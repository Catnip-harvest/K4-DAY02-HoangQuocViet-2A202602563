"""train.py - vòng huấn luyện dùng chung cho mọi thí nghiệm (B, T, F).

Một hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.

Lựa chọn cài đặt (ghi vào báo cáo):
    - LR cập nhật theo bước (iteration): warmup tuyến tính từ 0 trong `warmup_epochs`, rồi cosine về 0.
    - AMP = autocast FP16 + GradScaler (T4 không hỗ trợ BF16 nhanh). channels_last cho mọi model.
    - EMA cập nhật sau mỗi bước tối ưu, áp dụng cho cả tham số lẫn buffer float (running_mean/var của BN),
      giống timm ModelEmaV2. Khi bật EMA, checkpoint được chọn theo macro-F1 val CỦA MODEL EMA;
      model thường vẫn được đánh giá mỗi epoch để so sánh (I06).
    - Chọn checkpoint: epoch có macro-F1 val cao nhất, hoà thì giữ epoch sớm hơn (so sánh `>` chặt).
    - cudnn.benchmark = True để nhanh; vì vậy hai lần chạy cùng seed không trùng từng bit.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "eval.py").exists():
            return p
    raise FileNotFoundError("không tìm thấy eval.py của repo gốc ở các thư mục cha")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from eval import compute_metrics, save_predictions  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn, dùng trong tên ảnh curves/<exp_id>_<desc>.png
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | vflip
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    cache: str | None = None          # .npy ảnh đã giải mã (dataset.build_cache); None = đọc JPEG
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    save_checkpoint: bool = True
    max_steps_per_epoch: int | None = None   # chỉ dùng để chạy thử nhanh (smoke test)
    limit_eval: int | None = None            # chỉ dùng để chạy thử nhanh: N ảnh đầu của val/test
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    desc = cfg.desc or cfg.backbone
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}.png"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU + CUDA). Worker DataLoader được seed qua generator."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True       # nhanh hơn; không tái lập từng bit
    torch.backends.cudnn.deterministic = False


def build_optimizer(model, cfg: Config):
    """AdamW với các nhóm tham số của model.param_groups (norm/bias không weight decay)."""
    import torch
    from model import param_groups
    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Hệ số nhân LR tại bước `step`: warmup tuyến tính (0 -> 1) rồi cosine (1 -> 0)."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """LambdaLR cập nhật mỗi bước: warmup `warmup_epochs` rồi cosine về 0 ở bước cuối."""
    import torch
    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warm))


class EMA:
    """W_ema <- d * W_ema + (1 - d) * W sau mỗi bước (slide trang 56). Bản sao riêng để đánh giá.

    Buffer float (running_mean/var của BN) cũng được lấy trung bình động; buffer nguyên
    (num_batches_tracked) được chép thẳng.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model) -> None:
        import torch
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                src = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
                else:
                    v.copy_(src)


def _criterion_for(cfg: Config, train_df):
    from losses import build_criterion, class_weights
    if cfg.loss == "ce" and cfg.label_smoothing > 0:
        return build_criterion("ls", smoothing=cfg.label_smoothing)
    if cfg.loss == "ls":
        return build_criterion("ls", smoothing=cfg.label_smoothing or 0.1)
    if cfg.loss == "focal":
        return build_criterion("focal", gamma=cfg.focal_gamma)
    if cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"].to_numpy(), minlength=9)  # chỉ dùng TRAIN
        return build_criterion("ce_weighted", weight=class_weights(counts, cfg.class_weight_beta or 0.0))
    return build_criterion(cfg.loss)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch. Trả về {"train_loss", "train_acc" (vô nghĩa khi mix), "lrs": [lr head theo bước]}."""
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)
    rng = np.random.default_rng(cfg.seed * 100003 + int(scheduler.last_epoch))
    tot_loss, tot_n, correct, lrs = 0.0, 0, 0, []
    for step, (x, y, _) in enumerate(loader):
        if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
            break
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg.amp and device.type == "cuda"):
            if cfg.mix:
                x_in, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix, rng=rng)
                logits = model(x_in)
                loss = mixed_loss(criterion, logits.float(), targets)
            else:
                logits = model(x)
                loss = criterion(logits.float(), y)
        lrs.append(max(g["lr"] for g in optimizer.param_groups))  # LR dùng cho bước này
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        tot_loss += loss.item() * x.shape[0]
        tot_n += x.shape[0]
        correct += (logits.argmax(1) == y).sum().item()
    return {"train_loss": tot_loss / max(1, tot_n), "train_acc": correct / max(1, tot_n), "lrs": lrs}


def evaluate(model, loader, criterion, device, amp: bool = True):
    """Chạy model ở chế độ eval, không gradient. Trả về (filenames, y_true, logits[N,9], loss)."""
    import torch
    model.eval()
    names, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(x)
            outs.append(logits.float().cpu())
            ys.append(y)
            names.extend(f)
    logits = torch.cat(outs)
    y_true = torch.cat(ys)
    loss = torch.nn.functional.cross_entropy(logits, y_true).item()  # loss val luôn là CE thường để so sánh
    return names, y_true.numpy(), logits.numpy(), loss


def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics_from_logits(y_true, logits) -> dict:
    probs = softmax_np(logits)
    return compute_metrics(np.asarray(y_true), probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lrs: list[float] | None = None) -> None:
    """Ảnh 3 ô: loss train/val; macro-F1 và top-1 val (và EMA nếu có); LR theo bước."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train loss (loss huấn luyện)")
    ax[0].plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    if "val_loss_ema" in history[0]:
        ax[0].plot(ep, [h["val_loss_ema"] for h in history], "^--", label="val loss EMA")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax[1].plot(ep, [h["val_top1"] for h in history], "s-", label="val top-1")
    if "val_macro_f1_ema" in history[0]:
        ax[1].plot(ep, [h["val_macro_f1_ema"] for h in history], "^--", label="val macro-F1 EMA")
    best = max(range(len(history)), key=lambda i: (history[i]["select_f1"], -i))
    ax[1].axvline(ep[best], color="gray", ls=":", label=f"best epoch {ep[best]}")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("metric"); ax[1].set_title("Val metric"); ax[1].legend(); ax[1].grid(alpha=.3)
    if lrs:
        ax[2].plot(np.arange(1, len(lrs) + 1), lrs)
    ax[2].set_xlabel("step (iteration)"); ax[2].set_ylabel("LR (nhóm head)"); ax[2].set_title("Learning rate")
    ax[2].grid(alpha=.3)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict tóm tắt."""
    import torch
    import timm
    import dataset as D
    from model import build_model, count_gmacs, count_params

    t_start = time.time()
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    device = _device()

    # 2. dữ liệu
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)
    if cfg.limit_eval:
        val_df, test_df = val_df.head(cfg.limit_eval), test_df.head(cfg.limit_eval)
    tf_train = D.build_transforms(True, cfg.img_size, cfg.aug)
    tf_eval = D.build_transforms(False, cfg.img_size)
    train_loader = D.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size, True, cfg.sampler,
                                 cfg.num_workers, cfg.seed, cfg.cache)
    val_loader = D.make_loader(val_df, cfg.images_dir, tf_eval, 128, False, None, cfg.num_workers, cfg.seed, cfg.cache)

    # 4. model, loss, tối ưu
    model = build_model(cfg.backbone, True, D.NUM_CLASSES, cfg.drop_rate, cfg.init)
    tag = model.weight_tag
    params_m = count_params(model)
    gmacs = count_gmacs(model, cfg.img_size)
    model = model.to(device).to(memory_format=torch.channels_last)
    criterion = _criterion_for(cfg, train_df).to(device)
    optimizer = build_optimizer(model, cfg)
    steps = len(train_loader) if not cfg.max_steps_per_epoch else min(len(train_loader), cfg.max_steps_per_epoch)
    scheduler = build_scheduler(optimizer, cfg, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    print(f"[{cfg.exp_id} seed{cfg.seed}] {cfg.backbone} tag={tag} params={params_m:.2f}M GMAC={gmacs:.3f} "
          f"device={device} steps/epoch={steps}", flush=True)

    # 5. vòng epoch
    history, all_lrs = [], []
    best = {"f1": -1.0, "epoch": -1, "state": None, "raw_f1": -1.0, "raw_epoch": -1, "raw_state": None}
    train_times = []
    for epoch in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_times.append(time.time() - t0)
        all_lrs.extend(tr["lrs"])
        _, yv, lv, loss_v = evaluate(model, val_loader, None, device, cfg.amp)
        mv = metrics_from_logits(yv, lv)
        row = {"epoch": epoch, "train_loss": tr["train_loss"], "train_acc": tr["train_acc"],
               "val_loss": loss_v, "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"],
               "lr_end": tr["lrs"][-1], "train_time_s": train_times[-1]}
        if mv["macro_f1"] > best["raw_f1"]:
            best.update(raw_f1=mv["macro_f1"], raw_epoch=epoch,
                        raw_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        select_f1, cur = mv["macro_f1"], model
        if ema is not None:
            _, ye, le, loss_e = evaluate(ema.module, val_loader, None, device, cfg.amp)
            me = metrics_from_logits(ye, le)
            row.update(val_loss_ema=loss_e, val_macro_f1_ema=me["macro_f1"], val_top1_ema=me["top1"])
            select_f1, cur = me["macro_f1"], ema.module
        row["select_f1"] = select_f1
        if select_f1 > best["f1"]:  # hoà giữ epoch sớm hơn
            best.update(f1=select_f1, epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in cur.state_dict().items()})
        history.append(row)
        print(f"[{cfg.exp_id} seed{cfg.seed}] ep {epoch:2d} train_loss {tr['train_loss']:.4f} "
              f"val_loss {loss_v:.4f} val_F1 {mv['macro_f1']:.4f} val_top1 {mv['top1']:.4f}"
              + (f" | EMA F1 {row['val_macro_f1_ema']:.4f}" if ema else "")
              + f" | {train_times[-1]:.1f}s", flush=True)

    # 6. nạp checkpoint tốt nhất, lưu logit/dự đoán val
    model.load_state_dict(best["state"])
    if cfg.save_checkpoint:
        torch.save({"state_dict": best["state"], "config": dataclasses.asdict(cfg), "epoch": best["epoch"]},
                   rd / "best.pt")
    names_v, yv, lv, loss_v = evaluate(model, val_loader, None, device, cfg.amp)
    np.save(rd / "val_logits.npy", lv)
    np.save(rd / "val_labels.npy", yv)
    (rd / "val_filenames.txt").write_text("\n".join(names_v))
    save_predictions(pred_path(cfg, "val"), names_v, yv, softmax_np(lv))
    mv = metrics_from_logits(yv, lv)
    if ema is not None:  # logit val của model thường (không EMA) ở epoch tốt nhất của nó, cho I06
        raw = copy.deepcopy(model)
        raw.load_state_dict(best["raw_state"])
        _, _, lraw, _ = evaluate(raw, val_loader, None, device, cfg.amp)
        np.save(rd / "val_logits_raw.npy", lraw)
        del raw

    # 7. test: đúng một lần, chỉ khi được bật (Bước 4)
    if cfg.save_test_predictions:
        test_loader = D.make_loader(test_df, cfg.images_dir, tf_eval, 128, False, None, cfg.num_workers,
                                    cfg.seed, cfg.cache)
        names_t, yt, lt, _ = evaluate(model, test_loader, None, device, cfg.amp)
        np.save(rd / "test_logits.npy", lt)
        np.save(rd / "test_labels.npy", yt)
        (rd / "test_filenames.txt").write_text("\n".join(names_t))
        save_predictions(pred_path(cfg, "test"), names_t, yt, softmax_np(lt))

    # 8. log, biểu đồ, tóm tắt
    import pandas as pd
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lr_steps.npy", np.asarray(all_lrs))
    title = (f"{cfg.exp_id} - {cfg.backbone} (seed {cfg.seed}) {cfg.desc} | best epoch {best['epoch']}, "
             f"val macro-F1 {mv['macro_f1']:.4f}")
    plot_curves(history, curve_path(cfg), title, all_lrs)
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": tag, "init": cfg.init,
        "params_m": params_m, "gmacs": gmacs, "img_size": cfg.img_size, "epochs": cfg.epochs,
        "best_epoch": best["epoch"], "val_macro_f1": mv["macro_f1"], "val_top1": mv["top1"],
        "val_balanced_acc": mv["balanced_acc"], "val_ece": mv["ece"], "val_nll": mv["nll"],
        "val_f1_per_class": mv["f1"].tolist(), "val_recall_per_class": mv["recall"].tolist(),
        "train_time_per_epoch_s": float(np.mean(train_times)), "total_time_s": time.time() - t_start,
        "raw_best_epoch": best["raw_epoch"], "raw_val_macro_f1": best["raw_f1"],
        "torch": torch.__version__, "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "curve": curve_path(cfg).name,
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_trained(run_path: str | Path, device=None):
    """Nạp model đã train từ <run_dir>/best.pt. Trả về (model ở eval, Config)."""
    import torch
    from model import build_model
    ck = torch.load(Path(run_path) / "best.pt", map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    model = build_model(cfg.backbone, False, 9, cfg.drop_rate, "finetune")
    model.load_state_dict(ck["state_dict"])
    device = device or _device()
    return model.to(device).eval(), cfg


def _coerce(field: dataclasses.Field, raw: str):
    if raw.lower() in ("none", "null"):
        return None
    typ = str(field.type)
    if "bool" in typ:
        if raw.lower() in ("1", "true", "yes"):
            return True
        if raw.lower() in ("0", "false", "no"):
            return False
        raise ValueError(f"{field.name}: không hiểu giá trị bool {raw!r}")
    if typ.startswith("int"):
        return int(raw)
    if typ.startswith("float"):
        return float(raw)
    if "int" in typ and "float" not in typ:
        return int(raw)
    if "float" in typ:
        return float(raw)
    return raw


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict, ép kiểu theo field của Config."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"cần dạng KEY=VALUE, nhận {pair!r}")
        k, v = pair.split("=", 1)
        if k not in fields:
            raise KeyError(f"Config không có trường {k!r}")
        out[k] = _coerce(fields[k], v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2))


if __name__ == "__main__":
    main()
