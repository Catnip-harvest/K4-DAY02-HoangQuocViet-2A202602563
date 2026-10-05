"""inference_experiments.py - Bước 3: so sánh phương pháp suy luận trên VAL + đo độ trễ.

Không huấn luyện lại. Chỉ dùng VAL (test không được đụng tới ở bước này).

    python inference_experiments.py --data /tmp/deepweeds --out /kaggle/working/out \
        --best-run <runs/T12/seed0> --t00-runs <T00/seed0> <T00/seed1> <T00/seed2> \
        --ensemble-runs <runA> <runB> --ema-run <runs/T11/seed0> --bn-run <runs/B01/seed0>

Ghi ra <out>/inference/: inference_val.json (mỗi phương pháp một dòng), latency.json, val_views.npz.
Quy ước độ trễ: forward của model trên tensor đã ở GPU (không tính đọc/giải mã ảnh), batch 1 và 32,
warmup 10, 100 lần đo có torch.cuda.synchronize trước/sau; TTA chạy K forward tuần tự rồi gộp softmax.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark as B  # noqa: E402
import dataset as D  # noqa: E402
import inference as I  # noqa: E402
from train import compute_metrics, load_trained, softmax_np  # noqa: E402


def metrics_row(y, probs) -> dict:
    m = compute_metrics(y, probs.argmax(1), probs)
    return {"val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_balanced_acc": m["balanced_acc"],
            "val_ece": m["ece"], "val_nll": m["nll"],
            "val_f1_chinee": float(m["f1"][0]), "val_f1_snake": float(m["f1"][7])}


def changed(base_pred, new_pred, y) -> dict:
    """Số ca phương pháp mới đổi nhãn so với I00: sai->đúng và đúng->sai (GUIDE mục 8: TTA không miễn phí)."""
    b_ok, n_ok = base_pred == y, new_pred == y
    return {"n_changed": int((base_pred != new_pred).sum()), "wrong_to_right": int((~b_ok & n_ok).sum()),
            "right_to_wrong": int((b_ok & ~n_ok).sum())}


def crossfit_ece(z, y, k_folds: int = 2, seed: int = 0) -> dict:
    """ECE sau temperature scaling khi T khớp trên một nửa val và đo trên nửa còn lại (ngoài mẫu)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k_folds)
    before, after, ts = [], [], []
    for i in range(k_folds):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k_folds) if j != i])
        t = I.fit_temperature(z[tr], y[tr])
        p0, p1 = softmax_np(z[te]), I.apply_temperature(z[te], t)
        before.append(compute_metrics(y[te], p0.argmax(1), p0)["ece"])
        after.append(compute_metrics(y[te], p1.argmax(1), p1)["ece"])
        ts.append(t)
    return {"T_folds": ts, "ece_before_heldout": float(np.mean(before)), "ece_after_heldout": float(np.mean(after))}


def full_loader(df, data, workers):
    tf = D.build_transforms(False, 256, eval_mode="full")   # ảnh gốc 256, không resize
    return D.make_loader(df, f"{data}/images", tf, 64, False, None, workers, 0,
                         f"{data}/cache.npy" if Path(f"{data}/cache.npy").exists() else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--best-run", required=True)
    ap.add_argument("--t00-runs", nargs="*", default=[])
    ap.add_argument("--ensemble-runs", nargs="*", default=[])
    ap.add_argument("--ema-run")
    ap.add_argument("--bn-run")
    ap.add_argument("--latency-backbones", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="chỉ để chạy thử: dùng N ảnh val đầu tiên")
    args = ap.parse_args()

    out = Path(args.out) / "inference"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_df, _ = D.load_split(f"{args.data}/labels")
    if args.limit:
        val_df = val_df.head(args.limit)
    loader = full_loader(val_df, args.data, args.workers)
    rows, lat = [], []
    gpu = torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"
    print("GPU:", gpu, "| torch", torch.__version__, flush=True)

    # ---------- 1. mọi view của model tốt nhất trong một lượt ----------
    model, cfg = load_trained(args.best_run, dev)
    best_name = f"{cfg.exp_id}_seed{cfg.seed}"
    views = sorted({v for vs in I.METHODS.values() for v in vs})
    t0 = time.time()
    names, y, vl = I.compute_view_logits(model, loader, dev, views)
    print(f"{len(views)} view x {len(y)} ảnh val: {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out / "val_views.npz", y=y, names=np.array(names), **vl)
    saved = np.load(Path(args.best_run) / "val_logits.npy")[:len(y)]
    print("kiểm tra c224 == val_logits.npy lúc train: max |diff| =", float(np.abs(saved - vl["c224"]).max()), flush=True)

    base_pred = softmax_np(vl["c224"]).argmax(1)
    for meth, vs in I.METHODS.items():
        spaces = ["logit", "prob"] if len(vs) > 1 else ["logit"]
        for sp in spaces:
            p = I.method_probs(vl, meth, sp)
            code = meth.split("_")[0]
            rows.append({"exp_id": "I03" if sp == "prob" else code, "method": meth,
                         "aggregation": sp if len(vs) > 1 else "-", "model": best_name, "K": len(vs),
                         **metrics_row(y, p), **changed(base_pred, p.argmax(1), y)})

    # ---------- I07: temperature scaling (T khớp trên VAL) ----------
    temps = {}
    for meth in ("I00_1view", "I01_hflip"):
        z = np.mean(I.method_logits(vl, meth), axis=0)
        t = I.fit_temperature(z, y)
        before, after = metrics_row(y, softmax_np(z)), metrics_row(y, I.apply_temperature(z, t))
        cf = crossfit_ece(z, y)
        temps[meth] = {"T": t, "ece_before": before["val_ece"], "ece_after": after["val_ece"],
                       "nll_before": before["val_nll"], "nll_after": after["val_nll"], **cf}
        rows.append({"exp_id": "I07", "method": f"{meth}+TS", "aggregation": "logit", "model": best_name,
                     "K": len(I.METHODS[meth]), **after, **changed(base_pred, I.apply_temperature(z, t).argmax(1), y),
                     "note": f"T={t:.4f} (khớp trên val); ECE {before['val_ece']:.4f} -> {after['val_ece']:.4f}; "
                             f"ECE ngoài mẫu (2-fold trên val) {cf['ece_before_heldout']:.4f} -> "
                             f"{cf['ece_after_heldout']:.4f}"})
    print("temperature:", json.dumps(temps, indent=1), flush=True)

    # ---------- I05: ensemble (gộp xác suất) từ logit val đã lưu ----------
    def run_probs(rd):
        return softmax_np(np.load(Path(rd) / "val_logits.npy")[:len(y)])

    def run_name(rd):
        c = json.loads((Path(rd) / "config.json").read_text())
        return f"{c['exp_id']}_seed{c['seed']}"

    y_ref = y
    for rd in args.t00_runs + args.ensemble_runs:
        assert (Path(rd) / "val_filenames.txt").read_text().splitlines()[:len(y)] == names, f"{rd}: thứ tự file khác"
    if len(args.t00_runs) >= 2:
        ps = [run_probs(r) for r in args.t00_runs]
        singles = [metrics_row(y_ref, p)["val_macro_f1"] for p in ps]
        rows.append({"exp_id": "I05", "method": "ensemble_T00_seeds", "aggregation": "prob",
                     "model": "+".join(run_name(r) for r in args.t00_runs), "K": len(ps),
                     **metrics_row(y_ref, I.ensemble_probs(ps)),
                     "note": "từng model: " + ", ".join(f"{s:.4f}" for s in singles)})
    if args.ensemble_runs:
        ps = [softmax_np(vl["c224"])] + [run_probs(r) for r in args.ensemble_runs]
        rows.append({"exp_id": "I05", "method": "ensemble_backbones", "aggregation": "prob",
                     "model": "+".join([best_name] + [run_name(r) for r in args.ensemble_runs]), "K": len(ps),
                     **metrics_row(y_ref, I.ensemble_probs(ps)), **changed(base_pred, I.ensemble_probs(ps).argmax(1), y)})

    # ---------- I06: EMA vs trọng số thường ----------
    if args.ema_run:
        zr = np.load(Path(args.ema_run) / "val_logits_raw.npy")[:len(y)]
        ze = np.load(Path(args.ema_run) / "val_logits.npy")[:len(y)]
        for tagname, z in (("I06_raw_weights", zr), ("I06_ema_weights", ze)):
            rows.append({"exp_id": "I06", "method": tagname, "aggregation": "-", "model": run_name(args.ema_run),
                         "K": 1, **metrics_row(y_ref, softmax_np(z))})

    # ---------- I08: FP16 (model.half()) và gộp BN ----------
    import copy
    half = copy.deepcopy(model).half()
    with torch.inference_mode():
        zs = []
        for x, _, _ in loader:
            xi = I.VIEW_FNS["c224"](x.to(dev)).half().contiguous(memory_format=torch.channels_last)
            zs.append(half(xi).float().cpu())
    z16 = torch.cat(zs).numpy()
    del half
    rows.append({"exp_id": "I08", "method": "fp16_half", "aggregation": "-", "model": best_name, "K": 1,
                 **metrics_row(y, softmax_np(z16)), **changed(base_pred, z16.argmax(1), y),
                 "note": f"max |logit fp16 - logit AMP| = {float(np.abs(z16 - vl['c224']).max()):.4f}"})
    bn_model = None
    if args.bn_run:
        bn_model, bn_cfg = load_trained(args.bn_run, dev)
        fused = I.fuse_conv_bn(bn_model.float())
        xb = next(iter(loader))[0][:8].to(dev)
        xb = I.VIEW_FNS["c224"](xb)
        diff = I.max_abs_diff(bn_model.float(), fused, xb)
        print(f"BN fusion {bn_cfg.backbone}: {fused.n_fused} cặp, max |diff| FP32 = {diff:.3e}", flush=True)
        for tagname, mm in (("bn_unfused_fp32", bn_model), ("bn_fused_fp32", fused)):
            _, yb, vb = I.compute_view_logits(mm, loader, dev, ["c224"], amp=False)
            rows.append({"exp_id": "I08", "method": tagname, "aggregation": "-",
                         "model": f"{bn_cfg.exp_id}_seed{bn_cfg.seed}", "K": 1, **metrics_row(yb, softmax_np(vb["c224"])),
                         "note": f"{fused.n_fused} cặp Conv+BN; max |diff| đầu ra = {diff:.2e}"})

    (out / "inference_val.json").write_text(json.dumps({"rows": rows, "temperature": temps}, indent=2))
    print(json.dumps(rows, indent=1)[:6000], flush=True)

    # ---------- độ trễ ----------
    if dev.type != "cuda":
        print("Không có GPU: bỏ qua đo độ trễ")
        return
    it = args.iters

    def L(m, name, batch, size=224, dtype="amp", fused=False, k=1, view_fn=None, **extra):
        r = B.latency_report(m, batch, size, dtype, "cuda", 10, it, fused_bn=fused, k_views=k, view_fn=view_fn)
        r.update(config=name, **extra)
        lat.append(r)
        print(f"{name:<45} b{batch:<3} {dtype:<5} p50 {r['p50']:7.2f} p95 {r['p95']:7.2f} p99 {r['p99']:7.2f} ms "
              f"{r['images_per_s']:8.1f} img/s", flush=True)

    for b in (1, 32):
        for dt in ("fp32", "amp", "fp16"):
            L(model, f"{best_name} I00 1-view", b, 224, dt, method="I00_1view")
    for meth in ("I01_hflip", "I02a_5crop", "I02b_10crop", "I02c_multiscale", "I04_full256", "I04_full288",
                 "I04_full320"):
        vs = I.METHODS[meth]
        fn = (lambda vs_: (lambda x: [I.VIEW_FNS[v](x) for v in vs_]))(vs)
        for b in (1, 32):
            L(model, f"{best_name} {meth}", b, 256, "amp", k=len(vs), view_fn=fn, method=meth)
    if args.ensemble_runs:
        ens = torch.nn.ModuleList([model] + [load_trained(r, dev)[0] for r in args.ensemble_runs])

        class Ens(torch.nn.Module):
            def __init__(self, ms):
                super().__init__()
                self.ms = ms

            def forward(self, x):
                return torch.stack([m(x).float().softmax(-1) for m in self.ms]).mean(0)

        for b in (1, 32):
            L(Ens(ens), "ensemble_backbones", b, 224, "amp", k=len(ens), method="ensemble_backbones")
        del ens
    if bn_model is not None:
        fused = I.fuse_conv_bn(bn_model.float())
        for b in (1, 32):
            for dt in ("fp32", "fp16"):
                L(bn_model, f"{bn_cfg.backbone} unfused", b, 224, dt, fused=False, method="bn")
                L(fused, f"{bn_cfg.backbone} BN-fused", b, 224, dt, fused=True, method="bn")
    # mọi backbone (kiến trúc giống hệt; trọng số không ảnh hưởng độ trễ) - đo cô lập
    import timm
    for name in args.latency_backbones:
        m = timm.create_model(name.split(".")[0], pretrained=False, num_classes=9)
        for b, dt in ((1, "fp32"), (1, "amp"), (1, "fp16"), (32, "amp")):
            L(m, f"backbone {name}", b, 224, dt, method="backbone")
        del m
    (out / "latency.json").write_text(json.dumps(lat, indent=2))
    print("xong", flush=True)


if __name__ == "__main__":
    main()
