"""run_experiments.py - danh sách thí nghiệm và bộ chạy hàng loạt trên nhiều GPU.

Mọi thí nghiệm đều đi qua MỘT hàm `train.run(Config(...))`; file này chỉ định nghĩa các Config
(khác nhau đúng một yếu tố so với nền) và phân phối chúng lên các GPU (mỗi GPU một tiến trình con,
nhận việc từ hàng đợi thư mục; đổi tên file là thao tác nguyên tử nên không có job nào chạy hai lần).

    python run_experiments.py --stage backbones --data /tmp/deepweeds --out /kaggle/working/out
    python run_experiments.py --stage ablations --backbone convnext_tiny.fb_in1k ...
    python run_experiments.py --stage combo --backbone ... --set '{"T12_combo": {"aug": "trivial", ...}}'
    python run_experiments.py --stage final --backbone ... --set '{"aug": "trivial", ...}'
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Bước 1: backbone (tag trọng số ghi cố định để tái lập; mọi tag đều là ImageNet-1k).
BACKBONES = [
    ("B01", "resnet50.a1_in1k", "resnet50"),
    ("B02", "resnext50_32x4d.a1h_in1k", "resnext50"),
    ("B03", "convnext_tiny.fb_in1k", "convnext_tiny"),
    ("B04", "deit_small_patch16_224.fb_in1k", "deit_small"),
    ("B05", "swin_tiny_patch4_window7_224.ms_in1k", "swin_tiny"),
    ("B06", "efficientnet_b0.ra_in1k", "efficientnet_b0"),
    ("B07", "mobilenetv3_large_100.ra_in1k", "mobilenetv3_large"),
    # thêm sau phiên A: cùng kiến trúc ResNet-50 nhưng trọng số torchvision (CE, công thức cũ) thay cho a1 (BCE)
    ("B08", "resnet50.tv_in1k", "resnet50_tv"),
]

# Bước 2: mỗi dòng khác T00 đúng MỘT yếu tố (trục theo GUIDE.md mục 3).
ABLATIONS = [
    # exp_id, desc, trục, thay đổi so với T00
    ("T01", "frozen_linear_probe", "A", {"init": "frozen"}),
    ("T02", "from_scratch", "A", {"init": "scratch"}),
    ("T03", "color_jitter", "B", {"aug": "color"}),
    ("T04", "trivialaugment", "B", {"aug": "trivial"}),
    ("T05", "cutmix", "B", {"mix": "cutmix", "mix_alpha": 1.0}),
    ("T06", "mixup", "B", {"mix": "mixup", "mix_alpha": 0.2}),
    ("T07", "label_smoothing", "C", {"loss": "ls", "label_smoothing": 0.1}),
    ("T08", "focal_g2", "C", {"loss": "focal", "focal_gamma": 2.0}),
    ("T09", "class_weighted_ce", "C", {"loss": "ce_weighted"}),
    ("T10", "balanced_sampler", "D", {"sampler": "balanced"}),
    ("T11", "ema", "F", {"ema_decay": 0.998}),
]
NOISE_SEEDS = (0, 1, 2)   # T00 chạy 3 seed: ước lượng nhiễu, đồng thời là mốc cho Bước 4


def common_cfg(data: str, out: str, workers: int) -> dict:
    return {"images_dir": f"{data}/images", "labels_dir": f"{data}/labels", "cache": f"{data}/cache.npy",
            "out_dir": f"{out}/runs", "pred_dir": f"{out}/preds", "curves_dir": f"{out}/curves",
            "num_workers": workers}


def jobs_for(stage: str, common: dict, backbone: str | None, extra: dict) -> list[dict]:
    if stage == "backbones":
        return [{**common, "exp_id": e, "backbone": b, "desc": d, "seed": 0} for e, b, d in BACKBONES]
    if not backbone:
        raise SystemExit("--backbone là bắt buộc cho stage này")
    if stage == "ablations":
        jobs = [{**common, "exp_id": "T00", "backbone": backbone, "seed": s, "desc": f"seed{s}_baseline"}
                for s in NOISE_SEEDS]
        jobs += [{**common, "exp_id": e, "backbone": backbone, "seed": 0, "desc": d, **ch}
                 for e, d, _, ch in ABLATIONS]
        extra_bb = set(extra.get("with_backbones", []))   # backbone bổ sung chạy cùng hàng đợi
        jobs += [{**common, "exp_id": e, "backbone": b, "desc": d, "seed": 0} for e, b, d in BACKBONES if e in extra_bb]
        return jobs
    if stage == "combo":   # extra = {"T12_desc": {overrides}, ...}
        out = []
        for key, ch in extra.items():
            exp_id, desc = key.split("_", 1)
            out.append({**common, "exp_id": exp_id, "backbone": backbone, "seed": 0, "desc": desc, **ch})
        return out
    if stage == "final":   # extra = overrides của cấu hình chung kết
        seeds = extra.pop("seeds", [0, 1, 2])
        return [{**common, "exp_id": "F01", "backbone": backbone, "seed": s, "desc": f"seed{s}_final", **extra}
                for s in seeds]
    raise SystemExit(f"stage không hợp lệ: {stage}")


# --------------------------------------------------------------------------- #
# Hàng đợi thư mục + tiến trình con theo GPU
# --------------------------------------------------------------------------- #
def worker(queue: Path, gpu: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    sys.path.insert(0, str(HERE))
    import traceback

    import train
    pending, claimed, done = queue / "pending", queue / "claimed", queue / "done"
    while True:
        jobs = sorted(pending.glob("*.json"))
        if not jobs:
            return
        job = jobs[0]
        target = claimed / f"{job.stem}.gpu{gpu}.json"
        try:
            os.rename(job, target)
        except OSError:
            continue  # tiến trình khác đã nhận job này
        cfg_dict = json.loads(target.read_text())
        t0 = time.time()
        try:
            summary = train.run(train.Config(**cfg_dict))
            (done / f"{job.stem}.json").write_text(json.dumps({"ok": True, "summary": summary}, indent=2))
            print(f"[gpu{gpu}] DONE {job.stem} in {time.time() - t0:.0f}s val_F1={summary['val_macro_f1']:.4f}",
                  flush=True)
        except Exception as e:  # ghi lỗi, chạy tiếp job khác
            (done / f"{job.stem}.json").write_text(json.dumps({"ok": False, "error": repr(e),
                                                               "trace": traceback.format_exc()}, indent=2))
            print(f"[gpu{gpu}] FAILED {job.stem}: {e!r}\n{traceback.format_exc()}", flush=True)


def launch(jobs: list[dict], queue: Path, gpus: list[str], log_dir: Path) -> list[dict]:
    for sub in ("pending", "claimed", "done"):
        (queue / sub).mkdir(parents=True, exist_ok=True)
    for i, j in enumerate(jobs):
        (queue / "pending" / f"{i:02d}_{j['exp_id']}_seed{j['seed']}.json").write_text(json.dumps(j, indent=2))
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for g in gpus:
        log = open(log_dir / f"{queue.name}_gpu{g}.log", "w")
        procs.append((g, log, subprocess.Popen([sys.executable, "-u", __file__, "--worker", str(queue), "--gpu", g],
                                               stdout=log, stderr=subprocess.STDOUT, cwd=str(HERE),
                                               env=dict(os.environ, PYTHONIOENCODING="utf-8"))))
    t0 = time.time()
    while any(p.poll() is None for _, _, p in procs):
        time.sleep(30)
        n_done = len(list((queue / "done").glob("*.json")))
        print(f"  ... {n_done}/{len(jobs)} job xong sau {(time.time() - t0) / 60:.1f} phút", flush=True)
    results = []
    for g, log, p in procs:
        log.close()
        print(f"===== log gpu{g} (exit {p.returncode}) =====")
        print((log_dir / f"{queue.name}_gpu{g}.log").read_text(encoding="utf-8", errors="replace")[-20000:])
    for f in sorted((queue / "done").glob("*.json")):
        results.append({"job": f.stem, **json.loads(f.read_text())})
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["backbones", "ablations", "combo", "final"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--backbone")
    ap.add_argument("--set", default="{}", help="JSON: overrides (combo/final)")
    ap.add_argument("--gpus", default=None, help="ví dụ 0,1 (mặc định: mọi GPU thấy được)")
    ap.add_argument("--workers", type=int, default=3, help="num_workers DataLoader mỗi tiến trình")
    ap.add_argument("--no-cache", action="store_true", help="đọc JPEG trực tiếp, không dùng cache .npy")
    ap.add_argument("--smoke", action="store_true", help="chạy thử: 1 epoch x 2 bước mỗi job")
    ap.add_argument("--only", default=None, help="chỉ chạy các exp_id này (phân tách bằng dấu phẩy)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker(Path(args.worker), args.gpu)
        return
    if args.gpus is None:
        import torch
        n = torch.cuda.device_count()
        args.gpus = ",".join(str(i) for i in range(max(1, n)))
    common = common_cfg(args.data, args.out, args.workers)
    if args.no_cache:
        common["cache"] = None
    else:
        sys.path.insert(0, str(HERE))
        import dataset as D
        tr, va, te = D.load_split(common["labels_dir"])
        names = sorted(set(tr.Filename) | set(va.Filename) | set(te.Filename))
        t0 = time.time()
        D.build_cache(common["images_dir"], names, common["cache"])
        print(f"cache ảnh: {common['cache']} ({time.time() - t0:.0f}s)")
    jobs = jobs_for(args.stage, common, args.backbone, json.loads(args.set))
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["exp_id"] in keep]
    if args.smoke:
        for j in jobs:
            j.update(epochs=1, max_steps_per_epoch=2, batch_size=8, limit_eval=64)
    print(f"stage {args.stage}: {len(jobs)} job trên GPU {args.gpus}")
    queue = Path(args.out) / "queue" / args.stage
    results = launch(jobs, queue, args.gpus.split(","), Path(args.out) / "logs")
    ok = [r for r in results if r["ok"]]
    print(f"\n{len(ok)}/{len(jobs)} job thành công")
    for r in ok:
        s = r["summary"]
        print(f"{s['exp_id']:>4} seed{s['seed']} {s['backbone']:<38} val macro-F1 {s['val_macro_f1']:.4f} "
              f"top-1 {s['val_top1']:.4f} best_ep {s['best_epoch']:2d} {s['train_time_per_epoch_s']:.1f}s/epoch")
    if len(ok) != len(jobs):
        raise SystemExit("có job lỗi, xem log ở trên")


if __name__ == "__main__":
    main()
