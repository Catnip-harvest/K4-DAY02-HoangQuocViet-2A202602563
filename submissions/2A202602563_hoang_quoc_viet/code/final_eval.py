"""final_eval.py - Bước 4: áp dụng cấu hình suy luận đã chốt trên VAL cho các model chung kết và mốc,
chạy TEST đúng MỘT lần cho mỗi seed, ghi predictions/ đúng định dạng eval.py.

    python final_eval.py --data /tmp/deepweeds --out /kaggle/working/out \
        --final-runs <F01/seed0> <F01/seed1> <F01/seed2> --method I01_hflip --space logit --ts \
        --rt-method I00_1view --baseline-runs <T00/seed0> <T00/seed1> <T00/seed2>

Với mỗi model, mọi view cần thiết được tính trong MỘT lượt qua test (ảnh 256, view tạo trên GPU),
nên chung kết (F01), bản chưa hiệu chuẩn (F01_uncal) và cấu hình thời gian thực (R01, view c224 của
cùng lượt) dùng chung một lần chạy test. Nhiệt độ T khớp trên VAL rồi áp dụng nguyên sang test.

File ghi ra (<out>/predictions/):
    F01_seed<k>_test.csv, F01_seed<k>_val.csv      cấu hình chung kết (phương pháp + T)
    F01_uncal_seed<k>_test.csv                     cùng logit nhưng chưa temperature scaling (I4a)
    R01_seed<k>_test.csv, R01_seed<k>_val.csv      cấu hình thời gian thực (1 view + T riêng)
    T00_seed<k>_test.csv, T00_seed<k>_val.csv      mốc: công thức nền + I00 (1 view, không T)
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import benchmark as B  # noqa: E402
import dataset as D  # noqa: E402
import inference as I  # noqa: E402
from train import compute_metrics, load_trained, save_predictions  # noqa: E402


def loader_for(df, data, workers):
    tf = D.build_transforms(False, 256, eval_mode="full")
    cache = f"{data}/cache.npy" if Path(f"{data}/cache.npy").exists() else None
    return D.make_loader(df, f"{data}/images", tf, 64, False, None, workers, 0, cache)


def summarize(y, p):
    m = compute_metrics(y, p.argmax(1), p)
    return {k: m[k] for k in ("top1", "macro_f1", "balanced_acc", "ece", "nll")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--final-runs", nargs="+", required=True)
    ap.add_argument("--baseline-runs", nargs="+", required=True)
    ap.add_argument("--method", default="I00_1view")
    ap.add_argument("--space", default="logit", choices=["logit", "prob"])
    ap.add_argument("--ts", action="store_true", help="temperature scaling (T khớp trên val)")
    ap.add_argument("--rt-method", default="I00_1view")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="chỉ để chạy thử: N ảnh đầu của val/test")
    args = ap.parse_args()

    out = Path(args.out)
    pred_dir = out / "predictions"
    fin_dir = out / "final"
    fin_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_df, test_df = D.load_split(f"{args.data}/labels")
    if args.limit:
        val_df, test_df = val_df.head(args.limit), test_df.head(args.limit)
    vload, tload = loader_for(val_df, args.data, args.workers), loader_for(test_df, args.data, args.workers)
    views = sorted(set(I.METHODS[args.method]) | set(I.METHODS[args.rt_method]))
    record = {"method": args.method, "space": args.space, "ts": args.ts, "rt_method": args.rt_method,
              "views": views, "final": [], "baseline": []}

    for rd in args.final_runs:
        model, cfg = load_trained(rd, dev)
        k = cfg.seed
        nv, yv, vv = I.compute_view_logits(model, vload, dev, views)          # VAL
        T = I.fit_temperature_method(vv, yv, args.method, args.space) if args.ts else 1.0
        T_rt = I.fit_temperature_method(vv, yv, args.rt_method, "logit") if args.ts else 1.0
        nt, yt, tv = I.compute_view_logits(model, tload, dev, views)          # TEST: một lượt duy nhất
        np.savez_compressed(fin_dir / f"F01_seed{k}_views.npz", val_y=yv, test_y=yt,
                            **{f"val_{v}": a for v, a in vv.items()}, **{f"test_{v}": a for v, a in tv.items()})
        pv, pt = I.method_probs(vv, args.method, args.space, T), I.method_probs(tv, args.method, args.space, T)
        pt_uncal = I.method_probs(tv, args.method, args.space, 1.0)
        rv, rt = I.method_probs(vv, args.rt_method, "logit", T_rt), I.method_probs(tv, args.rt_method, "logit", T_rt)
        save_predictions(pred_dir / f"F01_seed{k}_test.csv", nt, yt, pt)
        save_predictions(pred_dir / f"F01_seed{k}_val.csv", nv, yv, pv)
        save_predictions(pred_dir / f"F01_uncal_seed{k}_test.csv", nt, yt, pt_uncal)
        save_predictions(pred_dir / f"R01_seed{k}_test.csv", nt, yt, rt)
        save_predictions(pred_dir / f"R01_seed{k}_val.csv", nv, yv, rv)
        record["final"].append({"run": rd, "seed": k, "T": T, "T_rt": T_rt, "best_epoch_cfg": cfg.exp_id,
                                "val": summarize(yv, pv), "test": summarize(yt, pt),
                                "test_uncal": summarize(yt, pt_uncal), "rt_val": summarize(yv, rv),
                                "rt_test": summarize(yt, rt)})
        print(f"F01 seed{k}: T={T:.4f} T_rt={T_rt:.4f} -> đã ghi predictions", flush=True)
        if k == min(int(json.loads((Path(r) / "config.json").read_text())["seed"]) for r in args.final_runs):
            final_model = copy.deepcopy(model)
        del model

    for rd in args.baseline_runs:
        model, cfg = load_trained(rd, dev)
        k = cfg.seed
        nv, yv, vv = I.compute_view_logits(model, vload, dev, ["c224"])
        nt, yt, tv = I.compute_view_logits(model, tload, dev, ["c224"])       # TEST: một lượt duy nhất
        np.savez_compressed(fin_dir / f"T00_seed{k}_views.npz", val_y=yv, test_y=yt, val_c224=vv["c224"],
                            test_c224=tv["c224"])
        pv, pt = I.method_probs(vv, "I00_1view"), I.method_probs(tv, "I00_1view")
        save_predictions(pred_dir / f"T00_seed{k}_test.csv", nt, yt, pt)
        save_predictions(pred_dir / f"T00_seed{k}_val.csv", nv, yv, pv)
        record["baseline"].append({"run": rd, "seed": k, "val": summarize(yv, pv), "test": summarize(yt, pt)})
        print(f"T00 seed{k}: đã ghi predictions", flush=True)
        del model

    # độ trễ của cấu hình chung kết và cấu hình thời gian thực (model F01 seed nhỏ nhất)
    lat = []
    if dev.type == "cuda":
        def fn_for(meth):
            vs = I.METHODS[meth]
            return lambda x: [I.VIEW_FNS[v](x) for v in vs]
        for meth, label in ((args.method, "F01 (chung kết)"), (args.rt_method, "R01 (thời gian thực)")):
            for b in (1, 32):
                for dt in ("fp32", "amp", "fp16"):
                    r = B.latency_report(final_model, b, 256, dt, "cuda", 10, 200, k_views=len(I.METHODS[meth]),
                                         view_fn=fn_for(meth))
                    r.update(config=f"{label} {meth}", method=meth)
                    lat.append(r)
                    print(f"{label:<22} {meth:<12} b{b:<3} {dt:<5} p50 {r['p50']:.2f} p95 {r['p95']:.2f} "
                          f"p99 {r['p99']:.2f} ms", flush=True)
    record["latency"] = lat
    (fin_dir / "final_record.json").write_text(json.dumps(record, indent=2))
    print("xong", flush=True)


if __name__ == "__main__":
    main()
