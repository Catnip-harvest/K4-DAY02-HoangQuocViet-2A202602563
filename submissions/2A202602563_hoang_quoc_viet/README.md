# Lab Day 2 — Hoàng Quốc Việt (2A202602563)

DeepWeeds, fold 0: so sánh 8 backbone, 11 ablation công thức huấn luyện + kết hợp, ≥ 8 phương pháp suy luận có đo
độ trễ, chung kết 3 seed. Kết quả: F01 macro-F1 test 0,9605 ± 0,0018 (mốc T00 0,9497 ± 0,0038), `eval.py grade` 19/20. Báo cáo: [`report.md`](report.md). Bảng số: [`results.xlsx`](results.xlsx).

## Notebook chạy lại được (Kaggle, GPU T4 x2)

Notebook chính: [`code/lab_day2.ipynb`](code/lab_day2.ipynb). Toàn bộ số liệu đến từ 4 phiên Kaggle chạy notebook này
(mỗi phiên một phần, chọn bằng biến môi trường `LAB_STAGE`; mỗi phiên clone nhánh `feat/day2-lab` của repo này rồi
`jupyter nbconvert --execute` notebook):

| Phiên | Kaggle notebook | `LAB_STAGE` | Nội dung |
|---|---|---|---|
| A | https://www.kaggle.com/code/viethwang3i/k4-day2-a-backbones | `A` | Bước 0 (kiểm tra chia dữ liệu, EDA, kiểm tra pipeline) + Bước 1 (B01–B07) |
| B | https://www.kaggle.com/code/viethwang3i/k4-day2-b-ablations | `B` | Bước 2: T00 × 3 seed, T01–T11, B08 |
| C+D | https://www.kaggle.com/code/viethwang3i/k4-day2-cd2-inference-final | `C+D` | kết hợp T12 + F01 × 3 seed; Bước 3 (suy luận, độ trễ); Bước 4 (test một lần mỗi seed, `eval.py score/grade`) |

Do nộp trễ, phiên C và D được gộp thành một phiên (xem report mục 8). Lần chạy đầu của phiên gộp
(`k4-day2-cd-inference-final`) lỗi trước bước suy luận/test (không tìm thấy output phiên trước) và bị bỏ; mọi số đến từ
`k4-day2-cd2-inference-final`. Notebook Kaggle đang để private; cần bật public nếu người chấm muốn mở.

Notebook đã thực thi của từng phiên (có output) nằm ở `code/executed/lab_day2_executed_<A|B|CD2>.ipynb`.
Mỗi phiên sau đọc output của phiên trước qua *kernel sources* của Kaggle (C+D đọc A và B).
Muốn chạy toàn bộ trong một phiên: đặt `LAB_STAGE=all` (mặc định), khoảng 2,5–3 giờ GPU T4 x2 (A ~45 phút, B ~1 giờ 10, C+D ~40 phút).

## Thứ tự chạy

```bash
# 1) GPU (Kaggle): code/lab_day2.ipynb với LAB_STAGE = A, B, C+D (các lựa chọn giữa các phiên đã ghi cứng trong
#    notebook, kèm lý do, và đều dựa trên val)
# 2) CPU: gom output 4 phiên thành sản phẩm nộp bài
cd code
python make_results.py --inputs <out_A> <out_B> <out_CD> --sub .. --data <thư mục có images/ và labels/>
# 3) kiểm tra
python -m unittest test_code -v                     # kiểm tra tự viết (focal γ=0 == CE, CutMix, gộp BN, ...)
cd ../../.. && python -m unittest discover -s tests  # test của repo (trên Windows: đặt PYTHONUTF8=1)
python eval.py score --pred "submissions/2A202602563_hoang_quoc_viet/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv --tag F01
python eval.py grade --final "submissions/2A202602563_hoang_quoc_viet/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602563_hoang_quoc_viet/predictions/T00_seed*_test.csv" \
    --uncal "submissions/2A202602563_hoang_quoc_viet/predictions/F01_uncal_seed*_test.csv" \
    --final-val "submissions/2A202602563_hoang_quoc_viet/predictions/F01_seed*_val.csv" \
    --val-csv data/labels/val_subset0.csv --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv \
    --latency-p95-ms 5.15 --latency-method proper   # p95 batch-1 FP16 của R01
```

Một thí nghiệm lẻ: `python code/train.py --set exp_id=T04 backbone=deit_small_patch16_224.fb_in1k aug=trivial seed=0
images_dir=... labels_dir=...`.

## Code

| File | Nội dung |
|---|---|
| `dataset.py` | đọc CSV fold 0, `check_split` (số ảnh, giao rỗng, hợp 17.509, file tồn tại), transform/augmentation, cache ảnh đã giải mã, DataLoader có seed |
| `model.py` | backbone timm (tag cố định), khởi tạo head, đóng băng (BN giữ eval), 4 nhóm tham số (không weight decay cho norm/bias), params, GMAC (`torch.utils.flop_counter`) |
| `losses.py` | label smoothing, focal, CE có trọng số, Mixup/CutMix (lam theo diện tích thật) |
| `train.py` | `Config` + **một** hàm `run(cfg)` cho mọi thí nghiệm: AMP, warmup + cosine theo bước, EMA, chọn checkpoint theo macro-F1 val, đường cong |
| `inference.py` | TTA (lật, 5/10 crop, đa tỉ lệ), gộp xác suất/logit, ensemble, temperature scaling, gộp BN |
| `benchmark.py` | đo độ trễ: warmup 10, ≥ 50 lần (dùng 100–200), `cuda.synchronize` trước/sau, p50/p95/p99 |
| `run_experiments.py` | danh sách thí nghiệm (B, T, F) và bộ chạy hàng loạt trên 2 GPU |
| `inference_experiments.py` | Bước 3 trên val + đo độ trễ |
| `final_eval.py` | Bước 4: test một lần mỗi seed, ghi `predictions/` bằng `eval.save_predictions` |
| `make_results.py` | Bước 5: `results.xlsx`, `curves/`, `figures/`, tính lại chỉ số test bằng `eval.py` |
| `test_code.py` | kiểm tra tự viết |

`eval.py` của repo không bị sửa.

## Phiên bản thư viện (phiên Kaggle)

Python 3.13.15 · torch 2.11.0+cu128 · torchvision 0.26.0+cu128 · CUDA 12.8 · cuDNN 9.19 · **timm 1.0.30** (cài bằng
pip ở ô đầu notebook; tiến trình notebook đã nạp sẵn 1.0.29 của Kaggle nên `env_*.json` ghi 1.0.29, còn mọi tiến trình
huấn luyện/suy luận chạy 1.0.30 — trường `timm` trong `summary.json` của từng run) · numpy 2.1.3 · pandas 2.3.3 ·
GPU 2 × Tesla T4 (16 GB), 4 vCPU. Phân tích CPU cục bộ: Python 3.13, pandas 3.0.5, openpyxl 3.1.5, matplotlib 3.11.

## Seed

- Bước 1 và 2: seed 0 cho mọi run (cùng seed cho mọi backbone); `T00` thêm seed 1, 2 để đo nhiễu.
- Bước 4: `F01` seed 0, 1, 2; mốc `T00` seed 0, 1, 2.
- Seed cố định `random`, `numpy`, `torch` (CPU+CUDA), generator và worker của DataLoader, hoán vị Mixup/CutMix.
  `cudnn.benchmark=True` nên hai lần chạy cùng seed không trùng từng bit.

## Thư mục

`results.xlsx` · `report.md` · `numbers.json` (số liệu báo cáo) · `curves/` (26 ảnh, một ảnh mỗi run B/T/F) · `predictions/` (test + val của F01, F01_uncal,
R01, T00; 3 seed) · `figures/` (EDA, augmentation, ma trận nhầm lẫn, đánh đổi độ chính xác–độ trễ, ảnh bị đoán sai) ·
`code/`. Không commit dữ liệu hay checkpoint.
