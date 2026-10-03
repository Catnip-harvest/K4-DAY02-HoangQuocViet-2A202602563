"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Giao diện (giữ nguyên theo starter/):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Thêm so với starter:
    build_cache(images_dir, filenames, out_path) -> giải mã JPEG một lần thành mảng uint8 (N, 256, 256, 3)
        lưu .npy; DeepWeedsDataset(cache=...) đọc bằng memmap nên nhiều tiến trình/worker dùng chung
        page cache của hệ điều hành. Ảnh giống hệt đọc bằng PIL (chỉ bỏ bước giải mã lặp lại mỗi epoch).

Quyết định tiền xử lý (ghi vào báo cáo):
    - Ảnh gốc 256x256. Val/test: CenterCrop(224) (không resize vì ảnh đã là 256), ToTensor, Normalize ImageNet.
    - Train "basic": RandomResizedCrop(224, scale=(0.08, 1)) + lật ngang.
    - Không lật dọc trong công thức nền (giữ đúng công thức nền của GUIDE 1.4).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
TOTAL_IMAGES = 17509
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
# Table 1 của bài báo (Olsen et al. 2019) để đối chiếu ở bước EDA.
PAPER_TABLE1 = {
    "Chinee Apple": 1125, "Lantana": 1064, "Parkinsonia": 1031, "Parthenium": 1022,
    "Prickly Acacia": 1062, "Rubber Vine": 1009, "Siam Weed": 1074, "Snake Weed": 1016,
    "Negatives": 9106,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # mọi backbone dùng ở đây đều có pretrained_cfg mean/std = ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Chia dữ liệu
# --------------------------------------------------------------------------- #
def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1). Không sửa gì."""
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        if not {"Filename", "Label"} <= set(df.columns):
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột Filename/Label")
        df["Label"] = df["Label"].astype(int)
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Lỗi thì raise AssertionError."""
    images_dir = Path(images_dir)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: int(len(v)) for k, v in splits.items()}
    total = sum(n.values())
    frac = {k: v / total for k, v in n.items()}
    per_class = {k: [int((v["Label"] == c).sum()) for c in range(NUM_CLASSES)] for k, v in splits.items()}

    names = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, v in splits.items():
        assert v["Filename"].is_unique, f"{k}: Filename bị trùng trong cùng một tập"
    overlap = {
        "train&val": len(names["train"] & names["val"]),
        "train&test": len(names["train"] & names["test"]),
        "val&test": len(names["val"] & names["test"]),
    }
    union = len(names["train"] | names["val"] | names["test"])
    on_disk = {p.name for p in images_dir.glob("*.jpg")}
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)

    assert all(v == 0 for v in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"
    assert union == TOTAL_IMAGES, f"hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}: {missing[:5]}"
    for k, expected in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(frac[k] - expected) <= 0.01, f"tỉ lệ {k} = {frac[k]:.4f}, lệch > 1 điểm % khỏi {expected}"

    result = {"n": n, "total": total, "fraction": frac, "per_class": per_class, "overlap": overlap,
              "union": union, "missing_files": len(missing), "images_on_disk": len(on_disk)}
    if verbose:
        print(f"Số ảnh: train {n['train']}, val {n['val']}, test {n['test']} (tổng {total}); "
              f"tỉ lệ {frac['train']:.4f}/{frac['val']:.4f}/{frac['test']:.4f}")
        print(f"Giao theo Filename: {overlap}; hợp = {union}; file thiếu trên đĩa = {len(missing)}")
        table = pd.DataFrame(per_class, index=CLASS_NAMES)
        table["total"] = table.sum(1)
        table["paper_table1"] = [PAPER_TABLE1[c] for c in CLASS_NAMES]
        print(table.to_string())
    return result


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
AUG_CHOICES = ("basic", "color", "trivial", "randaug", "vflip")


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic", eval_mode: str = "crop"):
    """Transform trên ảnh uint8 (PIL hoặc tensor CHW). Trả về tensor float đã chuẩn hoá.

    aug (trục B): "basic" = RandomResizedCrop + lật ngang; "color" = basic + ColorJitter(0.3,0.3,0.3,0.05);
                  "trivial" = basic + TrivialAugmentWide; "randaug" = basic + RandAugment(2, 9);
                  "vflip" = basic + lật dọc.
    eval_mode (chỉ khi train=False): "crop" = CenterCrop(img_size) từ ảnh 256 (mặc định, I00);
                  "full" = resize toàn ảnh về img_size (dùng cho dò độ phân giải I04).
    """
    import torch
    from torchvision.transforms import v2

    to_float = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        if eval_mode == "crop":
            geo = [v2.CenterCrop(img_size)] if img_size <= 256 else [v2.Resize(img_size, antialias=True),
                                                                     v2.CenterCrop(img_size)]
        elif eval_mode == "full":
            geo = [] if img_size == 256 else [v2.Resize((img_size, img_size), antialias=True)]
        else:
            raise ValueError(f"eval_mode không hợp lệ: {eval_mode}")
        return v2.Compose([v2.ToImage(), *geo, *to_float])

    if aug not in AUG_CHOICES:
        raise ValueError(f"aug phải thuộc {AUG_CHOICES}, nhận {aug!r}")
    ops = [v2.ToImage(), v2.RandomResizedCrop(img_size, antialias=True), v2.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(v2.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(v2.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(v2.RandAugment(num_ops=2, magnitude=9))
    elif aug == "vflip":
        ops.append(v2.RandomVerticalFlip())
    return v2.Compose([*ops, *to_float])


def denormalize(x):
    """Tensor (C,H,W) hoặc (N,C,H,W) đã chuẩn hoá -> giá trị [0,1] để vẽ."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


# --------------------------------------------------------------------------- #
# Cache ảnh đã giải mã
# --------------------------------------------------------------------------- #
def _decode(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def build_cache(images_dir: str | Path, filenames, out_path: str | Path, workers: int = 4) -> Path:
    """Giải mã mọi ảnh một lần, lưu mảng uint8 (N, 256, 256, 3) + danh sách tên file (.txt)."""
    from concurrent.futures import ThreadPoolExecutor
    out_path = Path(out_path)
    names = list(filenames)
    if out_path.exists() and out_path.with_suffix(".txt").exists():
        if out_path.with_suffix(".txt").read_text().split("\n") == names:
            return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = _decode(str(Path(images_dir) / names[0]))
    arr = np.lib.format.open_memmap(str(out_path) + ".tmp.npy", mode="w+", dtype=np.uint8,
                                    shape=(len(names), *first.shape))

    def work(i):
        img = _decode(str(Path(images_dir) / names[i]))
        assert img.shape == first.shape, f"{names[i]}: kích thước {img.shape} khác {first.shape}"
        arr[i] = img

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, range(len(names))))
    arr.flush()
    del arr
    Path(str(out_path) + ".tmp.npy").replace(out_path)
    out_path.with_suffix(".txt").write_text("\n".join(names))
    return out_path


# --------------------------------------------------------------------------- #
# Dataset và DataLoader
# --------------------------------------------------------------------------- #
try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # torch chưa cài: vẫn import được module để chạy kiểm tra chia dữ liệu
    _TorchDataset = object


class DeepWeedsDataset(_TorchDataset):
    """Dataset đọc ảnh theo DataFrame (Filename, Label). __getitem__ -> (tensor, int, filename).

    cache: đường dẫn .npy do build_cache tạo (tuỳ chọn). Nếu có, đọc ảnh từ memmap thay vì giải mã JPEG.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None, cache: str | Path | None = None):
        self.filenames = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.cache_path = Path(cache) if cache else None
        self._arr = None
        self._index = None
        if self.cache_path is not None:
            names = self.cache_path.with_suffix(".txt").read_text().split("\n")
            pos = {n: i for i, n in enumerate(names)}
            missing = [f for f in self.filenames if f not in pos]
            if missing:
                raise ValueError(f"cache {self.cache_path} thiếu {len(missing)} ảnh")
            self._index = [pos[f] for f in self.filenames]

    def __len__(self) -> int:
        return len(self.filenames)

    def load_image(self, i: int):
        """Ảnh gốc dạng PIL RGB (256x256)."""
        from PIL import Image
        if self._index is not None:
            if self._arr is None:  # mở memmap lười trong từng worker
                self._arr = np.load(self.cache_path, mmap_mode="r")
            return Image.fromarray(np.array(self._arr[self._index[i]]))
        with Image.open(self.images_dir / self.filenames[i]) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], self.filenames[i]


def seed_worker(worker_id: int) -> None:
    """Seed cho worker của DataLoader (torch đã đặt seed riêng cho mỗi worker từ generator)."""
    import torch
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0, cache: str | Path | None = None):
    """DataLoader. train=False giữ đúng thứ tự df (để ghép logit với Filename)."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform, cache=cache)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    shuffle = train
    if train and sampler == "balanced":
        counts = np.bincount(df["Label"].to_numpy(), minlength=NUM_CLASSES).astype(np.float64)
        w = 1.0 / counts[df["Label"].to_numpy()]
        smp = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df),
                                    replacement=True, generator=g)
        shuffle = False
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=smp, drop_last=train,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=g,
                      persistent_workers=num_workers > 0)
