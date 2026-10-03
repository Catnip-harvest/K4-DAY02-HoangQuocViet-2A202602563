"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mọi hàm chạy ở chế độ eval, không gradient. Phương pháp được chọn CHỈ dựa trên val; nhiệt độ T
khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện (giữ nguyên theo starter/):
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None, amp: bool = True):
    """Chạy model trên loader, gom logit theo đúng thứ tự file. `view(x)` biến đổi batch (hoặc None).

    Nếu `view` trả về list các batch (multi-crop/multi-scale), trả về list logit tương ứng (mỗi view một mảng).
    """
    model.eval()
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            xs = view(x) if view is not None else x
            multi = isinstance(xs, (list, tuple))
            xs = list(xs) if multi else [xs]
            if outs is None:
                outs = [[] for _ in xs]
            for i, xi in enumerate(xs):
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp and device.type == "cuda"):
                    outs[i].append(model(xi.contiguous(memory_format=torch.channels_last)).float().cpu())
            ys.append(y)
            names.extend(f)
    y_true = torch.cat(ys).numpy()
    logits = [torch.cat(o).numpy() for o in outs]
    return names, y_true, (logits if len(logits) > 1 else logits[0])


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W): đảo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_hflip_pair(x):
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch (N,C,H,W); tuỳ chọn thêm bản lật -> 10 view."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    tops = [0, 0, h - crop, h - crop, (h - crop) // 2]
    lefts = [0, w - crop, 0, w - crop, (w - crop) // 2]
    views = [x[..., t:t + crop, l:l + crop] for t, l in zip(tops, lefts)]
    if flip:
        views += [view_hflip(v) for v in views]
    return views


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias). CNN/ConvNeXt có global
    pooling nên nhận được mọi kích thước; ViT/Swin cần nội suy vị trí/cửa sổ nên không dùng ở đây."""
    out = []
    for s in sizes:
        out.append(x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear",
                                                            antialias=True, align_corners=False))
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: "prob" = trung bình softmax; "logit" = trung bình logit rồi softmax. Trả về xác suất."""
    stack = np.stack([np.asarray(v, dtype=np.float64) for v in logits_per_view])
    if space == "prob":
        p = np.mean([_softmax(v) for v in stack], axis=0)
    elif space == "logit":
        p = _softmax(stack.mean(0))
    else:
        raise ValueError("space phải là prob hoặc logit")
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình trên cùng tập ảnh, cùng thứ tự."""
    shapes = {np.asarray(p).shape for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"các mô hình có số dòng khác nhau: {shapes}")
    p = np.mean([np.asarray(p, dtype=np.float64) for p in list_of_probs], axis=0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL của softmax(logit / T). Tìm lưới log-T rồi tinh bằng LBFGS trên log T."""
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    nll = [F.cross_entropy(z / t, y).item() for t in grid]
    log_t = torch.tensor([np.log(grid[int(np.argmin(nll))])], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode).to(conv.weight.device)
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(conv.out_channels, dtype=torch.float64,
                                                                              device=w.device)
    std = torch.sqrt(bn.running_var.double() + bn.eps)
    gamma = bn.weight.detach().double() if bn.weight is not None else torch.ones_like(std)
    beta = bn.bias.detach().double() if bn.bias is not None else torch.zeros_like(std)
    fused.weight.data = (w * (gamma / std).reshape(-1, 1, 1, 1)).to(conv.weight.dtype)
    fused.bias.data = (beta + gamma * (b - bn.running_mean.double()) / std).to(conv.weight.dtype)
    return fused


def fuse_conv_bn(model):
    """Gộp mọi cặp (Conv2d -> BatchNorm2d) liền kề (theo thứ tự đăng ký trong cùng module cha) vào conv:

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    BN được thay bằng Identity. Trả về bản sao đã gộp (model gốc không đổi) ở chế độ eval, kèm thuộc tính
    `n_fused`. Hỗ trợ timm BatchNormAct2d (BN + activation): giữ lại phần activation/drop.
    Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm) -> n_fused = 0, không áp dụng.
    """
    try:
        from timm.layers import BatchNormAct2d
    except Exception:  # pragma: no cover
        BatchNormAct2d = ()
    m = copy.deepcopy(model).eval()
    n = 0
    for parent in m.modules():
        children = list(parent.named_children())
        for (n1, c1), (n2, c2) in zip(children, children[1:]):
            if isinstance(c1, nn.Conv2d) and isinstance(c2, nn.BatchNorm2d) and c2.track_running_stats:
                setattr(parent, n1, _fuse(c1, c2))
                if BatchNormAct2d and isinstance(c2, BatchNormAct2d):
                    act = nn.Sequential(c2.drop, c2.act)
                    setattr(parent, n2, act)
                else:
                    setattr(parent, n2, nn.Identity())
                n += 1
    # timm ConvNormAct: conv và bn nằm trong cùng module (thuộc tính .conv và .bn) -> đã được xử lý ở trên
    m.n_fused = n
    return m


def max_abs_diff(model_a, model_b, x) -> float:
    """Sai số lớn nhất giữa đầu ra hai model (FP32, eval) trên batch x."""
    with torch.inference_mode():
        return float((model_a.eval()(x) - model_b.eval()(x)).abs().max().item())


# --------------------------------------------------------------------------- #
# Tất cả view tính trong MỘT lượt qua dữ liệu (ảnh gốc 256x256 đã chuẩn hoá, view tạo trên GPU)
# --------------------------------------------------------------------------- #
def _crop(x, top, left, size=224):
    return x[..., top:top + size, left:left + size]


def _resize(x, s):
    return x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear", antialias=True,
                                                    align_corners=False)


VIEW_FNS = {
    "c224": lambda x: _crop(x, 16, 16),                       # = CenterCrop(224) của ảnh 256 (I00)
    "c224f": lambda x: view_hflip(_crop(x, 16, 16)),
    "tl": lambda x: _crop(x, 0, 0), "tr": lambda x: _crop(x, 0, 32),
    "bl": lambda x: _crop(x, 32, 0), "br": lambda x: _crop(x, 32, 32),
    "tlf": lambda x: view_hflip(_crop(x, 0, 0)), "trf": lambda x: view_hflip(_crop(x, 0, 32)),
    "blf": lambda x: view_hflip(_crop(x, 32, 0)), "brf": lambda x: view_hflip(_crop(x, 32, 32)),
    "full224": lambda x: _resize(x, 224), "full256": lambda x: x,
    "full288": lambda x: _resize(x, 288), "full320": lambda x: _resize(x, 320),
}

# Phương pháp = danh sách view (K = số view). Gộp theo "prob" hoặc "logit" chọn lúc tính.
METHODS = {
    "I00_1view": ["c224"],
    "I01_hflip": ["c224", "c224f"],
    "I02a_5crop": ["c224", "tl", "tr", "bl", "br"],
    "I02b_10crop": ["c224", "tl", "tr", "bl", "br", "c224f", "tlf", "trf", "blf", "brf"],
    "I02c_multiscale": ["full224", "full256", "full288"],
    "I04_full224": ["full224"], "I04_full256": ["full256"], "I04_full288": ["full288"],
    "I04_full320": ["full320"],
}


def compute_view_logits(model, loader, device, views, amp: bool = True) -> tuple[list, np.ndarray, dict]:
    """Một lượt qua `loader` (ảnh 256 đầy đủ); trả về (filenames, y_true, {view: logits[N,9]})."""
    model.eval()
    names, ys = [], []
    outs = {v: [] for v in views}
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            for v in views:
                xi = VIEW_FNS[v](x).contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=amp and device.type == "cuda"):
                    outs[v].append(model(xi).float().cpu())
            ys.append(y)
            names.extend(f)
    return names, torch.cat(ys).numpy(), {v: torch.cat(o).numpy() for v, o in outs.items()}


def method_logits(view_logits: dict, method: str):
    return [view_logits[v] for v in METHODS[method]]


def method_probs(view_logits: dict, method: str, space: str = "logit", T: float = 1.0):
    """Xác suất của một phương pháp. space="logit": softmax(mean_k z_k / T); "prob": mean_k softmax(z_k / T)."""
    zs = [z / T for z in method_logits(view_logits, method)]
    return aggregate_views(zs, space)


def fit_temperature_method(view_logits: dict, y, method: str, space: str = "logit") -> float:
    """T cho phương pháp nhiều view. Với gộp logit: z = mean_k z_k rồi fit_temperature như 1 view.
    Với gộp xác suất: tìm lưới log-T cực tiểu NLL của mean_k softmax(z_k / T)."""
    if space == "logit":
        return fit_temperature(np.mean(method_logits(view_logits, method), axis=0), y)
    zs = method_logits(view_logits, method)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 2000))
    y = np.asarray(y)

    def nll(t):
        p = aggregate_views([z / t for z in zs], "prob")
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()

    return float(grid[int(np.argmin([nll(t) for t in grid]))])
