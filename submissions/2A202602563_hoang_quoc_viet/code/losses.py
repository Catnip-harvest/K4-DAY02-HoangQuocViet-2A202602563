"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện (giữ nguyên theo starter/):
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar

Lựa chọn cài đặt:
    - LabelSmoothingCE tự cài đặt theo công thức q'(k) = (1-eps)*1[k=y] + eps/K (eps=0 trùng CE).
    - FocalLoss tự cài đặt; gamma=0, alpha=None trùng CE (kiểm tra trong test_code.py).
    - CutMix điều chỉnh lam theo diện tích thật của hộp sau khi cắt ở biên.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với nhãn mềm q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56)."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        k = logits.shape[-1]
        q = torch.full_like(logp, self.smoothing / k)
        q.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / k)
        return -(q * logp).sum(-1).mean()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), trung bình theo batch (slide trang 57)."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t).clamp(min=0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


class WeightedCE(nn.Module):
    """CE có trọng số theo lớp; chuẩn hoá theo tổng trọng số của batch (giống nn.CrossEntropyLoss)."""

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits, target):
        return F.cross_entropy(logits.float(), target, weight=self.weight.to(logits.device))


def build_criterion(kind: str = "ce", **kw):
    """kind: "ce" | "ls" (label smoothing, kw smoothing) | "focal" (kw gamma, alpha) | "ce_weighted" (kw weight)."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần weight (dùng class_weights(counts_train))")
        return WeightedCE(kw["weight"])
    raise ValueError(f"loss không hợp lệ: {kind}")


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp của TRAIN.

    beta = 0 : w_c ∝ 1 / n_c, chuẩn hoá để trung bình bằng 1.
    beta > 0 : class-balanced (Cui et al.): w_c = (1 - beta) / (1 - beta ** n_c), chuẩn hoá tổng = K.
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh")
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
    else:
        w = 1.0 / n
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator):
    """Hộp có diện tích ~ (1 - lam) * H * W, tâm ngẫu nhiên, cắt ở biên. Trả về (y1, y2, x1, x2)."""
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = int(rng.integers(h)), int(rng.integers(w))
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", rng: np.random.Generator | None = None):
    """Trộn batch. Trả về (x_mix, (y_a, y_b, lam)); y_a = y, y_b = y[perm].

    mixup : x_mix = lam * x + (1 - lam) * x[perm]
    cutmix: dán hộp từ x[perm] vào x; lam = 1 - diện tích hộp thật / (H * W).
    """
    rng = rng or np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
    lam = float(rng.beta(alpha, alpha))
    perm = torch.as_tensor(rng.permutation(x.shape[0]), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        y1, y2, x1, x2 = rand_bbox(h, w, lam, rng)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(h * w)
    else:
        raise ValueError(f"mode phải là mixup hoặc cutmix, nhận {mode!r}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
