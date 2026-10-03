"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện (giữ nguyên theo starter/):
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float

Thêm:
    weight_tag(model)            -> tag trọng số timm thực sự được tải (ví dụ "resnet50.a1_in1k")
    set_train_mode(model, init)  -> model.train() nhưng giữ backbone đóng băng ở eval (BN không cập nhật)
"""
from __future__ import annotations

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}
INIT_CHOICES = ("scratch", "frozen", "finetune")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", head_init_std: float | None = 0.01):
    """Tạo model phân loại 9 lớp qua timm; head mới khởi tạo ngẫu nhiên.

    init: "scratch" (không tải trọng số) | "frozen" (tải, đóng băng backbone) | "finetune" (tải, train hết).
    head_init_std: khởi tạo lại head mới bằng trunc_normal(std) và bias 0 cho MỌI backbone (đồng nhất).
        Lý do (kiểm tra loss ban đầu, GUIDE 1.3): khởi tạo mặc định của timm cho head EfficientNet/MobileNetV3
        là uniform(+-1/sqrt(9)) nên loss ban đầu ~5-7 thay vì ln 9 = 2.197. None = giữ khởi tạo của timm.
    """
    import timm
    if init not in INIT_CHOICES:
        raise ValueError(f"init phải thuộc {INIT_CHOICES}, nhận {init!r}")
    use_pretrained = pretrained and init != "scratch"
    model = timm.create_model(name, pretrained=use_pretrained, num_classes=num_classes, drop_rate=drop_rate)
    model.weight_tag = weight_tag(model) if use_pretrained else "none (random init)"
    if head_init_std is not None:
        import torch.nn as nn
        head = model.get_classifier()
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=head_init_std)
                nn.init.zeros_(m.bias)
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model) -> str:
    cfg = getattr(model, "pretrained_cfg", None) or {}
    arch = cfg.get("architecture", "")
    tag = cfg.get("tag", "")
    return f"{arch}.{tag}" if tag else (arch or "unknown")


def _head_param_ids(model) -> set[int]:
    head = model.get_classifier()
    return {id(p) for p in head.parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head; ghi lại để set_train_mode giữ backbone ở eval."""
    head_ids = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(); nếu backbone bị đóng băng thì mọi module ngoài head trở lại eval().

    Lý do: BatchNorm ở train mode vẫn cập nhật running_mean/var (và dùng thống kê batch) dù
    requires_grad=False, nên backbone "đóng băng" sẽ trôi khỏi trọng số tiền huấn luyện.
    """
    model.train()
    if getattr(model, "frozen_backbone", False):
        head = model.get_classifier()
        head_modules = set(head.modules())
        for m in model.modules():
            if m is not model and m not in head_modules:
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Nhóm tham số (slide trang 52):
       - backbone ndim > 1          : lr_backbone, weight_decay
       - backbone norm/bias (ndim<=1): lr_backbone, weight_decay = 0
       - head weight                : lr_head, weight_decay
       - head bias                  : lr_head, weight_decay = 0 (bias không bị phạt)
    Bỏ qua tham số requires_grad == False. Bỏ nhóm rỗng.
    """
    head_ids = _head_param_ids(model)
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "backbone"
        decay = "decay" if p.ndim > 1 else "no_decay"
        groups[f"{part}_{decay}"].append(p)
    spec = {
        "backbone_decay": (lr_backbone, weight_decay),
        "backbone_no_decay": (lr_backbone, 0.0),
        "head_decay": (lr_head, weight_decay),
        "head_no_decay": (lr_head, 0.0),
    }
    return [{"params": ps, "lr": spec[k][0], "weight_decay": spec[k][1], "name": k}
            for k, ps in groups.items() if ps]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size = FLOPs / 2 đếm bằng torch.utils.flop_counter.

    FlopCounterMode đếm FLOPs của các phép aten (conv, matmul/bmm, attention), tức đã gồm cả phép
    nhân ma trận trong self-attention (hook theo module Conv/Linear sẽ bỏ sót phần này).
    Đếm trên CPU với FP32, model ở eval.
    """
    import copy

    import torch
    from torch.utils.flop_counter import FlopCounterMode

    m = copy.deepcopy(model).float().cpu().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        m(x)
    return counter.get_total_flops() / 2 / 1e9
