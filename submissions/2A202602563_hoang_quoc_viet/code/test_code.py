"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Chạy được trên CPU, không cần dữ liệu:

    cd submissions/2A202602563_hoang_quoc_viet/code
    python -m unittest test_code -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark  # noqa: E402
import dataset as D  # noqa: E402
import inference as I  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
import train as T  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(64, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (64,), generator=g)

    def test_focal_gamma0_equals_ce(self):
        fl = L.FocalLoss(gamma=0.0)(self.logits, self.y)
        ce = F.cross_entropy(self.logits, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(L.FocalLoss(gamma=2.0)(self.logits, self.y).item(),
                        F.cross_entropy(self.logits, self.y).item())

    def test_label_smoothing_eps0_equals_ce_and_matches_torch(self):
        self.assertLess(abs(L.LabelSmoothingCE(0.0)(self.logits, self.y).item()
                            - F.cross_entropy(self.logits, self.y).item()), 1e-6)
        ours = L.LabelSmoothingCE(0.1)(self.logits, self.y).item()
        ref = F.cross_entropy(self.logits, self.y, label_smoothing=0.1).item()
        self.assertLess(abs(ours - ref), 1e-6)

    def test_class_weights(self):
        counts = [100, 100, 50, 200, 100, 100, 100, 100, 1000]
        w = L.class_weights(counts)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertGreater(w[2].item(), w[8].item())
        wb = L.class_weights(counts, beta=0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)

    def test_cutmix_mixes_images_and_labels_with_area_lam(self):
        x = torch.zeros(8, 3, 32, 32)
        for i in range(8):
            x[i] = i  # mỗi ảnh một giá trị hằng để biết pixel đến từ ảnh nào
        y = torch.arange(8)
        rng = np.random.default_rng(3)
        for _ in range(20):
            xm, (ya, yb, lam) = L.mix_batch(x, y, alpha=1.0, mode="cutmix", rng=rng)
            self.assertTrue(torch.equal(ya, y))
            for i in range(8):
                kept = (xm[i, 0] == i).float().mean().item()       # phần còn của ảnh gốc
                pasted = (xm[i, 0] == yb[i].item()).float().mean().item()
                if yb[i].item() == i:
                    continue
                # lam phải bằng đúng tỉ lệ diện tích còn lại của ảnh gốc (đã sửa theo biên)
                self.assertAlmostEqual(kept, lam, places=6)
                self.assertAlmostEqual(pasted, 1 - lam, places=6)

    def test_mixup_and_mixed_loss(self):
        x = torch.randn(4, 3, 8, 8)
        y = torch.tensor([0, 1, 2, 3])
        xm, (ya, yb, lam) = L.mix_batch(x, y, 0.4, "mixup", rng=np.random.default_rng(1))
        perm = [int((yb == k).nonzero()) for k in range(4)]  # noqa: F841 (chỉ để chắc yb là hoán vị)
        self.assertTrue(torch.allclose(xm, lam * x + (1 - lam) * x[yb], atol=1e-6))
        crit = nn.CrossEntropyLoss()
        logits = torch.randn(4, 9)
        expected = lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)
        self.assertAlmostEqual(L.mixed_loss(crit, logits, (ya, yb, lam)).item(), expected.item(), places=6)


class TestModelAndTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import timm
        torch.manual_seed(0)
        cls.net = timm.create_model("resnet18", pretrained=False, num_classes=9)

    def test_param_groups_no_decay_on_norm_bias(self):
        groups = M.param_groups(self.net, 1e-4, 1e-3, 0.05)
        by = {g["name"]: g for g in groups}
        self.assertEqual(by["backbone_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(by["head_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(by["backbone_decay"]["weight_decay"], 0.05)
        self.assertEqual(by["head_decay"]["lr"], 1e-3)
        self.assertTrue(all(p.ndim <= 1 for p in by["backbone_no_decay"]["params"]))
        n = sum(len(g["params"]) for g in groups)
        self.assertEqual(n, len(list(self.net.parameters())))

    def test_frozen_backbone_keeps_bn_in_eval(self):
        import copy
        net = copy.deepcopy(self.net)
        M.freeze_backbone(net)
        M.set_train_mode(net)
        self.assertFalse(net.bn1.training)
        self.assertTrue(net.get_classifier().training)
        trainable = [n for n, p in net.named_parameters() if p.requires_grad]
        self.assertEqual(sorted(trainable), ["fc.bias", "fc.weight"])

    def test_gmacs_resnet18(self):
        g = M.count_gmacs(self.net, 224)
        self.assertTrue(1.7 < g < 1.9, g)   # resnet18 ~1.82 GMAC

    def test_lr_schedule_warmup_then_cosine(self):
        total, warm = 100, 10
        f = [T.lr_factor(s, total, warm) for s in range(total)]
        self.assertAlmostEqual(f[9], 1.0)
        self.assertTrue(all(a < b for a, b in zip(f[:9], f[1:10])))
        self.assertTrue(all(a >= b for a, b in zip(f[10:], f[11:])))
        self.assertLess(f[-1], 0.01)

    def test_ema_update(self):
        lin = nn.Linear(2, 2)
        ema = T.EMA(lin, 0.9)
        with torch.no_grad():
            old = ema.module.weight.clone()
            lin.weight.add_(1.0)
        ema.update(lin)
        self.assertTrue(torch.allclose(ema.module.weight, 0.9 * old + 0.1 * lin.weight))

    def test_config_and_overrides(self):
        o = T.parse_overrides(["seed=2", "loss=focal", "ema_decay=none", "amp=false", "lr_head=0.002",
                               "sampler=balanced"])
        self.assertEqual(o, {"seed": 2, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 0.002,
                             "sampler": "balanced"})
        with self.assertRaises(KeyError):
            T.parse_overrides(["nope=1"])
        c = T.Config()
        self.assertEqual((c.epochs, c.batch_size, c.lr_backbone, c.lr_head, c.weight_decay),
                         (12, 64, 1e-4, 1e-3, 0.05))
        self.assertFalse(c.save_test_predictions)


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_exact(self):
        import timm
        torch.manual_seed(0)
        for name in ("resnet18", "efficientnet_b0", "mobilenetv3_large_100"):
            net = timm.create_model(name, pretrained=False, num_classes=9).eval()
            for m in net.modules():  # thống kê BN khác mặc định để phép gộp không tầm thường
                if isinstance(m, nn.BatchNorm2d):
                    m.running_mean.uniform_(-0.5, 0.5)
                    m.running_var.uniform_(0.5, 2.0)
                    m.weight.data.uniform_(0.5, 1.5)
                    m.bias.data.uniform_(-0.2, 0.2)
            fused = I.fuse_conv_bn(net)
            self.assertGreater(fused.n_fused, 10, name)
            remaining = sum(isinstance(m, nn.BatchNorm2d) for m in fused.modules())
            self.assertEqual(remaining, 0, f"{name}: còn {remaining} BN chưa gộp")
            x = torch.randn(2, 3, 224, 224)
            diff = I.max_abs_diff(net, fused, x)
            self.assertLessEqual(diff, 1e-5 * max(1.0, net(x).abs().max().item()), f"{name}: {diff}")

    def test_temperature_recovers_known_T(self):
        rng = np.random.default_rng(0)
        n, true_t = 20000, 2.5
        z = rng.normal(size=(n, 9)) * 4
        p = I.apply_temperature(z, true_t)
        y = np.array([rng.choice(9, p=row) for row in p])
        t = I.fit_temperature(z, y)
        self.assertLess(abs(t - true_t) / true_t, 0.05, t)

    def test_aggregate_views_and_ensemble(self):
        a = np.array([[2.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 0.0], [0.0, 3.0]])
        pl = I.aggregate_views([a, b], "logit")
        pp = I.aggregate_views([a, b], "prob")
        np.testing.assert_allclose(pl.sum(1), 1)
        np.testing.assert_allclose(pp, (I._softmax(a) + I._softmax(b)) / 2)
        np.testing.assert_allclose(I.ensemble_probs([I._softmax(a), I._softmax(b)]), pp)

    def test_views(self):
        x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
        self.assertTrue(torch.equal(I.view_hflip(I.view_hflip(x)), x))
        self.assertEqual(I.view_hflip(x)[0, 0, 0, 0].item(), x[0, 0, 0, -1].item())
        crops = I.views_multicrop(x, 6, flip=True)
        self.assertEqual(len(crops), 10)
        self.assertTrue(all(c.shape == (2, 3, 6, 6) for c in crops))
        self.assertEqual([v.shape[-1] for v in I.views_multiscale(x, [8, 12])], [8, 12])


class TestDataAndBenchmark(unittest.TestCase):
    def test_transforms_shapes(self):
        from PIL import Image
        img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
        for aug in D.AUG_CHOICES:
            self.assertEqual(tuple(D.build_transforms(True, 224, aug)(img).shape), (3, 224, 224))
        ev = D.build_transforms(False, 224)
        a, b = ev(img), ev(img)
        self.assertTrue(torch.equal(a, b))  # đánh giá không có ngẫu nhiên
        self.assertEqual(tuple(D.build_transforms(False, 320, eval_mode="full")(img).shape), (3, 320, 320))

    def test_bench_requires_50_and_reports_percentiles(self):
        with self.assertRaises(ValueError):
            benchmark.bench(lambda: None, iters=10)
        r = benchmark.bench(lambda: math.sqrt(2.0), warmup=2, iters=60)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])


if __name__ == "__main__":
    unittest.main()
