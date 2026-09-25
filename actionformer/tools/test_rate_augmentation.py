"""Checks for the feature-level rate augmentation (run from actionformer/: python tools/test_rate_augmentation.py)."""
import sys, os; sys.path.insert(0, os.getcwd())
import random, torch
from train_ssl import build_views, build_roi
from libs.modeling.selfsup import temporal_roi_align

torch.manual_seed(0); random.seed(0)
C, T = 3, 1000
ramp = torch.arange(T, dtype=torch.float32)
feats = torch.stack([ramp, torch.randn(T), torch.randn(T)])   # ch0 = raw position

# 1) rate 1 is an exact slice (bit-identical to the old behaviour)
v = [{'feats': feats, 'crop_box': ((100, 400), (250, 700)), 'crop_rate': (1.0, 1.0)}]
x, m, st, r = build_views(v, 1, 32, 2304, 'cpu')
assert torch.equal(x[0, :, :450], feats[:, 250:700]) and m[0, 0, :450].all() and not m[0, 0, 450:].any()
v_old = [{'feats': feats, 'crop_box': ((100, 400), (250, 700))}]      # no crop_rate key
x_old, *_ = build_views(v_old, 1, 32, 2304, 'cpu')
assert torch.equal(x, x_old)
print("ok: rate 1 == plain slice; missing crop_rate defaults to 1")

# 2) resampled view index j sits at raw position s + j*r, inside [s, e-1]
for rate in (0.5, 0.73, 1.6, 2.0):
    v = [{'feats': feats, 'crop_box': ((100, 400), (250, 700)), 'crop_rate': (rate, rate)}]
    x, m, st, r = build_views(v, 0, 32, 2304, 'cpu')
    L = int(m.sum())
    exp = 100 + torch.arange(L) * rate
    assert torch.allclose(x[0, 0, :L], exp, atol=1e-3), rate
    assert exp[-1] <= 399 + 1e-6 and 100 + L * rate > 399
    print("ok: rate %.2f -> %d vectors covering raw [100, %.1f]" % (rate, L, exp[-1]))

# 3) key invariant: with forward_view's box mapping, both views align the SAME
#    raw moments onto each ROI bin, whatever their rates
for r1, r2 in ((1.0, 1.0), (0.5, 2.0), (1.7, 0.6), (2.0, 2.0)):
    v = [{'feats': feats, 'crop_box': ((100, 400), (250, 700)), 'crop_rate': (r1, r2)}]
    roi = build_roi(v, 'cpu')
    outs, inside = [], torch.ones(32, dtype=torch.bool)
    target = 250 + (400 - 250) * (torch.arange(32) + 0.5) / 32
    for vi in (0, 1):
        x, m, st, rr = build_views(v, vi, 32, 2304, 'cpu')
        stride = 1.0 * rr                     # level-0 stride times view rate
        boxes = torch.stack([(roi[:, 0] - st) / stride, (roi[:, 1] - st) / stride], 1)
        a, am = temporal_roi_align(x, m, boxes, 32)
        outs.append(a[0, 0])
        last = float(st[0] + (int(m.sum()) - 1) * rr[0])   # raw pos of last valid sample
        inside &= target <= last + 1e-6
    for o in outs:
        assert torch.allclose(o[inside], target[inside], atol=1e-3), (r1, r2)
    print("ok: rates (%.1f, %.1f): both views hit identical raw ROI bins "
          "(%d/32 bins; %d edge bin(s) past a view's last sample, as at rate 1)"
          % (r1, r2, int(inside.sum()), 32 - int(inside.sum())))
print("ALL CHECKS PASSED")
