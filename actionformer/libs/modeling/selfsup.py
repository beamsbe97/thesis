"""
Self-supervised post-training for ActionFormer, following NeCo
(Pariza et al., "NeCo: Improving DINOv2's spatial representations with Patch
Neighbor Consistency", arXiv:2408.11054), adapted from spatial (2D) crops and
2D ROI-Align to *temporal (1D) crops* and a 1D ROI-Align.

The teacher / student encoders are the multi-scale transformer encoder of
ActionFormer (ConvTransformerBackbone + FPN neck). For each video we sample
two temporal crops. One is encoded by the student, the other by an EMA
teacher. Both feature maps are aligned to the same temporal ROI (the crop
intersection) with a 1D ROI Align, projected with a small MLP head, and the
loss enforces that the *order of nearest neighbors* (relative to a reference
pool of other videos in the batch) is consistent between student and teacher
(differentiable sorting + cross entropy, both directions).

This is an add-on: the original supervised ActionFormer training path is
untouched. Loss variants beyond 'neco' ('dino', 'byol', 'masked') are
reserved behind the same dispatch (see make_ssl_loss) for later extension.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .models import register_meta_arch, make_backbone, make_neck
from .blocks import LayerNorm, MaskedConv1D, Scale, AffineDropPath


SUPPORTED_SSL_METHODS = ('neco', 'dino', 'byol', 'masked')


################################################################################
def temporal_roi_align(x, mask, boxes, out_size):
    """
    1D ROI Align (spatial analog: 2D ROI Align of Mask R-CNN, hence 1D linear
    interpolation along time).

    Samples `out_size` equally-spaced bin centers inside each ROI box (defined
    in the coordinate system of `x`) and linearly interpolates the feature
    sequence at those positions.

    Args:
        x:      (B, C, T) feature sequence
        mask:   (B, 1, T) bool, valid positions of x
        boxes:  (B, 2) float, (start, end) ROI in x's temporal coordinates
        out_size: int, number of temporal bins
    Returns:
        out:      (B, C, out_size)
        out_mask: (B, 1, out_size) float, interpolated validity in [0, 1]
    """
    B, C, T = x.size()
    device = x.device
    dtype = x.dtype
    # bin centers in x's coords (same for all samples shape B x O)
    arange = torch.arange(out_size, device=device, dtype=dtype)
    rois = boxes[:, 0].unsqueeze(1) + \
        (boxes[:, 1] - boxes[:, 0]).unsqueeze(1) * (arange + 0.5) / out_size
    rois = rois.clamp(min=0.0, max=T - 1.0)          # (B, O)
    lo = rois.long()                                  # (B, O)
    hi = (lo + 1).clamp(max=T - 1)                    # (B, O)
    w = (rois - lo.to(dtype)).unsqueeze(1)            # (B, 1, O)

    lo_e = lo.unsqueeze(1).expand(B, C, out_size)
    hi_e = hi.unsqueeze(1).expand(B, C, out_size)
    out = x.gather(2, lo_e) * (1.0 - w) + x.gather(2, hi_e) * w

    mlo = mask.to(dtype).gather(2, lo.unsqueeze(1).expand(B, 1, out_size))
    mhi = mask.to(dtype).gather(2, hi.unsqueeze(1).expand(B, 1, out_size))
    out_mask = mlo * (1.0 - w) + mhi * w
    return out, out_mask


class SSLProjectionHead(nn.Module):
    """
    Position-wise MLP projection head, applied to every temporal position of
    *each* FPN level (DINO/NeCo style), followed by an optional linear mixer
    that fuses the ROI-aligned multi-scale tokens.
    """
    def __init__(
        self,
        in_dim,               # per-level input dim (fpn_dim)
        out_dim=256,
        hidden_dim=2048,
        num_layers=3,
        num_levels=1,
        with_mixer=True,
        dropout=0.0,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            cin = in_dim if i == 0 else hidden_dim
            cout = out_dim if (i == num_layers - 1) else hidden_dim
            layers.append(nn.Conv1d(cin, cout, 1))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(self._init_weights)

        self.with_mixer = with_mixer
        if with_mixer:
            self.mixer = nn.Conv1d(num_levels * out_dim, out_dim, 1)
            self.mixer_norm = LayerNorm(out_dim)
            self.mixer.apply(self._init_weights)
        else:
            self.mixer = None
            self.mixer_norm = None

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv1d):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, feats):
        # feats: list of (B, in_dim, T_l) for each FPN level (lengths differ)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        return [self.mlp(f) for f in feats]


################################################################################
def soft_sort_perm(dist, tau_rank=0.1, tau_perm=0.1, top_k=0):
    """
    Differentiable "soft sorting" of each row of a (N, R) distance matrix
    (lower = closer), based on SoftSort (Prillo & Eisenschlos, NeurIPS 2020).

    For every row, returns a (K, R) soft permutation Q[k, r] = probability
    that reference r occupies the k-th nearest-neighbor slot (k = 0 nearest).

        rank_r = sum_j sigmoid((dist_r - dist_j) / tau_rank)
        Q[k, r] = softmax_r( -|k - rank_r| / tau_perm )
    """
    N, R = dist.size()
    K = R if (top_k is None or top_k <= 0) else min(top_k, R)
    if K <= 0 or R == 0:
        return dist.new_zeros(0, 0, 0)

    rank = torch.sigmoid(
        (dist.unsqueeze(2) - dist.unsqueeze(1)) / tau_rank   # (N, R, R)
    ).sum(dim=2)                                             # (N, R) soft rank
    k = torch.arange(K, device=dist.device, dtype=dist.dtype).view(1, K, 1)
    logits = -(k - rank.unsqueeze(1)).abs() / tau_perm       # (N, K, R)
    return F.softmax(logits, dim=2)


class NeCoLoss(nn.Module):
    """
    NeCo: patch (neighbor) consistency across student and EMA teacher views.
    References are sampled from the *other* videos of the batch (student
    features, stop-gradient), mirroring NeCo's reference batch.
    """
    def __init__(self, ref_fraction=0.5, tau_rank=0.1, tau_perm=0.1, top_k=0):
        super().__init__()
        self.ref_fraction = ref_fraction
        self.tau_rank = tau_rank
        self.tau_perm = tau_perm
        self.top_k = top_k

    def forward(self, f_s, f_t, masks=None):
        # f_s, f_t: (B, N, D) L2-normalized anchor features (student / teacher)
        # masks:    (B, N) bool, valid anchors (aligned bins inside the crop)
        B, N, D = f_s.size()
        assert B >= 2, "NeCo requires batch_size >= 2 (reference pool)"

        if masks is None:
            masks = f_s.new_ones(B, N, dtype=torch.bool)

        n_ref_target = max(1, int(round(N * self.ref_fraction)))

        total_loss = f_s.new_zeros(())
        total_acc = f_s.new_zeros(())
        num_anchors = 0
        for b in range(B):
            vb = masks[b]
            a_s = f_s[b][vb]                 # (Na, D)
            a_t = f_t[b][vb]                 # (Na, D)
            if a_s.size(0) == 0:
                continue

            # reference pool = valid bins of the OTHER videos (student, no grad)
            refs = []
            for j in range(B):
                if j == b:
                    continue
                vj = masks[j]
                fj = f_s[j][vj]              # (Nj, D)
                mj = min(n_ref_target, fj.size(0))
                if mj == 0:
                    continue
                idxj = torch.randperm(fj.size(0), device=f_s.device)[:mj]
                refs.append(fj[idxj])
            if len(refs) == 0:
                continue
            o = torch.cat(refs, dim=0).detach()      # (R, D)
            R = o.size(0)
            K = R if self.top_k <= 0 else min(self.top_k, R)

            dist_s = -a_s @ o.t()           # (Na, R) cosine distance
            dist_t = -a_t @ o.t()           # (Na, R)
            q_s = soft_sort_perm(dist_s, self.tau_rank, self.tau_perm, K)
            q_t = soft_sort_perm(dist_t, self.tau_rank, self.tau_perm, K)
            q_s = q_s.clamp_min(1e-7)
            q_t = q_t.clamp_min(1e-7)

            # NeCo loss: CE of neighbor ordering in both directions
            ce = (-(q_t * torch.log(q_s)).sum(dim=(1, 2))
                  - (q_s * torch.log(q_t)).sum(dim=(1, 2))) / K
            total_loss = total_loss + ce.sum()
            num_anchors += a_s.size(0)

            with torch.no_grad():
                top_s = dist_s.argmin(dim=1)
                top_t = dist_t.argmin(dim=1)
                total_acc += (top_s == top_t).sum().to(total_acc.dtype)

        if num_anchors > 0:
            loss = total_loss / num_anchors
            acc = total_acc / num_anchors
        else:
            loss = total_loss
            acc = total_acc

        return {'neco_loss': loss,
                'neco_top1_agreement': acc,
                'final_loss': loss}


def make_ssl_loss(method, ssl_cfg=None):
    """Bounded dispatch for future ssl losses (add-on friendly flag)."""
    ssl_cfg = ssl_cfg or {}
    loss_cfg = ssl_cfg.get('loss', {}) or {}

    if method == 'neco':
        return NeCoLoss(
            ref_fraction=loss_cfg.get('ref_fraction', 0.5),
            tau_rank=loss_cfg.get('tau_rank', 0.1),
            tau_perm=loss_cfg.get('tau_perm', 0.1),
            top_k=loss_cfg.get('top_k', 0),
        )
    elif method in ('dino', 'byol', 'masked'):
        raise NotImplementedError(
            "ssl method '{:s}' is reserved but not implemented yet. "
            "Implemented methods: neco. Add new losses in "
            "libs/modeling/selfsup.py (see make_ssl_loss).".format(method))
    else:
        raise ValueError(
            "unknown ssl method '{:s}'. Supported: {:s}".format(
                method, ', '.join(SUPPORTED_SSL_METHODS)))


################################################################################
@register_meta_arch("PtTransformerSSL")
class PtTransformerSSL(nn.Module):
    """
    ActionFormer's multi-scale transformer encoder (backbone + neck) with a
    self-supervised projection head. Used as both the student and the EMA
    teacher. Exposes:

        forward(x, mask, view_start, roi) -> ((B, N, D) features, (B, N) mask)

    where `x` is a batch of *one temporal view* (already sliced + padded to a
    multiple of max_div_factor) and `roi` the shared crop-intersection in the
    original feature-grid coordinates.
    """
    def __init__(
        self,
        backbone_type='convTransformer',
        fpn_type='identity',
        backbone_arch=(2, 2, 5),
        scale_factor=2,
        input_dim=2304,
        max_seq_len=2304,
        n_head=4,
        n_mha_win_size=-1,
        embd_kernel_size=3,
        embd_dim=512,
        embd_with_ln=True,
        fpn_dim=512,
        fpn_with_ln=True,
        fpn_start_level=0,
        use_abs_pe=False,
        use_rel_pe=False,
        train_cfg=None,
        ssl_cfg=None,
        **kwargs,
    ):
        super().__init__()
        train_cfg = train_cfg or {}
        ssl_cfg = ssl_cfg or {}

        self.max_seq_len = max_seq_len
        self.fpn_type = fpn_type
        self.scale_factor = scale_factor

        # expand the (local) attention window size per transformer level
        if isinstance(n_mha_win_size, int):
            mha_win_size = [n_mha_win_size] * (1 + backbone_arch[-1])
        else:
            mha_win_size = list(n_mha_win_size)

        # max_div_factor: every input length must be a multiple of this
        # (mirrors the constraint of the original ActionFormer model)
        self.max_div_factor = 1
        for s, w in zip(
            [scale_factor ** i for i in range(fpn_start_level, backbone_arch[-1] + 1)],
            mha_win_size,
        ):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, \
                "max_seq_len must be divisible by fpn stride and window size"
            self.max_div_factor = max(self.max_div_factor, stride)

        # encoder: same multi-scale transformer as the supervised model
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                n_in=input_dim,
                n_embd=embd_dim,
                n_head=n_head,
                n_embd_ks=embd_kernel_size,
                max_len=max_seq_len,
                arch=backbone_arch,
                mha_win_size=mha_win_size,
                scale_factor=scale_factor,
                with_ln=embd_with_ln,
                attn_pdrop=0.0,
                proj_pdrop=train_cfg.get('dropout', 0.0),
                path_pdrop=train_cfg.get('droppath', 0.0),
                use_abs_pe=use_abs_pe,
                use_rel_pe=use_rel_pe,
            )
        else:
            self.backbone = make_backbone(
                'conv',
                n_in=input_dim,
                n_embd=embd_dim,
                n_embd_ks=embd_kernel_size,
                arch=backbone_arch,
                scale_factor=scale_factor,
                with_ln=embd_with_ln,
            )
        if isinstance(embd_dim, (list, tuple)):
            embd_dim = sum(embd_dim)

        self.neck = make_neck(
            fpn_type,
            in_channels=[embd_dim] * (backbone_arch[-1] + 1),
            out_channel=fpn_dim,
            scale_factor=scale_factor,
            start_level=fpn_start_level,
            with_ln=fpn_with_ln,
        )
        self.num_levels = self.neck.end_level - self.neck.start_level

        # temporal resolution of each FPN level
        #  - identity neck keeps the backbone's multi-scale lengths (stride 2^l)
        #  - the FPN neck resamples everything back to the full resolution
        if fpn_type == 'fpn':
            self.level_strides = [1.0] * self.num_levels
        else:
            self.level_strides = [
                float(scale_factor ** (l + fpn_start_level))
                for l in range(self.num_levels)
            ]

        # self-supervised parameters
        self.n_bins = int(ssl_cfg.get('n_bins', 32))
        head_cfg = ssl_cfg.get('head', {}) or {}
        self.head = SSLProjectionHead(
            in_dim=fpn_dim,
            out_dim=head_cfg.get('out_dim', 256),
            hidden_dim=head_cfg.get('hidden_dim', 2048),
            num_layers=head_cfg.get('num_layers', 3),
            num_levels=self.num_levels,
            with_mixer=head_cfg.get('with_mixer', True),
            dropout=head_cfg.get('dropout', 0.0),
        )

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    def encode(self, x, mask):
        feats, masks = self.backbone(x, mask)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        return fpn_feats, fpn_masks

    def forward_view(self, x, mask, view_start, roi):
        """
        Encode ONE temporal view, project it (position-wise), ROI-align every
        FPN level onto the shared temporal ROI and fuse the scales.

        x:          (B, C, P) padded view
        mask:       (B, 1, P) bool
        view_start: (B,) float, raw-coordinate start s_v of this view
        roi:        (B, 2) float, raw-coordinate intersection (shared ROI)

        Returns ((B, N, D) L2-normalized features, (B, N) bool valid mask).
        """
        fpn_feats, fpn_masks = self.encode(x, mask)
        aligned = []
        aligned_masks = []
        for l in range(self.num_levels):
            feat_l = fpn_feats[l]
            mask_l = fpn_masks[l]
            # project first, align second (NeCo's order)
            proj = self.head.forward(feat_l)[0]          # (B, D, T_l)
            stride = self.level_strides[l]
            st = view_start.to(roi.dtype)
            boxes = torch.stack(
                [(roi[:, 0] - st) / stride,
                 (roi[:, 1] - st) / stride], dim=1)      # (B, 2)
            a, am = temporal_roi_align(proj, mask_l, boxes, self.n_bins)
            aligned.append(a)
            aligned_masks.append(am)

        feat = torch.cat(aligned, dim=1)                 # (B, L*D, N)
        if self.head.mixer is not None:
            feat = self.head.mixer(feat)
            if self.head.mixer_norm is not None:
                feat = self.head.mixer_norm(feat)
        feat = F.normalize(feat, p=2, dim=1)             # L2 over channels
        amask = torch.cat(aligned_masks, dim=1).min(dim=1)[0]  # (B, N)
        return feat.transpose(1, 2), (amask > 0.5)

    def forward(self, x, mask, view_start, roi):
        return self.forward_view(x, mask, view_start, roi)


################################################################################
def make_ssl_optimizer(model, opt_cfg, head_lr_scale=1.0):
    """
    AdamW / SGD optimizer for the SSL model with *decoupled* learning rates:
    the MLP projection head (module prefix 'head.') trains at
    `head_lr_scale` x the encoder's learning rate (NeCo/DINO convention).

    The decay / no-decay split mirrors `libs.utils.make_optimizer`.
    """
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, nn.GroupNorm)

    lr = opt_cfg['learning_rate']
    wd = opt_cfg['weight_decay']
    # [enc_decay, enc_no_decay, head_decay, head_no_decay]
    groups = [[], [], [], []]
    for mn, m in model.named_modules():
        is_head = mn.split('.')[0] == 'head'
        gi0, gi1 = (2, 3) if is_head else (0, 1)
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if pn.endswith('bias'):
                groups[gi1].append(p)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                groups[gi0].append(p)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                groups[gi1].append(p)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                groups[gi1].append(p)
            elif pn.endswith('rel_pe'):
                groups[gi1].append(p)
            else:
                groups[gi1].append(p)

    optim_groups = [
        {'params': groups[0], 'lr': lr, 'weight_decay': wd},
        {'params': groups[1], 'lr': lr, 'weight_decay': 0.0},
        {'params': groups[2], 'lr': lr * head_lr_scale, 'weight_decay': wd},
        {'params': groups[3], 'lr': lr * head_lr_scale, 'weight_decay': 0.0},
    ]
    optim_groups = [g for g in optim_groups if len(g['params']) > 0]

    optim_type = opt_cfg.get('type', 'AdamW')
    if optim_type == 'AdamW':
        return torch.optim.AdamW(optim_groups)
    elif optim_type == 'SGD':
        return torch.optim.SGD(
            optim_groups, momentum=opt_cfg.get('momentum', 0.9))
    else:
        raise TypeError("Unsupported optimizer: {:s}".format(optim_type))


def ema_momentum_schedule(global_step, total_steps,
                          start_momentum=0.9995, end_momentum=1.0):
    """
    Cosine EMA momentum schedule (NeCo / DINOv2): m cosines from
    start_momentum up to end_momentum over the whole run.
    """
    if total_steps <= 1:
        return start_momentum
    ratio = global_step / (total_steps - 1)
    return end_momentum - (end_momentum - start_momentum) * \
        (1.0 + math.cos(math.pi * ratio)) / 2.0