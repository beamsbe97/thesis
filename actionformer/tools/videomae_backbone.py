"""
Minimal VideoMAE-L ViT (16x4x1) in plain PyTorch, matching the key layout of
OpenTAD's EPIC-finetuned checkpoints (e.g.
`videomae_l_epic_verb.pth`, keys `backbone.*` / `cls_head.*`).

Why this exists: `thesis_env` has no mmaction/mmcv, and we only need the
*dense token grid* (not the pooled classification feature), so a small,
dependency-free reimplementation is simpler than pulling in the whole
framework. It is used solely by `tools/extract_videomae_spatial.py`.

Architecture (VideoMAE-L):
    patch_embed: Conv3d(3 -> 1024, k=(2,16,16), s=(2,16,16))
    24 transformer blocks, 16 heads, MLP ratio 4, LayerNorm eps 1e-6
    learned pos_embed over (T/2) * (H/16) * (W/16) = 8 * 14 * 14 = 1568 tokens
    (no cls token), final `fc_norm` LayerNorm
Attention is VideoMAE-style: a single qkv weight plus separate q/v biases
(zero k bias).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
        qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # standard scaled dot-product attention (torch < 2.0 has no SDPA)
        attn = (q * (self.head_dim ** -0.5)) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    """Matches keys `mlp.layers.0.0.{weight,bias}` and `mlp.layers.1.{weight,bias}`."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU()),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=1024, tubelet=2, patch=16):
        super().__init__()
        self.projection = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet, patch, patch),
            stride=(tubelet, patch, patch),
        )

    def forward(self, x):
        # (B, 3, T, H, W) -> (B, N, C)
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class VideoMAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 num_frames=16, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(3, embed_dim, tubelet_size, patch_size)
        self.grid = img_size // patch_size
        self.n_t = num_frames // tubelet_size
        self.n_s = self.grid * self.grid
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_t * self.n_s, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    @torch.no_grad()
    def forward_tokens(self, x):
        """x: (B, 3, T, H, W) -> (B, n_t, grid, grid, D) dense token grid."""
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.fc_norm(x)
        B = x.shape[0]
        return x.reshape(B, self.n_t, self.grid, self.grid, -1)


def load_videomae(checkpoint_path, device='cpu'):
    """
    Load an OpenTAD EPIC-finetuned VideoMAE-L checkpoint into `VideoMAE`.

    The checkpoint keys are prefixed `backbone.` (and `cls_head.`), so we strip
    that prefix and ignore the classification head. Returns the eval-mode model.
    """
    model = VideoMAE()
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    backbone = {}
    for k, v in state.items():
        if k.startswith('backbone.'):
            backbone[k[len('backbone.'):]] = v
    missing, unexpected = model.load_state_dict(backbone, strict=False)
    missing = [m for m in missing if not m.startswith('cls_head')]
    if missing or unexpected:
        raise RuntimeError(
            'VideoMAE checkpoint mismatch:\n  missing={:s}\n  unexpected={:s}'.format(
                missing, unexpected))
    model.eval()
    model.to(device)
    return model
