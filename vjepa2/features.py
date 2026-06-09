"""
V-JEPA2 feature extraction for ActionFormer on Epic Kitchens (60 fps).

Produces one feature vector per snippet at a configurable stride,
using a sliding window centred on each snippet timestep. This matches
the feature format ActionFormer expects: a 1-D sequence of clip-level
vectors at uniform temporal resolution across the full video.

Temporal defaults (matching SlowFast EK-100 feature rate):
    FPS            = 60
    CLIP_FRAME_STEP = 2   → each clip covers 64*2/60 ≈ 2.13 s
    SNIPPET_STRIDE  = 30  → one vector every 30/60 = 0.5 s  (~2 feat/s)

Output per video:
    feats        : (N, D)  float32   — N snippets, D = model hidden dim
    timestamps_s : (N,)    float32   — centre time in seconds for each snippet
    fps          : scalar
    clip_frame_step, snippet_stride, num_frames : scalars (for reproducibility)
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torchvision.io import read_image
from transformers import AutoVideoProcessor, AutoModel

# ── Model defaults ────────────────────────────────────────────────────────────
HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 64       # frames fed to the model per forward pass
TUBELET_SIZE = 2      # V-JEPA2 temporal patch size
PATCH_SIZE = 16       # V-JEPA2 spatial patch size
CROP_SIZE = 256       # expected spatial resolution

# ── Temporal sampling defaults ────────────────────────────────────────────────
# CLIP_FRAME_STEP = 2  → 64 frames × 2 = 128 raw frames ≈ 2.13 s at 60 fps
# SNIPPET_STRIDE  = 30 → one feature every 0.5 s  (matches SlowFast EK rate)
DEFAULT_FPS = 60
DEFAULT_CLIP_FRAME_STEP = 2
DEFAULT_SNIPPET_STRIDE = 30


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_feature_sequence(
    frame_files: list,
    model,
    processor,
    device: str,
    clip_frame_step: int = DEFAULT_CLIP_FRAME_STEP,
    snippet_stride: int = DEFAULT_SNIPPET_STRIDE,
    fps: float = DEFAULT_FPS,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Slide a V-JEPA2 clip window across a video and return one feature
    vector per snippet.

    Args:
        frame_files      : sorted list of Path objects for every raw frame.
        model            : loaded V-JEPA2 model (eval mode).
        processor        : matching AutoVideoProcessor.
        device           : 'cuda' or 'cpu'.
        clip_frame_step  : raw-frame stride used when sampling the 64 clip
                           frames (default 2 → ~2.1 s clip at 60 fps).
        snippet_stride   : number of raw frames between snippet centres
                           (default 30 → 0.5 s at 60 fps, i.e. ~2 feat/s).
        fps              : source video frame rate (used for timestamp calc).

    Returns:
        feats        : float32 tensor of shape (N, D)
        timestamps_s : float32 array of shape (N,) — centre time per snippet
    """
    total = len(frame_files)
    if total == 0:
        raise ValueError("No frames provided.")

    # Half-span of the clip in raw frames (for centred window)
    half_span = (NUM_FRAMES * clip_frame_step) // 2

    all_vecs = []
    centres = list(range(0, total, snippet_stride))

    for centre in centres:
        start = centre - half_span
        indices = np.arange(
            start,
            start + NUM_FRAMES * clip_frame_step,
            clip_frame_step,
            dtype=int,
        )
        # Boundary clamping: reflect-pad at start/end instead of dropping clips
        indices = np.clip(indices, 0, total - 1)

        frames = [read_image(str(frame_files[i])) for i in indices]
        # Pad to NUM_FRAMES with the last frame if clamping collapsed the tail
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].clone())
        frames = frames[:NUM_FRAMES]

        video = torch.stack(frames, dim=0)  # (T, C, H, W)

        inputs = processor(video, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.get_vision_features(**inputs)

        # embeddings: (1, T_pos * S, D)
        emb = embeddings.cpu().squeeze(0)                         # (T_pos*S, D)
        T_pos = NUM_FRAMES // TUBELET_SIZE                        # 32
        S = (CROP_SIZE // PATCH_SIZE) ** 2                        # 256
        emb = emb.reshape(T_pos, S, -1).mean(dim=(0, 1))         # (D,)
        all_vecs.append(emb)

    feats = torch.stack(all_vecs, dim=0).float()                  # (N, D)
    timestamps_s = np.array(centres, dtype=np.float32) / fps

    return feats, timestamps_s


# ── Per-video driver ──────────────────────────────────────────────────────────

def process_folder(
    folder: Path,
    out_path: Path,
    model,
    processor,
    device: str,
    clip_frame_step: int,
    snippet_stride: int,
    fps: float,
):
    frame_files = sorted(folder.glob("frame_*.jpg"))
    if len(frame_files) == 0:
        raise ValueError(f"No frame_*.jpg files found in {folder}")

    feats, timestamps_s = extract_feature_sequence(
        frame_files=frame_files,
        model=model,
        processor=processor,
        device=device,
        clip_frame_step=clip_frame_step,
        snippet_stride=snippet_stride,
        fps=fps,
    )

    np.savez(
        out_path,
        feats=feats.numpy(),                  # (N, D)  ← what ActionFormer reads
        timestamps_s=timestamps_s,            # (N,)    ← useful for debugging
        fps=np.float32(fps),
        clip_frame_step=np.int32(clip_frame_step),
        snippet_stride=np.int32(snippet_stride),
        num_frames=np.int32(NUM_FRAMES),
    )
    return feats.shape


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA2 features for ActionFormer (Epic Kitchens 60 fps)"
    )
    parser.add_argument("--input",  "-i", type=str, required=True,
                        help="Root directory; recursively finds P*_* action folders.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for .npz feature files.")
    parser.add_argument("--model",  "-m", type=str, default=HF_REPO)
    parser.add_argument("--device", "-d", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Temporal sampling
    parser.add_argument(
        "--clip-frame-step", type=int, default=DEFAULT_CLIP_FRAME_STEP,
        help=(
            "Raw-frame stride for sampling the 64 clip frames. "
            "Default 2 → ~2.1 s clip at 60 fps (matches SlowFast temporal window). "
            "Increase to cover a wider context at lower density."
        ),
    )
    parser.add_argument(
        "--snippet-stride", type=int, default=DEFAULT_SNIPPET_STRIDE,
        help=(
            "Raw-frame gap between snippet centres. "
            "Default 30 → one feature every 0.5 s at 60 fps (~2 feat/s, "
            "matching SlowFast EK-100 feature rate). "
            "Use 15 for ~4 feat/s (denser, slower)."
        ),
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="Source video frame rate (default 60). Used only for timestamp metadata.",
    )

    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model     = AutoModel.from_pretrained(args.model).to(args.device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model)
    print(f"Model loaded on {args.device}")

    print(
        f"\nTemporal config:"
        f"\n  clip_frame_step = {args.clip_frame_step}  "
        f"→ clip covers {NUM_FRAMES * args.clip_frame_step / args.fps:.2f} s"
        f"\n  snippet_stride  = {args.snippet_stride}  "
        f"→ one feature every {args.snippet_stride / args.fps:.3f} s "
        f"({args.fps / args.snippet_stride:.1f} feat/s)"
    )

    action_folders = sorted([
        d for d in input_dir.rglob("*")
        if d.is_dir() and d.name.startswith("P") and "_" in d.name
    ])
    print(f"\nFound {len(action_folders)} video folder(s) in {input_dir}\n")

    for i, folder in enumerate(action_folders, 1):
        out_path = output_dir / f"{folder.name}.npz"
        print(f"[{i}/{len(action_folders)}] {folder.name}")
        try:
            shape = process_folder(
                folder=folder,
                out_path=out_path,
                model=model,
                processor=processor,
                device=args.device,
                clip_frame_step=args.clip_frame_step,
                snippet_stride=args.snippet_stride,
                fps=args.fps,
            )
            print(f"  -> saved {out_path.name}  feats shape={list(shape)}")
        except Exception as e:
            print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    main()