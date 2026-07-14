"""
V-JEPA2 feature extraction for ActionFormer on Epic Kitchens (Native 60 fps).

Processes ALL available video folders without filtering or dropping frames.
Uses a sliding window centred on each snippet timestep.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from torchvision.io import read_image
from transformers import AutoVideoProcessor, AutoModel

# ── Model defaults ────────────────────────────────────────────────────────────
HF_REPO = "facebook/vjepa2-vitl-fpc32-256-diving48"
NUM_FRAMES = 32       # frames fed to the model per forward pass
TUBELET_SIZE = 2      # V-JEPA2 temporal patch size
PATCH_SIZE = 16       # V-JEPA2 spatial patch size
CROP_SIZE = 256       # expected spatial resolution

# ── Native 60 fps Temporal defaults ──────────────────────────────────────────
DEFAULT_FPS = 30
DEFAULT_CLIP_FRAME_STEP = 1  # 64 consecutive frames at 60 fps ≈ 1.07 s context
DEFAULT_SNIPPET_STRIDE = 16  # Step forward by 32 frames at 60 fps ≈ 0.533 s stride


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
    # CHANGED: We now load ALL sorted frames directly without dropping any frames ([::2] is removed).
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
        feats=feats.numpy(),
        timestamps_s=timestamps_s,
        fps=np.float32(fps),
        clip_frame_step=np.int32(clip_frame_step),
        snippet_stride=np.int32(snippet_stride),
        num_frames=np.int32(NUM_FRAMES),
    )
    return feats.shape


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA2 features for ActionFormer (Native 60 fps, All Videos Mode)"
    )
    parser.add_argument("--input",  "-i", type=str, required=True,
                        help="Root directory containing participant folders (e.g., ~/EPIC-KITCHENS).")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for .npz feature files.")
    parser.add_argument("--model",  "-m", type=str, default=HF_REPO)
    parser.add_argument("--device", "-d", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Native 60 fps temporal parameters
    parser.add_argument(
        "--clip-frame-step", type=int, default=DEFAULT_CLIP_FRAME_STEP,
        help="Frame stride inside the clip window (Default: 1)."
    )
    parser.add_argument(
        "--snippet-stride", type=int, default=DEFAULT_SNIPPET_STRIDE,
        help="Frame shift between adjacent feature vectors (Default: 32)."
    )
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help="Operating framework frame rate (Default: 60)."
    )

    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model     = AutoModel.from_pretrained(args.model).to(args.device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model)
    print(f"Model loaded on {args.device}")

    # CHANGED: We find and process ALL folders matching the name structure on disk
    action_folders = sorted([
        d for d in input_dir.rglob("*")
        if d.is_dir() and d.name.startswith("P") and "_" in d.name
    ])
    
    print(f"\nDiscovered {len(action_folders)} video folder(s) in {input_dir} for processing.\n" + "="*60)

    print(
        f"\nTemporal config (60 fps native mode):"
        f"\n  clip_frame_step = {args.clip_frame_step}  "
        f"→ clip covers {NUM_FRAMES * args.clip_frame_step / args.fps:.2f} s"
        f"\n  snippet_stride  = {args.snippet_stride}  "
        f"→ one feature every {args.snippet_stride / args.fps:.3f} s "
        f"({args.fps / args.snippet_stride:.1f} feat/s)\n" + "="*60
    )

    for i, folder in enumerate(action_folders, 1):
        out_path = output_dir / f"{folder.name}.npz"
        print(f"[{i}/{len(action_folders)}] Processing: {folder.name}")
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
            print(f"  -> Saved {out_path.name} | Features shape = {list(shape)}")
        except Exception as e:
            print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    main()