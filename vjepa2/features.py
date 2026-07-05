"""
V-JEPA2 feature extraction for ActionFormer on Epic Kitchens (Aligned to 30 fps).
Filtered to target only videos outside the 59-60 fps range via EPIC_100_video_info.csv.
"""

import argparse
import torch
import numpy as np
import pandas as pd  # <-- Added for metadata filtering
from pathlib import Path
from torchvision.io import read_image
from transformers import AutoVideoProcessor, AutoModel

# ── Model defaults ────────────────────────────────────────────────────────────
HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 64       # frames fed to the model per forward pass
TUBELET_SIZE = 2      # V-JEPA2 temporal patch size
PATCH_SIZE = 16       # V-JEPA2 spatial patch size
CROP_SIZE = 256       # expected spatial resolution

# ── Aligned 30 fps Temporal defaults ──────────────────────────────────────────
DEFAULT_FPS = 30
DEFAULT_CLIP_FRAME_STEP = 1  
DEFAULT_SNIPPET_STRIDE = 16  


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
    total = len(frame_files)
    if total == 0:
        raise ValueError("No frames provided.")

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
        indices = np.clip(indices, 0, total - 1)

        frames = [read_image(str(frame_files[i])) for i in indices]
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].clone())
        frames = frames[:NUM_FRAMES]

        video = torch.stack(frames, dim=0)  # (T, C, H, W)

        inputs = processor(video, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.get_vision_features(**inputs)

        emb = embeddings.cpu().squeeze(0)                         
        T_pos = NUM_FRAMES // TUBELET_SIZE                        
        S = (CROP_SIZE // PATCH_SIZE) ** 2                        
        emb = emb.reshape(T_pos, S, -1).mean(dim=(0, 1))         
        all_vecs.append(emb)

    feats = torch.stack(all_vecs, dim=0).float()                  
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
    all_raw_frames = sorted(folder.glob("frame_*.jpg"))
    frame_files = all_raw_frames[::2] 
    
    if len(frame_files) == 0:
        raise ValueError(f"No frame_*.jpg files found after downsampling in {folder}")

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
        description="Extract V-JEPA2 features for ActionFormer (Targeted Range Mode)"
    )
    parser.add_argument("--input",  "-i", type=str, required=True,
                        help="Root directory containing participant folders (e.g., ~/EPIC-KITCHENS).")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory for .npz feature files.")
    # New argument to point to the video info file
    parser.add_argument("--video-info", "-c", type=str, default="EPIC_100_video_info.csv",
                        help="Path to the EPIC_100_video_info.csv metadata file.")
    parser.add_argument("--model",  "-m", type=str, default=HF_REPO)
    parser.add_argument("--device", "-d", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--clip-frame-step", type=int, default=DEFAULT_CLIP_FRAME_STEP)
    parser.add_argument("--snippet-stride", type=int, default=DEFAULT_SNIPPET_STRIDE)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)

    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = Path(args.video_info)
    if not csv_path.exists():
        print(f"Error: Metadata file '{csv_path}' not found.")
        return

    # ── Parse CSV and define target set ───────────────────────────────────────
    print(f"Parsing metadata from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Filter for videos strictly outside the 59-60 fps window
    target_df = df[(df['fps'] < 59) | (df['fps'] > 60)]
    target_video_ids = set(target_df['video_id'].unique())
    print(f"Loaded metadata. Identified {len(target_video_ids)} target video profiles matching criteria.")

    print(f"Loading model: {args.model}")
    model     = AutoModel.from_pretrained(args.model).to(args.device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model)
    print(f"Model loaded on {args.device}")

    # Discover folders matching standard Epic Kitchens structure
    all_folders = sorted([
        d for d in input_dir.rglob("*")
        if d.is_dir() and d.name.startswith("P") and "_" in d.name
    ])
    
    # ── Filter action folders down to matches in our target set ───────────────
    action_folders = [f for f in all_folders if f.name in target_video_ids]
    
    print(f"\nDiscovered {len(all_folders)} total video folders on disk.")
    print(f"Filtered down to {len(action_folders)} target folders requiring extraction.\n" + "="*60)

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