import argparse
import torch
import numpy as np
from pathlib import Path
from torchvision.io import read_image
from transformers import AutoVideoProcessor, AutoModel

# Why IS IT NOT UPDATING
HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 64
FRAME_STEP = 16
TUBELET_SIZE = 2
PATCH_SIZE = 16
CROP_SIZE = 256


def extract_embeddings_from_jpeg_folder(
    folder_path: Path, model, processor, device: str, frame_step: int = FRAME_STEP
):
    frame_files = sorted(folder_path.glob("frame_*.jpg"))
    total_frames = len(frame_files)

    if total_frames == 0:
        raise ValueError(f"No frames found in folder: {folder_path}")

    window_stride = NUM_FRAMES * frame_step
    S = (CROP_SIZE // PATCH_SIZE) ** 2
    T_pos = NUM_FRAMES // TUBELET_SIZE

    all_features = []

    for window_start in range(0, total_frames, window_stride):
        indices = np.arange(
            window_start, window_start + frame_step * NUM_FRAMES, frame_step, dtype=int
        )
        indices = indices[indices < total_frames]

        frames = []
        for idx in indices:
            img_path = frame_files[idx]
            img = read_image(str(img_path))
            frames.append(img)

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].clone())

        video = torch.stack(frames, dim=0)

        inputs = processor(video, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.get_vision_features(**inputs)

        emb = embeddings.cpu().squeeze(0)
        emb = emb.reshape(T_pos, S, -1).mean(dim=1)
        all_features.append(emb)

    feats = torch.cat(all_features, dim=0)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, default=HF_REPO)
    parser.add_argument("--step", "-s", type=int, default=FRAME_STEP,
                        help="Frame step (default: 16 for SlowFast density)")
    parser.add_argument("--device", "-d", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model)
    frame_step = args.step

    action_folders = sorted([
        d for d in input_dir.rglob("*")
        if d.is_dir() and d.name.startswith("P") and "_" in d.name
    ])

    print(f"Found {len(action_folders)} video folder(s) in {input_dir}")

    for i, folder in enumerate(action_folders, 1):
        out_path = output_dir / f"{folder.name}.npz"
        print(f"[{i}/{len(action_folders)}] Processing folder: {folder.name}  (step={frame_step})")
        try:
            emb = extract_embeddings_from_jpeg_folder(folder, model, processor, device, frame_step)
            np.savez(out_path, feats=emb.numpy(), frame_step=frame_step, num_frames=NUM_FRAMES)
            print(f"  -> saved {out_path.name}  shape={list(emb.shape)}")
        except Exception as e:
            print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    main()
