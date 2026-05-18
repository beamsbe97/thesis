import argparse
import torch
import numpy as np
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel


HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
NUM_FRAMES = 64


def extract_embeddings(video_path: Path, model, processor, device: str):
    vr = VideoDecoder(str(video_path))
    total_frames = vr.metadata.num_frames
    if total_frames == 0:
        raise ValueError("Video has no frames")

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    indices = np.clip(indices, 0, total_frames - 1)
    video = vr.get_frames_at(indices=indices).data
    inputs = processor(video, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_vision_features(**inputs)
    return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, default=HF_REPO)
    parser.add_argument("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model)

    video_files = sorted(
        f for f in input_dir.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS
    )
    print(f"Found {len(video_files)} video(s)")

    for i, vf in enumerate(video_files, 1):
        out_path = output_dir / f"{vf.stem}.pt"
        print(f"[{i}/{len(video_files)}] {vf.name}")
        try:
            emb = extract_embeddings(vf, model, processor, device)
            torch.save(emb, out_path)
            print(f"  -> saved {out_path.name}  shape={list(emb.shape)}")
        except Exception as e:
            print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    main()
