import argparse
import torch
import numpy as np
from pathlib import Path
from torchvision.io import read_image  # <-- Changed: replaced decord with torchvision
from transformers import AutoVideoProcessor, AutoModel

HF_REPO = "facebook/vjepa2-vitl-fpc64-256"
NUM_FRAMES = 64

def extract_embeddings_from_jpeg_folder(folder_path: Path, model, processor, device: str):
    # 1. Gather all jpg files inside this specific action/video folder
    # Frame format: frame_0000000001.jpg -> sorting ensures they are in correct temporal order
    frame_files = sorted(folder_path.glob("frame_*.jpg"))
    total_frames = len(frame_files)
    
    if total_frames == 0:
        raise ValueError(f"No frames found in folder: {folder_path}")

    # 2. Sample exactly NUM_FRAMES evenly across the sequence
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    indices = np.clip(indices, 0, total_frames - 1)
    
    # 3. Read the sampled images sequentially
    frames = []
    for idx in indices:
        img_path = frame_files[idx]
        img = read_image(str(img_path))  # Returns a tensor of shape (C, H, W)
        frames.append(img)
        
    # 4. Stack into a single tensor of shape: T x C x H x W
    video = torch.stack(frames, dim=0)
    
    # 5. Process and forward pass (remains identical to your video script)
    inputs = processor(video, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model.get_vision_features(**inputs)
    return embeddings.cpu()


def main():
    parser = argparse.ArgumentParser()
    # Provide the path to the root 'EPIC-KITCHENS' directory or a specific participant folder
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

    # CRITICAL CHANGE: Find folders (like P01_01) containing Epic-Kitchens frames,
    # rather than looking for standalone video files.
    action_folders = sorted([
        d for d in input_dir.rglob("*") 
        if d.is_dir() and d.name.startswith("P") and "_" in d.name
    ])
    
    print(f"Found {len(action_folders)} video folder(s)")

    for i, folder in enumerate(action_folders, 1):
        out_path = output_dir / f"{folder.name}.pt"
        print(f"[{i}/{len(action_folders)}] Processing folder: {folder.name}")
        try:
            emb = extract_embeddings_from_jpeg_folder(folder, model, processor, device)
            torch.save(emb, out_path)
            print(f"  -> saved {out_path.name}  shape={list(emb.shape)}")
        except Exception as e:
            print(f"  -> FAILED: {e}")


if __name__ == "__main__":
    main()