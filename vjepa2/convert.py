from pathlib import Path
import torch
import numpy as np

# Set your directory paths
pt_dir = Path("embeddings")
npz_dir = Path("embeddings")
npz_dir.mkdir(parents=True, exist_ok=True)

# Find all PyTorch files
pt_files = list(pt_dir.glob("*.pt"))
print(f"Found {len(pt_files)} .pt files to convert.")

for i, pt_path in enumerate(pt_files, 1):
    # 1. Load the PyTorch tensor
    tensor_data = torch.load(pt_path, map_location="cpu")
    
    # 2. Convert to a NumPy array
    numpy_data = tensor_data.numpy()
    
    # 3. Save as a compressed .npz archive
    npz_path = npz_dir / f"{pt_path.stem}.npz"
    
    # Change "features" to match the key name your downstream model expects!
    np.savez_compressed(npz_path, features=numpy_data)
    
    print(f"[{i}/{len(pt_files)}] Converted {pt_path.name} -> {npz_path.name}")

print("Conversion complete!")