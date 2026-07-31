import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def verify_embeddings_against_csv(
    csv_path: str,
    embeddings_dir: str,
):
    # Added .expanduser() to resolve tildes (~) to full home paths
    csv_file = Path(csv_path).expanduser().resolve()
    emb_path = Path(embeddings_dir).expanduser().resolve()

    if not csv_file.exists():
        print(f"Error: CSV metadata file '{csv_file}' not found.")
        return
    if not emb_path.exists():
        print(f"Error: Embeddings directory '{emb_path}' not found.")
        return

    # 1. Read metadata CSV and extract unique video_ids
    print(f"Loading metadata from: {csv_file.name}")
    df = pd.read_csv(csv_file)

    if "video_id" not in df.columns:
        print("Error: 'video_id' column not found in the CSV file.")
        return

    expected_video_ids = set(df["video_id"].unique())
    print(f"Total video IDs in CSV: {len(expected_video_ids)}")

    # 2. Find all .npz files on disk in the embeddings directory
    extracted_files = {p.stem: p for p in emb_path.glob("*.npz")}
    print(f"Total .npz files found in '{emb_path.name}': {len(extracted_files)}")

    # 3. Compare CSV vs Embeddings
    missing_video_ids = expected_video_ids - set(extracted_files.keys())
    extra_files = set(extracted_files.keys()) - expected_video_ids

    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)

    # 4. Audit integrity of existing files
    corrupted_files = []
    feature_shapes = []

    print("Auditing existing .npz files for corruption...")
    for video_id in expected_video_ids - missing_video_ids:
        npz_file = extracted_files[video_id]
        try:
            if npz_file.stat().st_size == 0:
                corrupted_files.append((video_id, "0-byte empty file"))
                continue

            with np.load(npz_file) as data:
                if "feats" not in data:
                    corrupted_files.append((video_id, "Missing 'feats' key"))
                elif data["feats"].shape[0] == 0:
                    corrupted_files.append((video_id, "Empty tensor (0 snippets)"))
                else:
                    feature_shapes.append(data["feats"].shape)
        except Exception as e:
            corrupted_files.append((video_id, f"Corrupted file ({str(e)})"))

    # ── Summary Details ──
    print(f"Valid .npz Files:         {len(feature_shapes)} / {len(expected_video_ids)}")
    print(f"Missing Extractions:       {len(missing_video_ids)}")
    print(f"Corrupted / Invalid Files: {len(corrupted_files)}")

    if extra_files:
        print(f"Extra/Unexpected Files:    {len(extra_files)}")

    if missing_video_ids:
        print("\n" + "!" * 60)
        print("MISSING VIDEO IDs:")
        for v in sorted(missing_video_ids):
            print(f"  - {v}")

    if corrupted_files:
        print("\n" + "!" * 60)
        print("CORRUPTED / INVALID FILES:")
        for v, reason in corrupted_files:
            print(f"  - {v}: {reason}")

    if not missing_video_ids and not corrupted_files:
        print("\nSUCCESS! Every single video in the CSV has a valid .npz file!")
        if feature_shapes:
            feat_dim = feature_shapes[0][1]
            min_seq = min(s[0] for s in feature_shapes)
            max_seq = max(s[0] for s in feature_shapes)
            print(f"Feature Dimension:     {feat_dim}")
            print(f"Sequence Length Range: {min_seq} to {max_seq} feature vectors.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify .npz feature file existence against EPIC_100_video_info.csv"
    )
    parser.add_argument(
        "--csv",
        "-c",
        type=str,
        default="EPIC_100_video_info.csv",
        help="Path to EPIC_100_video_info.csv",
    )
    parser.add_argument(
        "--embeddings",
        "-e",
        type=str,
        default="~/EPIC-KITCHENS/embeddings_32fpc",
        help="Path to the directory containing .npz feature files",
    )
    args = parser.parse_args()

    verify_embeddings_against_csv(args.csv, args.embeddings)