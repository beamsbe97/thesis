import os
import subprocess
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
DATASET_ROOT = Path("/zfsstore/courses/2025-2026/4343SADL6/g10/EPIC-KITCHENS").expanduser()  # Change this to your dataset base path
CSV_PATH = Path("~/EPIC-KITCHENS/EPIC_100_video_info.csv").expanduser()
TARGET_FPS = "60"
CRF_QUALITY = "20"  # 18-23 is standard. Lower means higher quality/larger file size.
# ---------------------


def process_single_video(row):
    """Worker function to re-encode a single video clip."""
    video_id = row['video_id']
    current_fps = row['fps']
    
    participant_id = video_id.split('_')[0]
    video_dir = DATASET_ROOT / participant_id / "videos"
    video_path = video_dir / f"{video_id}.MP4"
    temp_path = video_dir / f"{video_id}_temp.MP4"
    
    if not video_path.exists():
        return f"❌ Skipped (File not found): {video_id}"
        
    # Standard CPU encoding using libx264 with the 'veryfast' preset
    # -threads 1 ensures individual ffmpeg instances don't fight for the same cores
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-r", TARGET_FPS,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", CRF_QUALITY,
        "-threads", "1",
        "-c:a", "copy",
        str(temp_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        temp_path.replace(video_path)  # Safe atomic swap
        return f"✅ Successfully updated: {video_id} ({current_fps:.2f} -> {TARGET_FPS} FPS)"
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return f"❌ Failed to process {video_id}: {e}"

def main():
    if not CSV_PATH.exists():
        print(f"Error: Metadata file '{CSV_PATH}' not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    to_reencode = df[(df['fps'] < 59) | (df['fps'] > 60)]
    tasks = [row for _, row in to_reencode.iterrows()]
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("All videos are already within the 59-60 fps range.")
        return

    # Automatically read Slurm allocated CPUs, default to 4 if local testing
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    
    print(f"Total videos to re-encode: {total_tasks}")
    print(f"Allocating {allocated_cpus} parallel workers based on Slurm environment...\n" + "="*50)
    
    # Run the processing pool
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_single_video, task): task for task in tasks}
        
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{count}/{total_tasks}] {result}")

if __name__ == "__main__":
    main()