import os
import subprocess
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- CONFIGURATION ---
DATASET_ROOT = Path("~/done").expanduser()
CSV_PATH = Path("~/EPIC-KITCHENS/EPIC_100_video_info.csv").expanduser()
FRAME_EXT = ".jpg"  

# High-quality JPEG scale (1-31, lower is better quality. 2 is standard for CV)
JPEG_QUALITY = "2"  

# DISK I/O SAFETY CAP: Max parallel disk writers. 
# Keep this around 6-8 on a cluster to avoid dragging down shared file storage.
MAX_IO_WORKERS = 8  
# ---------------------

def extract_frames_for_video(row):
    video_id = row['video_id']
    participant_id = video_id.split('_')[0]
    
    video_path = DATASET_ROOT / participant_id / "videos" / f"{video_id}.MP4"
    output_dir = DATASET_ROOT / participant_id / "rgb_frames" / video_id
    
    if not video_path.exists():
        return f"❌ Skipped (Video not found): {video_id}"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Epic Kitchens standard: frame_0000000001.jpg (10-digit zero-padded)
    frame_pattern = output_dir / f"frame_%010d{FRAME_EXT}"
    

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", "scale=-1:256",        # Downscales height to 256px, keeps aspect ratio
        "-q:v", "4",                  # Matches official compression level
        "-start_number", "1",
        str(frame_pattern)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        num_frames = len(list(output_dir.glob(f"*{FRAME_EXT}")))
        return f"✅ Extracted {num_frames} .jpg frames for {video_id}"
    except subprocess.CalledProcessError as e:
        return f"❌ Failed extracting frames for {video_id}: {e}"

def main():
    if not CSV_PATH.exists():
        print(f"Error: Metadata file '{CSV_PATH}' not found.")
        return
        
    df = pd.read_csv(CSV_PATH)
    target_videos = df[(df['fps'] < 59) | (df['fps'] > 60)]
    tasks = [row for _, row in target_videos.iterrows()]
    
    print(f"Total videos to extract to .jpg: {len(tasks)}")
    print(f"Using {MAX_IO_WORKERS} parallel I/O workers...\n" + "="*60)
    
    with ProcessPoolExecutor(max_workers=MAX_IO_WORKERS) as executor:
        futures = {executor.submit(extract_frames_for_video, task): task for task in tasks}
        
        for count, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"[{count}/{len(tasks)}] {result}")

if __name__ == "__main__":
    main()