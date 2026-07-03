import os
import subprocess
from pathlib import Path
import pandas as pd

# --- CONFIGURATION ---
DATASET_ROOT = Path("/zfsstore/courses/2025-2026/4343SADL6/g10/EPIC-KITCHENS").expanduser()  # Change this to your dataset base path
CSV_PATH = Path("~/EPIC-KITCHENS/EPIC_100_video_info.csv").expanduser()
TARGET_FPS = "60"
CRF_QUALITY = "20"  # 18-23 is standard. Lower means higher quality/larger file size.
# ---------------------

def reencode_videos():
    if not CSV_PATH.exists():
        print(f"Error: Metadata file '{CSV_PATH}' not found.")
        return
        
    # Load dataset metadata
    df = pd.read_csv(CSV_PATH)
    
    # Filter for videos that are NOT between 59 and 60 fps
    to_reencode = df[(df['fps'] < 59) | (df['fps'] > 60)]
    
    total_to_process = len(to_reencode)
    print(f"Total videos in dataset: {len(df)}")
    print(f"Videos requiring re-encoding: {total_to_process}\n" + "="*40)
    
    if total_to_process == 0:
        print("All videos are already within the 59-60 fps range.")
        return

    for count, (idx, row) in enumerate(to_reencode.iterrows(), 1):
        video_id = row['video_id']
        current_fps = row['fps']
        
        # Extract participant ID (e.g., 'P01_101' -> 'P01')
        participant_id = video_id.split('_')[0]
        
        # Construct target paths based on dataset structure
        video_dir = DATASET_ROOT / participant_id / "videos"
        video_path = video_dir / f"{video_id}.MP4"
        temp_path = video_dir / f"{video_id}_temp.MP4"
        
        if not video_path.exists():
            print(f"[{count}/{total_to_process}] Skipped (File not found): {video_path}")
            continue
            
        print(f"[{count}/{total_to_process}] Processing: {video_id} ({current_fps:.2f} FPS -> {TARGET_FPS} FPS)")
        
        # FFmpeg command
        # -r 60 forces target frame rate
        # -c:v libx264 encodes video to H.264
        # -c:a copy streams the audio directly without re-encoding (saves time)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-r", TARGET_FPS,
            "-c:v", "libx264",
            "-crf", CRF_QUALITY,
            "-c:a", "copy",
            str(temp_path)
        ]
        
        try:
            # Execute transcoding
            subprocess.run(cmd, check=True)
            
            # Atomic swap to safely overwrite the original file
            temp_path.replace(video_path)
            print(f" -> Successfully updated {video_id}.MP4", flush=True)
            
        except subprocess.CalledProcessError as e:
            print(f" -> Error transcoding {video_id}. Action aborted.")
            if temp_path.exists():
                temp_path.unlink()  # Clean up the failed temp file
        except Exception as e:
            print(f" -> Unexpected error: {e}")
            if temp_path.exists():
                temp_path.unlink()

if __name__ == "__main__":
    reencode_videos()