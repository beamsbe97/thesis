import subprocess
import json
from pathlib import Path
import pandas as pd

# --- CONFIGURATION ---
DATASET_ROOT = Path("/zfsstore/courses/2025-2026/4343SADL6/g10/EPIC-KITCHENS").expanduser()
CSV_PATH = Path("~/EPIC-KITCHENS/EPIC_100_video_info.csv").expanduser()
# ---------------------

def get_actual_fps(video_path):
    """Uses ffprobe to extract the exact frame rate of a video file."""
    cmd = [
        "ffprobe", "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=r_frame_rate", 
        "-of", "json", 
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        # r_frame_rate is usually returned as a fraction string, e.g., "60/1" or "30000/1001"
        fps_string = data['streams'][0]['r_frame_rate']
        num, den = map(int, fps_string.split('/'))
        return num / den
    except Exception:
        return None

def verify_dataset():
    if not CSV_PATH.exists():
        print(f"Error: '{CSV_PATH}' not found.")
        return

    df = pd.read_csv(CSV_PATH)
    # Target the exact same subset that needed re-encoding
    target_videos = df[(df['fps'] < 59) | (df['fps'] > 60)]
    
    print(f"Verifying the {len(target_videos)} videos that required updating...")
    print("-" * 50)
    
    success_count = 0
    failed_videos = []
    missing_videos = []

    for idx, row in target_videos.iterrows():
        video_id = row['video_id']
        participant_id = video_id.split('_')[0]
        video_path = DATASET_ROOT / participant_id / "videos" / f"{video_id}.mp4"
        print(video_path)
        
        if not video_path.exists():
            missing_videos.append(video_id)
            continue
            
        actual_fps = get_actual_fps(video_path)
        
        # Check if the live FPS is now tightly bound to 60fps
        if actual_fps and (59.9 <= actual_fps <= 60.1):
            success_count += 1
        else:
            failed_videos.append((video_id, actual_fps))

    # --- REPORT RESULTS ---
    print("\n" + "="*40 + "\nVERIFICATION REPORT\n" + "="*40)
    print(f"✅ Successfully verified at 60fps: {success_count}/{len(target_videos)}")
    
    if missing_videos:
        print(f"⚠️ Missing files (not found on disk): {len(missing_videos)}")
        print(f"   Files: {missing_videos[:5]} ...")
        
    if failed_videos:
        print(f"❌ Failed validation (still not 60fps): {len(failed_videos)}")
        for vid, fps in failed_videos:
            print(f"   - {vid}: Current FPS is {fps}")
    else:
        if success_count == len(target_videos):
            print("\n🎉 Success! All targeted Epic Kitchens videos are now perfectly encoded at 60fps.")

if __name__ == "__main__":
    verify_dataset()