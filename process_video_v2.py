from moviepy import VideoFileClip
import os

input_file = "docs/videos/H3_TF2_Digest.mp4"
output_dir = "docs/videos/clips"
os.makedirs(output_dir, exist_ok=True)

# Extract 3 clips
clips = [
    (0, 20, "H3_TF2_Clip1_00-20.mp4"),
    (20, 40, "H3_TF2_Clip2_20-40.mp4"),
    (40, 60, "H3_TF2_Clip3_40-60.mp4")
]

try:
    with VideoFileClip(input_file) as video:
        print(f"Video duration: {video.duration} seconds")
        for start, end, name in clips:
            if start >= video.duration:
                print(f"Skipping {name} (start {start} > duration {video.duration})")
                continue
            
            output_path = os.path.join(output_dir, name)
            print(f"Creating {output_path} from {start} to {end}...")
            
            # Use subclip (returns a new clip)
            # subclip handles end > duration automatically usually, or we clamp it
            actual_end = min(end, video.duration)
            new_clip = video.subclip(start, actual_end)
            
            # Write using default codec (libx264 for mp4)
            # audio_codec='aac' is safer for compatibility
            new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
            print(f"Done: {name}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
