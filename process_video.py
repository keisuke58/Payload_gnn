from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os

input_file = "docs/videos/H3_TF2_Digest.mp4"
output_dir = "docs/videos/clips"
os.makedirs(output_dir, exist_ok=True)

# Extract 3 clips of 20 seconds each to cover the likely launch window
clips = [
    (0, 20, "H3_TF2_Clip1_00-20.mp4"),
    (20, 40, "H3_TF2_Clip2_20-40.mp4"),
    (40, 60, "H3_TF2_Clip3_40-60.mp4")
]

for start, end, name in clips:
    output_path = os.path.join(output_dir, name)
    print(f"Creating {output_path} from {start} to {end}...")
    try:
        ffmpeg_extract_subclip(input_file, start, end, targetname=output_path)
    except Exception as e:
        print(f"Error creating {name}: {e}")
