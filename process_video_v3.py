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
            output_path = os.path.join(output_dir, name)
            print(f"Creating {output_path} from {start} to {end}...")
            
            # Use subclipped (new API for v2.0?)
            # subclipped returns a COPY, subclip modifies in place?
            # Or maybe subclip is gone.
            try:
                new_clip = video.subclipped(start, end)
            except AttributeError:
                 # Fallback for older versions if needed, but we are on v2.2.1
                 # Maybe it's subclip on older versions?
                 # But previous error said no attribute 'subclip', did you mean 'subclipped'?
                 # So subclipped IS the method.
                 new_clip = video.subclipped(start, end)

            new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            print(f"Done: {name}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
