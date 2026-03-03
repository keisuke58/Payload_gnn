from moviepy import VideoFileClip
import os

input_file = "docs/videos/H3_TF2_Digest.mp4"
output_dir = "docs/videos/clips"
os.makedirs(output_dir, exist_ok=True)

# 0:58 to 1:31
start_time = 58
end_time = 91  # 1 min 31 sec

output_name = "H3_TF2_Clip_00-58_to_01-31.mp4"
output_path = os.path.join(output_dir, output_name)

try:
    with VideoFileClip(input_file) as video:
        print(f"Video duration: {video.duration} seconds")
        print(f"Cutting from {start_time} to {end_time}...")
        
        # moviepy v2.0+ uses subclipped (returns copy)
        # If it fails, we can try subclip (in-place or older version)
        # Based on previous turn, subclipped worked (or at least didn't crash with AttributeError like subclip did)
        # Wait, in the previous turn, the logs showed:
        # "AttributeError: 'VideoFileClip' object has no attribute 'subclip'. Did you mean: 'subclipped'?"
        # So I used subclipped in v3 and it worked.
        
        new_clip = video.subclipped(start_time, end_time)
        new_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Done: {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
