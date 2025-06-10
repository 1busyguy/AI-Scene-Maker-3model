from moviepy.editor import VideoFileClip, concatenate_videoclips

# List your video file paths in the order you want to stitch them
video_files = [
    "processed_chain_01.mp4",
    "processed_chain_02.mp4",
    "processed_chain_03.mp4"
]

# Load each video file as a VideoFileClip
clips = [VideoFileClip(v) for v in video_files]

# Concatenate video clips
final_clip = concatenate_videoclips(clips, method="compose")

# Export the result
final_clip.write_videofile("stitched_video.mp4", codec="libx264", audio_codec="aac")

# Close the clips to release resources
for clip in clips:
    clip.close()
final_clip.close()
