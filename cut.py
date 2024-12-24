import cv2
import numpy as np

# Path to the input video
input_video_path = 'video/video.avi'
output_video_path = 'video/output_video.avi'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID' or 'MJPG'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Define the segments to cut (start and end times in seconds)
segments = [
    (37, 47),
    (80, 130),   # From 1:20 to 2:10
    (161, 180),  # From 2:41 to 3:00
    (120+53, 120+58),
    # (214, 226),   # From 3:34 to 3:46

]

# Loop through each segment and write the frames to the output video
for start_time, end_time in segments:
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set the starting point for the segment
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read and write frames within the segment range
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Write the current frame to the output video
        out.write(frame)

# Release resources
cap.release()
out.release()

print("Video processing complete. Output saved at:", output_video_path)
