import cv2
from tqdm import tqdm


def read_video(path):
    frames = []
    frame_no = 0
    print(f"- reading from {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        frame_no += 1
        if not ret:
            print("  end of video or no frame captured.")
            break
        # if frame_no % 1 == 0:
        frames.append(frame)
    
    print(f"- read {len(frames)} frames from {path}")

    return frames


def write_video(video_path, frames, fps=24):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in tqdm(frames, desc='Writing Video'):
        out.write(frame)
    out.release()
