from modules.video_utils import read_video, write_video
from modules.detector import Detector


frames = read_video('input_videos/output_video.avi')

detector = Detector()

tracks = detector.get_object_tracks(frames, read_from_stubs=True, stub_path="stubs/stub.pkl")

# print(tracks)

output_frames, texts_by_frames, detections = detector.draw_annotations(frames, tracks)

# for d in detections:
#     print(d)

import pandas as pd

pd.DataFrame({
    "track_ids": detections.keys(),
    "detections": detections.values(),    
}).to_csv("output/detection_results.csv")

write_video('output/test_video2.avi', output_frames)
