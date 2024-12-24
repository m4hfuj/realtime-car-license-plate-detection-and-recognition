from ultralytics import YOLO
import supervision as sv
import cv2, os, pickle
from modules.lp_serializer import LPSerializer
lp_serializer = LPSerializer()


class Detector:
    def __init__(self):
        self.model = YOLO("models/weights_lpr/best.pt")
        self.cr = YOLO("models/weights_ocr/last.pt")
        self.tracker = sv.ByteTrack()


    def detect_lp(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.4)
            detections += detections_batch
        return detections


    def get_object_tracks(self, frames, read_from_stubs=False, stub_path=None):
        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            print(f"Loading track info stub from {stub_path}...")
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_lp(frames)        
        tracks = []

        for frame_num, detection in enumerate(detections):
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                conf = frame_detection[2]
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                tracks[frame_num][track_id] = {"track_id":track_id, "bbox":bbox, "conf":conf}

        if stub_path is not None:
            print(f"Saving track info stub into {stub_path}...")
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
    

    def detect_cr(self, bbox_frame):
        license_plate = []
        chars = self.cr(bbox_frame, conf=0.1)
        for char in chars:
            bbx = char.boxes
            text, conf = lp_serializer.serialize(bbx)
            license_plate.append({"text": text, "conf": conf})
        return license_plate


    def draw_annotations(self, frames, tracks):
        detections = {}

        output_frames = []
        confidences = {}
        max_conf = 0
        max_conf_frame_num = 0
        texts_by_frames = {0: "None"}

        for frame_num, track in enumerate(tracks):
            frame = frames[frame_num]
            # print(frame_num, track.items())

            for track_id, track in track.items():
                bbox = track["bbox"]
                bbox0 = int(bbox[0])
                bbox1 = int(bbox[1])
                bbox2 = int(bbox[2])
                bbox3 = int(bbox[3])
                w = bbox2 - bbox0
                h = bbox3 - bbox1
                cv2.imwrite(f"images/{track_id}_output.png", frame[bbox1:bbox1+h, bbox0:bbox0+w])
                license_plate = self.detect_cr(frame[bbox1:bbox1+h, bbox0:bbox0+w])
                print(track_id, license_plate)

                for lp in license_plate:
                    text = lp["text"]
                    texts_by_frames[frame_num] = text
                    conf = lp["conf"]
                    # cv2.putText(frame, text + "|" + str(conf)), (int(bbox[0]))
                    if track_id not in confidences:
                        confidences[track_id] = [conf]
                    else:
                        confidences[track_id].append(conf)

                prev_max_conf = max_conf
                max_conf = max(confidences[track_id])
                if prev_max_conf != max_conf:
                    max_conf_frame_num = frame_num

                    detections[track_id] = license_plate

                cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0, 0, 255), 2)
                # PARAMETERS
                font_scale = 2
                font = cv2.FONT_HERSHEY_DUPLEX
                thickness = 2
                # cv2.putText(frame, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                # Get the text size
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                # Define the bottom left corner of the text
                text_org = (bbox0 + 10, bbox3 + 50)
                # Draw a filled rectangle as the background
                cv2.rectangle(frame, 
                            (text_org[0]-10, text_org[1] - text_height - baseline), 
                            (text_org[0]+10 + text_width, text_org[1] + baseline), 
                            (0, 0, 255),  # Background color (BGR format)
                            -1)  # Thickness -1 means filled rectangle
                # Put the text over the rectangle
                cv2.putText(frame, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                # cv2.rectangle(frame, (50,1030), (), (255, 0, 0), 2)
                cv2.putText(frame, f"Best: {texts_by_frames[max_conf_frame_num]}", (50,1050), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Conf: {max_conf * 100:.2f} %", (50,1120), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

            output_frames.append(frame)

        return output_frames, texts_by_frames, detections




