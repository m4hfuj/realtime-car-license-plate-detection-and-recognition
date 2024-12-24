import cv2
from ultralytics import YOLO
import pandas as pd
import time

from modules.lp_serializer import LPSerializer
lp_serializer = LPSerializer()


classes = []
with open('classes.txt') as txtFile:
    for line in txtFile:
        classes.append(line.strip())


lpr_model = YOLO("models/weights_lpr/best.pt")
ocr_model = YOLO("models/weights_ocr/last.pt")

video_path = 'input_videos/output_video.avi'
cap = cv2.VideoCapture(video_path)

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# Frame skipping parameter (e.g., process every 5th frame)
frame_skip = 4
frame_count = 0


# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1 / fps  # in seconds


# Process each frame
while True:
    start_time = time.time()

    ret, frame = cap.read()
    
    if not ret:
        print("End of video or no frame captured.")
        break

    # Skip frames
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Resize the frame to the desired resolution
    resized_frame = cv2.resize(frame, (1280, 720))

    # cv2.rectangle(resized_frame, (500, 0), (750, 720), (0, 255, 0), 2)


    # Run YOLO model on the frame
    # frame = resized_frame[ : , 500:750 ]
    frame = resized_frame
    results = lpr_model(frame)

    for result in results:
        boxes = result.boxes
        if boxes.shape[0] > 0:
            bbox0 = int(boxes.xyxy[0][0].item())
            bbox1 = int(boxes.xyxy[0][1].item())
            bbox2 = int(boxes.xyxy[0][2].item())
            bbox3 = int(boxes.xyxy[0][3].item())
            w = int(boxes.xywh[0][2].item())
            h = int(boxes.xywh[0][3].item())

            cv2.rectangle(frame, (bbox0, bbox1), (bbox2, bbox3), (0, 0, 255), 1)
            # text = 'No Detection'

            ## HERE THE PLATE IMAGE IS SEPARATED AND FURTHER CHARACTERS DETECTED
            plate_image = frame[bbox1:bbox1+h, bbox0:bbox0+w]


            chars = ocr_model(plate_image, conf=0.1)

            for char in chars:
                bbx = char.boxes

                text = lp_serializer.serialize(bbx)


                

                # PARAMETERS
                font_scale = 1
                font = cv2.FONT_HERSHEY_DUPLEX
                thickness = 1

                # Add text to the frame
                # if len(lic) >= 8:

                # Get the text size
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                # Define the bottom left corner of the text
                text_org = (bbox0 + 10, bbox3 + 30)
                # Draw a filled rectangle as the background
                cv2.rectangle(resized_frame, 
                            (text_org[0]-10, text_org[1] - text_height - baseline), 
                            (text_org[0]+10 + text_width, text_org[1] + baseline), 
                            (0, 0, 255),  # Background color (BGR format)
                            -1)  # Thickness -1 means filled rectangle
                # Put the text over the rectangle
                cv2.putText(resized_frame, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)




    # Display the resulting frame
    cv2.imshow('YOLOv8 Detection', resized_frame)

    # Calculate processing time and adjust wait time
    elapsed_time = time.time() - start_time
    wait_time = max(1, int((frame_duration * frame_skip - elapsed_time) * 1000))

    # Press 'q' to exit the video
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
