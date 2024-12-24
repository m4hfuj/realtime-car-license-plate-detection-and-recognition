# 🚍 Automatic Bangla License Plate Recognition (ALPR)


![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/-YOLOv8-FF9900?logo=yolo&logoColor=white)
![ByteTrack](https://img.shields.io/badge/-ByteTrack-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
---


![gif](assets/project-alpr.gif)


## 🗺️ Overview

This project is designed for automatic Bangla License Plate Recognition (ALPR) using deep learning models. It includes functionality for detecting vehicles in a video, recognizing their license plates, and displaying relevant information on the video frames. The system is built using Python with libraries like YOLO, OpenCV, and Supervision.

The ALPR system performs the following tasks:

- Detects vehicles in video frames.
- Extracts license plates from vehicles.
- Recognizes the characters on the license plates using Optical Character Recognition (OCR).
- Annotates the video with detected license plates and confidence scores.
- Outputs the processed video with license plate annotations.


## 🚀 Key Features:

### **1. Vehicle Detection**

- **YOLOv8 Object Detection**: The system leverages the powerful **YOLOv8** (You Only Look Once) model, which is designed for fast and accurate object detection in images and videos.
- **Real-time Performance**: The vehicle detection process is optimized for real-time performance, allowing for efficient analysis of video frames, even in large video files.
- **Bounding Boxes**: Detected vehicles are highlighted with bounding boxes, allowing easy identification of each vehicle in the video frames.
- **Multiple Vehicle Detection**: The system can detect and track multiple vehicles simultaneously in each frame, making it suitable for crowded environments such as highways or city streets.

### **2. License Plate Recognition (LPR)**

- **Bangla License Plate Recognition**: This project focuses on recognizing **Bangla characters** on vehicle license plates. The system is equipped with a **pre-trained deep learning model** specifically designed for recognizing Bangla characters.
- **OCR (Optical Character Recognition)**: After detecting the license plate region, the **OCR model** extracts individual characters from the license plate using high-accuracy recognition algorithms. It is capable of detecting characters even under challenging conditions like occlusions or low light.
- **Accuracy and Precision**: The system has been trained on a dataset of Bangla license plates, ensuring accurate recognition of characters with high confidence. Each detected plate’s text is returned with an associated confidence score indicating the reliability of the recognition.
- **Character-Level Recognition**: Each character is identified individually, ensuring that even complex license plates with various combinations of letters and numbers are accurately decoded.

### **3. Video Annotation**

- **Bounding Boxes and Labels**: Once a vehicle and its license plate are detected, the system annotates the video frames by drawing bounding boxes around the vehicles and their corresponding license plates.
- **License Plate Text Annotation**: The detected license plate’s text (Bangla characters) is displayed directly on the video frame, with a confidence score showing how certain the model is about its recognition.
- **Real-Time Display**: The video frames are displayed with annotations in real time, giving you a live view of the detected vehicles and recognized license plates.
- **Visual Feedback**: The system draws colored rectangles around detected plates and adds text annotations that display the detected license plate number along with the confidence score for each detection.

### **4. CSV Output**

- **Detailed Detection Results**: The system generates a CSV file (`detection_results.csv`) that stores the detected license plate text and their associated confidence scores for each frame. This provides an easy way to analyze detection results post-processing.
- **Track IDs and Confidence Scores**: Each row in the CSV contains the track ID of the vehicle, the detected license plate text, and the confidence score for that particular license plate detection.
- **Data for Further Analysis**: The CSV format makes it easy to process, visualize, and analyze the data further, including for statistics, vehicle pattern analysis, or integration with other systems.
- **Example CSV Output**:

    ```
    track_ids,detections
    1,‘ঢাকা মেট্রো ব ১৪-১২৩৪’
    2,‘রাজশাহী চ ৭০-৪৬৪১’
    3,‘চট্টগ্রাম ব ৬৪-৮৪৮৬’
    ```
### **5. Video Output**
- **Annotated Output Video**: The processed video, with all detected vehicles and their license plates annotated, is saved in the output directory (e.g., `test_video2.avi`).
- **Real-time Processing**: The system processes the input video, annotates it with detected vehicles and their license plate texts, and generates a new video output, all in one step.
- **Track Vehicle Movement**: The output video includes tracking of vehicles across multiple frames. As the system detects a vehicle, it assigns it a unique track ID that is used to track that vehicle across frames.
- **Customizable Output**: The final output video can be adjusted to include different annotation types (e.g., bounding boxes, text overlays, confidence scores). The video format is customizable, and you can adjust parameters like **frame rate**, **resolution**, and **codec**.
- **Example Output**: The annotated output video shows each vehicle’s bounding box and its corresponding license plate text with confidence scores, providing a complete visual report of the detections.

---




## 📂 File Structure

```plaintext
/Project
│
├── /assets                # Assets for visualization (e.g., gifs, images)
├── /input_video           # Folder for input videos
│   └── video.avi          # Sample input video
├── /output                # Folder for output videos and results
│   └── output_video.avi   # Processed video output
│   └── detection_results.csv  # CSV file containing the detection results
├── /images                # Folder for saving individual frame images
├── /models                # Folder for pretrained models
│   ├── weights_lpr
│   │   └── best.pt        # Pretrained License Plate Recognition model
│   └── weights_ocr
│       └── best.pt        # Pretrained OCR model for character recognition
├── /modules               # Python scripts for different utilities
│   ├── detector.py        # Object detection and tracking logic
│   ├── video_utils.py     # Video reading and writing functions
│   └── lp_serializer.py   # License plate serialization and character extraction
├── /README.md             # Project documentation (this file)
├── /main.py               # Main script for running the ALPR system
├── /cut.py                # Script for cutting video segments
└── /classes.txt           # List of class labels used for ALPR
```


## 📌 Requirements

Dependencies:

- `opencv-python`: For video processing and displaying the output.
- `ultralytics`: For running YOLO models for vehicle and license plate detection.
- `supervision`: A tracking library for managing object detection.
- `pandas`: For data handling and saving detection results in CSV format.
- `tqdm`: For showing progress bars during video writing.

You can install the necessary libraries using the following command:

```bash
pip install opencv-python ultralytics supervision pandas tqdm
```


## ⚡ Running the Project

Running the ALPR System

To run the system and process a video, use the run.py script:

```bash
python main.py
```

The script will:

- Load the input video from input_videos/output_video.avi.
- Detect vehicles and license plates using YOLO and OCR models.
- Annotate the video with bounding boxes and license plate text.
- Save the annotated video to output/test_video2.avi.
- Save the detection results to output/detection_results.csv.



## 🧾 License

This project is open-source and licensed under the MIT License.



## 🪧 Acknowledgments

    Ultralytics YOLOv8 for object detection and license plate recognition models.
    OpenCV for video processing and annotation.
    Supervision for object tracking in video frames.