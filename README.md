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

### **1. License Plate Detection Model**
- **YOLO-based Localization**: Uses a pre-trained YOLO model to detect and localize license plates in video frames with bounding boxes.
- **Real-Time Performance**: Optimized for fast video processing, enabling quick detection in large or high-traffic videos.
- **Separation of Tasks**: Focuses on license plate localization while leaving character recognition as separate tasks.

### **2. Character Recognition**
- **Bangla Character Recognition (OCR)**: After detecting the plate, another **YOLO-based model** recognizes characters such as district names, letters, and numbers on the license plate.
- **OCR for Bangla License Plates**: The character recognition model is specialized for Bangla license plates, extracting district names (e.g., **Dhaka**, **Chattogram**) and numerical characters, as well as the alphabetic letters (e.g., **Cha**, **Ba**, **Da**). The model efficiently decodes the characters even under occlusions or varying light conditions.
- **Efficient Recognition**: Specialized for Bangla plates, even under occlusions or poor lighting conditions.
- **Confidence Scores**: Each detected character has a confidence score to indicate recognition accuracy.
- **Localized Detection**: Characters are detected individually with bounding boxes to handle partial obstructions or angled plates.

### **3. Video Annotation**
- **Bounding Boxes & Labels**: Annotates the video with bounding boxes around detected vehicles and license plates.
- **Text & Confidence**: Displays recognized license plate text (Bangla characters) with confidence scores directly on the video.
- **Real-Time Display**: Provides live annotations during video playback with visual feedback for each detected plate.

### **4. CSV Output**
- **Detection Results**: Generates a CSV (`detection_results.csv`) containing detected license plate text and confidence scores.
- **Track IDs**: Includes vehicle track IDs, license plate text, and confidence scores for each frame.
- **Data Analysis**: CSV format is ready for further analysis, visualization, or integration with other systems.
- **Example CSV**:

    ```
    track_ids,detections
    1,‘ঢাকা মেট্রো ব ১৪-১২৩৪’
    2,‘রাজশাহী চ ৭০-৪৬৪১’
    3,‘চট্টগ্রাম ব ৬৪-৮৪৮৬’
    ```

### **5. Video Output**
- **Annotated Video**: Outputs a video with bounding boxes, license plate text, and confidence scores annotated.
- **Real-Time Processing**: Processes and annotates the video in real-time.
- **Vehicle Tracking**: Tracks vehicles across frames using unique track IDs.
- **Customizable Output**: Allows customization of output video parameters (e.g., frame rate, resolution).
- **Example Output**: Shows each vehicle’s bounding box and plate text with confidence scores in the final annotated video.


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