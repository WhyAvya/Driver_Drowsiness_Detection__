# Driver Drowsiness Detection using Computer Vision and Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-FacialLandmarks-orange)
![Status](https://img.shields.io/badge/Status-Active-success)


A real-time driver drowsiness detection system that uses computer vision, MediaPipe, and a deep learning-based eye-state classification model to monitor driver alertness and trigger warnings during signs of fatigue.

The project combines real-time webcam processing with an EfficientNet-based classifier to detect prolonged eye closure and identify drowsiness in practical driving scenarios.

---

## Output Preview

### Awake State Detection

<p align="center">
  <img src="assets/awake_output.png" width="650"/>
</p>

<p align="center">
  <em>Real-time detection showing normal eye activity and awake status.</em>
</p>


### Drowsiness Detection

<p align="center">
  <img src="assets/drowsy_output.png" width="650"/>
</p>

<p align="center">
  <em>System detecting prolonged eye closure and triggering drowsiness alert.</em>
</p>

---

## Features

### Real-Time Driver Monitoring

The system continuously captures live webcam frames and performs real-time driver monitoring without requiring pre-recorded video input.


### Eye-State Based Drowsiness Detection

Instead of relying on full facial analysis, the project focuses specifically on eye-state classification to detect fatigue more efficiently and reduce unnecessary processing.


### EfficientNet-Based Deep Learning Model

An EfficientNet-based classifier is used to distinguish between open and closed eye states. The model is trained on labeled eye-image datasets for robust real-time prediction.


### MediaPipe Facial Landmark Detection

The project uses MediaPipe for lightweight and accurate facial landmark extraction, enabling reliable eye localization during live video processing.


### Frame-Based Drowsiness Logic

The system does not trigger alerts from a single closed-eye prediction. Consecutive frame analysis is used to differentiate normal blinking from actual drowsiness, reducing false alarms.


### Real-Time Alert System

An alarm mechanism is activated when prolonged eye closure is detected, helping provide immediate feedback during fatigue conditions.


### Modular Project Architecture

The codebase is organized into separate modules for preprocessing, inference, detection logic, webcam handling, and alarm management, making the project easier to maintain and extend.


### GPU-Compatible Training and Inference

The project supports CUDA-enabled GPU acceleration through PyTorch for faster model training and improved inference performance.


### Lightweight and Practical Pipeline

The implementation is designed to remain lightweight enough for academic demonstrations and real-time execution on standard systems without specialized hardware.


### Expandable System Design

The current pipeline can be extended with additional safety features such as yawning detection, head pose estimation, facial expression analysis, or embedded deployment.

---

## Project Structure

The project follows a modular architecture where different components of the real-time drowsiness detection pipeline are separated into dedicated modules for preprocessing, inference, detection logic, and alarm handling.

The repository is organized to keep the training, evaluation, and real-time inference workflows clean, maintainable, and easy to extend.


```bash
Driver_Drowsiness_Detection__/
│
├── assets/                          
│   ├── alarm.wav                    # Alarm audio used during drowsiness detection
│   ├── awake_output.png             # Awake-state output preview
│   └── drowsy_output.png            # Drowsiness detection output preview
│
├── data/
│   └── dataset/                     # Dataset directory
│       ├── train/
│       │   ├── open/
│       │   └── closed/
│       ├── val/
│       │   ├── open/
│       │   └── closed/
│       └── test/
│           ├── open/
│           └── closed/
│
├── models/                          # Model architecture and trained model weights
│   ├── model_arch.py
│   └── best_eye_state_effnet.pth
│
├── src/                             # Core project modules
│   ├── camera.py                    # Webcam handling and frame capture
│   ├── mediapipe_eye_detector.py    # Face and eye landmark detection
│   ├── preprocess.py                # Image preprocessing and transformations
│   ├── inference.py                 # Eye-state prediction pipeline
│   ├── drowsiness_logic.py          # Drowsiness scoring and decision logic
│   └── alarm.py                     # Alarm triggering and control
│
├── utility/
│   └── cudacheck.py                 # CUDA/GPU availability check
│
├── app.py                           # Main real-time drowsiness detection pipeline
├── main.py                          # Project execution entry point
├── train.py                         # Model training script
├── evaluate.py                      # Model evaluation and testing script
│
├── requirements.txt                 # Required project dependencies
├── .gitignore
└── README.md
```

---

## Tech Stack

The project combines computer vision and deep learning frameworks to support real-time eye-state classification, facial landmark detection, and driver drowsiness analysis.

The selected technologies were used to build a modular and efficient real-time detection pipeline for both training and inference workflows.

### Programming Language

- Python

### Deep Learning Framework

- PyTorch
- TorchVision

### Computer Vision

- OpenCV
- MediaPipe

### Data Processing

- NumPy

### Model Architecture

- EfficientNet

### Development Environment

- Visual Studio Code
- Jupyter Notebook

### Hardware Support

- CUDA-enabled GPU support (optional)

---

## Workflow

The system follows a real-time computer vision pipeline where webcam frames are processed continuously to detect eye states and identify signs of driver drowsiness.

The workflow combines facial landmark detection, deep learning-based eye-state classification, and frame-based decision logic to reduce false alerts and improve real-time reliability.

```text
Webcam Input
      │
      ▼
Face & Eye Detection (MediaPipe)
      │
      ▼
Eye Region Extraction
      │
      ▼
Image Preprocessing
      │
      ▼
EfficientNet Eye-State Prediction
      │
      ▼
Frame-Based Drowsiness Logic
      │
      ▼
Drowsiness Detection
      │
      ▼
Alarm Triggered
```

---

### Workflow Steps

1. The webcam continuously captures live video frames in real time.

2. MediaPipe facial landmark detection is used to identify the face and locate both eye regions.

3. The detected eye regions are extracted from the frame for further processing.

4. The extracted eye images are preprocessed and transformed into a format suitable for model inference.

5. The EfficientNet-based model predicts whether the eyes are open or closed.

6. Consecutive frame analysis is applied to monitor prolonged eye closure and differentiate normal blinking from drowsiness.

7. If the drowsiness threshold is exceeded, the system triggers an alarm alert.

---

### Key Concepts

- MediaPipe — A framework used for real-time facial landmark and eye detection.

- Preprocessing — The process of resizing, normalizing, and preparing images before model prediction.

- Inference — The stage where the trained deep learning model predicts the eye state from input images.

- EfficientNet — A deep learning model architecture used for eye-state classification.

- Frame-Based Logic — A detection approach that analyzes consecutive video frames to reduce false alerts caused by normal blinking.

---

## Installation

Follow the steps below to set up the project locally on your system.

### Clone the Repository

```bash
git clone https://github.com/WhyAvya/Driver_Drowsiness_Detection__.git
```

### Navigate to the Project Directory

```bash
cd Driver_Drowsiness_Detection__
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

### Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Instructions

### Run the Real-Time Drowsiness Detection System

```bash
python main.py
```

### Train the Model

```bash
python train.py
```

### Evaluate the Trained Model

```bash
python evaluate.py
```

---

## Model Details

The project uses an EfficientNet-based deep learning model for eye-state classification. The model is trained to classify eye images into two categories:

- Open Eyes
- Closed Eyes

The trained model is integrated with a real-time computer vision pipeline to detect prolonged eye closure and identify drowsiness conditions.

### Model Characteristics

- Deep learning-based eye-state classification
- Real-time inference support
- PyTorch implementation
- EfficientNet architecture
- CUDA-compatible GPU acceleration support

---

## Results

The system was successfully tested using real-time webcam input for driver drowsiness detection.

### Key Outcomes

- Accurate eye-state classification for open and closed eyes
- Real-time face and eye detection using MediaPipe
- Stable real-time inference performance
- Successful alarm triggering during prolonged eye closure
- Reduced false alerts through frame-based drowsiness logic

### Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Measures correct eye-state predictions |
| Precision | Measures prediction reliability |
| Recall | Measures detection sensitivity |
| Confusion Matrix | Shows prediction distribution across classes |

---

## Future Improvements

The current system can be further improved and extended with additional features and optimizations.

### Possible Enhancements

- Yawning detection integration
- Head pose estimation
- Mobile or embedded deployment
- Night-time driver monitoring
- Multi-driver support
- Improved low-light performance
- Real-time dashboard or web interface
- Advanced fatigue analysis using multiple facial cues

---

## Author

Developed by **Avya Sharma**

📧 Email: sharma.avya04@gmail.com

💼 LinkedIn: https://linkedin.com/in/avyasharma

🐙 GitHub: https://github.com/WhyAvya

Computer Vision and Deep Learning Project focused on real-time driver drowsiness detection using MediaPipe, OpenCV, and PyTorch.