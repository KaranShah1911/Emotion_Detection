# 🧠 Real-Time Face Emotion Detection

## 📌 Overview
This project is a **real-time face emotion detection system** built using **Python, OpenCV, and DeepFace**. It captures live video from the webcam, detects faces, and classifies the **dominant emotion** of each detected face. The system overlays bounding boxes and emotion labels on the video feed to provide an interactive visualization.

This project demonstrates the practical application of **computer vision** and **deep learning** in understanding human emotions through facial expressions.

---

## 🎯 Features
- 🔴 **Real-time video capture** using webcam  
- 😀 **Emotion detection** (happy, sad, angry, neutral, surprise, fear, disgust)  
- 🖼️ **Face detection** using OpenCV Haar Cascade  
- ⚡ **Performance optimization** (frame skipping & input resizing)  
- 🖊️ **On-screen overlays**: bounding boxes + emotion labels  
- ❌ **Graceful exit**: close the window or press `q`

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:**  
  - [OpenCV](https://opencv.org/) – video capture, detection, visualization  
  - [DeepFace](https://github.com/serengil/deepface) – pre-trained models for facial emotion recognition  
  - **NumPy** (dependency for OpenCV/DeepFace)

---

## ⚙️ Installation

1) **Clone the repository**
```bash
git clone https://github.com/KaranShah1911/Emotion_Detection.git
cd Emotion_Detection
```

2) **Install Dependencies**
```bash
pip install -r requirements.txt
```

3) **Run the Project**
```bash
python emotion_detection.py
```

## 📊 How It Works
1. **Capture frames** from the webcam using OpenCV.  

2. **Convert frames to grayscale** and **detect faces** with a Haar Cascade classifier.  

3. On every *n-th* frame (e.g., every 5th):  
   - **Extract** the face region (ROI)  
   - **Resize** to a smaller size (e.g., `224×224`) for faster inference  
   - **Analyze** emotions with DeepFace and store the **dominant emotion**  

4. **Overlay** bounding boxes and the **last known emotion** on the frame.  

5. **Display** processed frames in real time; **exit** on `q` or when the window is closed.  
