# 🤟 Sign Language Detection Using Machine Learning

A real-time hand gesture recognition system that detects American Sign Language (ASL) letters (A–Z) from a live webcam feed and translates them into text using deep learning.

---

## 📌 Description

This project uses **MediaPipe** for hand landmark extraction and an **LSTM-based neural network** to classify sign language gestures in real time. Instead of processing raw images, the model works on 21 3D hand keypoints per frame, making it lightweight, fast, and robust to background variation.

---

## 🗂️ Project Structure
---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Hand Detection | MediaPipe |
| Computer Vision | OpenCV |
| Data Handling | NumPy, scikit-learn |
| Training Monitoring | TensorBoard |

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Hand Detection | MediaPipe |
| Computer Vision | OpenCV |
| Data Handling | NumPy, scikit-learn |
| Training Monitoring | TensorBoard |

---

## 🧠 Model Architecture

The model processes sequences of **30 frames** (each with 63 keypoint values) through stacked LSTM layers:

## 🧠 Model Architecture

The model processes sequences of **30 frames** (each with 63 keypoint values) through stacked LSTM layers:
Input (30 frames × 63 keypoints)
→ LSTM(64, return_sequences=True)
→ LSTM(128, return_sequences=True)
→ LSTM(64, return_sequences=False)
→ Dense(64, relu)
→ Dense(32, relu)
→ Dense(num_classes, softmax)
- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 200  

---

## ⚙️ How It Works

1. **Data Collection** — `collectdata.py` captures webcam images for each letter (A–Z) and saves them to `Image/`.
2. **Keypoint Extraction** — `data.py` uses MediaPipe to extract 21 hand landmarks (x, y, z) per frame and saves 30-frame sequences as `.npy` files in `MP_Data/`.
3. **Model Training** — `trainmodel.py` loads the keypoint sequences, trains the LSTM model, and saves `model.json` + `model.h5`.
4. **Real-Time Detection** — `app.py` loads the trained model, reads live webcam frames, extracts keypoints, and predicts the sign with a confidence threshold of **80%**.

---

## 🚀 Getting Started

### Prerequisites

pip install tensorflow mediapipe opencv-python numpy scikit-learn

### Run Real-Time Detection

python app.py

Place your hand inside the highlighted rectangle on screen. The detected letter will appear at the top of the feed. Press **`q`** to quit.

### Retrain the Model

# Step 1: Collect images
python collectdata.py

# Step 2: Extract keypoints
python data.py

# Step 3: Train the model
python trainmodel.py

---

## 🎯 Key Features

- ✅ Keypoint-based — Works on hand landmarks, not raw pixels; fast and background-independent
- ✅ Temporal modeling — Uses 30-frame sequences for stable, context-aware predictions
- ✅ Confidence filtering — 80% threshold + 10-frame consistency check reduces false positives
- ✅ Active ROI — Defined detection zone guides the user for better accuracy
- ✅ Pre-trained model included — Run inference immediately without retraining

---




