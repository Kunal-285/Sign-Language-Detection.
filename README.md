# 🤟 SignAI — Neural Gesture Engine

> **Real-time Sign Language Detection using AI, MediaPipe & LSTM Neural Networks**

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?style=for-the-badge&logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Detection-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 About The Project

**SignAI** is a real-time sign language detection system that uses a webcam to recognize hand gestures and convert them into readable text. Built with Flask, MediaPipe, and a custom-trained LSTM neural network, it provides an accessible web-based interface — no special hardware or installation required beyond Python.

This project was developed as part of a tech fest submission at **Poornima University** to address communication barriers faced by hearing and speech-impaired individuals.

---

## ✨ Features

- 🎥 **Real-time Detection** — Processes live webcam feed at ~8 frames/second
- 🧠 **LSTM Neural Network** — 30-frame sequence-based gesture recognition
- ✋ **MediaPipe Hand Tracking** — 21 landmark keypoints, wrist-normalized
- 📊 **Confidence Filtering** — Only predictions above 75% confidence are shown
- 🔁 **Prediction Smoothing** — 5-frame history prevents flickering
- 📝 **Sentence Builder** — Add detected words to form full sentences
- 🌐 **Web-Based** — Works in any modern browser, no app install needed
- 🎨 **Neural Canvas UI** — Animated background visualizing the neural network

---

## 🤖 Supported Gestures

| Gesture | Sign |
|---------|------|
| HELLO | 👋 |
| NO | ❌ |
| PEACE | ✌️ |
| STOP | 🛑 |

> More gestures can be added by collecting new data with `collect_data.py` and retraining.

---

## 🏗️ Project Structure

```
Sign-Language-Detection/
│
├── Backend/
│   ├── Main.py              # Flask server + prediction API
│   ├── collect_data.py      # Data collection script
│   ├── train_model.py       # LSTM model training script
│   └── signetra_best.h5     # Trained model (not in repo — generate locally)
│
├── static/
│   ├── style.css            # UI styling
│   └── script.js            # Frontend logic, FPS counter, neural canvas
│
├── templates/
│   └── index.html           # Main web interface
│
├── requirements.txt         # Python dependencies
└── README.md
```

---

## ⚙️ How It Works

```
Webcam Input
     ↓
MediaPipe Hand Detection (21 landmarks)
     ↓
Wrist-Relative Normalization → (63 features)
     ↓
30-Frame Sequence Buffer
     ↓
LSTM Neural Network Prediction
     ↓
Confidence Filter (≥ 75%) + 5-Frame Smoothing
     ↓
Text Display on Web Interface
     ↓
Sentence Builder (copy full sentence)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Webcam

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Kunal-285/Sign-Language-Detection..git
cd Sign-Language-Detection.
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model (or use pretrained)**
```bash
# First collect data for each gesture
python Backend/collect_data.py

# Then train
python Backend/train_model.py
```

**4. Run the app**
```bash
python Backend/Main.py
```

**5. Open in browser**
```
http://127.0.0.1:5000/
```

---

## 📦 Requirements

```
flask
numpy
opencv-python
mediapipe
tensorflow
scikit-learn
matplotlib
seaborn
```

---

## 🧠 Model Architecture

```
Input: (30 frames × 63 features)
        ↓
LSTM Layer (128 units) + BatchNorm + Dropout(0.3)
        ↓
LSTM Layer (64 units) + BatchNorm + Dropout(0.3)
        ↓
Dense Layer (64 units, ReLU) + Dropout(0.2)
        ↓
Output: Softmax (4 classes)
```

- **Training Data:** 200 sequences per gesture × 3 (with augmentation) = 600 samples/gesture
- **Augmentation:** Gaussian noise + random scale + time shift
- **Optimizer:** Adam (lr=0.001) with ReduceLROnPlateau
- **Best model** saved automatically via ModelCheckpoint

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/predict` | POST | Send frame → get prediction |
| `/health` | GET | Server status + stats |
| `/reset` | POST | Clear frame buffer |

---

## 👨‍💻 Team

| Name | Role |
|------|------|
| Kunal Bairwa | Lead Developer |
| Prateek Sharma | ML & Training |
| Jhanvi Kumar Aloda | Frontend & UI |
| Neha Singhal | Data Collection & Testing |

**Guide:** Mrs. Priyanka Tiwari
**Institution:** Poornima University, Jaipur
**Year/Semester:** 1st Year / 2nd Semester

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Made with ❤️ at Poornima University | Tech Fest 2025</p>
