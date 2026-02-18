# 🛡️ Vision-Based Guard Monitoring System

An AI-powered real-time CCTV analytics system for **security guard compliance monitoring**, built using Computer Vision and Deep Learning.

This system upgrades traditional CCTV feeds into an intelligent monitoring platform capable of detecting:

- Guard presence / absence  
- Inactivity / idleness  
- Sleeping on duty  
- Mobile phone usage  
- Distraction behaviors  
- Real-time behavior violations  

---

# 📌 Project Scope (As Per Product Document)

## Core Detection & Alert Capabilities

✔ Real-time detection of people and behaviors from live video  
✔ Intelligent monitoring without hardware replacement  
✔ Automated supervision to reduce manual oversight  
✔ Real-time flagging of suspicious or negligent behavior  
✔ Enhanced operational efficiency  

---

## Personnel Monitoring (Guard Compliance)

✔ Detect guard absence from designated post  
✔ Detect sleeping on duty  
✔ Detect prolonged idleness  
✔ Detect mobile phone usage  
✔ Detect distraction behaviors  
✔ Behavior persistence → Possible / Confirmed violation logic  

---

# 🧠 System Architecture

The system follows a modular ML-based architecture:

```
Video Stream
     ↓
Person Detection (YOLOv8)
     ↓
Tracking (Centroid Tracker)
     ↓
Face Detection + Recognition
     ↓
Behavior Classification (CLIP-based Zero-Shot)
     ↓
Temporal Behavior Engine
     ↓
Violation Decision Engine
     ↓
Alert Trigger (Next Phase)
```

---

# 🏗️ Folder Structure

```
Video-monitoring-tool/
│
├── main.py
├── requirements.txt
├── yolov8n.pt
├── kinetics_labels.txt
│
├── analytics/
│   ├── behavior_classifier.py
│   ├── behavior_engine.py
│   ├── action_recognition.py
│   ├── presence.py
│   ├── inactivity.py
│   └── violation_engine.py
│
├── detection/
│   ├── person_detector.py
│   ├── object_detector.py
│   └── tracker.py
│
├── face/
│   ├── face_detector.py
│   ├── face_encoder.py
│   ├── face_recognizer.py
│   └── models/
│       ├── face_detection_yunet_2023mar.onnx
│       └── lbfmodel.yaml
│
├── video/
│   ├── stream_reader.py
│   └── video_utils.py
│
├── training/
│   └── train_behavior.py
│
├── alerts/
│   └── alert_manager.py
│
└── data/
    └── enrolled_faces/
        └── <Guard_Name>/
```

---

# 🚀 Installation Guide

## Step 1 — Clone Repository

```bash
git clone https://github.com/Pratyaksh0404/Video-monitoring-tool.git
cd Video-monitoring-tool
```

---

## Step 2 — Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows
```bash
venv\Scripts\activate
```

### Mac/Linux
```bash
source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Run the System

```bash
python src/main.py
```

Press `q` to exit the application.

---

# 🧠 Behavior Detection Approach

Instead of fragile rule-based logic, this system uses:

## 🔹 OpenCLIP (RN50)
Zero-shot behavior classification using natural language prompts:

- "a security guard standing alert"
- "a security guard sleeping on duty"
- "a security guard using a mobile phone"
- "a security guard sitting idle"
- "a distracted security guard talking to someone"

## 🔹 Temporal Confirmation Engine

Each detected behavior passes through time persistence filtering:

| Duration         | Output      |
|------------------|-------------|
| less than 5 sec  | ANALYZING   |
| 5–10 sec         | POSSIBLE_X  |
| more than 10 sec | CONFIRMED_X |

This significantly reduces false positives.

---

# 👤 Face Recognition

Guards are enrolled via:

```
data/enrolled_faces/<Guard_Name>/
```

Multiple images per guard improve multi-angle recognition stability.

Face encodings are generated using the `face_recognition` (dlib-based) library.

---

# ⚙️ Current Capabilities

| Feature | Status |
|----------|--------|
| Guard Presence | ✅ Stable |
| Active / Inactive Detection | ✅ Stable |
| Sleeping Detection | ✅ Stable |
| Phone Usage Detection | ✅ Stable |
| Idle Detection | ✅ Stable |
| Distraction Detection | ✅ Stable |
| Identity Persistence | ✅ Implemented |
| Trajectory Tracking | 🔜 Next Phase |
| Alert System (Email/SMS) | 🔜 Next Phase |

---

# ⚡ Performance Notes

- Optimized for CPU execution  
- CLIP inference reduced frequency to minimize lag  
- Identity memory prevents rapid UNKNOWN switching  
- Startup stabilization prevents early false positives  

GPU support will significantly improve performance.

---

# 🛠️ Future Enhancements

- Real-time email/SMS alert integration  
- Guard trajectory mapping  
- Patrol route compliance verification  
- Dashboard interface  
- Multi-camera deployment  
- Docker deployment configuration  

---

# 📊 Technology Stack

- Python  
- OpenCV  
- YOLOv8 (Ultralytics)  
- OpenCLIP  
- PyTorch  
- dlib (face recognition)  
- NumPy  

---

# 🧪 Development Status

System is currently:

> Stable for demo and prototype validation  
> Ready for installation and demonstration  

Further tuning can improve robustness under extreme posture variations.

---

# ⚠️ Disclaimer

This system is intended for authorized security monitoring use cases only.
