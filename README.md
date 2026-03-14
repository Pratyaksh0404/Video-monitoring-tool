# 🛡️ Vision-Based Guard Monitoring System

An **AI-powered real-time CCTV analytics system** for **security guard compliance monitoring**, built using Computer Vision and Deep Learning.

This system upgrades traditional CCTV feeds into an **intelligent monitoring platform** capable of automatically detecting guard behavior violations and operational risks.

The system analyzes live video feeds to detect:

* Guard presence / absence
* Guard inactivity or idleness
* Sleeping on duty
* Mobile phone usage
* Distraction behaviors
* Guard patrol movement inside the monitored area
* Real-time violation alerts

---

# 📌 Project Scope

The goal of this project is to transform **standard CCTV systems into intelligent monitoring tools** capable of assisting supervisors in ensuring guard compliance and operational safety.

The system performs **real-time behavioral analysis** without requiring any additional hardware.

### Key Objectives

* Real-time guard monitoring from CCTV feeds 
* Automated detection of security violations 
* Reduced need for manual supervision 
* Real-time alert generation 
* Patrol movement analysis inside the monitored zone

---

# 👮 Guard Monitoring Capabilities

The system monitors security guard activity using computer vision models.

### Guard Compliance Detection

* Guard absence from assigned post 
* Sleeping on duty 
* Prolonged inactivity 
* Mobile phone usage 
* Distraction behaviors

### Patrol Monitoring

* Guard trajectory tracking inside monitored area 
* Zone-based patrol tracking 
* Patrol movement logging

Example trajectory:

```
Zone: D → C → B → A
```

---

# 🧠 System Architecture

The system follows a **modular computer vision pipeline**:

```
Video Stream
     ↓
Person Detection (YOLOv8)
     ↓
Multi-Object Tracking (Centroid Tracker)
     ↓
Face Detection + Recognition
     ↓
Behavior Classification (CLIP-based)
     ↓
Temporal Behavior Engine
     ↓
Presence & Activity Monitoring
     ↓
Trajectory Tracking
     ↓
Violation Engine
     ↓
Alert Manager
     ↓
Logs / Console Output
```

This modular design allows easy scaling and component upgrades.

---

# 🏗️ Project Folder Structure

```
VIDEO_MONITORING_TOOL/
│
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── camera_config.yaml
│   └── rules_config.yaml
│
├── src/
│   ├── main.py
│   ├── kinetics_labels.txt
│   └── yolov8n.pt
│
├── src/alerts/
│   └── alert_manager.py
│
├── src/analytics/
│   ├── action_recognition.py
│   ├── behavior_classifier.py
│   ├── behavior_engine.py
│   ├── inactivity.py
│   ├── patrol_analyzer.py
│   ├── presence.py
│   ├── state_stabilizer.py
│   ├── trajectory_tracker.py
│   └── violation_engine.py
│
├── src/detection/
│   ├── object_detector.py
│   ├── person_detector.py
│   └── tracker.py
│
├── src/face/
│   ├── face_detector.py
│   ├── face_encoder.py
│   ├── face_recognizer.py
│   │
│   └── models/
│       ├── face_detection_yunet_2023mar.onnx
│       └── lbfmodel.yaml
│
├── src/data/
│   └── enrolled_faces/
│       ├── guard_1/
│       └── guard_2/
│
├── src/video/
│   ├── stream_reader.py
│   └── video_utils.py
│
├── src/utils/
│   ├── logger.py
│   └── timer.py
│
├── src/training/
│   └── train_behavior.py
│
└── src/logs/
    └── system.log
```

---

# 🚀 Installation Guide

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Pratyaksh0404/Video-monitoring-tool.git
cd Video-monitoring-tool
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```
venv\Scripts\activate
```

### Linux / Mac

```
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 4️⃣ Run the System

```
python src/main.py
```

Press **`q`** to exit the application.

---

# 🧠 Behavior Detection Approach

The system uses **CLIP-based zero-shot behavior classification** instead of fragile rule-based methods.

Example prompts used for classification:

* "a security guard standing alert"
* "a security guard sleeping on duty"
* "a security guard using a mobile phone"
* "a security guard sitting idle"
* "a distracted security guard talking to someone"

---

# ⏱️ Temporal Behavior Engine

Behavior detection uses **time-based persistence filtering** to reduce false positives.

| Duration   | Output      |
| ---------- | ----------- |
| < 5 sec    | ANALYZING   |
| 5 – 10 sec | POSSIBLE_X  |
| > 10 sec   | CONFIRMED_X |

Example:

```
CONFIRMED_SLEEPING
CONFIRMED_PHONE_USE
CONFIRMED_DISTRACTED
```

---

# 👤 Face Recognition

Guards are enrolled inside:

```
SRC/data/enrolled_faces/<Guard_Name>/
```

Multiple images per guard improve recognition stability.

Face embeddings are generated using:

```
face_recognition (dlib)
```

Recognition allows the system to associate **alerts with specific guards**.

---

# 📍 Trajectory Tracking

The monitored region is divided into **patrol zones**.

Example layout:

```
A | B
-----
C | D
```

The system tracks guard movement across zones to analyze patrol coverage.

Example:

```
Zone: D → C → B → A
```

Trajectory tracking enables:

* Patrol movement analysis
* Patrol compliance monitoring
* Guard activity validation

---

# 🚨 Alert System

The system generates alerts when violations occur.

Current alerts include:

* Guard Missing
* Guard Sleeping
* Phone Usage
* Guard Distracted
* Guard Idle

Alerts are currently:

* Printed to console
* Logged in `logs/system.log`

Future versions will support **SMS / email alerts**.

---

# ⚡ Performance Notes

* Designed to run on **CPU systems**
* Detection frequency optimized to reduce lag
* Identity memory prevents frequent UNKNOWN switching
* Startup stabilization prevents early false positives

GPU acceleration will significantly improve performance.

---

# 🛠️ Future Enhancements

Planned improvements include:

### System Improvements

* Patrol trajectory stabilization
* Structured alert logging
* Performance optimization

### Product Features

* Multi-camera support
* Web-based monitoring dashboard
* Alert notification system
* Guard patrol analytics
* Patrol compliance scoring

### Deployment

* Docker deployment
* Cloud monitoring pipeline
* Edge device optimization

---

# 📊 Technology Stack

| Component               | Technology              |
| ----------------------- | ----------------------- |
| Language                | Python                  |
| Computer Vision         | OpenCV                  |
| Detection Model         | YOLOv8 (Ultralytics)    |
| Behavior Classification | OpenCLIP                |
| Deep Learning           | PyTorch                 |
| Face Recognition        | dlib / face_recognition |
| Tracking                | Centroid Tracker        |
| Data Processing         | NumPy                   |

---

# 🧪 Development Status
## Current stage:

**Phase-1 Prototype**

Capabilities implemented:

* Guard presence monitoring
* Activity detection
* Sleeping detection
* Phone usage detection
* Distraction detection 
* Face recognition 
* Guard identity persistence 
* Patrol trajectory tracking 
* Alert generation

The system is **stable for prototype demonstrations and product validation**.

---

# ⚠️ Disclaimer

This system is intended only for **authorized security monitoring applications**.
Use of this system must comply with applicable privacy and surveillance regulations.

---
## 👤 Author

### Pratyaksh Agrawal