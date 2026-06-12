# NoviSentra — AI-Powered Video Analytics Platform

> **Real-time intelligent CCTV analytics. Detect, analyze, and alert automatically.**

NoviSentra transforms any standard CCTV feed into a smart analytics platform using computer vision and deep learning. Originally built for security guard compliance monitoring, the system is architected to support any video analytics use case — retail, warehousing, banking, manufacturing, and more without rewriting the core.

---

## What It Does

NoviSentra runs continuously on live or recorded video and detects behavioral events in real time:

- **Person presence and identity** — face recognition against enrolled persons, unknown person detection with stable tracking labels
- **Behavioral violations** — sleeping, phone usage, smoking, idleness (CLIP-based zero-shot classification)
- **Patrol and movement tracking** — zone-based trajectory analysis (A/B/C/D grid), patrol compliance logging
- **Threat detection** — weapon detection (gun, knife), fire and smoke detection
- **Scene integrity** — camera tamper detection (blackout, obstruction, scene shift)
- **Crowd monitoring** — crowd threshold alerts, post-crowd grace periods
- **Fight and violence detection** — proximity + motion magnitude analysis

All events generate real-time alerts via:
- **Email** — per-alert with 3-burst snapshot attachments, HTML shift report on session end
- **WhatsApp** — batched HIGH-severity alerts (configurable batch size and timeout)
- **Dashboard** — live SSE alert feed, analytics charts, snapshot grid, settings panel

---

## Live Dashboard

The web dashboard (`http://localhost:5000`) provides four tabs:

| Tab | What it shows |
|---|---|
| **Monitor** | Live MJPEG feed with detection overlays, real-time alert feed, model status pills |
| **Analytics** | Guard performance table, alert timeline chart, compliance scores, occupancy trend, snapshot grid |
| **System Log** | Live tail of `system.log` with severity colouring |
| **Settings** | Configurable alert thresholds and cooldowns — sliders, number inputs, toggles — saved live to `rules_config.yaml` without restart |

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web framework | Flask (MJPEG + SSE streaming) |
| Person detection | YOLOv8n (Ultralytics) |
| Weapon detection | YOLOv8n fine-tuned (`Subh775/Threat-Detection-YOLOv8n`) |
| Fire/smoke detection | YOLOv8 local model |
| Behavior classification | OpenCLIP (CLIP zero-shot, CPU) |
| Face recognition | dlib / face_recognition |
| Object tracking | Centroid Tracker (custom) |
| Deep learning runtime | PyTorch |
| Image processing | OpenCV, NumPy |
| Config | PyYAML + ruamel.yaml (comment-preserving writes) |
| Logging | Python logging (session-isolated, session archives) |

---

## Project Structure

```
VIDEO_MONITORING_TOOL/
│
├── requirements.txt
├── .gitignore
├── README.md
├── camera_manager.py       ← Multi-camera manager
├── email_alerter.py        ← SMTP email alerts with 3-snapshot attachments
├── PyWhatKit_DB.txt		← Here whatsapp message gets stored which we have to send
├── shift_report.py         ← HTML shift report generator (charts, guard table, alerts)
├── snapshot_manager.py     ← 3-burst snapshot capture, session folders, singleton
├── whatsapp_alerter.py     ← WhatsApp batch alerter via Chrome + pyautogui
├── flask_app.py            ← Flask web server, routes, SSE alert stream
├── main_web.py             ← Main CV pipeline (ALL detection logic lives here)
├── video_streamer.py       ← MJPEG frame pusher + stats dict
├── yolov8n.pt              ← YOLO person detection model
│
├── templates/
│    └── dashboard.html     ← Single-page dashboard (Monitor + Analytics tabs)
│
├── uploads/ (← Here the videos are stored for video analysis)
│
├── model_cache/
│       ├── .locks/
│       ├── models--timm--resnet50_clip.openai
│       ├── xet/
│       ├── yolo_threat
│			├── .cache/
│			├── _tmp
│       	├── Subh775_Threat-Detection-YOLOv8n__weights_best.pt  ← Weapon model (Gun, knife)
│       	├── fire_model.pt     ← Fire/smoke detection model (YOLOv8, classes: fire, smoke)
│			└── weapon_model.pt
|
├── config/
│   ├── camera_config.yaml   ← Multi-camera setup thresholds)
│   └── rules_config.yaml    ← All thresholds, cooldowns, email/WA config
│
├── src/
│   ├── main.py     ← Standalone CLI version (separate, not used in web)
│   ├── kinetics_labels.txt
│   └── yolov8n.pt
│
├── src/alerts/
│   └── alert_manager.py   ← AlertManager class + global alert_queue
│
├── src/analytics/
│   ├── action_recognition.py
│   ├── anomaly_detector.py      ← Weapon + fire detection (YOLOv8, background thread)
│   ├── behavior_classifier.py   ← CLIP-based behavior (sleeping/phone/idle/smoking/distracted)
│   ├── fight_detector.py        ← Proximity + motion magnitude fight detection
│   ├── behavior_engine.py       ← Majority-vote state machine (window=6, confirm=4 readings)
│   ├── inactivity.py            ← InactivityMonitor (centroid movement tracking)
│   ├── loitering_detector.py    ← Zone-time + movement accumulation loitering
│   ├── patrol_analyzer.py
│   ├── pose_analyzer.py
│   ├── presence.py              ← PresenceMonitor (PRESENT/TEMP_ABSENT/ABSENT)
│   ├── state_stabilizer.py
│   ├── trajectory_tracker.py    ← Zone A/B/C/D tracking per track_id
│   ├── unknown_tracker.py       ← Stable Unknown_N labels + per-slot alert suppression
│   ├── camera_tamper.py         ← Blackout / blur / scene-change detection
│   ├── smoking_detector.py      ← to derect a person smoking
│   └── violation_engine.py
│
├── src/detection/
│   ├── object_detector.py       ← To detect mobile phone (YOLOv8, class 67)
│   ├── person_detector.py       ← PersonDetector (YOLOv8, class 0)
│   └── tracker.py               ← CentroidTracker (max_disappeared=30, max_distance=300)
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
├── src/data/   ← Reference face images:
│   └── enrolled_faces/
│       ├── guard_1/
│       └── guard_2/
│
├── src/snapshots/
│   └── session_YYYYMMDD_HHMMSS/     ← Per-session burst snapshot folders
│
├── src/video/
│   ├── stream_reader.py
│   └── video_utils.py
│
├── src/utils/
│   ├── logger.py     ← Rotating file logger
│   └── timer.py      ← Pipeline latency profiler
│
├── src/training/
│   └── train_behavior.py  ← behaviour model training file for future use (if needed)
│
├── src/logs/
│       ├── system.log               ← Live current session log
│       └── sessions/
│           └── session_YYYY-MM-DD_HH-MM-SS.log   ← Archived on exit
```

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- Windows, Linux, or macOS
- Webcam or RTSP/file video source
- For WhatsApp alerts: Chrome browser + WhatsApp Web logged in

### 1. Clone the repository

```bash
git clone https://github.com/Novtek-Consulting-Ltd/Video-monitoring-tool.git
cd Video-monitoring-tool
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` (face recognition) requires cmake and a C++ compiler. On Windows install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first. On Linux: `sudo apt install cmake libboost-all-dev`.

### 4. Enroll faces

Place reference images for each person under `src/data/enrolled_faces/<Name>/`. Multiple images per person improve recognition stability. Supported formats: `.jpg`, `.png`.

```
src/data/enrolled_faces/
├── guard_2/
│   ├── img1.jpg
│   └── img2.jpg
└── guard_1/
    └── img1.jpg
```

### 5. Configure

Edit `config/rules_config.yaml` for thresholds, cooldowns, email credentials, and WhatsApp settings. All comments in the file are preserved on every save.

For email alerts, add your Gmail App Password (not your real password) and recipient addresses:

```yaml
email:
  enabled: true
  sender: "your@gmail.com"
  password: "your_app_password"   # Gmail App Password, not the account password
  recipients: "recipient@example.com"
```

For WhatsApp, set your recipient number in E.164 format:

```yaml
whatsapp:
  enabled: true
  transport: chrome
  recipient_number: "+919876543210"
```

### 6. Run

```bash
python flask_app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## Configuration Reference

All settings live in `config/rules_config.yaml`. The Settings tab in the dashboard edits these live — changes apply to the running pipeline without restart.

### Key sections

| Section | What it controls |
|---|---|
| `person` | Minimum person bounding box size, face recognition tolerance |
| `behavior.clip_thresholds` | CLIP confidence per behaviour class (sleeping, phone, smoking, idle) |
| `behavior.confirm_ratios` | Fraction of recent readings that must agree before alert fires |
| `alerts` | Cooldowns for missing, idle, and general alerts |
| `fire` | Confidence threshold, sustain duration, consecutive-hit requirement |
| `weapon` | Confidence threshold |
| `crowd` | Person count threshold, post-crowd grace period |
| `loitering` | Zone dwell time for known and unknown persons |
| `email` | SMTP config, recipient list, per-type enable/disable, cooldown |
| `whatsapp` | Transport, recipient, batch size and timeout, per-type enable/disable |

---

## Detection Capabilities

### Behavioral Detection (CLIP-based)

The system uses **zero-shot CLIP classification** — no labelled training data required. Each person crop is classified against text prompts every N frames (configurable `interval_frames`).

A two-stage confirmation pipeline prevents false positives:

| Stage | Condition | Output |
|---|---|---|
| Raw classification | CLIP score ≥ threshold | Candidate label |
| Temporal confirmation | ≥ confirm_ratio of recent readings agree | `CONFIRMED_X` state |
| Alert fires | `CONFIRMED_X` + cooldown elapsed | Alert sent |

Default thresholds (adjustable in Settings tab or `rules_config.yaml`):

| Behaviour | CLIP threshold | Confirm ratio |
|---|---|---|
| Sleeping | 0.75 | 0.95 (very strict — 9 of 10 readings) |
| Phone use | 0.50 | 0.60 |
| Smoking | 0.70 | 0.85 |
| Idle | 0.50 | 0.60 |

### Patrol and Zone Tracking

The monitored area is divided into a 2×2 zone grid:

```
A | B
─────
C | D
```

The system tracks each identified person's zone transitions with a stability filter (2.5 s dwell before zone change is confirmed). Patrol paths are logged and displayed:

```
PATROL Guard_Name — C -> D -> A -> B
```

### Weapon Detection

Two parallel scans run every N frames:

- **Full-frame scan** (every 20 frames) — detects weapons not associated with any tracked person
- **Per-person crop scan** (every 12 frames, staggered by `track_id`) — associates weapon with specific person

Both paths use a **2-second result freshness gate** — only results from the last 2 seconds generate alerts and snapshots, preventing stale detections from firing when the weapon is no longer in frame.

### Face Recognition

Persons are enrolled by placing reference images in `src/data/enrolled_faces/<Name>/`. Recognition runs every 8 frames using two matching strategies:

1. **IoU overlap** between face bounding box and person bounding box
2. **Face centroid containment** (fallback for stale face detections)

Unknown persons receive stable `Unknown_N` labels for the duration of their track.

---

## Alerts

### Per-Alert Emails

For every HIGH/MEDIUM severity event, an email is sent with:
- Alert type, person identity, zone, and timestamp in the subject
- 3-burst snapshots attached (frame at t=0, t+1s, t+2s)
- Configurable per-type enable/disable and cooldown

### WhatsApp Alerts

HIGH-severity alerts are batched and sent via WhatsApp Web automation:
- Configurable batch size (default 5) and timeout (default 300s)
- Fires when batch is full or timeout elapses, whichever comes first

### Shift Report

On session end (Ctrl+C or manual trigger from dashboard), an HTML shift report is generated and emailed containing:
- Summary statistics (total alerts, guard count, session duration)
- Alert timeline chart
- Per-guard compliance scores
- Full alert log table

---

## Session Logging

NoviSentra uses a two-file logging strategy:

```
src/logs/
├── system.log                           ← Live current session (always active)
└── sessions/
    ├── session_2026-06-11_09-00-00.log  ← Archived on program exit
    └── session_2026-06-12_08-57-53.log
```

**How it works:**

- `system.log` accumulates all log lines for the entire program run, including across source-switches
- On Ctrl+C or shutdown, `system.log` is copied to `sessions/session_YYYY-MM-DD_HH-MM-SS.log` (timestamp = session start time) and then truncated
- One archive file per program run — source-switches do not create new log files
- Session log archiving happens before the shift report email is sent, so the log is always preserved regardless of email outcome

Log format:
```
[2026-06-12 09:00:12] WARNING  gms.alert_manager  [ALERT] Guard Sleeping | Guard_Name | Zone C | HIGH
[2026-06-12 09:01:17] WARNING  gms.alert_manager  [ALERT] Weapon Detected: Gun | Guard_Name | Zone — | HIGH
```

---

## Snapshot System

For each alert, three burst snapshots are saved asynchronously:

| Frame | Timing |
|---|---|
| Frame 1 | Immediately at alert time (t=0) |
| Frame 2 | t + 1 second (live frame) |
| Frame 3 | t + 2 seconds (live frame) |

Each snapshot has a red banner overlay showing the alert type, person identity, and zone, plus a timestamp and burst indicator `[1/3]`, `[2/3]`, `[3/3]`.

Snapshots are organised by session:

```
src/snapshots/
└── session_20260612_085753/
    ├── 090012_guard_sleeping_1.jpg
    ├── 090012_guard_sleeping_2.jpg
    ├── 090012_guard_sleeping_3.jpg
    ├── 090117_weapon_detected_gun_1.jpg
    └── ...
```

Filename format: `HHMMSS_alert_type_burst_number.jpg`

---

## Settings Tab

The dashboard Settings tab allows live configuration without editing YAML or restarting:

- **Behavior (CLIP)** — sliders for sleeping/phone/smoking/idle CLIP thresholds and confirm ratios, behavior check interval
- **Alert Cooldowns** — number inputs for general, missing, idle, and email cooldowns
- **Detection Thresholds** — weapon confidence, fire confidence/sustain/consecutive hits, crowd threshold, face tolerance, unknown grace period
- **Alert Type Toggles** — enable/disable email and WhatsApp per alert type (e.g. disable WhatsApp for patrol events)
- **Advanced** — loitering timers, tracker max_disappeared and max_distance, post-crowd grace

Changes are deep-merged into `rules_config.yaml` using `ruamel.yaml` (preserving all comments and formatting) and hot-patched into the running pipeline on the next frame — no restart required.

---

## Multi-Camera Support

Configure multiple sources in `config/camera_config.yaml`. The dashboard camera selector switches between sources without restarting the pipeline — each source-switch sends an email shift report for the ending segment, and the pipeline restarts on the new source.

---

## Performance Notes

- Designed to run on **CPU** — no GPU required
- Face recognition: ~500–1200ms per run (every 8 frames)
- Person detection: ~90–300ms per frame
- Behavior (CLIP): runs in background thread every N frames (default 50), non-blocking
- Weapon and fire detection: background thread, staggered per track_id
- All snapshot saves are asynchronous — never block the pipeline

GPU acceleration (CUDA) will significantly improve throughput, especially for YOLOv8 inference.

---

## Alert Types Reference

| Alert | Severity | Trigger |
|---|---|---|
| Guard Missing | HIGH | Post absent beyond threshold |
| Guard Sleeping | HIGH | CLIP confirms sleeping state |
| Phone Usage | HIGH | CLIP confirms phone use |
| Guard Smoking | MEDIUM | CLIP confirms smoking |
| Guard Idle | MEDIUM | No significant movement |
| Weapon Detected | HIGH | YOLOv8 weapon model (gun/knife) |
| Unattended Weapon | HIGH | Weapon detected, no person associated |
| Fire Detected | HIGH | YOLOv8 fire model (sustain + consecutive hits) |
| Camera Tamper | HIGH | Blackout / obstruction / scene shift |
| Crowd Detected | MEDIUM | Person count ≥ threshold |
| Unknown Person Detected | HIGH | Unrecognised person past grace period |
| Fight / Violence | HIGH | Proximity + motion magnitude threshold |
| Guard Under Attack | HIGH | Known person in fight pair |
| Patrol | LOW | Zone transition logged |

---

## Disclaimer

NoviSentra is intended only for **authorised monitoring applications**. Use must comply with applicable privacy, surveillance, and data protection regulations in your jurisdiction.

---

## Author

**Pratyaksh Agrawal**