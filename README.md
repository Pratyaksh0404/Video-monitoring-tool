# NoviSentra — AI Video Surveillance Intelligence Platform

> Real-time threat detection, behavioral analysis, and automated alerting for existing CCTV infrastructure.

NoviSentra connects to your IP cameras or DVR/NVR hardware via RTSP, runs multiple AI models simultaneously on every frame, and delivers alerts with photo evidence to your team via email and WhatsApp — all on your own hardware, with no video sent to any external service.

---

## What It Detects

| Category | Alert Types |
|---|---|
| 🔫 **Weapon Detection** | Gun / Knife — attended and unattended |
| 🔥 **Fire & Smoke** | Fire Detected (sustained confirmation) |
| 😴 **Behavior** | Guard Sleeping, Phone Usage, Smoking, Idle |
| 👥 **Crowd** | Crowd Detected (configurable person threshold) |
| ⚡ **Violence** | Fight / Violence Detected, Guard Under Attack |
| 🚶 **Presence** | Guard / Staff / Worker Missing |
| 📷 **Camera Tamper** | Blackout, Obstruction, Freeze |
| 🗺 **Zone Tracking** | Patrol compliance, Movement alerts |
| 👤 **Face Recognition** | Known person identification + Unknown Person alert |

All detectors run **simultaneously** on every camera.

---

## Deployment Profiles

Switch profiles live from the dashboard — no restart required.

| Profile | Best For | Key Detections |
|---|---|---|
| `guard_monitoring` | Security guard posts | Sleeping, phone, smoking, idle, patrol compliance, missing, fight |
| `bank_security` | Bank branches | Weapon (priority), movement anomaly, face recognition, camera tamper |
| `retail_analytics` | Retail stores | Loitering, dwell analytics, crowd, idle detection, weapon, fire |
| `warehouse_ops` | Warehouses / factories | Worker sleeping, missing, phone, fire, crowd, zone movement |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Windows 10+ or Ubuntu 22.04+
- 8 GB RAM minimum (for multi-camera operation)
- IP cameras / DVR accessible via RTSP on the local network

### Installation

```bash
git clone https://github.com/Novtek-Consulting-Ltd/Video-monitoring-tool.git
cd Video-monitoring-tool
pip install -r requirements.txt
```

> **Note on dlib:** face_recognition requires dlib. Prebuilt wheels are available for most platforms via pip. If compilation is needed: `pip install cmake` first.

### Configuration

**1. Set credentials as environment variables (never in config files):**

```bash
# Email alerts
export SMTP_USER="your@email.com"
export SMTP_PASS="your-smtp-password"
export ALERT_RECIPIENT="recipient@email.com"

# Dashboard login
export NOVISENTRA_ADMIN_USER="admin"
export NOVISENTRA_ADMIN_PASS="your-secure-password"

# Camera credentials (if your DVR requires auth)
export CAM_3_USER="admin"
export CAM_3_PASS="your-dvr-password"
```

**2. Configure cameras in `config/camera_config.yaml`:**

```yaml
cameras:
  cam_1:
    name: "Main Entrance"
    source: 0                    
    enabled: true
    profile: guard_monitoring
  cam_2:
    name: "USB Camera"
    source: 1              
    enabled: false
    profile: guard_monitoring
  cam_3:
    name: "CAM01 — Lobby"
    source: "rtsp://IP_address:554/user=${CAM_3_USER}&password=${CAM_3_PASS}&channel=1&stream=0.sdp"
    enabled: true
    profile: guard_monitoring

  cam_4:
    name: "CAM02 — Corridor"
    source: "rtsp://IP_address:554/user=${CAM_3_USER}&password=${CAM_3_PASS}&channel=2&stream=0.sdp"
    enabled: true
    profile: guard_monitoring

  cam_5:
    name: "CAM03 — Exit"
    source: "rtsp://IP_address9:554/user=${CAM_3_USER}&password=${CAM_3_PASS}&channel=3&stream=0.sdp"
    enabled: true
    profile: guard_monitoring
```

**RTSP URL format for XMEye/TSEYE/Coreprix/Xiongmai-family DVRs:**
```
rtsp://<DVR_IP>:554/user=<username>&password=<password>&channel=<N>&stream=0.sdp
```
Where `channel=1` is CAM01, `channel=2` is CAM02, etc. `stream=0` = main (high-res), `stream=1` = sub-stream.

**3. Enroll faces for face recognition:**

```
src/data/enrolled_faces/
├── Mukesh/
│   ├── photo1.jpg
│   └── photo2.jpg
├── Guard_1/
│   └── photo1.jpg
└── ...
```

Add one or more photos per person. System auto-encodes on first run. Cache is stored at `enrolled_faces/.encoding_cache.pkl` and auto-invalidates when photos change.

### Run

```bash
python flask_app.py
```

**Startup sequence:**
```
[Startup] Pre-loading CLIP model...
[Startup] CLIP ready.
[Startup] Pre-loading face encodings...
[Startup] Face encodings ready.
[CameraManager] cam_1 (Main Entrance) — started.
[CameraManager] cam_3 (CAM01 — Lobby) — started.
...
  NoviSentra → http://127.0.0.1:5000/login
```

Open your browser to `http://127.0.0.1:5000` (or your server's IP on port 5000).

---

## Dashboard

The web dashboard has five sections:

| Tab | Purpose |
|---|---|
| **Monitor** | Live camera grid, real-time alert log, per-camera statistics |
| **Analytics** | Zone dwell time, person count trends, patrol compliance charts |
| **System Log** | Technical process logs, searchable and filterable |
| **Settings** | Live threshold tuning — changes apply immediately without restart |
| **Admin Panel** | Camera management, user management, profile switching |

### Alert Log

Every detected event appears in the live alert log with:
- Severity color (🔴 HIGH / 🟠 MEDIUM / ⚪ LOW)
- Timestamp, alert type, person name, zone, camera ID
- Expandable snapshot thumbnails (3 photos per HIGH/MEDIUM alert)

---

## Alert Delivery

### Email
- Sends for all HIGH and MEDIUM severity alerts
- 3 full-resolution snapshot images attached per alert
- Per-type cooldown (default 200s) prevents flooding
- Unknown Person alerts suppressed by default

### WhatsApp (via Twilio)
- Sends for HIGH severity alerts only
- Configure Twilio credentials in environment variables

### Shift Reports
- Auto-sent on profile switch and system shutdown
- Summarizes all alerts from the outgoing session
- HTML email with alert counts by type

---

## Performance Tuning (Multi-Camera)

Adjust in `config/rules_config.yaml`:

```yaml
pipeline:
  max_width: 640      # Downscale frames to 640px before AI processing (~9x CPU reduction)
  skip_frames: 2      # Process every 2nd frame (~2x CPU reduction)
```

Recommended settings:

| Setup | `max_width` | `skip_frames` |
|---|---|---|
| Laptop, 2-4 cameras | 640 | 2 |
| Dedicated server (8-core) | 0 (disabled) | 1 |
| Single camera, laptop | 640 | 1 |

> Snapshots and video feed always use the original full-resolution frame regardless of these settings.

---
## Configuration Reference

### `config/rules_config.yaml` (key sections)

```yaml
person:
  min_height: 60          # Minimum pixel height for a person detection to be tracked
  min_width: 30           # Minimum pixel width
  face_tolerance: 0.5     # Face recognition match distance (lower = stricter)

behavior:
  interval_frames: 50     # Run CLIP every N frames per tracked person
  idle_max_speed: 12.0    # Max px/s centroid movement to classify as idle
  clip_thresholds:        # Per-label CLIP minimum confidence
    SLEEPING: 0.85
    PHONE_USE: 0.50
    SMOKING: 0.70
  confirm_ratios:         # Fraction of rolling window that must agree
    SLEEPING: 0.95
    SMOKING: 0.85

fire:
  frame_interval: 20      # Run fire detection every N frames
  min_confidence: 0.55
  sustain_seconds: 6.0    # Must persist this long before alerting
  min_consecutive: 3      # Minimum consecutive positive detections

alerts:
  cooldown: 30            # Seconds between same-type alerts per person
  missing_cooldown: 30
  idle_cooldown: 60
  idle_startup_grace: 90  # Suppress idle alerts for N seconds after startup

pipeline:
  max_width: 640          # Downscale to this width before AI processing (0 = disabled)
  skip_frames: 2          # Process every Nth frame
```

---

## Verification

Run after any code change:

```bash
python check_imports.py
```

All 23+ modules must import cleanly before deployment.

---

## Known Limitations

- **USB webcam + concurrent RTSP**: opening multiple USB cameras simultaneously on Windows can cause a GIL-related freeze. Use RTSP/DVR for multi-camera deployments (recommended).
- **GPU acceleration**: not currently implemented — all inference runs on CPU. Performance scales with CPU cores.
- **Batch processing**: recorded video analysis is not supported in v1.0.


---

## License

Proprietary — All Rights Reserved. See LICENSE for terms.

---

## Author

**Pratyaksh Agrawal**