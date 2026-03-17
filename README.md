# 🧠 People Counter using YOLO

## 📌 Overview
This project performs real-time people detection and counting using RTSP camera streams.  
It also includes custom logic for direction-based counting and event tracking.

---

## 🚀 Features
- Real-time RTSP stream processing
- Person detection using YOLO (Ultralytics)
- Line crossing detection
- First frame capture per camera
- Direction-based counting (0–180 / 180–360 angle logic)
- Database update for entry/exit events

---

## 🧠 Custom Logic Implemented
- Angle-based movement detection:
  - `0°–180°` → Entry
  - `180°–360°` → Exit
- Fixed issues in detection box alignment across different screen resolutions
- Improved bounding box positioning for accurate alert zones (e.g., missing staff detection)

---

## 🛠 Tech Stack
- Python
- OpenCV
- YOLO (Ultralytics)
- RTSP Streaming

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python main.py
