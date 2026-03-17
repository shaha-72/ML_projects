# 🧠 People Counter using YOLO

## 📌 Overview
This project is a real-time people counting system built using YOLO and OpenCV.  
It processes RTSP camera streams to detect people, track movement, and count entries/exits based on direction.

---

## 🚀 Features
- 📡 Real-time RTSP stream processing
- 🧍 Person detection using YOLO (Ultralytics)
- ➖ Line crossing detection
- 🖼 First frame capture per camera
- 🔄 Direction-based counting system
- 🗃 Database integration for event logging
- ⚠️ Alert zone detection (e.g., missing staff area)

---

## 🧠 Custom Logic Implemented

### 🎯 Direction Detection (Core Logic)
- `0° – 180°` → Entry  
- `180° – 360°` → Exit  

This logic determines movement direction using object tracking and angle calculation.

---

### 📦 Bounding Box Fix
- Resolved misalignment due to screen resolution differences  
- Ensured consistent detection box placement across all devices  

---

### 📊 Event Logging
- Automatically pushes entry/exit data to database  
- Includes timestamp, camera IP, and use case  

---

## 🛠 Tech Stack
- Python
- OpenCV
- YOLO (Ultralytics)
- NumPy
- RTSP Streaming

---

## 📂 Project Structure
project/
│── main.py
│── config/  
│ └── usecase.json
│── models/
│── utils/
│── output/
│── requirements.txt

---

## ⚙️ Configuration

The system uses a JSON configuration file:

### `usecase.json`
- Camera IP
- Use case type
- Enable/Disable flag
- Camera name
- Detection zone coordinates

---

## ▶️ How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
