"""
people_counter.py  ── Main Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes applied:
  ✔ Webcam warm-up frames  → prevents instant close
  ✔ Frame drop mode        → reads latest frame always (no latency)
  ✔ Retry on failed grab   → doesn't exit on single bad frame
  ✔ YOLO verbose=False     → no terminal spam / number output
  ✔ Stream recovery loop   → auto reconnect if webcam disconnects
Visual changes:
  ✔ Thicker bounding box border (3px)
  ✔ Label shows "Person" only — no ID/tracking info
  ✔ No centroid/tracking text in putText
  ✔ "Person" label top-left above box — white text on green bg
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import cv2
import json
import logging
import signal
import sys
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

from ultralytics import YOLO

from linecrosscheck import checkIntersect, calcVectorAngle
from id_stabilizer  import IDStabilizer
from db_manager     import DBManager


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PeopleCounter")


# ─────────────────────────────────────────────
#  Config loader
# ─────────────────────────────────────────────
def load_config(path: str = "config.json") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(cfg_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
#  Camera open + warm-up
# ─────────────────────────────────────────────
def open_camera(cfg: dict):
    cam_cfg = cfg["camera"]
    source  = cam_cfg.get("source", "webcam").lower()

    if source == "rtsp":
        url = cam_cfg["rtsp_url"]
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(f"Opening RTSP: {url}")
    else:
        idx = int(cam_cfg.get("webcam_index", 0))
        # Use plain VideoCapture — CAP_DSHOW can cause black frames on some Windows drivers
        cap = cv2.VideoCapture(idx)
        logger.info(f"Opening webcam index: {idx}")

    if not cap.isOpened():
        logger.error("Cannot open camera. Check config.json → camera settings.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Give the webcam time to initialise before reading
    time.sleep(1)

    logger.info("Warming up camera …")
    for _ in range(30):   # increased from 10 → 30 for slow Windows webcams
        cap.read()

    logger.info("Camera ready.")
    return cap


# ─────────────────────────────────────────────
#  Grab latest frame (flush buffer → no latency)
# ─────────────────────────────────────────────
def grab_latest_frame(cap):
    for _ in range(1):   # reduced from 3 → 1, prevents black frames on slow webcams
        cap.grab()
    return cap.retrieve()


# ─────────────────────────────────────────────
#  Draw "Person" label — white text on green bg
#  top-left corner, just above the bounding box
# ─────────────────────────────────────────────
def draw_person_label(frame, x1, y1):
    label      = "Person"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2
    padding    = 4

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Background rectangle sits above the top-left corner of the box
    bg_x1 = x1
    bg_y1 = y1 - th - baseline - padding * 2
    bg_x2 = x1 + tw + padding * 2
    bg_y2 = y1

    # Clamp so it doesn't go off-screen at the top
    if bg_y1 < 0:
        bg_y1 = 0
        bg_y2 = th + baseline + padding * 2

    # Green background
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 200, 0), -1)

    # White text
    cv2.putText(
        frame, label,
        (bg_x1 + padding, bg_y2 - baseline - padding),
        font, font_scale, (255, 255, 255), thickness,
    )


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    cfg = load_config("config.json")

    det_cfg  = cfg["detection"]
    stab_cfg = cfg["stabilization"]
    disp_cfg = cfg["display"]
    db_cfg   = cfg["database"]
    cam_cfg  = cfg["camera"]

    # ── Model ──────────────────────────────────
    logger.info("Loading YOLO model …")
    model = YOLO(det_cfg.get("model", "yolov8n.pt"))
    logger.info("Model ready.")

    # ── DB ─────────────────────────────────────
    db = DBManager(db_cfg, camera_id="cam_main")
    if not db.connect():
        logger.warning("DB unavailable — counts will NOT be saved.")

    people_in, people_out = db.load_today()

    db.start_sync(
        get_counts_fn=lambda: (people_in, people_out),
        interval_sec=int(db_cfg.get("sync_interval_sec", 5)),
    )

    # ── ID Stabilizer ──────────────────────────
    stabilizer = IDStabilizer(stab_cfg)

    # ── Track history ──────────────────────────
    track_history: dict[int, list] = {}

    # ── Camera ─────────────────────────────────
    cap = open_camera(cfg)
    # ── Fullscreen window ─────────────────────
    window_name = disp_cfg.get("window_title", "People Counter")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    reconnect_attempts = int(cam_cfg.get("reconnect_attempts", 5))
    reconnect_delay    = int(cam_cfg.get("reconnect_delay_sec", 3))

    # ── Graceful shutdown ──────────────────────
    def shutdown(sig=None, frame=None):
        logger.info("Saving final counts and shutting down …")
        db.save(people_in, people_out)
        db.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Final — IN: {people_in}  OUT: {people_out}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    conf_thr    = float(det_cfg.get("confidence_threshold", 0.5))
    min_area    = int(det_cfg.get("min_box_area",           1500))
    hist_len    = int(det_cfg.get("track_history_len",      30))
    min_frames  = int(det_cfg.get("min_track_frames",       5))
    tracker_cfg = det_cfg.get("tracker", "bytetrack.yaml")

    fail_count = 0
    MAX_FAILS  = 20

    logger.info("Running … Press Q in the window to quit.")

    while True:
        # ── Grab latest frame ───────────────────
        ret, frame = grab_latest_frame(cap)

        # ── Handle bad frame ────────────────────
        if not ret or frame is None:
            fail_count += 1
            logger.warning(f"Bad frame ({fail_count}/{MAX_FAILS}) — retrying …")

            if fail_count >= MAX_FAILS:
                logger.error("Too many failed frames. Reconnecting …")
                cap.release()
                time.sleep(reconnect_delay)
                cap = open_camera(cfg)
                fail_count = 0

            time.sleep(0.05)
            continue

        fail_count = 0
        annotated  = frame.copy()
        h, w, _    = frame.shape

        # ── Crossing line (horizontal center) ───
        line_p1 = (0,     h // 2)
        line_p2 = (w - 1, h // 2)
        cv2.line(annotated, line_p1, line_p2, (0, 0, 255), 3)

        # ── YOLO Tracking ───────────────────────
        results = model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            classes=[0],
            verbose=False,
            imgsz=640,
        )

        raw_dets = []

        if results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                raw_id = int(box.id[0])
                conf   = float(box.conf[0])

                if conf < conf_thr:
                    continue
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                raw_dets.append({
                    "raw_id":   raw_id,
                    "box":      (x1, y1, x2, y2),
                    "centroid": (cx, cy),
                    "conf":     conf,
                })

        # ── ID Stabilization ────────────────────
        stable_dets = stabilizer.process(raw_dets)

        for det in stable_dets:
            x1, y1, x2, y2 = det["box"]
            sid             = det["stable_id"]
            centroid        = det["centroid"]

            # ── 1. Thicker bounding box (3px) ───────────────────────────
            if disp_cfg.get("show_bbox", True):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # ── 2 & 3 & 4. "Person" label — white on green, top-left ────
            #    No ID number, no centroid coords, no tracking clutter
            if disp_cfg.get("show_ids", True):
                draw_person_label(annotated, x1, y1)

            # ── Track history (counting logic — untouched) ───────────────
            if sid not in track_history:
                track_history[sid] = []

            track_history[sid].append(centroid)

            if len(track_history[sid]) > hist_len:
                track_history[sid] = track_history[sid][-hist_len:]

            # Trail drawing removed from frame (logic kept for line-crossing)

            if len(track_history[sid]) < min_frames:
                continue

            prev_pt = track_history[sid][-2]
            curr_pt = track_history[sid][-1]

            # ── Line crossing ───────────────────
            if checkIntersect(prev_pt, curr_pt, line_p1, line_p2):
                if stabilizer.can_count(sid):
                    angle = calcVectorAngle(prev_pt, curr_pt, line_p1, line_p2)
                    if angle < 180:
                        people_in  += 1
                        logger.info(f"ID {sid} IN  → total IN={people_in}")
                    else:
                        people_out += 1
                        logger.info(f"ID {sid} OUT → total OUT={people_out}")
                    stabilizer.mark_crossed(sid)

        # ── Counter panel — side by side, no background ─────────────
        cv2.putText(annotated, f"IN: {people_in}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(annotated, f"OUT: {people_out}",
                    (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 80, 255), 3)

        # ── Show ────────────────────────────────
        cv2.imshow(window_name, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    shutdown()


if __name__ == "__main__":
    main()
