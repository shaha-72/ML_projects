"""
id_stabilizer.py
Fixes all major tracking instability issues:
  1. ID Flickering   — centroid moves slightly, YOLO reassigns new ID.
  2. ID Switching    — two nearby tracks swap IDs.
  3. ID Overlap      — two bounding boxes overlap → one track disappears then reappears.
  4. Re-entry ghost  — person walks off screen, comes back, gets new ID.

Strategy:
  - IoU-based overlap suppression (merge overlapping boxes to dominant ID).
  - Centroid proximity merge (snap close centroids to single track).
  - Crossing cooldown per ID (prevent double-counts).
  - Ghost buffer (remember lost IDs briefly → re-assign if new centroid is close).
"""

import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger("IDStabilizer")


def iou(box_a, box_b) -> float:
    """Compute Intersection-over-Union of two boxes [x1,y1,x2,y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid_dist(c1, c2) -> float:
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


class IDStabilizer:
    def __init__(self, cfg: dict):
        iou_thr  = cfg.get("iou_overlap_threshold",    0.6)
        cd_thr   = cfg.get("centroid_merge_distance",  60)
        cooldown = cfg.get("crossing_cooldown_frames", 20)
        ghost_ttl= cfg.get("ghost_ttl_frames",         30)

        self.iou_threshold      = iou_thr
        self.centroid_threshold = cd_thr
        self.cooldown_frames    = cooldown
        self.ghost_ttl          = ghost_ttl

        # frame counter
        self._frame_no: int = 0

        # canonical ID mapping: raw_id → stable_id
        self._id_map: dict[int, int] = {}

        # last known centroid for each stable_id
        self._last_centroid: dict[int, tuple] = {}

        # last seen frame for each stable_id
        self._last_seen: dict[int, int] = {}

        # ghost buffer: stable_id → (centroid, expire_frame)
        self._ghosts: dict[int, tuple] = {}

        # crossing cooldown: stable_id → last_crossed_frame
        self._cross_frame: dict[int, int] = {}

        # next stable id counter
        self._next_stable: int = 1

    # ──────────────────────────────────────────
    #  Public: process one frame of detections
    # ──────────────────────────────────────────
    def process(self, detections: list[dict]) -> list[dict]:
        """
        detections: list of dicts with keys:
            raw_id, box (x1,y1,x2,y2), centroid (cx,cy), conf

        Returns same list but with added key: stable_id
        Also removes duplicate/overlapping boxes.
        """
        self._frame_no += 1
        self._expire_ghosts()

        if not detections:
            return []

        # Step 1: suppress overlapping boxes (keep highest conf)
        detections = self._suppress_overlaps(detections)

        # Step 2: assign stable IDs
        for det in detections:
            det["stable_id"] = self._assign_stable_id(
                det["raw_id"], det["centroid"]
            )
            self._last_centroid[det["stable_id"]] = det["centroid"]
            self._last_seen[det["stable_id"]]     = self._frame_no

        return detections

    # ──────────────────────────────────────────
    #  Crossing gate
    # ──────────────────────────────────────────
    def can_count(self, stable_id: int) -> bool:
        """Returns True only if this ID hasn't crossed recently (anti-flicker)."""
        last = self._cross_frame.get(stable_id, -9999)
        return (self._frame_no - last) > self.cooldown_frames

    def mark_crossed(self, stable_id: int):
        self._cross_frame[stable_id] = self._frame_no

    # ──────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────
    def _suppress_overlaps(self, dets: list[dict]) -> list[dict]:
        """NMS-style: for every pair with IoU > threshold, drop the lower-conf one."""
        keep = [True] * len(dets)
        dets_sorted = sorted(
            enumerate(dets), key=lambda x: x[1]["conf"], reverse=True
        )
        indices = [i for i, _ in dets_sorted]

        for i in range(len(indices)):
            if not keep[indices[i]]:
                continue
            for j in range(i + 1, len(indices)):
                if not keep[indices[j]]:
                    continue
                if iou(dets[indices[i]]["box"], dets[indices[j]]["box"]) > self.iou_threshold:
                    keep[indices[j]] = False
                    # remap suppressed raw_id → surviving raw_id
                    self._id_map[dets[indices[j]]["raw_id"]] = \
                        self._resolve_stable(dets[indices[i]]["raw_id"])

        return [d for i, d in enumerate(dets) if keep[i]]

    def _assign_stable_id(self, raw_id: int, centroid: tuple) -> int:
        # Already mapped?
        if raw_id in self._id_map:
            return self._id_map[raw_id]

        # Check ghosts first (re-entry detection)
        ghost_match = self._match_ghost(centroid)
        if ghost_match is not None:
            logger.debug(f"Re-entry: raw_id={raw_id} → ghost stable_id={ghost_match}")
            self._id_map[raw_id] = ghost_match
            del self._ghosts[ghost_match]
            return ghost_match

        # Check existing stables by centroid proximity (ID switching fix)
        for sid, last_c in self._last_centroid.items():
            if centroid_dist(centroid, last_c) < self.centroid_threshold:
                last_f = self._last_seen.get(sid, 0)
                if (self._frame_no - last_f) <= 5:   # seen very recently
                    self._id_map[raw_id] = sid
                    logger.debug(f"Centroid merge: raw_id={raw_id} → stable_id={sid}")
                    return sid

        # New ID
        stable = self._next_stable
        self._next_stable += 1
        self._id_map[raw_id] = stable
        return stable

    def _resolve_stable(self, raw_id: int) -> int:
        return self._id_map.get(raw_id, raw_id)

    def _match_ghost(self, centroid: tuple):
        best_sid  = None
        best_dist = self.centroid_threshold * 1.5   # slightly wider for re-entry
        for sid, (ghost_c, _) in self._ghosts.items():
            d = centroid_dist(centroid, ghost_c)
            if d < best_dist:
                best_dist = d
                best_sid  = sid
        return best_sid

    def _expire_ghosts(self):
        """Move recently-lost tracks to ghost buffer; expire old ghosts."""
        expired = []
        for sid, (c, exp) in self._ghosts.items():
            if self._frame_no > exp:
                expired.append(sid)
        for sid in expired:
            del self._ghosts[sid]

        # Promote stable IDs not seen in a while → ghost
        for sid, last_f in list(self._last_seen.items()):
            if (self._frame_no - last_f) > 5 and sid not in self._ghosts:
                last_c = self._last_centroid.get(sid)
                if last_c:
                    self._ghosts[sid] = (last_c, self._frame_no + self.ghost_ttl)
                    del self._last_seen[sid]
