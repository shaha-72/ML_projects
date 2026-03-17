"""
linecrosscheck.py
Geometry helpers for line-crossing detection.
"""

import math


def _cross_product(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, q, r):
    """Check if point q lies on segment pr."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def checkIntersect(p1, p2, p3, p4) -> bool:
    """
    Returns True if segment p1→p2 intersects segment p3→p4.
    Uses cross-product orientation method.
    """
    d1 = _cross_product(p3, p4, p1)
    d2 = _cross_product(p3, p4, p2)
    d3 = _cross_product(p1, p2, p3)
    d4 = _cross_product(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if d1 == 0 and _on_segment(p3, p1, p4): return True
    if d2 == 0 and _on_segment(p3, p2, p4): return True
    if d3 == 0 and _on_segment(p1, p3, p2): return True
    if d4 == 0 and _on_segment(p1, p4, p2): return True

    return False


def calcVectorAngle(prev_pt, curr_pt, line_p1, line_p2) -> float:
    """
    Returns the angle (0–360°) of movement vector prev_pt→curr_pt
    relative to the crossing line direction.

    Convention used here:
      - Crossing top-to-bottom  (cy increases) → angle < 180  → IN
      - Crossing bottom-to-top  (cy decreases) → angle >= 180 → OUT

    Works for any line orientation.
    """
    # Movement vector
    mv_x = curr_pt[0] - prev_pt[0]
    mv_y = curr_pt[1] - prev_pt[1]

    # Line direction vector
    lv_x = line_p2[0] - line_p1[0]
    lv_y = line_p2[1] - line_p1[1]

    # Angle of movement vector (screen coords: y down)
    angle_mv   = math.atan2(mv_y,   mv_x)
    angle_line = math.atan2(lv_y, lv_x)

    rel_angle = math.degrees(angle_mv - angle_line) % 360
    return rel_angle
