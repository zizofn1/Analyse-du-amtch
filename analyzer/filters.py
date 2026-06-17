"""Filtres adaptatifs — strict (spider cam) ou relâché (HD / gros plan)."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def nms_detections(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    if not detections:
        return []
    remaining = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if iou(best.bbox, d.bbox) < iou_threshold]
    return kept


def build_pitch_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (28, 30, 30), (92, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def is_on_pitch(mask: np.ndarray, bbox: tuple[float, float, float, float], threshold: float = 0.2) -> bool:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = bbox
    foot_x = int((x1 + x2) / 2)
    foot_y = int(min(y2 - 2, h - 1))
    foot_x = max(0, min(foot_x, w - 1))

    y_start = max(0, foot_y - 10)
    x_start = max(0, foot_x - 10)
    region = mask[y_start : min(h, foot_y + 2), x_start : min(w, foot_x + 11)]
    if region.size == 0:
        return False
    return float(np.mean(region > 0)) >= threshold


def _size_ok(det: Detection, h: int, w: int, relaxed: bool) -> bool:
    x1, y1, x2, y2 = det.bbox
    box_h = y2 - y1
    box_w = x2 - x1
    area = box_w * box_h
    frame_area = h * w
    aspect = box_h / max(box_w, 1)

    min_h = h * (0.012 if relaxed else 0.018)
    max_h = h * (0.45 if relaxed else 0.35)
    min_area = frame_area * (0.00008 if relaxed else 0.00015)
    max_area = frame_area * (0.08 if relaxed else 0.06)

    if box_h < min_h or box_h > max_h:
        return False
    if area < min_area or area > max_area:
        return False
    if aspect < 1.0 or aspect > 6.0:
        return False
    if y1 < h * (0.04 if relaxed else 0.08):
        return False
    return True


def filter_player_detections(
    frame: np.ndarray,
    players: list[Detection],
    pitch_mask: np.ndarray | None = None,
    max_players: int = 23,
    use_pitch_filter: bool = True,
) -> list[Detection]:
    if not players:
        return []

    h, w = frame.shape[:2]
    relaxed = h >= 720
    if pitch_mask is None:
        pitch_mask = build_pitch_mask(frame)

    pass_size: list[Detection] = [d for d in players if _size_ok(d, h, w, relaxed)]

    if not use_pitch_filter:
        filtered = pass_size
    else:
        on_pitch = [d for d in pass_size if is_on_pitch(pitch_mask, d.bbox, 0.2)]
        # Si le filtre terrain élimine trop de monde → mode relâché
        if len(on_pitch) < max(4, len(pass_size) * 0.35):
            filtered = pass_size
        else:
            filtered = on_pitch

    filtered = nms_detections(filtered, iou_threshold=0.35 if relaxed else 0.4)
    filtered.sort(key=lambda d: d.confidence, reverse=True)
    return filtered[:max_players]
