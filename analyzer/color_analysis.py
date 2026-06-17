"""
Analyse couleur professionnelle — espace LAB + KMeans (scikit-learn) + Delta E.
Plus fiable que HSV seul pour distinguer maillots proches.
"""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans


def delta_e_cie76(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    diff = lab_a.astype(np.float32) - lab_b.astype(np.float32)
    return float(np.sqrt(np.sum(diff * diff)))


def _is_grass_bgr(bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    return 30 <= h <= 90 and s >= 30 and v >= 30


def extract_torso_patch(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray | None:
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None

    pad_x = int((x2 - x1) * 0.1)
    torso_y1 = y1 + int((y2 - y1) * 0.12)
    torso_y2 = y1 + int((y2 - y1) * 0.55)
    rx1, rx2 = x1 + pad_x, x2 - pad_x
    if rx2 <= rx1 or torso_y2 <= torso_y1:
        return None
    return frame[torso_y1:torso_y2, rx1:rx2]


def extract_jersey_lab_pixels(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    patch = extract_torso_patch(frame, bbox)
    if patch is None or patch.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    kept: list[np.ndarray] = []
    for bgr in patch.reshape(-1, 3):
        if _is_grass_bgr(bgr):
            continue
        kept.append(bgr)

    if not kept:
        return np.empty((0, 3), dtype=np.float32)

    filtered = cv2.cvtColor(np.array(kept, dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    pixels = filtered.reshape(-1, 3).astype(np.float32)
    mask = (pixels[:, 0] > 35) & (np.std(pixels, axis=1) > 3)
    return pixels[mask] if np.any(mask) else pixels


def dominant_lab_color(frame: np.ndarray, bbox: tuple[float, float, float, float], n_clusters: int = 3) -> np.ndarray | None:
    pixels = extract_jersey_lab_pixels(frame, bbox)
    if len(pixels) < 12:
        return None

    k = min(n_clusters, max(1, len(pixels) // 15))
    if k == 1:
        return pixels.mean(axis=0)

    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(pixels)
    sizes = [(labels == i).sum() for i in range(k)]
    best = int(np.argmax(sizes))
    return km.cluster_centers_[best].astype(np.float32)


def lab_to_bgr(lab: np.ndarray) -> tuple[int, int, int]:
    patch = np.uint8([[lab.astype(np.uint8)]])
    bgr = cv2.cvtColor(patch, cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def is_referee_lab(lab: np.ndarray) -> bool:
    """Bleu ciel / cyan en LAB : b* négatif, a* proche de 0."""
    l_val, a_val, b_val = float(lab[0]), float(lab[1]), float(lab[2])
    return b_val < 118 and 120 < l_val < 200 and abs(a_val - 128) < 25
