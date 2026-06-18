"""Détection et suivi du ballon : YOLO + couleur + filtre de Kalman (OpenCV)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .filters import build_pitch_mask
from .types import SPORTS_BALL_CLASS, Detection


@dataclass
class BallState:
    center: Optional[tuple[float, float]] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    confidence: float = 0.0
    visible: bool = False
    source: str = "none"


class BallTracker:
    def __init__(self, search_radius: float = 90.0, max_lost_frames: int = 12):
        self.search_radius = search_radius
        self.max_lost_frames = max_lost_frames
        self._kalman = self._create_kalman()
        self._initialized = False
        self._lost_frames = 0
        self._last_state = BallState()
        self._pitch_mask: np.ndarray | None = None
        self._speed_history: list[float] = []

    def _create_kalman(self) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.8
        return kf

    def _predict(self) -> tuple[float, float]:
        if not self._initialized:
            return 0.0, 0.0
        pred = self._kalman.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def _correct(self, x: float, y: float) -> None:
        if not self._initialized:
            self._kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self._kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self._initialized = True
        else:
            self._kalman.correct(np.array([[x], [y]], dtype=np.float32))

    @property
    def speed(self) -> float:
        if len(self._speed_history) < 1:
            return 0.0
        return float(self._speed_history[-1])

    @property
    def avg_speed(self) -> float:
        if not self._speed_history:
            return 0.0
        return float(np.mean(self._speed_history[-3:]))

    def _detect_by_color(
        self,
        frame: np.ndarray,
        hint: Optional[tuple[float, float]],
    ) -> Optional[Detection]:
        h, w = frame.shape[:2]
        if self._pitch_mask is None:
            self._pitch_mask = build_pitch_mask(frame)

        if hint is not None:
            cx, cy = int(hint[0]), int(hint[1])
            r = int(self.search_radius)
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            roi = frame[y1:y2, x1:x2]
            mask_roi = self._pitch_mask[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            roi = frame
            mask_roi = self._pitch_mask
            offset = (0, 0)

        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Ballon blanc / clair
        white = cv2.inRange(hsv, (0, 0, 175), (180, 55, 255))
        # Ballon orange (certaines compétitions)
        orange = cv2.inRange(hsv, (5, 80, 120), (25, 255, 255))
        color_mask = cv2.bitwise_or(white, orange)
        color_mask = cv2.bitwise_and(color_mask, mask_roi)

        blur = cv2.GaussianBlur(color_mask, (5, 5), 0)
        contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(8, (h * w) / 500000)
        max_area = max(400, (h * w) / 8000)
        best: Optional[Detection] = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.45:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw / 2 + offset[0]
            cy = y + bh / 2 + offset[1]
            bx1, by1, bx2, by2 = x + offset[0], y + offset[1], x + bw + offset[0], y + bh + offset[1]

            score = circularity * min(1.0, area / 80.0)
            if hint is not None:
                dist = np.hypot(cx - hint[0], cy - hint[1])
                score *= max(0.1, 1.0 - dist / (self.search_radius * 1.5))

            if score > best_score:
                best_score = score
                best = Detection(
                    bbox=(bx1, by1, bx2, by2),
                    confidence=0.45 + 0.3 * circularity,
                    class_id=SPORTS_BALL_CLASS,
                    center=(cx, cy),
                )
        return best

    def update(self, frame: np.ndarray, yolo_ball: Optional[Detection]) -> BallState:
        hint = self._predict() if self._initialized else None
        if hint == (0.0, 0.0) and self._last_state.center:
            hint = self._last_state.center

        candidates: list[tuple[float, Detection, str]] = []
        if yolo_ball is not None:
            candidates.append((yolo_ball.confidence + 0.2, yolo_ball, "yolo"))

        # Désactivation de la détection par couleur car les maillots blancs perturbent le tracker
        # color_ball = self._detect_by_color(frame, hint)
        # if color_ball is not None:
        #     candidates.append((color_ball.confidence, color_ball, "couleur"))

        if not candidates:
            self._lost_frames += 1
            if self._initialized and self._lost_frames <= self.max_lost_frames:
                px, py = self._predict()
                self._last_state = BallState(
                    center=(px, py),
                    bbox=self._last_state.bbox,
                    confidence=max(0.1, self._last_state.confidence - 0.05),
                    visible=False,
                    source="kalman",
                )
            else:
                self._last_state = BallState(visible=False, source="perdu")
            return self._last_state

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, det, source = candidates[0]
        cx, cy = det.center

        if self._last_state.center:
            spd = np.hypot(cx - self._last_state.center[0], cy - self._last_state.center[1])
            self._speed_history.append(float(spd))
            if len(self._speed_history) > 8:
                self._speed_history.pop(0)

        self._correct(cx, cy)
        self._lost_frames = 0
        self._last_state = BallState(
            center=det.center,
            bbox=det.bbox,
            confidence=det.confidence,
            visible=True,
            source=source,
        )
        return self._last_state
