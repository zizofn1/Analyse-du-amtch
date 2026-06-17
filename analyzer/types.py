"""Types partagés pour la détection et le suivi."""

from __future__ import annotations

from dataclasses import dataclass


PERSON_CLASS = 0
SPORTS_BALL_CLASS = 32


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    center: tuple[float, float]
