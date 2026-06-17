"""Classification maillots via LAB + KMeans + Delta E (scikit-learn)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .color_analysis import (
    delta_e_cie76,
    dominant_lab_color,
    is_referee_lab,
    lab_to_bgr,
)
from .types import Detection

ROLE_TEAM_A = "équipe A"
ROLE_TEAM_B = "équipe B"
ROLE_REF = "arbitre"
ROLE_UNKNOWN = "inconnu"

DISPLAY_BGR: dict[str, tuple[int, int, int]] = {
    ROLE_TEAM_A: (60, 60, 255),
    ROLE_TEAM_B: (50, 220, 50),
    ROLE_REF: (255, 220, 80),
    ROLE_UNKNOWN: (140, 140, 140),
}

SLOT_LIMITS = {ROLE_TEAM_A: 11, ROLE_TEAM_B: 11, ROLE_REF: 1}

# Seuil Delta E : < 25 = couleurs similaires à l'œil humain
MAX_DELTA_E = 38.0


class TeamColorClassifier:
    def __init__(self) -> None:
        self.calibrated = False
        self.centroids_lab: np.ndarray | None = None
        self.cluster_roles: list[str] = []
        self.role_bgr: dict[str, tuple[int, int, int]] = dict(DISPLAY_BGR)

    def _features_for_players(self, frame: np.ndarray, players: list[Detection]) -> list[np.ndarray]:
        feats: list[np.ndarray] = []
        for det in players:
            lab = dominant_lab_color(frame, det.bbox)
            if lab is not None:
                feats.append(lab)
        return feats

    def calibrate(self, frame: np.ndarray, players: list[Detection]) -> bool:
        if len(players) < 4:
            return False

        features_list = self._features_for_players(frame, players)
        if len(features_list) < 4:
            return False

        data = np.vstack(features_list)
        k = min(3, len(data))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(data)
        centroids = km.cluster_centers_.astype(np.float32)
        self.centroids_lab = centroids

        counts = [int(np.sum(labels == i)) for i in range(k)]
        cluster_roles: list[str | None] = [None] * k

        ref_idx: int | None = None
        for i, c in enumerate(centroids):
            if is_referee_lab(c):
                ref_idx = i
                break
        if ref_idx is None and k >= 3:
            ref_idx = int(np.argmin(counts))

        team_names = [ROLE_TEAM_A, ROLE_TEAM_B]
        ti = 0
        for i in range(k):
            if ref_idx is not None and i == ref_idx:
                cluster_roles[i] = ROLE_REF
                self.role_bgr[ROLE_REF] = lab_to_bgr(centroids[i])
            elif ti < len(team_names):
                cluster_roles[i] = team_names[ti]
                self.role_bgr[team_names[ti]] = lab_to_bgr(centroids[i])
                ti += 1
            else:
                cluster_roles[i] = ROLE_UNKNOWN

        self.cluster_roles = [r or ROLE_UNKNOWN for r in cluster_roles]
        self.calibrated = True
        return True

    def assign(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[str, tuple[int, int, int]]:
        if not self.calibrated or self.centroids_lab is None:
            return ROLE_UNKNOWN, DISPLAY_BGR[ROLE_UNKNOWN]

        lab = dominant_lab_color(frame, bbox)
        if lab is None:
            return ROLE_UNKNOWN, DISPLAY_BGR[ROLE_UNKNOWN]

        dists = [delta_e_cie76(lab, c) for c in self.centroids_lab]
        idx = int(np.argmin(dists))
        if dists[idx] > MAX_DELTA_E:
            return ROLE_UNKNOWN, DISPLAY_BGR[ROLE_UNKNOWN]

        role = self.cluster_roles[idx]
        return role, self.role_bgr.get(role, DISPLAY_BGR[ROLE_UNKNOWN])
