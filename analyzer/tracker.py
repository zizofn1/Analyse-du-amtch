"""Suivi multi-objets avec confirmation et couleurs d'équipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .types import Detection
from .filters import iou


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    team: str = "inconnu"
    team_bgr: tuple[int, int, int] = (160, 160, 160)
    frames_seen: int = 1
    missed_frames: int = 0
    confirmed: bool = False


@dataclass
class TrackedBall:
    center: Optional[tuple[float, float]] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    confidence: float = 0.0
    visible: bool = False


@dataclass
class FrameTracks:
    players: list[TrackedObject] = field(default_factory=list)
    ball: TrackedBall = field(default_factory=TrackedBall)


def _match_score(
    det: Detection,
    track: TrackedObject,
    max_dist: float,
) -> float:
    iou_score = iou(det.bbox, track.bbox)
    dist = np.hypot(det.center[0] - track.center[0], det.center[1] - track.center[1])
    dist_score = max(0.0, 1.0 - dist / max_dist)
    return 0.65 * iou_score + 0.35 * dist_score


class SimpleTracker:
    def __init__(
        self,
        match_threshold: float = 0.25,
        max_missed: int = 12,
        min_hits_to_confirm: int = 3,
        max_tracks: int = 24,
    ):
        self.match_threshold = match_threshold
        self.max_missed = max_missed
        self.min_hits_to_confirm = min_hits_to_confirm
        self.max_tracks = max_tracks
        self._next_id = 1
        self._active: dict[int, TrackedObject] = {}
        self._last_ball: TrackedBall = TrackedBall()
        self._max_match_dist = 120.0

    def set_max_match_dist(self, value: float) -> None:
        self._max_match_dist = max(40.0, value)

    def update(
        self,
        player_dets: list[Detection],
        ball_det: Optional[Detection],
        det_team_info: list[tuple[str, tuple[int, int, int]]] | None = None,
    ) -> FrameTracks:
        det_team_info = det_team_info or [("inconnu", (160, 160, 160))] * len(player_dets)
        matched_ids: set[int] = set()
        updated: dict[int, TrackedObject] = {}

        det_order = sorted(
            range(len(player_dets)),
            key=lambda i: player_dets[i].confidence,
            reverse=True,
        )

        for di in det_order:
            det = player_dets[di]
            best_id = None
            best_score = self.match_threshold

            for tid, track in self._active.items():
                if tid in matched_ids:
                    continue
                score = _match_score(det, track, self._max_match_dist)
                if score > best_score:
                    best_score = score
                    best_id = tid

            if best_id is not None:
                matched_ids.add(best_id)
                prev = self._active[best_id]
                frames_seen = prev.frames_seen + 1
                team, team_bgr = det_team_info[di]
                if team == "inconnu" and prev.team != "inconnu":
                    team, team_bgr = prev.team, prev.team_bgr
                updated[best_id] = TrackedObject(
                    track_id=best_id,
                    bbox=det.bbox,
                    center=det.center,
                    confidence=det.confidence,
                    team=team,
                    team_bgr=team_bgr,
                    frames_seen=frames_seen,
                    missed_frames=0,
                    confirmed=frames_seen >= self.min_hits_to_confirm,
                )
            elif len(self._active) + len(updated) < self.max_tracks:
                tid = self._next_id
                self._next_id += 1
                matched_ids.add(tid)
                team, team_bgr = det_team_info[di]
                updated[tid] = TrackedObject(
                    track_id=tid,
                    bbox=det.bbox,
                    center=det.center,
                    confidence=det.confidence,
                    team=team,
                    team_bgr=team_bgr,
                    confirmed=False,
                )

        for tid, track in self._active.items():
            if tid not in matched_ids:
                track.missed_frames += 1
                limit = self.max_missed if track.confirmed else 5
                if track.missed_frames <= limit:
                    updated[tid] = track

        self._active = updated

        if ball_det is not None:
            self._last_ball = TrackedBall(
                center=ball_det.center,
                bbox=ball_det.bbox,
                confidence=ball_det.confidence,
                visible=True,
            )
        else:
            self._last_ball = TrackedBall(
                center=self._last_ball.center,
                bbox=self._last_ball.bbox,
                confidence=max(0, self._last_ball.confidence - 0.03),
                visible=False,
            )

        confirmed_players = [p for p in self._active.values() if p.confirmed]
        return FrameTracks(players=confirmed_players, ball=self._last_ball)
