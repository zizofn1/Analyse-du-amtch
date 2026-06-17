"""Suivi par slots fixes avec OpenCV Tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .filters import iou
from .team_colors import ROLE_REF, ROLE_TEAM_A, ROLE_TEAM_B, ROLE_UNKNOWN, SLOT_LIMITS
from .types import Detection


def slot_numeric_id(slot_id: str) -> int:
    if slot_id == "REF":
        return 99
    prefix = slot_id[0]
    num = int(slot_id[1:])
    return num if prefix == "A" else 11 + num


@dataclass
class TrackedObject:
    track_id: int
    slot_id: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    confidence: float
    team: str = ROLE_UNKNOWN
    team_bgr: tuple[int, int, int] = (140, 140, 140)
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


def _match_score(det: Detection, slot: TrackedObject, max_dist: float) -> float:
    if slot.bbox is None:
        return 0.0
    iou_score = iou(det.bbox, slot.bbox)
    dist = np.hypot(det.center[0] - slot.center[0], det.center[1] - slot.center[1])
    dist_score = max(0.0, 1.0 - dist / max_dist)
    return 0.7 * iou_score + 0.3 * dist_score



class RosterTracker:
    """11 slots équipe A, 11 équipe B, 1 arbitre."""

    def __init__(
        self,
        match_threshold: float = 0.18,
        max_missed: int = 18,
        min_hits_to_confirm: int = 1,
    ):
        self.match_threshold = match_threshold
        self.max_missed = max_missed
        self.min_hits_to_confirm = min_hits_to_confirm
        self._slots: dict[str, TrackedObject] = {}
        self._initialized = False
        self._max_match_dist = 100.0

    @property
    def slot_counts(self) -> dict[str, int]:
        counts = {"équipe A": 0, "équipe B": 0, ROLE_REF: 0}
        for s in self._slots.values():
            if s.team in counts:
                counts[s.team] += 1
        return counts

    def set_max_match_dist(self, value: float) -> None:
        self._max_match_dist = max(35.0, value)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _make_slot_id(self, role: str, index: int) -> str:
        if role == ROLE_REF:
            return "REF"
        prefix = "A" if role == ROLE_TEAM_A else "B"
        return f"{prefix}{index:02d}"

    def initialize_roster(
        self,
        frame: np.ndarray,
        frame_assignments: list[tuple[Detection, str, tuple[int, int, int]]],
    ) -> None:
        by_role: dict[str, list[tuple[Detection, tuple[int, int, int]]]] = {
            ROLE_TEAM_A: [],
            ROLE_TEAM_B: [],
            ROLE_REF: [],
        }
        for det, role, bgr in frame_assignments:
            if role in by_role:
                by_role[role].append((det, bgr))

        for role, items in by_role.items():
            limit = SLOT_LIMITS.get(role, 0)
            items.sort(key=lambda x: (x[0].center[1], x[0].center[0]))
            for i, (det, bgr) in enumerate(items[:limit], start=1):
                sid = self._make_slot_id(role, i)
                self._slots[sid] = TrackedObject(
                    track_id=slot_numeric_id(sid),
                    slot_id=sid,
                    bbox=det.bbox,
                    center=det.center,
                    confidence=det.confidence,
                    team=role,
                    team_bgr=bgr,
                    frames_seen=1,
                    confirmed=False
                )

        self._initialized = len(self._slots) > 0

    def update_trackers(self, frame: np.ndarray) -> list[TrackedObject]:
        # No more slow cv2 trackers! We rely solely on YOLO + IoU distance matching in update_with_detections
        for sid, slot in list(self._slots.items()):
            slot.missed_frames += 1
            if slot.missed_frames > self.max_missed:
                del self._slots[sid]

        
        return [
            s for s in self._slots.values()
            if s.confirmed or (s.missed_frames <= 4 and s.frames_seen >= 1)
        ]

    def update_with_detections(
        self,
        frame: np.ndarray,
        player_dets: list[Detection],
        det_roles: list[tuple[str, tuple[int, int, int]]],
    ) -> list[TrackedObject]:
        if self._initialized:
            self._match_frame(frame, player_dets, det_roles)

        return [
            s for s in self._slots.values()
            if s.confirmed or (s.missed_frames <= 4 and s.frames_seen >= 1)
        ]

    def _match_frame(
        self,
        frame: np.ndarray,
        player_dets: list[Detection],
        det_roles: list[tuple[str, tuple[int, int, int]]],
    ) -> None:
        matched_slots: set[str] = set()
        matched_dets: set[int] = set()

        slot_ids = list(self._slots.keys())
        det_indices = list(range(len(player_dets)))

        pairs: list[tuple[float, int, str]] = []
        for di in det_indices:
            det = player_dets[di]
            role, bgr = det_roles[di]
            for sid in slot_ids:
                slot = self._slots[sid]
                if slot.team != role:
                    continue
                score = _match_score(det, slot, self._max_match_dist)
                if score >= self.match_threshold:
                    pairs.append((score, di, sid))

        pairs.sort(reverse=True)
        for score, di, sid in pairs:
            if di in matched_dets or sid in matched_slots:
                continue
            det = player_dets[di]
            role, bgr = det_roles[di]
            prev = self._slots[sid]
            frames_seen = prev.frames_seen + 1
            self._slots[sid] = TrackedObject(
                track_id=prev.track_id,
                slot_id=sid,
                bbox=det.bbox,
                center=det.center,
                confidence=det.confidence,
                team=role,
                team_bgr=bgr,
                frames_seen=frames_seen,
                missed_frames=0,
                confirmed=frames_seen >= self.min_hits_to_confirm
            )
            matched_dets.add(di)
            matched_slots.add(sid)

        for sid, slot in list(self._slots.items()):
            if sid in matched_slots:
                continue
            slot.missed_frames += 1
            if slot.missed_frames <= self.max_missed:
                pass
            else:
                del self._slots[sid]

        self._add_new_slots(frame, player_dets, det_roles, matched_dets)

    def _add_new_slots(
        self,
        frame: np.ndarray,
        player_dets: list[Detection],
        det_roles: list[tuple[str, tuple[int, int, int]]],
        matched_dets: set[int],
    ) -> None:
        for di, det in enumerate(player_dets):
            if di in matched_dets:
                continue
            role, bgr = det_roles[di]
            if role == ROLE_UNKNOWN:
                continue
            count = sum(1 for s in self._slots.values() if s.team == role)
            limit = SLOT_LIMITS.get(role, 0)
            if count >= limit:
                continue
            sid = self._make_slot_id(role, count + 1)
            if sid in self._slots:
                continue
            self._slots[sid] = TrackedObject(
                track_id=slot_numeric_id(sid),
                slot_id=sid,
                bbox=det.bbox,
                center=det.center,
                confidence=det.confidence,
                team=role,
                team_bgr=bgr,
                frames_seen=1,
                confirmed=False
            )
