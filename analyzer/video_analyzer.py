"""Pipeline principal d'analyse vidéo de match."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .ball_tracker import BallTracker
from .detector import ObjectDetector
from .events import EventConfig, EventDetector, EventType
from .filters import build_pitch_mask
from .roster_tracker import FrameTracks, RosterTracker, TrackedBall
from .stats import MatchStats, build_match_stats
from .team_colors import ROLE_REF, TeamColorClassifier
from .types import Detection
from .pitch_mapper import PitchMapper

BALL_BGR = (0, 255, 255)
BALL_KALMAN_BGR = (0, 200, 255)
CALIBRATION_SCAN_FRAMES = 18


@dataclass
class AnalysisConfig:
    model_name: str = "yolov8s.pt"
    person_conf: float = 0.32
    ball_conf: float = 0.08
    frame_skip: int = 1
    detector_skip: int = 4  # YOLO runs every 4 processed frames
    max_frames: int = 900
    possession_radius: float = 75.0
    pass_min_distance: float = 28.0
    pass_cooldown: int = 8
    shot_cooldown: int = 60
    owner_confirm_frames: int = 2
    min_possession_pass: int = 3
    max_players: int = 23
    goal_zones: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def conf_threshold(self) -> float:
        return self.person_conf

    def event_config(self, fps: float) -> EventConfig:
        return EventConfig(
            possession_radius=self.possession_radius,
            pass_min_distance=self.pass_min_distance,
            pass_cooldown_frames=self.pass_cooldown,
            shot_cooldown_frames=self.shot_cooldown,
            owner_confirm_frames=self.owner_confirm_frames,
            min_possession_before_pass=self.min_possession_pass,
            goal_zones=self.goal_zones,
            fps=fps,
        )


@dataclass
class AnalysisResult:
    stats: MatchStats
    annotated_video_path: str
    preview_frames: list[np.ndarray]
    fps: float
    frame_count: int
    players_tracked: int = 0
    events: list[MatchEvent] = field(default_factory=list)


@dataclass
class VisualEffect:
    pts: list[tuple[int, int]]
    frames_left: int
    color: tuple[int, int, int]


def detect_goal_zones(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Auto-detects goal zones based on the pitch mask boundaries."""
    pitch_mask = build_pitch_mask(frame)
    h, w = pitch_mask.shape
    # Find bounding box of the pitch
    ys, xs = np.where(pitch_mask > 0)
    if len(xs) == 0:
        return []
    
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    
    # Left Goal Zone
    left_goal = (int(max(0, min_x - 20)), int(min_y), int(min_x + int(w * 0.05)), int(max_y))
    # Right Goal Zone
    right_goal = (int(max_x - int(w * 0.05)), int(min_y), int(min(w, max_x + 20)), int(max_y))
    return [left_goal, right_goal]


class VideoAnalyzer:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.detector = ObjectDetector(
            model_name=self.config.model_name,
            person_conf=self.config.person_conf,
            ball_conf=self.config.ball_conf,
            max_players=self.config.max_players,
        )

    def analyze(
        self,
        video_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        headless: bool = False,
    ) -> AnalysisResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = min(total_frames, self.config.max_frames)
        roster = RosterTracker()
        ball_tracker = BallTracker(search_radius=max(80, self.config.possession_radius * 1.2))
        team_classifier = TeamColorClassifier()
        teams_calibrated = False

        best_cal_frame: np.ndarray | None = None
        best_cal_players: list[Detection] = []
        best_cal_count = 0

        # Auto-detect goals if not provided
        ret, first_frame = cap.read()
        if ret and not self.config.goal_zones:
            self.config.goal_zones = detect_goal_zones(first_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        pitch_mapper = PitchMapper()
        if progress_callback:
            progress_callback(0.02, "Extraction et alignement 2D (Minimap)...")
        pitch_mapper.compute_homographies(video_path, num_keyframes=6)

        event_detector = EventDetector(self.config.event_config(fps))

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "match_analyse.mp4")
        writer = None
        if not headless:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        preview_frames: list[np.ndarray] = []
        processed = 0
        frame_idx = 0
        ball_source = "—"

        ball_trajectory: list[tuple[int, int]] = []
        visual_effects: list[VisualEffect] = []
        prev_event_count = 0

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.config.frame_skip != 0:
                frame_idx += 1
                continue

            run_yolo = (not teams_calibrated) or (processed % self.config.detector_skip == 0)

            if run_yolo:
                players, yolo_ball = self.detector.detect(frame)
                ball_state = ball_tracker.update(frame, yolo_ball)
            else:
                players = []
                ball_state = ball_tracker.update(frame, None)

            ball_source = ball_state.source

            if not teams_calibrated:
                if len(players) > best_cal_count:
                    best_cal_count = len(players)
                    best_cal_frame = frame.copy()
                    best_cal_players = players

                ready = (
                    processed >= CALIBRATION_SCAN_FRAMES
                    or best_cal_count >= 10
                    or (processed >= 5 and best_cal_count >= 6)
                )
                if ready and best_cal_frame is not None and best_cal_count >= 4:
                    teams_calibrated = team_classifier.calibrate(best_cal_frame, best_cal_players)
                    if teams_calibrated:
                        det_roles = [team_classifier.assign(best_cal_frame, d.bbox) for d in best_cal_players]
                        roster.initialize_roster(
                            best_cal_frame,
                            list(zip(best_cal_players, [r[0] for r in det_roles], [r[1] for r in det_roles]))
                        )
                        if progress_callback:
                            progress_callback(
                                processed / max(max_frames, 1),
                                f"Calibration ({best_cal_count} joueurs)",
                            )

            player_list: list = []
            if teams_calibrated:
                if run_yolo:
                    det_roles = [team_classifier.assign(frame, det.bbox) for det in players]
                    if players:
                        avg_h = float(np.mean([det.bbox[3] - det.bbox[1] for det in players]))
                        roster.set_max_match_dist(avg_h * 1.3)
                        event_detector.cfg.possession_radius = max(self.config.possession_radius, avg_h * 0.8)
                        event_detector.cfg.pass_min_distance = max(
                            self.config.pass_min_distance, avg_h * 0.4
                        )
                    player_list = roster.update_with_detections(frame, players, det_roles)
                else:
                    player_list = roster.update_trackers(frame)

            tracks = FrameTracks(
                players=player_list,
                ball=TrackedBall(
                    center=ball_state.center,
                    bbox=ball_state.bbox,
                    confidence=ball_state.confidence,
                    visible=ball_state.visible,
                ),
            )

            if ball_state.center:
                ball_trajectory.append((int(ball_state.center[0]), int(ball_state.center[1])))
                if len(ball_trajectory) > int(fps):
                    ball_trajectory.pop(0)

            if roster.initialized and ball_state.center:
                event_detector.process_frame(tracks, ball_tracker.speed, width)
                
                # Check for new events to spawn visual effects
                if not headless and len(event_detector.events) > prev_event_count:
                    new_event = event_detector.events[-1]
                    color = (0, 255, 0) if new_event.event_type == EventType.PASS else (0, 0, 255)
                    pts_to_draw = new_event.path if len(new_event.path) > 0 else list(ball_trajectory)
                    visual_effects.append(VisualEffect(
                        pts=pts_to_draw,
                        frames_left=int(fps * 1.5), # Affiché pendant 1.5 seconde
                        color=color
                    ))
                    prev_event_count = len(event_detector.events)

            if not headless:
                annotated = self._draw_frame(
                    frame, tracks, event_detector, teams_calibrated, roster, len(player_list), ball_source, visual_effects, pitch_mapper, frame_idx
                )
                if writer is not None:
                    writer.write(annotated)

                if len(preview_frames) < 6 and processed % max(1, max_frames // 6) == 0:
                    preview_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            processed += 1
            frame_idx += 1

            if progress_callback:
                progress_callback(processed / max_frames, f"Frame {processed}/{max_frames}")

        cap.release()
        if writer is not None:
            writer.release()

        stats = build_match_stats(event_detector, processed, fps)

        return AnalysisResult(
            stats=stats,
            annotated_video_path=out_path,
            preview_frames=preview_frames,
            fps=fps,
            frame_count=processed,
            players_tracked=len(stats.player_df),
            events=event_detector.events,
        )

    def _draw_frame(
        self,
        frame: np.ndarray,
        tracks: FrameTracks,
        events: EventDetector,
        teams_calibrated: bool,
        roster: RosterTracker,
        detected_count: int,
        ball_source: str,
        visual_effects: list[VisualEffect],
        pitch_mapper: Optional[PitchMapper] = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        out = frame.copy()

        # Draw Goal Zones
        for idx, (x1, y1, x2, y2) in enumerate(self.config.goal_zones):
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(out, f"BUT {idx+1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw Trajectory effects
        for effect in visual_effects:
            if len(effect.pts) > 1 and effect.frames_left > 0:
                pts = np.array(effect.pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(out, [pts], isClosed=False, color=effect.color, thickness=3)
            effect.frames_left -= 1
        
        # Remove expired effects
        visual_effects[:] = [e for e in visual_effects if e.frames_left > 0]

        for p in tracks.players:
            if p.missed_frames > 8:
                continue
            color = p.team_bgr
            x1, y1, x2, y2 = map(int, p.bbox)
            thickness = 3 if p.confirmed else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = "ARBITRE" if p.team == ROLE_REF else p.slot_id
            cv2.putText(out, label, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if tracks.ball.center:
            cx, cy = int(tracks.ball.center[0]), int(tracks.ball.center[1])
            bcol = BALL_BGR if tracks.ball.visible else BALL_KALMAN_BGR
            cv2.circle(out, (cx, cy), 11, bcol, -1)
            cv2.circle(out, (cx, cy), 13, (0, 0, 0), 2)
            if tracks.ball.bbox and tracks.ball.visible:
                bx1, by1, bx2, by2 = map(int, tracks.ball.bbox)
                cv2.rectangle(out, (bx1, by1), (bx2, by2), bcol, 2)
            cv2.putText(out, f"BALLE({ball_source})", (cx + 14, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcol, 2)

        counts = roster.slot_counts
        n_a, n_b, n_r = counts["équipe A"], counts["équipe B"], counts[ROLE_REF]
        if teams_calibrated:
            status = f"Det:{detected_count} A:{n_a}/11 B:{n_b}/11 ARB:{n_r}/1"
        else:
            status = f"Scan... det:{detected_count}"
        cv2.putText(out, status, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        recent = events.events[-3:] if events.events else []
        y_offset = 28
        for ev in recent:
            src = ev.from_slot or f"#{ev.from_player}"
            text = f"{ev.timestamp_sec:.1f}s {ev.event_type.value} ({src})"
            cv2.putText(out, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            y_offset += 22

        if pitch_mapper is not None:
            minimap = pitch_mapper.draw_minimap(tracks.players, tracks.ball.center, frame_idx)
            mh, mw = minimap.shape[:2]
            scale = 0.35
            m_sw, m_sh = int(mw * scale), int(mh * scale)
            mini_rsz = cv2.resize(minimap, (m_sw, m_sh))
            x_off = out.shape[1] - m_sw - 20
            y_off = 20
            if y_off+m_sh < out.shape[0] and x_off+m_sw < out.shape[1]:
                roi = out[y_off:y_off+m_sh, x_off:x_off+m_sw]
                cv2.addWeighted(roi, 0.1, mini_rsz, 0.9, 0, roi)

        return out
