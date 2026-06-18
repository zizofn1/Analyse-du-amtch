"""Détection d'événements : passes, tirs, possession — logique affinée par machine à états."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .roster_tracker import FrameTracks, TrackedObject
from .team_colors import ROLE_REF


class EventType(str, Enum):
    PASS = "passe"
    SHOT = "tir"
    POSSESSION = "possession"


@dataclass
class MatchEvent:
    frame: int
    timestamp_sec: float
    event_type: EventType
    from_player: Optional[int] = None
    to_player: Optional[int] = None
    from_slot: str = ""
    to_slot: str = ""
    speed: float = 0.0
    path: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PassAttempt:
    start_frame: int
    from_player: int
    from_team: str
    from_slot: str
    path: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PlayerState:
    track_id: int
    slot_id: str = ""
    team: str = "inconnu"
    passes_made: int = 0
    passes_received: int = 0
    shots: int = 0
    touches: int = 0
    distance_px: float = 0.0
    possession_frames: int = 0
    last_center: Optional[tuple[float, float]] = None


@dataclass
class EventConfig:
    possession_radius: float = 75.0
    pass_min_distance: float = 28.0
    pass_cooldown_frames: int = 8
    shot_cooldown_frames: int = 60
    owner_confirm_frames: int = 2
    min_possession_before_pass: int = 3
    fps: float = 30.0
    goal_zones: list[tuple[int, int, int, int]] = field(default_factory=list)


def in_goal_zone(x: float, y: float, zones: list[tuple[int, int, int, int]]) -> bool:
    for zx1, zy1, zx2, zy2 in zones:
        if zx1 <= x <= zx2 and zy1 <= y <= zy2:
            return True
    return False

def ray_intersects_rect(px: float, py: float, vx: float, vy: float, rect: tuple[int, int, int, int]) -> float:
    rx1, ry1, rx2, ry2 = rect
    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
        return -1.0

    tmin, tmax = 0.0, 100000.0
    if abs(vx) > 1e-3:
        tx1 = (rx1 - px) / vx
        tx2 = (rx2 - px) / vx
        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))
    elif px < rx1 or px > rx2:
        return -1.0
        
    if abs(vy) > 1e-3:
        ty1 = (ry1 - py) / vy
        ty2 = (ry2 - py) / vy
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))
    elif py < ry1 or py > ry2:
        return -1.0
        
    if tmax >= tmin and tmax > 0:
        return tmin if tmin > 0 else 0.0
    return -1.0

class EventDetector:
    def __init__(self, config: Optional[EventConfig] = None):
        self.cfg = config or EventConfig()
        self.events: list[MatchEvent] = []
        self.players: dict[int, PlayerState] = {}
        
        # Machine à états pour les passes
        self._confirmed_owner: Optional[int] = None
        self._current_pass: Optional[PassAttempt] = None
        
        self._frames_since_shot = 999
        self._frame_idx = 0
        self._ball_history: list[tuple[float, float]] = []

    def _ensure_player(self, track: TrackedObject) -> PlayerState:
        if track.track_id not in self.players:
            self.players[track.track_id] = PlayerState(track_id=track.track_id, slot_id=track.slot_id)
        state = self.players[track.track_id]
        state.slot_id = track.slot_id
        if track.team != "inconnu":
            state.team = track.team
        return state

    def _is_field_player(self, track: TrackedObject) -> bool:
        return track.confirmed and track.team not in ("inconnu", ROLE_REF)

    def _get_track(self, tracks: FrameTracks, track_id: int) -> Optional[TrackedObject]:
        for p in tracks.players:
            if p.track_id == track_id:
                return p
        return None

    def _nearest_player(self, ball_pos: tuple[float, float], players: list[TrackedObject]) -> tuple[Optional[int], float]:
        best_id = None
        best_dist = 999999.0
        for p in players:
            if not self._is_field_player(p):
                continue
            dist = np.hypot(p.center[0] - ball_pos[0], p.center[1] - ball_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_id = p.track_id
        return best_id, best_dist

    def process_frame(
        self,
        tracks: FrameTracks,
        ball_speed: float,
        frame_width: int,
    ) -> None:
        self._frame_idx += 1
        self._frames_since_shot += 1
        timestamp = self._frame_idx / self.cfg.fps

        # Mise à jour des distances parcourues
        for p in tracks.players:
            if not p.confirmed:
                continue
            state = self._ensure_player(p)
            if state.last_center is not None:
                state.distance_px += np.hypot(
                    p.center[0] - state.last_center[0],
                    p.center[1] - state.last_center[1],
                )
            state.last_center = p.center

        ball_pos = tracks.ball.center
        if ball_pos is None:
            return

        # Lissage de la vitesse sur 5 frames
        self._ball_history.append(ball_pos)
        if len(self._ball_history) > 5:
            self._ball_history.pop(0)

        vx, vy = 0.0, 0.0
        if len(self._ball_history) >= 3:
            dx = self._ball_history[-1][0] - self._ball_history[0][0]
            dy = self._ball_history[-1][1] - self._ball_history[0][1]
            frames = len(self._ball_history) - 1
            vx = dx / frames
            vy = dy / frames
            
        smoothed_speed = np.hypot(vx, vy)

        closest_id, min_dist = self._nearest_player(ball_pos, tracks.players)
        is_possessed = min_dist <= self.cfg.possession_radius

        # --- GESTION DES TIRS ---
        is_shot = False
        shot_reason = ""
        
        # 1. Balle physiquement dans la zone de but (avec marge)
        for zone in self.cfg.goal_zones:
            zx1, zy1, zx2, zy2 = zone
            if zx1 - 50 <= ball_pos[0] <= zx2 + 50 and zy1 - 50 <= ball_pos[1] <= zy2 + 50:
                is_shot = True
                shot_reason = "Balle entrée physiquement dans la zone de but"
                break
                
        # 2. Balle très rapide pointant vers le but (impact < 8 frames)
        if not is_shot and smoothed_speed > 25.0:
            for zone in self.cfg.goal_zones:
                t = ray_intersects_rect(ball_pos[0], ball_pos[1], vx, vy, zone)
                if 0 <= t < 8.0:
                    is_shot = True
                    shot_reason = f"Trajectoire ultra-rapide ({smoothed_speed:.1f}px/f) impactant le but dans {t:.1f} frames"
                    break

        if is_shot and self._frames_since_shot >= self.cfg.shot_cooldown_frames:
            shooter_id = None
            shot_path = []
            
            if self._current_pass is not None:
                # Une passe s'est transformée en tir !
                shooter_id = self._current_pass.from_player
                shot_path = self._current_pass.path
                self._current_pass = None
                print(f"[EVENEMENT] ⚠️ Une passe en vol s'est transformée en TIR ! Joueur {shooter_id}.")
            else:
                shooter_id = self._confirmed_owner or closest_id
            
            if shooter_id is not None:
                shooter_track = self._get_track(tracks, shooter_id)
                if shooter_track and self._is_field_player(shooter_track):
                    shooter = self._ensure_player(shooter_track)
                    shooter.shots += 1
                    
                    print(f"[EVENEMENT] 🎯 TIR détecté ! Joueur {shooter_id} ({shooter_track.team}). Raison : {shot_reason}")
                    
                    self.events.append(
                        MatchEvent(
                            frame=self._frame_idx,
                            timestamp_sec=timestamp,
                            event_type=EventType.SHOT,
                            from_player=shooter_id,
                            from_slot=shooter.slot_id,
                            speed=smoothed_speed,
                            path=shot_path
                        )
                    )
                    self._frames_since_shot = 0
                    self._fast_ball_streak = 0
                    self._confirmed_owner = shooter_id

        # --- MACHINE À ÉTATS DES PASSES ---
        if is_possessed and closest_id is not None:
            # ÉTATS 1 & 4 : La balle est chez quelqu'un (Possession ou Réception)
            new_owner_track = self._get_track(tracks, closest_id)
            if new_owner_track:
                new_owner_state = self._ensure_player(new_owner_track)
                new_owner_state.touches += 1
                new_owner_state.possession_frames += 1
                
                # Si une passe était en vol (Réception)
                if self._current_pass is not None:
                    if closest_id != self._current_pass.from_player: # Pas le même joueur qu'au départ
                        pass_dist = 0.0
                        if len(self._current_pass.path) > 0:
                            start_pos = self._current_pass.path[0]
                            pass_dist = np.hypot(ball_pos[0] - start_pos[0], ball_pos[1] - start_pos[1])
                            
                        # ÉTAT 5 : Validation
                        if pass_dist >= self.cfg.pass_min_distance:
                            if new_owner_state.team == self._current_pass.from_team:
                                # Passe réussie (même équipe)
                                print(f"[EVENEMENT] ⚽ PASSE réussie ! De {self._current_pass.from_player} à {closest_id} ({new_owner_state.team}). "
                                      f"Raison: Réception après un vol de {pass_dist:.1f}px (min requis {self.cfg.pass_min_distance})")
                                from_p = self.players.get(self._current_pass.from_player)
                                if from_p: from_p.passes_made += 1
                                new_owner_state.passes_received += 1
                                
                                self.events.append(MatchEvent(
                                    frame=self._frame_idx,
                                    timestamp_sec=timestamp,
                                    event_type=EventType.PASS,
                                    from_player=self._current_pass.from_player,
                                    to_player=closest_id,
                                    from_slot=self._current_pass.from_slot,
                                    to_slot=new_owner_track.slot_id,
                                    speed=ball_speed,
                                    path=self._current_pass.path
                                ))
                            else:
                                # Interception / Passe ratée (équipe différente)
                                print(f"[EVENEMENT] 🛑 INTERCEPTION / Passe ratée. Le joueur {closest_id} ({new_owner_state.team}) "
                                      f"a intercepté la balle venant de {self._current_pass.from_player} ({self._current_pass.from_team}).")
                                pass
                        else:
                            print(f"[DEBUG] Fausse passe ignorée entre {self._current_pass.from_player} et {closest_id} : distance {pass_dist:.1f}px trop courte (< {self.cfg.pass_min_distance}).")

                    # La passe est terminée (réussie, ratée, ou reprise par le même joueur)
                    self._current_pass = None
            
            # Mise à jour du possesseur actuel SEULEMENT SI le paramètre est respecté (tolérance 0)
            if new_owner_state and new_owner_state.possession_frames >= self.cfg.owner_confirm_frames:
                if self._confirmed_owner != closest_id:
                    print(f"[DEBUG] 🛡️ Joueur {closest_id} ({new_owner_state.team}) prend la possession officielle (Confirmé après {new_owner_state.possession_frames} frames).")
                self._confirmed_owner = closest_id
            
        else:
            # ÉTATS 2 & 3 : La balle n'est chez personne (Départ ou Vol)
            
            # ÉTAT 2 : Départ d'une passe
            if self._current_pass is None and self._confirmed_owner is not None:
                owner_track = self._get_track(tracks, self._confirmed_owner)
                if owner_track:
                    owner_state = self._ensure_player(owner_track)
                    # Vérification stricte : le départ de passe nécessite une vraie possession
                    if owner_state.possession_frames >= self.cfg.min_possession_before_pass:
                        print(f"[DEBUG] 🚀 La balle quitte le joueur {self._confirmed_owner} => Début d'une passe en vol.")
                        self._current_pass = PassAttempt(
                            start_frame=self._frame_idx,
                            from_player=self._confirmed_owner,
                            from_team=owner_state.team,
                            from_slot=owner_track.slot_id,
                            path=[]
                        )
                        # Reset possession frames as ball left
                        owner_state.possession_frames = 0
            
            # ÉTAT 3 : En vol, on enregistre le trajet de la balle
            if self._current_pass is not None:
                self._current_pass.path.append((int(ball_pos[0]), int(ball_pos[1])))
