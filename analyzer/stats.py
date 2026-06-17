"""Agrégation des statistiques de match et joueurs."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .events import EventDetector, EventType, PlayerState
from .team_colors import ROLE_REF


@dataclass
class TeamStats:
    name: str
    passes: int = 0
    shots: int = 0
    possession_pct: float = 0.0
    players: int = 0


@dataclass
class MatchStats:
    total_frames: int = 0
    duration_sec: float = 0.0
    total_passes: int = 0
    total_shots: int = 0
    teams: dict[str, TeamStats] = field(default_factory=dict)
    player_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    events_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_match_stats(detector: EventDetector, total_frames: int, fps: float) -> MatchStats:
    teams: dict[str, TeamStats] = {
        "équipe A": TeamStats(name="équipe A"),
        "équipe B": TeamStats(name="équipe B"),
        "inconnu": TeamStats(name="inconnu"),
    }

    total_possession = sum(p.possession_frames for p in detector.players.values()) or 1

    player_rows = []
    for pid, p in sorted(detector.players.items()):
        if p.team == ROLE_REF:
            continue
        if p.touches == 0 and p.distance_px < 20 and p.possession_frames == 0:
            continue
        team = teams.get(p.team, teams["inconnu"])
        team.passes += p.passes_made
        team.shots += p.shots
        team.players += 1
        team.possession_pct += (p.possession_frames / total_possession) * 100

        player_rows.append(
            {
                "Joueur": p.slot_id or f"#{pid}",
                "ID": pid,
                "Équipe": p.team,
                "Passes": p.passes_made,
                "Passes reçues": p.passes_received,
                "Tirs": p.shots,
                "Touches": p.touches,
                "Distance (px)": round(p.distance_px, 1),
                "Possession (%)": round((p.possession_frames / total_possession) * 100, 1),
            }
        )

    event_rows = [
        {
            "Temps (s)": round(e.timestamp_sec, 2),
            "Événement": e.event_type.value,
            "De": e.from_slot or (f"#{e.from_player}" if e.from_player else "-"),
            "Vers": e.to_slot or (f"#{e.to_player}" if e.to_player else "-"),
            "Vitesse": round(e.speed, 1),
        }
        for e in detector.events
    ]

    return MatchStats(
        total_frames=total_frames,
        duration_sec=round(total_frames / fps, 2),
        total_passes=sum(1 for e in detector.events if e.event_type == EventType.PASS),
        total_shots=sum(1 for e in detector.events if e.event_type == EventType.SHOT),
        teams=teams,
        player_df=pd.DataFrame(player_rows) if player_rows else pd.DataFrame(),
        events_df=pd.DataFrame(event_rows) if event_rows else pd.DataFrame(),
    )
