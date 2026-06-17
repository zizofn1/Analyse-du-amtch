"""Détection joueurs/ballon — YOLOv8s + résolution adaptative + retry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

from .filters import build_pitch_mask, filter_player_detections
from .types import PERSON_CLASS, SPORTS_BALL_CLASS, Detection


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        person_conf: float = 0.32,
        ball_conf: float = 0.10,
        max_players: int = 23,
    ):
        base_name = Path(model_name).stem
        openvino_model_path = f"{base_name}_openvino_model"
        
        self.openvino = False
        if os.path.exists(openvino_model_path):
            self.model = YOLO(openvino_model_path, task='detect')
            self.openvino = True
        else:
            self.model = YOLO(model_name)
            try:
                print("Exportation vers OpenVINO (NPU) en cours...")
                self.model.export(format="openvino", dynamic=True)
                self.model = YOLO(openvino_model_path, task='detect')
                self.openvino = True
            except Exception as e:
                print(f"Échec de l'exportation OpenVINO : {e}")

        self.person_conf = person_conf
        self.ball_conf = ball_conf
        self.max_players = max_players
        self._pitch_mask: np.ndarray | None = None

    def _infer_size(self, frame: np.ndarray) -> int:
        if self.openvino:
            return 640
            
        h, w = frame.shape[:2]
        # On bloque la résolution maximale à 736px pour optimiser massivement la vitesse
        # (L'ancienne version montait à 1280px ce qui divisait la vitesse par 4)
        if max(h, w) >= 1280:
            return 736
        return 640

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], Optional[Detection]]:
        min_conf = min(self.person_conf, self.ball_conf, 0.12)
        results = self.model(frame, verbose=False, conf=min_conf, imgsz=self._infer_size(frame), iou=0.5)[0]

        players: list[Detection] = []
        ball: Optional[Detection] = None

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                c = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                det = Detection(bbox=(x1, y1, x2, y2), confidence=c, class_id=cls_id, center=center)

                if cls_id == PERSON_CLASS and c >= self.person_conf:
                    players.append(det)
                elif cls_id == SPORTS_BALL_CLASS and c >= self.ball_conf:
                    if ball is None or c > ball.confidence:
                        ball = det

        if self._pitch_mask is None:
            self._pitch_mask = build_pitch_mask(frame)

        players = filter_player_detections(
            frame, players, pitch_mask=self._pitch_mask, max_players=self.max_players
        )

        # Retry avec confiance plus basse si peu de joueurs
        if len(players) < 6:
            low_conf = max(0.18, self.person_conf - 0.12)
            retry_players: list[Detection] = []
            if results.boxes is not None:
                for box in results.boxes:
                    if int(box.cls[0].item()) != PERSON_CLASS:
                        continue
                    c = float(box.conf[0].item())
                    if c < low_conf:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    retry_players.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=c,
                            class_id=PERSON_CLASS,
                            center=((x1 + x2) / 2, (y1 + y2) / 2),
                        )
                    )
            retry_players = filter_player_detections(
                frame, retry_players, pitch_mask=self._pitch_mask, max_players=self.max_players
            )
            if len(retry_players) > len(players):
                players = retry_players

        return players, ball
