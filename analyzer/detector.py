"""Détection joueurs/ballon — YOLOv8s + OpenVINO (CPU/GPU/NPU) + retry.

CORRECTION NPU :
  L'ancienne version exportait vers OpenVINO mais n'indiquait PAS le device NPU.
  OpenVINO utilise le CPU par défaut, même avec un modèle OpenVINO.
  
  Pour utiliser le vrai NPU Intel (Core Ultra / Meteor Lake+), il faut :
  1. Exporter sans dynamic=True (le NPU Intel ne supporte pas les shapes dynamiques)
  2. Passer device='NPU' lors de l'inférence
  3. Garder un fallback propre si le NPU n'est pas disponible
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from .filters import build_pitch_mask, filter_player_detections
from .types import PERSON_CLASS, SPORTS_BALL_CLASS, Detection


def _try_import_ultralytics():
    from ultralytics import YOLO
    return YOLO


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8m.pt",
        person_conf: float = 0.32,
        ball_conf: float = 0.10,
        max_players: int = 23,
    ):
        YOLO = _try_import_ultralytics()
        base_name = Path(model_name).stem
        openvino_model_path = f"{base_name}_openvino_model"

        self.openvino = False
        self._device = "cpu"  # device utilisé pour l'inférence
        
        if os.path.exists(openvino_model_path):
            # Modèle OpenVINO déjà exporté
            self.model = YOLO(openvino_model_path, task="detect")
            self.openvino = True
            self._device = self._detect_best_openvino_device()
            print(f"[Detector] OpenVINO chargé — device: {self._device}")
        else:
            self.model = YOLO(model_name)
            print("[Detector] Tentative d'exportation OpenVINO...")
            try:
                # CORRECTION: dynamic=False pour compatibilité NPU Intel
                self.model.export(format="openvino", dynamic=False, half=False)
                self.model = YOLO(openvino_model_path, task="detect")
                self.openvino = True
                self._device = self._detect_best_openvino_device()
                print(f"[Detector] OpenVINO exporté — device: {self._device}")
            except Exception as e:
                print(f"[Detector] Échec OpenVINO ({e}) — mode CPU standard")
                self.model = YOLO(model_name)

        self.person_conf = person_conf
        self.ball_conf = ball_conf
        self.max_players = max_players
        self._pitch_mask: np.ndarray | None = None

    def _detect_best_openvino_device(self) -> str:
        """
        Détecte le meilleur device OpenVINO disponible.
        Priorité : NPU > GPU > CPU
        
        Nécessite Intel OpenVINO Runtime installé.
        """
        try:
            from openvino.runtime import Core
            ie = Core()
            available = ie.available_devices
            print(f"[Detector] Devices OpenVINO disponibles : {available}")
            for preferred in ["NPU", "GPU", "CPU"]:
                if preferred in available:
                    return preferred
        except ImportError:
            pass
        except Exception as e:
            print(f"[Detector] OpenVINO device check échoué : {e}")
        return "cpu"

    def _infer_size(self, frame: np.ndarray) -> int:
        """
        Taille d'inférence adaptée.
        
        NOTE NPU : Les NPU Intel supportent mieux les tailles fixes (640).
        Les résolutions trop élevées peuvent causer des erreurs sur NPU.
        """
        if self.openvino:
            return 640  # Taille fixe pour OpenVINO/NPU
        h, w = frame.shape[:2]
        if max(h, w) >= 1280:
            return 736
        return 640

    def detect(self, frame: np.ndarray) -> tuple[list[Detection], Optional[Detection]]:
        min_conf = min(self.person_conf, self.ball_conf, 0.12)
        infer_kwargs: dict = {
            "verbose": False,
            "conf": min_conf,
            "imgsz": self._infer_size(frame),
            "iou": 0.5,
            "classes": [PERSON_CLASS, SPORTS_BALL_CLASS],
        }
        
        # Spécifier le device seulement pour OpenVINO (et si ce n'est pas cpu, qui est le défaut)
        if self.openvino and self._device.upper() in ("NPU", "GPU"):
            infer_kwargs["device"] = self._device
        
        try:
            results = self.model(frame, **infer_kwargs)[0]
        except Exception as e:
            # Fallback sans device spécifié si le NPU/GPU échoue
            print(f"[Detector] Inférence {self._device} échouée ({e}), fallback CPU")
            self._device = "cpu"
            results = self.model(frame, verbose=False, conf=min_conf, imgsz=self._infer_size(frame), iou=0.5, classes=[PERSON_CLASS, SPORTS_BALL_CLASS])[0]

        players: list[Detection] = []
        ball: Optional[Detection] = None

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                c = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                det = Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=c,
                    class_id=cls_id,
                    center=center,
                )
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

        # Retry avec confiance plus basse si trop peu de joueurs
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