"""Module de mapping 2D (Minimap) pour transformer la perspective caméra vers un terrain 2D."""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Pitch2DConfig:
    width_m: float = 105.0  # Mètres
    height_m: float = 68.0  # Mètres
    scale: int = 6          # 1 mètre = 6 pixels pour la minimap (630 x 408 px)

class PitchMapper:
    def __init__(self, config: Optional[Pitch2DConfig] = None):
        self.config = config or Pitch2DConfig()
        self.w = int(self.config.width_m * self.config.scale)
        self.h = int(self.config.height_m * self.config.scale)
        self.pitch_img = self._create_pitch_template()
        self.pitch_edges = cv2.Canny(self.pitch_img, 50, 150)
        self.homographies: dict[int, np.ndarray] = {}  # index frame -> Matrice 3x3
        self.keyframes: list[int] = []

    def _create_pitch_template(self) -> np.ndarray:
        """Dessine le terrain 2D parfait (vert avec lignes blanches)."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)  # Vert gazon
        
        color = (255, 255, 255)
        thick = 2
        
        # Contour extérieur
        cv2.rectangle(img, (0, 0), (self.w-1, self.h-1), color, thick)
        # Ligne médiane
        cv2.line(img, (self.w//2, 0), (self.w//2, self.h), color, thick)
        # Rond central
        cv2.circle(img, (self.w//2, self.h//2), int(9.15 * self.config.scale), color, thick)
        
        # Surfaces de réparation
        pen_w = int(16.5 * self.config.scale)
        pen_h = int(40.3 * self.config.scale)
        cv2.rectangle(img, (0, self.h//2 - pen_h//2), (pen_w, self.h//2 + pen_h//2), color, thick)
        cv2.rectangle(img, (self.w - pen_w, self.h//2 - pen_h//2), (self.w, self.h//2 + pen_h//2), color, thick)
        
        return img

    def preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Applique l'idée de l'utilisateur : filtre B&W pour isoler les lignes sur l'herbe."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # Masque pour isoler le vert
        lower_green = np.array([30, 30, 30])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Détection de contours (Canny) sur l'image en niveaux de gris
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # On ne garde que les contours qui sont dans la zone verte (les lignes blanches)
        lines_bw = cv2.bitwise_and(edges, edges, mask=mask)
        return lines_bw

    def compute_homographies(self, video_path: str, num_keyframes: int = 6) -> None:
        """Extrait N frames (Keyframes), calcule leur homographie, puis interpole."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return

        step = max(1, total // num_keyframes)
        
        # Initialiseur de features OpenCV pour trouver les correspondances
        orb = cv2.ORB_create(nfeatures=1000)
        kp_pitch, des_pitch = orb.detectAndCompute(self.pitch_edges, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Matrice de secours : simple mise à l'échelle si la détection 3D échoue
        scale_x = self.w / max(1, w_vid)
        scale_y = self.h / max(1, h_vid)
        fallback_H = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        last_valid_H = fallback_H.copy()

        for i in range(num_keyframes + 1):
            idx = min(i * step, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            # Traitement de la frame (B&W)
            bw_frame = self.preprocess_frame(frame)
            
            # Essai de corrélation avec le terrain 2D
            kp_frame, des_frame = orb.detectAndCompute(bw_frame, None)
            
            H = last_valid_H.copy()
            if des_frame is not None and des_pitch is not None and len(des_frame) > 10:
                matches = bf.match(des_frame, des_pitch)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Si on a assez de bons matchs, on calcule la perspective
                if len(matches) > 10:
                    src_pts = np.float32([kp_frame[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_pitch[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
                    
                    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if matrix is not None:
                        H = matrix
                        last_valid_H = matrix

            self.homographies[idx] = H
            self.keyframes.append(idx)
            
        cap.release()

    def get_homography(self, frame_idx: int) -> np.ndarray:
        """Interpole la matrice de perspective pour n'importe quelle frame."""
        if not self.keyframes:
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            
        if frame_idx in self.homographies:
            return self.homographies[frame_idx]
            
        self.keyframes.sort()
        if frame_idx < self.keyframes[0]:
            return self.homographies[self.keyframes[0]]
        if frame_idx >= self.keyframes[-1]:
            return self.homographies[self.keyframes[-1]]
            
        # Trouver les deux keyframes encadrantes
        idx1, idx2 = self.keyframes[0], self.keyframes[-1]
        for i in range(len(self.keyframes)-1):
            if self.keyframes[i] <= frame_idx < self.keyframes[i+1]:
                idx1 = self.keyframes[i]
                idx2 = self.keyframes[i+1]
                break
                
        # Interpolation linéaire simple des matrices
        h1 = self.homographies[idx1]
        h2 = self.homographies[idx2]
        ratio = (frame_idx - idx1) / (idx2 - idx1)
        h_interp = h1 * (1.0 - ratio) + h2 * ratio
        return h_interp

    def map_point(self, x: float, y: float, frame_idx: int) -> tuple[int, int]:
        """Convertit un point (x,y) de la vidéo vers la carte 2D."""
        H = self.get_homography(frame_idx)
        pt = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, H)
        mx, my = mapped[0][0]
        # Restreindre dans les limites du terrain
        return int(max(0, min(self.w-1, mx))), int(max(0, min(self.h-1, my)))

    def draw_minimap(self, players: list, ball: tuple[float, float], frame_idx: int) -> np.ndarray:
        """Crée l'image de la minimap avec tous les joueurs dessinés dessus."""
        map_img = self.pitch_img.copy()
        
        # Joueurs
        for p in players:
            # On projette le milieu de la base de la bbox (les pieds du joueur)
            feet_y = p.bbox[3]
            mx, my = self.map_point(p.center[0], feet_y, frame_idx)
            color = p.team_bgr if hasattr(p, 'team_bgr') else (255, 0, 0)
            cv2.circle(map_img, (mx, my), 5, color, -1)
            cv2.circle(map_img, (mx, my), 5, (255, 255, 255), 1)
            
        # Balle
        if ball is not None:
            bx, by = self.map_point(ball[0], ball[1], frame_idx)
            cv2.circle(map_img, (bx, by), 4, (0, 255, 255), -1) # Balle jaune
            cv2.circle(map_img, (bx, by), 4, (0, 0, 0), 1)
            
        return map_img
