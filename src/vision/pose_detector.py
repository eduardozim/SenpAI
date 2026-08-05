"""
Detector de Poses baseado em MediaPipe Pose para captura de movimentos de Kendo.
Rastreia as 33 articulações (landmarks) e calcula centros de massa e orientação.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Any

class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.6):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2, # Alta precisão
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Any]], np.ndarray]:
        """
        Processa um frame BGR e retorna (landmarks_dict, frame_desenhado).
        """
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        annotated_frame = frame.copy()
        landmarks_dict = None
        
        if results.pose_landmarks:
            # Desenhar skeleton no frame
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Extrair dicionário de landmarks chave em pixel e normalizado
            landmarks_dict = {}
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                name = self.mp_pose.PoseLandmark(idx).name
                landmarks_dict[name] = {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                    "px": int(lm.x * w),
                    "py": int(lm.y * h)
                }
                
        return landmarks_dict, annotated_frame

    def release(self):
        self.pose.close()
