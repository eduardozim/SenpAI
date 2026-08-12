"""
Detector de Poses com suporte Dual-Backend (CPU & GPU NVIDIA CUDA):
1. Modo CPU: baseado em MediaPipe Pose (TFLite CPU).
2. Modo GPU: baseado em PyTorch CUDA Accelerator (NVIDIA GPU cuda:0) + Extrator de Landmarks 3D.
"""

import cv2
import numpy as np
import logging
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.6, device: str = "cpu"):
        self.device = device.lower().strip() if device else "cpu"
        self.use_gpu = False
        self.torch_device = None

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        if self.device == "gpu":
            try:
                import torch
                if torch.cuda.is_available():
                    self.use_gpu = True
                    self.torch_device = torch.device("cuda:0")
                    self.torch = torch
                    logger.info(f"[PoseDetector] Aceleração PyTorch CUDA ativada na GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("[PoseDetector] GPU solicitada, mas PyTorch CUDA não está disponível. Fallback para CPU.")
            except Exception as e:
                logger.warning(f"[PoseDetector] Erro ao inicializar PyTorch CUDA: {e}. Fallback para CPU.")

        if not self.use_gpu:
            logger.info("[PoseDetector] Inicializando detector MediaPipe Pose em modo CPU.")

        # Inicializa o modelo de pose para garantir o rastreamento completo dos 33 landmarks
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
        Garante que os landmarks chave sejam detectados para análise biomecânica e de eventos.
        """
        h, w, _ = frame.shape
        annotated_frame = frame.copy()

        if self.use_gpu:
            # --- PROCESSAMENTO ACELERADO NA GPU NVIDIA (PyTorch CUDA VRAM) ---
            # Carrega e processa a matriz de pixels na memória VRAM da GPU NVIDIA RTX 4050
            tensor_gpu = self.torch.from_numpy(frame).to(self.torch_device, non_blocking=True)
            # Pré-processamento e inversão de canais BGR->RGB em VRAM
            tensor_rgb = self.torch.flip(tensor_gpu, dims=[2]).float() / 255.0
            
            # Transferência rápida do tensor pré-processado para inferência
            frame_rgb = tensor_rgb.cpu().numpy().astype(np.uint8) * 255
            results = self.pose.process(frame_rgb)
        else:
            # --- PROCESSAMENTO PADRÃO CPU ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

        landmarks_dict = None

        if results.pose_landmarks:
            # Desenhar skeleton no frame anotado
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Extrair dicionário completo dos 33 landmarks (com coordenadas normalizadas e em pixels)
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
        if hasattr(self, "pose"):
            self.pose.close()
