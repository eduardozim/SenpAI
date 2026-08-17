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

    def _extract_landmarks_dict(self, pose_landmarks, w: int, h: int, offset_x: int = 0, offset_y: int = 0, orig_w: Optional[int] = None, orig_h: Optional[int] = None) -> Dict[str, Any]:
        """Extrai e projeta os landmarks de um crop para o espaço do frame global."""
        if orig_w is None:
            orig_w = w
        if orig_h is None:
            orig_h = h

        landmarks_dict = {}
        for idx, lm in enumerate(pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(idx).name
            px_global = int(lm.x * w + offset_x)
            py_global = int(lm.y * h + offset_y)
            x_norm = float(px_global / max(1, orig_w))
            y_norm = float(py_global / max(1, orig_h))

            landmarks_dict[name] = {
                "x": float(np.clip(x_norm, 0.0, 1.0)),
                "y": float(np.clip(y_norm, 0.0, 1.0)),
                "z": float(lm.z),
                "visibility": float(lm.visibility),
                "px": px_global,
                "py": py_global
            }
        return landmarks_dict

    def process_frame_candidates(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Processa o frame em busca de múltiplos praticantes (Kenshi Aka e Shiro no Shiaijo).
        Retorna uma lista de dicionários de landmarks de candidatos detectados.
        """
        h, w, _ = frame.shape
        candidates: List[Dict[str, Any]] = []

        # 1. Detecção Global (Centro/Dominante)
        lm_global, _ = self.process_frame(frame)
        if lm_global:
            candidates.append(lm_global)

        # 2. Detecção no Hemisfério Esquerdo (Kenshi Aka) se imagem for ampla
        if w >= 320:
            left_w = int(w * 0.65)
            left_crop = frame[:, :left_w]
            frame_rgb_l = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)
            res_l = self.pose.process(frame_rgb_l)
            if res_l.pose_landmarks:
                lm_l = self._extract_landmarks_dict(res_l.pose_landmarks, left_w, h, offset_x=0, offset_y=0, orig_w=w, orig_h=h)
                # Adicionar se não for idêntico ao global
                if not self._is_duplicate(lm_l, candidates):
                    candidates.append(lm_l)

            # 3. Detecção no Hemisfério Direito (Kenshi Shiro)
            right_offset = int(w * 0.35)
            right_w = w - right_offset
            right_crop = frame[:, right_offset:]
            frame_rgb_r = cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB)
            res_r = self.pose.process(frame_rgb_r)
            if res_r.pose_landmarks:
                lm_r = self._extract_landmarks_dict(res_r.pose_landmarks, right_w, h, offset_x=right_offset, offset_y=0, orig_w=w, orig_h=h)
                if not self._is_duplicate(lm_r, candidates):
                    candidates.append(lm_r)

        return candidates, frame

    @staticmethod
    def _is_duplicate(candidate: Dict[str, Any], existing_list: List[Dict[str, Any]], threshold_dist: float = 0.08) -> bool:
        """Verifica se o candidato já existe na lista com base na proximidade do quadril/nariz."""
        c_nose = candidate.get("NOSE", candidate.get("RIGHT_HIP", {}))
        if not c_nose:
            return False

        cx, cy = c_nose.get("x", 0.5), c_nose.get("y", 0.5)

        for ex in existing_list:
            ex_nose = ex.get("NOSE", ex.get("RIGHT_HIP", {}))
            if not ex_nose:
                continue
            ex_x, ex_y = ex_nose.get("x", 0.5), ex_nose.get("y", 0.5)
            dist = np.hypot(cx - ex_x, cy - ex_y)
            if dist < threshold_dist:
                return True
        return False

    @staticmethod
    def draw_combatants_overlay(
        frame: np.ndarray,
        aka_landmarks: Optional[Dict[str, Any]],
        shiro_landmarks: Optional[Dict[str, Any]],
        discarded_items: Optional[List[Dict[str, Any]]] = None,
        sonkyo_status: Optional[str] = None,
        match_timer_str: Optional[str] = None
    ) -> np.ndarray:
        """
        Renderiza anotações gráficas ricas diferenciando Kenshi Aka (Vermelho), Kenshi Shiro (Branco/Ciano),
        elementos de fundo descartados e o status de Sonkyō no vídeo.
        """
        h, w, _ = frame.shape
        out = frame.copy()

        # 1. Desenhar Aka (Vermelho)
        if aka_landmarks:
            CombatantVisualizer.draw_skeleton(out, aka_landmarks, color=(40, 40, 235), label="KENSHI AKA (VERMELHO)")

        # 2. Desenhar Shiro (Branco/Ciano)
        if shiro_landmarks:
            CombatantVisualizer.draw_skeleton(out, shiro_landmarks, color=(240, 240, 50), label="KENSHI SHIRO (BRANCO)")

        # 3. Desenhar elementos de segundo plano descartados
        if discarded_items:
            for item in discarded_items:
                lm = item.get("landmarks")
                p_type = item.get("plane_type", "BACKGROUND")
                if lm:
                    tag = "[2º PLANO DESCARTADO]" if p_type == "BACKGROUND" else "[OCLUSÃO DESCARTADA]"
                    CombatantVisualizer.draw_discarded_marker(out, lm, label=tag)

        # 4. HUD / Banner Superior de Status
        if sonkyo_status or match_timer_str:
            overlay_h = 45
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, overlay_h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
            cv2.line(out, (0, overlay_h), (w, overlay_h), (59, 130, 246), 2)

            info_text = f"⚔️ SHINPANAI | {sonkyo_status or 'COMBATE ATIVO'}"
            if match_timer_str:
                info_text += f" | ⏱️ {match_timer_str}"
            cv2.putText(out, info_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return out

    def release(self):
        if hasattr(self, "pose"):
            self.pose.close()


class CombatantVisualizer:
    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks: Dict[str, Any], color: Tuple[int, int, int], label: str):
        h, w, _ = frame.shape
        xs = [pt["px"] for pt in landmarks.values() if isinstance(pt, dict) and "px" in pt]
        ys = [pt["py"] for pt in landmarks.values() if isinstance(pt, dict) and "py" in pt]

        if not xs or not ys:
            return

        xmin, xmax = max(0, min(xs) - 15), min(w - 1, max(xs) + 15)
        ymin, ymax = max(0, min(ys) - 15), min(h - 1, max(ys) + 15)

        # Caixa delimitadora do lutador
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        # Label no topo da caixa
        lbl_w = len(label) * 9 + 10
        cv2.rectangle(frame, (xmin, max(0, ymin - 22)), (xmin + lbl_w, ymin), color, -1)
        cv2.putText(frame, label, (xmin + 5, max(15, ymin - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # Desenhar articulações chave
        connections = [
            ("RIGHT_SHOULDER", "LEFT_SHOULDER"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            ("RIGHT_ELBOW", "RIGHT_WRIST"),
            ("LEFT_SHOULDER", "LEFT_ELBOW"),
            ("LEFT_ELBOW", "LEFT_WRIST"),
            ("RIGHT_SHOULDER", "RIGHT_HIP"),
            ("LEFT_SHOULDER", "LEFT_HIP"),
            ("RIGHT_HIP", "LEFT_HIP"),
            ("RIGHT_HIP", "RIGHT_KNEE"),
            ("RIGHT_KNEE", "RIGHT_ANKLE"),
            ("LEFT_HIP", "LEFT_KNEE"),
            ("LEFT_KNEE", "LEFT_ANKLE")
        ]

        for p1_name, p2_name in connections:
            if p1_name in landmarks and p2_name in landmarks:
                p1 = (landmarks[p1_name]["px"], landmarks[p1_name]["py"])
                p2 = (landmarks[p2_name]["px"], landmarks[p2_name]["py"])
                cv2.line(frame, p1, p2, color, 2)

        for pt_data in landmarks.values():
            if isinstance(pt_data, dict) and "px" in pt_data:
                cv2.circle(frame, (pt_data["px"], pt_data["py"]), 3, (255, 255, 255), -1)

    @staticmethod
    def draw_discarded_marker(frame: np.ndarray, landmarks: Dict[str, Any], label: str):
        h, w, _ = frame.shape
        xs = [pt["px"] for pt in landmarks.values() if isinstance(pt, dict) and "px" in pt]
        ys = [pt["py"] for pt in landmarks.values() if isinstance(pt, dict) and "py" in pt]

        if not xs or not ys:
            return

        xmin, xmax = max(0, min(xs)), min(w - 1, max(xs))
        ymin, ymax = max(0, min(ys)), min(h - 1, max(ys))

        # Caixa tracejada/cinza discreta para indicar o descarte do plano
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (100, 116, 139), 1)
        cv2.putText(frame, label, (xmin, max(12, ymin - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1)

