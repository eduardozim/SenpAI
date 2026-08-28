"""
Detector de Poses com suporte Dual-Backend de Alta Performance:
1. Modo GPU NVIDIA CUDA: baseado em YOLOv8-Pose (PyTorch CUDA VRAM cuda:0) para inferência multi-person em tempo real (100+ FPS).
2. Modo CPU: baseado em MediaPipe Pose (TFLite CPU) com extração precisa de landmarks 3D.
"""

import os
import warnings

# Suprime aviso benigno interno de depreciação do protobuf com mediapipe
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import numpy as np
import logging
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Mapeamento oficial dos 17 keypoints COCO para o padrão de nomenclatura SenpAI / MediaPipe
COCO_INDEX_TO_LANDMARK = {
    0: "NOSE",
    1: "LEFT_EYE",
    2: "RIGHT_EYE",
    3: "LEFT_EAR",
    4: "RIGHT_EAR",
    5: "LEFT_SHOULDER",
    6: "RIGHT_SHOULDER",
    7: "LEFT_ELBOW",
    8: "RIGHT_ELBOW",
    9: "LEFT_WRIST",
    10: "RIGHT_WRIST",
    11: "LEFT_HIP",
    12: "RIGHT_HIP",
    13: "LEFT_KNEE",
    14: "RIGHT_KNEE",
    15: "LEFT_ANKLE",
    16: "RIGHT_ANKLE"
}

class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.6, device: str = "cpu"):
        self.device = device.lower().strip() if device else "cpu"
        self.use_gpu = False
        self.yolo_model = None
        self.torch_device = None

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        if self.device == "gpu":
            try:
                import torch
                from ultralytics import YOLO
                if torch.cuda.is_available():
                    self.use_gpu = True
                    self.torch_device = "cuda:0"
                    self.torch = torch
                    
                    # Localizar modelo YOLOv8-pose
                    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolov8n-pose.pt")
                    if not os.path.exists(model_path):
                        model_path = "yolov8n-pose.pt"
                    
                    self.yolo_model = YOLO(model_path)
                    self.yolo_model.to("cuda:0")
                    
                    # Aquecimento de inferência (warmup)
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    _ = self.yolo_model(dummy, device="cuda:0", verbose=False)
                    
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"[PoseDetector] 🚀 Aceleração Nativa NVIDIA CUDA ativada com sucesso: {gpu_name} (YOLOv8-Pose)")
                else:
                    logger.warning("[PoseDetector] GPU solicitada, mas PyTorch CUDA não está disponível. Fallback para CPU.")
            except Exception as e:
                logger.warning(f"[PoseDetector] Erro ao inicializar aceleração GPU NVIDIA: {e}. Fallback para CPU MediaPipe.")

        if not self.use_gpu:
            logger.info("[PoseDetector] Inicializando detector MediaPipe Pose em modo CPU.")
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1, # Otimizado para CPU
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

    def _yolo_results_to_landmarks_list(self, yolo_res, w: int, h: int) -> List[Dict[str, Any]]:
        """Converte as predições de múltiplos esqueletos do YOLOv8-Pose para o formato de landmarks do SenpAI."""
        candidates = []
        if not yolo_res or len(yolo_res) == 0:
            return candidates

        res = yolo_res[0]
        if res.keypoints is None or res.keypoints.data is None:
            return candidates

        kpts_data = res.keypoints.data.cpu().numpy() # [num_persons, 17, 3] (x_px, y_px, conf)

        for person_idx in range(len(kpts_data)):
            person_kpts = kpts_data[person_idx]
            lm_dict = {}

            for coco_idx, (px, py, conf) in enumerate(person_kpts):
                name = COCO_INDEX_TO_LANDMARK.get(coco_idx)
                if not name:
                    continue

                x_norm = float(np.clip(px / max(1, w), 0.0, 1.0))
                y_norm = float(np.clip(py / max(1, h), 0.0, 1.0))

                lm_dict[name] = {
                    "x": x_norm,
                    "y": y_norm,
                    "z": 0.0,
                    "visibility": float(conf),
                    "px": int(px),
                    "py": int(py)
                }

            # Sintetizar pés/calcanhares para compatibilidade total com os módulos biomecânicos
            if "RIGHT_ANKLE" in lm_dict:
                lm_dict["RIGHT_FOOT_INDEX"] = dict(lm_dict["RIGHT_ANKLE"])
                lm_dict["RIGHT_HEEL"] = dict(lm_dict["RIGHT_ANKLE"])
            if "LEFT_ANKLE" in lm_dict:
                lm_dict["LEFT_FOOT_INDEX"] = dict(lm_dict["LEFT_ANKLE"])
                lm_dict["LEFT_HEEL"] = dict(lm_dict["LEFT_ANKLE"])

            # Validar se o esqueleto contém pontos suficientes (ombros e quadris)
            has_shoulders = "RIGHT_SHOULDER" in lm_dict and "LEFT_SHOULDER" in lm_dict
            has_hips = "RIGHT_HIP" in lm_dict and "LEFT_HIP" in lm_dict
            if has_shoulders and has_hips:
                candidates.append(lm_dict)

        return candidates

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Any]], np.ndarray]:
        """
        Processa um frame BGR e retorna (landmarks_dict, frame_desenhado).
        """
        h, w, _ = frame.shape
        annotated_frame = frame.copy()

        if self.use_gpu and self.yolo_model is not None:
            # --- INFERÊNCIA ACELERADA NA GPU NVIDIA CUDA (VRAM cuda:0) ---
            results = self.yolo_model(frame, device="cuda:0", verbose=False, conf=0.25, imgsz=640)
            candidates = self._yolo_results_to_landmarks_list(results, w, h)
            
            if candidates:
                # Escolher o candidato mais central/dominante
                primary_lm = max(candidates, key=lambda c: (1.0 - abs(c.get("NOSE", {}).get("x", 0.5) - 0.5)))
                CombatantVisualizer.draw_skeleton(annotated_frame, primary_lm, color=(59, 130, 246), label="KENSHI DETECTADO")
                return primary_lm, annotated_frame
            return None, annotated_frame

        # --- PROCESSAMENTO CPU MEDIAPIPE ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        landmarks_dict = None
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
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

    def process_frame_candidates(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Processa o frame em busca de múltiplos praticantes (Kenshi Aka e Shiro no Shiaijo).
        Retorna uma lista de dicionários de landmarks de candidatos detectados.
        """
        h, w, _ = frame.shape

        if self.use_gpu and self.yolo_model is not None:
            # --- INFERÊNCIA PARALELA MULTI-PESSOA NA GPU NVIDIA (1 ÚNICO PASSO EM VRAM) ---
            results = self.yolo_model(frame, device="cuda:0", verbose=False, conf=0.25, imgsz=640)
            candidates = self._yolo_results_to_landmarks_list(results, w, h)
            return candidates, frame

        # --- MODO CPU: DETECÇÃO GLOBAL + HEMISFÉRIOS ---
        candidates: List[Dict[str, Any]] = []
        lm_global, _ = self.process_frame(frame)
        if lm_global:
            candidates.append(lm_global)

        if w >= 320 and hasattr(self, "pose"):
            left_w = int(w * 0.65)
            left_crop = frame[:, :left_w]
            frame_rgb_l = cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB)
            res_l = self.pose.process(frame_rgb_l)
            if res_l.pose_landmarks:
                lm_l = self._extract_landmarks_dict(res_l.pose_landmarks, left_w, h, offset_x=0, offset_y=0, orig_w=w, orig_h=h)
                if not self._is_duplicate(lm_l, candidates):
                    candidates.append(lm_l)

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

            info_text = f"⚔️ SENPAI | {sonkyo_status or 'COMBATE ATIVO'}"
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

