"""
Detector de Poses com suporte Dual-Backend de Alta Performance:
1. Modo GPU NVIDIA CUDA: baseado em YOLOv8-Pose (PyTorch CUDA VRAM cuda:0) para inferência multi-person em tempo real (100+ FPS).
2. Modo CPU: baseado em MediaPipe Pose (TFLite CPU) com extração precisa de landmarks 3D.
"""

import os
import warnings

# Suprime avisos benignos internos
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", message=".*'half' is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import logging
try:
    import mediapipe as mp
except ImportError:
    mp = None

from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Suprime logs de avisos benignos do Ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)

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
        self.pose = None

        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        else:
            self.mp_pose = None
            self.mp_drawing = None
            self.mp_drawing_styles = None

        if self.device == "gpu":
            try:
                import torch
                from ultralytics import YOLO
                if torch.cuda.is_available():
                    self.use_gpu = True
                    self.torch_device = "cuda:0"
                    self.torch = torch
                    
                    # Ativar otimizações de kernel cuDNN e inferência
                    torch.backends.cudnn.benchmark = True
                    try:
                        torch.set_grad_enabled(False)
                    except Exception:
                        pass

                    # Localizar modelo YOLOv8-pose
                    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolov8n-pose.pt")
                    if not os.path.exists(model_path):
                        model_path = "yolov8n-pose.pt"
                    
                    self.yolo_model = YOLO(model_path)
                    self.yolo_model.to("cuda:0")
                    if hasattr(self.yolo_model.model, "half"):
                        try:
                            self.yolo_model.model.half()
                        except Exception:
                            pass
                    
                    # Aquecimento de inferência (warmup)
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    _ = self.yolo_model(dummy, device="cuda:0", verbose=False)
                    
                    gpu_name = torch.cuda.get_device_name(0)
                    logger.info(f"[PoseDetector] 🚀 Aceleração Nativa NVIDIA CUDA ativada com sucesso: {gpu_name} (YOLOv8-Pose FP16 + cuDNN Benchmark)")
                else:
                    logger.warning("[PoseDetector] GPU solicitada, mas PyTorch CUDA não está disponível. Fallback para CPU.")
            except Exception as e:
                logger.warning(f"[PoseDetector] Erro ao inicializar aceleração GPU NVIDIA: {e}. Fallback para CPU MediaPipe.")

        if not self.use_gpu:
            if self.mp_pose is not None:
                logger.info("[PoseDetector] Inicializando detector MediaPipe Pose em modo CPU.")
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1, # Otimizado para CPU
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            else:
                logger.info("[PoseDetector] MediaPipe Pose indisponível. Inicializando fallback YOLO em CPU.")
                try:
                    from ultralytics import YOLO
                    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolov8n-pose.pt")
                    if not os.path.exists(model_path):
                        model_path = "yolov8n-pose.pt"
                    if os.path.exists(model_path):
                        self.yolo_model = YOLO(model_path)
                        self.yolo_model.to("cpu")
                except Exception as e:
                    logger.warning(f"[PoseDetector] Fallback YOLO em CPU indisponível: {e}")

    def _single_yolo_result_to_landmarks(self, res, w: int, h: int) -> List[Dict[str, Any]]:
        """Converte as predições de múltiplos esqueletos de um resultado YOLOv8-Pose para o formato de landmarks do SenpAI."""
        candidates = []
        if res is None:
            return candidates

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

    def _yolo_results_to_landmarks_list(self, yolo_res, w: int, h: int) -> List[Dict[str, Any]]:
        """Converte as predições de múltiplos esqueletos do YOLOv8-Pose para o formato de landmarks do SenpAI (retrocompatibilidade)."""
        if not yolo_res or len(yolo_res) == 0:
            return []
        return self._single_yolo_result_to_landmarks(yolo_res[0], w, h)

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Any]], np.ndarray]:
        """
        Processa um frame BGR e retorna (landmarks_dict, frame_desenhado).
        """
        h, w, _ = frame.shape
        annotated_frame = frame.copy()

        if self.yolo_model is not None:
            # --- INFERÊNCIA YOLO (GPU OU CPU) ---
            dev = "cuda:0" if self.use_gpu else "cpu"
            results = self.yolo_model(frame, device=dev, verbose=False, conf=0.25, imgsz=640)
            candidates = self._yolo_results_to_landmarks_list(results, w, h)
            
            if candidates:
                # Escolher o candidato mais central/dominante
                primary_lm = max(candidates, key=lambda c: (1.0 - abs(c.get("NOSE", {}).get("x", 0.5) - 0.5)))
                CombatantVisualizer.draw_skeleton(annotated_frame, primary_lm, color=(59, 130, 246), label="KENSHI DETECTADO")
                return primary_lm, annotated_frame
            return None, annotated_frame

        if self.pose is not None:
            # --- PROCESSAMENTO CPU MEDIAPIPE ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            landmarks_dict = None
            if results and results.pose_landmarks:
                if self.mp_drawing and self.mp_pose:
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style() if self.mp_drawing_styles else None
                    )
                landmarks_dict = {}
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    name = self.mp_pose.PoseLandmark(idx).name if self.mp_pose else f"LANDMARK_{idx}"
                    landmarks_dict[name] = {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                        "px": int(lm.x * w),
                        "py": int(lm.y * h)
                    }

            return landmarks_dict, annotated_frame

        return None, annotated_frame

    def process_frame_candidates_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Processa um lote de múltiplos frames simultaneamente em busca de praticantes (Kenshi Aka e Shiro).
        Retorna uma lista de listas de dicionários de landmarks (uma lista de candidatos por frame).
        """
        if not frames:
            return []

        h, w = frames[0].shape[:2]

        if self.yolo_model is not None:
            # --- INFERÊNCIA PARALELA EM LOTE (BATCH INFERENCE) NA GPU OU CPU ---
            dev = "cuda:0" if self.use_gpu else "cpu"
            results = self.yolo_model(
                frames,
                device=dev,
                batch=len(frames),
                verbose=False,
                conf=0.25,
                imgsz=640
            )
            batch_candidates = []
            for res in results:
                cands = self._single_yolo_result_to_landmarks(res, w, h)
                batch_candidates.append(cands)
            return batch_candidates

        # --- MODO CPU (MEDIAPIPE): PROCESSAMENTO SEQUENCIAL TRANSPARENTE ---
        batch_candidates = []
        for frame in frames:
            cands, _ = self.process_frame_candidates(frame)
            batch_candidates.append(cands)
        return batch_candidates

    def process_frame_candidates(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Processa o frame em busca de múltiplos praticantes (Kenshi Aka e Shiro no Shiaijo).
        Retorna uma lista de dicionários de landmarks de candidatos detectados.
        """
        h, w, _ = frame.shape

        if self.yolo_model is not None:
            # --- INFERÊNCIA PARALELA MULTI-PESSOA YOLO (GPU OU CPU) ---
            dev = "cuda:0" if self.use_gpu else "cpu"
            results = self.yolo_model(frame, device=dev, verbose=False, conf=0.25, imgsz=640)
            candidates = self._yolo_results_to_landmarks_list(results, w, h)
            return candidates, frame

        # --- MODO CPU: DETECÇÃO GLOBAL + HEMISFÉRIOS ---
        candidates: List[Dict[str, Any]] = []
        lm_global, _ = self.process_frame(frame)
        if lm_global:
            candidates.append(lm_global)

        if w >= 320 and self.pose is not None:
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
        match_timer_str: Optional[str] = None,
        active_strikes: Optional[List[Dict[str, Any]]] = None,
        current_frame_idx: int = 0,
        aka_label: Optional[str] = None,
        shiro_label: Optional[str] = None
    ) -> np.ndarray:
        """
        Renderiza anotações gráficas ricas diferenciando Kenshi Aka (Vermelho), Kenshi Shiro (Branco/Ciano),
        elementos de fundo descartados, Shinai (espada), detecção de golpes em execução e status de Sonkyō.
        """
        h, w, _ = frame.shape
        out = frame.copy()

        # Determinar se algum combatente está executando golpe neste frame
        aka_is_striking = False
        shiro_is_striking = False
        if active_strikes:
            for ev in active_strikes:
                atk_id = ev.get("event_info", {}).get("attacker_id", "")
                if atk_id == "KENSHI_AKA":
                    aka_is_striking = True
                elif atk_id == "KENSHI_SHIRO":
                    shiro_is_striking = True

        # 1. Desenhar Aka (Vermelho) e seu Shinai
        label_aka = aka_label or "KENSHI AKA (VERMELHO)"
        if aka_landmarks:
            CombatantVisualizer.draw_skeleton(
                out,
                aka_landmarks,
                color=(40, 40, 235) if not aka_is_striking else (0, 0, 255),
                label=f"🔴 {label_aka}" + (" [⚡ ATAQUE]" if aka_is_striking else ""),
                is_striking=aka_is_striking
            )
            CombatantVisualizer.draw_shinai(out, aka_landmarks, is_striking=aka_is_striking)

        # 2. Desenhar Shiro (Branco/Ciano) e seu Shinai
        label_shiro = shiro_label or "KENSHI SHIRO (BRANCO)"
        if shiro_landmarks:
            CombatantVisualizer.draw_skeleton(
                out,
                shiro_landmarks,
                color=(240, 240, 50) if not shiro_is_striking else (255, 255, 0),
                label=f"⚪ {label_shiro}" + (" [⚡ ATAQUE]" if shiro_is_striking else ""),
                is_striking=shiro_is_striking
            )
            CombatantVisualizer.draw_shinai(out, shiro_landmarks, is_striking=shiro_is_striking)

        # 3. Desenhar elementos de segundo plano descartados
        if discarded_items:
            for item in discarded_items:
                lm = item.get("landmarks")
                p_type = item.get("plane_type", "BACKGROUND")
                if lm:
                    tag = "[2º PLANO DESCARTADO]" if p_type == "BACKGROUND" else "[OCLUSÃO DESCARTADA]"
                    CombatantVisualizer.draw_discarded_marker(out, lm, label=tag)

        # 4. Desenhar Alvos e Destaques de Golpes em Execução (Yuko-Datotsu / Ki-Ken-Tai-Ichi)
        if active_strikes:
            for strike_ev in active_strikes:
                CombatantVisualizer.draw_strike_overlay(
                    out,
                    strike_ev,
                    aka_landmarks=aka_landmarks,
                    shiro_landmarks=shiro_landmarks,
                    current_frame=current_frame_idx
                )

        # 5. HUD / Banner Superior de Status
        if sonkyo_status or match_timer_str:
            overlay_h = 42
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, overlay_h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.80, out, 0.20, 0, out)
            cv2.line(out, (0, overlay_h), (w, overlay_h), (59, 130, 246), 2)

            info_text = f"⚔️ SENPAI | {sonkyo_status or 'COMBATE ATIVO'}"
            if match_timer_str:
                info_text += f" | ⏱️ {match_timer_str}"
            cv2.putText(out, info_text, (20, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        return out

    def release(self):
        if hasattr(self, "pose"):
            self.pose.close()


class CombatantVisualizer:
    @staticmethod
    def _extract_pt(pt_data: Any, w: int, h: int) -> Optional[Tuple[int, int]]:
        """Extrai coordenadas em pixels de um landmark com fallback seguro para coordenadas normalizadas."""
        if not isinstance(pt_data, dict):
            return None
        if "px" in pt_data and "py" in pt_data and pt_data["px"] is not None and pt_data["py"] is not None:
            return int(pt_data["px"]), int(pt_data["py"])
        if "x" in pt_data and "y" in pt_data and pt_data["x"] is not None and pt_data["y"] is not None:
            return int(np.clip(pt_data["x"] * w, 0, w - 1)), int(np.clip(pt_data["y"] * h, 0, h - 1))
        return None

    @staticmethod
    def draw_skeleton(frame: np.ndarray, landmarks: Dict[str, Any], color: Tuple[int, int, int], label: str, is_striking: bool = False):
        h, w, _ = frame.shape
        if not landmarks:
            return

        pts_map: Dict[str, Tuple[int, int]] = {}
        for k, v in landmarks.items():
            pt = CombatantVisualizer._extract_pt(v, w, h)
            if pt is not None:
                pts_map[k] = pt

        if not pts_map:
            return

        xs = [p[0] for p in pts_map.values()]
        ys = [p[1] for p in pts_map.values()]

        xmin, xmax = max(0, min(xs) - 15), min(w - 1, max(xs) + 15)
        ymin, ymax = max(0, min(ys) - 15), min(h - 1, max(ys) + 15)

        # Caixa delimitadora do lutador (mais grossa se estiver executando golpe)
        box_thickness = 3 if is_striking else 2
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, box_thickness)

        # Label no topo da caixa com fundo colorido
        lbl_w = len(label) * 9 + 14
        cv2.rectangle(frame, (xmin, max(0, ymin - 22)), (min(w - 1, xmin + lbl_w), ymin), color, -1)
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

        bone_thickness = 3 if is_striking else 2
        for p1_name, p2_name in connections:
            if p1_name in pts_map and p2_name in pts_map:
                cv2.line(frame, pts_map[p1_name], pts_map[p2_name], color, bone_thickness, cv2.LINE_AA)

        for pt in pts_map.values():
            cv2.circle(frame, pt, 3 if not is_striking else 4, (255, 255, 255), -1, cv2.LINE_AA)

    @staticmethod
    def draw_shinai(frame: np.ndarray, landmarks: Dict[str, Any], is_striking: bool = False):
        """Desenha a espada Shinai projetada a partir dos pulsos ao longo do eixo do antebraço."""
        h, w, _ = frame.shape
        r_wrist = CombatantVisualizer._extract_pt(landmarks.get("RIGHT_WRIST"), w, h)
        r_elbow = CombatantVisualizer._extract_pt(landmarks.get("RIGHT_ELBOW"), w, h)

        if not r_wrist:
            return

        if r_elbow:
            dx = float(r_wrist[0] - r_elbow[0])
            dy = float(r_wrist[1] - r_elbow[1])
            norm = np.hypot(dx, dy)
            if norm > 5:
                # Projeta o comprimento da lâmina do Shinai
                shinai_len = max(38, int(norm * 1.40))
                tip_x = int(r_wrist[0] + (dx / norm) * shinai_len)
                tip_y = int(r_wrist[1] + (dy / norm) * shinai_len)
            else:
                tip_x, tip_y = r_wrist[0] + 30, r_wrist[1] - 45
        else:
            tip_x, tip_y = r_wrist[0] + 30, r_wrist[1] - 45

        tip_x = int(np.clip(tip_x, 0, w - 1))
        tip_y = int(np.clip(tip_y, 0, h - 1))

        sword_color = (0, 215, 255) if not is_striking else (0, 69, 255) # Ouro ou Vermelho Neon
        thickness = 3 if not is_striking else 5

        # Haste do Shinai
        cv2.line(frame, r_wrist, (tip_x, tip_y), sword_color, thickness, cv2.LINE_AA)
        # Kensen (Ponta da espada)
        cv2.circle(frame, (tip_x, tip_y), 4 if not is_striking else 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (tip_x, tip_y), 6 if not is_striking else 9, sword_color, 2, cv2.LINE_AA)

    @staticmethod
    def draw_strike_overlay(
        frame: np.ndarray,
        strike_event: Dict[str, Any],
        aka_landmarks: Optional[Dict[str, Any]],
        shiro_landmarks: Optional[Dict[str, Any]],
        current_frame: int
    ):
        """Renderiza anotações específicas do golpe em execução (alvo atingido e banner de diagnóstico)."""
        h, w, _ = frame.shape
        ev_info = strike_event.get("event_info", {})
        eval_d = strike_event.get("evaluation", {})

        attacker_id = ev_info.get("attacker_id", "KENSHI_AKA")
        strike_type = str(ev_info.get("type", "MEN")).upper()
        is_valid = eval_d.get("is_valid", False)
        score = eval_d.get("total_score", 0.0)
        impact_f = ev_info.get("impact_frame", current_frame)

        is_near_impact = abs(current_frame - impact_f) <= 4

        if attacker_id == "KENSHI_AKA":
            atk_name = "KENSHI AKA (VERMELHO)"
            def_lm = shiro_landmarks
        else:
            atk_name = "KENSHI SHIRO (BRANCO)"
            def_lm = aka_landmarks

        # 1. Desenhar Alvo Anatômico no Defensor
        if def_lm:
            target_pt = None
            if strike_type == "MEN":
                target_pt = CombatantVisualizer._extract_pt(def_lm.get("NOSE"), w, h)
                if not target_pt:
                    ys = [pt["y"] for pt in def_lm.values() if isinstance(pt, dict) and "y" in pt]
                    xs = [pt["x"] for pt in def_lm.values() if isinstance(pt, dict) and "x" in pt]
                    if ys and xs:
                        target_pt = (int(np.mean(xs) * w), int(min(ys) * h))
            elif strike_type == "KOTE":
                target_pt = CombatantVisualizer._extract_pt(def_lm.get("RIGHT_WRIST"), w, h) or CombatantVisualizer._extract_pt(def_lm.get("RIGHT_ELBOW"), w, h)
            elif strike_type == "DO":
                r_hip = CombatantVisualizer._extract_pt(def_lm.get("RIGHT_HIP"), w, h)
                r_sh = CombatantVisualizer._extract_pt(def_lm.get("RIGHT_SHOULDER"), w, h)
                if r_hip and r_sh:
                    target_pt = ((r_hip[0] + r_sh[0]) // 2, (r_hip[1] + r_sh[1]) // 2)
            elif strike_type == "TSUKI":
                nose = CombatantVisualizer._extract_pt(def_lm.get("NOSE"), w, h)
                sh = CombatantVisualizer._extract_pt(def_lm.get("RIGHT_SHOULDER"), w, h)
                if nose and sh:
                    target_pt = ((nose[0] + sh[0]) // 2, (nose[1] + sh[1]) // 2)

            if target_pt:
                radius = 24 if is_near_impact else 16
                target_col = (50, 220, 50) if is_valid else (0, 165, 255)
                cv2.circle(frame, target_pt, radius, target_col, 2, cv2.LINE_AA)
                cv2.circle(frame, target_pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.line(frame, (target_pt[0] - radius - 4, target_pt[1]), (target_pt[0] + radius + 4, target_pt[1]), target_col, 1, cv2.LINE_AA)
                cv2.line(frame, (target_pt[0], target_pt[1] - radius - 4), (target_pt[0], target_pt[1] + radius + 4), target_col, 1, cv2.LINE_AA)

                lbl_txt = f"ALVO: {strike_type}"
                cv2.putText(frame, lbl_txt, (target_pt[0] - 28, max(20, target_pt[1] - radius - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Banner de Destaque do Golpe no Rodapé do Vídeo
        banner_h = 42
        banner_y = h - banner_h
        overlay = frame.copy()

        bg_color = (15, 60, 20) if is_valid else (20, 30, 60)
        cv2.rectangle(overlay, (0, banner_y), (w, h), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        border_col = (34, 197, 94) if is_valid else (239, 68, 68)
        cv2.line(frame, (0, banner_y), (w, banner_y), border_col, 2)

        valid_str = f"IPPON VÁLIDO ✅ ({score:.1f}%)" if is_valid else f"GOLPE INVÁLIDO ⚠️ ({score:.1f}%)"
        strike_title = f"⚔️ GOLPE: {strike_type} | {atk_name} -> {valid_str}"
        cv2.putText(frame, strike_title, (20, banner_y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def draw_discarded_marker(frame: np.ndarray, landmarks: Dict[str, Any], label: str):
        h, w, _ = frame.shape
        if not landmarks:
            return

        pts_map: Dict[str, Tuple[int, int]] = {}
        for k, v in landmarks.items():
            pt = CombatantVisualizer._extract_pt(v, w, h)
            if pt is not None:
                pts_map[k] = pt

        if not pts_map:
            return

        xs = [p[0] for p in pts_map.values()]
        ys = [p[1] for p in pts_map.values()]

        xmin, xmax = max(0, min(xs)), min(w - 1, max(xs))
        ymin, ymax = max(0, min(ys)), min(h - 1, max(ys))

        # Caixa tracejada/cinza discreta para indicar o descarte do plano
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (100, 116, 139), 1)
        cv2.putText(frame, label, (xmin, max(12, ymin - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)


