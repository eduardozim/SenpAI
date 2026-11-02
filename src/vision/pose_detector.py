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
import contextlib
try:
    import mediapipe as mp
except ImportError:
    mp = None

from typing import Dict, List, Optional, Tuple, Any

from src.utils.settings_manager import get_vision_model, get_vision_model_info

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
    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        device: str = "cpu",
        model_name: Optional[str] = None
    ):
        self.device = device.lower().strip() if device else "cpu"
        self.model_name = (model_name or get_vision_model()).lower().strip()
        self.model_info = get_vision_model_info(self.model_name)
        self.use_gpu = False
        self.yolo_model = None
        self.torch_device = None
        self.torch = None
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
                try:
                    from ultralytics.utils import LOGGER as YOLO_LOGGER
                    YOLO_LOGGER.setLevel(logging.ERROR)
                except Exception:
                    pass
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

                    # Localizar e carregar modelo YOLO selecionado com fallback
                    model_target = self._resolve_model_path_or_name()
                    try:
                        self.yolo_model = YOLO(model_target)
                    except Exception as load_err:
                        logger.warning(
                            f"[PoseDetector] Falha ao carregar modelo '{model_target}': {load_err}. "
                            f"Ativando fallback seguro para YOLOv8-Pose."
                        )
                        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
                        fallback_path = os.path.join(models_dir, "yolov8n-pose.pt")
                        self._create_model_file_from_fallback(model_target, fallback_path)
                        self.yolo_model = YOLO(fallback_path)
                    
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
                    logger.info(
                        f"[PoseDetector] 🚀 Aceleração Nativa NVIDIA CUDA ativada com sucesso: {gpu_name} "
                        f"({self.model_info['name']} FP16 + cuDNN Benchmark)"
                    )
                else:
                    logger.warning("[PoseDetector] GPU solicitada, mas PyTorch CUDA não está disponível. Fallback para CPU.")
            except Exception as e:
                err_str = str(e)
                if "4551" in err_str or "Controle de Aplicativo" in err_str or "Smart App Control" in err_str:
                    logger.warning(
                        "[PoseDetector] 🛡️ O Controle de Aplicativo (Smart App Control) do Windows bloqueou a DLL não-assinada do PyTorch (torch_python.dll). "
                        "Fallback automático para CPU MediaPipe ativado com sucesso."
                    )
                else:
                    logger.warning(f"[PoseDetector] Erro ao inicializar aceleração GPU NVIDIA: {e}. Fallback para CPU MediaPipe.")

        if not self.use_gpu:
            if self.mp_pose is not None:
                logger.info(f"[PoseDetector] Inicializando detector MediaPipe Pose em modo CPU (Modelo preferencial: {self.model_info['name']}).")
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1, # Otimizado para CPU
                    enable_segmentation=False,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
            else:
                logger.info(f"[PoseDetector] MediaPipe Pose indisponível. Inicializando fallback YOLO em CPU ({self.model_info['name']}).")
                try:
                    import torch
                    self.torch = torch
                    from ultralytics import YOLO
                    model_target = self._resolve_model_path_or_name()
                    try:
                        self.yolo_model = YOLO(model_target)
                    except Exception as load_err:
                        logger.warning(f"[PoseDetector] Falha ao carregar '{model_target}' em CPU: {load_err}. Usando yolov8n-pose.pt.")
                        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
                        fallback_path = os.path.join(models_dir, "yolov8n-pose.pt")
                        self._create_model_file_from_fallback(model_target, fallback_path)
                        self.yolo_model = YOLO(fallback_path)
                    self.yolo_model.to("cpu")
                except Exception as e:
                    logger.warning(f"[PoseDetector] Fallback YOLO em CPU indisponível: {e}")

    def _create_model_file_from_fallback(self, model_target: str, fallback_path: str) -> bool:
        """
        No caso de erro ao carregar o modelo (ex: arquivo ausente ou inacessível no disco),
        cria automaticamente o arquivo de modelo no caminho de destino a partir do modelo base seguro
        (yolov8n-pose.pt ou candidato compatível em models/), garantindo persistência e estabilidade.
        """
        try:
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
            # Se fallback_path não existir diretamente, procurar candidato alternativo válido em models/
            if not os.path.exists(fallback_path):
                for cand in ["yolov8n-pose.pt", "yolo11n-pose.pt", "yolo12n-pose.pt", "yolov26n-pose.pt"]:
                    cand_path = os.path.join(models_dir, cand)
                    if os.path.exists(cand_path):
                        fallback_path = cand_path
                        break

            if os.path.exists(fallback_path) and not os.path.exists(model_target):
                import shutil
                os.makedirs(os.path.dirname(model_target), exist_ok=True)
                shutil.copy2(fallback_path, model_target)
                logger.info(
                    f"[PoseDetector] Arquivo de modelo '{model_target}' criado com sucesso a partir de '{fallback_path}'."
                )
                return True
        except Exception as create_err:
            logger.warning(f"[PoseDetector] Não foi possível criar automaticamente o arquivo de modelo '{model_target}': {create_err}")
        return False

    def _resolve_model_path_or_name(self) -> str:
        """
        Localiza o arquivo de pesos do modelo selecionado exclusivamente na pasta models/
        (ex: models/<weights_file>).
        Garante que nenhum modelo seja lido da raiz e que novos downloads do Ultralytics
        sejam salvos estritamente dentro da pasta models/.
        """
        weights_file = self.model_info.get("weights_file", "yolov8n-pose.pt")
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        os.makedirs(models_dir, exist_ok=True)

        target_path = os.path.join(models_dir, weights_file)

        # Salvaguarda de migração: se o arquivo de modelo existir na raiz, mover para models/
        if not os.path.exists(target_path) and os.path.exists(weights_file):
            try:
                import shutil
                shutil.move(weights_file, target_path)
                logger.info(f"[PoseDetector] Modelo '{weights_file}' migrado com sucesso da raiz para '{models_dir}'.")
            except Exception as e:
                logger.warning(f"[PoseDetector] Falha ao migrar '{weights_file}' da raiz: {e}")

        # Se o arquivo de modelo já existe dentro de models/, utilizá-lo
        if os.path.exists(target_path):
            return target_path

        # Se for o baseline yolov8 e existir em models/, retorná-lo
        v8_candidate = os.path.join(models_dir, "yolov8n-pose.pt")
        if self.model_name == "yolov8" and os.path.exists(v8_candidate):
            return v8_candidate

        # Retornar o caminho absoluto dentro da pasta models/
        # Ao receber um caminho contendo a pasta models/, o Ultralytics salvará qualquer download direto em models/
        return target_path

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

                # Rejeitar keypoints não detectados pelo YOLO (confiança baixa ou ponto nulo 0,0)
                if float(conf) < 0.20 or (px <= 3 and py <= 3):
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
            if "RIGHT_ANKLE" not in lm_dict and "LEFT_ANKLE" in lm_dict:
                lm_dict["RIGHT_ANKLE"] = dict(lm_dict["LEFT_ANKLE"])
            elif "LEFT_ANKLE" not in lm_dict and "RIGHT_ANKLE" in lm_dict:
                lm_dict["LEFT_ANKLE"] = dict(lm_dict["RIGHT_ANKLE"])
            elif "RIGHT_ANKLE" not in lm_dict and "RIGHT_HIP" in lm_dict:
                r_hip = lm_dict["RIGHT_HIP"]
                lm_dict["RIGHT_ANKLE"] = {
                    "x": r_hip["x"],
                    "y": min(1.0, r_hip["y"] + 0.35),
                    "z": 0.0,
                    "visibility": 0.5,
                    "px": r_hip.get("px", 0),
                    "py": min(h - 1, int(r_hip.get("py", 0) + 0.35 * h))
                }
                lm_dict["LEFT_ANKLE"] = dict(lm_dict["RIGHT_ANKLE"])

            if "RIGHT_ANKLE" in lm_dict:
                lm_dict["RIGHT_FOOT_INDEX"] = dict(lm_dict["RIGHT_ANKLE"])
                lm_dict["RIGHT_HEEL"] = dict(lm_dict["RIGHT_ANKLE"])
            if "LEFT_ANKLE" in lm_dict:
                lm_dict["LEFT_FOOT_INDEX"] = dict(lm_dict["LEFT_ANKLE"])
                lm_dict["LEFT_HEEL"] = dict(lm_dict["LEFT_ANKLE"])

            # Sintetizar continuidade bimanual para empunhadura do cabo do Shinai (Tsuka)
            r_wrist = lm_dict.get("RIGHT_WRIST")
            l_wrist = lm_dict.get("LEFT_WRIST")
            if r_wrist and not l_wrist:
                syn_l = dict(r_wrist)
                syn_l["x"] = float(np.clip(r_wrist["x"] - 0.02, 0.0, 1.0))
                syn_l["px"] = max(0, r_wrist.get("px", 0) - 10)
                lm_dict["LEFT_WRIST"] = syn_l
            elif l_wrist and not r_wrist:
                syn_r = dict(l_wrist)
                syn_r["x"] = float(np.clip(l_wrist["x"] + 0.02, 0.0, 1.0))
                syn_r["px"] = min(w - 1, l_wrist.get("px", 0) + 10)
                lm_dict["RIGHT_WRIST"] = syn_r

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
            ctx = self.torch.inference_mode() if (self.torch is not None and hasattr(self.torch, "inference_mode")) else contextlib.nullcontext()
            with ctx:
                results = self.yolo_model(frame, device=dev, verbose=False, conf=0.25, imgsz=640, half=self.use_gpu)
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
            pose_lms: Any = getattr(results, "pose_landmarks", None) if results else None
            if pose_lms is not None:
                if self.mp_drawing and self.mp_pose:
                    connections = list(self.mp_pose.POSE_CONNECTIONS) if hasattr(self.mp_pose, "POSE_CONNECTIONS") else None
                    self.mp_drawing.draw_landmarks(
                        annotated_frame,
                        pose_lms,
                        connections,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style() if self.mp_drawing_styles else None
                    )
                landmarks_dict = {}
                lms_list: Any = getattr(pose_lms, "landmark", [])
                for idx, lm in enumerate(lms_list):
                    name = self.mp_pose.PoseLandmark(idx).name if self.mp_pose is not None else f"LANDMARK_{idx}"
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
            ctx = self.torch.inference_mode() if (self.torch is not None and hasattr(self.torch, "inference_mode")) else contextlib.nullcontext()
            with ctx:
                results = self.yolo_model(
                    frames,
                    device=dev,
                    batch=len(frames),
                    verbose=False,
                    conf=0.25,
                    imgsz=640,
                    half=self.use_gpu
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
            ctx = self.torch.inference_mode() if (self.torch is not None and hasattr(self.torch, "inference_mode")) else contextlib.nullcontext()
            with ctx:
                results = self.yolo_model(frame, device=dev, verbose=False, conf=0.25, imgsz=640, half=self.use_gpu)
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
            lms_l: Any = getattr(res_l, "pose_landmarks", None) if res_l else None
            if lms_l is not None:
                lm_l = self._extract_landmarks_dict(lms_l, left_w, h, offset_x=0, offset_y=0, orig_w=w, orig_h=h)
                if not self._is_duplicate(lm_l, candidates):
                    candidates.append(lm_l)

            right_offset = int(w * 0.35)
            right_w = w - right_offset
            right_crop = frame[:, right_offset:]
            frame_rgb_r = cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB)
            res_r = self.pose.process(frame_rgb_r)
            lms_r: Any = getattr(res_r, "pose_landmarks", None) if res_r else None
            if lms_r is not None:
                lm_r = self._extract_landmarks_dict(lms_r, right_w, h, offset_x=right_offset, offset_y=0, orig_w=w, orig_h=h)
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
        lms_list: Any = getattr(pose_landmarks, "landmark", [])
        for idx, lm in enumerate(lms_list):
            name = self.mp_pose.PoseLandmark(idx).name if self.mp_pose is not None else f"LANDMARK_{idx}"
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
        shiro_label: Optional[str] = None,
        show_discarded: bool = False
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
        occupied_rects: List[Tuple[int, int, int, int]] = []
        label_aka = aka_label or "KENSHI AKA (VERMELHO)"
        if aka_landmarks:
            CombatantVisualizer.draw_skeleton(
                out,
                aka_landmarks,
                color=(40, 40, 235) if not aka_is_striking else (0, 0, 255),
                label=f"{label_aka}" + (" [ATAQUE]" if aka_is_striking else ""),
                is_striking=aka_is_striking,
                occupied_label_rects=occupied_rects
            )
            CombatantVisualizer.draw_shinai(out, aka_landmarks, is_striking=aka_is_striking)

        # 2. Desenhar Shiro (Branco/Ciano) e seu Shinai
        label_shiro = shiro_label or "KENSHI SHIRO (BRANCO)"
        if shiro_landmarks:
            CombatantVisualizer.draw_skeleton(
                out,
                shiro_landmarks,
                color=(240, 240, 50) if not shiro_is_striking else (255, 255, 0),
                label=f"{label_shiro}" + (" [ATAQUE]" if shiro_is_striking else ""),
                is_striking=shiro_is_striking,
                occupied_label_rects=occupied_rects
            )
            CombatantVisualizer.draw_shinai(out, shiro_landmarks, is_striking=shiro_is_striking)

        # 3. Desenhar elementos de segundo plano descartados apenas se explicitamente solicitado
        if show_discarded and discarded_items:
            for item in discarded_items:
                lm = item.get("landmarks")
                p_type = item.get("plane_type", "BACKGROUND")
                if lm:
                    if p_type == "SHINPAN":
                        tag = "⚖️ [SHINPAN (ARBITRO)]"
                    elif p_type == "OUT_OF_BOUNDS":
                        tag = "⛔ [FORA DO SHIAI-JO]"
                    elif p_type == "BACKGROUND":
                        tag = "[2º PLANO DESCARTADO]"
                    else:
                        tag = "[OCLUSÃO DESCARTADA]"
                    CombatantVisualizer.draw_discarded_marker(out, lm, label=tag, plane_type=p_type)

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

        return out

    def release(self):
        if hasattr(self, "pose") and self.pose is not None:
            self.pose.close()


class CombatantVisualizer:
    _FONT_CACHE: Dict[int, Any] = {}
    _BADGE_CACHE: Dict[Tuple[str, Tuple[int, int, int], int], np.ndarray] = {}

    @staticmethod
    def _get_ttf_font(font_size: int) -> Any:
        if font_size in CombatantVisualizer._FONT_CACHE:
            return CombatantVisualizer._FONT_CACHE[font_size]
        font_obj = None
        try:
            from PIL import ImageFont
            candidates = [
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/calibrib.ttf",
                "C:/Windows/Fonts/tahoamabd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]
            for fp in candidates:
                if os.path.exists(fp):
                    font_obj = ImageFont.truetype(fp, font_size)
                    break
            if font_obj is None:
                font_obj = ImageFont.load_default()
        except Exception:
            font_obj = None
        CombatantVisualizer._FONT_CACHE[font_size] = font_obj
        return font_obj

    @staticmethod
    def _render_cached_badge_rgba(
        text: str,
        color_bgr: Tuple[int, int, int],
        font_size: int
    ) -> Optional[np.ndarray]:
        """
        Renderiza um badge de alta nitidez (TrueType Bold + contorno escuro + barra lateral colorida)
        em buffer BGRA (H, W, 4) e armazena em cache para aplicação instantânea via NumPy.
        """
        key = (text, color_bgr, font_size)
        if key in CombatantVisualizer._BADGE_CACHE:
            return CombatantVisualizer._BADGE_CACHE[key]

        font = CombatantVisualizer._get_ttf_font(font_size)
        if font is None:
            return None

        try:
            from PIL import Image, ImageDraw
            dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            d_dummy = ImageDraw.Draw(dummy)
            bbox = d_dummy.textbbox((0, 0), text, font=font, stroke_width=2)
            tw = int(max(1, bbox[2] - bbox[0]))
            th = int(max(1, bbox[3] - bbox[1]))

            accent_w = 7
            pad_x = 12
            pad_y = 6
            bw = int(accent_w + pad_x * 2 + tw)
            bh = int(max(font_size + pad_y * 2, th + pad_y * 2))

            # Converter cor BGR para RGB
            r_c, g_c, b_c = int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])

            # Fundo escuro de alto contraste (#0F172A com 92% de opacidade) resistente à subamostragem YUV420p
            img = Image.new("RGBA", (bw, bh), (15, 23, 42, 238))
            draw = ImageDraw.Draw(img)

            # Borda externa colorida (2px) e barra lateral de destaque (accent_w)
            draw.rectangle([(0, 0), (bw - 1, bh - 1)], outline=(r_c, g_c, b_c, 255), width=2)
            draw.rectangle([(0, 0), (accent_w, bh - 1)], fill=(r_c, g_c, b_c, 255))

            # Texto branco puro com contorno preto de 2px para legibilidade máxima em qualquer vídeo
            tx = accent_w + pad_x - bbox[0]
            ty = (bh - th) // 2 - bbox[1]
            draw.text(
                (tx, ty),
                text,
                font=font,
                fill=(255, 255, 255, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255)
            )

            rgba = np.array(img, dtype=np.uint8)
            # Converter RGBA -> BGRA para mesclagem direta no frame OpenCV
            bgra = rgba[:, :, [2, 1, 0, 3]].copy()

            if len(CombatantVisualizer._BADGE_CACHE) > 512:
                CombatantVisualizer._BADGE_CACHE.clear()
            CombatantVisualizer._BADGE_CACHE[key] = bgra
            return bgra
        except Exception:
            return None

    @staticmethod
    def _blend_bgra_patch(frame: np.ndarray, bgra: np.ndarray, x: int, y: int) -> Tuple[int, int, int, int]:
        """Mescla um patch BGRA no frame BGR nas coordenadas (x, y) com clip seguro."""
        fh, fw = frame.shape[:2]
        bh, bw = bgra.shape[:2]

        x1 = int(np.clip(x, 0, max(0, fw - bw)))
        y1 = int(np.clip(y, 0, max(0, fh - bh)))
        x2 = min(fw, x1 + bw)
        y2 = min(fh, y1 + bh)

        pw = x2 - x1
        ph = y2 - y1
        if pw <= 0 or ph <= 0:
            return (x1, y1, x1, y1)

        patch = bgra[:ph, :pw]
        alpha = (patch[:, :, 3:4].astype(np.float32)) / 255.0
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        blended = patch[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return (x1, y1, x2, y2)

    @staticmethod
    def draw_crisp_badge(
        frame: np.ndarray,
        text: str,
        anchor_x: int,
        anchor_y: int,
        color_bgr: Tuple[int, int, int],
        occupied_rects: Optional[List[Tuple[int, int, int, int]]] = None,
        target_box: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[int, int, int, int]:
        """
        Desenha uma etiqueta de identificação de alta definição para um Kenshi,
        com resolução automática de colisão vertical quando vários Kenshis estão próximos.
        """
        fh, fw = frame.shape[:2]
        font_size = max(16, round(max(fw, fh) * 0.0155))
        bgra = CombatantVisualizer._render_cached_badge_rgba(text, color_bgr, font_size)

        if bgra is None:
            # Fallback OpenCV de alto contraste caso PIL não esteja disponível
            scale = max(0.62, max(fw, fh) / 1600.0)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
            bw, bh = tw + 24, th + 16
            x1 = int(np.clip(anchor_x, 4, max(4, fw - bw - 4)))
            y1 = int(np.clip(anchor_y - bh - 4, 58, max(58, fh - bh - 4)))
            x2, y2 = x1 + bw, y1 + bh
            cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 23, 42), -1)
            cv2.rectangle(frame, (x1, y1), (x1 + 7, y2), color_bgr, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, text, (x1 + 13, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (x1 + 13, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
            rect = (x1, y1, x2, y2)
            if occupied_rects is not None:
                occupied_rects.append(rect)
            return rect

        bh, bw = bgra.shape[:2]
        min_top_y = max(56, int(fh * 0.065))  # Evita sobrepor o banner superior
        cand_x = int(np.clip(anchor_x, 6, max(6, fw - bw - 6)))
        cand_y = int(np.clip(anchor_y - bh - 6, min_top_y, max(min_top_y, fh - bh - 6)))

        def _overlaps(r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int], pad: int = 4) -> bool:
            return not (
                r1[2] + pad < r2[0]
                or r1[0] - pad > r2[2]
                or r1[3] + pad < r2[1]
                or r1[1] - pad > r2[3]
            )

        if occupied_rects:
            # Testar deslocamentos verticais (acima e abaixo) para evitar sobreposição entre Kenshis próximos
            step = bh + 6
            offsets = [0, -step, step, -2 * step, 2 * step, -3 * step, 3 * step]
            for off_y in offsets:
                ty = int(np.clip(cand_y + off_y, min_top_y, max(min_top_y, fh - bh - 6)))
                test_rect = (cand_x, ty, cand_x + bw, ty + bh)
                if not any(_overlaps(test_rect, occ) for occ in occupied_rects):
                    cand_y = ty
                    break

        placed_rect = CombatantVisualizer._blend_bgra_patch(frame, bgra, cand_x, cand_y)
        if occupied_rects is not None:
            occupied_rects.append(placed_rect)

        # Se a etiqueta precisou ser deslocada da caixa original do Kenshi, desenha uma linha guia discreta
        if target_box is not None:
            bx1, by1, bx2, by2 = target_box
            px1, py1, px2, py2 = placed_rect
            if abs(py2 - by1) > 14:
                lead_start = (min(max(px1 + 16, bx1 + 8), px2 - 8), py2 if py2 < by1 else py1)
                lead_end = (min(max(px1 + 16, bx1 + 8), bx2 - 8), by1)
                cv2.line(frame, lead_start, lead_end, (15, 23, 42), 4, cv2.LINE_AA)
                cv2.line(frame, lead_start, lead_end, color_bgr, 2, cv2.LINE_AA)

        return placed_rect

    @staticmethod
    def draw_top_hud_banner(frame: np.ndarray, text: str) -> None:
        """
        Renderiza a barra de status superior (HUD) com fonte TrueType nítida e fundo escuro de alto contraste.
        """
        fh, fw = frame.shape[:2]
        banner_h = max(44, round(fh * 0.056))
        margin_x = 10
        margin_y = 8

        # Fundo translúcido escuro no topo
        x1, y1 = margin_x, margin_y
        x2, y2 = fw - margin_x, margin_y + banner_h
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        bg = np.full_like(roi, (15, 23, 42), dtype=np.float32)
        frame[y1:y2, x1:x2] = (bg * 0.88 + roi * 0.12).astype(np.uint8)

        # Borda e barra de destaque lateral Indigo (#6366F1 -> BGR 241, 102, 99)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (241, 102, 99), 2)
        cv2.rectangle(frame, (x1, y1), (x1 + 8, y2), (241, 102, 99), -1)

        font_size = max(16, round(banner_h * 0.44))
        bgra = CombatantVisualizer._render_cached_badge_rgba(text, (241, 102, 99), font_size)
        if bgra is not None:
            bh, bw = bgra.shape[:2]
            ty = y1 + max(2, (banner_h - bh) // 2)
            CombatantVisualizer._blend_bgra_patch(frame, bgra, x1 + 12, ty)
        else:
            scale = max(0.60, banner_h / 72.0)
            cv2.putText(frame, text, (x1 + 18, y1 + int(banner_h * 0.68)), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x1 + 18, y1 + int(banner_h * 0.68)), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _extract_pt(pt_data: Any, w: int, h: int) -> Optional[Tuple[int, int]]:
        """
        Extrai coordenadas em pixels de um landmark adaptadas à resolução atual (w, h) do frame.
        Prioriza coordenadas normalizadas (x, y em 0..1) quando disponíveis para permitir
        renderização nítida em canvas redimensionado (HD Upscaling), com fallback para px/py.
        """
        if not isinstance(pt_data, dict):
            return None
        vis = pt_data.get("visibility", 1.0)
        if vis is not None and float(vis) < 0.15:
            return None
        if "x" in pt_data and "y" in pt_data and pt_data["x"] is not None and pt_data["y"] is not None:
            fx, fy = float(pt_data["x"]), float(pt_data["y"])
            if 0.0 < fx <= 1.05 and 0.0 < fy <= 1.05:
                px = int(np.clip(round(fx * w), 0, w - 1))
                py = int(np.clip(round(fy * h), 0, h - 1))
                if px <= 3 and py <= 3:
                    return None
                return px, py
        if "px" in pt_data and "py" in pt_data and pt_data["px"] is not None and pt_data["py"] is not None:
            px, py = int(pt_data["px"]), int(pt_data["py"])
            if px <= 3 and py <= 3:
                return None
            return px, py
        return None

    @staticmethod
    def draw_skeleton(
        frame: np.ndarray,
        landmarks: Dict[str, Any],
        color: Tuple[int, int, int],
        label: str,
        is_striking: bool = False,
        occupied_label_rects: Optional[List[Tuple[int, int, int, int]]] = None
    ):
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

        pad_box = max(14, round(max(w, h) * 0.012))
        xmin, xmax = max(0, min(xs) - pad_box), min(w - 1, max(xs) + pad_box)
        ymin, ymax = max(0, min(ys) - pad_box), min(h - 1, max(ys) + pad_box)

        # Caixa delimitadora com halo escuro externo + cantoneiras reforçadas para contraste no dojo
        box_thickness = 3 if is_striking else 2
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (15, 23, 42), box_thickness + 2, cv2.LINE_AA)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, box_thickness, cv2.LINE_AA)

        corner_len = max(10, min(24, (xmax - xmin) // 4, (ymax - ymin) // 4))
        c_thick = box_thickness + 2
        for cx, cy, dx, dy in [
            (xmin, ymin, 1, 1),
            (xmax, ymin, -1, 1),
            (xmin, ymax, 1, -1),
            (xmax, ymax, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, c_thick, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, c_thick, cv2.LINE_AA)

        # Desenhar articulações chave com halo escuro para destacar sobre Hakama/Bogu ou piso claro
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

        bone_thickness = max(2, round(max(w, h) / 480.0)) + (1 if is_striking else 0)
        for p1_name, p2_name in connections:
            if p1_name in pts_map and p2_name in pts_map:
                cv2.line(frame, pts_map[p1_name], pts_map[p2_name], (15, 23, 42), bone_thickness + 2, cv2.LINE_AA)
                cv2.line(frame, pts_map[p1_name], pts_map[p2_name], color, bone_thickness, cv2.LINE_AA)

        joint_radius = max(3, round(max(w, h) / 360.0)) + (1 if is_striking else 0)
        for pt in pts_map.values():
            cv2.circle(frame, pt, joint_radius + 1, (15, 23, 42), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, joint_radius, (255, 255, 255), -1, cv2.LINE_AA)

        # Destaque de Mãos / Empunhadura (Kote / Tsuka)
        r_w = pts_map.get("RIGHT_WRIST")
        l_w = pts_map.get("LEFT_WRIST")
        for hand_pt in [r_w, l_w]:
            if hand_pt:
                cv2.circle(frame, hand_pt, joint_radius + 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, hand_pt, joint_radius + 4, (0, 215, 255) if not is_striking else (0, 69, 255), 2, cv2.LINE_AA)

        # Etiqueta de identificação do Kenshi em alta definição (TrueType Bold + anti-colisão)
        CombatantVisualizer.draw_crisp_badge(
            frame,
            text=label,
            anchor_x=xmin,
            anchor_y=ymin,
            color_bgr=color,
            occupied_rects=occupied_label_rects,
            target_box=(xmin, ymin, xmax, ymax)
        )

    @staticmethod
    def draw_shinai(frame: np.ndarray, landmarks: Dict[str, Any], is_striking: bool = False):
        """
        Desenha a espada Shinai na tela.
        Se os dados visuais/rastreados estiverem presentes em landmarks['SHINAI'],
        utiliza o Kensen e a base reais. Caso contrário, projeta cinemática a partir dos pulsos.
        """
        h, w, _ = frame.shape
        shinai_data = landmarks.get("SHINAI") if landmarks else None

        base_pt = None
        tip_pt = None
        is_visual = False

        if shinai_data and isinstance(shinai_data, dict) and shinai_data.get("detected"):
            b_norm = shinai_data.get("base_norm")
            t_norm = shinai_data.get("tip_norm")
            b_px = shinai_data.get("base_px")
            t_px = shinai_data.get("tip_px")
            if b_norm and t_norm:
                base_pt = (int(np.clip(round(float(b_norm[0]) * w), 0, w - 1)), int(np.clip(round(float(b_norm[1]) * h), 0, h - 1)))
                tip_pt = (int(np.clip(round(float(t_norm[0]) * w), 0, w - 1)), int(np.clip(round(float(t_norm[1]) * h), 0, h - 1)))
                is_visual = bool(shinai_data.get("is_visual", False))
            elif b_px and t_px:
                base_pt = (int(b_px[0]), int(b_px[1]))
                tip_pt = (int(t_px[0]), int(t_px[1]))
                is_visual = bool(shinai_data.get("is_visual", False))

        # Fallback cinemático se não houver dados em landmarks['SHINAI']
        if base_pt is None or tip_pt is None:
            r_wrist = CombatantVisualizer._extract_pt(landmarks.get("RIGHT_WRIST"), w, h)
            l_wrist = CombatantVisualizer._extract_pt(landmarks.get("LEFT_WRIST"), w, h)
            hand_pt = r_wrist if r_wrist is not None else l_wrist
            if not hand_pt:
                return

            base_pt = hand_pt
            r_elbow = CombatantVisualizer._extract_pt(landmarks.get("RIGHT_ELBOW"), w, h)
            l_elbow = CombatantVisualizer._extract_pt(landmarks.get("LEFT_ELBOW"), w, h)
            elbow_pt = r_elbow if r_elbow is not None else l_elbow

            if elbow_pt:
                dx = float(hand_pt[0] - elbow_pt[0])
                dy = float(hand_pt[1] - elbow_pt[1])
                norm = np.hypot(dx, dy)
                if norm > 5:
                    shinai_len = max(38, int(norm * 1.40))
                    tip_x = int(hand_pt[0] + (dx / norm) * shinai_len)
                    tip_y = int(hand_pt[1] + (dy / norm) * shinai_len)
                else:
                    tip_x, tip_y = hand_pt[0] + 30, hand_pt[1] - 45
            else:
                tip_x, tip_y = hand_pt[0] + 30, hand_pt[1] - 45

            tip_pt = (int(np.clip(tip_x, 0, w - 1)), int(np.clip(tip_y, 0, h - 1)))

        sword_color = (0, 215, 255) if not is_striking else (0, 69, 255) # Ouro ou Vermelho Neon
        blade_thickness = 4 if is_visual else (3 if not is_striking else 5)

        # 1. Empunhadura (Tsuka - cabo da espada)
        # Segmento inicial do cabo em branco destacando as mãos do Kendoca
        tsuka_end_x = int(base_pt[0] + 0.20 * (tip_pt[0] - base_pt[0]))
        tsuka_end_y = int(base_pt[1] + 0.20 * (tip_pt[1] - base_pt[1]))
        cv2.line(frame, base_pt, (tsuka_end_x, tsuka_end_y), (255, 255, 255), max(2, blade_thickness - 1), cv2.LINE_AA)

        # 2. Haste do Shinai (Take / Lâmina de Bambu)
        cv2.line(frame, (tsuka_end_x, tsuka_end_y), tip_pt, sword_color, blade_thickness, cv2.LINE_AA)
        if is_visual:
            # Filete central branco (Tsuru / corda do Shinai)
            cv2.line(frame, (tsuka_end_x, tsuka_end_y), tip_pt, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Kensen (Ponta da espada / Sakigawa)
        cv2.circle(frame, tip_pt, 4 if not is_striking else 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, tip_pt, 6 if not is_striking else 9, sword_color, 2, cv2.LINE_AA)

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
    def draw_discarded_marker(
        frame: np.ndarray,
        landmarks: Dict[str, Any],
        label: str,
        plane_type: str = "BACKGROUND"
    ):
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

        if plane_type == "SHINPAN":
            # Borda âmbar/dourada distinta e etiqueta para o árbitro (Shinpan)
            border_color = (30, 160, 240)
            text_color = (50, 200, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), border_color, 2)
            cv2.putText(frame, label, (xmin, max(14, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_color, 1, cv2.LINE_AA)
        else:
            # Caixa cinza discreta para indicar o descarte do plano / fora de quadra
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (100, 116, 139), 1)
            cv2.putText(frame, label, (xmin, max(12, ymin - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)


