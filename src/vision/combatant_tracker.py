"""
Módulo de Rastreamento dos 2 Combatentes Principais e Filtragem de Planos.
Identifica e rastreia os 2 Kenshi (Aka & Shiro) que realizaram o Sonkyō inicial de abertura,
descartando elementos de segundo plano (outras lutas/fundo), transeuntes de primeiro plano (frente da câmera),
árbitros (Shinpans) e detecções fora da área regulamentar do Shiai-jo.
Elimina inversões espúrias através de associação bipartida inercial contínua, barreira de histerese anti-swap
e ancoragem geométrica via Shinai.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.vision.shinai_tracker import ShinaiTracker


class CombatantProfile:
    def __init__(self, combatant_id: str, name: str, color_bgr: Tuple[int, int, int]):
        self.id = combatant_id       # "KENSHI_AKA" ou "KENSHI_SHIRO"
        self.name = name             # "Kenshi Aka (Vermelho)" ou "Kenshi Shiro (Branco)"
        self.color_bgr = color_bgr   # (B, G, R)
        self.last_center_x: Optional[float] = None
        self.last_center_y: Optional[float] = None
        self.last_bbox: Optional[Tuple[float, float, float, float]] = None # (xmin, ymin, xmax, ymax)
        self.last_valid_landmarks: Optional[Dict[str, Any]] = None
        self.history: List[Optional[Dict[str, Any]]] = []
        self.shinai: Optional[Dict[str, Any]] = None
        self.facing: Optional[str] = None

        # Predição de Movimento e Resiliência Temporal (Filtro Inercial / Kalman simplificado)
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.pred_x: Optional[float] = None
        self.pred_y: Optional[float] = None
        self.occluded_frames: int = 0
        self.is_locked: bool = False

    def update(self, landmarks: Optional[Dict[str, Any]]):
        if landmarks:
            cx, cy, bbox = CombatantTracker.extract_bbox_and_center(landmarks)
            if self.last_center_x is not None and self.last_center_y is not None:
                # Atualização suave da velocidade vetorial
                self.vx = 0.60 * self.vx + 0.40 * (cx - self.last_center_x)
                self.vy = 0.60 * self.vy + 0.40 * (cy - self.last_center_y)

            # Persistência anatômica dos pulsos/mãos (Kote/Shinai grip)
            if self.last_valid_landmarks:
                dx = cx - (self.last_center_x if self.last_center_x is not None else cx)
                dy = cy - (self.last_center_y if self.last_center_y is not None else cy)
                for wrist_key in ["RIGHT_WRIST", "LEFT_WRIST"]:
                    if wrist_key not in landmarks and wrist_key in self.last_valid_landmarks:
                        prev_w = self.last_valid_landmarks[wrist_key]
                        if isinstance(prev_w, dict) and "x" in prev_w and "y" in prev_w:
                            syn_w = dict(prev_w)
                            syn_w["x"] = float(np.clip(prev_w["x"] + dx, 0.0, 1.0))
                            syn_w["y"] = float(np.clip(prev_w["y"] + dy, 0.0, 1.0))
                            if "px" in prev_w and "py" in prev_w:
                                syn_w["px"] = int(prev_w["px"] + dx * 640)
                                syn_w["py"] = int(prev_w["py"] + dy * 480)
                            syn_w["visibility"] = float(prev_w.get("visibility", 0.8) * 0.9)
                            landmarks[wrist_key] = syn_w

                for foot_key in ["RIGHT_ANKLE", "LEFT_ANKLE", "RIGHT_FOOT_INDEX", "LEFT_FOOT_INDEX", "RIGHT_HEEL", "LEFT_HEEL"]:
                    if foot_key not in landmarks and foot_key in self.last_valid_landmarks:
                        prev_f = self.last_valid_landmarks[foot_key]
                        if isinstance(prev_f, dict) and "x" in prev_f and "y" in prev_f:
                            syn_f = dict(prev_f)
                            syn_f["x"] = float(np.clip(prev_f["x"] + dx, 0.0, 1.0))
                            syn_f["y"] = float(np.clip(prev_f["y"] + dy, 0.0, 1.0))
                            if "px" in prev_f and "py" in prev_f:
                                syn_f["px"] = int(prev_f["px"] + dx * 640)
                                syn_f["py"] = int(prev_f["py"] + dy * 480)
                            syn_f["visibility"] = float(prev_f.get("visibility", 0.8) * 0.9)
                            landmarks[foot_key] = syn_f

            if "SHINAI" in landmarks:
                self.shinai = landmarks["SHINAI"]
            if "FACING" in landmarks:
                self.facing = landmarks["FACING"]

            self.last_center_x = cx
            self.last_center_y = cy
            self.last_bbox = bbox
            self.last_valid_landmarks = landmarks
            self.pred_x = float(np.clip(cx + self.vx, 0.0, 1.0))
            self.pred_y = float(np.clip(cy + self.vy, 0.0, 1.0))
            self.occluded_frames = 0
            self.history.append(landmarks)
        else:
            self.occluded_frames += 1
            if self.pred_x is not None and self.pred_y is not None:
                # Projeção inercial amortecida durante oclusão
                self.pred_x = float(np.clip(self.pred_x + self.vx * 0.70, 0.0, 1.0))
                self.pred_y = float(np.clip(self.pred_y + self.vy * 0.70, 0.0, 1.0))
            self.history.append(None)

    def get_persisted_landmarks(self, max_gap: int = 6) -> Optional[Dict[str, Any]]:
        """
        Retorna landmarks persistidos/projetados inercialmente durante pequenas falhas
        temporárias de detecção (motion blur, oclusão parcial temporária de até max_gap frames).
        Garante que o esqueleto, mãos e Shinai não pisquem ou sumam do Kendoca.
        """
        if self.history and self.history[-1] is not None:
            return self.history[-1]

        if self.last_valid_landmarks and self.occluded_frames <= max_gap:
            decay = 0.85 ** self.occluded_frames
            dx = self.vx * decay
            dy = self.vy * decay
            projected: Dict[str, Any] = {}
            for k, v in self.last_valid_landmarks.items():
                if isinstance(v, dict) and "x" in v and "y" in v:
                    p_copy = dict(v)
                    p_copy["x"] = float(np.clip(v["x"] + dx, 0.0, 1.0))
                    p_copy["y"] = float(np.clip(v["y"] + dy, 0.0, 1.0))
                    if "px" in v and "py" in v and v["px"] is not None and v["py"] is not None:
                        p_copy["px"] = int(v["px"] + dx * 640)
                        p_copy["py"] = int(v["py"] + dy * 480)
                    projected[k] = p_copy
                else:
                    projected[k] = v

            if self.shinai:
                projected["SHINAI"] = dict(self.shinai)
            if self.facing:
                projected["FACING"] = self.facing

            return projected

        return None


class CombatantTracker:
    def __init__(
        self,
        min_background_scale_ratio: float = 0.68,
        max_foreground_scale_ratio: float = 1.38,
        ground_line_tolerance: float = 0.14,
        invert_assignment: bool = False,
        shiaijo_polygon: Optional[List[Tuple[float, float]]] = None,
        min_kenshi_score: float = 0.25,
        track_buffer: int = 60,
        lock_tracks: bool = True
    ):
        self.min_bg_ratio = min_background_scale_ratio
        self.max_fg_ratio = max_foreground_scale_ratio
        self.ground_tolerance = ground_line_tolerance
        self.invert_assignment = invert_assignment
        self.shiaijo_polygon = shiaijo_polygon
        self.min_kenshi_score = min_kenshi_score
        self.track_buffer = track_buffer
        self.lock_tracks = lock_tracks

        # Módulo de Shinai dedicado para ancoragem postural
        self.shinai_tracker = ShinaiTracker()

        # Perfis dos 2 lutadores
        self.aka = CombatantProfile("KENSHI_AKA", "Kenshi Aka (Vermelho)", (40, 40, 230))
        self.shiro = CombatantProfile("KENSHI_SHIRO", "Kenshi Shiro (Branco)", (240, 240, 240))

        # Evidência da fita vermelha (Tasuki) acumulada exclusivamente na inicialização
        self.candidate_left_red_score = 0.0
        self.candidate_right_red_score = 0.0
        self.red_evidence_frames_left = 0
        self.red_evidence_frames_right = 0
        self.flag_decision = "POSITION_DEFAULT_OPPOSITE_JUDGES"
        self.flag_confidence = 0.50

        # Barreira de histerese anti-swap (impede trocas espúrias de lado)
        self.pending_swap_frames = 0
        self.SWAP_CONFIRMATION_THRESHOLD = 8   # Frames consecutivos necessários para confirmar crossover real
        self.SWAP_HYSTERESIS_MARGIN = 0.25     # Custo adicional exigido para superar a inércia do track atual

        # Estado do Sistema de Rastreamento (K=2)
        self.tracking_state = "UNINITIALIZED" # "UNINITIALIZED", "LOCKED_COMBAT"
        self.is_calibrated = False
        self.ref_height = 0.42
        self.ref_bbox_area = 0.08
        self.ref_ground_y = 0.78
        self.ref_shoulder_width = 0.12
        # Rastreamento formal dos 3 Shinpans (Árbitros)
        self.shinpans: Dict[str, Optional[Dict[str, Any]]] = {
            "SHINPAN_LEFT": None,
            "SHINPAN_CENTER": None,
            "SHINPAN_RIGHT": None
        }
        self.discarded_shinpan_count = 0

        # Contadores estatísticos
        self.discarded_background_count = 0
        self.discarded_foreground_count = 0
        self.discarded_out_of_shiaijo_count = 0
        self.discarded_shinpan_posture_count = 0
        self.total_detections_processed = 0
        self.occlusion_recovery_count = 0

    def is_within_shiaijo(self, ground_x: float, ground_y: float) -> bool:
        """Verifica se o ponto de solo está estritamente contido no polígono do Shiai-jo."""
        if not self.shiaijo_polygon or len(self.shiaijo_polygon) < 3:
            # Margem padrão de segurança da quadra ativa de combate: Kenshis combatem entre x in [0.18, 0.82] e y in [0.20, 0.98]
            return 0.18 <= ground_x <= 0.82 and 0.20 <= ground_y <= 0.98

        poly = np.array(self.shiaijo_polygon, dtype=np.float32)
        res = cv2.pointPolygonTest(poly, (ground_x, ground_y), False)
        return res >= 0

    @staticmethod
    def detect_red_flag_score(frame: Optional[np.ndarray], landmarks: Optional[Dict[str, Any]]) -> float:
        """
        Analisa a presença da fita vermelha (Aka Tasuki) nas costas/tronco do praticante.
        Retorna score de 0.0 a 1.0 (densidade de vermelho na ROI dorsal).
        """
        if frame is None or not landmarks:
            return 0.0

        h, w = frame.shape[:2]
        shoulder_pts = [landmarks[k] for k in ["LEFT_SHOULDER", "RIGHT_SHOULDER"] if k in landmarks and isinstance(landmarks[k], dict)]
        hip_pts = [landmarks[k] for k in ["LEFT_HIP", "RIGHT_HIP"] if k in landmarks and isinstance(landmarks[k], dict)]

        if not shoulder_pts and not hip_pts:
            _, _, (xmin, ymin, xmax, ymax) = CombatantTracker.extract_bbox_and_center(landmarks)
            top_y = ymin + (ymax - ymin) * 0.20
            bottom_y = ymin + (ymax - ymin) * 0.65
            left_x = xmin
            right_x = xmax
        else:
            top_y = min([p["y"] for p in shoulder_pts]) if shoulder_pts else (min([p["y"] for p in hip_pts]) - 0.25)
            bottom_y = max([p["y"] for p in hip_pts]) if hip_pts else (max([p["y"] for p in shoulder_pts]) + 0.35)
            all_xs = [p["x"] for p in (shoulder_pts + hip_pts)]
            left_x = min(all_xs) - 0.03
            right_x = max(all_xs) + 0.03

        px_min = max(0, int(left_x * w))
        px_max = min(w, int(right_x * w))
        py_min = max(0, int(top_y * h))
        py_max = min(h, int(bottom_y * h))

        if px_max - px_min < 5 or py_max - py_min < 5:
            return 0.0

        roi = frame[py_min:py_max, px_min:px_max]
        if roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 75], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
        mask2 = cv2.inRange(hsv, np.array([168, 100, 75], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return 0.0

        red_ratio = red_pixels / float(total_pixels)
        # Rejeitar ruído de piso/verniz/cadeiras (< 2.5% da área do tronco)
        if red_ratio < 0.025:
            return 0.0
        score = float(np.clip((red_ratio - 0.025) / 0.08, 0.0, 1.0))
        return score

    @staticmethod
    def extract_bbox_and_center(landmarks: Optional[Dict[str, Any]]) -> Tuple[float, float, Tuple[float, float, float, float]]:
        """Extrai centro (cx, cy) e bounding box normalizado (xmin, ymin, xmax, ymax)."""
        if not landmarks:
            return 0.5, 0.5, (0.4, 0.4, 0.6, 0.6)

        xs = [pt["x"] for pt in landmarks.values() if isinstance(pt, dict) and "x" in pt]
        ys = [pt["y"] for pt in landmarks.values() if isinstance(pt, dict) and "y" in pt]

        if not xs or not ys:
            return 0.5, 0.5, (0.4, 0.4, 0.6, 0.6)

        xmin, xmax = float(min(xs)), float(max(xs))
        ymin, ymax = float(min(ys)), float(max(ys))
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        return cx, cy, (xmin, ymin, xmax, ymax)

    @staticmethod
    def interpolate_missing_poses(
        timeline: List[Optional[Dict[str, Any]]],
        max_gap: int = 10
    ) -> List[Optional[Dict[str, Any]]]:
        """Interpola linearmente poses e membros ausentes em gaps temporários."""
        n = len(timeline)
        result = list(timeline)
        i = 0
        while i < n:
            if result[i] is None:
                prev_idx = i - 1
                next_idx = i
                while next_idx < n and result[next_idx] is None:
                    next_idx += 1
                if prev_idx >= 0 and next_idx < n and (next_idx - prev_idx) <= max_gap:
                    p1 = result[prev_idx]
                    p2 = result[next_idx]
                    if p1 and p2:
                        for mid in range(prev_idx + 1, next_idx):
                            alpha = (mid - prev_idx) / float(next_idx - prev_idx)
                            interp: Dict[str, Any] = {}
                            all_keys = set(p1.keys()).union(set(p2.keys()))
                            for k in all_keys:
                                pt1 = p1.get(k)
                                pt2 = p2.get(k)
                                if isinstance(pt1, dict) and isinstance(pt2, dict) and "x" in pt1 and "x" in pt2:
                                    interp[k] = {
                                        "x": float((1.0 - alpha) * pt1["x"] + alpha * pt2["x"]),
                                        "y": float((1.0 - alpha) * pt1["y"] + alpha * pt2["y"]),
                                        "z": 0.0,
                                        "visibility": float(min(pt1.get("visibility", 0.8), pt2.get("visibility", 0.8))),
                                        "px": int((1.0 - alpha) * pt1.get("px", 0) + alpha * pt2.get("px", 0)),
                                        "py": int((1.0 - alpha) * pt1.get("py", 0) + alpha * pt2.get("py", 0)),
                                    }
                                elif isinstance(pt1, dict):
                                    interp[k] = dict(pt1)
                                elif isinstance(pt2, dict):
                                    interp[k] = dict(pt2)
                            if "SHINAI" in p1:
                                interp["SHINAI"] = dict(p1["SHINAI"])
                            result[mid] = interp
                i = next_idx
            else:
                i += 1
        return result

    @staticmethod
    def get_skeleton_metrics(landmarks: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de escala, altura, área e linha dos pés de um conjunto de landmarks."""
        _, _, (xmin, ymin, xmax, ymax) = CombatantTracker.extract_bbox_and_center(landmarks)
        w = max(0.01, xmax - xmin)
        h = max(0.01, ymax - ymin)
        area = w * h

        if "RIGHT_ANKLE" in landmarks and "LEFT_ANKLE" in landmarks:
            ground_y = (landmarks["RIGHT_ANKLE"]["y"] + landmarks["LEFT_ANKLE"]["y"]) / 2.0
            ground_x = (landmarks["RIGHT_ANKLE"]["x"] + landmarks["LEFT_ANKLE"]["x"]) / 2.0
        elif "RIGHT_FOOT_INDEX" in landmarks:
            ground_y = landmarks["RIGHT_FOOT_INDEX"]["y"]
            ground_x = landmarks["RIGHT_FOOT_INDEX"]["x"]
        else:
            ground_y = ymax
            ground_x = (xmin + xmax) / 2.0

        if "RIGHT_SHOULDER" in landmarks and "LEFT_SHOULDER" in landmarks:
            shoulder_w = abs(landmarks["RIGHT_SHOULDER"]["x"] - landmarks["LEFT_SHOULDER"]["x"])
        else:
            shoulder_w = w * 0.45

        return {
            "width": w,
            "height": h,
            "area": area,
            "ground_x": ground_x,
            "ground_y": ground_y,
            "shoulder_width": shoulder_w,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax
        }

    def compute_kenshi_feature_score(self, landmarks: Optional[Dict[str, Any]], frame: Optional[np.ndarray] = None) -> float:
        """
        Calcula a probabilidade (0.0 a 1.0) de um esqueleto detectado ser um Kenshi (lutador)
        em vez de um Shinpan (árbitro de Kendo segurando bandeiras) ou espectador ao fundo.
        """
        if not landmarks:
            return 0.0

        cx, cy, (xmin, ymin, xmax, ymax) = CombatantTracker.extract_bbox_and_center(landmarks)
        h = max(0.01, ymax - ymin)
        m = self.get_skeleton_metrics(landmarks)
        score = 0.35

        # 1. Proximidade dos pulsos (Empunhadura bimanual do Shinai - marcador definitivo de Kenshi vs Shinpan)
        r_wrist = landmarks.get("RIGHT_WRIST")
        l_wrist = landmarks.get("LEFT_WRIST")
        if r_wrist and l_wrist and "x" in r_wrist and "x" in l_wrist:
            wrist_dist = np.hypot(r_wrist["x"] - l_wrist["x"], r_wrist["y"] - l_wrist["y"])
            if wrist_dist < (0.16 * h):
                score += 0.40
            elif wrist_dist < (0.26 * h):
                score += 0.20
            elif wrist_dist > (0.40 * h):
                score -= 0.35  # Árbitros com braços soltos ou segurando bandeiras separadas

        # 2. Centralidade horizontal no Shiaijo
        dist_center_x = abs(cx - 0.50)
        if dist_center_x <= 0.28:
            score += 0.20 * (1.0 - (dist_center_x / 0.28))
        elif dist_center_x >= 0.42:
            score -= 0.30

        # 3. Profundidade de solo (Solo de combate dos Kenshis fica tipicamente entre 0.65 e 0.92)
        if m["ground_y"] < 0.62:
            score -= 0.40
        elif 0.68 <= m["ground_y"] <= 0.92:
            score += 0.20

        # 4. Elevação de braços para corte (Furikaburi / Shinai elevado)
        r_sh = landmarks.get("RIGHT_SHOULDER")
        if r_wrist and r_sh and "y" in r_wrist and "y" in r_sh:
            if r_wrist["y"] <= r_sh["y"]:
                score += 0.25

        # 5. Flexão de joelhos / Agachamento de Sonkyō
        r_hip = landmarks.get("RIGHT_HIP")
        r_knee = landmarks.get("RIGHT_KNEE")
        r_ankle = landmarks.get("RIGHT_ANKLE")
        if r_hip and r_knee and r_ankle and "y" in r_hip and "y" in r_knee and "y" in r_ankle:
            leg_span = abs(r_ankle["y"] - r_hip["y"])
            if leg_span < (0.38 * h):
                score += 0.30

        return float(np.clip(score, 0.0, 1.0))

    def calibrate_main_plane(self, candidate_poses: List[Dict[str, Any]]):
        """Calibra as métricas de referência do Plano Principal com base nas poses dos dois lutadores."""
        if not candidate_poses:
            return

        valid_metrics = [self.get_skeleton_metrics(p) for p in candidate_poses if p]
        if not valid_metrics:
            return

        heights = [m["height"] for m in valid_metrics]
        avg_h = float(np.mean(heights)) if heights else 0.42
        # Bounded calibration to prevent foreground arbiters from inflating the plane
        # and ensure standing up from Sonkyo never triggers scale_h > 1.35
        self.ref_height = max(0.38, float(np.clip(avg_h, 0.25, 0.60)))
        self.ref_bbox_area = float(np.clip(np.mean([m["area"] for m in valid_metrics]), 0.05, 0.18))
        self.ref_ground_y = float(np.clip(np.mean([m["ground_y"] for m in valid_metrics]), 0.68, 0.88))
        self.ref_shoulder_width = float(np.mean([m["shoulder_width"] for m in valid_metrics]))
        self.is_calibrated = True

    def classify_shinpan(self, landmarks: Optional[Dict[str, Any]], frame: Optional[np.ndarray] = None) -> Tuple[bool, str, float]:
        """
        Identifica se um esqueleto detectado pertence a um dos 3 Shinpans (árbitros oficiais).
        Critérios:
        1. Mãos separadas portando bandeiras (hand_dist >= 0.09 vs empunhadura bimanual da Shinai <= 0.065).
        2. Cabeça descoberta (sem capacete Men metálico do Kendo).
        3. Posicionamento na geometria perimetral do triângulo de arbitragem da FIK:
           - Lateral Esquerda (cx <= 0.22)
           - Lateral Direita (cx >= 0.80)
           - Fundo de Quadra / Mesas (ground_y <= 0.66 ou cy <= 0.52).
        4. Presença de bandeiras vermelha e branca nas imediações dos punhos.
        """
        if not landmarks:
            return False, "NONE", 0.0

        cx, cy, _ = self.extract_bbox_and_center(landmarks)
        m = self.get_skeleton_metrics(landmarks)
        rw = landmarks.get("RIGHT_WRIST")
        lw = landmarks.get("LEFT_WRIST")
        h = max(0.01, m["height"])

        # Distância entre as mãos
        hand_dist = float(np.hypot(rw["x"] - lw["x"], rw["y"] - lw["y"])) if rw and lw and "x" in rw and "x" in lw else 1.0

        # Zonas típicas dos 3 Shinpans
        is_left_perimeter = (cx <= 0.22)
        is_right_perimeter = (cx >= 0.80)
        is_court_depth = (m["ground_y"] <= 0.66 or cy <= 0.52)
        is_shinpan_zone = is_left_perimeter or is_right_perimeter or is_court_depth

        # Mãos separadas: árbitros seguram uma bandeira em cada mão
        is_hands_apart = (hand_dist >= 0.09) or (hand_dist >= 0.28 * h)

        # Cabeça descoberta: orelhas visíveis sem o capacete Men
        re = landmarks.get("RIGHT_EAR")
        le = landmarks.get("LEFT_EAR")
        has_exposed_ears = bool(re or le)

        # Detecção de cor de bandeiras nas mãos (se frame fornecido)
        has_flag_colors = False
        if frame is not None and (rw or lw):
            fh, fw = frame.shape[:2]
            for w_pt in [rw, lw]:
                if w_pt and "x" in w_pt:
                    wx, wy = int(w_pt["x"] * fw), int(w_pt["y"] * fh)
                    x1, x2 = max(0, wx - 25), min(fw, wx + 25)
                    y1, y2 = max(0, wy - 15), min(fh, wy + 55)
                    hand_roi = frame[y1:y2, x1:x2]
                    if hand_roi.size > 0:
                        hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
                        m_red1 = cv2.inRange(hsv, np.array([0, 100, 75]), np.array([12, 255, 255]))
                        m_red2 = cv2.inRange(hsv, np.array([168, 100, 75]), np.array([180, 255, 255]))
                        m_red = cv2.bitwise_or(m_red1, m_red2)
                        m_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
                        if cv2.countNonZero(m_red) > 10 or cv2.countNonZero(m_white) > 25:
                            has_flag_colors = True
                            break

        confidence = 0.0
        # No miolo de combate central (área dos lutadores com solo normal),
        # um indivíduo só pode ser árbitro se explicitamente tiver cabeça descoberta E mãos afastadas com bandeiras
        if (0.22 < cx < 0.78) and m["ground_y"] > 0.66:
            if not has_exposed_ears or not is_hands_apart:
                return False, "NONE", 0.0

        if is_shinpan_zone:
            confidence += 0.35
        if is_hands_apart:
            confidence += 0.35
        if has_exposed_ears:
            confidence += 0.20
        if has_flag_colors:
            confidence += 0.20

        # Posição lateral típica dos 2 árbitros de borda (triângulo da FIK)
        if (0.10 <= cx <= 0.22 or 0.78 <= cx <= 0.90):
            if is_hands_apart and (has_exposed_ears or has_flag_colors or m["ground_y"] >= 0.80):
                confidence = max(confidence, 0.85)

        if confidence >= 0.50:
            if is_left_perimeter:
                role = "SHINPAN_LEFT"
            elif is_right_perimeter:
                role = "SHINPAN_RIGHT"
            else:
                role = "SHINPAN_CENTER"
            return True, role, confidence

        return False, "NONE", confidence

    @property
    def state(self) -> str:
        """Alias ergonômico para tracking_state."""
        return self.tracking_state

    def classify_plane(self, landmarks: Optional[Dict[str, Any]], frame: Optional[np.ndarray] = None) -> Tuple[str, float, str]:
        """Classifica um esqueleto em MAIN_PLANE, BACKGROUND, FOREGROUND_OCCLUDER ou SHINPAN."""
        if not landmarks:
            return "INVALID", 0.0, "Sem landmarks válidos"

        self.total_detections_processed += 1
        m = self.get_skeleton_metrics(landmarks)

        ref_h = self.ref_height if self.is_calibrated else 0.42
        ref_area = self.ref_bbox_area if self.is_calibrated else 0.08
        ref_ground = self.ref_ground_y if self.is_calibrated else 0.78

        scale_h = m["height"] / max(0.01, ref_h)
        scale_area = m["area"] / max(0.001, ref_area)

        # 1. Identificação direta de Árbitro Oficial (SHINPAN)
        is_sp, sp_role, sp_conf = self.classify_shinpan(landmarks, frame=frame)
        if is_sp:
            self.discarded_shinpan_count += 1
            reason = f"Árbitro ({sp_role}) identificado fora do combate (Conf: {sp_conf:.2f})"
            return "SHINPAN", scale_h, reason

        # 2. Verificação de Primeiro Plano Excessivo (FOREGROUND_OCCLUDER)
        # Árbitro ou operador de câmera em frente ao plano de combate
        is_fg_bottom = (m["ground_y"] >= 0.92 and m["height"] > 0.36)
        is_fg_huge = (m["height"] > 0.65)
        is_fg_edge_crop = (m["ymin"] <= 0.01 and m["ymax"] >= 0.95 and scale_area > 1.1)

        if is_fg_bottom or is_fg_huge or is_fg_edge_crop:
            self.discarded_foreground_count += 1
            reason = f"Árbitro/Oclusão na frente da câmera (Escala: {scale_h:.2f}x, Área: {scale_area:.2f}x ref, Pé Y: {m['ground_y']:.2f})"
            return "FOREGROUND_OCCLUDER", scale_h, reason

        # 3. Verificação de Segundo Plano (BACKGROUND)
        # Pessoas nas mesas ao fundo, árbitro central afastado ou plateia
        is_bg_court_depth = (m["ground_y"] < 0.68 and m["height"] < 0.22) or (m["ground_y"] < 0.60)
        is_bg_distant = self.is_calibrated and (m["ground_y"] < (ref_ground - 0.10)) and (scale_h < 0.75)
        is_bg_tiny = (m["height"] < 0.13) or (self.is_calibrated and scale_h < 0.35 and m["height"] < 0.18)

        if is_bg_court_depth or is_bg_distant or is_bg_tiny:
            self.discarded_background_count += 1
            reason = f"Elemento de Segundo Plano detectado (Escala: {scale_h:.2f}x, Área: {scale_area:.2f}x ref, Pé Y: {m['ground_y']:.2f})"
            return "BACKGROUND", scale_h, reason

        # 4. Plano Principal de Combate
        return "MAIN_PLANE", scale_h, "Plano Principal de Combate"

    def select_best_combatant_pair(
        self,
        candidates: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """Seleciona com máxima precisão o par (Kenshi_Left, Kenshi_Right) dentre múltiplos candidatos."""
        if not candidates:
            return None, None, []

        if len(candidates) == 1:
            return candidates[0], None, []

        if len(candidates) == 2:
            c1, c2 = candidates[0], candidates[1]
            cx1, _, _ = self.extract_bbox_and_center(c1)
            cx2, _, _ = self.extract_bbox_and_center(c2)
            c_left, c_right = (c1, c2) if cx1 <= cx2 else (c2, c1)
            return c_left, c_right, []

        # Para N >= 3 candidatos: computar pontuação postural e geométrica de todas as combinações 2 a 2
        best_pair = (candidates[0], candidates[1])
        best_pair_score = -1e9

        m_list = [self.get_skeleton_metrics(c) for c in candidates]
        k_scores = [self.compute_kenshi_feature_score(c, frame=frame) for c in candidates]

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                c_i, c_j = candidates[i], candidates[j]
                m_i, m_j = m_list[i], m_list[j]
                ks_i, ks_j = k_scores[i], k_scores[j]

                cx_i = (m_i["xmin"] + m_i["xmax"]) / 2.0
                cx_j = (m_j["xmin"] + m_j["xmax"]) / 2.0
                pair_center = (cx_i + cx_j) / 2.0
                pair_dist = abs(cx_i - cx_j)

                # 1. Pontuação individual de Kenshi (Shinai / Kamae / Sonkyō)
                score = (ks_i + ks_j) * 3.0

                # 2. Compatibilidade de escala
                h_max = max(m_i["height"], m_j["height"], 0.01)
                h_min = min(m_i["height"], m_j["height"])
                scale_ratio = h_min / h_max
                score += scale_ratio * 2.0

                # 3. Alinhamento de solo
                ground_diff = abs(m_i["ground_y"] - m_j["ground_y"])
                if ground_diff < 0.12:
                    score += 1.5
                else:
                    score -= ground_diff * 4.0

                # 4. Centralidade conjunta da dupla no Shiaijo
                center_dist = abs(pair_center - 0.50)
                score += max(0.0, 1.0 - center_dist * 2.5) * 2.0

                # Penalidade severa se algum estiver nas bordas do enquadramento
                if cx_i < 0.14 or cx_i > 0.86: score -= 5.0
                if cx_j < 0.14 or cx_j > 0.86: score -= 5.0

                # 5. Distância mútua de combate (Maai típico: 0.12 a 0.50)
                if 0.12 <= pair_dist <= 0.50:
                    score += 2.0
                else:
                    score -= max(1.0, (pair_dist - 0.50) * 8.0)

                # 6. Continuidade temporal com posições rastreadas anteriormente
                target_aka_x = self.aka.pred_x if self.aka.pred_x is not None else self.aka.last_center_x
                target_shiro_x = self.shiro.pred_x if self.shiro.pred_x is not None else self.shiro.last_center_x
                if target_aka_x is not None and target_shiro_x is not None:
                    d_tracked = min(
                        abs(cx_i - target_aka_x) + abs(cx_j - target_shiro_x),
                        abs(cx_i - target_shiro_x) + abs(cx_j - target_aka_x)
                    )
                    score += max(0.0, 3.0 - d_tracked * 6.0)

                if score > best_pair_score:
                    best_pair_score = score
                    best_pair = (c_i, c_j)

        c_a, c_b = best_pair
        cxa, _, _ = self.extract_bbox_and_center(c_a)
        cxb, _, _ = self.extract_bbox_and_center(c_b)
        cand_left, cand_right = (c_a, c_b) if cxa <= cxb else (c_b, c_a)

        ref_h = (self.get_skeleton_metrics(cand_left)["height"] + self.get_skeleton_metrics(cand_right)["height"]) / 2.0
        discarded = []
        for c in candidates:
            if c is not cand_left and c is not cand_right:
                m_c = self.get_skeleton_metrics(c)
                rel_scale = m_c["height"] / max(0.01, ref_h)
                if rel_scale > 1.25:
                    p_type = "FOREGROUND_OCCLUDER"
                    reason = f"Árbitro (Shinpan) em 1º plano descartado (Escala: {rel_scale:.2f}x)"
                    self.discarded_foreground_count += 1
                else:
                    p_type = "BACKGROUND"
                    reason = f"Elemento de 2º plano / Árbitro ao fundo descartado (Escala: {rel_scale:.2f}x)"
                    self.discarded_background_count += 1

                self.discarded_shinpan_posture_count += 1
                discarded.append({
                    "landmarks": c,
                    "plane_type": p_type,
                    "scale": rel_scale,
                    "reason": reason
                })

        return cand_left, cand_right, discarded

    def associate_and_filter(
        self,
        frame_landmarks_list: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
        return_persisted: bool = False
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Associa os 2 Kenshis principais com imunidade a inversões e ancoragem do Shinai.
        """
        def _wrap_result(a_res, s_res, d_res):
            if not return_persisted:
                return a_res, s_res, d_res
            final_a = a_res if a_res is not None else self.aka.get_persisted_landmarks(max_gap=self.track_buffer)
            final_s = s_res if s_res is not None else self.shiro.get_persisted_landmarks(max_gap=self.track_buffer)
            return final_a, final_s, d_res

        if not frame_landmarks_list:
            self.aka.update(None)
            self.shiro.update(None)
            return _wrap_result(None, None, [])

        # Resetar árbitros identificados no frame atual
        self.shinpans = {"SHINPAN_LEFT": None, "SHINPAN_CENTER": None, "SHINPAN_RIGHT": None}

        # --- ETAPA 1: FILTRAGEM POR DELIMITAÇÃO DA QUADRA (SHIAI-JO ROI) ---
        shiaijo_filtered: List[Dict[str, Any]] = []
        discarded_items: List[Dict[str, Any]] = []

        for lm in frame_landmarks_list:
            m = self.get_skeleton_metrics(lm)
            if not self.is_within_shiaijo(m["ground_x"], m["ground_y"]):
                # Verificar se é um dos Shinpans perimetrais
                is_sp, role, _ = self.classify_shinpan(lm, frame=frame)
                if is_sp:
                    self.shinpans[role] = lm
                    self.discarded_shinpan_count += 1
                    plane_t = "SHINPAN"
                    reason = f"Árbitro (Shinpan) lateral/perimetral descartado ({role})"
                else:
                    self.discarded_out_of_shiaijo_count += 1
                    plane_t = "OUT_OF_BOUNDS"
                    reason = f"Detecção fora dos limites do Shiai-jo ativo (Solo: X={m['ground_x']:.2f}, Y={m['ground_y']:.2f})"
                discarded_items.append({
                    "landmarks": lm,
                    "plane_type": plane_t,
                    "scale": 1.0,
                    "reason": reason
                })
            else:
                shiaijo_filtered.append(lm)

        if not shiaijo_filtered:
            self.aka.update(None)
            self.shiro.update(None)
            return _wrap_result(None, None, discarded_items)

        # --- ETAPA 2: PRÉ-FILTRO POSTURAL E SELEÇÃO DE CANDIDATOS VÁLIDOS ---
        main_plane_candidates: List[Dict[str, Any]] = []
        for lm in shiaijo_filtered:
            plane_type, scale, reason = self.classify_plane(lm, frame=frame)
            if plane_type == "MAIN_PLANE":
                main_plane_candidates.append(lm)
            else:
                if plane_type == "SHINPAN":
                    _, role, _ = self.classify_shinpan(lm, frame=frame)
                    self.shinpans[role] = lm
                discarded_items.append({
                    "landmarks": lm,
                    "plane_type": plane_type,
                    "scale": scale,
                    "reason": reason
                })

        # NUNCA reintegra árbitros ou não-combatentes no modo LOCKED_COMBAT!
        # Apenas na inicialização antes da calibração se houver falha temporária
        if not self.is_calibrated and len(main_plane_candidates) < 2 and len(shiaijo_filtered) >= 2:
            eligible = [c for c in shiaijo_filtered if not self.classify_shinpan(c, frame=frame)[0]]
            if len(eligible) >= 2:
                sorted_by_kenshi = sorted(eligible, key=lambda c: self.compute_kenshi_feature_score(c), reverse=True)
                main_plane_candidates = sorted_by_kenshi[:2]
                discarded_items = [item for item in discarded_items if item["landmarks"] not in main_plane_candidates]

        cand_left: Optional[Dict[str, Any]] = None
        cand_right: Optional[Dict[str, Any]] = None

        if len(main_plane_candidates) == 0:
            self.aka.update(None)
            self.shiro.update(None)
            return _wrap_result(None, None, discarded_items)
        elif len(main_plane_candidates) == 1:
            cand_left = main_plane_candidates[0]
            cand_right = None
        elif len(main_plane_candidates) == 2:
            c1, c2 = main_plane_candidates[0], main_plane_candidates[1]
            cx1, _, _ = self.extract_bbox_and_center(c1)
            cx2, _, _ = self.extract_bbox_and_center(c2)
            cand_left, cand_right = (c1, c2) if cx1 <= cx2 else (c2, c1)
        else:
            cand_left, cand_right, disc_shinpan = self.select_best_combatant_pair(main_plane_candidates, frame=frame)
            discarded_items.extend(disc_shinpan)

        if cand_left is None and cand_right is None:
            self.aka.update(None)
            self.shiro.update(None)
            return _wrap_result(None, None, discarded_items)

        # Caso onde apenas 1 combatente foi detectado (oclusão momentânea do oponente)
        if cand_left is None or cand_right is None:
            single_cand = cand_left if cand_left is not None else cand_right
            if single_cand is None:
                self.aka.update(None)
                self.shiro.update(None)
                return _wrap_result(None, None, discarded_items)

            cx, cy, _ = self.extract_bbox_and_center(single_cand)
            target_aka_x = self.aka.pred_x if self.aka.pred_x is not None else (self.aka.last_center_x if self.aka.last_center_x is not None else 0.70)
            target_aka_y = self.aka.pred_y if self.aka.pred_y is not None else (self.aka.last_center_y if self.aka.last_center_y is not None else 0.50)
            target_shiro_x = self.shiro.pred_x if self.shiro.pred_x is not None else (self.shiro.last_center_x if self.shiro.last_center_x is not None else 0.30)
            target_shiro_y = self.shiro.pred_y if self.shiro.pred_y is not None else (self.shiro.last_center_y if self.shiro.last_center_y is not None else 0.50)

            d_aka = float(np.hypot(cx - target_aka_x, cy - target_aka_y))
            d_shiro = float(np.hypot(cx - target_shiro_x, cy - target_shiro_y))

            if self.aka.last_center_x is not None or self.shiro.last_center_x is not None:
                if d_aka <= d_shiro:
                    if self.aka.occluded_frames > 0:
                        self.occlusion_recovery_count += 1
                    single_cand["SHINAI"] = self.shinai_tracker.track_shinai(frame, single_cand, self.aka.shinai)
                    self.aka.update(single_cand)
                    self.shiro.update(None)
                    res_a, res_s = (single_cand, None) if not self.invert_assignment else (None, single_cand)
                    return _wrap_result(res_a, res_s, discarded_items)
                else:
                    if self.shiro.occluded_frames > 0:
                        self.occlusion_recovery_count += 1
                    single_cand["SHINAI"] = self.shinai_tracker.track_shinai(frame, single_cand, self.shiro.shinai)
                    self.shiro.update(single_cand)
                    self.aka.update(None)
                    res_a, res_s = (None, single_cand) if not self.invert_assignment else (single_cand, None)
                    return _wrap_result(res_a, res_s, discarded_items)
            else:
                # Inicialização sem histórico prévio
                if not self.invert_assignment:
                    if cx <= 0.50:
                        single_cand["SHINAI"] = self.shinai_tracker.track_shinai(frame, single_cand, self.shiro.shinai)
                        self.shiro.update(single_cand)
                        self.aka.update(None)
                        return _wrap_result(None, single_cand, discarded_items)
                    else:
                        single_cand["SHINAI"] = self.shinai_tracker.track_shinai(frame, single_cand, self.aka.shinai)
                        self.aka.update(single_cand)
                        self.shiro.update(None)
                        return _wrap_result(single_cand, None, discarded_items)
                else:
                    if cx <= 0.50:
                        single_cand["SHINAI"] = self.shinai_tracker.track_shinai(frame, single_cand, self.aka.shinai)
                        self.aka.update(single_cand)
                        self.shiro.update(None)
                        return _wrap_result(single_cand, None, discarded_items)
                    else:
                        single_cand["SHINAI"] = self.shinai_tracker.track_shinai(frame, single_cand, self.shiro.shinai)
                        self.shiro.update(single_cand)
                        self.aka.update(None)
                        return _wrap_result(None, single_cand, discarded_items)

        # --- ETAPA 3: AMOSTRAGEM DA FITA VERMELHA (TASUKI) ---
        score_left = 0.0
        score_right = 0.0
        if frame is not None:
            score_left = self.detect_red_flag_score(frame, cand_left)
            score_right = self.detect_red_flag_score(frame, cand_right)

            if self.tracking_state != "LOCKED_COMBAT":
                if score_left >= 0.10:
                    self.candidate_left_red_score += score_left
                    self.red_evidence_frames_left += 1

                if score_right >= 0.10:
                    self.candidate_right_red_score += score_right
                    self.red_evidence_frames_right += 1

        # --- ETAPA 4: ASSOCIAÇÃO BIPARTIDA E TRAVAMENTO DE IDS (K=2) ---
        cxl, cyl, _ = self.extract_bbox_and_center(cand_left)
        cxr, cyr, _ = self.extract_bbox_and_center(cand_right)

        if self.tracking_state == "LOCKED_COMBAT":
            target_aka_x = self.aka.pred_x if self.aka.pred_x is not None else (self.aka.last_center_x if self.aka.last_center_x is not None else 0.70)
            target_aka_y = self.aka.pred_y if self.aka.pred_y is not None else (self.aka.last_center_y if self.aka.last_center_y is not None else 0.50)
            target_shiro_x = self.shiro.pred_x if self.shiro.pred_x is not None else (self.shiro.last_center_x if self.shiro.last_center_x is not None else 0.30)
            target_shiro_y = self.shiro.pred_y if self.shiro.pred_y is not None else (self.shiro.last_center_y if self.shiro.last_center_y is not None else 0.50)

            pair_dist = abs(cxl - cxr)
            prev_aka_is_left = (target_aka_x < target_shiro_x)
            flag_diff = score_left - score_right

            # REGRA 1: Evidência contundente de fita vermelha (Aka Tasuki inequívoco com contraste >= 0.50)
            # Previne trocas espúrias por ruído de piso (<0.30), mas respeita fitas vermelhas reais confirmadas
            if flag_diff >= 0.50:
                aka_lm = cand_left
                shiro_lm = cand_right
            elif flag_diff <= -0.50:
                aka_lm = cand_right
                shiro_lm = cand_left
            # REGRA 2: BARREIRA FÍSICA ESPACIAL ANTI-TELEPORTE
            # Se os atletas estão separados por mais de 15% da largura da tela e sem fita contundente,
            # é fisicamente impossível terem trocado de posição em um único frame (33ms).
            elif pair_dist > 0.15:
                if prev_aka_is_left:
                    aka_lm = cand_left
                    shiro_lm = cand_right
                else:
                    aka_lm = cand_right
                    shiro_lm = cand_left
            else:
                # REGRA 3: CRUZAMENTO FÍSICO / TAIATARI (pair_dist <= 0.15)
                # Os lutadores estão em contato próximo ou cruzando. Usamos distância contínua com histerese.
                dist_l_to_shiro = float(np.hypot(cxl - target_shiro_x, cyl - target_shiro_y))
                dist_r_to_aka = float(np.hypot(cxr - target_aka_x, cyr - target_aka_y))
                dist_l_to_aka = float(np.hypot(cxl - target_aka_x, cyl - target_aka_y))
                dist_r_to_shiro = float(np.hypot(cxr - target_shiro_x, cyr - target_shiro_y))

                # Opção A: cand_left = Shiro, cand_right = Aka
                cost_a = dist_l_to_shiro + dist_r_to_aka
                # Opção B: cand_left = Aka, cand_right = Shiro
                cost_b = dist_l_to_aka + dist_r_to_shiro

                # Histerese contra inversão espúria momentânea
                if prev_aka_is_left:
                    if cost_a < (cost_b - 0.18):
                        aka_lm = cand_right
                        shiro_lm = cand_left
                    else:
                        aka_lm = cand_left
                        shiro_lm = cand_right
                else:
                    if cost_b < (cost_a - 0.18):
                        aka_lm = cand_left
                        shiro_lm = cand_right
                    else:
                        aka_lm = cand_right
                        shiro_lm = cand_left

        else:
            # Inicialização / Primeiro frame de Sonkyō
            diff = self.candidate_right_red_score - self.candidate_left_red_score
            if diff >= 0.40:
                aka_lm = cand_right
                shiro_lm = cand_left
                self.flag_decision = "FLAG_DETECTED_RIGHT_IS_AKA"
                self.flag_confidence = float(np.clip(diff / 2.0, 0.60, 0.98))
            elif diff <= -0.40:
                aka_lm = cand_left
                shiro_lm = cand_right
                self.flag_decision = "FLAG_DETECTED_LEFT_IS_AKA"
                self.flag_confidence = float(np.clip(abs(diff) / 2.0, 0.60, 0.98))
            else:
                aka_lm = cand_right
                shiro_lm = cand_left
                self.flag_decision = "POSITION_DEFAULT_OPPOSITE_JUDGES"
                self.flag_confidence = 0.50

            if self.invert_assignment:
                aka_lm, shiro_lm = shiro_lm, aka_lm

            if self.lock_tracks and aka_lm and shiro_lm:
                self.tracking_state = "LOCKED_COMBAT"
                self.aka.is_locked = True
                self.shiro.is_locked = True

        # Anexar Shinai rastreado aos landmarks finais com orientação corporal e espacial correta
        if aka_lm and shiro_lm:
            cxa, _, _ = self.extract_bbox_and_center(aka_lm)
            cxs, _, _ = self.extract_bbox_and_center(shiro_lm)
            facing_aka = "RIGHT" if cxa < cxs else "LEFT"
            facing_shiro = "LEFT" if cxa < cxs else "RIGHT"
        else:
            facing_aka = "LEFT" if not self.invert_assignment else "RIGHT"
            facing_shiro = "RIGHT" if not self.invert_assignment else "LEFT"

        if aka_lm:
            aka_lm["SHINAI"] = self.shinai_tracker.track_shinai(
                frame,
                aka_lm,
                prev_state=self.aka.shinai,
                expected_facing=facing_aka
            )
        if shiro_lm:
            shiro_lm["SHINAI"] = self.shinai_tracker.track_shinai(
                frame,
                shiro_lm,
                prev_state=self.shiro.shinai,
                expected_facing=facing_shiro
            )

        # Calibrar plano principal se ainda não calibrado
        if not self.is_calibrated and aka_lm and shiro_lm:
            self.calibrate_main_plane([aka_lm, shiro_lm])

        # Verificar recuperação de oclusão
        if self.aka.occluded_frames > 0 and aka_lm:
            self.occlusion_recovery_count += 1
        if self.shiro.occluded_frames > 0 and shiro_lm:
            self.occlusion_recovery_count += 1

        self.aka.update(aka_lm)
        self.shiro.update(shiro_lm)

        return _wrap_result(aka_lm, shiro_lm, discarded_items)

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo das estatísticas de rastreamento, filtragem de planos, Shiaijo e detecção de flag."""
        return {
            "is_calibrated": self.is_calibrated,
            "tracking_state": self.tracking_state,
            "shiaijo_polygon_configured": bool(self.shiaijo_polygon),
            "ref_height": round(self.ref_height, 3),
            "ref_bbox_area": round(self.ref_bbox_area, 3),
            "ref_ground_y": round(self.ref_ground_y, 3),
            "discarded_background_count": self.discarded_background_count,
            "discarded_foreground_count": self.discarded_foreground_count,
            "discarded_out_of_shiaijo_count": self.discarded_out_of_shiaijo_count,
            "discarded_shinpan_posture_count": self.discarded_shinpan_posture_count,
            "discarded_shinpan_count": self.discarded_shinpan_count,
            "shinpans_tracked": {k: bool(v is not None) for k, v in self.shinpans.items()},
            "occlusion_recovery_count": self.occlusion_recovery_count,
            "total_detections_processed": self.total_detections_processed,
            "flag_decision": self.flag_decision,
            "flag_confidence": round(self.flag_confidence, 2),
            "invert_assignment": self.invert_assignment,
            "candidate_left_red_score": round(self.candidate_left_red_score, 2),
            "candidate_right_red_score": round(self.candidate_right_red_score, 2),
            "tracked_combatants": [
                {"id": self.aka.id, "name": self.aka.name, "frames_tracked": len([p for p in self.aka.history if p])},
                {"id": self.shiro.id, "name": self.shiro.name, "frames_tracked": len([p for p in self.shiro.history if p])}
            ]
        }
