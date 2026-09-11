"""
Módulo de Rastreamento dos 2 Combatentes Principais e Filtragem de Planos.
Identifica e rastreia os 2 Kenshi (Aka & Shiro) que realizaram o Sonkyō inicial de abertura,
descartando elementos de segundo plano (outras lutas/fundo), transeuntes de primeiro plano (frente da câmera),
árbitros (Shinpans) e detecções fora da área regulamentar do Shiai-jo.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

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
            # Se um pulso não foi detectado no frame mas estava presente no frame anterior,
            # propaga a posição do pulso acompanhando o deslocamento do tronco.
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
            # Projetar cada ponto somando o deslocamento inercial amortecido
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
        """
        - min_background_scale_ratio: Limiar abaixo do qual o elemento é classificado como Segundo Plano (Fundo).
        - max_foreground_scale_ratio: Limiar acima do qual o elemento é classificado como Oclusão de Primeiro Plano (Frente da Câmera).
        - ground_line_tolerance: Tolerância de deslocamento vertical dos pés em relação ao solo do Shiaijo.
        - invert_assignment: Se True, inverte manualmente as identidades de Aka e Shiro.
        - shiaijo_polygon: Polígono 2D de coordenadas normalizadas (x, y) definindo a área do Shiaijo.
        - min_kenshi_score: Pontuação mínima de postura (empunhadura Shinai vs bandeiras de árbitro).
        - track_buffer: Número máximo de frames de buffer para manter Kenshi ocluído sem perder ID.
        - lock_tracks: Se True, trava o rastreador estritamente em K=2 combatentes após a inicialização.

        Configuração Padrão de Posição (Câmera Oposta à Mesa dos Juízes):
        Em Kendo oficial, observando do lado oposto à mesa dos juízes (visão padrão da câmera):
        - Esquerda do enquadramento (x <= 0.50) = Kenshi Shiro (Branco)
        - Direita do enquadramento (x > 0.50) = Kenshi Aka (Vermelho)
        """
        self.min_bg_ratio = min_background_scale_ratio
        self.max_fg_ratio = max_foreground_scale_ratio
        self.ground_tolerance = ground_line_tolerance
        self.invert_assignment = invert_assignment
        self.shiaijo_polygon = shiaijo_polygon
        self.min_kenshi_score = min_kenshi_score
        self.track_buffer = track_buffer
        self.lock_tracks = lock_tracks

        # Perfis dos 2 lutadores
        self.aka = CombatantProfile("KENSHI_AKA", "Kenshi Aka (Vermelho)", (40, 40, 230)) # Vermelho BGR
        self.shiro = CombatantProfile("KENSHI_SHIRO", "Kenshi Shiro (Branco)", (240, 240, 240)) # Branco BGR

        # Rastreamento de Evidência da Flag Vermelha (Tasukuki nas costas)
        self.candidate_left_red_score = 0.0
        self.candidate_right_red_score = 0.0
        self.red_evidence_frames_left = 0
        self.red_evidence_frames_right = 0
        self.flag_decision = "POSITION_DEFAULT_OPPOSITE_JUDGES"
        self.flag_confidence = 0.50

        # Estado do Sistema de Rastreamento (K=2)
        self.tracking_state = "UNINITIALIZED" # "UNINITIALIZED", "LOCKED_COMBAT"
        self.is_calibrated = False
        self.ref_height = 0.35
        self.ref_bbox_area = 0.08
        self.ref_ground_y = 0.78
        self.ref_shoulder_width = 0.12

        # Contadores estatísticos de descartes e recuperação
        self.discarded_background_count = 0
        self.discarded_foreground_count = 0
        self.discarded_out_of_shiaijo_count = 0
        self.discarded_shinpan_posture_count = 0
        self.total_detections_processed = 0
        self.occlusion_recovery_count = 0

    def is_within_shiaijo(self, ground_x: float, ground_y: float) -> bool:
        """
        Verifica se as coordenadas de contato com o solo (ground_x, ground_y)
        estão estritamente contidas no polígono do Shiai-jo.
        Se nenhum polígono for configurado, aplica margens automáticas padrão da quadra
        para filtrar árbitros laterais e pessoas nas bordas da câmera (9 a 11 metros centrais).
        """
        if not self.shiaijo_polygon or len(self.shiaijo_polygon) < 3:
            # Margem padrão de segurança: Kenshis lutam entre x in [0.12, 0.88] e y in [0.20, 0.98]
            return 0.12 <= ground_x <= 0.88 and 0.20 <= ground_y <= 0.98

        poly = np.array(self.shiaijo_polygon, dtype=np.float32)
        res = cv2.pointPolygonTest(poly, (ground_x, ground_y), False)
        return res >= 0

    @staticmethod
    def detect_red_flag_score(frame: Optional[np.ndarray], landmarks: Optional[Dict[str, Any]]) -> float:
        """
        Analisa a presença da fita vermelha (Aka Tasukuki / Mejirushi) nas costas/tronco do praticante.
        O Keikogi pode ser de qualquer cor (azul escuro, branco, preto), mas a fita vermelha tem
        alta saturação e matiz vermelho característico no dorso (região entre ombros e quadril).
        Retorna uma pontuação de 0.0 a 1.0 (densidade/intensidade de vermelho na ROI dorsal).
        """
        if frame is None or not landmarks:
            return 0.0

        h, w = frame.shape[:2]
        
        # Obter bounding box da região dorsal / tronco
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

        # Converter ROI para HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Máscaras para a cor vermelha da fita (Tasukuki)
        # Faixa 1: H [0, 14], S >= 70, V >= 50
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50], dtype=np.uint8), np.array([14, 255, 255], dtype=np.uint8))
        # Faixa 2: H [166, 180], S >= 70, V >= 50
        mask2 = cv2.inRange(hsv, np.array([166, 70, 50], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return 0.0

        red_ratio = red_pixels / float(total_pixels)
        score = float(np.clip(red_ratio / 0.05, 0.0, 1.0))
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
        """
        Interpola linearmente poses e membros ausentes em gaps temporários (<= max_gap frames).
        Garante continuidade absoluta no vídeo renderizado e nas análises de velocidade e trajetória.
        """
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

        # Posição dos pés / tornozelos
        if "RIGHT_ANKLE" in landmarks and "LEFT_ANKLE" in landmarks:
            ground_y = (landmarks["RIGHT_ANKLE"]["y"] + landmarks["LEFT_ANKLE"]["y"]) / 2.0
            ground_x = (landmarks["RIGHT_ANKLE"]["x"] + landmarks["LEFT_ANKLE"]["x"]) / 2.0
        elif "RIGHT_FOOT_INDEX" in landmarks:
            ground_y = landmarks["RIGHT_FOOT_INDEX"]["y"]
            ground_x = landmarks["RIGHT_FOOT_INDEX"]["x"]
        else:
            ground_y = ymax
            ground_x = (xmin + xmax) / 2.0

        # Largura dos ombros
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

    @staticmethod
    def compute_kenshi_feature_score(landmarks: Optional[Dict[str, Any]]) -> float:
        """
        Calcula a probabilidade (0.0 a 1.0) de um esqueleto detectado ser um Kenshi (lutador)
        em vez de um Shinpan (árbitro de Kendo segurando bandeiras) ou espectador.
        
        Critérios avaliados:
        1. Empunhadura de Shinai em Kamae: As duas mãos (RIGHT_WRIST e LEFT_WRIST) estão próximas,
           empunhando o Tsuka (cabo da espada) na linha central do abdômen/solar plexus.
           (Árbitros mantêm as mãos afastadas segurando bandeiras nas laterais ou abaixadas).
        2. Centralidade horizontal no Shiaijo: Kenshis combatem na área central (x=0.20 a x=0.80),
           enquanto árbitros ocupam as bordas e esquinas do Shiaijo.
        3. Postura de corte / elevação de braços (Furikaburi): Mãos elevadas acima dos ombros.
        4. Postura de Sonkyō / Flexão atlética de pernas.
        """
        if not landmarks:
            return 0.0

        cx, cy, (xmin, ymin, xmax, ymax) = CombatantTracker.extract_bbox_and_center(landmarks)
        h = max(0.01, ymax - ymin)
        score = 0.35

        # 1. Proximidade dos pulsos (Empunhadura bimanual do Shinai)
        r_wrist = landmarks.get("RIGHT_WRIST")
        l_wrist = landmarks.get("LEFT_WRIST")
        if r_wrist and l_wrist and "x" in r_wrist and "x" in l_wrist:
            wrist_dist = np.hypot(r_wrist["x"] - l_wrist["x"], r_wrist["y"] - l_wrist["y"])
            # No Kendo Kamae, as duas mãos seguram o mesmo cabo (< 0.18 de h)
            if wrist_dist < (0.18 * h):
                score += 0.35
            elif wrist_dist < (0.28 * h):
                score += 0.15
            else:
                # Mãos bem abertas / separadas (característico de árbitro com bandeiras nas duas mãos)
                score -= 0.25

        # 2. Posição no Shiaijo (Kenshis no miolo, árbitros nas extremidades)
        dist_center_x = abs(cx - 0.50)
        if dist_center_x <= 0.28:
            score += 0.25 * (1.0 - (dist_center_x / 0.28))
        elif dist_center_x >= 0.42:
            # Posição periférica extrema (típico de árbitro lateral)
            score -= 0.20

        # 3. Elevação de braços para corte (Furikaburi / Shinai elevado)
        r_sh = landmarks.get("RIGHT_SHOULDER")
        if r_wrist and r_sh and "y" in r_wrist and "y" in r_sh:
            if r_wrist["y"] <= r_sh["y"]:
                score += 0.25

        # 4. Flexão de joelhos / Agachamento de Sonkyō
        r_hip = landmarks.get("RIGHT_HIP")
        r_knee = landmarks.get("RIGHT_KNEE")
        r_ankle = landmarks.get("RIGHT_ANKLE")
        if r_hip and r_knee and r_ankle and "y" in r_hip and "y" in r_knee and "y" in r_ankle:
            leg_span = abs(r_ankle["y"] - r_hip["y"])
            if leg_span < (0.38 * h):
                score += 0.25  # Sonkyō ou agachamento atlético

        return float(np.clip(score, 0.0, 1.0))

    def calibrate_main_plane(self, candidate_poses: List[Dict[str, Any]]):
        """
        Calibra as métricas de referência do Plano Principal com base nas poses dos dois lutadores.
        """
        if not candidate_poses:
            return

        valid_metrics = [self.get_skeleton_metrics(p) for p in candidate_poses if p]
        if not valid_metrics:
            return

        heights = [m["height"] for m in valid_metrics]
        avg_h = float(np.mean(heights)) if heights else 0.35
        self.ref_height = max(0.20, avg_h)
        self.ref_bbox_area = float(np.mean([m["area"] for m in valid_metrics]))
        self.ref_ground_y = float(np.mean([m["ground_y"] for m in valid_metrics]))
        self.ref_shoulder_width = float(np.mean([m["shoulder_width"] for m in valid_metrics]))
        self.is_calibrated = True

    def classify_plane(self, landmarks: Optional[Dict[str, Any]]) -> Tuple[str, float, str]:
        """
        Classifica um esqueleto detectado em:
        - "MAIN_PLANE": Pertence aos 2 Kenshi no plano principal da luta.
        - "BACKGROUND": Segundo plano / fundo (outras lutas, árbitros distantes, arquibancada).
        - "FOREGROUND_OCCLUDER": Primeiro plano excessivo / Árbitro em primeiro plano na frente dos Kenshis.
        - "INVALID": Dados insuficientes.
        """
        if not landmarks:
            return "INVALID", 0.0, "Sem landmarks válidos"

        self.total_detections_processed += 1
        m = self.get_skeleton_metrics(landmarks)

        ref_h = self.ref_height if self.is_calibrated else 0.55
        ref_area = self.ref_bbox_area if self.is_calibrated else 0.12
        ref_ground = self.ref_ground_y if self.is_calibrated else 0.85

        scale_h = m["height"] / max(0.01, ref_h)
        scale_area = m["area"] / max(0.001, ref_area)

        # 1. Verificação de Segundo Plano (BACKGROUND)
        is_on_ground_line = (m["ground_y"] >= (ref_ground - self.ground_tolerance))
        is_bg_distant = (m["ground_y"] < (ref_ground - self.ground_tolerance)) and (scale_h < 0.80)
        is_bg_tiny = (scale_h < 0.35) and (scale_area < 0.18)

        if (is_bg_distant or is_bg_tiny) and not is_on_ground_line:
            self.discarded_background_count += 1
            reason = f"Elemento de Segundo Plano detectado (Escala: {scale_h:.2f}x, Área: {scale_area:.2f}x ref, Pé Y: {m['ground_y']:.2f})"
            return "BACKGROUND", scale_h, reason

        # 2. Verificação de Primeiro Plano Excessivo / Árbitro Oclusor (FOREGROUND_OCCLUDER)
        is_fg_scale = (scale_h > self.max_fg_ratio) or (scale_area > (self.max_fg_ratio ** 2))
        is_fg_edge_crop = (m["ymin"] <= 0.01 and m["ymax"] >= 0.98 and scale_area > 1.2)

        if is_fg_scale or is_fg_edge_crop:
            self.discarded_foreground_count += 1
            reason = f"Árbitro/Oclusão na frente da câmera (Escala: {scale_h:.2f}x, Área: {scale_area:.2f}x ref)"
            return "FOREGROUND_OCCLUDER", scale_h, reason

        # 3. Classificação como Plano Principal (MAIN_PLANE)
        return "MAIN_PLANE", scale_h, "Plano Principal de Combate"

    def select_best_combatant_pair(
        self,
        candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Dentre todos os esqueletos detectados no frame (que podem incluir Kenshis, Shinpans/árbitros e transeuntes),
        seleciona com precisão o par (Kenshi_Left, Kenshi_Right) que melhor representa os 2 combatentes.
        """
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
        k_scores = [self.compute_kenshi_feature_score(c) for c in candidates]

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
                score = (ks_i + ks_j) * 2.5

                # 2. Compatibilidade de escala (ambos os Kenshis estão no mesmo plano)
                h_max = max(m_i["height"], m_j["height"], 0.01)
                h_min = min(m_i["height"], m_j["height"])
                scale_ratio = h_min / h_max
                score += scale_ratio * 2.0

                # 3. Alinhamento de solo (mesma linha de pés no Shiaijo)
                ground_diff = abs(m_i["ground_y"] - m_j["ground_y"])
                if ground_diff < 0.12:
                    score += 1.2
                else:
                    score -= ground_diff * 3.0

                # 4. Centralidade conjunta da dupla no Shiaijo
                center_dist = abs(pair_center - 0.50)
                score += max(0.0, 1.0 - center_dist * 2.0) * 1.5

                # Penalidade severa se algum dos membros estiver colado nas bordas do enquadramento
                if cx_i < 0.15 or cx_i > 0.85:
                    score -= 4.0
                if cx_j < 0.15 or cx_j > 0.85:
                    score -= 4.0

                # 5. Distância mútua de combate (Maai típico: 0.10 a 0.55 de distância horizontal)
                if 0.10 <= pair_dist <= 0.55:
                    score += 1.5
                else:
                    score -= max(1.0, (pair_dist - 0.55) * 6.0)

                # 6. Continuidade temporal com posições rastreadas anteriormente
                target_aka_x = self.aka.pred_x if self.aka.pred_x is not None else self.aka.last_center_x
                target_shiro_x = self.shiro.pred_x if self.shiro.pred_x is not None else self.shiro.last_center_x
                if target_aka_x is not None and target_shiro_x is not None:
                    d_tracked = min(
                        abs(cx_i - target_aka_x) + abs(cx_j - target_shiro_x),
                        abs(cx_i - target_shiro_x) + abs(cx_j - target_aka_x)
                    )
                    score += max(0.0, 2.5 - d_tracked * 5.0)

                # 7. Penalidade severa para disparidade de 1º plano (ex: árbitro em 1º plano vs Kenshi no fundo)
                if (m_i["height"] > 0.72 and m_j["height"] < 0.55) or (m_j["height"] > 0.72 and m_i["height"] < 0.55):
                    score -= 3.0

                if score > best_pair_score:
                    best_pair_score = score
                    best_pair = (c_i, c_j)

        # Ordenar o par vencedor da esquerda para a direita
        c_a, c_b = best_pair
        cxa, _, _ = self.extract_bbox_and_center(c_a)
        cxb, _, _ = self.extract_bbox_and_center(c_b)
        cand_left, cand_right = (c_a, c_b) if cxa <= cxb else (c_b, c_a)

        # Tratar os demais candidatos como descartados (árbitros, transeuntes ou fundo)
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
        Recebe a lista de esqueletos/poses detectados no frame, aplica:
        1. Delimitação poligonal da quadra (Shiai-jo ROI Mask).
        2. Pré-filtro postural e geométrico de planos (descarte de Shinpans e fundos).
        3. Rastreamento e travamento de IDs estrito (K=2) com predição temporal inercial.
        4. Segmentação da cor da fita vermelha (Aka Tasuki) para garantia de identidade contínua.
        5. Persistência inercial opcional para preenchimento de dropouts temporários.
        Retorna:
            - aka_landmarks: Optional[Dict]
            - shiro_landmarks: Optional[Dict]
            - discarded_items: List[Dict] com dados dos elementos descartados
        """
        def _wrap_result(a_res, s_res, d_res):
            if not return_persisted:
                return a_res, s_res, d_res
            final_a = a_res if a_res is not None else self.aka.get_persisted_landmarks(max_gap=self.track_buffer)
            final_s = s_res if s_res is not None else self.shiro.get_persisted_landmarks(max_gap=self.track_buffer)
            return final_a, final_s, d_res

        if not frame_landmarks_list:
            # Ambos os Kenshis ocluídos / sem detecção no frame
            self.aka.update(None)
            self.shiro.update(None)
            return _wrap_result(None, None, [])

        # --- ETAPA 1: FILTRAGEM POR DELIMITAÇÃO DA QUADRA (SHIAI-JO ROI) ---
        shiaijo_filtered: List[Dict[str, Any]] = []
        discarded_items: List[Dict[str, Any]] = []

        for lm in frame_landmarks_list:
            m = self.get_skeleton_metrics(lm)
            if not self.is_within_shiaijo(m["ground_x"], m["ground_y"]):
                self.discarded_out_of_shiaijo_count += 1
                discarded_items.append({
                    "landmarks": lm,
                    "plane_type": "OUT_OF_BOUNDS",
                    "scale": 1.0,
                    "reason": f"Detecção fora dos limites do Shiai-jo (Solo: X={m['ground_x']:.2f}, Y={m['ground_y']:.2f})"
                })
            else:
                shiaijo_filtered.append(lm)

        if not shiaijo_filtered:
            self.aka.update(None)
            self.shiro.update(None)
            return _wrap_result(None, None, discarded_items)

        # --- ETAPA 2: PRÉ-FILTRO POSTURAL E SELEÇÃO DE CANDIDATOS VÁLIDOS ---
        cand_left: Optional[Dict[str, Any]] = None
        cand_right: Optional[Dict[str, Any]] = None

        if len(shiaijo_filtered) <= 2:
            main_plane_candidates: List[Dict[str, Any]] = []
            for lm in shiaijo_filtered:
                plane_type, scale, reason = self.classify_plane(lm)
                if plane_type == "MAIN_PLANE":
                    main_plane_candidates.append(lm)
                else:
                    discarded_items.append({
                        "landmarks": lm,
                        "plane_type": plane_type,
                        "scale": scale,
                        "reason": reason
                    })

            if not main_plane_candidates:
                self.aka.update(None)
                self.shiro.update(None)
                return _wrap_result(None, None, discarded_items)

            if len(main_plane_candidates) == 1:
                cand_left = main_plane_candidates[0]
                cand_right = None
            else:
                c1, c2 = main_plane_candidates[0], main_plane_candidates[1]
                cx1, _, _ = self.extract_bbox_and_center(c1)
                cx2, _, _ = self.extract_bbox_and_center(c2)
                if cx1 <= cx2:
                    cand_left, cand_right = c1, c2
                else:
                    cand_left, cand_right = c2, c1
        else:
            # 3 ou mais candidatos: Usar a seleção combinatória de par ótimo e descartar Shinpans
            cand_left, cand_right, disc_shinpan = self.select_best_combatant_pair(shiaijo_filtered)
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

            cx, _, _ = self.extract_bbox_and_center(single_cand)
            target_aka_x = self.aka.pred_x if self.aka.pred_x is not None else (self.aka.last_center_x if self.aka.last_center_x is not None else 0.70)
            target_shiro_x = self.shiro.pred_x if self.shiro.pred_x is not None else (self.shiro.last_center_x if self.shiro.last_center_x is not None else 0.30)

            d_aka = abs(cx - target_aka_x)
            d_shiro = abs(cx - target_shiro_x)
            red_score = self.detect_red_flag_score(frame, single_cand) if frame is not None else 0.0
            if red_score >= 0.20:
                d_aka -= 0.30
                d_shiro += 0.30

            if self.aka.last_center_x is not None or self.shiro.last_center_x is not None:
                if d_aka <= d_shiro:
                    if self.aka.occluded_frames > 0:
                        self.occlusion_recovery_count += 1
                    self.aka.update(single_cand)
                    self.shiro.update(None)
                    res_a, res_s = (single_cand, None) if not self.invert_assignment else (None, single_cand)
                    return _wrap_result(res_a, res_s, discarded_items)
                else:
                    if self.shiro.occluded_frames > 0:
                        self.occlusion_recovery_count += 1
                    self.shiro.update(single_cand)
                    self.aka.update(None)
                    res_a, res_s = (None, single_cand) if not self.invert_assignment else (single_cand, None)
                    return _wrap_result(res_a, res_s, discarded_items)
            else:
                # Inicialização sem histórico prévio
                if not self.invert_assignment:
                    if cx <= 0.50:
                        self.shiro.update(single_cand)
                        self.aka.update(None)
                        return _wrap_result(None, single_cand, discarded_items)
                    else:
                        self.aka.update(single_cand)
                        self.shiro.update(None)
                        return _wrap_result(single_cand, None, discarded_items)
                else:
                    if cx <= 0.50:
                        self.aka.update(single_cand)
                        self.shiro.update(None)
                        return _wrap_result(single_cand, None, discarded_items)
                    else:
                        self.shiro.update(single_cand)
                        self.aka.update(None)
                        return _wrap_result(None, single_cand, discarded_items)

        # --- ETAPA 3: AMOSTRAGEM DA FITA VERMELHA (TASUKI) NAS COSTAS ---
        score_left = 0.0
        score_right = 0.0
        if frame is not None:
            score_left = self.detect_red_flag_score(frame, cand_left)
            score_right = self.detect_red_flag_score(frame, cand_right)

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

            # Distâncias Euclidianas 2D completas
            dist_l_to_shiro = float(np.hypot(cxl - target_shiro_x, cyl - target_shiro_y))
            dist_r_to_aka = float(np.hypot(cxr - target_aka_x, cyr - target_aka_y))

            dist_l_to_aka = float(np.hypot(cxl - target_aka_x, cyl - target_aka_y))
            dist_r_to_shiro = float(np.hypot(cxr - target_shiro_x, cyr - target_shiro_y))

            # Opção A: cand_left = Shiro, cand_right = Aka
            cost_a = dist_l_to_shiro + dist_r_to_aka
            if score_left >= 0.15: cost_a += 1.50
            if score_right >= 0.15: cost_a -= 0.50

            # Opção B: cand_left = Aka, cand_right = Shiro
            cost_b = dist_l_to_aka + dist_r_to_shiro
            if score_left >= 0.15: cost_b -= 0.50
            if score_right >= 0.15: cost_b += 1.50

            if cost_b < cost_a:
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

            if self.lock_tracks and aka_lm and shiro_lm:
                self.tracking_state = "LOCKED_COMBAT"
                self.aka.is_locked = True
                self.shiro.is_locked = True

        # Inversão manual se solicitada pelo operador
        if self.invert_assignment:
            aka_lm, shiro_lm = shiro_lm, aka_lm

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
