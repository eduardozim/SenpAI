"""
Módulo de Rastreamento dos 2 Combatentes Principais e Filtragem de Planos.
Identifica e rastreia os 2 Kenshi (Aka & Shiro) que realizaram o Sonkyō inicial de abertura,
descartando elementos de segundo plano (outras lutas/fundo) e transeuntes de primeiro plano (frente da câmera).
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
        self.history: List[Optional[Dict[str, Any]]] = []

    def update(self, landmarks: Optional[Dict[str, Any]]):
        self.history.append(landmarks)
        if landmarks:
            cx, cy, bbox = CombatantTracker.extract_bbox_and_center(landmarks)
            self.last_center_x = cx
            self.last_center_y = cy
            self.last_bbox = bbox


class CombatantTracker:
    def __init__(
        self,
        min_background_scale_ratio: float = 0.68,
        max_foreground_scale_ratio: float = 1.38,
        ground_line_tolerance: float = 0.14,
        invert_assignment: bool = False
    ):
        """
        - min_background_scale_ratio: Limiar abaixo do qual o elemento é classificado como Segundo Plano (Fundo).
        - max_foreground_scale_ratio: Limiar acima do qual o elemento é classificado como Oclusão de Primeiro Plano (Frente da Câmera).
        - ground_line_tolerance: Tolerância de deslocamento vertical dos pés em relação ao solo do Shiaijo.
        - invert_assignment: Se True, inverte manualmente as identidades de Aka e Shiro.
        """
        self.min_bg_ratio = min_background_scale_ratio
        self.max_fg_ratio = max_foreground_scale_ratio
        self.ground_tolerance = ground_line_tolerance
        self.invert_assignment = invert_assignment

        # Perfis dos 2 lutadores
        self.aka = CombatantProfile("KENSHI_AKA", "Kenshi Aka (Vermelho)", (40, 40, 230)) # Vermelho BGR
        self.shiro = CombatantProfile("KENSHI_SHIRO", "Kenshi Shiro (Branco)", (240, 240, 240)) # Branco BGR

        # Rastreamento de Evidência da Flag Vermelha (Tasukuki nas costas)
        self.candidate_left_red_score = 0.0
        self.candidate_right_red_score = 0.0
        self.red_evidence_frames_left = 0
        self.red_evidence_frames_right = 0
        self.flag_decision = "POSITION_DEFAULT"
        self.flag_confidence = 0.50

        # Referência do Plano Principal de Combate (calibrado a partir do Sonkyō ou dos primeiros frames)
        self.is_calibrated = False
        self.ref_height = 0.60
        self.ref_bbox_area = 0.15
        self.ref_ground_y = 0.88
        self.ref_shoulder_width = 0.18

        # Contadores estatísticos de descartes
        self.discarded_background_count = 0
        self.discarded_foreground_count = 0
        self.total_detections_processed = 0

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

        red_pixels = int(cv2.countNonZero(red_mask))
        total_pixels = int(roi.shape[0] * roi.shape[1])
        if total_pixels == 0:
            return 0.0

        red_ratio = red_pixels / float(total_pixels)
        score = float(np.clip(red_ratio / 0.05, 0.0, 1.0))
        return score

    @staticmethod
    def extract_bbox_and_center(landmarks: Dict[str, Any]) -> Tuple[float, float, Tuple[float, float, float, float]]:
        """Extrai centro (cx, cy) e bounding box normalizado (xmin, ymin, xmax, ymax)."""
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
    def get_skeleton_metrics(landmarks: Dict[str, Any]) -> Dict[str, float]:
        """Calcula métricas de escala, altura, área e linha dos pés de um conjunto de landmarks."""
        _, _, (xmin, ymin, xmax, ymax) = CombatantTracker.extract_bbox_and_center(landmarks)
        w = max(0.01, xmax - xmin)
        h = max(0.01, ymax - ymin)
        area = w * h

        # Posição dos pés / tornozelos
        if "RIGHT_ANKLE" in landmarks and "LEFT_ANKLE" in landmarks:
            ground_y = (landmarks["RIGHT_ANKLE"]["y"] + landmarks["LEFT_ANKLE"]["y"]) / 2.0
        elif "RIGHT_FOOT_INDEX" in landmarks:
            ground_y = landmarks["RIGHT_FOOT_INDEX"]["y"]
        else:
            ground_y = ymax

        # Largura dos ombros
        if "RIGHT_SHOULDER" in landmarks and "LEFT_SHOULDER" in landmarks:
            shoulder_w = abs(landmarks["RIGHT_SHOULDER"]["x"] - landmarks["LEFT_SHOULDER"]["x"])
        else:
            shoulder_w = w * 0.45

        return {
            "width": float(w),
            "height": float(h),
            "area": float(area),
            "ground_y": float(ground_y),
            "shoulder_width": float(shoulder_w),
            "xmin": float(xmin),
            "ymin": float(ymin),
            "xmax": float(xmax),
            "ymax": float(ymax)
        }

    def calibrate_main_plane(self, candidate_poses: List[Dict[str, Any]]):
        """
        Calibra as métricas de referência do Plano Principal com base nas poses dos dois lutadores.
        """
        if not candidate_poses:
            return

        valid_metrics = [self.get_skeleton_metrics(p) for p in candidate_poses if p]
        if not valid_metrics:
            return

        # Considerar altura de pé típica (mínimo de 0.45 para evitar calibrar com valores agachados)
        heights = [m["height"] for m in valid_metrics]
        max_h = max(heights) if heights else 0.60
        self.ref_height = float(max(0.45, max_h))
        self.ref_bbox_area = float(np.mean([m["area"] for m in valid_metrics]))
        self.ref_ground_y = float(np.mean([m["ground_y"] for m in valid_metrics]))
        self.ref_shoulder_width = float(np.mean([m["shoulder_width"] for m in valid_metrics]))
        self.is_calibrated = True

    def classify_plane(self, landmarks: Optional[Dict[str, Any]]) -> Tuple[str, float, str]:
        """
        Classifica um esqueleto detectado em:
        - "MAIN_PLANE": Pertence aos 2 Kenshi no plano principal da luta.
        - "BACKGROUND": Segundo plano / fundo (outras lutas, árbitros distantes, arquibancada).
        - "FOREGROUND_OCCLUDER": Primeiro plano excessivo (pessoa passando na frente da câmera).
        - "INVALID": Dados insuficientes.

        Retorna: (plane_type, scale_factor, reason)
        """
        if not landmarks:
            return "INVALID", 0.0, "Sem landmarks válidos"

        self.total_detections_processed += 1
        m = self.get_skeleton_metrics(landmarks)

        # Se ainda não calibrado, usar valores base típicos
        ref_h = self.ref_height if self.is_calibrated else 0.60
        ref_area = self.ref_bbox_area if self.is_calibrated else 0.15
        ref_ground = self.ref_ground_y if self.is_calibrated else 0.88

        scale_h = m["height"] / max(0.01, ref_h)
        scale_area = m["area"] / max(0.001, ref_area)

        # 1. Verificação de Segundo Plano (BACKGROUND)
        # Se os pés estiverem fisicamente no solo do Shiaijo (ground_y), manter como MAIN_PLANE mesmo se agachado (Sonkyō)
        is_on_ground_line = (m["ground_y"] >= (ref_ground - self.ground_tolerance))
        
        # Descartar como background apenas se a linha dos pés estiver visivelmente elevada (ao fundo) E a escala for reduzida,
        # ou se a escala for extremamente minúscula (menos de 35% de ref_h)
        is_bg_distant = (m["ground_y"] < (ref_ground - self.ground_tolerance)) and (scale_h < 0.80)
        is_bg_tiny = (scale_h < 0.35) and (scale_area < 0.18)

        if (is_bg_distant or is_bg_tiny) and not is_on_ground_line:
            self.discarded_background_count += 1
            reason = f"Elemento de Segundo Plano detectado (Escala: {scale_h:.2f}x, Área: {scale_area:.2f}x ref, Pé Y: {m['ground_y']:.2f})"
            return "BACKGROUND", float(scale_h), reason

        # 2. Verificação de Primeiro Plano Excessivo / Oclusão (FOREGROUND_OCCLUDER)
        # Se a escala for desproporcionalmente grande ou cortar drasticamente o enquadramento
        is_fg_scale = (scale_h > self.max_fg_ratio) or (scale_area > (self.max_fg_ratio ** 2))
        is_fg_edge_crop = (m["ymin"] <= 0.01 and m["ymax"] >= 0.98 and scale_area > 1.2)

        if is_fg_scale or is_fg_edge_crop:
            self.discarded_foreground_count += 1
            reason = f"Transeunte/Oclusão na frente da câmera (Escala: {scale_h:.2f}x, Área: {scale_area:.2f}x ref)"
            return "FOREGROUND_OCCLUDER", float(scale_h), reason

        # 3. Classificação como Plano Principal (MAIN_PLANE)
        return "MAIN_PLANE", float(scale_h), "Plano Principal de Combate"

    def associate_and_filter(
        self,
        frame_landmarks_list: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Recebe a lista de esqueletos/poses detectados no frame, filtra ruídos de planos diferentes,
        analisa a presença da fita vermelha nas costas (Aka Tasukuki) e associa aos 2 combatentes (Aka e Shiro).
        Retorna:
            - aka_landmarks: Optional[Dict]
            - shiro_landmarks: Optional[Dict]
            - discarded_items: List[Dict] com dados dos elementos descartados
        """
        if not frame_landmarks_list:
            return None, None, []

        main_plane_candidates = []
        discarded_items = []

        for lm in frame_landmarks_list:
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

        # Se nenhum candidato no plano principal
        if not main_plane_candidates:
            return None, None, discarded_items

        # Se tiver 1 candidato no plano principal
        if len(main_plane_candidates) == 1:
            cand = main_plane_candidates[0]
            cx, _, _ = self.extract_bbox_and_center(cand)
            
            # Se Aka estava mais à esquerda
            if self.aka.last_center_x is not None and self.shiro.last_center_x is not None:
                d_aka = abs(cx - self.aka.last_center_x)
                d_shiro = abs(cx - self.shiro.last_center_x)
                if d_aka <= d_shiro:
                    return cand, None, discarded_items
                else:
                    return None, cand, discarded_items
            else:
                # Default: Se x < 0.5 é Aka (lado esquerdo), senão Shiro
                if not self.invert_assignment:
                    if cx <= 0.50:
                        return cand, None, discarded_items
                    else:
                        return None, cand, discarded_items
                else:
                    if cx <= 0.50:
                        return None, cand, discarded_items
                    else:
                        return cand, None, discarded_items

        # Se tiver 2 ou mais candidatos no plano principal:
        # Ordenar geometricamente no Shiaijo (candidato à esquerda e candidato à direita)
        main_plane_candidates.sort(key=lambda lm: self.extract_bbox_and_center(lm)[0])
        cand_left = main_plane_candidates[0]
        cand_right = main_plane_candidates[1]

        # Amostragem da Cor da Flag (Tasukuki Vermelho) nas costas de cada combatente
        if frame is not None:
            score_left = self.detect_red_flag_score(frame, cand_left)
            score_right = self.detect_red_flag_score(frame, cand_right)

            if score_left >= 0.10:
                self.candidate_left_red_score += score_left
                self.red_evidence_frames_left += 1

            if score_right >= 0.10:
                self.candidate_right_red_score += score_right
                self.red_evidence_frames_right += 1

        # Decisão de Atribuição de Identidade baseada na evidência da Flag Vermelha
        diff = self.candidate_right_red_score - self.candidate_left_red_score
        
        if diff >= 0.40:
            # O lutador da DIREITA possui a flag vermelha (Aka) -> Câmera invertida ou posição invertida
            aka_lm = cand_right
            shiro_lm = cand_left
            self.flag_decision = "FLAG_DETECTED_RIGHT_IS_AKA"
            self.flag_confidence = float(np.clip(diff / 2.0, 0.60, 0.98))
        elif diff <= -0.40:
            # O lutador da ESQUERDA possui a flag vermelha (Aka)
            aka_lm = cand_left
            shiro_lm = cand_right
            self.flag_decision = "FLAG_DETECTED_LEFT_IS_AKA"
            self.flag_confidence = float(np.clip(abs(diff) / 2.0, 0.60, 0.98))
        else:
            # Evidência insuficiente / inicial: usar padrão posicional do Shiaijo (Aka à esquerda)
            aka_lm = cand_left
            shiro_lm = cand_right
            self.flag_decision = "POSITION_DEFAULT"
            self.flag_confidence = 0.50

        # Se o usuário solicitou inversão manual de Aka e Shiro
        if self.invert_assignment:
            aka_lm, shiro_lm = shiro_lm, aka_lm

        # Se houver excedentes além dos 2 principais no mesmo frame, descartar os mais distantes
        if len(main_plane_candidates) > 2:
            for extra_lm in main_plane_candidates[2:]:
                discarded_items.append({
                    "landmarks": extra_lm,
                    "plane_type": "BACKGROUND",
                    "scale": 1.0,
                    "reason": "Excedente de combatentes no plano de combate"
                })
                self.discarded_background_count += 1

        # Calibrar plano principal se ainda não calibrado
        if not self.is_calibrated:
            self.calibrate_main_plane([aka_lm, shiro_lm])

        self.aka.update(aka_lm)
        self.shiro.update(shiro_lm)

        return aka_lm, shiro_lm, discarded_items

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo das estatísticas de rastreamento, filtragem de planos e detecção de flag."""
        return {
            "is_calibrated": self.is_calibrated,
            "ref_height": round(self.ref_height, 3),
            "ref_bbox_area": round(self.ref_bbox_area, 3),
            "ref_ground_y": round(self.ref_ground_y, 3),
            "discarded_background_count": self.discarded_background_count,
            "discarded_foreground_count": self.discarded_foreground_count,
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
