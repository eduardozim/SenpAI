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

        Configuração Padrão de Posição (Câmera Oposta à Mesa dos Juízes):
        Em Kendo oficial, observando do lado oposto à mesa dos juízes (visão padrão da câmera):
        - Esquerda do enquadramento (x <= 0.50) = Kenshi Shiro (Branco)
        - Direita do enquadramento (x > 0.50) = Kenshi Aka (Vermelho)
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
        self.flag_decision = "POSITION_DEFAULT_OPPOSITE_JUDGES"
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
        score = 0.0

        # 1. Proximidade dos pulsos (Empunhadura bimanual do Shinai)
        r_wrist = landmarks.get("RIGHT_WRIST")
        l_wrist = landmarks.get("LEFT_WRIST")
        if r_wrist and l_wrist and "x" in r_wrist and "x" in l_wrist:
            wrist_dist = np.hypot(r_wrist["x"] - l_wrist["x"], r_wrist["y"] - l_wrist["y"])
            # No Kendo Kamae, as duas mãos seguram o mesmo cabo (< 0.18 de h)
            if wrist_dist < (0.18 * h):
                score += 0.40
            elif wrist_dist < (0.28 * h):
                score += 0.25
            else:
                # Mãos bem abertas / separadas (característico de árbitro com bandeiras nas duas mãos)
                score -= 0.10

        # 2. Posição no Shiaijo (Kenshis no miolo, árbitros nas extremidades)
        dist_center_x = abs(cx - 0.50)
        if dist_center_x <= 0.28:
            score += 0.30 * (1.0 - (dist_center_x / 0.28))
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
                score += 0.30  # Sonkyō ou agachamento atlético

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
        - "FOREGROUND_OCCLUDER": Primeiro plano excessivo / Árbitro em primeiro plano na frente dos Kenshis.
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
        is_on_ground_line = (m["ground_y"] >= (ref_ground - self.ground_tolerance))
        is_bg_distant = (m["ground_y"] < (ref_ground - self.ground_tolerance)) and (scale_h < 0.80)
        is_bg_tiny = (scale_h < 0.35) and (scale_area < 0.18)

        if (is_bg_distant or is_bg_tiny) and not is_on_ground_line:
            self.discarded_background_count += 1
            reason = f"Elemento de Segundo Plano detectado (Escala: {scale_h:.2f}x, Área: {scale_area:.2f}x ref, Pé Y: {m['ground_y']:.2f})"
            return "BACKGROUND", float(scale_h), reason

        # 2. Verificação de Primeiro Plano Excessivo / Árbitro Oclusor (FOREGROUND_OCCLUDER)
        is_fg_scale = (scale_h > self.max_fg_ratio) or (scale_area > (self.max_fg_ratio ** 2))
        is_fg_edge_crop = (m["ymin"] <= 0.01 and m["ymax"] >= 0.98 and scale_area > 1.2)

        if is_fg_scale or is_fg_edge_crop:
            self.discarded_foreground_count += 1
            reason = f"Árbitro/Oclusão na frente da câmera (Escala: {scale_h:.2f}x, Área: {scale_area:.2f}x ref)"
            return "FOREGROUND_OCCLUDER", float(scale_h), reason

        # 3. Classificação como Plano Principal (MAIN_PLANE)
        return "MAIN_PLANE", float(scale_h), "Plano Principal de Combate"

    def select_best_combatant_pair(
        self,
        candidates: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Dentre todos os esqueletos detectados no frame (que podem incluir Kenshis, Shinpans/árbitros e transeuntes),
        seleciona com precisão o par (Kenshi_Left, Kenshi_Right) que melhor representa os 2 combatentes.
        
        Isola os Kenshis eliminando Shinpans em primeiro plano através de:
        - Empunhadura de Shinai (mãos unidas no cabo) vs mãos com bandeiras.
        - Compatibilidade mútua de escala de profundidade (os 2 Kenshis compartilham o mesmo plano no Shiaijo).
        - Centralidade no Shiaijo e distância de Maai.
        - Alinhamento da linha de solo dos pés.
        """
        if not candidates:
            return None, None, []

        if len(candidates) == 1:
            return candidates[0], None, []

        if len(candidates) == 2:
            c1, c2 = candidates[0], candidates[1]
            cx1, _, _ = self.extract_bbox_and_center(c1)
            cx2, _, _ = self.extract_bbox_and_center(c2)
            if cx1 <= cx2:
                return c1, c2, []
            else:
                return c2, c1, []

        # Para 3 ou mais candidatos: avaliar combinatória de pares (i, j)
        m_list = [self.get_skeleton_metrics(c) for c in candidates]
        k_scores = [self.compute_kenshi_feature_score(c) for c in candidates]

        best_pair_score = -999.0
        best_pair = (candidates[0], candidates[1])

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

                # 5. Distância mútua de combate (Maai típico: 0.10 a 0.65 de distância horizontal)
                if 0.10 <= pair_dist <= 0.65:
                    score += 1.5
                else:
                    score -= 1.0

                # 6. Continuidade temporal com posições rastreadas anteriormente
                if self.aka.last_center_x is not None and self.shiro.last_center_x is not None:
                    d_tracked = min(
                        abs(cx_i - self.aka.last_center_x) + abs(cx_j - self.shiro.last_center_x),
                        abs(cx_i - self.shiro.last_center_x) + abs(cx_j - self.aka.last_center_x)
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
                rel_scale = float(m_c["height"] / max(0.01, ref_h))
                if rel_scale > 1.25:
                    p_type = "FOREGROUND_OCCLUDER"
                    reason = f"Árbitro (Shinpan) em 1º plano descartado (Escala: {rel_scale:.2f}x)"
                    self.discarded_foreground_count += 1
                else:
                    p_type = "BACKGROUND"
                    reason = f"Elemento de 2º plano / Árbitro ao fundo descartado (Escala: {rel_scale:.2f}x)"
                    self.discarded_background_count += 1

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
        frame: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Recebe a lista de esqueletos/poses detectados no frame, filtra ruídos de planos e árbitros (Shinpans),
        analisa a presença da fita vermelha nas costas (Aka Tasukuki) e associa aos 2 combatentes (Aka e Shiro).
        Retorna:
            - aka_landmarks: Optional[Dict]
            - shiro_landmarks: Optional[Dict]
            - discarded_items: List[Dict] com dados dos elementos descartados
        """
        if not frame_landmarks_list:
            return None, None, []

        # Se houver 1 ou 2 candidatos, fazer a triagem clássica com classify_plane
        if len(frame_landmarks_list) <= 2:
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

            if not main_plane_candidates:
                return None, None, discarded_items

            if len(main_plane_candidates) == 1:
                cand = main_plane_candidates[0]
                cx, _, _ = self.extract_bbox_and_center(cand)
                
                if self.aka.last_center_x is not None and self.shiro.last_center_x is not None:
                    d_aka = abs(cx - self.aka.last_center_x)
                    d_shiro = abs(cx - self.shiro.last_center_x)
                    if d_aka <= d_shiro:
                        return cand, None, discarded_items
                    else:
                        return None, cand, discarded_items
                else:
                    if not self.invert_assignment:
                        if cx <= 0.50:
                            return None, cand, discarded_items  # Shiro à esquerda
                        else:
                            return cand, None, discarded_items  # Aka à direita
                    else:
                        if cx <= 0.50:
                            return cand, None, discarded_items  # Aka à esquerda (invertido)
                        else:
                            return None, cand, discarded_items  # Shiro à direita (invertido)

            cand_left, cand_right = main_plane_candidates[0], main_plane_candidates[1]
            cx1, _, _ = self.extract_bbox_and_center(cand_left)
            cx2, _, _ = self.extract_bbox_and_center(cand_right)
            if cx1 > cx2:
                cand_left, cand_right = cand_right, cand_left
        else:
            # 3 ou mais candidatos: Usar a seleção do melhor par de combatentes com discriminação de árbitro (Shinpan)
            cand_left, cand_right, discarded_items = self.select_best_combatant_pair(frame_landmarks_list)
            if not cand_left and not cand_right:
                return None, None, discarded_items

        # Amostragem da Cor da Flag (Tasukuki Vermelho) nas costas de cada combatente
        if frame is not None and cand_left and cand_right:
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
            # O lutador da DIREITA possui a flag vermelha (Aka)
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
            # Padrão da câmera oposta à mesa dos juízes: Shiro à esquerda, Aka à direita
            aka_lm = cand_right
            shiro_lm = cand_left
            self.flag_decision = "POSITION_DEFAULT_OPPOSITE_JUDGES"
            self.flag_confidence = 0.50

        # Se o usuário solicitou inversão manual de Aka e Shiro
        if self.invert_assignment:
            aka_lm, shiro_lm = shiro_lm, aka_lm

        # Calibrar plano principal se ainda não calibrado
        if not self.is_calibrated and aka_lm and shiro_lm:
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
