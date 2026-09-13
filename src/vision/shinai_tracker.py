"""
Módulo de Rastreamento Avançado do Shinai (Espada de Bambu) e Regiões de Ataque Alvo.
Combina visão computacional (segmentação de cor de bambu, detecção de linhas e gradientes)
com cinemática postural bimanual e filtragem temporal inercial para ancorar com máxima
estabilidade a posição, orientação corporal e alvos anatômicos dos Kendocas.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


class ShinaiTracker:
    def __init__(self, shinai_length_ratio: float = 1.6):
        """
        shinai_length_ratio: Proporção aproximada entre a distância cotovelo-pulso e o comprimento da lâmina do Shinai.
        """
        self.shinai_length_ratio = shinai_length_ratio

    def track_shinai(
        self,
        frame: Optional[np.ndarray],
        landmarks: Optional[Dict[str, Any]],
        prev_state: Optional[Dict[str, Any]] = None,
        expected_facing: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Rastreia e mapeia o Shinai (base, lâmina e ponta/Kensen) a partir dos landmarks das mãos
        e das características visuais (bambu/linhas retas) presentes no frame.

        Retorna dicionário com:
          - detected: bool
          - base_norm: (x, y) normalizado
          - tip_norm: (x, y) normalizado
          - base_px: (px, py) pixels
          - tip_px: (px, py) pixels
          - angle_deg: float (graus em relação ao plano horizontal)
          - length_norm: float
          - facing: "RIGHT", "LEFT" ou "CENTER"
          - confidence: float (0.0 a 1.0)
          - is_visual: bool (True se detectado por visão, False se cinemático)
        """
        if not landmarks:
            return None

        h, w = (frame.shape[:2]) if frame is not None else (480, 640)

        # 1. Localizar centro das mãos / empunhadura do Tsuka
        r_w = landmarks.get("RIGHT_WRIST")
        l_w = landmarks.get("LEFT_WRIST")
        r_e = landmarks.get("RIGHT_ELBOW") or landmarks.get("LEFT_ELBOW") or landmarks.get("RIGHT_SHOULDER")

        if not r_w and not l_w:
            return None

        # Ponto base (Tsuka): centro ponderado entre as mãos (mão direita à frente)
        if r_w and l_w and "x" in r_w and "x" in l_w:
            bx = 0.65 * float(r_w["x"]) + 0.35 * float(l_w["x"])
            by = 0.65 * float(r_w["y"]) + 0.35 * float(l_w["y"])
        else:
            main_w = r_w if r_w else l_w
            bx = float(main_w["x"])
            by = float(main_w["y"])

        b_px = (int(np.clip(bx * w, 0, w - 1)), int(np.clip(by * h, 0, h - 1)))

        # Vetor do antebraço para estimativa de direção padrão
        forearm_vec = np.array([0.0, -1.0])
        forearm_len = 0.15
        if r_w and r_e and "x" in r_w and "x" in r_e:
            dx = float(r_w["x"]) - float(r_e["x"])
            dy = float(r_w["y"]) - float(r_e["y"])
            norm = float(np.hypot(dx, dy))
            if norm > 0.02:
                forearm_vec = np.array([dx / norm, dy / norm])
                forearm_len = norm

        # Orientação esperada (se Kendoca está à esquerda e ataca para a direita ou vice-versa)
        if expected_facing == "RIGHT" and forearm_vec[0] < -0.2:
            forearm_vec[0] = abs(forearm_vec[0])
        elif expected_facing == "LEFT" and forearm_vec[0] > 0.2:
            forearm_vec[0] = -abs(forearm_vec[0])

        visual_tip_px = None
        is_visual = False
        visual_conf = 0.0

        # 2. Mapeamento Visual no Frame (Bambu + Linhas de Alta Rigidez)
        if frame is not None:
            # Definir ROI ao redor das mãos se estendendo na direção provável da espada
            # Dimensão da ROI: ~25% da altura da imagem
            roi_rad = int(max(60, min(w, h) * 0.28))
            y1 = max(0, b_px[1] - roi_rad)
            y2 = min(h, b_px[1] + roi_rad)
            x1 = max(0, b_px[0] - roi_rad)
            x2 = min(w, b_px[0] + roi_rad)

            roi = frame[y1:y2, x1:x2]
            if roi.shape[0] > 20 and roi.shape[1] > 20:
                # Otimização de performance: redimensionar para resolução máxima de 180px para acelerar Canny/Hough
                scale_factor = 1.0
                if max(roi.shape[0], roi.shape[1]) > 180:
                    scale_factor = 160.0 / float(max(roi.shape[0], roi.shape[1]))
                    roi_proc = cv2.resize(roi, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                else:
                    roi_proc = roi

                hsv = cv2.cvtColor(roi_proc, cv2.COLOR_BGR2HSV)
                # Máscara de cor de bambu (tons bege, amarelo-claro e madeira clara)
                mask = cv2.inRange(hsv, np.array([12, 20, 75], dtype=np.uint8), np.array([42, 225, 255], dtype=np.uint8))

                # Detecção de bordas nas faixas de bambu
                edges = cv2.Canny(mask, 40, 140)

                # Buscar segmentos lineares pelo HoughLinesP em resolução otimizada
                min_len = max(12, int(22 * scale_factor))
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=16, minLineLength=min_len, maxLineGap=int(10 * scale_factor))

                if lines is not None and len(lines) > 0:
                    best_line = None
                    best_score = -1e9
                    base_in_roi = np.array([(b_px[0] - x1) * scale_factor, (b_px[1] - y1) * scale_factor], dtype=np.float32)

                    for line in lines:
                        coords = np.asarray(line).ravel()
                        if len(coords) < 4:
                            continue
                        lx1, ly1, lx2, ly2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        p1 = np.array([lx1, ly1], dtype=np.float32)
                        p2 = np.array([lx2, ly2], dtype=np.float32)

                        seg_vec = p2 - p1
                        seg_len = float(np.linalg.norm(seg_vec))
                        if seg_len < (14 * scale_factor):
                            continue

                        # O Kensen é a ponta mais distante da base
                        d1 = float(np.linalg.norm(p1 - base_in_roi))
                        d2 = float(np.linalg.norm(p2 - base_in_roi))

                        tip_pt = p2 if d1 <= d2 else p1
                        dist_to_root = min(d1, d2)

                        dir_vec = (tip_pt - base_in_roi)
                        dir_norm = float(np.linalg.norm(dir_vec))
                        if dir_norm < (8 * scale_factor):
                            continue
                        dir_unit = dir_vec / dir_norm

                        # Pontuação da linha:
                        score = seg_len * 1.5 - dist_to_root * 1.8
                        align = float(dir_unit[0] * forearm_vec[0] + dir_unit[1] * forearm_vec[1])
                        score += align * 35.0

                        if expected_facing == "RIGHT" and dir_unit[0] > 0.1:
                            score += 25.0
                        elif expected_facing == "LEFT" and dir_unit[0] < -0.1:
                            score += 25.0

                        if score > best_score:
                            best_score = score
                            # Mapear de volta à escala original do ROI
                            orig_tip_x = tip_pt[0] / scale_factor
                            orig_tip_y = tip_pt[1] / scale_factor
                            best_line = (int(orig_tip_x + x1), int(orig_tip_y + y1), seg_len / scale_factor)

                    if best_line is not None and best_score > 8.0:
                        visual_tip_px = (best_line[0], best_line[1])
                        is_visual = True
                        visual_conf = float(np.clip(best_score / (100.0 * scale_factor), 0.70, 0.95))

        # 3. Fallback Cinemático Suave se não houver linha visual nítida
        estimated_len = max(0.18, forearm_len * self.shinai_length_ratio)
        if visual_tip_px is not None:
            tx = float(np.clip(visual_tip_px[0] / max(1, w), 0.0, 1.0))
            ty = float(np.clip(visual_tip_px[1] / max(1, h), 0.0, 1.0))
            conf = visual_conf
        else:
            tx = float(np.clip(bx + forearm_vec[0] * estimated_len, 0.0, 1.0))
            ty = float(np.clip(by + forearm_vec[1] * estimated_len, 0.0, 1.0))
            conf = 0.55

        # 4. Filtragem Temporal Inercial (Suavização Exponencial)
        if prev_state and prev_state.get("tip_norm") is not None:
            prev_tx, prev_ty = prev_state["tip_norm"]
            # Suavizar ruídos de alta frequência
            alpha = 0.70 if is_visual else 0.40
            tx = float(alpha * tx + (1.0 - alpha) * prev_tx)
            ty = float(alpha * ty + (1.0 - alpha) * prev_ty)

        t_px = (int(np.clip(tx * w, 0, w - 1)), int(np.clip(ty * h, 0, h - 1)))

        # 5. Cálculo de Vetores, Ângulo e Orientação Corporal (Facing Direction)
        dx = tx - bx
        dy = ty - by
        actual_len = float(np.hypot(dx, dy))
        angle_rad = np.arctan2(-dy, dx) # Positivo para cima
        angle_deg = float(np.degrees(angle_rad))

        if dx > 0.035:
            facing = "RIGHT"
        elif dx < -0.035:
            facing = "LEFT"
        else:
            facing = expected_facing or "CENTER"

        return {
            "detected": True,
            "base_norm": (bx, by),
            "tip_norm": (tx, ty),
            "base_px": b_px,
            "tip_px": t_px,
            "angle_deg": angle_deg,
            "length_norm": actual_len,
            "facing": facing,
            "confidence": conf,
            "is_visual": is_visual
        }

    def estimate_shinai_tip(self, landmarks: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        """
        Estima as coordenadas 3D (x, y, z) do Kensen (ponta do Shinai) a partir da posição dos pulsos e cotovelos.
        Mantido para retrocompatibilidade.
        """
        if not landmarks:
            return None

        res = self.track_shinai(frame=None, landmarks=landmarks)
        if res and "tip_norm" in res:
            tx, ty = res["tip_norm"]
            return float(tx), float(ty), 0.0

        return None

    @staticmethod
    def get_target_zones(landmarks: Dict[str, Any]) -> Dict[str, Tuple[float, float, float]]:
        """
        Mapeia os alvos anatômicos válidos de Kendo a partir das articulações do praticante.
        - Men: Topo do capacete / Cabeça
        - Kote: Pulso/Antebraço direito
        - Do: Lateral do tronco (flanco direito/esquerdo)
        - Tsuki: Garganta (Abaixo do queixo, centro das clavículas)
        """
        targets = {}
        if not landmarks:
            return targets

        # 1. Men (Topo da Cabeça)
        nose_pt = landmarks.get("NOSE") or {"x": 0.5, "y": 0.25, "z": 0.0}
        r_ear_pt = landmarks.get("RIGHT_EAR") or nose_pt
        l_ear_pt = landmarks.get("LEFT_EAR") or nose_pt

        nose = np.array([nose_pt["x"], nose_pt["y"], nose_pt.get("z", 0.0)])
        r_ear = np.array([r_ear_pt["x"], r_ear_pt["y"], r_ear_pt.get("z", 0.0)])
        l_ear = np.array([l_ear_pt["x"], l_ear_pt["y"], l_ear_pt.get("z", 0.0)])
        head_center = (nose + r_ear + l_ear) / 3.0
        targets["MEN"] = (float(head_center[0]), float(head_center[1] - 0.05), float(head_center[2]))

        # 2. Kote (Pulso Direito)
        r_wrist_pt = landmarks.get("RIGHT_WRIST") or landmarks.get("LEFT_WRIST")
        if r_wrist_pt:
            targets["KOTE"] = (r_wrist_pt["x"], r_wrist_pt["y"], r_wrist_pt.get("z", 0.0))

        # 3. Do (Lateral do Tronco - Hip/Ribs)
        r_hip_pt = landmarks.get("RIGHT_HIP") or landmarks.get("LEFT_HIP")
        r_sh_pt = landmarks.get("RIGHT_SHOULDER") or landmarks.get("LEFT_SHOULDER")
        if r_hip_pt and r_sh_pt:
            r_hip = np.array([r_hip_pt["x"], r_hip_pt["y"], r_hip_pt.get("z", 0.0)])
            r_shoulder = np.array([r_sh_pt["x"], r_sh_pt["y"], r_sh_pt.get("z", 0.0)])
            do_point = (r_hip * 0.6) + (r_shoulder * 0.4)
            targets["DO"] = (float(do_point[0]), float(do_point[1]), float(do_point[2]))

        # 4. Tsuki (Garganta)
        l_sh_pt = landmarks.get("LEFT_SHOULDER") or r_sh_pt
        if r_sh_pt and l_sh_pt:
            r_shoulder = np.array([r_sh_pt["x"], r_sh_pt["y"], r_sh_pt.get("z", 0.0)])
            l_shoulder = np.array([l_sh_pt["x"], l_sh_pt["y"], l_sh_pt.get("z", 0.0)])
            shoulder_center = (r_shoulder + l_shoulder) / 2.0
            throat_point = (shoulder_center * 0.7) + (nose * 0.3)
            targets["TSUKI"] = (float(throat_point[0]), float(throat_point[1]), float(throat_point[2]))

        return targets
