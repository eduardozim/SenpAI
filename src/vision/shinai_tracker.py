"""
Módulo de rastreamento do Shinai (Espada) e Regiões de Ataca Target (Men, Kote, Do, Tsuki).
Utiliza a extensão vetorial dos pulsos/mãos para estimar a ponta da espada (Kensen) e os pontos de contato.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

class ShinaiTracker:
    def __init__(self, shinai_length_ratio: float = 1.6):
        """
        shinai_length_ratio: Proporção aproximada entre a distância cotovelo-pulso e o comprimento da lâmina do Shinai.
        """
        self.shinai_length_ratio = shinai_length_ratio

    def estimate_shinai_tip(self, landmarks: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        """
        Estima as coordenadas 3D (x, y, z) do Kensen (ponta do Shinai) a partir da posição dos pulsos e cotovelos.
        """
        if not landmarks:
            return None

        # Posição dos pulsos e cotovelos
        r_w_pt = landmarks.get("RIGHT_WRIST") or landmarks.get("LEFT_WRIST")
        r_e_pt = landmarks.get("RIGHT_ELBOW") or landmarks.get("LEFT_ELBOW") or landmarks.get("RIGHT_SHOULDER")
        if not r_w_pt or not r_e_pt:
            return None

        r_wrist = np.array([r_w_pt["x"], r_w_pt["y"], r_w_pt.get("z", 0.0)])
        r_elbow = np.array([r_e_pt["x"], r_e_pt["y"], r_e_pt.get("z", 0.0)])
        
        l_w_pt = landmarks.get("LEFT_WRIST") or r_w_pt
        l_wrist = np.array([l_w_pt["x"], l_w_pt["y"], l_w_pt.get("z", 0.0)])

        # Centro do punho (mão direita na frente, mão esquerda na base do Tsuka)
        hand_center = (r_wrist * 0.7) + (l_wrist * 0.3)
        
        # Vetor de direção do antebraço direito
        forearm_vec = r_wrist - r_elbow
        norm = np.linalg.norm(forearm_vec)
        if norm == 0:
            return tuple(hand_center)
        
        direction = forearm_vec / norm
        
        # Estimar ponta do Shinai projetada
        shinai_length = norm * self.shinai_length_ratio
        kensen_3d = hand_center + direction * shinai_length
        
        return float(kensen_3d[0]), float(kensen_3d[1]), float(kensen_3d[2])

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
