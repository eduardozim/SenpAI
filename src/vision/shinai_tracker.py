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
        r_wrist = np.array([landmarks["RIGHT_WRIST"]["x"], landmarks["RIGHT_WRIST"]["y"], landmarks["RIGHT_WRIST"]["z"]])
        r_elbow = np.array([landmarks["RIGHT_ELBOW"]["x"], landmarks["RIGHT_ELBOW"]["y"], landmarks["RIGHT_ELBOW"]["z"]])
        
        l_wrist = np.array([landmarks["LEFT_WRIST"]["x"], landmarks["LEFT_WRIST"]["y"], landmarks["LEFT_WRIST"]["z"]])

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
        nose = np.array([landmarks["NOSE"]["x"], landmarks["NOSE"]["y"], landmarks["NOSE"]["z"]])
        r_ear = np.array([landmarks["RIGHT_EAR"]["x"], landmarks["RIGHT_EAR"]["y"], landmarks["RIGHT_EAR"]["z"]])
        l_ear = np.array([landmarks["LEFT_EAR"]["x"], landmarks["LEFT_EAR"]["y"], landmarks["LEFT_EAR"]["z"]])
        head_center = (nose + r_ear + l_ear) / 3.0
        targets["MEN"] = (float(head_center[0]), float(head_center[1] - 0.05), float(head_center[2]))

        # 2. Kote (Pulso Direito)
        r_wrist = landmarks["RIGHT_WRIST"]
        targets["KOTE"] = (r_wrist["x"], r_wrist["y"], r_wrist["z"])

        # 3. Do (Lateral do Tronco - Hip/Ribs)
        r_hip = np.array([landmarks["RIGHT_HIP"]["x"], landmarks["RIGHT_HIP"]["y"], landmarks["RIGHT_HIP"]["z"]])
        r_shoulder = np.array([landmarks["RIGHT_SHOULDER"]["x"], landmarks["RIGHT_SHOULDER"]["y"], landmarks["RIGHT_SHOULDER"]["z"]])
        do_point = (r_hip * 0.6) + (r_shoulder * 0.4)
        targets["DO"] = (float(do_point[0]), float(do_point[1]), float(do_point[2]))

        # 4. Tsuki (Garganta)
        l_shoulder = np.array([landmarks["LEFT_SHOULDER"]["x"], landmarks["LEFT_SHOULDER"]["y"], landmarks["LEFT_SHOULDER"]["z"]])
        shoulder_center = (r_shoulder + l_shoulder) / 2.0
        throat_point = (shoulder_center * 0.7) + (nose * 0.3)
        targets["TSUKI"] = (float(throat_point[0]), float(throat_point[1]), float(throat_point[2]))

        return targets
