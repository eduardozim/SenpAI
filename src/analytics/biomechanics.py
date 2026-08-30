"""
Módulo Biomecânico de Avaliação do Kendo.
Calcula métricas numéricas precisas para:
1. Impacto no Alvo (Target Contact)
2. Sincronismo Pé-Mão (Fumikomi / Ki-Ken-Tai-Ichi)
3. Postura Corporal (Spine Alignment)
4. Manutenção de Guarda Pós-Golpe (Zanshin)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Sequence

class BiomechanicsAnalyzer:
    @staticmethod
    def calculate_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calcula o ângulo em graus no ponto p2 formado pelos vetores p1-p2 e p3-p2."""
        v1 = p1 - p2
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 180.0
        
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_angle)))

    def evaluate_target_impact(self, strike_type: str, landmarks: Optional[Dict[str, Any]]) -> float:
        """
        Avalia o grau de precisão do impacto no alvo correto. Retorna um score entre 0.0 e 1.0.
        """
        if not landmarks:
            return 0.0

        r_wrist = np.array([landmarks["RIGHT_WRIST"]["x"], landmarks["RIGHT_WRIST"]["y"]])
        
        st_clean = str(strike_type).replace("メ ", "").replace("コ ", "").replace("ド ", "").replace("ツ ", "").replace("メ", "").replace("コ", "").replace("ド", "").replace("ツ", "").strip().upper()

        if st_clean == "MEN":
            # Alvo Men: Acima da linha dos olhos/nariz
            nose_y = landmarks["NOSE"]["y"]
            diff = abs(r_wrist[1] - nose_y)
            score = max(0.0, 1.0 - (diff * 2.5))
        elif st_clean == "KOTE":
            # Alvo Kote: Linha da cintura/ombro com boa extensão de cotovelo
            r_elbow = np.array([landmarks["RIGHT_ELBOW"]["x"], landmarks["RIGHT_ELBOW"]["y"], landmarks["RIGHT_ELBOW"]["z"]])
            r_shoulder = np.array([landmarks["RIGHT_SHOULDER"]["x"], landmarks["RIGHT_SHOULDER"]["y"], landmarks["RIGHT_SHOULDER"]["z"]])
            p_wrist = np.array([landmarks["RIGHT_WRIST"]["x"], landmarks["RIGHT_WRIST"]["y"], landmarks["RIGHT_WRIST"]["z"]])
            elbow_angle = self.calculate_angle_3d(r_shoulder, r_elbow, p_wrist)
            # No Kote a extensão do cotovelo deve ser forte (140° a 170°)
            score = 1.0 - (abs(155.0 - elbow_angle) / 60.0)
        elif st_clean == "DO":
            # Alvo Do: Mãos na altura do peito, trajetória lateral
            hip_y = landmarks["RIGHT_HIP"]["y"]
            diff = abs(r_wrist[1] - hip_y)
            score = max(0.0, 1.0 - (diff * 2.0))
        else: # TSUKI
            shoulder_y = landmarks["RIGHT_SHOULDER"]["y"]
            diff = abs(r_wrist[1] - shoulder_y)
            score = max(0.0, 1.0 - (diff * 3.0))

        return float(np.clip(score, 0.0, 1.0))

    def evaluate_fumikomi_sync(self, pose_history: Sequence[Optional[Dict[str, Any]]], impact_frame: int) -> Tuple[float, float]:
        """
        Avalia o Ki-Ken-Tai-Ichi: Sincronismo do impacto do pé direito (Fumikomi) com a batida das mãos.
        Retorna (score, offset_ms).
        """
        if impact_frame >= len(pose_history) or impact_frame < 5:
            return 0.5, 0.0

        # Rastrear movimento do pé direito (RIGHT_ANKLE / RIGHT_FOOT_INDEX)
        foot_velocities = []
        for f in range(impact_frame - 5, min(impact_frame + 6, len(pose_history))):
            lm = pose_history[f]
            if not lm:
                foot_velocities.append(0.0)
                continue
            r_foot = np.array([lm["RIGHT_FOOT_INDEX"]["x"], lm["RIGHT_FOOT_INDEX"]["y"]])
            if f == impact_frame - 5 or not pose_history[f - 1]:
                foot_velocities.append(0.0)
            else:
                prev_lm = pose_history[f - 1]
                if not prev_lm:
                    foot_velocities.append(0.0)
                else:
                    prev_foot = np.array([prev_lm["RIGHT_FOOT_INDEX"]["x"], prev_lm["RIGHT_FOOT_INDEX"]["y"]])
                    foot_velocities.append(float(np.linalg.norm(r_foot - prev_foot)))

        # Encontrar instante do impacto do pé
        peak_foot_offset = int(np.argmax(foot_velocities) - 5)
        offset_ms = float(peak_foot_offset * 33.3) # ~33ms por frame a 30fps

        # Sincronia perfeita é quando offset é próximo de 0 (pé e mão batem juntos)
        sync_score = max(0.0, 1.0 - (abs(offset_ms) / 150.0))
        return float(sync_score), float(offset_ms)

    def evaluate_posture(self, landmarks: Optional[Dict[str, Any]]) -> float:
        """
        Avalia a postura corporal (verticalidade da coluna, ombros nivelados).
        No Kendo, o tronco não deve inclinar demasiadamente para a frente nem colapsar.
        """
        if not landmarks:
            return 0.0

        r_shoulder = np.array([landmarks["RIGHT_SHOULDER"]["x"], landmarks["RIGHT_SHOULDER"]["y"]])
        r_hip = np.array([landmarks["RIGHT_HIP"]["x"], landmarks["RIGHT_HIP"]["y"]])
        
        # Vetor da coluna (quadril ao ombro)
        spine_vec = r_shoulder - r_hip # y cresce para baixo na imagem
        # Vetor vertical puro (0, -1)
        vertical_vec = np.array([0.0, -1.0])
        
        norm_spine = float(np.linalg.norm(spine_vec))
        if norm_spine == 0:
            return 0.0
            
        cosine_tilt = np.dot(spine_vec, vertical_vec) / norm_spine
        tilt_degrees = np.degrees(np.arccos(np.clip(cosine_tilt, -1.0, 1.0)))
        
        # Uma inclinação aceitável no Kendo é de 0° a 15°. Acima de 25° a postura é ruim.
        score = 1.0 - max(0.0, (tilt_degrees - 10.0) / 25.0)
        return float(np.clip(score, 0.0, 1.0))

    def evaluate_zanshin(self, pose_history: Sequence[Optional[Dict[str, Any]]], impact_frame: int, end_frame: int) -> float:
        """
        Avalia a manutenção de Zanshin (guarda e estabilidade corporal após o golpe).
        """
        if end_frame >= len(pose_history) or impact_frame >= end_frame:
            return 0.5

        # Analisar estabilidade da postura nos frames posteriores ao impacto
        posture_scores = []
        for f in range(impact_frame + 1, min(end_frame, len(pose_history))):
            curr_f = pose_history[f]
            if curr_f:
                posture_scores.append(self.evaluate_posture(curr_f))

        if not posture_scores:
            return 0.5

        zanshin_score = float(np.mean(posture_scores))
        return float(np.clip(zanshin_score, 0.0, 1.0))
