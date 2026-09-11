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

        r_wrist_pt = landmarks.get("RIGHT_WRIST") or landmarks.get("LEFT_WRIST")
        if not r_wrist_pt:
            return 0.5
        r_wrist = np.array([r_wrist_pt["x"], r_wrist_pt["y"]])
        
        st_clean = str(strike_type).replace("メ ", "").replace("コ ", "").replace("ド ", "").replace("ツ ", "").replace("メ", "").replace("コ", "").replace("ド", "").replace("ツ", "").strip().upper()

        if st_clean == "MEN":
            # Alvo Men: Acima da linha dos olhos/nariz
            nose_pt = landmarks.get("NOSE") or landmarks.get("RIGHT_EAR") or landmarks.get("LEFT_EAR")
            nose_y = nose_pt["y"] if nose_pt else 0.25
            diff = abs(r_wrist[1] - nose_y)
            score = max(0.0, 1.0 - (diff * 2.5))
        elif st_clean == "KOTE":
            # Alvo Kote: Linha da cintura/ombro com boa extensão de cotovelo
            r_elbow_pt = landmarks.get("RIGHT_ELBOW") or landmarks.get("LEFT_ELBOW")
            r_shoulder_pt = landmarks.get("RIGHT_SHOULDER") or landmarks.get("LEFT_SHOULDER")
            if r_elbow_pt and r_shoulder_pt:
                r_elbow = np.array([r_elbow_pt["x"], r_elbow_pt["y"], r_elbow_pt.get("z", 0.0)])
                r_shoulder = np.array([r_shoulder_pt["x"], r_shoulder_pt["y"], r_shoulder_pt.get("z", 0.0)])
                p_wrist = np.array([r_wrist_pt["x"], r_wrist_pt["y"], r_wrist_pt.get("z", 0.0)])
                elbow_angle = self.calculate_angle_3d(r_shoulder, r_elbow, p_wrist)
                score = 1.0 - (abs(155.0 - elbow_angle) / 60.0)
            else:
                score = 0.70
        elif st_clean == "DO":
            # Alvo Do: Mãos na altura do peito, trajetória lateral
            hip_pt = landmarks.get("RIGHT_HIP") or landmarks.get("LEFT_HIP")
            hip_y = hip_pt["y"] if hip_pt else 0.60
            diff = abs(r_wrist[1] - hip_y)
            score = max(0.0, 1.0 - (diff * 2.0))
        else: # TSUKI
            shoulder_pt = landmarks.get("RIGHT_SHOULDER") or landmarks.get("LEFT_SHOULDER")
            shoulder_y = shoulder_pt["y"] if shoulder_pt else 0.40
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
            r_foot_pt = lm.get("RIGHT_FOOT_INDEX") or lm.get("RIGHT_ANKLE") or lm.get("LEFT_FOOT_INDEX") or lm.get("LEFT_ANKLE")
            if not r_foot_pt or "x" not in r_foot_pt or "y" not in r_foot_pt:
                foot_velocities.append(0.0)
                continue

            r_foot = np.array([r_foot_pt["x"], r_foot_pt["y"]])
            if f == impact_frame - 5 or not pose_history[f - 1]:
                foot_velocities.append(0.0)
            else:
                prev_lm = pose_history[f - 1]
                if not prev_lm:
                    foot_velocities.append(0.0)
                else:
                    prev_foot_pt = prev_lm.get("RIGHT_FOOT_INDEX") or prev_lm.get("RIGHT_ANKLE") or prev_lm.get("LEFT_FOOT_INDEX") or prev_lm.get("LEFT_ANKLE")
                    if not prev_foot_pt or "x" not in prev_foot_pt or "y" not in prev_foot_pt:
                        foot_velocities.append(0.0)
                    else:
                        prev_foot = np.array([prev_foot_pt["x"], prev_foot_pt["y"]])
                        vel = float(np.linalg.norm(r_foot - prev_foot))
                        foot_velocities.append(vel)

        if not foot_velocities or max(foot_velocities) < 0.005:
            # Sem rastreamento de pés ou pés em repouso: atribui escore regulamentar neutro
            return 0.75, 0.0

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

        r_shoulder_pt = landmarks.get("RIGHT_SHOULDER") or landmarks.get("LEFT_SHOULDER")
        r_hip_pt = landmarks.get("RIGHT_HIP") or landmarks.get("LEFT_HIP")
        if not r_shoulder_pt or not r_hip_pt:
            return 0.75

        r_shoulder = np.array([r_shoulder_pt["x"], r_shoulder_pt["y"]])
        r_hip = np.array([r_hip_pt["x"], r_hip_pt["y"]])
        
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
