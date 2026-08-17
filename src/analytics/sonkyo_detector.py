"""
Módulo de Detecção e Análise de Sonkyō para Arbitragem de Kendo.
Identifica a postura ritualística de Sonkyō (agachamento sobre a ponta dos pés com coluna ereta e joelhos flexionados)
para determinar com precisão o Início Oficial e o Término Oficial da Luta.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class SonkyoInterval:
    def __init__(self, start_frame: int, end_frame: int, fps: float, interval_type: str = "INITIAL"):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.fps = fps
        self.interval_type = interval_type  # "INITIAL" ou "FINAL"

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.end_frame - self.start_frame) / self.fps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interval_type": self.interval_type,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_seconds": round(self.duration_seconds, 2),
            "start_timestamp": self.frame_to_timestamp(self.start_frame, self.fps),
            "end_timestamp": self.frame_to_timestamp(self.end_frame, self.fps)
        }

    @staticmethod
    def frame_to_timestamp(frame: int, fps: float) -> str:
        seconds = max(0.0, frame / fps)
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"


class SonkyoDetector:
    def __init__(
        self,
        min_sonkyo_duration_frames: int = 10,
        hip_drop_ratio_threshold: float = 0.46,
        knee_angle_max_threshold: float = 110.0,
        spine_tilt_max_threshold: float = 35.0
    ):
        """
        Parâmetros de detecção da postura biomecânica de Sonkyō:
        - min_sonkyo_duration_frames: Duração mínima (em frames) para consolidar um intervalo de Sonkyō.
        - hip_drop_ratio_threshold: Razão máxima (ankle_y - hip_y) / (ankle_y - nose_y). No Sonkyō fica <= 0.45.
        - knee_angle_max_threshold: Ângulo máximo do joelho para agachamento (no Sonkyō é agudo, <= 110°).
        - spine_tilt_max_threshold: Inclinação máxima do tronco em relação à vertical (coluna ereta <= 35°).
        """
        self.min_sonkyo_duration_frames = min_sonkyo_duration_frames
        self.hip_drop_ratio_threshold = hip_drop_ratio_threshold
        self.knee_angle_max_threshold = knee_angle_max_threshold
        self.spine_tilt_max_threshold = spine_tilt_max_threshold

    @staticmethod
    def calculate_angle_2d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calcula o ângulo em graus no vértice p2 formado pelos pontos p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 180.0
        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_angle)))

    def evaluate_sonkyo_pose(self, landmarks: Optional[Dict[str, Any]]) -> Tuple[bool, float, Dict[str, float]]:
        """
        Avalia se a pose de um indivíduo em um determinado frame corresponde ao Sonkyō.
        Retorna:
            - is_sonkyo: bool
            - confidence_score: float (0.0 a 1.0)
            - metrics: dict com medições intermediárias (hip_ratio, knee_angle, spine_tilt)
        """
        if not landmarks:
            return False, 0.0, {"hip_ratio": 1.0, "knee_angle": 180.0, "spine_tilt": 0.0}

        # Extrair pontos essenciais
        has_hips = "RIGHT_HIP" in landmarks and "LEFT_HIP" in landmarks
        has_knees = "RIGHT_KNEE" in landmarks or "LEFT_KNEE" in landmarks
        has_ankles = "RIGHT_ANKLE" in landmarks or "LEFT_ANKLE" in landmarks
        has_shoulders = "RIGHT_SHOULDER" in landmarks and "LEFT_SHOULDER" in landmarks
        has_nose = "NOSE" in landmarks

        if not (has_hips and has_shoulders):
            return False, 0.0, {"hip_ratio": 1.0, "knee_angle": 180.0, "spine_tilt": 0.0}

        # 1. Posições Y
        hip_y = (landmarks["RIGHT_HIP"]["y"] + landmarks["LEFT_HIP"]["y"]) / 2.0
        shoulder_y = (landmarks["RIGHT_SHOULDER"]["y"] + landmarks["LEFT_SHOULDER"]["y"]) / 2.0
        shoulder_x = (landmarks["RIGHT_SHOULDER"]["x"] + landmarks["LEFT_SHOULDER"]["x"]) / 2.0
        hip_x = (landmarks["RIGHT_HIP"]["x"] + landmarks["LEFT_HIP"]["x"]) / 2.0

        # Ankle / Foot
        if "RIGHT_ANKLE" in landmarks and "LEFT_ANKLE" in landmarks:
            ankle_y = (landmarks["RIGHT_ANKLE"]["y"] + landmarks["LEFT_ANKLE"]["y"]) / 2.0
        elif "RIGHT_ANKLE" in landmarks:
            ankle_y = landmarks["RIGHT_ANKLE"]["y"]
        elif "LEFT_ANKLE" in landmarks:
            ankle_y = landmarks["LEFT_ANKLE"]["y"]
        elif "RIGHT_FOOT_INDEX" in landmarks:
            ankle_y = landmarks["RIGHT_FOOT_INDEX"]["y"]
        else:
            ankle_y = hip_y + 0.35

        nose_y = landmarks["NOSE"]["y"] if has_nose else (shoulder_y - 0.15)

        # 2. Razão de rebaixamento do quadril (Hip Drop Ratio)
        total_height = max(0.1, ankle_y - nose_y)
        hip_to_ankle = max(0.0, ankle_y - hip_y)
        hip_ratio = hip_to_ankle / total_height

        # 3. Ângulo do joelho (Knee Flexion Angle)
        knee_angle = 180.0
        if "RIGHT_HIP" in landmarks and "RIGHT_KNEE" in landmarks and "RIGHT_ANKLE" in landmarks:
            p_hip = np.array([landmarks["RIGHT_HIP"]["x"], landmarks["RIGHT_HIP"]["y"]])
            p_knee = np.array([landmarks["RIGHT_KNEE"]["x"], landmarks["RIGHT_KNEE"]["y"]])
            p_ankle = np.array([landmarks["RIGHT_ANKLE"]["x"], landmarks["RIGHT_ANKLE"]["y"]])
            knee_angle = self.calculate_angle_2d(p_hip, p_knee, p_ankle)
        elif "LEFT_HIP" in landmarks and "LEFT_KNEE" in landmarks and "LEFT_ANKLE" in landmarks:
            p_hip = np.array([landmarks["LEFT_HIP"]["x"], landmarks["LEFT_HIP"]["y"]])
            p_knee = np.array([landmarks["LEFT_KNEE"]["x"], landmarks["LEFT_KNEE"]["y"]])
            p_ankle = np.array([landmarks["LEFT_ANKLE"]["x"], landmarks["LEFT_ANKLE"]["y"]])
            knee_angle = self.calculate_angle_2d(p_hip, p_knee, p_ankle)
        else:
            # Estimativa sintética baseada no hip_ratio caso joelho esteja ocluso pelo hakama
            if hip_ratio < 0.40:
                knee_angle = 60.0 + (hip_ratio / 0.40) * 40.0
            else:
                knee_angle = 150.0

        # 4. Inclinação da coluna (Spine Tilt)
        spine_vec = np.array([shoulder_x - hip_x, shoulder_y - hip_y])
        vert_vec = np.array([0.0, -1.0])
        norm_spine = np.linalg.norm(spine_vec)
        if norm_spine > 0:
            cos_tilt = np.dot(spine_vec, vert_vec) / norm_spine
            spine_tilt = float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))
        else:
            spine_tilt = 0.0

        # 5. Cálculo do Score de Confiança do Sonkyō
        # Hip score: 1.0 se hip_ratio <= 0.35, decai até 0.48
        score_hip = float(np.clip((self.hip_drop_ratio_threshold - hip_ratio) / 0.20 + 0.5, 0.0, 1.0))
        # Knee score: 1.0 se knee_angle <= 75°, decai até knee_angle_max
        score_knee = float(np.clip((self.knee_angle_max_threshold - knee_angle) / 50.0 + 0.3, 0.0, 1.0))
        # Spine score: coluna ereta
        score_spine = float(np.clip(1.0 - (spine_tilt / self.spine_tilt_max_threshold), 0.0, 1.0))

        confidence = (0.50 * score_hip) + (0.35 * score_knee) + (0.15 * score_spine)
        is_sonkyo = (hip_ratio <= self.hip_drop_ratio_threshold) and (knee_angle <= self.knee_angle_max_threshold) and (confidence >= 0.55)

        metrics = {
            "hip_ratio": round(float(hip_ratio), 3),
            "knee_angle": round(float(knee_angle), 1),
            "spine_tilt": round(float(spine_tilt), 1),
            "confidence": round(float(confidence), 3)
        }

        return is_sonkyo, float(confidence), metrics

    def detect_match_boundaries(
        self,
        pose_history: List[Optional[Dict[str, Any]]],
        fps: float = 30.0,
        secondary_pose_history: Optional[List[Optional[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Escaneia o histórico de poses dos combatentes e determina os limites exatos da luta:
        - match_start_frame: Momento em que os atletas se levantam do Sonkyō inicial (ou início do vídeo).
        - match_end_frame: Momento em que os atletas agacham no Sonkyō final (ou fim do vídeo).
        """
        total_frames = len(pose_history)
        if total_frames < self.min_sonkyo_duration_frames:
            return {
                "is_bounded": False,
                "has_initial_sonkyo": False,
                "has_final_sonkyo": False,
                "match_start_frame": 0,
                "match_end_frame": max(0, total_frames - 1),
                "match_start_timestamp": "00:00.000",
                "match_end_timestamp": SonkyoInterval.frame_to_timestamp(max(0, total_frames - 1), fps),
                "effective_combat_duration_seconds": round(total_frames / fps, 2),
                "initial_sonkyo": None,
                "final_sonkyo": None,
                "sonkyo_intervals": [],
                "status_message": "Vídeo muito curto para análise de Sonkyō."
            }

        # 1. Identificar frames em Sonkyō
        sonkyo_flags = []
        for idx in range(total_frames):
            p1 = pose_history[idx]
            is_s1, _, _ = self.evaluate_sonkyo_pose(p1)
            
            if secondary_pose_history and idx < len(secondary_pose_history):
                p2 = secondary_pose_history[idx]
                is_s2, _, _ = self.evaluate_sonkyo_pose(p2)
                # Se ambos ou o principal estiverem em Sonkyō
                is_frame_sonkyo = is_s1 or is_s2
            else:
                is_frame_sonkyo = is_s1

            sonkyo_flags.append(is_frame_sonkyo)

        # 2. Agrupar em blocos contínuos de Sonkyō
        intervals: List[SonkyoInterval] = []
        in_block = False
        block_start = 0

        for f_idx, is_s in enumerate(sonkyo_flags):
            if is_s and not in_block:
                in_block = True
                block_start = f_idx
            elif not is_s and in_block:
                in_block = False
                block_len = f_idx - block_start
                if block_len >= self.min_sonkyo_duration_frames:
                    intervals.append(SonkyoInterval(block_start, f_idx, fps))

        if in_block:
            block_len = total_frames - block_start
            if block_len >= self.min_sonkyo_duration_frames:
                intervals.append(SonkyoInterval(block_start, total_frames - 1, fps))

        # 3. Determinar Sonkyō Inicial e Sonkyō Final
        initial_sonkyo = None
        final_sonkyo = None
        has_initial = False
        has_final = False

        # Sonkyō Inicial: Procurar nos primeiros 40% do vídeo
        first_quarter_frames = int(total_frames * 0.40)
        early_intervals = [it for it in intervals if it.start_frame <= first_quarter_frames]

        if early_intervals:
            initial_sonkyo = early_intervals[0]
            initial_sonkyo.interval_type = "INITIAL"
            has_initial = True
            # Início da luta ativa: momento em que os lutadores se levantam do Sonkyō inicial
            match_start_frame = min(total_frames - 1, initial_sonkyo.end_frame + 2)
        else:
            match_start_frame = 0

        # Sonkyō Final: Procurar nos últimos 40% do vídeo
        last_quarter_frames = int(total_frames * 0.60)
        late_intervals = [it for it in intervals if it.end_frame >= last_quarter_frames and (initial_sonkyo is None or it.start_frame > initial_sonkyo.end_frame + 30)]

        if late_intervals:
            final_sonkyo = late_intervals[-1]
            final_sonkyo.interval_type = "FINAL"
            has_final = True
            # Fim da luta ativa: momento em que os lutadores agacham no Sonkyō de encerramento
            match_end_frame = max(match_start_frame, final_sonkyo.start_frame - 2)
        else:
            match_end_frame = max(0, total_frames - 1)

        # Garantir consistência
        if match_end_frame <= match_start_frame:
            match_start_frame = 0
            match_end_frame = max(0, total_frames - 1)
            is_bounded = False
        else:
            is_bounded = has_initial or has_final

        effective_duration = max(0.0, (match_end_frame - match_start_frame) / fps)

        status_msg = "Combate delimitado por Sonkyō (Início e Fim detectados)."
        if has_initial and not has_final:
            status_msg = "Sonkyō de Início detectado. Fim considerado até o término da gravação."
        elif not has_initial and has_final:
            status_msg = "Sonkyō de Fechamento detectado. Início considerado desde o primeiro quadro."
        elif not is_bounded:
            status_msg = "Nenhum ritual de Sonkyō detectado. Todo o vídeo foi considerado para análise."

        return {
            "is_bounded": is_bounded,
            "has_initial_sonkyo": has_initial,
            "has_final_sonkyo": has_final,
            "match_start_frame": match_start_frame,
            "match_end_frame": match_end_frame,
            "match_start_timestamp": SonkyoInterval.frame_to_timestamp(match_start_frame, fps),
            "match_end_timestamp": SonkyoInterval.frame_to_timestamp(match_end_frame, fps),
            "effective_combat_duration_seconds": round(effective_duration, 2),
            "initial_sonkyo": initial_sonkyo.to_dict() if initial_sonkyo else None,
            "final_sonkyo": final_sonkyo.to_dict() if final_sonkyo else None,
            "sonkyo_intervals": [it.to_dict() for it in intervals],
            "status_message": status_msg
        }
