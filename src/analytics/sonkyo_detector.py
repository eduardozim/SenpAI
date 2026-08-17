"""
Módulo de Detecção e Análise de Sonkyō para Arbitragem de Kendo.
Identifica a postura ritualística de Sonkyō (agachamento sobre a ponta dos pés com coluna ereta e joelhos flexionados)
para determinar com precisão o Início Oficial e o Término Oficial da Luta.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class SonkyoInterval:
    def __init__(self, start_frame: int, end_frame: int, fps: float, interval_type: str = "INITIAL", confidence: float = 0.85):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.fps = fps
        self.interval_type = interval_type  # "INITIAL", "FINAL" ou "INTERMEDIATE"
        self.confidence = confidence

    @property
    def duration_seconds(self) -> float:
        return max(0.0, (self.end_frame - self.start_frame) / self.fps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interval_type": self.interval_type,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_seconds": round(self.duration_seconds, 2),
            "confidence": round(self.confidence, 3),
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
        min_sonkyo_duration_frames: int = 6,
        hip_drop_ratio_threshold: float = 0.48,
        knee_angle_max_threshold: float = 120.0,
        spine_tilt_max_threshold: float = 40.0
    ):
        """
        Parâmetros de detecção da postura biomecânica de Sonkyō:
        - min_sonkyo_duration_frames: Duração mínima (em frames) para consolidar um intervalo de Sonkyō (default 6 quadros ~ 0.2s).
        - hip_drop_ratio_threshold: Razão máxima (ankle_y - hip_y) / (ankle_y - nose_y). No Sonkyō fica <= 0.48.
        - knee_angle_max_threshold: Ângulo máximo do joelho para agachamento (no Sonkyō é agudo/flexionado <= 120°).
        - spine_tilt_max_threshold: Inclinação máxima do tronco em relação à vertical (coluna ereta <= 40°).
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
        Utiliza múltiplos sinais biomecânicos robustos a oclusões por Hakama / Kendogi.
        Retorna:
            - is_sonkyo: bool
            - confidence_score: float (0.0 a 1.0)
            - metrics: dict com medições intermediárias
        """
        if not landmarks:
            return False, 0.0, {"hip_ratio": 1.0, "torso_ratio": 0.35, "knee_angle": 180.0, "spine_tilt": 0.0}

        # Extrair pontos essenciais
        has_hips = "RIGHT_HIP" in landmarks or "LEFT_HIP" in landmarks
        has_shoulders = "RIGHT_SHOULDER" in landmarks or "LEFT_SHOULDER" in landmarks

        if not (has_hips and has_shoulders):
            return False, 0.0, {"hip_ratio": 1.0, "torso_ratio": 0.35, "knee_angle": 180.0, "spine_tilt": 0.0}

        # 1. Posições Y e Bounding Box
        hip_ys = [landmarks[k]["y"] for k in ["RIGHT_HIP", "LEFT_HIP"] if k in landmarks]
        hip_y = float(np.mean(hip_ys))
        hip_xs = [landmarks[k]["x"] for k in ["RIGHT_HIP", "LEFT_HIP"] if k in landmarks]
        hip_x = float(np.mean(hip_xs))

        shoulder_ys = [landmarks[k]["y"] for k in ["RIGHT_SHOULDER", "LEFT_SHOULDER"] if k in landmarks]
        shoulder_y = float(np.mean(shoulder_ys))
        shoulder_xs = [landmarks[k]["x"] for k in ["RIGHT_SHOULDER", "LEFT_SHOULDER"] if k in landmarks]
        shoulder_x = float(np.mean(shoulder_xs))

        # Pés / Tornozelos / Limite inferior
        ankle_ys = [landmarks[k]["y"] for k in ["RIGHT_ANKLE", "LEFT_ANKLE", "RIGHT_FOOT_INDEX", "LEFT_FOOT_INDEX", "RIGHT_HEEL", "LEFT_HEEL"] if k in landmarks]
        if ankle_ys:
            ankle_y = float(max(ankle_ys))
        else:
            all_ys = [pt["y"] for pt in landmarks.values() if isinstance(pt, dict) and "y" in pt]
            ankle_y = float(max(all_ys)) if all_ys else (hip_y + 0.30)

        # Cabeça / Nariz / Limite superior
        if "NOSE" in landmarks:
            top_y = landmarks["NOSE"]["y"]
        else:
            top_y = max(0.0, shoulder_y - 0.14)

        # 2. Métricas de Altura e Rebaixamento
        total_height = max(0.08, ankle_y - top_y)
        hip_to_ankle = max(0.0, ankle_y - hip_y)
        hip_ratio = hip_to_ankle / total_height

        torso_height = max(0.04, hip_y - shoulder_y)
        torso_ratio = torso_height / max(0.08, ankle_y - shoulder_y)

        # 3. Ângulo do joelho (Knee Flexion Angle com fallback inteligente)
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
            # Estimativa cinemática baseada no hip_ratio caso o hakama oclua as pernas
            if hip_ratio < 0.42:
                knee_angle = 60.0 + (hip_ratio / 0.42) * 45.0
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
        # Hip score: 1.0 se hip_ratio <= 0.20, zero se hip_ratio >= 0.38 (em pé é ~0.45-0.55)
        score_hip = float(np.clip((0.36 - hip_ratio) / 0.16, 0.0, 1.0))
        # Torso proportion score: no Sonkyō o torso ocupa >= 60% do espaço vertical (em pé é ~0.35-0.45)
        score_torso = float(np.clip((torso_ratio - 0.50) / 0.20, 0.0, 1.0))
        # Knee score: flexão de joelho aguda (em pé é 170-180°)
        score_knee = float(np.clip((130.0 - knee_angle) / 50.0, 0.0, 1.0))
        # Spine score: coluna ereta
        score_spine = float(np.clip(1.0 - (spine_tilt / self.spine_tilt_max_threshold), 0.0, 1.0))

        confidence = (0.35 * score_hip) + (0.35 * score_torso) + (0.20 * score_knee) + (0.10 * score_spine)
        is_sonkyo = (confidence >= 0.48) and (score_hip > 0.05 or score_torso > 0.05) and (knee_angle <= 140.0)

        metrics = {
            "hip_ratio": round(float(hip_ratio), 3),
            "torso_ratio": round(float(torso_ratio), 3),
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
        Escaneia o histórico de poses dos combatentes com análise multi-sinal adaptativa e temporal
        para determinar os limites exatos da luta:
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

        # 1. Obter curvas de probabilidade por frame para cada combatente
        def compute_combatant_sonkyo_probs(history: List[Optional[Dict[str, Any]]]) -> np.ndarray:
            probs = np.zeros(total_frames, dtype=np.float32)
            valid_heights = []
            valid_hip_ys = []

            # Extração de baseline de altura e quadril
            for p in history:
                if p:
                    all_ys = [pt["y"] for pt in p.values() if isinstance(pt, dict) and "y" in pt]
                    if all_ys:
                        h = max(all_ys) - min(all_ys)
                        valid_heights.append(h)
                    if "RIGHT_HIP" in p or "LEFT_HIP" in p:
                        hy = [p[k]["y"] for k in ["RIGHT_HIP", "LEFT_HIP"] if k in p]
                        valid_hip_ys.append(float(np.mean(hy)))

            standing_height = float(np.percentile(valid_heights, 85)) if len(valid_heights) >= 5 else 0.60
            standing_hip_y = float(np.percentile(valid_hip_ys, 15)) if len(valid_hip_ys) >= 5 else 0.58

            for f_i in range(total_frames):
                p = history[f_i] if f_i < len(history) else None
                if not p:
                    continue

                is_s, conf_pose, _ = self.evaluate_sonkyo_pose(p)

                # Avaliação da compressão de altura relativa ao atleta em pé
                all_ys = [pt["y"] for pt in p.values() if isinstance(pt, dict) and "y" in pt]
                if all_ys and standing_height > 0.05:
                    h_curr = max(all_ys) - min(all_ys)
                    rel_height_ratio = h_curr / standing_height
                    rel_height_score = float(np.clip((0.82 - rel_height_ratio) / 0.22, 0.0, 1.0))
                else:
                    rel_height_score = 0.0

                # Avaliação de rebaixamento do quadril em relação ao baseline em pé
                hip_ys = [p[k]["y"] for k in ["RIGHT_HIP", "LEFT_HIP"] if k in p]
                if hip_ys and standing_hip_y > 0.05:
                    h_y_curr = float(np.mean(hip_ys))
                    hip_drop_delta = h_y_curr - standing_hip_y
                    hip_drop_score = float(np.clip(hip_drop_delta / 0.12, 0.0, 1.0))
                else:
                    hip_drop_score = 0.0

                # Fusão dos sinais biométricos e relativos
                rel_score = 0.50 * rel_height_score + 0.50 * hip_drop_score
                combined_score = max(conf_pose, rel_score)
                if is_s:
                    combined_score = max(combined_score, 0.75)

                probs[f_i] = combined_score

            return probs

        p_prob1 = compute_combatant_sonkyo_probs(pose_history)
        if secondary_pose_history:
            p_prob2 = compute_combatant_sonkyo_probs(secondary_pose_history)
            combined_probs = np.maximum(p_prob1, p_prob2)
            simultaneous_mask = (p_prob1 >= 0.40) & (p_prob2 >= 0.40)
            combined_probs[simultaneous_mask] = np.clip(combined_probs[simultaneous_mask] + 0.15, 0.0, 1.0)
        else:
            combined_probs = p_prob1

        # 2. Suavização Temporal e Fechamento Morfológico (Preenchimento de falhas de até 8 frames)
        smoothed = np.copy(combined_probs)
        if len(smoothed) >= 3:
            smoothed = np.convolve(smoothed, np.ones(3)/3.0, mode='same')

        # Ponte para pequenas falhas momentâneas (dropout gap bridging)
        binary_mask = (smoothed >= 0.48)
        gap_limit = 8
        in_gap = False
        gap_start = 0

        for i in range(len(binary_mask)):
            if not binary_mask[i] and not in_gap:
                in_gap = True
                gap_start = i
            elif binary_mask[i] and in_gap:
                in_gap = False
                gap_len = i - gap_start
                if gap_len <= gap_limit and gap_start > 0:
                    binary_mask[gap_start:i] = True

        # 3. Agrupamento em Intervalos Consistentes de Sonkyō
        raw_intervals: List[Tuple[int, int, float]] = []
        in_block = False
        block_start = 0

        for f_idx in range(total_frames):
            if binary_mask[f_idx] and not in_block:
                in_block = True
                block_start = f_idx
            elif not binary_mask[f_idx] and in_block:
                in_block = False
                block_len = f_idx - block_start
                if block_len >= self.min_sonkyo_duration_frames:
                    mean_conf = float(np.mean(smoothed[block_start:f_idx]))
                    raw_intervals.append((block_start, f_idx, mean_conf))

        if in_block:
            block_len = total_frames - block_start
            if block_len >= self.min_sonkyo_duration_frames:
                mean_conf = float(np.mean(smoothed[block_start:total_frames]))
                raw_intervals.append((block_start, total_frames - 1, mean_conf))

        # Mesclar intervalos adjacentes muito próximos (gap <= 15 frames)
        merged_intervals: List[SonkyoInterval] = []
        for s_f, e_f, c in raw_intervals:
            if not merged_intervals:
                merged_intervals.append(SonkyoInterval(s_f, e_f, fps, confidence=c))
            else:
                last_it = merged_intervals[-1]
                if s_f - last_it.end_frame <= 15:
                    # Mesclar
                    last_it.end_frame = e_f
                    last_it.confidence = (last_it.confidence + c) / 2.0
                else:
                    merged_intervals.append(SonkyoInterval(s_f, e_f, fps, confidence=c))

        # 4. Determinação Precisa de Sonkyō Inicial e Final
        initial_sonkyo = None
        final_sonkyo = None
        has_initial = False
        has_final = False

        half_frames = int(total_frames * 0.50)

        # Sonkyō Inicial: Procurar nos primeiros 50% do vídeo
        early_candidates = [it for it in merged_intervals if it.start_frame <= half_frames]
        if early_candidates:
            # Selecionar o candidato mais próximo do início
            initial_sonkyo = early_candidates[0]
            initial_sonkyo.interval_type = "INITIAL"
            has_initial = True
            match_start_frame = min(total_frames - 1, initial_sonkyo.end_frame + 2)
        else:
            match_start_frame = 0

        # Sonkyō Final: Procurar na segunda metade do vídeo (após o início da luta)
        min_final_start = (initial_sonkyo.end_frame + 15) if initial_sonkyo else int(total_frames * 0.40)
        late_candidates = [it for it in merged_intervals if it.end_frame >= half_frames and it.start_frame >= min_final_start]
        
        if late_candidates:
            # Selecionar o candidato mais próximo do término
            final_sonkyo = late_candidates[-1]
            final_sonkyo.interval_type = "FINAL"
            has_final = True
            match_end_frame = max(match_start_frame, final_sonkyo.start_frame - 2)
        else:
            match_end_frame = max(0, total_frames - 1)

        # Garantir consistência lógica
        if match_end_frame <= match_start_frame:
            match_start_frame = 0
            match_end_frame = max(0, total_frames - 1)
            is_bounded = False
        else:
            is_bounded = has_initial or has_final

        effective_duration = max(0.0, (match_end_frame - match_start_frame) / fps)

        status_msg = "Combate delimitado por Sonkyō (Início e Fim detectados com sucesso)."
        if has_initial and not has_final:
            status_msg = "Sonkyō de Início detectado com sucesso. Fim considerado até o término da gravação."
        elif not has_initial and has_final:
            status_msg = "Sonkyō de Fechamento detectado com sucesso. Início considerado desde o primeiro quadro."
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
            "sonkyo_intervals": [it.to_dict() for it in merged_intervals],
            "status_message": status_msg
        }
