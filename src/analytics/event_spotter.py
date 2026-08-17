"""
Action Spotting & Event Detection Module para Kendo.
Analisa séries temporais de aceleração e trajetória dos pulsos/espada para identificar o início,
impacto e fim de um golpe (Men, Kote, Do, Tsuki).
"""

import numpy as np
from typing import List, Dict, Any, Optional

class StrikeEvent:
    def __init__(
        self,
        strike_type: str,
        start_frame: int,
        impact_frame: int,
        end_frame: int,
        fps: float,
        attacker_id: str = "KENSHI_AKA",
        attacker_name: str = "Kenshi Aka",
        is_within_sonkyo_bounds: bool = True
    ):
        self.type = strike_type  # "MEN", "KOTE", "DO", "TSUKI"
        self.start_frame = start_frame
        self.impact_frame = impact_frame
        self.end_frame = end_frame
        self.fps = fps
        self.attacker_id = attacker_id
        self.attacker_name = attacker_name
        self.is_within_sonkyo_bounds = is_within_sonkyo_bounds

    @property
    def timestamp_impact(self) -> str:
        seconds = self.impact_frame / self.fps
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "start_frame": self.start_frame,
            "impact_frame": self.impact_frame,
            "end_frame": self.end_frame,
            "timestamp": self.timestamp_impact,
            "attacker_id": self.attacker_id,
            "attacker_name": self.attacker_name,
            "is_within_sonkyo_bounds": self.is_within_sonkyo_bounds
        }

class EventSpotter:
    def __init__(self, velocity_threshold: float = 0.025, min_event_gap_frames: int = 15):
        self.velocity_threshold = velocity_threshold
        self.min_event_gap_frames = min_event_gap_frames

    def detect_strikes(
        self,
        pose_history: List[Optional[Dict[str, Any]]],
        fps: float = 30.0,
        start_bound_frame: int = 0,
        end_bound_frame: Optional[int] = None,
        attacker_id: str = "KENSHI_AKA",
        attacker_name: str = "Kenshi Aka",
        filter_out_of_bounds: bool = True
    ) -> List[StrikeEvent]:
        """
        Escaneia a história de poses frame a frame e retorna uma lista de StrikeEvents detectados.
        Se filter_out_of_bounds for True, descarta os golpes fora da janela delimitada pelo Sonkyō.
        """
        if len(pose_history) < 10:
            return []

        if end_bound_frame is None:
            end_bound_frame = len(pose_history) - 1

        # 1. Calcular a velocidade 2D/3D dos pulsos
        velocities = []
        hand_y_positions = []
        
        for idx in range(len(pose_history)):
            lm = pose_history[idx]
            if not lm or "RIGHT_WRIST" not in lm:
                velocities.append(0.0)
                hand_y_positions.append(1.0)
                continue

            r_wrist = np.array([lm["RIGHT_WRIST"]["x"], lm["RIGHT_WRIST"]["y"]])
            hand_y_positions.append(r_wrist[1])

            if idx == 0 or not pose_history[idx - 1] or "RIGHT_WRIST" not in pose_history[idx - 1]:
                velocities.append(0.0)
            else:
                prev_lm = pose_history[idx - 1]
                prev_r_wrist = np.array([prev_lm["RIGHT_WRIST"]["x"], prev_lm["RIGHT_WRIST"]["y"]])
                dist = float(np.linalg.norm(r_wrist - prev_r_wrist))
                velocities.append(dist)

        # 2. Identificar picos de velocidade (impactos) com limiar adaptativo
        events: List[StrikeEvent] = []
        n_frames = len(pose_history)
        
        valid_vels = [v for v in velocities if v > 0]
        if not valid_vels:
            return []
            
        mean_vel = np.mean(valid_vels)
        std_vel = np.std(valid_vels)
        max_vel = np.max(valid_vels)
        # Limiar adaptativo robusto
        dynamic_threshold = min(max(0.005, mean_vel + 0.8 * std_vel), max(0.02, max_vel * 0.60))

        i = 5
        while i < n_frames - 10:
            # Buscar início de aceleração brusca
            if velocities[i] > dynamic_threshold:
                # Encontrar o pico local de velocidade (momento do impacto)
                window_end = min(i + 15, n_frames - 1)
                peak_idx = i + int(np.argmax(velocities[i:window_end]))
                
                # Definir janela do golpe: ~10 frames antes do pico até ~25 frames depois
                start_f = max(0, peak_idx - 10)
                end_f = min(n_frames - 1, peak_idx + 25)

                # Classificar o tipo de golpe baseado no movimento da mão e na altura do esqueleto
                strike_type = self._classify_technique(pose_history, start_f, peak_idx)
                
                is_within = (start_bound_frame <= peak_idx <= end_bound_frame)

                event = StrikeEvent(
                    strike_type=strike_type,
                    start_frame=start_f,
                    impact_frame=peak_idx,
                    end_frame=end_f,
                    fps=fps,
                    attacker_id=attacker_id,
                    attacker_name=attacker_name,
                    is_within_sonkyo_bounds=is_within
                )

                if not filter_out_of_bounds or is_within:
                    events.append(event)

                # Pular os próximos frames para evitar duplicidade do mesmo golpe
                i = peak_idx + self.min_event_gap_frames
            else:
                i += 1

        return events

    def _classify_technique(self, pose_history: List[Optional[Dict[str, Any]]], start_f: int, impact_f: int) -> str:
        """
        Classifica a técnica (MEN, KOTE, DO, TSUKI) analisando a trajetória da mão e ombro no momento do impacto.
        """
        lm = pose_history[impact_f] if impact_f < len(pose_history) else None
        if not lm or "RIGHT_WRIST" not in lm or "RIGHT_SHOULDER" not in lm or "RIGHT_HIP" not in lm:
            return "MEN" # Default fallback

        hand_y = lm["RIGHT_WRIST"]["y"]
        shoulder_y = lm["RIGHT_SHOULDER"]["y"]
        hip_y = lm["RIGHT_HIP"]["y"]
        hand_x = lm["RIGHT_WRIST"]["x"]
        elbow_x = lm.get("RIGHT_ELBOW", {}).get("x", hand_x)

        # Se as mãos estão bem elevadas (acima do ombro ou na linha do topo do Men) -> MEN
        if hand_y < shoulder_y:
            return "MEN"
        # Se o golpe é horizontal ou descendente na altura do tronco (entre ombro e quadril)
        elif hand_y >= shoulder_y and hand_y < hip_y:
            # Se houve deslocamento lateral forte -> DO, senão KOTE ou TSUKI
            dx = abs(hand_x - elbow_x)
            if dx > 0.15:
                return "DO"
            elif hand_y > (shoulder_y + hip_y) / 2:
                return "KOTE"
            else:
                return "TSUKI"
        else:
            return "KOTE"

