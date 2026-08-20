"""
Módulo de Detecção e Análise de Sonkyō para Kendo.
Identifica a postura ritualística de Sonkyō (agachamento sobre a ponta dos pés com coluna ereta e joelhos flexionados)
para determinar com precisão o Início Oficial e o Término Oficial da Luta.
"""

import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.utils.logger_manager import log_event

DEFAULT_SONKYO_LEARNED_PATH = "config/sonkyo_learned_profile.json"

DEFAULT_SONKYO_LEARNED_PROFILE: Dict[str, Any] = {
    "samples_count": 0,
    "learned_rel_height_threshold": 0.82,
    "learned_hip_drop_threshold": 0.08,
    "learned_hip_ratio_threshold": 0.48,
    "learned_knee_angle_threshold": 120.0,
    "learned_torso_ratio": 0.60,
    "learned_min_duration_frames": 6,
    "exemplars": [],
    "last_updated_at": None
}

class SonkyoInterval:
    def __init__(self, start_frame: int, end_frame: int, fps: float, interval_type: str = "INITIAL", confidence: float = 0.85, is_detected: bool = True):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.fps = fps
        self.interval_type = interval_type  # "INITIAL", "FINAL" ou "INTERMEDIATE"
        self.confidence = confidence
        self.is_detected = is_detected

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
            "is_detected": self.is_detected,
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

    @staticmethod
    def timestamp_to_frame(timestamp_str: str, fps: float) -> int:
        """Converte uma string de timestamp ('MM:SS.mmm', 'SS.mmm' ou 'SS') em número de quadro."""
        if not timestamp_str:
            return 0
        try:
            ts = str(timestamp_str).strip().lower().replace("s", "")
            if ":" in ts:
                parts = ts.split(":")
                mins = float(parts[0])
                secs = float(parts[1])
                total_sec = mins * 60.0 + secs
            else:
                total_sec = float(ts)
            return max(0, int(round(total_sec * fps)))
        except Exception:
            return 0


class SonkyoDetector:
    def __init__(
        self,
        min_sonkyo_duration_frames: int = 6,
        hip_drop_ratio_threshold: float = 0.48,
        knee_angle_max_threshold: float = 120.0,
        spine_tilt_max_threshold: float = 40.0,
        learned_profile_path: str = DEFAULT_SONKYO_LEARNED_PATH
    ):
        """
        Parâmetros de detecção da postura biomecânica de Sonkyō:
        - min_sonkyo_duration_frames: Duração mínima (em frames) para consolidar um intervalo de Sonkyō.
        - hip_drop_ratio_threshold: Razão máxima (ankle_y - hip_y) / (ankle_y - nose_y). No Sonkyō fica <= 0.48.
        - knee_angle_max_threshold: Ângulo máximo do joelho para agachamento (<= 120°).
        - spine_tilt_max_threshold: Inclinação máxima do tronco em relação à vertical (<= 40°).
        - learned_profile_path: Caminho para persistência do perfil adaptativo de aprendizado do Sonkyō.
        """
        self.min_sonkyo_duration_frames = min_sonkyo_duration_frames
        self.hip_drop_ratio_threshold = hip_drop_ratio_threshold
        self.knee_angle_max_threshold = knee_angle_max_threshold
        self.spine_tilt_max_threshold = spine_tilt_max_threshold
        self.learned_profile_path = learned_profile_path
        
        self.learned_profile: Dict[str, Any] = self._load_learned_profile()
        self._apply_learned_profile()

    def _load_learned_profile(self) -> Dict[str, Any]:
        """Carrega o perfil de aprendizado contínuo do Sonkyō."""
        if os.path.exists(self.learned_profile_path):
            try:
                with open(self.learned_profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    base = DEFAULT_SONKYO_LEARNED_PROFILE.copy()
                    base.update(data)
                    return base
            except Exception as e:
                log_event("WARNING", f"Erro ao ler perfil de aprendizado de Sonkyō ({e}). Usando padrões.", "sonkyo_detector")
        return DEFAULT_SONKYO_LEARNED_PROFILE.copy()

    def _save_learned_profile(self) -> None:
        """Persiste o perfil de aprendizado contínuo do Sonkyō."""
        try:
            os.makedirs(os.path.dirname(self.learned_profile_path), exist_ok=True)
            with open(self.learned_profile_path, "w", encoding="utf-8") as f:
                json.dump(self.learned_profile, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_event("ERROR", f"Falha ao salvar perfil de aprendizado do Sonkyō em '{self.learned_profile_path}': {e}", "sonkyo_detector")

    def _apply_learned_profile(self) -> None:
        """Aplica os hiperparâmetros adaptados pelo histórico de aprendizado."""
        if self.learned_profile.get("samples_count", 0) > 0:
            self.hip_drop_ratio_threshold = self.learned_profile.get("learned_hip_ratio_threshold", self.hip_drop_ratio_threshold)
            self.knee_angle_max_threshold = self.learned_profile.get("learned_knee_angle_threshold", self.knee_angle_max_threshold)
            self.min_sonkyo_duration_frames = self.learned_profile.get("learned_min_duration_frames", self.min_sonkyo_duration_frames)

    def get_learned_stats(self) -> Dict[str, Any]:
        """Retorna resumo das métricas e amostras aprendidas pelo modelo."""
        return {
            "samples_count": self.learned_profile.get("samples_count", 0),
            "learned_rel_height_threshold": self.learned_profile.get("learned_rel_height_threshold", 0.82),
            "learned_hip_drop_threshold": self.learned_profile.get("learned_hip_drop_threshold", 0.08),
            "learned_hip_ratio_threshold": self.learned_profile.get("learned_hip_ratio_threshold", 0.48),
            "learned_knee_angle_threshold": self.learned_profile.get("learned_knee_angle_threshold", 120.0),
            "exemplars_count": len(self.learned_profile.get("exemplars", [])),
            "last_updated_at": self.learned_profile.get("last_updated_at", "Nenhum aprendizado registrado")
        }

    def reset_learned_profile(self) -> None:
        """Restaura o perfil de aprendizado do Sonkyō aos padrões de fábrica."""
        self.learned_profile = DEFAULT_SONKYO_LEARNED_PROFILE.copy()
        self._save_learned_profile()
        self.hip_drop_ratio_threshold = 0.48
        self.knee_angle_max_threshold = 120.0
        self.min_sonkyo_duration_frames = 6
        log_event("INFO", "Perfil de aprendizado do Sonkyō resetado aos padrões de fábrica.", "sonkyo_detector")

    def learn_from_annotation(
        self,
        pose_history: List[Optional[Dict[str, Any]]],
        start_frame: int,
        end_frame: int,
        fps: float = 30.0,
        interval_type: str = "INITIAL",
        save_to_disk: bool = True
    ) -> Dict[str, Any]:
        """
        Extrai as características biomecânicas da pose no intervalo informado pelo usuário
        e atualiza o perfil de aprendizado contínuo persistido para este e todos os futuros vídeos.
        """
        if not pose_history:
            return {"status": "warning", "message": "Histórico de poses vazio para aprendizado."}

        start_f = max(0, min(len(pose_history) - 1, start_frame))
        end_f = max(start_f, min(len(pose_history) - 1, end_frame))
        
        annotated_poses = [pose_history[i] for i in range(start_f, end_f + 1) if i < len(pose_history) and pose_history[i]]
        if not annotated_poses:
            return {
                "status": "warning",
                "message": "Nenhuma pose válida encontrada no intervalo fornecido para aprendizado.",
                "samples_count": self.learned_profile.get("samples_count", 0)
            }
        
        # 1. Baseline de altura do atleta em pé ao longo do vídeo
        valid_heights = []
        valid_hip_ys = []
        for p in pose_history:
            if p:
                all_ys = [pt["y"] for pt in p.values() if isinstance(pt, dict) and "y" in pt]
                if all_ys:
                    valid_heights.append(max(all_ys) - min(all_ys))
                if "RIGHT_HIP" in p or "LEFT_HIP" in p:
                    hy = [p[k]["y"] for k in ["RIGHT_HIP", "LEFT_HIP"] if k in p]
                    valid_hip_ys.append(float(np.mean(hy)))

        standing_height = float(np.percentile(valid_heights, 85)) if len(valid_heights) >= 5 else 0.60
        standing_hip_y = float(np.percentile(valid_hip_ys, 15)) if len(valid_hip_ys) >= 5 else 0.58

        # 2. Extrair métricas médias da janela anotada
        observed_hip_ratios = []
        observed_torso_ratios = []
        observed_knee_angles = []
        observed_spine_tilts = []
        observed_rel_heights = []
        observed_hip_drops = []

        for p in annotated_poses:
            _, _, m = self.evaluate_sonkyo_pose(p)
            observed_hip_ratios.append(m["hip_ratio"])
            observed_torso_ratios.append(m["torso_ratio"])
            observed_knee_angles.append(m["knee_angle"])
            observed_spine_tilts.append(m["spine_tilt"])

            all_ys = [pt["y"] for pt in p.values() if isinstance(pt, dict) and "y" in pt]
            if all_ys and standing_height > 0.05:
                observed_rel_heights.append((max(all_ys) - min(all_ys)) / standing_height)

            hip_ys = [p[k]["y"] for k in ["RIGHT_HIP", "LEFT_HIP"] if k in p]
            if hip_ys and standing_hip_y > 0.05:
                observed_hip_drops.append(float(np.mean(hip_ys)) - standing_hip_y)

        mean_hip_ratio = float(np.mean(observed_hip_ratios)) if observed_hip_ratios else self.hip_drop_ratio_threshold
        mean_torso_ratio = float(np.mean(observed_torso_ratios)) if observed_torso_ratios else 0.60
        mean_knee_angle = float(np.mean(observed_knee_angles)) if observed_knee_angles else self.knee_angle_max_threshold
        mean_spine_tilt = float(np.mean(observed_spine_tilts)) if observed_spine_tilts else self.spine_tilt_max_threshold
        mean_rel_height = float(np.mean(observed_rel_heights)) if observed_rel_heights else 0.70
        mean_hip_drop = float(np.mean(observed_hip_drops)) if observed_hip_drops else 0.12

        duration_frames = end_f - start_f + 1

        # 3. Atualizar o Perfil com Média Ponderada
        samples = self.learned_profile.get("samples_count", 0)
        alpha = 0.35 if samples > 0 else 0.80  # Taxa de aprendizado adaptativa

        curr_rel_h = self.learned_profile.get("learned_rel_height_threshold", 0.82)
        curr_hip_drop = self.learned_profile.get("learned_hip_drop_threshold", 0.08)
        curr_hip_r = self.learned_profile.get("learned_hip_ratio_threshold", self.hip_drop_ratio_threshold)
        curr_knee = self.learned_profile.get("learned_knee_angle_threshold", self.knee_angle_max_threshold)

        new_rel_h = float((1 - alpha) * curr_rel_h + alpha * (mean_rel_height + 0.06))
        new_hip_drop = float((1 - alpha) * curr_hip_drop + alpha * max(0.04, mean_hip_drop - 0.02))
        new_hip_r = float((1 - alpha) * curr_hip_r + alpha * (mean_hip_ratio + 0.04))
        new_knee = float((1 - alpha) * curr_knee + alpha * min(150.0, mean_knee_angle + 10.0))

        self.learned_profile["learned_rel_height_threshold"] = round(new_rel_h, 3)
        self.learned_profile["learned_hip_drop_threshold"] = round(new_hip_drop, 3)
        self.learned_profile["learned_hip_ratio_threshold"] = round(new_hip_r, 3)
        self.learned_profile["learned_knee_angle_threshold"] = round(new_knee, 1)
        self.learned_profile["learned_torso_ratio"] = round(float(mean_torso_ratio), 3)
        self.learned_profile["samples_count"] = samples + 1
        self.learned_profile["last_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Armazenar exemplar de referência
        exemplars = self.learned_profile.get("exemplars", [])
        exemplar_entry = {
            "interval_type": interval_type,
            "duration_frames": duration_frames,
            "mean_hip_ratio": round(mean_hip_ratio, 3),
            "mean_rel_height": round(mean_rel_height, 3),
            "mean_knee_angle": round(mean_knee_angle, 1),
            "mean_spine_tilt": round(mean_spine_tilt, 1)
        }
        exemplars.append(exemplar_entry)
        if len(exemplars) > 50:
            exemplars = exemplars[-50:]
        self.learned_profile["exemplars"] = exemplars

        self._apply_learned_profile()

        if save_to_disk:
            self._save_learned_profile()

        log_event(
            "INFO",
            f"SonkyoDetector aprendeu novo padrão de Sonkyō ({interval_type} #{samples+1}): "
            f"RelHeight={new_rel_h:.2f}, HipRatio={new_hip_r:.2f}, HipDrop={new_hip_drop:.2f}, KneeAngle={new_knee:.1f}°.",
            "sonkyo_detector"
        )

        return {
            "status": "success",
            "message": f"Padrão de Sonkyō aprendido com sucesso (Amostra #{samples+1}).",
            "samples_count": self.learned_profile["samples_count"],
            "learned_rel_height": new_rel_h,
            "learned_hip_ratio": new_hip_r,
            "learned_hip_drop": new_hip_drop
        }

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
        secondary_pose_history: Optional[List[Optional[Dict[str, Any]]]] = None,
        initial_sonkyo_override: Optional[Dict[str, Any]] = None,
        final_sonkyo_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Escaneia o histórico de poses dos combatentes com análise multi-sinal adaptativa e temporal
        para determinar os limites exatos da luta.
        Permite receber correções manuais de Sonkyō do usuário (overrides), aprendendo imediatamente
        as posturas e persistindo o aprendizado para este e todos os futuros vídeos.
        """
        total_frames = len(pose_history)
        if total_frames < self.min_sonkyo_duration_frames:
            half = max(1, total_frames // 2)
            init_s = SonkyoInterval(0, half, fps, interval_type="INITIAL", confidence=0.50, is_detected=False)
            fin_s = SonkyoInterval(half, max(half, total_frames - 1), fps, interval_type="FINAL", confidence=0.50, is_detected=False)
            return {
                "is_bounded": True,
                "has_initial_sonkyo": True,
                "has_final_sonkyo": True,
                "match_start_frame": half,
                "match_end_frame": max(half, total_frames - 1),
                "match_start_timestamp": SonkyoInterval.frame_to_timestamp(half, fps),
                "match_end_timestamp": SonkyoInterval.frame_to_timestamp(max(0, total_frames - 1), fps),
                "effective_combat_duration_seconds": round(total_frames / fps, 2),
                "initial_sonkyo": init_s.to_dict(),
                "final_sonkyo": fin_s.to_dict(),
                "sonkyo_intervals": [init_s.to_dict(), fin_s.to_dict()],
                "status_message": "Vídeo curto: rituais de Sonkyō definidos no início e término do vídeo (ajustáveis na edição).",
                "learning_logs": [],
                "learned_samples_total": self.learned_profile.get("samples_count", 0)
            }

        # 1. Obter curvas de probabilidade por frame para cada combatente
        learned_rel_h = self.learned_profile.get("learned_rel_height_threshold", 0.82)
        learned_hip_drop = self.learned_profile.get("learned_hip_drop_threshold", 0.08)

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

                # Avaliação da compressão de altura relativa ao atleta em pé (com threshold adaptado)
                all_ys = [pt["y"] for pt in p.values() if isinstance(pt, dict) and "y" in pt]
                if all_ys and standing_height > 0.05:
                    h_curr = max(all_ys) - min(all_ys)
                    rel_height_ratio = h_curr / standing_height
                    rel_height_score = float(np.clip((learned_rel_h - rel_height_ratio) / 0.22, 0.0, 1.0))
                else:
                    rel_height_score = 0.0

                # Avaliação de rebaixamento do quadril em relação ao baseline em pé
                hip_ys = [p[k]["y"] for k in ["RIGHT_HIP", "LEFT_HIP"] if k in p]
                if hip_ys and standing_hip_y > 0.05:
                    h_y_curr = float(np.mean(hip_ys))
                    hip_drop_delta = h_y_curr - standing_hip_y
                    hip_drop_score = float(np.clip(hip_drop_delta / max(0.04, learned_hip_drop * 1.5), 0.0, 1.0))
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

        # 2. Suavização Temporal e Fechamento Morfológico
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

        # Sonkyō Inicial: Automático ou Fallback no Início do Vídeo
        early_candidates = [it for it in merged_intervals if it.start_frame <= half_frames]
        if early_candidates:
            initial_sonkyo = early_candidates[0]
            initial_sonkyo.interval_type = "INITIAL"
            initial_sonkyo.is_detected = True
            has_initial = True
            match_start_frame = min(total_frames - 1, initial_sonkyo.end_frame + 2)
        else:
            # Caso não detectado automaticamente: Incluir movimento de Sonkyō no início do vídeo
            def_init_len = min(int(fps * 1.5), max(self.min_sonkyo_duration_frames, int(total_frames * 0.15)))
            initial_sonkyo = SonkyoInterval(0, max(1, def_init_len), fps, interval_type="INITIAL", confidence=0.50, is_detected=False)
            has_initial = True
            match_start_frame = min(total_frames - 1, initial_sonkyo.end_frame + 2)

        # Sonkyō Final: Automático ou Fallback no Final do Vídeo
        min_final_start = (initial_sonkyo.end_frame + 15) if initial_sonkyo else int(total_frames * 0.40)
        late_candidates = [it for it in merged_intervals if it.end_frame >= half_frames and it.start_frame >= min_final_start]
        
        if late_candidates:
            final_sonkyo = late_candidates[-1]
            final_sonkyo.interval_type = "FINAL"
            final_sonkyo.is_detected = True
            has_final = True
            match_end_frame = max(match_start_frame, final_sonkyo.start_frame - 2)
        else:
            # Caso não detectado automaticamente: Incluir movimento de Sonkyō no final do vídeo
            def_fin_len = min(int(fps * 1.5), max(self.min_sonkyo_duration_frames, int(total_frames * 0.15)))
            fin_start_f = max(match_start_frame + 5, total_frames - def_fin_len)
            final_sonkyo = SonkyoInterval(fin_start_f, max(fin_start_f + 1, total_frames - 1), fps, interval_type="FINAL", confidence=0.50, is_detected=False)
            has_final = True
            match_end_frame = max(match_start_frame, final_sonkyo.start_frame - 2)

        # 5. Aplicação de Overrides do Usuário e Aprendizado Contínuo
        learning_logs = []
        if initial_sonkyo_override:
            s_f = initial_sonkyo_override.get("start_frame")
            if s_f is None:
                s_f = SonkyoInterval.timestamp_to_frame(initial_sonkyo_override.get("start_timestamp", "00:00.000"), fps)
            e_f = initial_sonkyo_override.get("end_frame")
            if e_f is None:
                e_f = SonkyoInterval.timestamp_to_frame(initial_sonkyo_override.get("end_timestamp", "00:00.000"), fps)
            
            s_f = max(0, min(total_frames - 1, int(s_f)))
            e_f = max(s_f, min(total_frames - 1, int(e_f)))
            
            initial_sonkyo = SonkyoInterval(s_f, e_f, fps, interval_type="INITIAL", confidence=1.0, is_detected=True)
            has_initial = True
            match_start_frame = min(total_frames - 1, initial_sonkyo.end_frame + 2)
            
            # Executar aprendizado sobre a movimentação do Sonkyō Inicial
            learn_res = self.learn_from_annotation(pose_history, s_f, e_f, fps, interval_type="INITIAL")
            learning_logs.append(f"Sonkyō Inicial aprendido: {learn_res.get('message', '')}")

        if final_sonkyo_override:
            s_f = final_sonkyo_override.get("start_frame")
            if s_f is None:
                s_f = SonkyoInterval.timestamp_to_frame(final_sonkyo_override.get("start_timestamp", "00:00.000"), fps)
            e_f = final_sonkyo_override.get("end_frame")
            if e_f is None:
                e_f = SonkyoInterval.timestamp_to_frame(final_sonkyo_override.get("end_timestamp", "00:00.000"), fps)
            
            s_f = max(0, min(total_frames - 1, int(s_f)))
            e_f = max(s_f, min(total_frames - 1, int(e_f)))
            
            final_sonkyo = SonkyoInterval(s_f, e_f, fps, interval_type="FINAL", confidence=1.0, is_detected=True)
            has_final = True
            match_end_frame = max(match_start_frame, final_sonkyo.start_frame - 2)
            
            # Executar aprendizado sobre a movimentação do Sonkyō Final
            learn_res = self.learn_from_annotation(pose_history, s_f, e_f, fps, interval_type="FINAL")
            learning_logs.append(f"Sonkyō Final aprendido: {learn_res.get('message', '')}")

        # Garantir consistência lógica
        if match_end_frame <= match_start_frame:
            match_start_frame = 0
            match_end_frame = max(0, total_frames - 1)
            is_bounded = True
        else:
            is_bounded = True

        effective_duration = max(0.0, (match_end_frame - match_start_frame) / fps)

        init_detected = initial_sonkyo.is_detected
        fin_detected = final_sonkyo.is_detected

        if initial_sonkyo_override or final_sonkyo_override:
            status_msg = f"Limites de Sonkyō atualizados e aprendidos pelo usuário ({len(learning_logs)} ritual(is) calibrado(s))."
        elif init_detected and fin_detected:
            status_msg = "Combate delimitado por Sonkyō (Início e Fim detectados com sucesso)."
        elif init_detected and not fin_detected:
            status_msg = "Sonkyō Inicial detectado com sucesso. Sonkyō Final posicionado no encerramento do vídeo."
        elif not init_detected and fin_detected:
            status_msg = "Sonkyō Inicial posicionado no início do vídeo. Sonkyō Final detectado com sucesso."
        else:
            status_msg = "Movimentos de Sonkyō definidos no início e término do vídeo (ajustáveis na edição)."

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
            "status_message": status_msg,
            "learning_logs": learning_logs,
            "learned_samples_total": self.learned_profile.get("samples_count", 0)
        }
