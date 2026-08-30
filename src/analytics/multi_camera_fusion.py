"""
Motor de Fusão e Consenso Multi-Câmeras para Arbitragem de Kendo (ShinpanAI).
Implementa a validação conjunta de técnicas (Yuko-Datotsu) baseada no conjunto de imagens
das câmeras ativas, com escalonamento de quórum e confirmação em múltiplos ângulos de visão.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.analytics.event_spotter import EventSpotter, StrikeEvent
from src.analytics.biomechanics import BiomechanicsAnalyzer
from src.utils.logger_manager import log_event


class CameraFrameEvidence:
    """
    Evidência e métricas extraídas dos frames de uma câmera específica
    durante a janela temporal de um candidato a golpe.
    """
    def __init__(
        self,
        camera_id: int,
        camera_label: str,
        impact_frame: int,
        timestamp: str,
        technique_detected: Optional[str],
        is_confirmed: bool,
        confidence_score: float,
        wrist_velocity: float,
        target_proximity: float,
        posture_stability: float,
        visibility_ok: bool = True,
        notes: str = "",
        snapshot_frame: Optional[np.ndarray] = None
    ):
        self.camera_id = camera_id
        self.camera_label = camera_label
        self.impact_frame = impact_frame
        self.timestamp = timestamp
        self.technique_detected = technique_detected
        self.is_confirmed = is_confirmed
        self.confidence_score = round(confidence_score, 1)  # 0 a 100
        self.wrist_velocity = round(wrist_velocity, 4)
        self.target_proximity = round(target_proximity, 1)  # 0 a 100
        self.posture_stability = round(posture_stability, 1) # 0 a 100
        self.visibility_ok = visibility_ok
        self.notes = notes
        self.snapshot_frame = snapshot_frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "camera_label": self.camera_label,
            "impact_frame": self.impact_frame,
            "timestamp": self.timestamp,
            "technique_detected": self.technique_detected,
            "is_confirmed": self.is_confirmed,
            "confidence_score": self.confidence_score,
            "wrist_velocity": self.wrist_velocity,
            "target_proximity": self.target_proximity,
            "posture_stability": self.posture_stability,
            "visibility_ok": self.visibility_ok,
            "notes": self.notes
        }


class MultiCameraStrikeEvaluation:
    """
    Resultado da avaliação e julgamento conjunto do golpe através do conjunto de câmeras.
    """
    def __init__(
        self,
        technique: str,
        impact_frame_ref: int,
        timestamp_ref: str,
        attacker_id: str,
        attacker_name: str,
        num_active_cameras: int,
        num_confirming_cameras: int,
        required_quorum: int,
        consensus_ratio: float,
        is_strike_confirmed: bool,
        joint_score: float,
        decision_status: str,
        camera_evidences: List[CameraFrameEvidence],
        summary_text: str
    ):
        self.technique = technique
        self.impact_frame_ref = impact_frame_ref
        self.timestamp_ref = timestamp_ref
        self.attacker_id = attacker_id
        self.attacker_name = attacker_name
        self.num_active_cameras = num_active_cameras
        self.num_confirming_cameras = num_confirming_cameras
        self.required_quorum = required_quorum
        self.consensus_ratio = round(consensus_ratio, 3)
        self.is_strike_confirmed = is_strike_confirmed
        self.joint_score = round(joint_score, 1)
        self.decision_status = decision_status
        self.camera_evidences = camera_evidences
        self.summary_text = summary_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "technique": self.technique,
            "impact_frame_ref": self.impact_frame_ref,
            "timestamp_ref": self.timestamp_ref,
            "attacker_id": self.attacker_id,
            "attacker_name": self.attacker_name,
            "num_active_cameras": self.num_active_cameras,
            "num_confirming_cameras": self.num_confirming_cameras,
            "required_quorum": self.required_quorum,
            "consensus_ratio": self.consensus_ratio,
            "consensus_pct": round(self.consensus_ratio * 100, 1),
            "is_strike_confirmed": self.is_strike_confirmed,
            "joint_score": self.joint_score,
            "decision_status": self.decision_status,
            "summary_text": self.summary_text,
            "camera_evidences": [ev.to_dict() for ev in self.camera_evidences]
        }


class MultiCameraFusionEngine:
    """
    Motor de Consenso e Fusão Multi-Câmeras.
    
    Princípio Central:
    A definição de haver ou não o golpe deve ser tomada com base no conjunto das imagens
    das câmeras. Quanto mais câmeras disponíveis, mais rigorosa e necessária é a confirmação
    em imagens/frames da realização da técnica (quorum escalonado).
    """

    # Quórum Mínimo por Perfil e Quantidade de Câmeras
    QUORUM_RATIOS = {
        "permissivo": 0.50, # Pelo menos 50% de confirmação (ex: 2/3 ou 2/4)
        "normal": 0.60,     # Maioria qualificada (ex: 2/2, 2/3, 3/4)
        "rigido": 0.80      # Unanimidade / alta exigência (ex: 2/2, 3/3, 4/4)
    }

    def __init__(
        self,
        sync_window_frames: int = 10,
        profile_name: str = "normal",
        cooldown_frames: int = 20
    ):
        self.sync_window_frames = sync_window_frames
        self.profile_name = profile_name
        self.cooldown_frames = cooldown_frames
        self.event_spotter = EventSpotter()
        self.biomechanics = BiomechanicsAnalyzer()
        self.last_confirmed_frame = -999

    @classmethod
    def calculate_required_quorum(cls, num_cameras: int, profile_name: str = "normal") -> int:
        """
        Calcula o quórum mínimo de câmeras necessárias para validar um golpe.
        Regra fundamental:
          - N = 1: 1 câmera (100% monocular)
          - N = 2: 2 câmeras (100% de confirmação cruzada obrigatória)
          - N = 3: 2 câmeras (permissivo/normal) ou 3 câmeras (rígido)
          - N = 4: 3 câmeras (normal) ou 4 câmeras (rígido)
        """
        if num_cameras <= 1:
            return 1
        if num_cameras == 2:
            return 2 # Com 2 câmeras, ambas devem confirmar para evitar artefatos de perspectiva unilateral

        ratio = cls.QUORUM_RATIOS.get(profile_name.lower(), 0.60)
        req = math.ceil(num_cameras * ratio)
        return max(2, min(num_cameras, req))

    def evaluate_camera_evidence_in_window(
        self,
        pose_history: List[Optional[Dict[str, Any]]],
        camera_id: int,
        camera_label: str,
        center_frame: int,
        fps: float,
        target_technique: str,
        window_size: Optional[int] = None,
        snapshot_frame: Optional[np.ndarray] = None
    ) -> CameraFrameEvidence:
        """
        Avalia os frames de uma câmera específica dentro da janela temporal [center_frame - W, center_frame + W].
        Examina a presença de aceleração de pulso, trajetória do Shinai e estabilidade postural.
        """
        w = window_size if window_size is not None else self.sync_window_frames
        total_len = len(pose_history)
        
        if total_len == 0:
            return CameraFrameEvidence(
                camera_id=camera_id,
                camera_label=camera_label,
                impact_frame=center_frame,
                timestamp=self._frame_to_timestamp(center_frame, fps),
                technique_detected=None,
                is_confirmed=False,
                confidence_score=0.0,
                wrist_velocity=0.0,
                target_proximity=0.0,
                posture_stability=0.0,
                visibility_ok=False,
                notes="Sem sinal de vídeo / pose indisponível",
                snapshot_frame=snapshot_frame
            )

        start_f = max(0, center_frame - w)
        end_f = min(total_len - 1, center_frame + w)

        # Analisar poses disponíveis na janela
        valid_poses = [pose_history[f] for f in range(start_f, end_f + 1) if pose_history[f] is not None]
        if len(valid_poses) < max(2, (end_f - start_f + 1) // 3):
            return CameraFrameEvidence(
                camera_id=camera_id,
                camera_label=camera_label,
                impact_frame=center_frame,
                timestamp=self._frame_to_timestamp(center_frame, fps),
                technique_detected=None,
                is_confirmed=False,
                confidence_score=10.0,
                wrist_velocity=0.0,
                target_proximity=0.0,
                posture_stability=0.0,
                visibility_ok=False,
                notes="Oclusão severa de praticante no ângulo",
                snapshot_frame=snapshot_frame
            )

        # 1. Calcular velocidades e pico no intervalo
        max_vel = 0.0
        best_peak_f = center_frame
        
        for f in range(start_f + 1, end_f + 1):
            curr_lm = pose_history[f]
            prev_lm = pose_history[f - 1]
            if curr_lm and prev_lm and "RIGHT_WRIST" in curr_lm and "RIGHT_WRIST" in prev_lm:
                c_w = np.array([curr_lm["RIGHT_WRIST"]["x"], curr_lm["RIGHT_WRIST"]["y"]])
                p_w = np.array([prev_lm["RIGHT_WRIST"]["x"], prev_lm["RIGHT_WRIST"]["y"]])
                vel = float(np.linalg.norm(c_w - p_w))
                if vel > max_vel:
                    max_vel = vel
                    best_peak_f = f

        # 2. Avaliação de Proximidade e Alvo
        impact_pose = pose_history[best_peak_f] if best_peak_f < total_len else None
        if not impact_pose:
            impact_pose = valid_poses[-1]

        target_score = self.biomechanics.evaluate_target_impact(target_technique, impact_pose)
        posture_score = self.biomechanics.evaluate_posture(impact_pose)

        # Classificação da técnica neste ângulo
        detected_tech = self.event_spotter._classify_technique(pose_history, max(0, best_peak_f - 10), best_peak_f)

        # Critério de confirmação nesta câmera:
        # Velocidade acima do limiar mínimo + pontuação de impacto no alvo razoável + compatibilidade de técnica
        has_motion = max_vel >= 0.015
        has_target_contact = target_score >= 0.40
        tech_matches = (detected_tech == target_technique or target_score >= 0.60)

        is_confirmed = (has_motion and has_target_contact and tech_matches)

        # Score individual de confiança (0 a 100)
        confidence = (
            min(1.0, max_vel / 0.06) * 40.0 +
            target_score * 40.0 +
            posture_score * 20.0
        )

        notes = []
        if is_confirmed:
            notes.append(f"Técnica {detected_tech} confirmada no frame #{best_peak_f}")
        else:
            if not has_motion:
                notes.append("Sem aceleração significativa de pulso")
            if not has_target_contact:
                notes.append("Fora da área de alvo no ângulo")
            if not tech_matches:
                notes.append(f"Classificado como {detected_tech} em vez de {target_technique}")

        return CameraFrameEvidence(
            camera_id=camera_id,
            camera_label=camera_label,
            impact_frame=best_peak_f,
            timestamp=self._frame_to_timestamp(best_peak_f, fps),
            technique_detected=detected_tech if has_motion else None,
            is_confirmed=is_confirmed,
            confidence_score=confidence,
            wrist_velocity=max_vel,
            target_proximity=target_score * 100.0,
            posture_stability=posture_score * 100.0,
            visibility_ok=True,
            notes="; ".join(notes),
            snapshot_frame=snapshot_frame
        )

    def evaluate_multi_camera_strike(
        self,
        camera_histories: List[List[Optional[Dict[str, Any]]]],
        camera_labels: List[str],
        reference_cam_idx: int,
        impact_frame: int,
        technique: str,
        fps: float = 30.0,
        attacker_id: str = "KENSHI_AKA",
        attacker_name: str = "Kenshi Aka",
        snapshot_frames: Optional[List[Optional[np.ndarray]]] = None
    ) -> MultiCameraStrikeEvaluation:
        """
        Executa o julgamento conjunto do golpe através do conjunto de câmeras ativas.
        """
        num_cameras = len(camera_histories)
        required_quorum = self.calculate_required_quorum(num_cameras, self.profile_name)

        evidences: List[CameraFrameEvidence] = []

        for idx in range(num_cameras):
            history = camera_histories[idx]
            label = camera_labels[idx] if idx < len(camera_labels) else f"Câmera {idx + 1}"
            snap = snapshot_frames[idx] if (snapshot_frames and idx < len(snapshot_frames)) else None

            # Para a câmera de referência onde o evento disparou, avaliar diretamente
            if idx == reference_cam_idx:
                ev = self.evaluate_camera_evidence_in_window(
                    pose_history=history,
                    camera_id=idx + 1,
                    camera_label=label,
                    center_frame=impact_frame,
                    fps=fps,
                    target_technique=technique,
                    window_size=self.sync_window_frames,
                    snapshot_frame=snap
                )
                # A câmera que disparou é garantidamente um candidato
                ev.is_confirmed = True
                ev.confidence_score = max(ev.confidence_score, 60.0)
            else:
                # Câmeras secundárias são cruzadas na janela de sincronização
                ev = self.evaluate_camera_evidence_in_window(
                    pose_history=history,
                    camera_id=idx + 1,
                    camera_label=label,
                    center_frame=impact_frame,
                    fps=fps,
                    target_technique=technique,
                    window_size=self.sync_window_frames,
                    snapshot_frame=snap
                )

            evidences.append(ev)

        confirming_cams = [ev for ev in evidences if ev.is_confirmed]
        num_confirming = len(confirming_cams)
        consensus_ratio = num_confirming / max(1, num_cameras)

        # Fusão de Scores das Câmeras Confirmadoras
        if confirming_cams:
            joint_score = float(np.mean([ev.confidence_score for ev in confirming_cams]))
        else:
            joint_score = 0.0

        # Verificação do Quórum
        is_confirmed = (num_confirming >= required_quorum)

        # Definição do Status de Decisão
        if is_confirmed:
            decision_status = "CONFIRMED_MULTICAM"
            summary_text = (
                f"🎯 GOLPE VALIDADO POR CONSENSO MULTI-CÂMERAS ({num_confirming}/{num_cameras} câmeras confirmaram {technique}, "
                f"Quórum Requerido: {required_quorum}/{num_cameras}, Confiança Conjunta: {joint_score:.1f}%)."
            )
        elif num_confirming == 1 and num_cameras > 1:
            decision_status = "REJECTED_SINGLE_ANGLE"
            summary_text = (
                f"⚠️ GOLPE REJEITADO (VISÃO UNILATERAL): Detectado apenas na Câmera {confirming_cams[0].camera_id}, "
                f"mas não confirmado nas demais {num_cameras - 1} câmeras (Quórum exigido: {required_quorum}/{num_cameras})."
            )
        elif num_confirming < required_quorum:
            decision_status = "REJECTED_INSUFFICIENT_CONSENSUS"
            summary_text = (
                f"⚠️ GOLPE REJEITADO POR FALTA DE CONSENSO: Apenas {num_confirming}/{num_cameras} câmeras confirmaram a técnica "
                f"(Necessário pelo menos {required_quorum}/{num_cameras} para validação oficial)."
            )
        else:
            decision_status = "OCCLUDED_OR_UNCERTAIN"
            summary_text = "⚠️ GOLPE INCONCLUSIVO: Oclusão generalizada ou baixa visibilidade no conjunto de câmeras."

        ref_ts = self._frame_to_timestamp(impact_frame, fps)

        return MultiCameraStrikeEvaluation(
            technique=technique,
            impact_frame_ref=impact_frame,
            timestamp_ref=ref_ts,
            attacker_id=attacker_id,
            attacker_name=attacker_name,
            num_active_cameras=num_cameras,
            num_confirming_cameras=num_confirming,
            required_quorum=required_quorum,
            consensus_ratio=consensus_ratio,
            is_strike_confirmed=is_confirmed,
            joint_score=joint_score,
            decision_status=decision_status,
            camera_evidences=evidences,
            summary_text=summary_text
        )

    def evaluate_live_step(
        self,
        live_pose_histories: List[List[Optional[Dict[str, Any]]]],
        camera_configs: List[Dict[str, Any]],
        current_fps: float = 30.0,
        current_frame_idx: int = 0,
        latest_frames: Optional[List[Optional[np.ndarray]]] = None
    ) -> Optional[MultiCameraStrikeEvaluation]:
        """
        Passo de avaliação em tempo real (Modo Ao Vivo Multi-Câmeras).
        Examina todas as câmeras ativas. Se alguma detectar um candidato de golpe
        e houver decorrido o período de resfriamento (cooldown), dispara a verificação
        de consenso em todo o conjunto de câmeras.
        """
        num_cameras = len(live_pose_histories)
        if num_cameras == 0:
            return None

        # Checar se estamos no cooldown
        if current_frame_idx - self.last_confirmed_frame < self.cooldown_frames:
            return None

        camera_labels = [cfg.get("label", f"Câmera {i+1}") for i, cfg in enumerate(camera_configs)]

        # Varrer cada câmera para buscar disparo candidato
        for k in range(num_cameras):
            hist_k = live_pose_histories[k]
            if len(hist_k) < 15:
                continue

            # Buscar strikes na janela recente (últimos 30 frames)
            recent_slice = hist_k[-30:]
            detected_strikes = self.event_spotter.detect_strikes(recent_slice, fps=current_fps)
            
            if detected_strikes:
                last_strike = detected_strikes[-1]
                # Frame de impacto relativo à história global
                rel_impact = last_strike.impact_frame
                global_impact = len(hist_k) - (30 - rel_impact)
                
                # Executar a fusão e validação no conjunto completo de câmeras
                multicam_eval = self.evaluate_multi_camera_strike(
                    camera_histories=live_pose_histories,
                    camera_labels=camera_labels,
                    reference_cam_idx=k,
                    impact_frame=global_impact,
                    technique=last_strike.type,
                    fps=current_fps,
                    attacker_id=last_strike.attacker_id,
                    attacker_name=last_strike.attacker_name,
                    snapshot_frames=latest_frames
                )

                self.last_confirmed_frame = current_frame_idx

                # Log do evento de fusão
                log_lvl = "INFO" if multicam_eval.is_strike_confirmed else "WARNING"
                log_event(log_lvl, multicam_eval.summary_text, "multi_camera_fusion")

                return multicam_eval

        return None

    def _frame_to_timestamp(self, frame: int, fps: float) -> str:
        seconds = max(0.0, frame / max(0.1, fps))
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"
