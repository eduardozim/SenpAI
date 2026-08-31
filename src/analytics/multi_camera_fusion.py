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
from src.engine.calibrator import CalibrationEngine
from src.engine.reporter import DiagnosticReporter
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
        fumikomi_sync: float = 0.0,
        fumikomi_offset_ms: float = 0.0,
        zanshin_score: float = 0.0,
        is_ippon: bool = False,
        failed_subcriteria: Optional[List[str]] = None,
        diagnostic_report: str = "",
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
        self.fumikomi_sync = round(fumikomi_sync, 1)
        self.fumikomi_offset_ms = round(fumikomi_offset_ms, 1)
        self.zanshin_score = round(zanshin_score, 1)
        self.is_ippon = is_ippon
        self.failed_subcriteria = failed_subcriteria or []
        self.diagnostic_report = diagnostic_report
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
            "fumikomi_sync": self.fumikomi_sync,
            "fumikomi_offset_ms": self.fumikomi_offset_ms,
            "zanshin_score": self.zanshin_score,
            "is_ippon": self.is_ippon,
            "failed_subcriteria": self.failed_subcriteria,
            "diagnostic_report": self.diagnostic_report,
            "visibility_ok": self.visibility_ok,
            "notes": self.notes
        }


class MultiCameraStrikeEvaluation:
    """
    Resultado da avaliação e julgamento conjunto do golpe através do conjunto de câmeras.
    Contém a análise integral de Yūko-Datotsu (Ki-Ken-Tai-Ichi) para cada golpe (Ippon ou não).
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
        summary_text: str,
        yuko_datotsu_analysis: Optional[Dict[str, Any]] = None
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
        self.yuko_datotsu_analysis = yuko_datotsu_analysis or {}

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
            "camera_evidences": [ev.to_dict() for ev in self.camera_evidences],
            "yuko_datotsu_analysis": self.yuko_datotsu_analysis
        }


class MultiCameraFusionEngine:
    """
    Motor de Consenso e Fusão Multi-Câmeras ancorado no Modelo de Calibração e Treinamento.
    
    Princípio Central:
    A definição de haver ou não o golpe é tomada com base nas evidências biomecânicas
    das câmeras ativas, avaliadas rigorosamente pelo modelo de calibração (Ki-Ken-Tai-Ichi).
    Com 1 câmera, o golpe só é marcado se houver movimentação real e conformidade biomecânica.
    Com múltiplas câmeras, escalona-se o quórum de confirmação entre os ângulos de visão.
    """

    # Quórum Mínimo por Perfil e Quantidade de Câmeras
    QUORUM_RATIOS = {
        "permissivo": 0.50, # Pelo menos 50% de confirmação (ex: 2/3 ou 2/4)
        "normal": 0.60,     # Maioria qualificada (ex: 2/2, 2/3, 3/4)
        "rigido": 0.80      # Unanimidade / alta exigência (ex: 2/2, 3/3, 4/4)
    }

    # Limiares de velocidade mínima de pulso (rejeição de postura estática e ruído de câmera)
    PROFILE_VELOCITY_THRESHOLDS = {
        "permissivo": 0.018,
        "normal": 0.025,
        "rigido": 0.032
    }

    def __init__(
        self,
        sync_window_frames: int = 10,
        profile_name: str = "normal",
        cooldown_frames: int = 20,
        calibrator: Optional[CalibrationEngine] = None,
        config_path: str = "config/calibration_profiles.json"
    ):
        self.sync_window_frames = sync_window_frames
        self._profile_name = profile_name
        self.cooldown_frames = cooldown_frames
        self.calibrator = calibrator or CalibrationEngine(config_path=config_path, profile_name=profile_name)
        self.biomechanics = BiomechanicsAnalyzer()
        self.last_confirmed_frame = -999
        
        self.active_velocity_threshold = self.PROFILE_VELOCITY_THRESHOLDS.get(profile_name.lower(), 0.025)
        self.event_spotter = EventSpotter(velocity_threshold=self.active_velocity_threshold)

    @property
    def profile_name(self) -> str:
        return self._profile_name

    @profile_name.setter
    def profile_name(self, value: str):
        self._profile_name = value
        if hasattr(self, "calibrator") and self.calibrator is not None:
            self.calibrator.set_profile(value)
        self.active_velocity_threshold = self.PROFILE_VELOCITY_THRESHOLDS.get(str(value).lower(), 0.025)
        if hasattr(self, "event_spotter") and self.event_spotter is not None:
            self.event_spotter.velocity_threshold = self.active_velocity_threshold

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
        Examina a presença de movimentação física real e conformidade com o modelo de calibração de Kendo.
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
                confidence_score=0.0,
                wrist_velocity=0.0,
                target_proximity=0.0,
                posture_stability=0.0,
                visibility_ok=False,
                notes="Oclusão severa de praticante no ângulo",
                snapshot_frame=snapshot_frame
            )

        # 1. Calcular velocidades e pico de movimentação no intervalo
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

        # 2. Avaliação Biomecânica no Ponto de Impacto (Ki-Ken-Tai-Ichi)
        impact_pose = pose_history[best_peak_f] if best_peak_f < total_len else None
        if not impact_pose:
            impact_pose = valid_poses[-1]

        target_score = self.biomechanics.evaluate_target_impact(target_technique, impact_pose)
        fumikomi_score, fumikomi_offset = self.biomechanics.evaluate_fumikomi_sync(pose_history, best_peak_f)
        fumikomi_offset_ms = float(fumikomi_offset * (1000.0 / max(1.0, fps)))
        posture_score = self.biomechanics.evaluate_posture(impact_pose)
        zanshin_score = self.biomechanics.evaluate_zanshin(pose_history, best_peak_f, min(total_len - 1, best_peak_f + 15))

        # 3. Avaliação pelo Motor de Calibração (Training Model)
        calib_eval = self.calibrator.evaluate_strike(
            target_score=target_score,
            fumikomi_score=fumikomi_score,
            posture_score=posture_score,
            zanshin_score=zanshin_score
        )

        # Classificação da técnica neste ângulo
        detected_tech = self.event_spotter._classify_technique(pose_history, max(0, best_peak_f - 10), best_peak_f)

        # Critérios de confirmação baseados estritamente no modelo de treinamento:
        # 1. Movimentação real acima do limiar físico mínimo do perfil (descarta repouso/ruído)
        # 2. Validação biomecânica completa dos critérios de Yuko-Datotsu (Ki-Ken-Tai-Ichi)
        # 3. Compatibilidade técnica ou proximidade alta do alvo
        has_motion = (max_vel >= self.active_velocity_threshold)
        is_biomechanically_valid = calib_eval["is_valid"]
        tech_matches = (detected_tech == target_technique or target_score >= 0.50)

        is_confirmed = (has_motion and is_biomechanically_valid and tech_matches)
        confidence = calib_eval["total_score"] if has_motion else 0.0

        diag_report = DiagnosticReporter.generate_strike_report(
            event_info={
                "type": detected_tech or target_technique,
                "timestamp": self._frame_to_timestamp(best_peak_f, fps),
                "attacker_name": f"Kenshi (Câmera {camera_id})"
            },
            evaluation=calib_eval,
            fumikomi_offset_ms=fumikomi_offset_ms
        )

        notes = []
        if is_confirmed:
            notes.append(f"Técnica {detected_tech} validada ({confidence:.1f}%)")
        else:
            if not has_motion:
                notes.append("Sem movimentação detectada (velocidade abaixo do limiar)")
            if not is_biomechanically_valid:
                failed = ", ".join(calib_eval.get("failed_subcriteria", [])) or "Abaixo do score mínimo"
                notes.append(f"Reprovado na calibração ({failed})")
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
            fumikomi_sync=fumikomi_score * 100.0,
            fumikomi_offset_ms=fumikomi_offset_ms,
            zanshin_score=zanshin_score * 100.0,
            is_ippon=is_biomechanically_valid,
            failed_subcriteria=calib_eval.get("failed_subcriteria", []),
            diagnostic_report=diag_report,
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
        Para 1 câmera (monocular), a validação depende estritamente do modelo de calibração.
        Para 2+ câmeras, requer quórum de confirmação entre os ângulos de visão.
        """
        num_cameras = len(camera_histories)
        required_quorum = self.calculate_required_quorum(num_cameras, self._profile_name)

        evidences: List[CameraFrameEvidence] = []

        for idx in range(num_cameras):
            history = camera_histories[idx]
            label = camera_labels[idx] if idx < len(camera_labels) else f"Câmera {idx + 1}"
            snap = snapshot_frames[idx] if (snapshot_frames and idx < len(snapshot_frames)) else None

            # Avaliar cada câmera contra o modelo de calibração e limiar cinemático
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

        # Fusão de Scores das Câmeras
        if confirming_cams:
            joint_score = float(np.mean([ev.confidence_score for ev in confirming_cams]))
        elif evidences:
            joint_score = float(np.mean([ev.confidence_score for ev in evidences]))
        else:
            joint_score = 0.0

        # Verificação do Quórum de Validação
        is_confirmed = (num_confirming >= required_quorum and num_confirming > 0)

        # Análise consolidada de Yuko-Datotsu (Ki-Ken-Tai-Ichi)
        avg_target = float(np.mean([ev.target_proximity for ev in evidences])) if evidences else 0.0
        avg_fumikomi = float(np.mean([ev.fumikomi_sync for ev in evidences])) if evidences else 0.0
        avg_posture = float(np.mean([ev.posture_stability for ev in evidences])) if evidences else 0.0
        avg_zanshin = float(np.mean([ev.zanshin_score for ev in evidences])) if evidences else 0.0
        avg_offset = float(np.mean([ev.fumikomi_offset_ms for ev in evidences])) if evidences else 0.0

        joint_calib = self.calibrator.evaluate_strike(
            target_score=avg_target / 100.0,
            fumikomi_score=avg_fumikomi / 100.0,
            posture_score=avg_posture / 100.0,
            zanshin_score=avg_zanshin / 100.0
        )

        is_overall_ippon = is_confirmed and joint_calib["is_valid"]

        ref_ts = self._frame_to_timestamp(impact_frame, fps)

        diag_report_text = DiagnosticReporter.generate_strike_report(
            event_info={
                "type": technique,
                "timestamp": ref_ts,
                "attacker_name": attacker_name
            },
            evaluation=joint_calib,
            fumikomi_offset_ms=avg_offset
        )

        yuko_datotsu = {
            "is_valid": is_overall_ippon,
            "total_score": joint_calib["total_score"],
            "min_required": joint_calib["min_required"],
            "sub_scores": {
                "target_impact": round(avg_target, 1),
                "fumikomi_sync": round(avg_fumikomi, 1),
                "posture": round(avg_posture, 1),
                "zanshin": round(avg_zanshin, 1)
            },
            "fumikomi_offset_ms": round(avg_offset, 1),
            "failed_subcriteria": joint_calib.get("failed_subcriteria", []),
            "diagnostic_report": diag_report_text,
            "technique": DiagnosticReporter.format_strike_name(technique),
            "attacker_id": attacker_id,
            "attacker_name": attacker_name,
            "timestamp": ref_ts
        }

        # Definição do Status de Decisão
        if is_confirmed:
            if num_cameras == 1:
                decision_status = "CONFIRMED_MULTICAM"
                summary_text = (
                    f"🎯 GOLPE VALIDADO (CÂMERA ÚNICA): {technique} validado com confiança de {joint_score:.1f}% "
                    f"pelo modelo de treinamento ({self._profile_name.capitalize()})."
                )
            else:
                decision_status = "CONFIRMED_MULTICAM"
                summary_text = (
                    f"🎯 GOLPE VALIDADO POR CONSENSO MULTI-CÂMERAS ({num_confirming}/{num_cameras} câmeras confirmaram {technique}, "
                    f"Quórum Requerido: {required_quorum}/{num_cameras}, Confiança Conjunta: {joint_score:.1f}%)."
                )
        elif num_cameras == 1:
            decision_status = "REJECTED_NO_MOTION_OR_INVALID"
            first_ev = evidences[0] if evidences else None
            note_reason = first_ev.notes if first_ev and first_ev.notes else "Critérios não atingidos"
            summary_text = (
                f"⚠️ GOLPE REJEITADO (CÂMERA ÚNICA): {note_reason} pelo modelo de calibração ({self._profile_name.capitalize()})."
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
            summary_text=summary_text,
            yuko_datotsu_analysis=yuko_datotsu
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
