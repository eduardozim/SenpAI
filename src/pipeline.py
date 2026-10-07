"""
Pipeline Principal de Processamento do SenpAI.
Orquestra Leitura de Vídeo -> Pose Tracking -> Action Spotting -> Avaliação Biomecânica -> Calibração -> Relatório.
"""

import cv2
import os
import time
import queue
import threading
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple

from src.vision.pose_detector import PoseDetector
from src.vision.shinai_tracker import ShinaiTracker
from src.vision.combatant_tracker import CombatantTracker
from src.analytics.event_spotter import EventSpotter, StrikeEvent
from src.analytics.sonkyo_detector import SonkyoDetector
from src.analytics.biomechanics import BiomechanicsAnalyzer
from src.analytics.multimodal_yuko_datotsu import MultimodalYukoDatotsuEngine
from src.analytics.camera_invariance import CameraInvarianceEngine
from src.analytics.multi_camera_fusion import MultiCameraFusionEngine, MultiCameraStrikeEvaluation
from src.analytics.training_analyzer import TrainingAnalyzer
from src.engine.calibrator import CalibrationEngine
from src.engine.reporter import DiagnosticReporter
from src.utils.hardware import get_effective_device, ensure_browser_compatible_video, get_optimal_batch_size
from src.utils.settings_manager import get_vision_model, get_vision_model_info
from src.utils.logger_manager import log_event


class AsyncVideoBatchReader:
    """
    Leitor assíncrono de vídeo com prefetching em thread dedicada de CPU.
    Desacopla a decodificação de frames de disco do pipeline de inferência da GPU,
    mantendo uma fila em memória RAM para que a GPU opere em saturação contínua.
    """
    def __init__(self, video_path: str, batch_size: int = 64, max_queue: int = 4):
        self.video_path = video_path
        self.batch_size = max(1, batch_size)
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self.stopped = False
        self.error: Optional[Exception] = None

        self.cap = cv2.VideoCapture(video_path)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        self.thread = threading.Thread(target=self._worker, daemon=True, name="AsyncVideoBatchReader")
        self.thread.start()

    def _worker(self):
        try:
            while not self.stopped and self.cap.isOpened():
                batch = []
                for _ in range(self.batch_size):
                    if self.stopped:
                        break
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        break
                    batch.append(frame)

                if not batch:
                    break

                # Enfileirar o lote de frames na fila com timeout
                while not self.stopped:
                    try:
                        self.queue.put(batch, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            self.error = e
        finally:
            try:
                self.queue.put(None, timeout=0.5)
            except Exception:
                pass
            if self.cap.isOpened():
                self.cap.release()

    def read_batch(self) -> Optional[List[np.ndarray]]:
        """Retorna o próximo lote de frames decodificados ou None ao atingir o fim do vídeo."""
        if self.stopped:
            return None
        while True:
            try:
                item = self.queue.get(timeout=0.05)
                return item
            except queue.Empty:
                if not self.thread.is_alive():
                    return None
                continue

    def stop(self):
        """Interrompe a thread de leitura e libera buffers."""
        self.stopped = True
        try:
            while not self.queue.empty():
                _ = self.queue.get_nowait()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class SenpAIPipeline:
    def __init__(
        self,
        calibration_profile: str = "normal",
        device_preference: str = "cpu",
        custom_batch_size: Optional[int] = None,
        vision_model: Optional[str] = None
    ):
        self.device_preference = device_preference
        self.vision_model = (vision_model or get_vision_model()).lower().strip()
        self.vision_model_info = get_vision_model_info(self.vision_model)
        self.effective_device, self.device_status_message, self.gpu_info = get_effective_device(device_preference)
        self.batch_size = get_optimal_batch_size(self.effective_device, custom_batch_size)
        
        self.pose_detector = PoseDetector(device=self.effective_device, model_name=self.vision_model)
        self.shinai_tracker = ShinaiTracker()
        self.combatant_tracker = CombatantTracker()
        self.sonkyo_detector = SonkyoDetector()
        self.event_spotter = EventSpotter()
        self.biomechanics = BiomechanicsAnalyzer()
        self.multimodal_engine = MultimodalYukoDatotsuEngine()
        self.camera_invariance = CameraInvarianceEngine()
        self.training_analyzer = TrainingAnalyzer()
        self.calibrator = CalibrationEngine(profile_name=calibration_profile)
        self.multicam_fusion = MultiCameraFusionEngine(profile_name=calibration_profile)
        try:
            from src.analytics.kenshi_style_model import KinestheticProfileManager
            self.kinesthetic_manager = KinestheticProfileManager()
        except Exception:
            self.kinesthetic_manager = None

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        initial_sonkyo_override: Optional[Dict[str, Any]] = None,
        final_sonkyo_override: Optional[Dict[str, Any]] = None,
        invert_combatants: bool = False,
        training_modality_override: Optional[str] = None,
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        shiaijo_polygon: Optional[List[Tuple[float, float]]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executa a análise completa de um arquivo de vídeo de luta/treino de Kendo:
        1. Rastreamento e associação exclusiva dos 2 Kenshi (Aka e Shiro) no Plano Principal com detecção da cor da flag dorsal (Tasukuki).
        2. Descarte automático de elementos fora do Shiai-jo, Segundo Plano (Background) e Oclusões na frente da câmera.
        3. Detecção e verificação dos momentos de Sonkyō (Abertura e Encerramento) ou aplicação de ajustes manuais com aprendizado contínuo.
        4. Delimitação estrita do início (match_start_frame) e fim (match_end_frame) da luta.
        5. Detecção e avaliação biomecânica exclusiva dos golpes dentro da janela de Sonkyō.
        6. Análise especializada de Treinamento & Aprendizado (14 modalidades oficiais com Kanji, 3 pilares, diagnósticos pedagógicos).
        7. Controle de pontuação (Placar Oficial de Ippon) e renderização de vídeo anotado.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

        start_time = time.time()

        # Resetar o rastreador para uma nova análise de vídeo com configuração de inversão e polígono do Shiai-jo
        self.combatant_tracker = CombatantTracker(
            invert_assignment=invert_combatants,
            shiaijo_polygon=shiaijo_polygon
        )

        # Coleta de histórico dos combatentes
        aka_history: List[Optional[Dict[str, Any]]] = []
        shiro_history: List[Optional[Dict[str, Any]]] = []
        discarded_per_frame: List[List[Dict[str, Any]]] = []

        batch_size = self.batch_size if (self.effective_device == "gpu" and self.pose_detector.use_gpu) else 1
        frame_idx = 0

        # Passo 1: Extração Paralela com Prefetching Assíncrono em Thread
        with AsyncVideoBatchReader(video_path, batch_size=batch_size, max_queue=4) as reader:
            fps = reader.fps
            total_frames = reader.total_frames
            width = reader.width
            height = reader.height

            try:
                while True:
                    if is_cancelled and is_cancelled():
                        elapsed_cancel = time.time() - start_time
                        log_event("WARNING", f"Processamento de vídeo cancelado pelo usuário no frame {frame_idx}/{total_frames} (Tempo decorrido: {elapsed_cancel:.2f}s).", "pipeline")
                        reader.stop()
                        return None

                    batch_frames = reader.read_batch()
                    if not batch_frames:
                        break

                    # 1. Extração de candidatos a praticantes (Inferência Paralela em Lote na GPU / Fallback CPU)
                    batch_candidates = self.pose_detector.process_frame_candidates_batch(batch_frames)

                    for frame_in_batch, candidates in zip(batch_frames, batch_candidates):
                        # Fallback sintético para modo demo (se for vídeo esquemático 2D)
                        if (not candidates or len(candidates) == 0) and "demo" in video_path.lower():
                            # Simular Sonkyō de abertura nos primeiros 25 frames, seguido de corte aos 48 frames
                            is_sonkyo_frame = (frame_idx < 25)
                            hand_y = 0.65 if is_sonkyo_frame else (0.50 if frame_idx < 35 else (0.25 if frame_idx < 48 else 0.60))
                            foot_x = 0.50 if frame_idx < 45 else 0.58
                            hip_y_val = 0.80 if is_sonkyo_frame else 0.65 # Quadril desce no Sonkyō

                            synthetic_lm = {
                                "RIGHT_WRIST": {"x": 0.52, "y": float(hand_y), "z": 0.0, "visibility": 0.9, "px": int(0.52*width), "py": int(hand_y*height)},
                                "LEFT_WRIST": {"x": 0.48, "y": float(hand_y + 0.02), "z": 0.0, "visibility": 0.9, "px": int(0.48*width), "py": int(hand_y*height)},
                                "RIGHT_ELBOW": {"x": 0.55, "y": float(hand_y + 0.12), "z": 0.0, "visibility": 0.9, "px": int(0.55*width), "py": int((hand_y+0.12)*height)},
                                "RIGHT_SHOULDER": {"x": 0.55, "y": 0.45 if is_sonkyo_frame else 0.40, "z": 0.0, "visibility": 0.9, "px": int(0.55*width), "py": int(0.40*height)},
                                "LEFT_SHOULDER": {"x": 0.45, "y": 0.45 if is_sonkyo_frame else 0.40, "z": 0.0, "visibility": 0.9, "px": int(0.45*width), "py": int(0.40*height)},
                                "RIGHT_HIP": {"x": 0.53, "y": float(hip_y_val), "z": 0.0, "visibility": 0.9, "px": int(0.53*width), "py": int(hip_y_val*height)},
                                "LEFT_HIP": {"x": 0.47, "y": float(hip_y_val), "z": 0.0, "visibility": 0.9, "px": int(0.47*width), "py": int(hip_y_val*height)},
                                "RIGHT_KNEE": {"x": 0.54, "y": float(hip_y_val + 0.08), "z": 0.0, "visibility": 0.9, "px": int(0.54*width), "py": int((hip_y_val+0.08)*height)},
                                "LEFT_KNEE": {"x": 0.46, "y": float(hip_y_val + 0.08), "z": 0.0, "visibility": 0.9, "px": int(0.46*width), "py": int((hip_y_val+0.08)*height)},
                                "NOSE": {"x": 0.50, "y": 0.35 if is_sonkyo_frame else 0.25, "z": 0.0, "visibility": 0.9, "px": int(0.50*width), "py": int(0.25*height)},
                                "RIGHT_EAR": {"x": 0.53, "y": 0.34 if is_sonkyo_frame else 0.24, "z": 0.0, "visibility": 0.9, "px": int(0.53*width), "py": int(0.24*height)},
                                "LEFT_EAR": {"x": 0.47, "y": 0.34 if is_sonkyo_frame else 0.24, "z": 0.0, "visibility": 0.9, "px": int(0.47*width), "py": int(0.24*height)},
                                "RIGHT_ANKLE": {"x": float(foot_x), "y": 0.90, "z": 0.0, "visibility": 0.9, "px": int(foot_x*width), "py": int(0.90*height)},
                                "LEFT_ANKLE": {"x": float(foot_x - 0.04), "y": 0.90, "z": 0.0, "visibility": 0.9, "px": int((foot_x-0.04)*width), "py": int(0.90*height)},
                                "RIGHT_FOOT_INDEX": {"x": float(foot_x), "y": 0.90, "z": 0.0, "visibility": 0.9, "px": int(foot_x*width), "py": int(0.90*height)}
                            }
                            candidates = [synthetic_lm]

                        # 2. Filtragem de Planos, Detecção de Flag (Tasukuki) e Associação dos 2 Combatentes
                        aka_lm, shiro_lm, discarded = self.combatant_tracker.associate_and_filter(
                            candidates,
                            frame=frame_in_batch,
                            return_persisted=True
                        )
                        aka_history.append(aka_lm)
                        shiro_history.append(shiro_lm)
                        discarded_per_frame.append(discarded)

                        frame_idx += 1

                    if progress_callback:
                        progress_callback(min(0.60, (frame_idx / total_frames) * 0.60)) # 60% para extração de poses
            finally:
                if self.effective_device == "gpu":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

        # Interpolação suave para quaisquer gaps residuais de detecção (ex: oclusões mútuas de até 10 frames)
        aka_history = self.combatant_tracker.interpolate_missing_poses(aka_history, max_gap=10)
        shiro_history = self.combatant_tracker.interpolate_missing_poses(shiro_history, max_gap=10)

        # Checagem de cancelamento antes de processamento dos eventos
        if is_cancelled and is_cancelled():
            elapsed_cancel = time.time() - start_time
            log_event("WARNING", f"Processamento de vídeo cancelado antes da análise de eventos de golpe (Tempo decorrido: {elapsed_cancel:.2f}s).", "pipeline")
            return None

        # 3. Detecção dos momentos de Sonkyō e Bounding da Luta
        # Se Aka tem mais detecções, usa Aka como referência principal; senão Shiro
        primary_history = aka_history if len([p for p in aka_history if p]) >= len([p for p in shiro_history if p]) else shiro_history
        secondary_history = shiro_history if primary_history is aka_history else aka_history

        sonkyo_analysis = self.sonkyo_detector.detect_match_boundaries(
            primary_history,
            fps=fps,
            secondary_pose_history=secondary_history,
            initial_sonkyo_override=initial_sonkyo_override,
            final_sonkyo_override=final_sonkyo_override
        )

        match_start_f = sonkyo_analysis["match_start_frame"]
        match_end_f = sonkyo_analysis["match_end_frame"]

        # 4. Detecção Temporal de Golpes estritamente entre Sonkyōs
        all_raw_strikes: List[StrikeEvent] = []

        # Golpes do Combatente Aka
        aka_strikes = self.event_spotter.detect_strikes(
            aka_history,
            fps=fps,
            start_bound_frame=match_start_f,
            end_bound_frame=match_end_f,
            attacker_id="KENSHI_AKA",
            attacker_name="Kenshi Aka (Vermelho)",
            filter_out_of_bounds=True
        )
        all_raw_strikes.extend(aka_strikes)

        # Golpes do Combatente Shiro (se presente)
        if len([p for p in shiro_history if p]) >= 15:
            shiro_strikes = self.event_spotter.detect_strikes(
                shiro_history,
                fps=fps,
                start_bound_frame=match_start_f,
                end_bound_frame=match_end_f,
                attacker_id="KENSHI_SHIRO",
                attacker_name="Kenshi Shiro (Branco)",
                filter_out_of_bounds=True
            )
            # Evitar duplicatas muito próximas entre Aka e Shiro (mínimo 16 frames ~0.53s)
            for s_ev in shiro_strikes:
                if not any(abs(s_ev.impact_frame - a_ev.impact_frame) < 16 for a_ev in all_raw_strikes):
                    all_raw_strikes.append(s_ev)

        # Ordenar eventos cronologicamente
        all_raw_strikes.sort(key=lambda ev: ev.impact_frame)

        # 5. Avaliação Biomecânica e Calibração dos Golpes Válidos
        analyzed_events = []
        for ev in all_raw_strikes:
            impact_f = ev.impact_frame
            history_used = aka_history if ev.attacker_id == "KENSHI_AKA" else shiro_history
            opponent_history = shiro_history if ev.attacker_id == "KENSHI_AKA" else aka_history
            landmarks_at_impact = history_used[impact_f] if impact_f < len(history_used) else None
            opponent_lm = opponent_history[impact_f] if impact_f < len(opponent_history) else None

            if not landmarks_at_impact:
                landmarks_at_impact = primary_history[impact_f] if impact_f < len(primary_history) else None

            # Análise de Invariância de Câmera, Normalização 3D e Diagnóstico de Perspectiva (Eixo 5)
            spatial_invariance = self.camera_invariance.process_frame_spatial_invariance(
                aka_landmarks=landmarks_at_impact,
                shiro_landmarks=opponent_lm
            )
            camera_angle_deg = spatial_invariance.get("angle_info", {}).get("estimated_angle_deg", 65.0)
            maai_3d = spatial_invariance.get("maai_3d")
            quality_diag = spatial_invariance.get("quality_diagnostic", {})

            # Métricas Ki-Ken-Tai-Ichi e Avaliação Multimodal Eixo 3
            # 1. Avaliação Multimodal de Yuko-Datotsu (Colisão, Hasuji, Seme, Contrataque, Áudio e Action Spotting)
            mm_eval = self.multimodal_engine.evaluate_complete_strike(
                strike_type=ev.type,
                attacker_history=history_used,
                defender_history=opponent_history,
                impact_frame=impact_f,
                video_path=video_path,
                fps=fps
            )

            # Impacto no alvo considerando colisão com Bogu do oponente
            target_score = mm_eval.get("collision_score", 0.60)
            hasuji_score = mm_eval.get("hasuji_score", 0.75)
            seme_score = mm_eval.get("seme_score", 0.70)
            is_ku_totsu = mm_eval.get("is_ku_totsu", False)
            counter_data = mm_eval.get("counterattack", {})
            action_spot_data = mm_eval.get("action_spotting", {})

            # Discriminação de Contato e Alcance Físico (Maai 2D e 3D):
            # Se o Maai 3D for excessivo (> 0.52), reforça a classificação de Ku-totsu
            if maai_3d is not None and maai_3d > 0.55:
                is_ku_totsu = True

            is_contact_range = not is_ku_totsu
            kenshi_dist = maai_3d if maai_3d is not None else (mm_eval.get("target_collision", {}).get("maai_distance") or 0.0)

            fumikomi_score, offset_ms = self.biomechanics.evaluate_fumikomi_sync(history_used, impact_f)
            # Postura com compensação geométrica de perspectiva (Eixo 5.1)
            posture_score = self.biomechanics.evaluate_posture(landmarks_at_impact, camera_angle_deg=camera_angle_deg)
            zanshin_score = self.biomechanics.evaluate_zanshin(history_used, impact_f, ev.end_frame)

            # Calibração com pesos especializados por tipo de golpe e 5° Pilar Hasuji (Eixo 1.4 & Eixo 3.2)
            evaluation = self.calibrator.evaluate_strike(
                target_score,
                fumikomi_score,
                posture_score,
                zanshin_score,
                strike_type=ev.type,
                hasuji_score=hasuji_score,
                seme_score=seme_score,
                is_ku_totsu=is_ku_totsu
            )

            # Se for falso disparo de Tsubazeriai ou guarda no vazio suprimida
            if action_spot_data.get("suppress_false_trigger", False):
                evaluation["is_valid"] = False
                evaluation["notes"] = f"Ação suprimida por Action Spotting ({action_spot_data.get('predicted_class')})."
            
            # Se foi constatada falta de alcance/contato, não pode ser Ippon válido
            if not is_contact_range:
                evaluation["is_valid_ippon"] = False
                evaluation["is_valid"] = False
                evaluation["contact_valid"] = False
                evaluation["notes"] = "Fora do Maai (Sem contato com oponente / Ku-totsu)"
            else:
                evaluation["contact_valid"] = True

            # Relatório textual com identificação do atacante
            ev_dict = ev.to_dict()
            ev_dict["contact_detected"] = is_contact_range
            ev_dict["maai_distance"] = round(kenshi_dist, 3) if kenshi_dist > 0 else None
            ev_dict["hasuji_score"] = round(hasuji_score * 100, 1)
            ev_dict["seme_score"] = round(seme_score * 100, 1)
            ev_dict["action_spotting_class"] = action_spot_data.get("predicted_class")
            ev_dict["is_counterattack"] = counter_data.get("is_counterattack", False)
            ev_dict["camera_angle_deg"] = round(camera_angle_deg, 1)
            ev_dict["maai_3d"] = round(maai_3d, 3) if maai_3d is not None else None
            ev_dict["camera_category"] = spatial_invariance.get("angle_info", {}).get("camera_category", "LATERAL")
            ev_dict["camera_quality_score"] = quality_diag.get("overall_quality_score", 85.0)

            # Eixo 6: Modelagem do Estilo Individual do Kenshi
            kin_eval = {}
            if self.kinesthetic_manager:
                try:
                    atk_disp = (custom_kendoka_names or {}).get(ev.attacker_id, ev.attacker_name)
                    # Estimativa de inclinação da coluna no impacto
                    spine_tilt_val = None
                    if opponent_history and len(opponent_history) > ev.impact_frame:
                        opp_p = opponent_history[ev.impact_frame]
                        if opp_p:
                            spine_tilt_val = opp_p.get("spine_tilt_deg")
                    kin_eval = self.kinesthetic_manager.record_strike_event(
                        kenshi_id=ev.attacker_id,
                        strike_type=ev.type,
                        spine_tilt_deg=spine_tilt_val,
                        fumikomi_offset_ms=offset_ms,
                        display_name=atk_disp
                    )
                except Exception:
                    pass

            ev_dict["kinesthetic_baseline_evaluation"] = kin_eval
            ev_dict["kinesthetic_insights"] = kin_eval.get("insights", [])
            
            report_text = DiagnosticReporter.generate_strike_report(ev_dict, evaluation, offset_ms)
            if not is_contact_range:
                report_text += " [Aviso: Golpe sem contato físico / Ku-totsu fora do Maai]"
            if ev_dict["is_counterattack"]:
                report_text += f" [{counter_data.get('counterattack_type')}]"

            analyzed_events.append({
                "event_info": ev_dict,
                "evaluation": evaluation,
                "multimodal_details": mm_eval,
                "spatial_invariance": spatial_invariance,
                "kinesthetic_baseline": kin_eval,
                "fumikomi_offset_ms": offset_ms,
                "diagnostic_report": report_text
            })

        # 6. Gravação do Vídeo Anotado com HUD de Sonkyō, Rastreamento dos Kenshis e Marcação de Golpes
        if output_video_path and total_frames > 0:
            # Mapear eventos de golpe por frame para anotação visual síncrona
            strike_events_by_frame: Dict[int, List[Dict[str, Any]]] = {}
            for ev in analyzed_events:
                ev_info = ev.get("event_info", {})
                impact_f = ev_info.get("impact_frame", 0)
                start_f = ev_info.get("start_frame", max(0, impact_f - 8))
                end_f = ev_info.get("end_frame", min(total_frames - 1, impact_f + 12))

                w_start = max(0, start_f - 4)
                w_end = min(total_frames - 1, end_f + 10)
                for f in range(w_start, w_end + 1):
                    if f not in strike_events_by_frame:
                        strike_events_by_frame[f] = []
                    strike_events_by_frame[f].append(ev)

            aka_custom = (custom_kendoka_names or {}).get("KENSHI_AKA")
            shiro_custom = (custom_kendoka_names or {}).get("KENSHI_SHIRO")

            cap_render = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            try:
                f_idx = 0
                while cap_render.isOpened():
                    if is_cancelled and is_cancelled():
                        elapsed_cancel = time.time() - start_time
                        log_event("WARNING", f"Renderização de vídeo anotado cancelada pelo usuário no frame {f_idx}/{total_frames} (Tempo decorrido: {elapsed_cancel:.2f}s).", "pipeline")
                        return None

                    ret, raw_f = cap_render.read()
                    if not ret or raw_f is None:
                        break

                    aka_p = aka_history[f_idx] if f_idx < len(aka_history) else None
                    shiro_p = shiro_history[f_idx] if f_idx < len(shiro_history) else None
                    disc_p = discarded_per_frame[f_idx] if f_idx < len(discarded_per_frame) else None
                    active_stk = strike_events_by_frame.get(f_idx, [])

                    # Determinar status visual de Sonkyō / Combate
                    if sonkyo_analysis["has_initial_sonkyo"] and f_idx < match_start_f:
                        hud_status = "🥋 SONKYŌ (INÍCIO DO COMBATE)"
                        timer_txt = "Aguardando Início"
                    elif sonkyo_analysis["has_final_sonkyo"] and f_idx >= match_end_f:
                        hud_status = "🥋 SONKYŌ (FIM DO COMBATE)"
                        timer_txt = "Combate Encerrado"
                    elif f_idx >= match_start_f and f_idx <= match_end_f:
                        elapsed_combat_sec = max(0.0, (f_idx - match_start_f) / fps)
                        hud_status = "⚔️ LUTA EM ANDAMENTO (IPPIN / YUKO-DATOTSU)"
                        timer_txt = f"{int(elapsed_combat_sec // 60):02d}:{int(elapsed_combat_sec % 60):02d}"
                    else:
                        hud_status = "FORA DA JANELA OFICIAL"
                        timer_txt = "--:--"

                    annotated_f = self.pose_detector.draw_combatants_overlay(
                        raw_f,
                        aka_landmarks=aka_p,
                        shiro_landmarks=shiro_p,
                        discarded_items=disc_p,
                        sonkyo_status=hud_status,
                        match_timer_str=timer_txt,
                        active_strikes=active_stk,
                        current_frame_idx=f_idx,
                        aka_label=aka_custom,
                        shiro_label=shiro_custom
                    )
                    writer.write(annotated_f)
                    f_idx += 1

                    if progress_callback and f_idx % 10 == 0:
                        progress_callback(0.60 + (f_idx / total_frames) * 0.40)
            finally:
                cap_render.release()
                writer.release()

            # Transcodifica para H.264/AVC1 YUV420p com +faststart garantindo renderização no HTML5 dos navegadores
            try:
                ensure_browser_compatible_video(output_video_path)
            except Exception as e_transcode:
                log_event("WARNING", f"Transcodificação para H.264 do navegador não pôde ser completada: {e_transcode}", "pipeline")

        if progress_callback:
            progress_callback(1.0)

        total_elapsed_sec = round(time.time() - start_time, 2)
        processing_fps = round(total_frames / max(0.001, total_elapsed_sec), 1)
        tracker_summary = self.combatant_tracker.get_summary()

        # 7. Controle de Pontuação Oficial dos Combatentes (Placar de Ippon)
        aka_valid_strikes = [ev for ev in analyzed_events if ev["event_info"]["attacker_id"] == "KENSHI_AKA" and ev["evaluation"].get("is_valid", False)]
        shiro_valid_strikes = [ev for ev in analyzed_events if ev["event_info"]["attacker_id"] == "KENSHI_SHIRO" and ev["evaluation"].get("is_valid", False)]
        
        aka_score = len(aka_valid_strikes)
        shiro_score = len(shiro_valid_strikes)

        if aka_score > shiro_score:
            winner = "AKA"
            winner_name = "Kenshi Aka (Vermelho)"
            result_description = f"Vitória de Aka ({aka_score} - {shiro_score})"
        elif shiro_score > aka_score:
            winner = "SHIRO"
            winner_name = "Kenshi Shiro (Branco)"
            result_description = f"Vitória de Shiro ({shiro_score} - {aka_score})"
        else:
            winner = "DRAW"
            winner_name = "Empate (Hikiwake)"
            result_description = f"Empate ({aka_score} - {shiro_score})"

        scoreboard = {
            "aka_score": aka_score,
            "shiro_score": shiro_score,
            "winner": winner,
            "winner_name": winner_name,
            "result_description": result_description,
            "aka_valid_strikes": [s["event_info"] for s in aka_valid_strikes],
            "shiro_valid_strikes": [s["event_info"] for s in shiro_valid_strikes],
            "flag_detection": {
                "flag_decision": tracker_summary.get("flag_decision", "POSITION_DEFAULT"),
                "confidence": tracker_summary.get("flag_confidence", 0.50),
                "candidate_left_red_score": tracker_summary.get("candidate_left_red_score", 0.0),
                "candidate_right_red_score": tracker_summary.get("candidate_right_red_score", 0.0),
                "invert_assignment": tracker_summary.get("invert_assignment", False)
            }
        }

        # Registro do Resumo do Processamento no Log do Sistema
        init_sonkyo_str = f"Detectado (Início: {sonkyo_analysis.get('match_start_timestamp', '00:00.000')})" if sonkyo_analysis["has_initial_sonkyo"] else "Não detectado"
        final_sonkyo_str = f"Detectado (Fim: {sonkyo_analysis.get('match_end_timestamp', f'{round(total_frames / fps, 2)}s')})" if sonkyo_analysis["has_final_sonkyo"] else "Não detectado"
        
        # 7. Análise de Treinamento & Aprendizado (14 Modalidades Oficiais com Kanji e 3 Pilares)
        has_sonkyo_in_video = bool(sonkyo_analysis.get("has_initial_sonkyo") or sonkyo_analysis.get("has_final_sonkyo"))
        training_analysis = self.training_analyzer.analyze_session(
            primary_history=primary_history,
            secondary_history=secondary_history,
            detected_strikes=all_raw_strikes,
            modality_override=training_modality_override,
            fps=fps,
            custom_kendoka_names=custom_kendoka_names,
            has_sonkyo=has_sonkyo_in_video
        )

        summary_log = (
            f"RESUMO DO PROCESSAMENTO DE VÍDEO CONCLUÍDO:\n"
            f"  • Arquivo de Vídeo: {os.path.basename(video_path)} ({video_path})\n"
            f"  • Tempo Total de Processamento: {total_elapsed_sec:.2f}s ({total_frames} frames a {processing_fps:.1f} FPS)\n"
            f"  • Duração do Vídeo: {round(total_frames / fps, 2)}s (Tempo Efetivo de Combate: {sonkyo_analysis['effective_combat_duration_seconds']}s)\n"
            f"  • Dispositivo / Acelerador: {self.effective_device.upper()} ({self.device_status_message})\n"
            f"  • Perfil de Avaliação Aplicado: {self.calibrator.active_config.get('name', 'Custom')}\n"
            f"  • Modalidade de Treinamento: {training_analysis.modality_name} (Confiança: {int(training_analysis.detection_confidence*100)}%)\n"
            f"  • Placar Oficial: Aka {aka_score} x {shiro_score} Shiro — {result_description}\n"
            f"  • Identificação de Flag (Tasukuki): {tracker_summary.get('flag_decision', 'N/A')} (Confiança: {int(tracker_summary.get('flag_confidence', 0.5)*100)}%)\n"
            f"  • Sonkyō Inicial: {init_sonkyo_str}\n"
            f"  • Sonkyō Final: {final_sonkyo_str}\n"
            f"  • Golpes Regulamentares Detectados: {len(analyzed_events)} evento(s)\n"
            f"  • Filtragem de Planos Descartados: Fundo={tracker_summary.get('discarded_background_count', 0)}, Frente={tracker_summary.get('discarded_foreground_count', 0)}"
        )
        log_event("INFO", summary_log, "pipeline")

        return {
            "video_path": video_path,
            "total_frames": total_frames,
            "duration_seconds": round(total_frames / fps, 2),
            "processing_time_seconds": total_elapsed_sec,
            "processing_fps": processing_fps,
            "effective_combat_duration_seconds": sonkyo_analysis["effective_combat_duration_seconds"],
            "events_detected_count": len(analyzed_events),
            "profile_applied": self.calibrator.active_config.get("name", "Custom"),
            "device_used": self.effective_device,
            "device_status": self.device_status_message,
            "sonkyo_analysis": sonkyo_analysis,
            "plane_filtering": tracker_summary,
            "scoreboard": scoreboard,
            "events": analyzed_events,
            "training_analysis": training_analysis.to_dict(),
            "camera_invariance_analysis": {
                "detected_angle_deg": round(float(np.mean([ev["event_info"].get("camera_angle_deg", 65.0) for ev in analyzed_events])), 1) if analyzed_events else 65.0,
                "camera_category": analyzed_events[0]["event_info"].get("camera_category", "LATERAL") if analyzed_events else "LATERAL",
                "quality_diagnostic": analyzed_events[0]["spatial_invariance"].get("quality_diagnostic", {}) if analyzed_events and "spatial_invariance" in analyzed_events[0] else self.camera_invariance.diagnostic.diagnose_camera_angle(65.0)
            }
        }


class AnalysisWorker:
    """
    Worker assíncrono para execução de processamento de vídeo em background thread,
    permitindo monitoramento em tempo real, cronômetro contínuo e cancelamento cooperativo instantâneo.
    """
    def __init__(
        self,
        pipeline: "SenpAIPipeline",
        video_path: str,
        output_video_path: Optional[str] = None,
        initial_sonkyo_override: Optional[Dict[str, Any]] = None,
        final_sonkyo_override: Optional[Dict[str, Any]] = None,
        invert_combatants: bool = False,
        training_modality_override: Optional[str] = None,
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        shiaijo_polygon: Optional[List[Tuple[float, float]]] = None
    ):
        self.pipeline = pipeline
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.initial_sonkyo_override = initial_sonkyo_override
        self.final_sonkyo_override = final_sonkyo_override
        self.invert_combatants = invert_combatants
        self.training_modality_override = training_modality_override
        self.custom_kendoka_names = custom_kendoka_names
        self.shiaijo_polygon = shiaijo_polygon
        
        self.progress: float = 0.0
        self.status_message: str = "Inicializando pipeline de visão e pose tracking..."
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.is_done: bool = False
        self.is_cancelled: bool = False
        
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        
        self._cancel_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def elapsed_seconds(self) -> float:
        if self.end_time is not None:
            return max(0.0, self.end_time - self.start_time)
        return max(0.0, time.time() - self.start_time)

    @property
    def elapsed_formatted(self) -> str:
        sec = self.elapsed_seconds
        mins = int(sec // 60)
        remaining_sec = sec % 60
        return f"{mins:02d}:{remaining_sec:04.1f}"

    def start(self):
        """Inicia a thread de processamento em background e reseta o cronômetro."""
        self.start_time = time.time()
        self.end_time = None
        self._thread.start()

    def cancel(self):
        """Sinaliza interrupção imediata ao pipeline e congela o cronômetro."""
        self._cancel_event.set()
        self.is_cancelled = True
        self.end_time = time.time()
        self.status_message = "Interrupção solicitada pelo usuário..."

    def _run(self):
        try:
            def on_progress(p: float):
                self.progress = min(1.0, max(0.0, p))
                self.status_message = f"Processando frames... {int(self.progress * 100)}%"

            def check_cancel() -> bool:
                return self._cancel_event.is_set()

            res = self.pipeline.process_video(
                video_path=self.video_path,
                output_video_path=self.output_video_path,
                progress_callback=on_progress,
                is_cancelled=check_cancel,
                initial_sonkyo_override=self.initial_sonkyo_override,
                final_sonkyo_override=self.final_sonkyo_override,
                invert_combatants=self.invert_combatants,
                training_modality_override=self.training_modality_override,
                custom_kendoka_names=self.custom_kendoka_names,
                shiaijo_polygon=self.shiaijo_polygon
            )

            self.end_time = time.time()

            if self._cancel_event.is_set() or res is None:
                self.is_cancelled = True
                self.result = None
                self.status_message = "Processamento cancelado."
            else:
                self.result = res
                self.progress = 1.0
                self.status_message = "Processamento concluído com sucesso!"
        except Exception as ex:
            self.end_time = time.time()
            self.error = str(ex)
            log_event("ERROR", f"Erro no worker de análise: {ex}", "pipeline")
        finally:
            if self.end_time is None:
                self.end_time = time.time()
            self.is_done = True


# Aliases de compatibilidade retroativa
SenpaiPipeline = SenpAIPipeline
ShinpanAIPipeline = SenpAIPipeline
ShinpanaiPipeline = SenpAIPipeline
