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
from src.vision.training_multi_tracker import TrainingMultiKenshiTracker
from src.analytics.event_spotter import EventSpotter, StrikeEvent
from src.analytics.sonkyo_detector import SonkyoDetector
from src.analytics.biomechanics import BiomechanicsAnalyzer
from src.analytics.multimodal_yuko_datotsu import MultimodalYukoDatotsuEngine
from src.analytics.camera_invariance import CameraInvarianceEngine
from src.analytics.multi_camera_fusion import MultiCameraFusionEngine, MultiCameraStrikeEvaluation
from src.analytics.training_analyzer import TrainingAnalyzer, modality_confidence_label
from src.engine.calibrator import CalibrationEngine
from src.engine.reporter import DiagnosticReporter
from src.utils.hardware import get_effective_device, ensure_browser_compatible_video, get_optimal_batch_size, HighQualityVideoWriter
from src.utils.settings_manager import get_vision_model, get_vision_model_info
from src.utils.logger_manager import log_event
from src.utils import feature_cache
from src.utils.test_isolation import is_test_mode


# Curadoria de todas as dúvidas (Pendência I12)
FILTER_DOUBT_MARGIN = 0.10          # probabilidade do filtro a até 10 p.p. do limiar
TECHNIQUE_DOUBT_MAX_PROB = 0.60     # técnica com certeza abaixo de 60%
AIUCHI_WINDOW_SEC = 1.0             # Aka e Shiro pontuando quase juntos
AFTER_POINT_WINDOW_SEC = 3.0        # ponto logo depois de outro ponto (antes do reinício)
MAX_EXTRA_DOUBTS_PER_VIDEO = 10


class AsyncVideoBatchReader:
    """
    Leitor assíncrono de vídeo com prefetching em thread dedicada de CPU.
    Desacopla a decodificação de frames de disco do pipeline de inferência da GPU,
    mantendo uma fila em memória RAM para que a GPU opere em saturação contínua.
    """
    def __init__(self, video_path: str, batch_size: int = 64, max_queue: int = 4, frame_step: int = 1):
        self.video_path = video_path
        self.batch_size = max(1, batch_size)
        self.frame_step = max(1, frame_step or 1)
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self.stopped = False
        self.error: Optional[Exception] = None

        # Validação antecipada de arquivo para evitar falso positivo de análise em vídeos corrompidos
        is_url = isinstance(video_path, str) and video_path.startswith(("http://", "https://", "rtsp://", "rtmp://"))
        if not is_url:
            if not os.path.exists(video_path):
                raise ValueError(f"Arquivo de vídeo não encontrado: '{video_path}'")
            if os.path.isfile(video_path) and os.path.getsize(video_path) == 0:
                raise ValueError(f"Arquivo de vídeo corrompido (tamanho 0 bytes): '{os.path.basename(video_path)}'")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap or not self.cap.isOpened():
            if self.cap:
                self.cap.release()
            raise ValueError(f"Não foi possível abrir o arquivo de vídeo (arquivo corrompido ou formato não suportado): '{os.path.basename(video_path)}'")

        raw_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if raw_frames <= 0:
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.cap.release()
                raise ValueError(f"O arquivo de vídeo '{os.path.basename(video_path)}' está corrompido ou não contém quadros legíveis (0 frames decodificados).")
            # Reset posição do stream para o início
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            raw_frames = 1

        raw_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.fps = max(1.0, raw_fps / self.frame_step)
        self.total_frames = max(1, raw_frames // self.frame_step)
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
                    if self.frame_step > 1:
                        for _ in range(self.frame_step - 1):
                            if not self.cap.grab():
                                break

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
        self.training_multi_tracker = TrainingMultiKenshiTracker()
        self.sonkyo_detector = SonkyoDetector()
        self.event_spotter = EventSpotter()
        # Etapa 2 do detector (Pendência I3): filtro aprendido, ativo só se promovido pelo Golden Benchmark
        self.strike_filter = None
        # Classificador de técnica (Pendência I9): ativo só se promovido pelo Golden Benchmark
        self.technique_classifier = None
        # Classificador luta × 14 modalidades (Pendência I14): ativo só se superar o chute e o modelo atual
        self.modality_classifier = None
        if not is_test_mode():
            try:
                from src.analytics.modality_classifier import ModalityClassifier
                self.modality_classifier = ModalityClassifier.load_active()
            except Exception as e:
                log_event("WARNING", f"Classificador de modalidade indisponível: {e}", "pipeline")
            try:
                from src.analytics.strike_classifier import StrikeFilter
                self.strike_filter = StrikeFilter.load_active()
            except Exception as e:
                log_event("WARNING", f"Filtro de golpes indisponível: {e}", "pipeline")
            try:
                from src.analytics.technique_classifier import TechniqueClassifier
                self.technique_classifier = TechniqueClassifier.load_active()
            except Exception as e:
                log_event("WARNING", f"Classificador de técnica indisponível: {e}", "pipeline")
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

    def extract_match_features(
        self,
        video_path: str,
        frame_step: int = 1,
        invert_combatants: bool = False,
        shiaijo_polygon: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Extrai (ou reaproveita) somente as features de pose e rastreamento de um vídeo de luta,
        gravando-as no cache de features. Usado para pré-processar lotes de vídeos de treino.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")
        features = self._extract_match_features(
            video_path,
            invert_combatants=invert_combatants,
            shiaijo_polygon=shiaijo_polygon,
            frame_step=frame_step,
            use_feature_cache=True
        )
        if features is None:
            raise RuntimeError("Extração de features interrompida.")
        return features

    def _extract_match_features(
        self,
        video_path: str,
        progress_callback: Optional[Callable[..., None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        invert_combatants: bool = False,
        shiaijo_polygon: Optional[List[Tuple[float, float]]] = None,
        frame_step: int = 1,
        use_feature_cache: Optional[bool] = None,
        start_time: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Passos 1-2 do modo luta: inferência de pose em lote e associação exclusiva Aka/Shiro.
        É a etapa mais cara da análise; com o cache de features ligado, cada vídeo é processado
        pela rede neural uma única vez e as reanálises seguintes leem as poses do disco.
        Retorna None se cancelado. As histórias retornadas ainda não foram interpoladas.
        """
        if start_time is None:
            start_time = time.time()
        if use_feature_cache is None:
            use_feature_cache = feature_cache.is_feature_cache_enabled_by_default()

        cache_key = None
        if use_feature_cache:
            try:
                cache_key = feature_cache.build_cache_key(
                    video_path,
                    vision_model=self.vision_model,
                    frame_step=frame_step,
                    invert_combatants=invert_combatants,
                    shiaijo_polygon=shiaijo_polygon
                )
                cached = feature_cache.load_match_features(cache_key)
            except Exception as e:
                log_event("WARNING", f"Cache de features indisponível: {e}", "pipeline")
                cache_key, cached = None, None

            if cached is not None:
                # Rastreador novo para não carregar estado de análises anteriores
                self.combatant_tracker = CombatantTracker(
                    invert_assignment=invert_combatants,
                    shiaijo_polygon=shiaijo_polygon
                )
                log_event("INFO", f"Features de pose carregadas do cache para '{os.path.basename(video_path)}' ({cached['total_frames']} frames).", "pipeline")
                if progress_callback:
                    msg = "Poses carregadas do cache de features (vídeo já processado anteriormente)"
                    try:
                        progress_callback(0.60, msg)
                    except TypeError:
                        progress_callback(0.60)
                cached["from_cache"] = True
                cached["cache_key"] = cache_key
                return cached

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
        with AsyncVideoBatchReader(video_path, batch_size=batch_size, max_queue=4, frame_step=frame_step) as reader:
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
                        pct = int((frame_idx / total_frames) * 100)
                        elapsed = max(0.001, time.time() - start_time)
                        fps_cur = frame_idx / elapsed
                        eta_sec = max(0, (total_frames - frame_idx) / max(0.1, fps_cur))
                        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if eta_sec >= 60 else f"{int(eta_sec)}s"
                        msg = f"Rastreando poses na GPU ({pct}% • {frame_idx:,}/{total_frames:,} quadros • {fps_cur:.1f} FPS • Restante: ~{eta_str})"
                        try:
                            progress_callback(min(0.60, (frame_idx / total_frames) * 0.60), msg)
                        except TypeError:
                            progress_callback(min(0.60, (frame_idx / total_frames) * 0.60))
            finally:
                if self.effective_device == "gpu":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

        # Validação pós-leitura: abortar imediatamente se nenhum frame foi decodificado
        if frame_idx == 0:
            raise ValueError(
                f"O arquivo de vídeo '{os.path.basename(video_path)}' está corrompido ou não possui quadros legíveis "
                f"(0 frames decodificados). A análise foi interrompida para proteger a integridade dos dados."
            )

        features = {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "frames_decoded": frame_idx,
            "aka_history": aka_history,
            "shiro_history": shiro_history,
            "discarded_per_frame": discarded_per_frame,
            "tracker_summary": self.combatant_tracker.get_summary(),
            "video_name": os.path.basename(video_path),
            "vision_model": self.vision_model,
            "frame_step": frame_step
        }
        if cache_key:
            feature_cache.save_match_features(cache_key, features)
        features["from_cache"] = False
        features["cache_key"] = cache_key
        return features

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        progress_callback: Optional[Callable[..., None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        initial_sonkyo_override: Optional[Dict[str, Any]] = None,
        final_sonkyo_override: Optional[Dict[str, Any]] = None,
        invert_combatants: bool = False,
        training_modality_override: Optional[str] = None,
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        shiaijo_polygon: Optional[List[Tuple[float, float]]] = None,
        analysis_mode: str = "match",
        training_segments_override: Optional[List[Dict[str, Any]]] = None,
        frame_step: int = 1,
        use_feature_cache: Optional[bool] = None,
        record_learning_side_effects: bool = True,
        use_strike_filter: Optional[bool] = None
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

        use_feature_cache: reaproveita poses já extraídas deste vídeo (None = padrão; ligado fora de testes).
        record_learning_side_effects: False desliga a gravação de baselines do Kenshi e da fila de
        curadoria ativa, para reanálises repetidas (ex.: ajuste de limiares) não contaminarem os dados.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")

        # BIFURCAÇÃO ESTRITA: Modo Treinamento & Aprendizado usa rastreador dedicado
        if analysis_mode == "training":
            return self._process_training_video(
                video_path=video_path,
                output_video_path=output_video_path,
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
                training_modality_override=training_modality_override,
                custom_kendoka_names=custom_kendoka_names,
                training_segments_override=training_segments_override,
                frame_step=frame_step,
                use_feature_cache=use_feature_cache
            )

        start_time = time.time()

        # Passos 1-2: Extração de poses e associação Aka/Shiro (reaproveitada do cache de features quando disponível)
        features = self._extract_match_features(
            video_path,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
            invert_combatants=invert_combatants,
            shiaijo_polygon=shiaijo_polygon,
            frame_step=frame_step,
            use_feature_cache=use_feature_cache,
            start_time=start_time
        )
        if features is None:
            return None

        fps = features["fps"]
        total_frames = features["total_frames"]
        width = features["width"]
        height = features["height"]
        discarded_per_frame = features["discarded_per_frame"]
        feature_cache_hit = features["from_cache"]

        # Interpolação suave para quaisquer gaps residuais de detecção (ex: oclusões mútuas de até 10 frames)
        aka_history = self.combatant_tracker.interpolate_missing_poses(features["aka_history"], max_gap=10)
        shiro_history = self.combatant_tracker.interpolate_missing_poses(features["shiro_history"], max_gap=10)

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

        # 4b. Etapa 2 do detector (I3): o filtro aprendido descarta candidatos que não parecem golpe que vira ponto
        filter_probs: Dict[int, float] = {}
        filter_doubts: List[Tuple[Any, float]] = []
        technique_doubts: List[Tuple[Any, Dict[str, Any]]] = []
        active_filter = self.strike_filter if use_strike_filter is not False else None
        candidates_before_filter = len(all_raw_strikes)
        if active_filter is not None:
            kept = []
            for ev in all_raw_strikes:
                atk_h = aka_history if ev.attacker_id == "KENSHI_AKA" else shiro_history
                def_h = shiro_history if ev.attacker_id == "KENSHI_AKA" else aka_history
                prob = active_filter.probability(atk_h, def_h, ev.impact_frame, fps)
                if getattr(active_filter, "threshold", None) is not None and abs(prob - active_filter.threshold) <= FILTER_DOUBT_MARGIN:
                    filter_doubts.append((ev, prob))
                if active_filter.keep(prob):
                    filter_probs[id(ev)] = prob
                    kept.append(ev)
            log_event("INFO", f"Filtro de golpes: {len(kept)} de {candidates_before_filter} candidatos mantidos.", "pipeline")
            all_raw_strikes = kept

        # 4c. Técnica (I9): o classificador aprendido substitui a regra fixa de altura da mão
        technique_info: Dict[int, Dict[str, Any]] = {}
        if self.technique_classifier is not None:
            for ev in all_raw_strikes:
                atk_h = aka_history if ev.attacker_id == "KENSHI_AKA" else shiro_history
                def_h = shiro_history if ev.attacker_id == "KENSHI_AKA" else aka_history
                pred = self.technique_classifier.predict(atk_h, def_h, ev.impact_frame, fps)
                technique_info[id(ev)] = dict(pred, regra_fixa=ev.type)
                if pred["probabilidade"] < TECHNIQUE_DOUBT_MAX_PROB:
                    technique_doubts.append((ev, pred))
                ev.type = pred["tecnica"]

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
            if id(ev) in filter_probs:
                ev_dict["strike_filter_probability"] = round(filter_probs[id(ev)], 4)
            if id(ev) in technique_info:
                ev_dict["technique_source"] = "modelo"
                ev_dict["technique_probabilities"] = technique_info[id(ev)]["probabilidades"]
                ev_dict["technique_rule_based"] = technique_info[id(ev)]["regra_fixa"]
            else:
                ev_dict["technique_source"] = "regra_fixa"

            # Eixo 6: Modelagem do Estilo Individual do Kenshi
            kin_eval = {}
            if self.kinesthetic_manager and record_learning_side_effects:
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

            # Curadoria Ativa: Se o lance recai na faixa de incerteza arbitral (45%-65%), extrair corte real da luta
            conf_val = float(evaluation.get("total_score", 0.0)) / 100.0 if float(evaluation.get("total_score", 0.0)) > 1.0 else float(evaluation.get("total_score", 0.0))
            if record_learning_side_effects and 0.45 <= conf_val <= 0.65 and video_path and os.path.isfile(video_path):
                try:
                    from src.utils.strike_clip_manager import extract_strike_subclip, get_clips_dir
                    from src.engine.active_learning import UncertaintySampler

                    target_time = float(ev_dict.get("impact_frame", 0)) / float(fps) if fps > 0 else 0.0
                    clip_id = f"unc_strike_{int(time.time() * 1000)}_{len(analyzed_events)}"
                    os.makedirs(get_clips_dir(), exist_ok=True)
                    clip_out = os.path.join(get_clips_dir(), f"{clip_id}_corte.mp4")
                    real_clip = extract_strike_subclip(video_path, clip_out, target_time_sec=target_time)

                    sampler = UncertaintySampler()
                    strike_pkg = {
                        "strike_type": ev_dict.get("type") or ev_dict.get("strike_type", "Men"),
                        "is_valid": bool(evaluation.get("is_valid")),
                        "kenshi": "AKA" if ev_dict.get("attacker_id") == "KENSHI_AKA" else "SHIRO",
                        "fps": fps,
                        "scores": evaluation.get("sub_scores", {}),
                        "sub_scores": evaluation.get("sub_scores", {}),
                        "total_score": evaluation.get("total_score", conf_val * 100.0),
                        "timestamp_seconds": target_time,
                        "timestamp": ev_dict.get("timestamp", f"{target_time:.1f}s"),
                        "impact_frame": ev_dict.get("impact_frame", 0)
                    }
                    sampler.evaluate_and_enqueue(
                        strike_data=strike_pkg,
                        confidence=conf_val,
                        video_source=video_path,
                        profile_name=getattr(self.calibrator, "profile_name", "normal"),
                        clip_path=real_clip
                    )
                except Exception as e:
                    log_event("WARNING", f"Falha ao extrair corte real de golpe incerto no pipeline: {e}", "pipeline")

        # 5b. Curadoria de todas as dúvidas (I12): filtro de golpes, técnica e regras (ai-uchi, golpe após o yame)
        if record_learning_side_effects and video_path and os.path.isfile(video_path):
            try:
                self._enqueue_extra_doubts(video_path, fps, filter_doubts, technique_doubts, analyzed_events,
                                           getattr(active_filter, "threshold", None))
            except Exception as e:
                log_event("WARNING", f"Falha ao enfileirar dúvidas para curadoria: {e}", "pipeline")

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
            if width < 1280:
                render_w = 1280
                render_h = round(height * (1280.0 / max(1, width)))
                render_h += render_h % 2
            else:
                render_w = width + (width % 2)
                render_h = height + (height % 2)
            writer = HighQualityVideoWriter(output_video_path, fps, (render_w, render_h))

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

                    if raw_f.shape[1] != render_w or raw_f.shape[0] != render_h:
                        raw_f = cv2.resize(raw_f, (render_w, render_h), interpolation=cv2.INTER_LINEAR)

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
        tracker_summary = features["tracker_summary"]

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
            has_sonkyo=has_sonkyo_in_video,
            session_segments_override=training_segments_override
        )
        modality_check = self._check_modality([primary_history, secondary_history], fps, total_frames, chosen_mode="luta")

        summary_log = (
            f"RESUMO DO PROCESSAMENTO DE VÍDEO CONCLUÍDO:\n"
            f"  • Arquivo de Vídeo: {os.path.basename(video_path)} ({video_path})\n"
            f"  • Tempo Total de Processamento: {total_elapsed_sec:.2f}s ({total_frames} frames a {processing_fps:.1f} FPS)\n"
            f"  • Cache de Features: {'Reaproveitado (poses lidas do disco)' if feature_cache_hit else 'Extração completa'}\n"
            f"  • Duração do Vídeo: {round(total_frames / fps, 2)}s (Tempo Efetivo de Combate: {sonkyo_analysis['effective_combat_duration_seconds']}s)\n"
            f"  • Dispositivo / Acelerador: {self.effective_device.upper()} ({self.device_status_message})\n"
            f"  • Perfil de Avaliação Aplicado: {self.calibrator.active_config.get('name', 'Custom')}\n"
            f"  • Modalidade de Treinamento: {training_analysis.modality_name} ({modality_confidence_label(training_analysis.detection_method, training_analysis.detection_confidence)})\n"
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
            "feature_cache_hit": feature_cache_hit,
            "strike_filter": {
                "ativo": active_filter is not None,
                "modelo": os.path.basename(active_filter.path) if active_filter is not None and active_filter.path else None,
                "candidatos": candidates_before_filter,
                "mantidos": len(all_raw_strikes),
            },
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
            "modality_check": modality_check,
            "camera_invariance_analysis": {
                "detected_angle_deg": round(float(np.mean([ev["event_info"].get("camera_angle_deg", 65.0) for ev in analyzed_events])), 1) if analyzed_events else 65.0,
                "camera_category": analyzed_events[0]["event_info"].get("camera_category", "LATERAL") if analyzed_events else "LATERAL",
                "quality_diagnostic": analyzed_events[0]["spatial_invariance"].get("quality_diagnostic", {}) if analyzed_events and "spatial_invariance" in analyzed_events[0] else self.camera_invariance.diagnostic.diagnose_camera_angle(65.0)
            }
        }

    def _enqueue_extra_doubts(self, video_path: str, fps: float, filter_doubts, technique_doubts,
                              analyzed_events: List[Dict[str, Any]], filter_threshold: Optional[float]) -> int:
        """Enfileira na curadoria as dúvidas do filtro, da técnica e de regra, das mais informativas primeiro."""
        from src.engine.active_learning import UncertaintySampler
        from src.utils.strike_clip_manager import extract_strike_subclip, DEFAULT_CLIPS_DIR

        def _kenshi(attacker_id):
            return "AKA" if attacker_id == "KENSHI_AKA" else "SHIRO"

        jobs = []
        for ev, prob in filter_doubts:
            unc = 1.0 - abs(prob - filter_threshold) / FILTER_DOUBT_MARGIN
            jobs.append(("golpe_filtro", ev.impact_frame, ev.type, _kenshi(ev.attacker_id), unc, 0.8,
                         f"Filtro de golpes em dúvida ({prob:.0%}; limiar {filter_threshold:.0%}): é um golpe que vira ponto?", {}, prob))
        for ev, pred in technique_doubts:
            jobs.append(("tecnica", ev.impact_frame, pred["tecnica"], _kenshi(ev.attacker_id), 1.0 - pred["probabilidade"], 0.6,
                         f"Técnica incerta: o modelo diz {pred['tecnica']} com {pred['probabilidade']:.0%}.", pred["probabilidades"],
                         pred["probabilidade"]))
        valid = sorted((e["event_info"] for e in analyzed_events if e["evaluation"].get("is_valid")), key=lambda i: i["impact_frame"])
        for a, b in zip(valid, valid[1:]):
            dt = (b["impact_frame"] - a["impact_frame"]) / max(1.0, float(fps))
            if a["attacker_id"] != b["attacker_id"] and dt <= AIUCHI_WINDOW_SEC:
                jobs.append(("regra", a["impact_frame"], a.get("type", "MEN"), None, 0.9, 1.0,
                             f"Possível ai-uchi: Aka e Shiro pontuaram com {dt:.1f} s de diferença. Só um, nenhum ou os dois valem?", {}, None))
            elif dt <= AFTER_POINT_WINDOW_SEC:
                jobs.append(("regra", b["impact_frame"], b.get("type", "MEN"), _kenshi(b["attacker_id"]), 0.9, 1.0,
                             f"Ponto {dt:.1f} s depois de outro ponto: possível golpe após o yame (não deveria valer).", {}, None))

        jobs.sort(key=lambda j: j[4] * j[5], reverse=True)
        sampler = UncertaintySampler()
        profile = getattr(self.calibrator, "profile_name", "normal")
        added = 0
        for doubt_type, frame, strike_type, kenshi, unc, impact, reason, options, model_conf in jobs[:MAX_EXTRA_DOUBTS_PER_VIDEO]:
            t = float(frame) / max(1.0, float(fps))
            clip = None
            try:
                clip_out = os.path.join(DEFAULT_CLIPS_DIR, f"unc_{doubt_type}_{int(time.time() * 1000)}_{frame}_corte.mp4")
                clip = extract_strike_subclip(video_path, clip_out, target_time_sec=t)
            except Exception:
                pass
            entry = sampler.enqueue_doubt(
                doubt_type=doubt_type, video_source=video_path, uncertainty=unc, impact=impact, reason=reason,
                strike_data={"strike_type": strike_type, "timestamp_seconds": t, "impact_frame": frame, "fps": fps, "kenshi": kenshi},
                profile_name=profile, clip_path=clip, options=options, kenshi=kenshi, confidence=model_conf,
            )
            added += int(entry is not None)
        return added

    def _process_training_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        progress_callback: Optional[Callable[..., None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
        training_modality_override: Optional[str] = None,
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        training_segments_override: Optional[List[Dict[str, Any]]] = None,
        frame_step: int = 1,
        use_feature_cache: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executa a análise dedicada do Modo de Treinamento e Aprendizado:
        1. Rastreamento simultâneo de N praticantes com TrainingMultiKenshiTracker.
        2. Re-identificação contínua (Re-ID) de praticantes que saem e retornam ao enquadramento
           via Zekken (frente do Tare), Tenugui (dorsal sob o Men), Tare lateral e Biometria.
        3. Detecção e atribuição de golpes sem barreiras de Sonkyō.
        4. Identificação de múltiplas sessões de treinamento dentro da mesma sequência
           e concentração em blocos por modalidade.
        """
        start_time = time.time()

        # Cache do Modo Treinamento (I13): pose + re-identificação já feitas para este vídeo
        if use_feature_cache is None:
            use_feature_cache = feature_cache.is_feature_cache_enabled_by_default()
        cache_key = None
        cached = None
        if use_feature_cache and os.path.isfile(video_path):
            try:
                cache_key = feature_cache.build_cache_key(video_path, self.vision_model, frame_step=frame_step, mode="training")
                cached = feature_cache.load_training_features(cache_key)
            except Exception as e:
                log_event("WARNING", f"Cache do Modo Treinamento indisponível: {e}", "pipeline")
        if cached is not None:
            fps, total_frames = cached["fps"], cached["total_frames"]
            width, height = cached.get("width", 0), cached.get("height", 0)
            from types import SimpleNamespace
            tracks_view = {k: SimpleNamespace(**v) for k, v in cached["tracks"].items()}
            summary_tracker = cached["summary"]
            total_reidentifications = cached["total_reidentifications"]
            self.training_multi_tracker = SimpleNamespace(tracks=tracks_view, total_reidentifications=total_reidentifications)
            log_event("INFO", f"Modo Treinamento: rastreamento reaproveitado do cache ({os.path.basename(video_path)}).", "pipeline")
            if progress_callback:
                try:
                    progress_callback(0.65, "Rastreamento reaproveitado do cache de poses")
                except TypeError:
                    progress_callback(0.65)
        else:
            tracks_view, summary_tracker, total_reidentifications, fps, total_frames, width, height = self._track_training_video(
                video_path, frame_step, progress_callback, is_cancelled, start_time)
            if tracks_view is None:
                return None
            if cache_key:
                feature_cache.save_match_features(cache_key, {
                    "tracks": {k: {"history": t.history, "display_name": t.display_name, "reentry_count": t.reentry_count}
                               for k, t in tracks_view.items()},
                    "summary": summary_tracker, "total_reidentifications": total_reidentifications,
                    "fps": fps, "total_frames": total_frames, "width": width, "height": height,
                    "video_name": os.path.basename(video_path), "vision_model": self.vision_model,
                    "frame_step": frame_step, "mode": "training",
                })
        feature_cache_hit = cached is not None
        return self._finish_training_analysis(
            video_path, output_video_path, progress_callback, is_cancelled, training_modality_override,
            custom_kendoka_names, training_segments_override, start_time, tracks_view, summary_tracker,
            total_reidentifications, fps, total_frames, feature_cache_hit)

    def _track_training_video(self, video_path, frame_step, progress_callback, is_cancelled, start_time):
        """Etapa cara do Modo Treinamento: pose em lote + rastreamento/re-identificação de N praticantes."""
        tracker = TrainingMultiKenshiTracker(
            max_lost_frames=90,
            spatial_dist_threshold=0.44,
            reid_similarity_threshold=0.46,
            min_hits=3
        )
        self.training_multi_tracker = tracker

        batch_size = self.batch_size if (self.effective_device == "gpu" and self.pose_detector.use_gpu) else 1
        frame_idx = 0

        with AsyncVideoBatchReader(video_path, batch_size=batch_size, max_queue=4, frame_step=frame_step) as reader:
            fps = reader.fps
            total_frames = reader.total_frames
            width = reader.width
            height = reader.height

            try:
                while True:
                    if is_cancelled and is_cancelled():
                        log_event("WARNING", f"Análise de treino cancelada pelo usuário no frame {frame_idx}/{total_frames}.", "pipeline")
                        reader.stop()
                        return (None,) * 7

                    batch_frames = reader.read_batch()
                    if not batch_frames:
                        break

                    batch_candidates = self.pose_detector.process_frame_candidates_batch(batch_frames)

                    for frame_in_batch, candidates in zip(batch_frames, batch_candidates):
                        tracker.process_frame(candidates, frame=frame_in_batch)
                        frame_idx += 1

                    if progress_callback:
                        pct = int((frame_idx / total_frames) * 100)
                        elapsed = max(0.001, time.time() - start_time)
                        fps_cur = frame_idx / elapsed
                        eta_sec = max(0, (total_frames - frame_idx) / max(0.1, fps_cur))
                        eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s" if eta_sec >= 60 else f"{int(eta_sec)}s"
                        msg = f"Rastreando poses na GPU ({pct}% • {frame_idx:,}/{total_frames:,} quadros • {fps_cur:.1f} FPS • Restante: ~{eta_str})"
                        try:
                            progress_callback(min(0.65, (frame_idx / total_frames) * 0.65), msg)
                        except TypeError:
                            progress_callback(min(0.65, (frame_idx / total_frames) * 0.65))
            finally:
                if self.effective_device == "gpu":
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

        # Validação pós-leitura: abortar imediatamente se nenhum frame foi decodificado
        if frame_idx == 0:
            raise ValueError(
                f"O arquivo de vídeo '{os.path.basename(video_path)}' está corrompido ou não possui quadros legíveis "
                f"(0 frames decodificados). A análise de treinamento foi interrompida para proteger a integridade dos dados."
            )

        tracker.finish_session(total_frames)

        if is_cancelled and is_cancelled():
            return (None,) * 7
        return (dict(tracker.tracks), tracker.get_session_summary(), tracker.total_reidentifications,
                fps, total_frames, width, height)

    def _check_modality(self, histories, fps, total_frames, chosen_mode: str) -> Optional[Dict[str, Any]]:
        """
        Classificador luta × 14 modalidades (I14). Sem modelo ativo devolve None (nada é inventado).
        Com modelo: trechos previstos, classe dominante com a probabilidade medida e um aviso quando o
        modo escolhido (luta/treino) não bate com o que o modelo vê.
        """
        if self.modality_classifier is None:
            return None
        try:
            from src.analytics.modality_classifier import LUTA
            pred = self.modality_classifier.predict_segments([h for h in histories if h], int(total_frames), float(fps))
        except Exception as e:
            log_event("WARNING", f"Classificador de modalidade falhou: {e}", "pipeline")
            return None
        dom, prob = pred.get("dominante"), pred.get("probabilidade")
        aviso = None
        if dom and chosen_mode == "luta" and dom != LUTA:
            aviso = f"O classificador vê treino ({dom}, {prob:.0%}) neste vídeo analisado como luta. Confira o modo de análise."
        elif dom == LUTA and chosen_mode == "treino":
            aviso = (f"O classificador vê luta ou shiai-geiko ({prob:.0%}) neste vídeo analisado como treino. "
                     "Confira o modo de análise.")
        return {"predicao": pred, "classe_prevista": dom, "probabilidade": prob, "modo_escolhido": chosen_mode,
                "aviso": aviso, "modelo": self.modality_classifier.path}

    def _finish_training_analysis(self, video_path, output_video_path, progress_callback, is_cancelled,
                                  training_modality_override, custom_kendoka_names, training_segments_override,
                                  start_time, tracks, summary_tracker, total_reidentifications, fps, total_frames,
                                  feature_cache_hit=False):
        """Golpes, sessões, blocos de modalidade e vídeo anotado a partir do rastreamento (do cache ou recém-feito)."""
        # Detecção de golpes individualizada para cada Kenshi rastreado
        strikes_by_kenshi: Dict[str, List[Any]] = {}
        all_training_events: List[Dict[str, Any]] = []

        kenshi_histories = {k_id: t.history for k_id, t in tracks.items()}

        for k_id, t in tracks.items():
            valid_poses = [p for p in t.history if p]
            if len(valid_poses) < 10:
                continue

            k_name = (custom_kendoka_names or {}).get(k_id, t.display_name)
            k_strikes = self.event_spotter.detect_strikes(
                t.history,
                fps=fps,
                attacker_id=k_id,
                attacker_name=k_name,
                filter_out_of_bounds=False
            )
            strikes_by_kenshi[k_id] = k_strikes

            for st in k_strikes:
                ev_dict = st.to_dict()
                ev_dict["contact_detected"] = True
                all_training_events.append({
                    "event_info": ev_dict,
                    "evaluation": {"is_valid": True, "total_score": 85.0},
                    "diagnostic_report": f"🥋 Golpe {st.type} executado por {k_name} ({st.timestamp_impact})."
                })

        all_training_events.sort(key=lambda e: e.get("event_info", {}).get("impact_frame", 0))
        for idx_ev, ev_entry in enumerate(all_training_events, start=1):
            ev_entry["event_index"] = idx_ev
            if isinstance(ev_entry.get("event_info"), dict):
                ev_entry["event_info"]["event_index"] = idx_ev

        # Classificador luta × 14 modalidades (I14): trechos e probabilidade medida, e aviso se parecer luta
        modality_check = self._check_modality(list(kenshi_histories.values()), fps, total_frames, chosen_mode="treino")

        # Analisar sessões e blocos de modalidade antes da renderização para anotar o vídeo com a sessão corrente
        training_session_result = self.training_analyzer.analyze_multi_session(
            kenshi_histories=kenshi_histories,
            detected_strikes_by_kenshi=strikes_by_kenshi,
            modality_override=training_modality_override,
            fps=fps,
            custom_kendoka_names=custom_kendoka_names,
            kenshis_metadata=summary_tracker.get("kenshis", []),
            session_segments_override=training_segments_override,
            model_prediction=modality_check.get("predicao") if modality_check else None
        )
        detected_sub_sessions = training_session_result.sub_sessions or []

        # Renderização do vídeo anotado com N Kenshis se solicitado
        if output_video_path and total_frames > 0:
            ok_render = self.render_training_annotated_video(
                video_path=video_path,
                output_video_path=output_video_path,
                tracks=tracks,
                custom_kendoka_names=custom_kendoka_names,
                sub_sessions=detected_sub_sessions,
                default_modality_name=training_session_result.modality_name,
                total_reidentifications=total_reidentifications,
                progress_callback=(lambda p: progress_callback(0.65 + p * 0.35)) if progress_callback else None,
                is_cancelled=is_cancelled
            )
            if not ok_render and is_cancelled and is_cancelled():
                return None

        total_elapsed_sec = round(time.time() - start_time, 2)
        processing_fps = round(total_frames / max(0.001, total_elapsed_sec), 1)

        summary_log = (
            f"RESUMO DE TREINAMENTO MULTI-KENSHI CONCLUÍDO:\n"
            f"  • Kenshis Detectados: {len(tracks)}\n"
            f"  • Re-identificações de Retorno ao Quadro: {total_reidentifications}\n"
            f"  • Cache de Poses: {'reaproveitado' if feature_cache_hit else 'extraído agora'}\n"
            f"  • Modalidade Principal: {training_session_result.modality_name}\n"
            f"  • Sessões na Sequência: {len(training_session_result.sub_sessions)} | Blocos de Modalidade: {len(training_session_result.modality_blocks)}\n"
            f"  • Duração: {round(total_frames / fps, 2)}s ({total_elapsed_sec:.2f}s a {processing_fps} FPS)"
        )
        log_event("INFO", summary_log, "pipeline")

        if progress_callback:
            progress_callback(1.0)

        try:
            import pickle
            cache_payload = {
                "video_path": os.path.abspath(video_path),
                "total_reidentifications": total_reidentifications,
                "tracks": {
                    k_id: {
                        "history": t.history,
                        "display_name": t.display_name,
                        "reentry_count": t.reentry_count
                    }
                    for k_id, t in tracks.items()
                }
            }
            cache_file = os.path.abspath("annotated_tracks_cache.pkl")
            with open(cache_file, "wb") as f_cache:
                pickle.dump(cache_payload, f_cache)
        except Exception:
            pass

        return {
            "video_path": video_path,
            "total_frames": total_frames,
            "duration_seconds": round(total_frames / fps, 2),
            "processing_time_seconds": total_elapsed_sec,
            "processing_fps": processing_fps,
            "effective_combat_duration_seconds": round(total_frames / fps, 2),
            "events_detected_count": len(all_training_events),
            "profile_applied": "Treinamento & Aprendizado Multi-Kenshi",
            "device_used": self.effective_device,
            "device_status": self.device_status_message,
            "sonkyo_analysis": {"has_initial_sonkyo": False, "has_final_sonkyo": False},
            "plane_filtering": {"mode": "training_multi_kenshi"},
            "scoreboard": {
                "aka_score": 0,
                "shiro_score": 0,
                "winner": "TRAINING",
                "winner_name": "Sessão de Treino",
                "result_description": f"Treinamento com {len(tracks)} Kenshi(s)"
            },
            "feature_cache_hit": feature_cache_hit,
            "events": all_training_events,
            "training_analysis": training_session_result.to_dict(),
            "modality_check": modality_check,
            "multi_kenshi_tracker_summary": summary_tracker
        }

    @staticmethod
    def render_training_annotated_video(
        video_path: str,
        output_video_path: str,
        tracks: Dict[str, Any],
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        merged_id_map: Optional[Dict[str, str]] = None,
        sub_sessions: Optional[List[Any]] = None,
        default_modality_name: str = "Treinamento",
        total_reidentifications: int = 0,
        progress_callback: Optional[Callable[..., None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None
    ) -> bool:
        """
        Renderiza (ou re-renderiza rapidamente) o vídeo anotado de treinamento com os nomes
        personalizados, IDs mesclados e sessões atualizadas sem precisar repetir a detecção de pose.
        """
        if not video_path or not os.path.exists(video_path) or not output_video_path or not tracks:
            return False

        from src.vision.pose_detector import CombatantVisualizer
        from src.analytics.training_analyzer import get_kendoka_id_aliases

        cap_render = cv2.VideoCapture(video_path)
        if not cap_render.isOpened():
            return False

        fps = float(cap_render.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0
        total_frames = max(1, int(cap_render.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        width = int(cap_render.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap_render.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        if width < 1280:
            render_w = 1280
            render_h = round(height * (1280.0 / max(1, width)))
            render_h += render_h % 2
        else:
            render_w = width + (width % 2)
            render_h = height + (height % 2)

        writer = HighQualityVideoWriter(output_video_path, fps, (render_w, render_h))

        palette = [
            (246, 130, 59),  # Azul Brilhante
            (59, 214, 115),  # Verde Esmeralda
            (37, 99, 235),   # Laranja Vibrante
            (217, 70, 239),  # Magenta / Roxo
            (14, 165, 233),  # Ouro / Âmbar
            (235, 180, 30),  # Ciano Turquesa
            (153, 72, 236),  # Rosa Coral
            (128, 222, 74)   # Verde Lima
        ]

        names_map = custom_kendoka_names or {}
        m_map = merged_id_map or {}
        track_keys = list(tracks.keys())
        color_idx_by_id = {tk: i for i, tk in enumerate(track_keys)}
        unique_active_ids = len({m_map.get(tk, tk) for tk in track_keys})

        def _resolve_track_label_and_color(raw_k_id: str, track_obj: Any, idx_t: int) -> Tuple[str, Tuple[int, int, int]]:
            eff_id = m_map.get(raw_k_id, raw_k_id)
            col_idx = color_idx_by_id.get(eff_id, idx_t)
            col = palette[col_idx % len(palette)]

            def_disp = getattr(track_obj, "display_name", None) or (
                track_obj.get("display_name") if isinstance(track_obj, dict) else f"Kenshi {eff_id.replace('KENSHI_', '')}"
            )
            c_name = None
            for al in get_kendoka_id_aliases(eff_id):
                val = names_map.get(al)
                if val and val.strip():
                    c_name = val.strip()
                    break
            if not c_name:
                c_name = str(def_disp)

            reentries = getattr(track_obj, "reentry_count", 0) if not isinstance(track_obj, dict) else int(track_obj.get("reentry_count", 0))
            label_txt = f"{c_name}"
            if reentries > 0 or eff_id != raw_k_id:
                label_txt += f" [Re-ID x{max(1, reentries)}]"
            return label_txt, col

        try:
            f_idx = 0
            while cap_render.isOpened():
                if is_cancelled and is_cancelled():
                    return False
                ret, raw_f = cap_render.read()
                if not ret or raw_f is None:
                    break

                if raw_f.shape[1] != render_w or raw_f.shape[0] != render_h:
                    ann_f = cv2.resize(raw_f, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
                else:
                    ann_f = raw_f.copy()

                occupied_label_rects: List[Tuple[int, int, int, int]] = []
                active_eff_ids_in_frame = set()
                for idx_track, (k_id, track) in enumerate(tracks.items()):
                    hist = getattr(track, "history", None) if not isinstance(track, dict) else track.get("history", [])
                    if hist and f_idx < len(hist):
                        lm = hist[f_idx]
                        if lm is not None:
                            active_eff_ids_in_frame.add(m_map.get(k_id, k_id))
                            label_txt, col = _resolve_track_label_and_color(k_id, track, idx_track)
                            CombatantVisualizer.draw_skeleton(
                                ann_f,
                                lm,
                                color=col,
                                label=label_txt,
                                occupied_label_rects=occupied_label_rects
                            )

                curr_ss_label = default_modality_name.split("(")[0].strip()
                for ss_item in (sub_sessions or []):
                    ss_sf = ss_item.get("start_frame", 0) if isinstance(ss_item, dict) else getattr(ss_item, "start_frame", 0)
                    ss_ef = ss_item.get("end_frame", total_frames) if isinstance(ss_item, dict) else getattr(ss_item, "end_frame", total_frames)
                    if ss_sf <= f_idx <= ss_ef:
                        raw_mname = ss_item.get("modality_name", "") if isinstance(ss_item, dict) else getattr(ss_item, "modality_name", "")
                        ss_mod_short = str(raw_mname).split("(")[0].strip()
                        ss_idx = ss_item.get("segment_index", 1) if isinstance(ss_item, dict) else getattr(ss_item, "segment_index", 1)
                        ss_ts_s = ss_item.get("start_timestamp", "00:00") if isinstance(ss_item, dict) else getattr(ss_item, "start_timestamp", "00:00")
                        ss_ts_e = ss_item.get("end_timestamp", "00:00") if isinstance(ss_item, dict) else getattr(ss_item, "end_timestamp", "00:00")
                        curr_ss_label = f"Sessão #{ss_idx} ({ss_ts_s}-{ss_ts_e}): {ss_mod_short}"
                        break

                active_now = len(active_eff_ids_in_frame)
                txt_banner = f"SENPAI TREINO  |  {curr_ss_label}  |  Kenshis: {active_now}/{max(1, unique_active_ids)}  |  Re-ID: {total_reidentifications}"
                CombatantVisualizer.draw_top_hud_banner(ann_f, txt_banner)

                writer.write(ann_f)
                f_idx += 1

                if progress_callback and f_idx % 12 == 0:
                    progress_callback(min(0.99, f_idx / total_frames))
        finally:
            cap_render.release()
            writer.release()

        try:
            ensure_browser_compatible_video(output_video_path)
        except Exception as e_transcode:
            log_event("WARNING", f"Transcodificação para H.264 não pôde ser completada: {e_transcode}", "pipeline")

        if progress_callback:
            progress_callback(1.0)
        return True

    @staticmethod
    def load_cached_training_tracks(video_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Carrega do disco o cache de poses rastreadas (annotated_tracks_cache.pkl)
        caso o objeto em memória da sessão tenha sido reciclado pelo Streamlit.
        """
        cache_file = os.path.abspath("annotated_tracks_cache.pkl")
        if not os.path.exists(cache_file):
            return None
        try:
            import pickle
            with open(cache_file, "rb") as f_cache:
                data = pickle.load(f_cache)
            if not isinstance(data, dict) or not data.get("tracks"):
                return None
            if video_path and data.get("video_path"):
                if os.path.abspath(video_path) != os.path.abspath(str(data["video_path"])):
                    return None
            return data
        except Exception:
            return None

    def process_training_video(self, *args, **kwargs):
        """Atalho de conveniência para processamento dedicado em modo treinamento."""
        kwargs["analysis_mode"] = "training"
        return self.process_video(*args, **kwargs)


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
        shiaijo_polygon: Optional[List[Tuple[float, float]]] = None,
        analysis_mode: str = "match",
        training_segments_override: Optional[List[Dict[str, Any]]] = None,
        frame_step: int = 1
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
        self.analysis_mode = analysis_mode
        self.training_segments_override = training_segments_override
        self.frame_step = max(1, frame_step or 1)
        
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
            def on_progress(p: float, msg: Optional[str] = None):
                self.progress = min(1.0, max(0.0, p))
                if msg:
                    self.status_message = msg
                else:
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
                shiaijo_polygon=self.shiaijo_polygon,
                analysis_mode=self.analysis_mode,
                training_segments_override=self.training_segments_override,
                frame_step=self.frame_step
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
