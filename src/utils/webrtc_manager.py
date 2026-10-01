"""
Módulo de Integração WebRTC para Captura de Câmeras do Navegador no SenpAI.
Permite que usuários acessem o SenpAI publicado na Web (Streamlit Cloud, servidores em nuvem)
e transmitam a webcam do computador/celular local via protocolo WebRTC com baixa latência,
processando os modelos de IA e devolvendo o vídeo anotado diretamente no navegador.
"""

import time
import threading
import html
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING, cast

import cv2
import numpy as np


class _WebRtcModeFallback:
    SENDRECV: Any = "SENDRECV"
    RECVONLY: Any = "RECVONLY"
    SENDONLY: Any = "SENDONLY"


class _DummyWebRtcContext:
    video_processor: Any = None


def _dummy_webrtc_streamer(*args: Any, **kwargs: Any) -> Any:
    return _DummyWebRtcContext()


if TYPE_CHECKING:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        VideoProcessorBase
    )
    import av
    HAS_WEBRTC: bool = True
else:
    try:
        from streamlit_webrtc import (
            webrtc_streamer,
            WebRtcMode,
            RTCConfiguration,
            VideoProcessorBase
        )
        import av
        HAS_WEBRTC = True
    except (ImportError, ModuleNotFoundError):
        HAS_WEBRTC = False
        VideoProcessorBase = object
        RTCConfiguration = None
        WebRtcMode = _WebRtcModeFallback
        webrtc_streamer = _dummy_webrtc_streamer
        av = None

from src.engine.reporter import DiagnosticReporter
from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA


def _patch_aioice_for_python314() -> None:
    """
    Previne exceção não tratada no aioice quando rodando em Python 3.14+ em ambientes de nuvem.
    No Python 3.14, ao falhar uma transação UDP de STUN e fechar o transport, o _sock interno
    é anulado e a chamada Transaction.__retry() lança AttributeError: 'NoneType' object has no attribute 'sendto'.
    """
    try:
        import aioice.stun
        orig_retry = getattr(aioice.stun.Transaction, "_Transaction__retry", None)
        if orig_retry and not getattr(aioice.stun.Transaction, "_senpai_patched", False):
            def safe_retry(self: Any) -> Any:
                try:
                    return orig_retry(self)
                except (AttributeError, OSError):
                    pass
            aioice.stun.Transaction._Transaction__retry = safe_retry  # type: ignore
            aioice.stun.Transaction._senpai_patched = True  # type: ignore
    except Exception:
        pass


if HAS_WEBRTC:
    _patch_aioice_for_python314()


def get_rtc_configuration() -> Optional[Any]:
    """
    Retorna a configuração otimizada de servidores STUN e TURN para transposição rápida de NAT,
    firewalls e contêineres de nuvem (Streamlit Community Cloud).
    Prioriza STUNs ultra-rápidos (Google e Cloudflare) para resolução em milissegundos,
    com fallback para servidores TURN (UDP e TCP porta 443) caso o cliente ou o servidor
    estejam atrás de redes corporativas restritas ou proxies.
    Permite também configuração personalizada via st.secrets["RTC_CONFIGURATION"] ou st.secrets["webrtc"].
    """
    if not HAS_WEBRTC or RTCConfiguration is None:
        return None

    # Verifica se há configuração customizada no st.secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "RTC_CONFIGURATION" in st.secrets:
                cfg = st.secrets["RTC_CONFIGURATION"]
                if isinstance(cfg, dict):
                    return cast(Any, cfg)
            if "webrtc" in st.secrets and isinstance(st.secrets["webrtc"], dict):
                ice_servers = st.secrets["webrtc"].get("iceServers")
                if ice_servers:
                    return RTCConfiguration({"iceServers": ice_servers})
    except Exception:
        pass

    # Servidores STUN de altíssima velocidade e disponibilidade global
    fast_stun_servers = [
        "stun:stun.l.google.com:19302",
        "stun:stun1.l.google.com:19302",
        "stun:stun2.l.google.com:19302",
        "stun:stun.cloudflare.com:3478",
    ]

    # Servidores TURN de fallback (com suporte a UDP, TCP e TLS/443 para contêineres)
    turn_server = {
        "urls": [
            "turn:openrelay.metered.ca:80",
            "turn:openrelay.metered.ca:443",
            "turn:openrelay.metered.ca:443?transport=tcp",
            "turns:openrelay.metered.ca:443?transport=tcp",
        ],
        "username": "openrelayproject",
        "credential": "openrelayproject",
    }

    return RTCConfiguration(
        {
            "iceServers": [
                {"urls": fast_stun_servers},
                turn_server,
            ]
        }
    )



class SenpAIMatchWebRtcProcessor(VideoProcessorBase):  # type: ignore
    """
    Processador de vídeo WebRTC em tempo real para o Modo de Análise de Lutas (Shiai).
    Recebe frames do navegador do cliente, executa o rastreamento dos competidores (Aka/Shiro),
    avaliação de golpes Ki-Ken-Tai-Ichi e sobrepõe o HUD biomecânico do SenpAI.
    """

    def __init__(self, pipeline: Any, profile_name: str = "normal"):
        self.pipeline = pipeline
        self.profile_name = profile_name
        self.lock = threading.Lock()

        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 30.0

        # Placar e métricas da sessão
        self.score_shiro = 0
        self.score_aka = 0
        self.total_shiro_strikes = 0
        self.total_aka_strikes = 0
        self.live_modality_name = "Shiai (Combate)"

        # Histórico de eventos e alertas
        self.latest_strike_alert: Optional[Dict[str, Any]] = None
        self.strike_events: List[Dict[str, Any]] = []
        self.pose_history: List[Optional[Dict[str, Any]]] = []

    def recv(self, frame: Any) -> Any:
        if not HAS_WEBRTC or av is None:
            return frame

        img: np.ndarray = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        elapsed = max(0.001, time.time() - self.start_time)
        self.current_fps = self.frame_count / elapsed

        try:
            # 1. Rastreamento e estimativa de pose
            candidates, _ = self.pipeline.pose_detector.process_frame_candidates(img)
            aka_lm, shiro_lm, disc = self.pipeline.combatant_tracker.associate_and_filter(
                candidates,
                frame=img,
                return_persisted=True
            )
            drawn_frame = self.pipeline.pose_detector.draw_combatants_overlay(
                img,
                aka_landmarks=aka_lm,
                shiro_landmarks=shiro_lm,
                discarded_items=disc
            )

            active_lm = aka_lm or shiro_lm
            self.pose_history.append(active_lm)
            if len(self.pose_history) > 150:
                self.pose_history = self.pose_history[-120:]

            # 2. Identificação periódica da modalidade (a cada 45 quadros)
            if self.frame_count % 45 == 0 and len(self.pose_history) >= 20:
                try:
                    m_k, m_c, _ = self.pipeline.training_analyzer.detect_training_modality(
                        primary_history=self.pose_history[-90:],
                        secondary_history=[],
                        fps=self.current_fps or 30.0
                    )
                    with self.lock:
                        self.live_modality_name = f"{TRAINING_MODALITIES_METADATA.get(m_k, {}).get('name', m_k)} ({int(m_c * 100)}%)"
                except Exception:
                    pass

            # 3. Avaliação periódica de golpes Ki-Ken-Tai-Ichi (a cada 3 quadros)
            if self.frame_count % 3 == 0 and len(self.pose_history) >= 15:
                cam_cfg = [{"id": 1, "type": "webrtc", "source": 0, "label": "Navegador WebRTC"}]
                multicam_eval = self.pipeline.multicam_fusion.evaluate_live_step(
                    live_pose_histories=[self.pose_history],
                    camera_configs=cam_cfg,
                    current_fps=self.current_fps or 30.0,
                    current_frame_idx=self.frame_count,
                    latest_frames=[drawn_frame]
                )

                if multicam_eval:
                    yuko = multicam_eval.yuko_datotsu_analysis or {}
                    is_ippon = yuko.get("is_valid", False)
                    tot_sc = yuko.get("total_score", multicam_eval.joint_score)
                    tech_mark = DiagnosticReporter.format_strike_name(multicam_eval.technique)
                    sub = yuko.get("sub_scores", {})
                    atk_name = multicam_eval.attacker_name or "Kenshi"
                    atk_id_val = str(getattr(multicam_eval, "attacker_id", "") or yuko.get("attacker_id", "KENSHI_AKA")).upper()

                    with self.lock:
                        if "SHIRO" in atk_id_val or "BRANCO" in atk_name.upper():
                            self.total_shiro_strikes += 1
                            if is_ippon:
                                self.score_shiro += 1
                            competitor_label = "⚪ SHIRO (Branco)"
                            badge_bg = "rgba(255, 255, 255, 0.15)"
                            badge_color = "#FFFFFF"
                        else:
                            self.total_aka_strikes += 1
                            if is_ippon:
                                self.score_aka += 1
                            competitor_label = "🔴 AKA (Vermelho)"
                            badge_bg = "rgba(239, 68, 68, 0.2)"
                            badge_color = "#F87171"

                        alert_data = {
                            "timestamp": multicam_eval.timestamp_ref,
                            "is_ippon": is_ippon,
                            "tech": tech_mark,
                            "competitor": competitor_label,
                            "badge_bg": badge_bg,
                            "badge_color": badge_color,
                            "score": int(tot_sc * 100),
                            "sub_target": int(sub.get("target_impact", 0) * 100),
                            "sub_fumi": int(sub.get("fumikomi_sync", 0) * 100),
                            "sub_posture": int(sub.get("posture", 0) * 100),
                            "sub_zanshin": int(sub.get("zanshin", 0) * 100),
                            "status_text": "IPPON VÁLIDO!" if is_ippon else "GOLPE INCOMPLETO",
                            "status_color": "#4ADE80" if is_ippon else "#FBBF24"
                        }
                        self.latest_strike_alert = alert_data
                        self.strike_events.insert(0, alert_data)
                        if len(self.strike_events) > 50:
                            self.strike_events = self.strike_events[:50]

            return av.VideoFrame.from_ndarray(drawn_frame, format="bgr24")
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Retorna snapshot seguro para leitura pela interface Streamlit.
        """
        with self.lock:
            return {
                "frame_count": self.frame_count,
                "fps": self.current_fps,
                "score_shiro": self.score_shiro,
                "score_aka": self.score_aka,
                "total_shiro_strikes": self.total_shiro_strikes,
                "total_aka_strikes": self.total_aka_strikes,
                "live_modality_name": self.live_modality_name,
                "latest_strike_alert": self.latest_strike_alert,
                "strike_events": list(self.strike_events[:15])
            }


class SenpAITrainingWebRtcProcessor(VideoProcessorBase):  # type: ignore
    """
    Processador de vídeo WebRTC em tempo real para o Modo de Treinamento & Aprendizado.
    Recebe frames da webcam do navegador do usuário, rastreia os movimentos biomecânicos,
    calcula os 3 Pilares e envia feedbacks pedagógicos instantâneos de Kendo.
    """

    def __init__(self, pipeline: Any, live_train_mgr: Any):
        self.pipeline = pipeline
        self.live_train_mgr = live_train_mgr
        self.lock = threading.Lock()

        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 30.0
        self.pose_history: List[Optional[Dict[str, Any]]] = []

    def recv(self, frame: Any) -> Any:
        if not HAS_WEBRTC or av is None:
            return frame

        img: np.ndarray = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        elapsed = max(0.001, time.time() - self.start_time)
        self.current_fps = self.frame_count / elapsed

        try:
            candidates, _ = self.pipeline.pose_detector.process_frame_candidates(img)
            aka_lm, shiro_lm, disc = self.pipeline.combatant_tracker.associate_and_filter(
                candidates,
                frame=img,
                return_persisted=True
            )
            drawn_frame = self.pipeline.pose_detector.draw_combatants_overlay(
                img,
                aka_landmarks=aka_lm,
                shiro_landmarks=shiro_lm,
                discarded_items=disc
            )

            active_lm = aka_lm or shiro_lm
            self.pose_history.append(active_lm)
            if len(self.pose_history) > 150:
                self.pose_history = self.pose_history[-120:]

            # Processar passo de treinamento no motor de repetições
            with self.lock:
                self.live_train_mgr.process_live_frame(
                    live_pose_histories=[self.pose_history],
                    fps=self.current_fps or 30.0,
                    current_frame_idx=self.frame_count
                )

            return av.VideoFrame.from_ndarray(drawn_frame, format="bgr24")
        except Exception:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Retorna snapshot seguro para leitura pela interface Streamlit.
        """
        with self.lock:
            return {
                "frame_count": self.frame_count,
                "fps": self.current_fps,
                "reps_count": self.live_train_mgr.rep_count,
                "modality_key": self.live_train_mgr.current_modality_key,
                "movement_score": self.live_train_mgr.movement_score,
                "precision_score": self.live_train_mgr.precision_score,
                "constancy_score": self.live_train_mgr.constancy_score,
                "overall_score": self.live_train_mgr.overall_score,
                "latest_feedback": self.live_train_mgr.latest_feedback,
                "feedback_severity": self.live_train_mgr.feedback_severity,
                "rep_history": list(self.live_train_mgr.rep_history[:12]),
                "hud_html": self.live_train_mgr.render_live_hud_html()
            }
