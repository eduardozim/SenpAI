"""
Módulo de Captura e Gerenciamento Assíncrono de Streams de Vídeo (RTSP, RTMP, HTTP e Webcams).
Fornece streaming de alta performance com zero latência de buffer, transporte TCP forçado para RTSP,
descarte automático de frames defasados, reconexão resiliente e diagnóstico prévio de conexões.
"""

import os
import time
import threading
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from src.utils.logger_manager import log_event


# Configuração padrão de flags FFmpeg para RTSP e Streams de Rede no OpenCV
FFMPEG_RTSP_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;500000|"
    "analyzeduration;1000000|"
    "probesize;1000000"
)


def apply_ffmpeg_network_optimizations() -> None:
    """
    Aplica as variáveis de ambiente e opções do FFmpeg para garantir que
    conexões RTSP/IP usem TCP e operem com a menor latência possível sem drop de pacotes UDP.
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_RTSP_OPTIONS


def normalize_stream_source(source_input: Union[str, int]) -> Union[str, int]:
    """
    Normaliza e valida a fonte de vídeo fornecida pelo usuário.
    Converte strings numéricas em inteiros (para webcams locais) e limpa URLs RTSP/HTTP.

    Args:
        source_input: Índice de câmera (int ou str) ou URL de stream (RTSP, RTMP, HTTP, HTTPS).

    Returns:
        int para índices de webcam ou str sanitizada para URLs.
    """
    if isinstance(source_input, int):
        return source_input

    if not isinstance(source_input, str):
        return 0

    cleaned = source_input.strip()

    # Se for string numérica (ex: "0", "1", "2"), converter para int
    if cleaned.isdigit():
        return int(cleaned)

    # Remover aspas externas se existirem
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    return cleaned


class ThreadedVideoStream:
    """
    Leitor de vídeo em thread dedicada para Webcams locais e streams de rede (RTSP, RTMP, HTTP/MJPEG).
    
    Principais Vantagens:
    1. Buffer Size = 1: Nunca acumula frames defasados, eliminando lag em transmissões ao vivo.
    2. Zero Blocking: O método .read() retorna instantaneamente o último frame decodificado.
    3. Resiliência: Reconexão automática em caso de oscilações de rede sem travar a interface.
    4. Métricas em Tempo Real: Acompanhamento de FPS real, resolução, latência e status.
    """

    def __init__(
        self,
        src: Union[str, int],
        name: str = "Stream",
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 1.5,
        auto_start: bool = True
    ):
        self.raw_src = src
        self.src = normalize_stream_source(src)
        self.name = name
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        # Identificação de protocolo
        self.is_network = isinstance(self.src, str) and any(
            self.src.lower().startswith(p) for p in ["rtsp://", "rtmp://", "http://", "https://"]
        )
        self.is_rtsp = isinstance(self.src, str) and self.src.lower().startswith("rtsp://")
        self.is_file = isinstance(self.src, str) and not self.is_network

        # Estado e Controle
        self.cap: Optional[cv2.VideoCapture] = None
        self.status: str = "INITIALIZING"  # INITIALIZING, CONNECTED, RECONNECTING, DISCONNECTED, ERROR, STOPPED
        self.error_message: str = ""
        
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_time: float = 0.0
        self.frame_count: int = 0
        self.dropped_frames: int = 0
        self.fps: float = 0.0
        self.resolution: Tuple[int, int] = (0, 0)
        self.latency_ms: float = 0.0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if auto_start:
            self.start()

    def _open_capture(self) -> bool:
        """Inicializa e configura a conexão do VideoCapture."""
        try:
            if self.is_network:
                apply_ffmpeg_network_optimizations()
                # Para URLs de rede no OpenCV, CAP_FFMPEG oferece suporte direto e otimizado
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            elif isinstance(self.src, int):
                backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.src, backend)
            else:
                self.cap = cv2.VideoCapture(self.src)

            if not self.cap or not self.cap.isOpened():
                self.status = "ERROR"
                self.error_message = f"Não foi possível abrir a fonte: {self.src}"
                return False

            # Limitar tamanho de buffer para evitar atraso (latência) acumulada
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Capturar primeiro frame para validação e resolução (com retries para aguardar I-frame/keyframe inicial)
            ret, frame = False, None
            max_initial_reads = 5 if self.is_network else 2
            for _ in range(max_initial_reads):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    break
                time.sleep(0.1)

            if ret and frame is not None:
                h, w = frame.shape[:2]
                self.resolution = (w, h)
                stream_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.fps = stream_fps if (0 < stream_fps <= 120) else 30.0
                with self._lock:
                    self.last_frame = frame
                    self.last_frame_time = time.time()
                    self.frame_count += 1
                self.status = "CONNECTED"
                self.error_message = ""
                return True
            else:
                self.status = "ERROR"
                self.error_message = "Conexão aberta, mas nenhum frame de vídeo foi recebido."
                return False

        except Exception as ex:
            self.status = "ERROR"
            self.error_message = f"Exceção ao inicializar captura: {str(ex)}"
            log_event("ERROR", f"[{self.name}] Falha na captura de vídeo: {ex}", "stream_capture")
            return False

    def wait_until_connected(self, timeout_seconds: float = 4.0) -> bool:
        """
        Aguarda de forma não-bloqueante ativa até que o stream esteja conectado e com o primeiro frame decodificado.
        """
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if self.is_connected():
                return True
            if self.status in ["DISCONNECTED", "ERROR"] and (self._thread is None or not self._thread.is_alive()):
                return False
            time.sleep(0.05)
        return self.is_connected()


    def start(self) -> "ThreadedVideoStream":
        """Inicia a thread de captura contínua."""
        if self._thread is not None and self._thread.is_alive():
            return self

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name=f"ThreadedStream-{self.name}", daemon=True)
        self._thread.start()
        return self

    def _worker_loop(self) -> None:
        """Loop contínuo em segundo plano para consumir frames e manter o stream atualizado."""
        # Tentativa inicial de conexão
        opened = self._open_capture()
        reconnect_count = 0

        fps_calc_time = time.time()
        fps_frame_counter = 0

        while not self._stop_event.is_set():
            if not opened or self.cap is None or not self.cap.isOpened():
                if reconnect_count >= self.max_reconnect_attempts:
                    self.status = "DISCONNECTED"
                    self.error_message = f"Desconectado após {self.max_reconnect_attempts} tentativas de reconexão."
                    break

                self.status = "RECONNECTING"
                reconnect_count += 1
                log_event("WARNING", f"[{self.name}] Tentando reconectar ({reconnect_count}/{self.max_reconnect_attempts})...", "stream_capture")
                time.sleep(self.reconnect_delay)

                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                opened = self._open_capture()
                continue

            # Captura de frame
            t_before = time.time()
            ret, frame = self.cap.read()
            t_after = time.time()

            if ret and frame is not None:
                reconnect_count = 0  # Reseta contador de reconexões após sucesso
                with self._lock:
                    self.last_frame = frame
                    self.last_frame_time = t_after
                    self.frame_count += 1
                    self.status = "CONNECTED"
                    self.error_message = ""
                    self.latency_ms = (t_after - t_before) * 1000.0

                # Cálculo de FPS real efetivo
                fps_frame_counter += 1
                elapsed_fps = t_after - fps_calc_time
                if elapsed_fps >= 1.0:
                    self.fps = fps_frame_counter / elapsed_fps
                    fps_frame_counter = 0
                    fps_calc_time = t_after

            else:
                # Se for arquivo local e atingiu o final (EOF), reiniciar para simular stream contínuo
                if self.is_file and self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_loop, frame_loop = self.cap.read()
                    if ret_loop and frame_loop is not None:
                        with self._lock:
                            self.last_frame = frame_loop
                            self.last_frame_time = time.time()
                            self.frame_count += 1
                        continue

                self.dropped_frames += 1
                # Se falhar leituras consecutivas, acionar reconexão
                if self.dropped_frames % 10 == 0:
                    log_event("WARNING", f"[{self.name}] Falha na leitura de frame ({self.dropped_frames} frames perdidos).", "stream_capture")
                    if self.dropped_frames > 25:
                        opened = False

            # Cadência de tempo para não saturar CPU (em arquivo local respeita FPS nativo)
            if self.is_file and self.fps > 0:
                time.sleep(max(0.005, 1.0 / self.fps))
            else:
                time.sleep(0.002)


        # Finalização da thread
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if self.status != "DISCONNECTED":
            self.status = "STOPPED"

    def read(self, copy: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retorna o frame mais recente decodificado sem bloquear a thread chamadora.

        Args:
            copy: Se True, retorna uma cópia independente do frame para manipulação segura.

        Returns:
            Tuple (sucesso: bool, frame_bgr: Optional[np.ndarray])
        """
        with self._lock:
            if self.last_frame is None or self.status in ["DISCONNECTED", "ERROR"]:
                return False, None
            frame = self.last_frame.copy() if copy else self.last_frame
            return True, frame

    def read_rgb(self, copy: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retorna o frame mais recente convertido para o espaço de cor RGB (pronto para Streamlit).
        """
        ret, frame = self.read(copy=False)
        if not ret or frame is None:
            return False, None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return True, frame_rgb

    def is_connected(self) -> bool:
        """Verifica se o stream está conectado e recebendo frames ativos."""
        return self.status == "CONNECTED" and self.last_frame is not None

    def get_stats(self) -> Dict[str, Any]:
        """Retorna dicionário de métricas e status operacional do stream."""
        return {
            "name": self.name,
            "source": str(self.src),
            "status": self.status,
            "is_connected": self.is_connected(),
            "fps": round(self.fps, 1),
            "resolution": self.resolution,
            "frame_count": self.frame_count,
            "dropped_frames": self.dropped_frames,
            "latency_ms": round(self.latency_ms, 1),
            "error_message": self.error_message,
            "is_network": self.is_network,
            "is_rtsp": self.is_rtsp
        }

    def stop(self) -> None:
        """Para a thread de captura e libera todos os recursos de rede/vídeo."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.status = "STOPPED"
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def release(self) -> None:
        """Alias para compatibilidade com a API de cv2.VideoCapture."""
        self.stop()

    def __enter__(self) -> "ThreadedVideoStream":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def probe_stream_connection(
    source: Union[str, int],
    timeout_seconds: float = 3.5
) -> Dict[str, Any]:
    """
    Testa de forma não-bloqueante a conectividade com uma câmera local ou stream de rede (RTSP/HTTP).
    Captura um frame de amostra e avalia latência e resolução.

    Args:
        source: Índice de webcam ou URL de stream RTSP/HTTP.
        timeout_seconds: Tempo limite máximo de resposta em segundos.

    Returns:
        Dict com status ('success'), mensagem explicativa, 'frame_rgb' (miniatura), 'fps' e 'resolution'.
    """
    norm_src = normalize_stream_source(source)
    start_t = time.time()

    stream = ThreadedVideoStream(
        src=norm_src,
        name="ConnectionTest",
        max_reconnect_attempts=1,
        auto_start=True
    )

    try:
        # Aguardar pelo primeiro frame válido até o timeout
        while time.time() - start_t < timeout_seconds:
            ret, frame_rgb = stream.read_rgb()
            if ret and frame_rgb is not None:
                elapsed_ms = (time.time() - start_t) * 1000.0
                stats = stream.get_stats()
                return {
                    "success": True,
                    "message": f"Conectado com sucesso! Resolução: {stats['resolution'][0]}x{stats['resolution'][1]} ({stats['fps']:.1f} FPS) — Latência de abertura: {elapsed_ms:.0f}ms",
                    "frame_rgb": frame_rgb,
                    "resolution": stats["resolution"],
                    "fps": stats["fps"],
                    "latency_ms": elapsed_ms,
                    "source": str(norm_src)
                }
            time.sleep(0.05)

        # Se atingiu o timeout sem frame
        stats = stream.get_stats()
        err = stats.get("error_message") or "Tempo limite esgotado sem receber sinal de vídeo."
        return {
            "success": False,
            "message": f"Falha na conexão: {err}",
            "frame_rgb": None,
            "resolution": (0, 0),
            "fps": 0.0,
            "latency_ms": (time.time() - start_t) * 1000.0,
            "source": str(norm_src)
        }

    finally:
        stream.stop()


# Alias retrocompatível
test_stream_connection = probe_stream_connection

