"""
Módulo de Captura e Gerenciamento Assíncrono de Streams de Vídeo (RTSP, RTMP, HTTP e Webcams).
Fornece streaming de alta performance com zero latência de buffer, transporte TCP forçado para RTSP,
descarte automático de frames defasados, reconexão resiliente e diagnóstico prévio de conexões.
"""

import os
import time
import threading
from collections import deque
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np
import cv2

from src.utils.logger_manager import log_event


# Configuração padrão de flags FFmpeg para RTSP e Streams de Rede de Ultra-Baixa Latência no OpenCV
FFMPEG_RTSP_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "max_delay;0|"
    "analyzeduration;100000|"
    "probesize;100000|"
    "reconnect;1|"
    "reconnect_streamed;1|"
    "reconnect_delay_max;2"
)


def set_c_runtime_env(key: str, value: str) -> None:
    """
    Configura a variável de ambiente tanto no Python (os.environ) quanto nas
    tabelas de ambiente das C runtimes (msvcrt.dll e ucrtbase.dll) no Windows.
    Essencial para que DLLs compiladas em C/C++ (como opencv_videoio_ffmpeg)
    obtenham os valores corretos via getenv().
    """
    os.environ[key] = value
    try:
        import ctypes
        if hasattr(ctypes, "cdll") and hasattr(ctypes.cdll, "msvcrt"):
            ctypes.cdll.msvcrt._putenv(f"{key}={value}".encode("ascii"))
    except Exception:
        pass
    try:
        import ctypes
        ucrt = ctypes.CDLL("ucrtbase.dll")
        if hasattr(ucrt, "_putenv"):
            ucrt._putenv(f"{key}={value}".encode("ascii"))
    except Exception:
        pass


def apply_ffmpeg_network_optimizations() -> None:
    """
    Aplica as variáveis de ambiente e opções do FFmpeg tanto no Python quanto no C runtime (msvcrt/ucrt)
    para garantir reconexão automática resiliente, transporte TCP sem drop de UDP,
    e supressão de mensagens de baixo nível de decodificação no console.
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_RTSP_OPTIONS
    # Garante que o C-runtime global não retenha 'nobuffer' por padrão, evitando corromper demuxers de arquivos locais MP4/MOV
    set_c_runtime_env("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = FFMPEG_RTSP_OPTIONS
    set_c_runtime_env("OPENCV_FFMPEG_LOGLEVEL", "-8")
    set_c_runtime_env("OPENCV_LOG_LEVEL", "OFF")
    set_c_runtime_env("OPENCV_VIDEOIO_DEBUG", "0")


# Aplica imediatamente na carga do módulo
apply_ffmpeg_network_optimizations()


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
        playback_speed: float = 1.0,
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
        self.fps: float = 30.0
        self.native_fps: float = 30.0
        self.resolution: Tuple[int, int] = (0, 0)
        self.latency_ms: float = 0.0

        self._lock = threading.Lock()
        self._seek_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Controle de Velocidade e Posição (0.5x, 1.0x, 1.5x, 2.0x e Retroceder/Avançar)
        self.playback_speed: float = max(0.25, min(4.0, float(playback_speed)))
        self.is_paused: bool = False
        self._seek_requested_sec: Optional[float] = None

        if auto_start:
            self.start()

    def _open_capture(self) -> bool:
        """Inicializa e configura a conexão do VideoCapture."""
        try:
            if self.is_rtsp:
                rtsp_opts = FFMPEG_RTSP_OPTIONS
                src_str = str(self.src).lower()
                if "transport=udp" in src_str or "?udp" in src_str or "&udp" in src_str:
                    rtsp_opts = rtsp_opts.replace("rtsp_transport;tcp", "rtsp_transport;udp")
                set_c_runtime_env("OPENCV_FFMPEG_CAPTURE_OPTIONS", rtsp_opts)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                set_c_runtime_env("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
            elif self.is_network:
                # Streams HTTP/HTTPS/TLS (YouTube, HLS, DASH, etc.): reconexão resiliente sem 'nobuffer'
                http_stream_opts = "reconnect;1|reconnect_streamed;1|reconnect_delay_max;5|flags;low_delay"
                set_c_runtime_env("OPENCV_FFMPEG_CAPTURE_OPTIONS", http_stream_opts)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                set_c_runtime_env("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
            elif isinstance(self.src, int):
                set_c_runtime_env("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
                backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.src, backend)
            else:
                set_c_runtime_env("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
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
                self.native_fps = float(stream_fps) if (0 < stream_fps <= 120) else 30.0
                self.fps = self.native_fps
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
        def _drain_stale_rtsp_frames() -> None:
            """Drena frames defasados retidos no buffer interno do FFmpeg/RTSP para sincronizar em tempo real."""
            if not self.is_rtsp or not self.cap or not self.cap.isOpened():
                return
            drained = 0
            while drained < 90 and not self._stop_event.is_set():
                t0 = time.time()
                ret_g = self.cap.grab()
                t1 = time.time()
                if not ret_g:
                    break
                drained += 1
                # Se demorou mais que 8ms, significa que o buffer interno esvaziou e atingimos o sinal ao vivo da câmera
                if (t1 - t0) >= 0.008:
                    break
            if drained > 0:
                ret_r, f_r = self.cap.retrieve()
                if ret_r and f_r is not None:
                    with self._lock:
                        self.last_frame = f_r
                        self.last_frame_time = time.time()
                log_event("INFO", f"[{self.name}] Buffer RTSP inicial drenado: {drained} frames defasados descartados.", "stream_capture")

        # Tentativa inicial de conexão
        opened = self._open_capture()
        if opened:
            _drain_stale_rtsp_frames()
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
                if opened:
                    _drain_stale_rtsp_frames()
                continue

            # Processamento de requisição pendente de Seek (Avançar / Retroceder)
            with self._seek_lock:
                if self._seek_requested_sec is not None and self.cap and self.cap.isOpened():
                    target_sec = self._seek_requested_sec
                    self._seek_requested_sec = None
                    self.dropped_frames = 0
                    try:
                        target_ms = max(0.0, target_sec * 1000.0)
                        seek_ok = self.cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
                        if not seek_ok and self.fps > 0:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * self.fps))
                        # Se estiver pausado, atualiza o last_frame imediatamente para a nova posição
                        if self.is_paused:
                            ret_seek, frame_seek = self.cap.read()
                            if ret_seek and frame_seek is not None:
                                with self._lock:
                                    self.last_frame = frame_seek
                    except Exception as ex:
                        log_event("WARNING", f"[{self.name}] Falha no seek para {target_sec:.1f}s: {ex}", "stream_capture")

            # Se estiver pausado, aguardar sem ler novos frames
            if self.is_paused:
                fps_calc_time = time.time()
                fps_frame_counter = 0
                time.sleep(0.04)
                continue

            # Captura de frame
            t_before = time.time()
            ret, frame = self.cap.read()
            t_after = time.time()

            # Para streams RTSP: drenagem contínua de frames acumulados no socket/demuxer
            # Se a leitura demorou < 3ms, significa que o frame veio de um buffer local acumulado.
            if self.is_rtsp and ret and (t_after - t_before) < 0.003 and not self.is_paused:
                drain_count = 0
                while drain_count < 15 and not self._stop_event.is_set():
                    t_d0 = time.time()
                    ret_d = self.cap.grab()
                    t_d1 = time.time()
                    if not ret_d:
                        break
                    drain_count += 1
                    if (t_d1 - t_d0) >= 0.005:
                        break
                if drain_count > 0:
                    ret_last, frame_last = self.cap.retrieve()
                    if ret_last and frame_last is not None:
                        frame = frame_last
                        t_after = time.time()

            if ret and frame is not None:
                reconnect_count = 0  # Reseta contador de reconexões após sucesso
                self.dropped_frames = 0
                with self._lock:
                    self.last_frame = frame
                    self.last_frame_time = t_after
                    self.frame_count += 1
                    self.status = "CONNECTED"
                    self.error_message = ""
                    self.latency_ms = (t_after - t_before) * 1000.0
                    cur_speed = self.playback_speed

                # Para velocidades aceleradas (1.5x ou 2.0x), adiantar leitura de frames
                if cur_speed >= 1.75 and self.cap and self.cap.isOpened():
                    self.cap.grab()  # Pula 1 frame para reproduzir a 2.0x em tempo de vídeo
                elif cur_speed >= 1.4 and self.cap and self.cap.isOpened() and self.frame_count % 2 == 0:
                    self.cap.grab()  # Pula 1 frame alternado para reproduzir a ~1.5x

                # Cálculo de FPS real efetivo
                fps_frame_counter += 1
                elapsed_fps = t_after - fps_calc_time
                if elapsed_fps >= 1.0:
                    measured = fps_frame_counter / elapsed_fps
                    if measured > 0:
                        self.fps = measured
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
                # Se falhar leituras consecutivas, acionar reconexão resiliente
                max_allowed_dropped = 45 if self.is_network else 15
                if self.dropped_frames % 15 == 0:
                    log_event("WARNING", f"[{self.name}] Falha na leitura de frame ({self.dropped_frames} frames perdidos).", "stream_capture")
                if self.dropped_frames >= max_allowed_dropped:
                    opened = False

            # Cadência de tempo:
            # - Arquivos locais gravados (self.is_file): respeitam a velocidade de reprodução (0.5x, 1.0x, etc.)
            # - Streams ao vivo (RTSP, RTMP, Webcams): rodam sem atraso artificial (tempo real estrito)
            if self.is_file:
                with self._lock:
                    eff_speed = max(0.25, min(4.0, self.playback_speed))
                base_fps = self.native_fps if (hasattr(self, "native_fps") and self.native_fps > 0) else 30.0
                cadence_delay = (1.0 / max(10.0, base_fps)) / eff_speed
                time.sleep(max(0.002, cadence_delay))
            else:
                time.sleep(0.001)


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

    def is_alive(self) -> bool:
        """Verifica se a thread de captura assíncrona está ativa e em execução."""
        return self._thread is not None and self._thread.is_alive() and self.status not in ["DISCONNECTED", "STOPPED", "ERROR"]

    def set_speed(self, speed: float) -> None:
        """
        Define a velocidade de reprodução do stream.
        Suporta velocidades: 0.5x, 1.0x (normal), 1.5x e 2.0x.
        """
        with self._lock:
            self.playback_speed = max(0.25, min(4.0, float(speed)))
        log_event("INFO", f"[{self.name}] Velocidade de reprodução alterada para {self.playback_speed:.1f}x", "stream_capture")

    def get_speed(self) -> float:
        """Retorna a velocidade de reprodução atual."""
        with self._lock:
            return self.playback_speed

    def seek(self, delta_seconds: float) -> bool:
        """
        Avança ou retrocede a reprodução no stream por delta_seconds.
        delta_seconds < 0: retrocede (ex: -5s, -10s)
        delta_seconds > 0: avança (ex: +5s, +10s)
        """
        with self._seek_lock:
            curr = self.get_position_seconds()
            target = max(0.0, curr + delta_seconds)
            self._seek_requested_sec = target
            log_event("INFO", f"[{self.name}] Seek relativo solicitado ({delta_seconds:+.1f}s -> {target:.1f}s)", "stream_capture")
            return True

    def seek_to(self, target_seconds: float) -> bool:
        """Posiciona a reprodução no instante específico em segundos."""
        with self._seek_lock:
            target = max(0.0, float(target_seconds))
            self._seek_requested_sec = target
            log_event("INFO", f"[{self.name}] Seek absoluto solicitado para {target:.1f}s", "stream_capture")
            return True

    def toggle_pause(self) -> bool:
        """Alterna o estado de pausa da reprodução."""
        with self._lock:
            self.is_paused = not self.is_paused
            is_now_p = self.is_paused
            if not is_now_p:
                self.dropped_frames = 0
            state_str = "PAUSADO" if is_now_p else "RETOMADO"
        log_event("INFO", f"[{self.name}] Estado de reprodução alterado: {state_str}", "stream_capture")
        return is_now_p

    def get_position_seconds(self) -> float:
        """Retorna a posição temporal estimada do stream em segundos."""
        with self._lock:
            if not self.cap or not self.cap.isOpened():
                return 0.0
            try:
                pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms > 0:
                    return pos_ms / 1000.0
                pos_f = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                if pos_f > 0 and self.fps > 0:
                    return pos_f / self.fps
            except Exception:
                pass
            return float(self.frame_count) / max(1.0, self.fps)

    def get_duration_seconds(self) -> float:
        """Retorna a duração total do vídeo em segundos (se aplicável/conhecida)."""
        with self._lock:
            if not self.cap or not self.cap.isOpened():
                return 0.0
            try:
                total_f = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total_f > 0 and self.fps > 0:
                    return total_f / self.fps
            except Exception:
                pass
            return 0.0

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
            "is_rtsp": self.is_rtsp,
            "playback_speed": self.playback_speed,
            "position_seconds": round(self.get_position_seconds(), 1),
            "is_paused": self.is_paused
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


def format_stream_time(seconds: float) -> str:
    """
    Formata tempo em segundos para representação amigável MM:SS.s.
    Ex: 65.4 -> '01:05.4'
    """
    sec = max(0.0, float(seconds))
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:04.1f}"


def probe_stream_connection(
    source: Union[str, int],
    timeout_seconds: float = 6.0
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
            if stream.status in ["DISCONNECTED", "ERROR"] and (stream._thread is None or not stream._thread.is_alive()):
                break
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


def is_web_streaming_url(url: Any) -> bool:
    """
    Verifica se a fonte informada é uma URL de streaming da web ou de rede
    (YouTube, YouTube Live, Twitch, Vimeo, links HLS .m3u8, RTMP, RTSP, HTTP/HTTPS).
    """
    if not url or not isinstance(url, str):
        return False
    clean = url.strip().lower()
    return any(clean.startswith(proto) for proto in ["rtsp://", "rtmp://", "http://", "https://"])


def resolve_streaming_url(
    url: str,
    quality: str = "media",
    timeout: int = 15,
    cookie_file: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve uma URL de streaming da web (YouTube, YouTube Live, Twitch, Vimeo, HLS, RTMP, RTSP)
    para um endereço direto decodificável pelo OpenCV / FFmpeg em tempo real com baixa latência.
    
    Retorna:
    - direct_stream_url (str): URL direta do fluxo de vídeo decodificável.
    - info (dict): Dicionário com metadados do stream (título, autor, status ao vivo, resolução, thumbnail).
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL de streaming inválida ou vazia.")

    clean_url = url.strip()
    clean_lower = clean_url.lower()

    # 1. Fluxos diretos de rede (RTSP, RTMP)
    if clean_lower.startswith(("rtsp://", "rtmp://")):
        return clean_url, {
            "id": "direct_stream",
            "title": "Stream de Rede IP (RTSP/RTMP)",
            "uploader": "Câmera / Servidor Local",
            "is_live": True,
            "resolution": "HD",
            "fps": 30.0,
            "duration_formatted": "🔴 AO VIVO",
            "thumbnail": "",
            "direct_url": clean_url
        }

    # 2. Arquivos de stream diretos (.m3u8 HLS ou arquivos diretos de vídeo HTTP/HTTPS)
    if ".m3u8" in clean_lower or any(clean_lower.split("?")[0].endswith(ext) for ext in [".mp4", ".webm", ".avi", ".mov"]):
        is_hls = ".m3u8" in clean_lower
        return clean_url, {
            "id": "direct_hls" if is_hls else "direct_video",
            "title": "Fluxo HLS Ao Vivo (.m3u8)" if is_hls else "Stream de Vídeo HTTP Direto",
            "uploader": "Servidor de Streaming",
            "is_live": is_hls,
            "resolution": "HD",
            "fps": 30.0,
            "duration_formatted": "🔴 AO VIVO" if is_hls else "00:00",
            "thumbnail": "",
            "direct_url": clean_url
        }

    # 3. Plataformas Web (YouTube, YouTube Live, Vimeo, Twitch, etc.) via yt-dlp
    try:
        from src.utils.video_downloader import get_base_ydl_opts, VideoDownloadError
        import yt_dlp
    except ImportError:
        return clean_url, {
            "id": "web_stream",
            "title": "Transmissão Web",
            "uploader": "Web",
            "is_live": True,
            "resolution": "HD",
            "fps": 30.0,
            "duration_formatted": "🔴 AO VIVO",
            "thumbnail": "",
            "direct_url": clean_url
        }

    q = quality.lower().strip() if quality else "media"
    if q in ["alta", "high"]:
        fmt = "best[protocol^=m3u8]/best[ext=mp4]/best/22/18"
    elif q in ["baixa", "low"]:
        fmt = "worst[protocol^=m3u8]/worst[ext=mp4]/worst/18"
    else:  # media
        fmt = "best[protocol^=m3u8][height<=720]/best[height<=720][ext=mp4]/best[height<=720]/22/18/best"

    client_strategies = [
        ["android", "visionos"],
        ["visionos", "android"],
        ["android"],
        ["visionos"],
        ["web", "android"],
        ["default"]
    ]

    last_error = ""
    for clients in client_strategies:
        ydl_opts = get_base_ydl_opts(timeout=timeout, client_list=clients, cookie_file=cookie_file)
        ydl_opts.update({
            "format": fmt,
            "skip_download": True,
            "extract_flat": False
        })
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=False)
                if not info:
                    continue
                if "entries" in info and info["entries"]:
                    info = info["entries"][0]

                direct = info.get("url")
                if not direct and "formats" in info and info["formats"]:
                    for f in reversed(info["formats"]):
                        if f.get("url"):
                            direct = f["url"]
                            break

                if direct:
                    w = info.get("width") or 0
                    h = info.get("height") or 0
                    res_str = f"{w}x{h}" if w and h else (info.get("resolution") or "HD")
                    fps_val = float(info.get("fps") or 30.0)
                    is_live = bool(info.get("is_live", False))
                    duration_sec = float(info.get("duration") or 0.0)
                    dur_str = "🔴 AO VIVO" if is_live else f"{int(duration_sec // 60):02d}:{int(duration_sec % 60):02d}"

                    meta = {
                        "id": info.get("id", "stream"),
                        "title": info.get("title", "Transmissão de Kendo"),
                        "uploader": info.get("uploader", info.get("channel", "Canal Web")),
                        "is_live": is_live,
                        "resolution": res_str,
                        "fps": fps_val,
                        "duration_seconds": duration_sec,
                        "duration_formatted": dur_str,
                        "thumbnail": info.get("thumbnail", ""),
                        "direct_url": direct
                    }
                    log_event("INFO", f"Stream resolvido com sucesso ({clients}): {meta['title']} ({res_str})", "stream_capture")
                    return direct, meta
        except Exception as e:
            last_error = str(e)
            continue

    # Se falhou e for URL do YouTube, tenta recuperar erros de digitação comuns em caracteres ambíguos ('l' vs 'I', '0' vs 'O')
    if "youtube.com" in clean_lower or "youtu.be" in clean_lower:
        import re
        match_id = re.search(r'(?:v=|\/live\/|\/shorts\/|youtu\.be\/)([a-zA-Z0-9_-]{11})', clean_url)
        if match_id:
            orig_id = match_id.group(1)
            candidate_ids = []
            if 'l' in orig_id:
                candidate_ids.append(orig_id.replace('l', 'I'))
            if 'I' in orig_id:
                candidate_ids.append(orig_id.replace('I', 'l'))
            if '0' in orig_id:
                candidate_ids.append(orig_id.replace('0', 'O'))
            if 'O' in orig_id:
                candidate_ids.append(orig_id.replace('O', '0'))

            for cand_id in candidate_ids:
                if cand_id == orig_id:
                    continue
                cand_url = clean_url.replace(orig_id, cand_id)
                try:
                    direct, meta = resolve_streaming_url(cand_url, quality=quality, timeout=timeout, cookie_file=cookie_file)
                    meta["corrected_url"] = cand_url
                    meta["corrected_from_url"] = clean_url
                    log_event("INFO", f"Stream recuperado com sucesso via autocorreção de ID ({orig_id} -> {cand_id}): {meta['title']}", "stream_capture")
                    return direct, meta
                except Exception:
                    pass

    log_event("ERROR", f"Falha ao resolver streaming para {clean_url}: {last_error}", "stream_capture")
    if "This video is unavailable" in last_error or "Video unavailable" in last_error:
        user_msg = "Este vídeo não está disponível no YouTube (verifique se há erro de digitação no link ou se o vídeo foi tornado privado/removido)."
    else:
        user_msg = f"Não foi possível obter o fluxo de vídeo deste streaming: {last_error or 'Link inacessível ou formato incompatível.'}"
    raise VideoDownloadError(user_msg)


def format_stream_time(seconds: float) -> str:
    """
    Formata tempo em segundos para representação amigável MM:SS.s.
    Exemplo: 65.4 -> '01:05.4'
    
    Args:
        seconds: Tempo em segundos (float ou int).
        
    Returns:
        String formatada em MM:SS.s.
    """
    if seconds is None or seconds < 0:
        return "00:00.0"
    m = int(seconds // 60)
    s = int(seconds % 60)
    fraction = int((seconds - int(seconds)) * 10)
    return f"{m:02d}:{s:02d}.{fraction:01d}"


class StreamSyncTracker:
    """
    Monitor e Calibrador Automático de Atraso Temporal Multi-Câmeras.

    Funcionalidades Principais:
    1. Ancoragem na Webcam Local: A webcam local (DirectShow/USB) é sempre a Referência Zero (Ground Truth 0ms).
    2. Calibração Contínua de Atraso: Mede continuamente o atraso temporal (delay offset em milissegundos)
       entre o fluxo RTSP/IP e a webcam local através de correlação cruzada de perfil de movimento.
    3. Tempo Real Estrito: Não impõe atraso artificial — todas as câmeras exibem seus frames mais
       recentes de forma fluida e direta.
    4. Diagnóstico em Tempo Real: Fornece badges e métricas de latência para legendas e HUDs do SenpAI.
    """

    def __init__(
        self,
        reference_index: int = 0,
        window_size: int = 90,
        buffer_capacity: int = 150,
        fps: float = 30.0,
        smoothing_alpha: float = 0.20,
        initial_delay_ms: float = 100.0
    ):
        self.reference_index = reference_index
        self.window_size = window_size
        self.buffer_capacity = buffer_capacity
        self.fps = max(1.0, float(fps))
        self.smoothing_alpha = smoothing_alpha
        self.initial_delay_ms = max(0.0, float(initial_delay_ms))

        # Buffer temporal de frames para compensação visual e biomecânica da webcam (referência)
        # Armazena tuplas: (timestamp: float, frame: np.ndarray, drawn_frame: Optional[np.ndarray], landmarks: Optional[Any])
        self._frame_buffers: Dict[int, deque] = {}

        # Estruturas por câmera: histórico de movimento e último frame em escala reduzida
        self._prev_thumbnails: Dict[int, np.ndarray] = {}
        self._motion_histories: Dict[int, deque] = {}  # deque de (timestamp, motion_energy)
        self._measured_delays_ms: Dict[int, float] = {reference_index: 0.0}
        self._last_calc_time = 0.0
        self._lock = threading.Lock()

    def set_reference_index(self, ref_idx: int) -> None:
        """Define o índice da câmera de referência (Webcam local = 0ms)."""
        with self._lock:
            self.reference_index = ref_idx
            self._measured_delays_ms[ref_idx] = 0.0

    def register_processed_frame(
        self,
        cam_idx: int,
        frame: Optional[np.ndarray],
        drawn_frame: Optional[np.ndarray] = None,
        landmarks: Optional[Any] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Registra um frame capturado e processado no calibrador.
        Para a câmera de referência (Webcam), enfileira no buffer FIFO para permitir a compensação do atraso.
        Também atualiza o histórico de movimento para calibração contínua de latência.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return

        ts = timestamp if timestamp is not None else time.time()

        with self._lock:
            # Armazena no buffer temporal da câmera de referência para compensação visual
            if cam_idx == self.reference_index:
                if cam_idx not in self._frame_buffers:
                    self._frame_buffers[cam_idx] = deque(maxlen=self.buffer_capacity)
                self._frame_buffers[cam_idx].append((ts, frame, drawn_frame, landmarks))

            # Se for stream secundário (RTSP) e ainda não tiver delay registrado, inicializa com initial_delay_ms
            if cam_idx != self.reference_index and cam_idx not in self._measured_delays_ms:
                self._measured_delays_ms[cam_idx] = self.initial_delay_ms

        # Atualiza métricas de movimento para a correlação cruzada
        self.update_frame(cam_idx=cam_idx, frame=frame, timestamp=ts)

    def get_synced_data(
        self,
        cam_idx: int,
        current_frame: np.ndarray,
        current_drawn: Optional[np.ndarray] = None,
        current_landmarks: Optional[Any] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """
        Retorna o frame, overlay anotado e landmarks sincronizados no mesmo instante temporal exato.

        - Para a Webcam (referência): busca no buffer FIFO o instante (t_now - delta_t),
          compensando o atraso da câmera RTSP para que ambas as imagens exibam o MESMO momento.
        - Para streams RTSP / IP: retorna o frame atual (que já é o mais recente decodificado).
        """
        if cam_idx != self.reference_index:
            return current_frame, current_drawn, current_landmarks

        with self._lock:
            delays = [d for idx, d in self._measured_delays_ms.items() if idx != self.reference_index]
            max_delay_ms = max(delays) if delays else 0.0

            # Se o atraso for desprezível (< 15ms) ou o buffer ainda estiver vazio, exibe tempo real direto
            if max_delay_ms < 15.0 or cam_idx not in self._frame_buffers or len(self._frame_buffers[cam_idx]) == 0:
                return current_frame, current_drawn, current_landmarks

            now = time.time()
            target_ts = now - (max_delay_ms / 1000.0)
            buf = self._frame_buffers[cam_idx]

            # Busca o frame com timestamp mais próximo de target_ts
            best_entry = None
            best_diff = float("inf")
            for entry in reversed(buf):
                diff = abs(entry[0] - target_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_entry = entry
                elif entry[0] < target_ts:
                    break

            # Limpa frames antigos do buffer (mais velhos que 1.5s antes do alvo)
            cutoff = target_ts - 1.5
            while len(buf) > 10 and buf[0][0] < cutoff:
                buf.popleft()

            if best_entry is not None:
                _, s_frame, s_drawn, s_lm = best_entry
                return (
                    s_frame if s_frame is not None else current_frame,
                    s_drawn if s_drawn is not None else current_drawn,
                    s_lm if s_lm is not None else current_landmarks
                )

            return current_frame, current_drawn, current_landmarks

    def update_frame(
        self,
        cam_idx: int,
        frame: Optional[np.ndarray],
        timestamp: Optional[float] = None
    ) -> None:
        """
        Registra um novo frame recebido da câmera, calculando a energia de movimento instantânea.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return

        ts = timestamp if timestamp is not None else time.time()

        # Reduz resolução para cálculo ultrarrápido (< 0.05ms)
        try:
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                return
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            thumb = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_NEAREST)
        except Exception:
            return

        with self._lock:
            if cam_idx not in self._motion_histories:
                self._motion_histories[cam_idx] = deque(maxlen=self.window_size)

            prev_thumb = self._prev_thumbnails.get(cam_idx)
            self._prev_thumbnails[cam_idx] = thumb

            if prev_thumb is not None and prev_thumb.shape == thumb.shape:
                diff = cv2.absdiff(thumb, prev_thumb)
                motion_energy = float(np.mean(diff))
            else:
                motion_energy = 0.0

            self._motion_histories[cam_idx].append((ts, motion_energy))

            # Dispara calibração periódica (a cada ~8 frames ou 0.25s)
            now = ts
            if cam_idx != self.reference_index and (now - self._last_calc_time) >= 0.25:
                self.recalculate_delays()
                self._last_calc_time = now

    def recalculate_delays(self) -> None:
        """Recalcula a correlação cruzada entre a câmera de referência e os streams secundários."""
        ref_hist = self._motion_histories.get(self.reference_index)
        if not ref_hist or len(ref_hist) < 20:
            return

        ref_energies = np.array([m for _, m in ref_hist], dtype=np.float32)
        ref_std = float(np.std(ref_energies))
        # Se não há movimentação detectável na cena (kendokas imóveis), manter última estimativa
        if ref_std < 0.2:
            return

        ref_norm = (ref_energies - float(np.mean(ref_energies))) / (ref_std + 1e-6)

        for cam_idx, target_hist in self._motion_histories.items():
            if cam_idx == self.reference_index or len(target_hist) < 20:
                continue

            target_energies = np.array([m for _, m in target_hist], dtype=np.float32)
            target_std = float(np.std(target_energies))
            if target_std < 0.2:
                continue

            # Alinha tamanhos
            min_len = min(len(ref_norm), len(target_energies))
            s1 = ref_norm[-min_len:]
            t_slice = target_energies[-min_len:]
            s2 = (t_slice - float(np.mean(t_slice))) / (target_std + 1e-6)

            # Correlação cruzada
            corr = np.correlate(s2, s1, mode="full") / min_len
            lags = np.arange(-min_len + 1, min_len)

            # Restringe faixa razoável de atraso (-5 a +60 frames, ou seja, -150ms a +2000ms)
            valid_mask = (lags >= -5) & (lags <= 60)
            if not np.any(valid_mask):
                continue

            sub_corr = corr[valid_mask]
            sub_lags = lags[valid_mask]
            best_idx = np.argmax(sub_corr)
            best_lag_frames = int(sub_lags[best_idx])
            best_corr_val = float(sub_corr[best_idx])

            # Apenas atualiza se a correlação for estatisticamente consistente (> 0.35)
            if best_corr_val > 0.35:
                measured_ms = max(0.0, best_lag_frames * (1000.0 / self.fps))
                prev = self._measured_delays_ms.get(cam_idx, measured_ms)
                # Suavização exponencial (EMA)
                smoothed = (self.smoothing_alpha * measured_ms) + ((1.0 - self.smoothing_alpha) * prev)
                self._measured_delays_ms[cam_idx] = round(smoothed, 1)

    def get_delay_ms(self, cam_idx: int) -> float:
        """Retorna o atraso em milissegundos relativo à Webcam local (0.0 para referência)."""
        with self._lock:
            return self._measured_delays_ms.get(cam_idx, 0.0)

    def get_sync_badge(self, cam_idx: int) -> str:
        """Retorna badge descritivo de sincronização para legendas e HUDs."""
        with self._lock:
            if cam_idx == self.reference_index:
                delays = [d for idx, d in self._measured_delays_ms.items() if idx != self.reference_index]
                comp_ms = max(delays) if delays else 0.0
                if comp_ms < 15.0:
                    return "⏱️ Ref. Local 0ms"
                elif comp_ms <= 150.0:
                    return f"⏱️ Ref. Local (Compensado -{comp_ms:.0f}ms)"
                else:
                    return f"⏱️ Ref. Local (Compensado -{comp_ms / 1000.0:.2f}s)"

            d_ms = self._measured_delays_ms.get(cam_idx, 0.0)
            if d_ms <= 40.0:
                return f"⏱️ Sincronizado (+{d_ms:.0f}ms)"
            elif d_ms <= 150.0:
                return f"⏱️ Atraso Real: +{d_ms:.0f}ms"
            else:
                return f"⏱️ Atraso Real: +{d_ms / 1000.0:.2f}s"

    def get_all_delays(self) -> Dict[int, float]:
        """Retorna dicionário de atrasos em ms para todas as câmeras."""
        with self._lock:
            return dict(self._measured_delays_ms)




