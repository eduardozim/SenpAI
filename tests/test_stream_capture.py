"""
Testes Unitários e de Integração para o Módulo de Captura de Streams (stream_capture.py).
Valida normalização de fontes, ThreadedVideoStream, diagnóstico prévio e compatibilidade RTSP/HTTP.
"""

import os
import time
import unittest
import numpy as np
import cv2

from src.utils.stream_capture import (
    normalize_stream_source,
    apply_ffmpeg_network_optimizations,
    ThreadedVideoStream,
    probe_stream_connection,
    FFMPEG_RTSP_OPTIONS
)
from src.utils.demo_generator import generate_demo_kendo_video


class TestStreamCapture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo_video_path = os.path.abspath("tests_demo_stream.mp4")
        generate_demo_kendo_video(cls.demo_video_path, duration_sec=2)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.demo_video_path):
            try:
                os.remove(cls.demo_video_path)
            except Exception:
                pass

    def test_normalize_stream_source_int(self):
        """Valida que índices numéricos nativos são preservados como inteiros."""
        self.assertEqual(normalize_stream_source(0), 0)
        self.assertEqual(normalize_stream_source(1), 1)

    def test_normalize_stream_source_string_digits(self):
        """Valida que strings numéricas são convertidas corretamente para inteiros."""
        self.assertEqual(normalize_stream_source("0"), 0)
        self.assertEqual(normalize_stream_source(" 2 "), 2)

    def test_normalize_stream_source_urls(self):
        """Valida limpeza e normalização de URLs de streams de rede."""
        self.assertEqual(
            normalize_stream_source(" rtsp://192.168.1.100:554/live.sdp "),
            "rtsp://192.168.1.100:554/live.sdp"
        )
        self.assertEqual(
            normalize_stream_source('"http://192.168.1.50:8080/video"'),
            "http://192.168.1.50:8080/video"
        )
        self.assertEqual(
            normalize_stream_source("'rtmp://live.stream/shiai'"),
            "rtmp://live.stream/shiai"
        )

    def test_ffmpeg_optimizations_environment(self):
        """Valida que as flags de otimização de rede FFmpeg/TCP são configuradas."""
        apply_ffmpeg_network_optimizations()
        self.assertIn("OPENCV_FFMPEG_CAPTURE_OPTIONS", os.environ)
        self.assertIn("rtsp_transport;tcp", os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"])
        self.assertIn("nobuffer", os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"])

    def test_threaded_video_stream_lifecycle(self):
        """Valida o ciclo de vida completo do ThreadedVideoStream em arquivo/stream local."""
        stream = ThreadedVideoStream(
            src=self.demo_video_path,
            name="TestStream",
            auto_start=True
        )

        time.sleep(0.3)
        self.assertTrue(stream.is_connected())

        ret, frame = stream.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)

        ret_rgb, frame_rgb = stream.read_rgb()
        self.assertTrue(ret_rgb)
        self.assertIsNotNone(frame_rgb)

        stats = stream.get_stats()
        self.assertEqual(stats["name"], "TestStream")
        self.assertEqual(stats["status"], "CONNECTED")
        self.assertGreater(stats["resolution"][0], 0)
        self.assertGreater(stats["resolution"][1], 0)
        self.assertGreaterEqual(stats["frame_count"], 1)

        stream.stop()
        self.assertIn(stream.status, ["STOPPED", "DISCONNECTED"])
        self.assertIsNone(stream.cap)

    def test_threaded_video_stream_context_manager(self):
        """Valida o uso do ThreadedVideoStream através do protocolo with."""
        with ThreadedVideoStream(src=self.demo_video_path, name="ContextStream") as stream:
            time.sleep(0.2)
            ret, frame = stream.read()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)

        self.assertIn(stream.status, ["STOPPED", "DISCONNECTED"])

    def test_probe_stream_connection_valid_source(self):
        """Valida a rotina de diagnóstico probe_stream_connection para uma fonte funcional."""
        diag = probe_stream_connection(self.demo_video_path, timeout_seconds=2.0)
        self.assertTrue(diag["success"])
        self.assertIn("Conectado com sucesso", diag["message"])
        self.assertIsNotNone(diag["frame_rgb"])
        self.assertGreater(diag["resolution"][0], 0)
        self.assertGreater(diag["resolution"][1], 0)

    def test_probe_stream_connection_invalid_source(self):
        """Valida a rotina de diagnóstico probe_stream_connection para uma fonte inexistente."""
        diag = probe_stream_connection("caminho_inexistente_video_12345.mp4", timeout_seconds=0.5)
        self.assertFalse(diag["success"])
        self.assertIn("Falha na conexão", diag["message"])
        self.assertIsNone(diag["frame_rgb"])

    def test_is_web_streaming_url(self):
        """Valida a identificação correta de URLs de streaming de rede e web."""
        from src.utils.stream_capture import is_web_streaming_url
        self.assertTrue(is_web_streaming_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_web_streaming_url("https://www.youtube.com/live/dQw4w9WgXcQ"))
        self.assertTrue(is_web_streaming_url("rtsp://192.168.1.50:554/live"))
        self.assertTrue(is_web_streaming_url("rtmp://stream.server/live/feed"))
        self.assertTrue(is_web_streaming_url("http://192.168.1.10:8080/video.m3u8"))
        self.assertFalse(is_web_streaming_url("0"))
        self.assertFalse(is_web_streaming_url(1))
        self.assertFalse(is_web_streaming_url(""))
        self.assertFalse(is_web_streaming_url(None))

    def test_resolve_streaming_url_direct_protocols(self):
        """Valida que URLs diretas de rede RTSP, RTMP e HLS (.m3u8) são resolvidas diretamente sem yt-dlp."""
        from src.utils.stream_capture import resolve_streaming_url
        rtsp_url = "rtsp://192.168.1.100:554/ch1"
        u1, m1 = resolve_streaming_url(rtsp_url)
        self.assertEqual(u1, rtsp_url)
        self.assertTrue(m1["is_live"])
        self.assertEqual(m1["id"], "direct_stream")

        hls_url = "https://example.com/live/kendo_stream.m3u8"
        u2, m2 = resolve_streaming_url(hls_url)
        self.assertEqual(u2, hls_url)
        self.assertTrue(m2["is_live"])
        self.assertEqual(m2["id"], "direct_hls")

        direct_mp4 = "https://example.com/vod/match.mp4"
        u3, m3 = resolve_streaming_url(direct_mp4)
        self.assertEqual(u3, direct_mp4)
        self.assertEqual(m3["id"], "direct_video")

    def test_resolve_streaming_url_invalid_input(self):
        """Valida que entradas vazias ou inválidas levantam ValueError."""
        from src.utils.stream_capture import resolve_streaming_url
        with self.assertRaises(ValueError):
            resolve_streaming_url("")
        with self.assertRaises(ValueError):
            resolve_streaming_url(None)  # type: ignore

    def test_resolve_streaming_url_mocked_ytdlp(self):
        """Valida que a extração via yt-dlp formata metadados e retorna a URL direta."""
        from unittest.mock import patch, MagicMock
        from src.utils.stream_capture import resolve_streaming_url

        mock_info = {
            "id": "mock_yt_123",
            "title": "Final Mundial Kendo Ao Vivo",
            "uploader": "FIK Official",
            "is_live": True,
            "width": 1280,
            "height": 720,
            "fps": 60.0,
            "duration": 0,
            "url": "https://manifest.googlevideo.com/live/feed.m3u8",
            "thumbnail": "https://img.youtube.com/mock.jpg"
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl_class = MagicMock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance

        with patch("yt_dlp.YoutubeDL", mock_ydl_class):
            direct_u, meta = resolve_streaming_url("https://www.youtube.com/watch?v=mock_yt_123", quality="alta")
            self.assertEqual(direct_u, "https://manifest.googlevideo.com/live/feed.m3u8")
            self.assertEqual(meta["title"], "Final Mundial Kendo Ao Vivo")
            self.assertTrue(meta["is_live"])
            self.assertEqual(meta["resolution"], "1280x720")
            self.assertEqual(meta["duration_formatted"], "🔴 AO VIVO")

    def test_threaded_video_stream_speed_control(self):
        """Valida que o ThreadedVideoStream aceita e aplica velocidades 0.5x, 1.0x, 1.5x e 2.0x."""
        from src.utils.stream_capture import ThreadedVideoStream
        stream = ThreadedVideoStream(self.demo_video_path, name="TestSpeed", auto_start=False)
        self.assertEqual(stream.get_speed(), 1.0)
        
        # Testar velocidade 0.5x (câmera lenta)
        stream.set_speed(0.5)
        self.assertEqual(stream.get_speed(), 0.5)
        
        # Testar velocidade 1.5x (acelerada)
        stream.set_speed(1.5)
        self.assertEqual(stream.get_speed(), 1.5)

        # Testar velocidade 2.0x (rápida)
        stream.set_speed(2.0)
        self.assertEqual(stream.get_speed(), 2.0)

        # Testar retorno aos 1.0x (normal)
        stream.set_speed(1.0)
        self.assertEqual(stream.get_speed(), 1.0)

        stats = stream.get_stats()
        self.assertEqual(stats["playback_speed"], 1.0)
        stream.stop()

    def test_threaded_video_stream_seek_and_pause(self):
        """Valida que seek relativo, absoluto e pause alternam o estado sem travar a thread."""
        from src.utils.stream_capture import ThreadedVideoStream
        stream = ThreadedVideoStream(self.demo_video_path, name="TestSeek", auto_start=True)
        connected = stream.wait_until_connected(timeout_seconds=3.0)
        self.assertTrue(connected)

        # Testar pause e resume
        self.assertFalse(stream.is_paused)
        stream.toggle_pause()
        self.assertTrue(stream.is_paused)
        self.assertTrue(stream.get_stats()["is_paused"])
        stream.toggle_pause()
        self.assertFalse(stream.is_paused)

        # Testar seek relativo e absoluto
        self.assertTrue(stream.is_alive())
        self.assertTrue(stream.seek(1.0))
        self.assertTrue(stream.seek(-0.5))
        self.assertTrue(stream.seek_to(0.0))

        stream.stop()
        self.assertFalse(stream.is_alive())

    def test_format_stream_time(self):
        """Valida a formatação de timestamps de streaming em formato amigável MM:SS.s."""
        from src.utils.stream_capture import format_stream_time
        self.assertEqual(format_stream_time(0.0), "00:00.0")
        self.assertEqual(format_stream_time(9.5), "00:09.5")
        self.assertEqual(format_stream_time(65.4), "01:05.4")
        self.assertEqual(format_stream_time(125.9), "02:05.9")
        self.assertEqual(format_stream_time(-5.0), "00:00.0")


if __name__ == "__main__":
    unittest.main()

