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


if __name__ == "__main__":
    unittest.main()
