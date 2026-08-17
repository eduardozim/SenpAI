"""
Testes Unitários para Interrupção e Cancelamento do Pipeline no Modo Gravado.
"""

import unittest
import os
import tempfile
import cv2
import numpy as np

from src.pipeline import ShinpanaiPipeline
from src.utils.demo_generator import generate_demo_kendo_video
from src.utils.logger_manager import get_memory_logs


class TestPipelineCancellation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_video = os.path.join(cls.temp_dir.name, "test_cancel_video.mp4")
        generate_demo_kendo_video(cls.test_video, duration_sec=3, fps=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_immediate_cancellation(self):
        """Verifica que o pipeline interrompe imediatamente quando is_cancelled é True desde o início."""
        pipeline = ShinpanaiPipeline(calibration_profile="normal", device_preference="cpu")
        out_vid = os.path.join(self.temp_dir.name, "out_immediate_cancel.mp4")

        result = pipeline.process_video(
            video_path=self.test_video,
            output_video_path=out_vid,
            is_cancelled=lambda: True
        )

        self.assertIsNone(result, "O resultado deve ser None quando o processamento for cancelado imediatamente.")

    def test_midway_cancellation_during_frame_extraction(self):
        """Verifica que o pipeline interrompe no meio da extração de frames e libera recursos."""
        pipeline = ShinpanaiPipeline(calibration_profile="normal", device_preference="cpu")
        out_vid = os.path.join(self.temp_dir.name, "out_midway_cancel.mp4")

        calls = [0]
        def cancel_after_15_calls():
            calls[0] += 1
            return calls[0] > 15

        result = pipeline.process_video(
            video_path=self.test_video,
            output_video_path=out_vid,
            is_cancelled=cancel_after_15_calls
        )

        self.assertIsNone(result, "O resultado deve ser None quando cancelado no meio da extração.")
        self.assertGreater(calls[0], 15)

    def test_cancellation_logs_warning(self):
        """Verifica que o cancelamento registra um evento de WARNING nos logs do sistema."""
        pipeline = ShinpanaiPipeline(calibration_profile="normal", device_preference="cpu")

        _ = pipeline.process_video(
            video_path=self.test_video,
            is_cancelled=lambda: True
        )

        logs = get_memory_logs(max_entries=50, level_filter="WARNING")
        cancel_logs = [l for l in logs if "cancelado" in l.get("message", "").lower() or "interrompido" in l.get("message", "").lower()]
        self.assertTrue(len(cancel_logs) > 0, "Deve haver registro de log com aviso de cancelamento.")

    def test_normal_execution_without_cancellation(self):
        """Verifica que quando is_cancelled=lambda: False, o processamento ocorre normalmente."""
        pipeline = ShinpanaiPipeline(calibration_profile="normal", device_preference="cpu")
        out_vid = os.path.join(self.temp_dir.name, "out_normal.mp4")

        result = pipeline.process_video(
            video_path=self.test_video,
            output_video_path=out_vid,
            is_cancelled=lambda: False
        )

        self.assertIsNotNone(result)
        self.assertIn("events_detected_count", result)
        self.assertIn("sonkyo_analysis", result)

    def test_analysis_worker_cancellation(self):
        """Verifica que AnalysisWorker pode ser iniciado em background e cancelado imediatamente."""
        import time
        from src.pipeline import AnalysisWorker

        pipeline = ShinpanaiPipeline(calibration_profile="normal", device_preference="cpu")
        out_vid = os.path.join(self.temp_dir.name, "out_worker_cancel.mp4")

        worker = AnalysisWorker(
            pipeline=pipeline,
            video_path=self.test_video,
            output_video_path=out_vid
        )
        worker.start()
        time.sleep(0.05)
        worker.cancel()
        worker._thread.join(timeout=3.0)

        self.assertTrue(worker.is_done)
        self.assertTrue(worker.is_cancelled)
        self.assertIsNone(worker.result)

    def test_analysis_worker_normal_completion(self):
        """Verifica que AnalysisWorker conclui a execução em background com sucesso."""
        from src.pipeline import AnalysisWorker

        pipeline = ShinpanaiPipeline(calibration_profile="normal", device_preference="cpu")
        out_vid = os.path.join(self.temp_dir.name, "out_worker_success.mp4")

        worker = AnalysisWorker(
            pipeline=pipeline,
            video_path=self.test_video,
            output_video_path=out_vid
        )
        worker.start()
        worker._thread.join(timeout=10.0)

        self.assertTrue(worker.is_done)
        self.assertFalse(worker.is_cancelled)
        self.assertIsNotNone(worker.result)
        self.assertEqual(worker.progress, 1.0)


if __name__ == "__main__":
    unittest.main()
