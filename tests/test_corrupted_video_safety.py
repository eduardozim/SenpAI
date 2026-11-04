"""
Testes de Integridade e Proteção contra Vídeos Corrompidos no SenpAI.
Garante que arquivos corrompidos, vazios (0 bytes) ou ilegíveis:
1. Sejam rejeitados imediatamente com exceção descritiva.
2. Nunca gerem análises concluídas com placar fictício 0-0.
3. Nunca sejam contabilizados como sessões nos perfis cinestésicos dos praticantes.
"""

import os
import tempfile
import unittest
from src.pipeline import SenpAIPipeline, AsyncVideoBatchReader
from src.analytics.kenshi_style_model import KinestheticProfileManager

class TestCorruptedVideoSafety(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.empty_video = os.path.join(self.tmp_dir.name, "empty_video.mp4")
        with open(self.empty_video, "wb") as f:
            f.write(b"")

        self.corrupt_video = os.path.join(self.tmp_dir.name, "corrupt_video.mp4")
        with open(self.corrupt_video, "wb") as f:
            f.write(b"NOT_A_VALID_MP4_HEADER_GARBAGE_BYTES_1234567890")

        self.pipeline = SenpAIPipeline(vision_model="mediapipe", device_preference="cpu")
        self.baseline_path = os.path.join(self.tmp_dir.name, "test_baselines.json")
        self.kin_mgr = KinestheticProfileManager(storage_path=self.baseline_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_reader_rejects_empty_zero_byte_video(self):
        with self.assertRaises(ValueError) as ctx:
            with AsyncVideoBatchReader(self.empty_video):
                pass
        self.assertIn("0 bytes", str(ctx.exception))

    def test_reader_rejects_corrupted_video_bytes(self):
        with self.assertRaises(ValueError) as ctx:
            with AsyncVideoBatchReader(self.corrupt_video):
                pass
        self.assertTrue(
            "corrompido" in str(ctx.exception).lower() or
            "não foi possível abrir" in str(ctx.exception).lower() or
            "legíveis" in str(ctx.exception).lower()
        )

    def test_process_video_rejects_corrupt_video_without_scoring_0_0(self):
        with self.assertRaises(ValueError) as ctx:
            self.pipeline.process_video(self.corrupt_video)
        self.assertTrue(
            "corrompido" in str(ctx.exception).lower() or
            "não foi possível abrir" in str(ctx.exception).lower() or
            "legíveis" in str(ctx.exception).lower()
        )

    def test_process_training_video_rejects_corrupt_video_and_does_not_count_session(self):
        initial_sessions = self.kin_mgr.get_or_create_profile("KENSHI_SHIRO").sessions_count
        with self.assertRaises(ValueError):
            self.pipeline.process_training_video(self.corrupt_video)

        # Recarregar perfil e verificar que nenhuma sessão foi adicionada
        reloaded_mgr = KinestheticProfileManager(storage_path=self.baseline_path)
        final_sessions = reloaded_mgr.get_or_create_profile("KENSHI_SHIRO").sessions_count
        self.assertEqual(initial_sessions, final_sessions, "Nenhuma sessão deve ser adicionada para vídeo corrompido.")

if __name__ == "__main__":
    unittest.main()
