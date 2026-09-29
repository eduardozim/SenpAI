"""
Testes unitários para o gerenciador de Treinamento em Tempo Real (LiveTrainingSessionManager).
"""

import unittest
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.analytics.training_live_manager import LiveTrainingSessionManager
from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA


class TestLiveTrainingSessionManager(unittest.TestCase):
    def setUp(self):
        self.manager = LiveTrainingSessionManager(
            modality_override="suburi",
            kendoka_name="Test Kendoka",
            target_dan=3
        )

    def test_initialization(self):
        self.assertEqual(self.manager.kendoka_name, "Test Kendoka")
        self.assertEqual(self.manager.current_modality_key, "suburi")
        self.assertFalse(self.manager.is_auto_detected)
        self.assertEqual(self.manager.rep_count, 0)
        self.assertEqual(len(self.manager.rep_history), 0)

    def test_process_empty_frame(self):
        res = self.manager.process_live_frame(live_pose_histories=[[]], fps=30.0)
        self.assertIn("modality_name", res)
        self.assertEqual(res["rep_count"], 0)
        self.assertIn("movement_score", res)
        self.assertIn("precision_score", res)
        self.assertIn("constancy_score", res)

    def test_furikaburi_and_strike_detection(self):
        # Simular quadro de Furikaburi (punho acima dos ombros)
        pose_furikaburi = {
            "RIGHT_WRIST": {"x": 0.5, "y": 0.2, "z": 0.0, "visibility": 0.9},
            "LEFT_WRIST": {"x": 0.5, "y": 0.2, "z": 0.0, "visibility": 0.9},
            "RIGHT_SHOULDER": {"x": 0.52, "y": 0.45, "z": 0.0, "visibility": 0.9},
            "LEFT_SHOULDER": {"x": 0.48, "y": 0.45, "z": 0.0, "visibility": 0.9},
            "RIGHT_HIP": {"x": 0.51, "y": 0.70, "z": 0.0, "visibility": 0.9},
            "LEFT_HIP": {"x": 0.49, "y": 0.70, "z": 0.0, "visibility": 0.9},
        }

        # 1. Passo com Furikaburi
        res1 = self.manager.process_live_frame([[pose_furikaburi]], fps=30.0)
        self.assertEqual(self.manager._strike_phase, "FURIKABURI")
        self.assertEqual(self.manager.rep_count, 0)

        # 2. Simular quadro de Uchi (punho descendo rapidamente para altura dos ombros)
        pose_strike = {
            "RIGHT_WRIST": {"x": 0.5, "y": 0.44, "z": 0.0, "visibility": 0.9},
            "LEFT_WRIST": {"x": 0.5, "y": 0.44, "z": 0.0, "visibility": 0.9},
            "RIGHT_SHOULDER": {"x": 0.52, "y": 0.45, "z": 0.0, "visibility": 0.9},
            "LEFT_SHOULDER": {"x": 0.48, "y": 0.45, "z": 0.0, "visibility": 0.9},
            "RIGHT_HIP": {"x": 0.51, "y": 0.70, "z": 0.0, "visibility": 0.9},
            "LEFT_HIP": {"x": 0.49, "y": 0.70, "z": 0.0, "visibility": 0.9},
        }

        time.sleep(0.4)  # Garantir intervalo mínimo
        res2 = self.manager.process_live_frame([[pose_strike]], fps=30.0)
        self.assertTrue(res2["new_rep_detected"])
        self.assertEqual(self.manager.rep_count, 1)
        self.assertEqual(len(self.manager.rep_history), 1)

    def test_render_live_hud_html(self):
        html_out = self.manager.render_live_hud_html()
        self.assertIn("REPETIÇÕES", html_out)
        self.assertIn("CADÊNCIA", html_out)
        self.assertIn("Movimentação", html_out)
        self.assertIn("Precisão", html_out)
        self.assertIn("Constância", html_out)

    def test_generate_final_session_report(self):
        rep = self.manager.generate_final_session_report()
        self.assertIn("markdown_report", rep)
        self.assertIn("duration_seconds", rep)
        self.assertIn("average_movement", rep)
        self.assertIn("average_precision", rep)
        self.assertIn("average_constancy", rep)
        self.assertIn("strengths", rep)
        self.assertIn("prescribed_drills", rep)


if __name__ == "__main__":
    unittest.main()
