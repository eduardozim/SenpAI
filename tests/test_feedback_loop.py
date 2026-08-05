"""
Testes Automatizados para Gerenciamento de Feedback e Aprendizagem por Reforço no ShinpanAI.
"""

import os
import json
import unittest
from src.engine.feedback_manager import FeedbackManager
from src.engine.calibrator import CalibrationEngine

class TestFeedbackLoop(unittest.TestCase):
    def setUp(self):
        self.test_dataset_path = "data/test_feedback_dataset.json"
        if os.path.exists(self.test_dataset_path):
            os.remove(self.test_dataset_path)
        self.feedback_mgr = FeedbackManager(dataset_path=self.test_dataset_path)

    def tearDown(self):
        if os.path.exists(self.test_dataset_path):
            os.remove(self.test_dataset_path)

    def test_save_and_load_feedback(self):
        entry = self.feedback_mgr.save_feedback(
            video_name="match1.mp4",
            profile_key="normal",
            event_id="ev_1",
            label="TP",
            sub_scores={"target_impact": 85.0, "fumikomi_sync": 70.0, "posture": 80.0, "zanshin": 75.0},
            total_score=78.5,
            strike_type="MEN",
            timestamp="00:01.500"
        )
        self.assertEqual(entry["label"], "TP")

        stats = self.feedback_mgr.get_stats(profile_key="normal")
        self.assertEqual(stats["total_feedback"], 1)
        self.assertEqual(stats["true_positives"], 1)
        self.assertEqual(stats["false_positives"], 0)

    def test_optimize_profile_on_false_positives(self):
        # Simular gravação de um Falso Positivo (FP) onde o total_score obtido foi 68% no perfil normal (limiar min 65%)
        self.feedback_mgr.save_feedback(
            video_name="match1.mp4",
            profile_key="normal",
            event_id="ev_fp1",
            label="FP",
            sub_scores={"target_impact": 62.0, "fumikomi_sync": 50.0, "posture": 50.0, "zanshin": 45.0},
            total_score=68.0,
            strike_type="KOTE",
            timestamp="00:03.200"
        )

        base_config = {
            "name": "Treino Geral (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        new_config, summary = self.feedback_mgr.optimize_profile_config("normal", base_config)

        self.assertEqual(summary["status"], "success")
        # O min_total_score deve ter sido elevado acima de 68% (0.68 + 0.02 = 0.70)
        self.assertGreater(new_config["min_total_score"], 0.65)
        self.assertGreaterEqual(new_config["min_total_score"], 0.70)

if __name__ == "__main__":
    unittest.main()
