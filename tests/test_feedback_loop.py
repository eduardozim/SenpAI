"""
Testes Automatizados para Gerenciamento de Feedback e Aprendizagem por Reforço no SenpAI.
"""

import os
import json
import unittest
from src.engine.feedback_manager import FeedbackManager
from src.engine.calibrator import CalibrationEngine

class TestFeedbackLoop(unittest.TestCase):
    def setUp(self):
        self.test_dataset_path = "data/test_feedback_dataset.json"
        self.test_history_path = "data/test_training_history.json"
        self.test_profiles_path = "config/test_calibration_profiles.json"
        self.test_shinpan_path = "data/test_shinpan_reviewed_videos.json"
        for p in [self.test_dataset_path, self.test_history_path, self.test_profiles_path, self.test_shinpan_path]:
            if os.path.exists(p):
                os.remove(p)
        self.feedback_mgr = FeedbackManager(
            dataset_path=self.test_dataset_path,
            history_path=self.test_history_path,
            profiles_path=self.test_profiles_path,
            shinpan_registry_path=self.test_shinpan_path
        )

    def tearDown(self):
        for p in [self.test_dataset_path, self.test_history_path, self.test_profiles_path, self.test_shinpan_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_save_and_load_feedback(self):
        """Valida o registro de feedback (True Positive), persistência em JSON e cálculo de precisão."""
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
        """Verifica a otimização adaptativa por aprendizagem por reforço sobre Falsos Positivos (ajuste de min_total_score)."""
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
        # Com a decisão de produção (pesos do KOTE), o FP tem que ficar rejeitado pela nova calibração —
        # sem a antiga regra "pior FP + 0,02", que empurrava o limiar por cima de uma nota informada (68%)
        # diferente da nota que a análise de fato calcula.
        from src.engine.calibrator import decide_strike
        decision = decide_strike(0.62, 0.50, 0.50, 0.45, config=new_config, strike_type="KOTE")
        self.assertFalse(decision["is_valid"])
        self.assertLessEqual(new_config["min_total_score"], 0.80)  # sem salto arbitrário do limiar

if __name__ == "__main__":
    unittest.main()
