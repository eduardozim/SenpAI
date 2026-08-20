"""
Testes Automatizados para Edição por Dan, Retreinamento e Governança de Treinamento no SenpAI.
"""

import os
import json
import unittest
from src.engine.feedback_manager import FeedbackManager, DAN_NAMES

class TestDanTrainingGovernance(unittest.TestCase):
    def setUp(self):
        self.test_dataset_path = "data/test_feedback_dataset.json"
        self.test_history_path = "data/test_training_history.json"
        self.test_profiles_path = "config/test_calibration_profiles.json"

        for p in [self.test_dataset_path, self.test_history_path, self.test_profiles_path]:
            if os.path.exists(p):
                os.remove(p)

        self.mgr = FeedbackManager(
            dataset_path=self.test_dataset_path,
            history_path=self.test_history_path,
            profiles_path=self.test_profiles_path
        )

    def tearDown(self):
        for p in [self.test_dataset_path, self.test_history_path, self.test_profiles_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_save_review_session_with_dan(self):
        """Valida o salvamento de sessão de revisão com Dan, calculando métricas e histórico de auditoria."""
        items = [
            {
                "event_id": "ev_1",
                "label": "TP",
                "strike_type": "MEN",
                "timestamp": "00:01.200",
                "total_score": 82.0,
                "sub_scores": {"target_impact": 85.0, "fumikomi_sync": 80.0, "posture": 80.0, "zanshin": 80.0},
                "is_confirmed": True
            },
            {
                "event_id": "ev_2",
                "label": "FP",
                "strike_type": "KOTE",
                "timestamp": "00:04.500",
                "total_score": 68.0,
                "sub_scores": {"target_impact": 60.0, "fumikomi_sync": 50.0, "posture": 50.0, "zanshin": 45.0},
                "is_edited": True,
                "notes": "Alvo incorreto"
            }
        ]

        current_config = {
            "name": "Treino Geral (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        new_config, record = self.mgr.save_review_session(
            video_name="match_kendo.mp4",
            profile_key="normal",
            reviewer_dan=4,  # Yondan (4º Dan)
            review_items=items,
            current_profile_config=current_config
        )

        self.assertEqual(record["reviewer_dan"], 4)
        self.assertEqual(record["reviewer_dan_name"], "4º Dan (Yondan)")
        self.assertEqual(record["items_count"], 2)

        data = self.mgr.load_feedback()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["reviewer_dan"], 4)
        self.assertEqual(data[0]["reviewer_dan_name"], "4º Dan (Yondan)")
        self.assertIn("review_date", data[0])

    def test_get_training_metrics_dan_distribution(self):
        """Verifica o cálculo de métricas de governança: total de sessões, nível médio Dan e distribuição 1º ao 8º Dan."""
        current_config = {
            "name": "Normal",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        # Sessão 1: 3º Dan (Sandan)
        self.mgr.save_review_session("m1.mp4", "normal", 3, [{"event_id": "e1", "label": "TP", "strike_type": "MEN"}], current_config)
        # Sessão 2: 5º Dan (Godan)
        self.mgr.save_review_session("m2.mp4", "normal", 5, [{"event_id": "e2", "label": "TP", "strike_type": "KOTE"}], current_config)

        metrics = self.mgr.get_training_metrics()

        self.assertEqual(metrics["total_trainings_count"], 2)
        self.assertEqual(metrics["average_dan_level"], 4.0)  # (3 + 5) / 2 = 4.0
        self.assertIn("4.0º Dan", metrics["average_dan_label"])

        dan_dist = metrics["dan_distribution"]
        self.assertEqual(len(dan_dist), 8)  # 1º ao 8º Dan
        
        dan3_entry = next(item for item in dan_dist if item["Dan"] == "3º Dan")
        dan5_entry = next(item for item in dan_dist if item["Dan"] == "5º Dan")
        
        self.assertEqual(dan3_entry["Quantidade Treinamentos"], 1)
        self.assertEqual(dan5_entry["Quantidade Treinamentos"], 1)

    def test_export_and_import_training_package(self):
        """Testa a exportação e importação de pacotes de treinamento JSON com preservação de datas e graduação Dan."""
        current_config = {
            "name": "Normal",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        self.mgr.save_review_session("m1.mp4", "normal", 6, [{"event_id": "e1", "label": "TP", "strike_type": "MEN", "timestamp": "00:01.000"}], current_config)

        pkg = self.mgr.export_training_package()
        self.assertEqual(pkg["system_name"], "SenpAI")
        self.assertIn("exported_at", pkg)
        self.assertEqual(len(pkg["review_items"]), 1)
        self.assertEqual(pkg["review_items"][0]["reviewer_dan"], 6)

        # Resetar gerenciador e importar o pacote
        self.mgr.reset_all_training_data()
        self.assertEqual(self.mgr.get_training_metrics()["total_trainings_count"], 0)

        import_res = self.mgr.import_training_package(pkg)
        self.assertEqual(import_res["status"], "success")
        self.assertEqual(import_res["imported_items_count"], 1)

        metrics_post = self.mgr.get_training_metrics()
        self.assertEqual(metrics_post["total_trainings_count"], 1)
        self.assertEqual(metrics_post["average_dan_level"], 6.0)

    def test_reset_all_training_data(self):
        """Valida o reset completo do histórico de treinamento e dataset restaurando o sistema ao estágio inicial."""
        self.mgr.save_feedback("v.mp4", "normal", "ev1", "TP", strike_type="MEN", reviewer_dan=2)
        self.mgr.reset_all_training_data()

        self.assertEqual(len(self.mgr.load_feedback()), 0)
        self.assertEqual(len(self.mgr.load_history()), 0)

    def test_import_raw_list_json(self):
        """Valida a importação de listas JSON brutas de feedbacks e recalibração automática do Dan médio."""
        raw_list = [
            {
                "id": "item_1",
                "video_name": "match_test.mp4",
                "profile_key": "normal",
                "label": "TP",
                "strike_type": "MEN",
                "reviewer_dan": 5
            }
        ]
        res = self.mgr.import_training_package(raw_list)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["new_items_added"], 1)
        self.assertEqual(self.mgr.get_training_metrics()["total_trainings_count"], 1)
        self.assertEqual(self.mgr.get_training_metrics()["average_dan_level"], 5.0)

if __name__ == "__main__":
    unittest.main()
