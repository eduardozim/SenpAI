"""
Testes do Motor de Conhecimento do SenpAI (auto_trainer) depois da Pendência P1.

O antigo "treinamento automático por tempo" (loop que só dormia, acurácia por fórmula, amostras
aleatórias, ganhos fixos e checkpoint) foi removido; o treino real é o Ciclo de Treino
(src/engine/training_cycle.py). Aqui ficam a base de conhecimento de referência e o pacote de treino.
"""

import os
import json
import unittest

from src.engine import auto_trainer as at
from src.engine.auto_trainer import AutoTrainingEngine, KENDO_KNOWLEDGE_RESOURCES
from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA

LEGACY_SIMULATION_FIELDS = ("initial_accuracy", "current_accuracy", "mastery_level", "sessions_count")


class TestAutoTrainer(unittest.TestCase):
    def setUp(self):
        self.test_kb_path = "config/test_ai_knowledge_base.json"
        self.test_profiles_path = "config/test_calibration_profiles.json"
        self.test_history_path = "data/test_training_history.json"
        self.test_feedback_path = "data/test_feedback_dataset.json"
        for p in [self.test_kb_path, self.test_profiles_path, self.test_history_path, self.test_feedback_path]:
            if os.path.exists(p):
                os.remove(p)
        self.engine = AutoTrainingEngine(
            knowledge_base_path=self.test_kb_path,
            profiles_path=self.test_profiles_path,
            history_path=self.test_history_path,
            feedback_path=self.test_feedback_path,
        )

    def tearDown(self):
        for p in [self.test_kb_path, self.test_profiles_path, self.test_history_path, self.test_feedback_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_simulation_is_gone(self):
        """Nenhuma das peças do treinamento simulado por tempo existe mais."""
        for name in ("run_auto_training", "start_training_thread", "load_checkpoint", "diagnose_latent_need",
                     "get_scope_current_accuracy", "retrain_detection_model", "get_evolution_statistics",
                     "get_modalities_accuracy_summary"):
            self.assertFalse(hasattr(self.engine, name), name)
        self.assertFalse(hasattr(at, "AUTO_TRAINING_SCOPES"))
        self.assertFalse(hasattr(at, "MODALITY_BASE_ACCURACIES"))

    def test_knowledge_base_initialization_without_fake_metrics(self):
        kb = self.engine.load_knowledge_base()
        self.assertIn("sources", kb)
        self.assertEqual(kb["sources"], {})
        self.assertEqual(kb.get("total_web_sources_indexed", 0), 0)
        mods = kb["learned_parameters"]["training_modalities"]
        self.assertEqual(set(mods), set(TRAINING_MODALITIES_METADATA))
        for m in mods.values():
            for field in LEGACY_SIMULATION_FIELDS:
                self.assertNotIn(field, m)
            self.assertTrue(m["principles_learned"])

    def test_legacy_kb_with_simulated_accuracy_is_cleaned(self):
        """Uma base antiga com acurácias inventadas perde esses campos ao ser carregada."""
        kb = self.engine.load_knowledge_base()
        kb["learned_parameters"]["training_modalities"]["suburi"].update(
            {"current_accuracy": 93.2, "initial_accuracy": 38.0, "mastery_level": "Excelente / Shiai", "sessions_count": 25})
        with open(self.test_kb_path, "w", encoding="utf-8") as f:
            json.dump(kb, f)
        clean = AutoTrainingEngine(knowledge_base_path=self.test_kb_path, profiles_path=self.test_profiles_path,
                                   history_path=self.test_history_path, feedback_path=self.test_feedback_path)
        suburi = clean.load_knowledge_base()["learned_parameters"]["training_modalities"]["suburi"]
        for field in LEGACY_SIMULATION_FIELDS:
            self.assertNotIn(field, suburi)

    def test_modalities_reference(self):
        ref = self.engine.get_modalities_reference()
        self.assertEqual(len(ref), len(TRAINING_MODALITIES_METADATA))
        for m in ref:
            self.assertEqual(set(m), {"key", "name", "japanese", "category", "pillar_movement_pct",
                                      "pillar_precision_pct", "pillar_constancy_pct", "cadence_optimal",
                                      "accuracy_pct", "accuracy_display", "accuracy_status"})
        info = self.engine.get_modality_learned_knowledge("suburi")
        for field in ("current_accuracy", "mastery_level", "training_sessions_count", "samples_estimated"):
            self.assertNotIn(field, info)
        self.assertTrue(info["principles_learned"])

    def test_knowledge_sources_by_type(self):
        # Quando a base está zerada
        self.assertEqual(self.engine.get_knowledge_sources_by_type(), {})
        self.assertEqual(self.engine.get_consulted_knowledge_sources(), [])

        # Adicionando uma fonte ao corpus
        kb = self.engine.load_knowledge_base()
        kb.setdefault("sources", {})["fik_regulations"] = dict(KENDO_KNOWLEDGE_RESOURCES["fik_regulations"])
        self.engine.save_knowledge_base(kb)

        counts = self.engine.get_knowledge_sources_by_type()
        self.assertEqual(sum(counts.values()), len(self.engine.get_consulted_knowledge_sources()))
        self.assertEqual(list(counts.values()), sorted(counts.values(), reverse=True))
        self.assertGreaterEqual(sum(counts.values()), 1)
        self.assertGreaterEqual(len(KENDO_KNOWLEDGE_RESOURCES), 2)

    def test_reset_knowledge_base(self):
        kb = self.engine.load_knowledge_base()
        kb["learned_parameters"]["training_modalities"]["suburi"]["principles_learned"].append("Princípio extra")
        self.engine.save_knowledge_base(kb)
        self.engine.reset_knowledge_base()
        fresh = self.engine.load_knowledge_base()["learned_parameters"]["training_modalities"]["suburi"]
        self.assertNotIn("Princípio extra", fresh["principles_learned"])

    def test_export_and_merge_knowledge_data(self):
        """O pacote de treino leva e traz princípios e fontes, sem checkpoint nem acurácias simuladas."""
        kb = self.engine.load_knowledge_base()
        kb["learned_parameters"]["training_modalities"]["suburi"]["principles_learned"].append("Hasuji Estrito no Suburi")
        self.engine.save_knowledge_base(kb)

        exported = self.engine.export_knowledge_data()
        self.assertEqual(set(exported), {"ai_knowledge_base", "exported_at"})

        self.engine.reset_knowledge_base()
        res = self.engine.merge_knowledge_data(exported["ai_knowledge_base"])
        self.assertEqual(res["status"], "success")
        restored = self.engine.load_knowledge_base()["learned_parameters"]["training_modalities"]["suburi"]
        self.assertIn("Hasuji Estrito no Suburi", restored["principles_learned"])
        for field in LEGACY_SIMULATION_FIELDS:
            self.assertNotIn(field, restored)

    def test_merge_ignores_legacy_checkpoint_and_accuracy(self):
        """Pacotes antigos (com checkpoint e acurácias da simulação) ainda importam, sem trazer esses dados."""
        legacy = self.engine.load_knowledge_base()
        legacy["learned_parameters"]["training_modalities"]["kihon"]["current_accuracy"] = 97.0
        res = self.engine.merge_knowledge_data(legacy, imported_checkpoint={"status": "completed_consolidated", "current_accuracy": 97.0})
        self.assertEqual(res["status"], "success")
        kihon = self.engine.load_knowledge_base()["learned_parameters"]["training_modalities"]["kihon"]
        self.assertNotIn("current_accuracy", kihon)
        self.assertFalse(os.path.exists("data/test_auto_training_checkpoint.json"))


if __name__ == "__main__":
    unittest.main()
