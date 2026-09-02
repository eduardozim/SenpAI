"""
Testes Automatizados para o Motor de Treinamento Automático por IA, Retreinamento de Modelo e Painel de Evolução.
"""

import os
import json
import time
import unittest
from src.engine.auto_trainer import AutoTrainingEngine, AUTO_TRAINING_SCOPES, KENDO_KNOWLEDGE_RESOURCES
from src.engine.feedback_manager import FeedbackManager

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
            feedback_path=self.test_feedback_path
        )

    def tearDown(self):
        for p in [self.test_kb_path, self.test_profiles_path, self.test_history_path, self.test_feedback_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_knowledge_base_initialization(self):
        """Valida a inicialização automática e integridade da base de conhecimento de IA de Kendo."""
        kb = self.engine.load_knowledge_base()
        self.assertIn("version", kb)
        self.assertIn("sources", kb)
        self.assertIn("learned_parameters", kb)
        self.assertGreaterEqual(len(kb["sources"]), 2)
        self.assertIn("fik_regulations", kb["sources"])

    def test_diagnose_latent_need_empty_state(self):
        """Valida o diagnóstico de necessidade mais latente em estado inicial do sistema."""
        diag = self.engine.diagnose_latent_need()
        self.assertIn("chosen_scope", diag)
        self.assertIn("scope_name", diag)
        self.assertIn("diagnosis_reasons", diag)
        self.assertGreaterEqual(len(diag["diagnosis_reasons"]), 1)

    def test_diagnose_latent_need_with_false_positives(self):
        """Valida o diagnóstico quando há acúmulo de Falsos Positivos elegendo Shiai/Arbitragem."""
        fps = [
            {"event_id": f"fp_{i}", "label": "FP", "category": "INVALID_HIT", "strike_type": "MEN", "timestamp": "00:01.000"}
            for i in range(8)
        ]
        with open(self.test_feedback_path, "w", encoding="utf-8") as f:
            json.dump(fps, f, indent=2)

        diag = self.engine.diagnose_latent_need()
        self.assertEqual(diag["chosen_scope"], "recorded_shiai")
        self.assertTrue(any("Falsos Positivos" in r for r in diag["diagnosis_reasons"]))

    def test_retrain_detection_model(self):
        """Valida o retreinamento efetivo do modelo de detecção de golpes e calibração de perfis."""
        sources = [
            {"id": "fik_regulations", "title": "FIK Regulations", "type": "Regulamento FIK", "focus": "Yuko-Datotsu"}
        ]
        res = self.engine.retrain_detection_model(
            effective_scope="recorded_shiai",
            sources_consulted=sources,
            intensity="padrao"
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("normal", res["profiles_retrained"])
        self.assertGreaterEqual(len(res["improvements"]), 1)

        # Verificar se os perfis foram gravados em disco com novos pesos
        profiles = self.engine.calibrator.get_all_profiles()
        self.assertIn("normal", profiles)
        self.assertEqual(profiles["normal"]["weights"]["target_impact"], 0.40)

    def test_run_auto_training_quick_execution(self):
        """Valida a execução de um ciclo de auto-treinamento rápido respeitando o tempo e retreinando o modelo."""
        callbacks_received = []

        def on_progress(p_data):
            callbacks_received.append(p_data)

        # Duração ultrarrápida (0.05 min ~ 3-5s para teste)
        result = self.engine.run_auto_training(
            scope_key="latent_need",
            duration_minutes=0.08,
            intensity="rapido",
            include_video=True,
            include_text_guidelines=True,
            progress_callback=on_progress
        )

        self.assertIn(result["status"], ["success", "stopped_early"])
        self.assertGreater(result["final_accuracy_pct"], result["initial_accuracy_pct"])
        self.assertGreaterEqual(len(result["sources_consulted"]), 1)
        self.assertGreaterEqual(len(result["improvements_summary"]), 1)
        self.assertGreater(len(callbacks_received), 0)
        self.assertIn("retrain_summary", result)

        # Validar persistência no histórico de governança
        history = self.engine.feedback_mgr.load_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["reviewer_dan"], 0)
        self.assertTrue(history[0].get("is_auto_training", False))
        self.assertEqual(history[0]["optimization_summary"]["mode"], "auto_training_ai")

    def test_get_evolution_statistics_and_sources(self):
        """Valida a geração de estatísticas consolidadas e consulta de corpus para o Painel de Evolução."""
        # Executar um ciclo de auto-treinamento para popular estatísticas
        self.engine.run_auto_training(
            scope_key="recorded_shiai",
            duration_minutes=0.08,
            intensity="rapido"
        )

        stats = self.engine.get_evolution_statistics()
        self.assertEqual(stats["total_auto_trainings"], 1)
        self.assertGreater(stats["total_duration_seconds"], 0)
        self.assertIn("s", stats["total_duration_formatted"])
        self.assertGreaterEqual(stats["average_accuracy_pct"], 75.0)
        self.assertGreaterEqual(len(stats["accuracy_timeline"]), 1)
        self.assertGreaterEqual(stats["total_sources_indexed"], 2)

        # Validar corpus de fontes consultadas
        sources_list = self.engine.get_consulted_knowledge_sources()
        self.assertGreaterEqual(len(sources_list), 2)
        self.assertTrue(any(s["id"] == "fik_regulations" for s in sources_list))

    def test_run_auto_training_for_modalities(self):
        """Valida o treinamento específico para as 14 modalidades pedagógicas de Kendo."""
        result = self.engine.run_auto_training(
            scope_key="all_14_modalities",
            duration_minutes=0.08,
            intensity="padrao"
        )
        self.assertEqual(result["scope_key"], "all_14_modalities")
        self.assertTrue(any("14 Modalidades" in imp for imp in result["improvements_summary"]))

    def test_modalities_accuracy_summary(self):
        """Valida a geração do sumário de acurácia para todas as 14 modalidades de aprendizado de Kendo."""
        summary = self.engine.get_modalities_accuracy_summary()
        self.assertEqual(len(summary), 14)
        
        # Validar estrutura de cada modalidade
        for mod in summary:
            self.assertIn("key", mod)
            self.assertIn("name", mod)
            self.assertIn("japanese", mod)
            self.assertIn("category", mod)
            self.assertGreaterEqual(mod["current_accuracy"], 70.0)
            self.assertLessEqual(mod["current_accuracy"], 100.0)
            self.assertIn("gain_formatted", mod)
            self.assertIn(mod["status"], ["Excelente", "Calibrado", "Otimizado"])
            self.assertIn("pillar_movement_pct", mod)
            self.assertIn("pillar_precision_pct", mod)
            self.assertIn("pillar_constancy_pct", mod)
            self.assertIn("cadence_optimal", mod)

        # Validar que get_evolution_statistics inclui o sumário e média
        stats = self.engine.get_evolution_statistics()
        self.assertIn("modalities_accuracy_summary", stats)
        self.assertEqual(len(stats["modalities_accuracy_summary"]), 14)
        self.assertIn("average_modality_accuracy_pct", stats)
        self.assertGreaterEqual(stats["average_modality_accuracy_pct"], 80.0)

    def test_auto_training_cooperative_stop(self):
        """Valida a parada graciosa e salvamento parcial do treinamento sob interrupção manual."""
        self.engine._stop_requested = True
        result = self.engine.run_auto_training(
            scope_key="recorded_shiai",
            duration_minutes=0.5
        )
        self.assertIn(result["status"], ["stopped_early", "success"])


if __name__ == "__main__":
    unittest.main()
