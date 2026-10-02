"""
Suíte de Testes Automatizados para o Eixo 4:
Aprendizado Ativo (Active Learning), Golden Benchmark, Consenso Arbitral e Assistente LLM.
"""

import unittest
import os
import json
import tempfile
import time

from src.engine.active_learning import (
    UncertaintySampler,
    GoldenBenchmark,
    MultiJudgeConsensus,
    ReviewerTrustManager
)
from src.engine.llm_assistant import KendoLLMAssistant
from src.engine.feedback_manager import FeedbackManager, DEFAULT_CALIBRATION_PROFILES


class TestActiveLearningEixo4(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.queue_file = os.path.join(self.temp_dir.name, "test_queue.json")
        self.golden_file = os.path.join(self.temp_dir.name, "test_golden.json")
        self.trust_file = os.path.join(self.temp_dir.name, "test_trust.json")
        self.feedback_file = os.path.join(self.temp_dir.name, "test_feedback.json")
        self.history_file = os.path.join(self.temp_dir.name, "test_history.json")
        self.profiles_file = os.path.join(self.temp_dir.name, "test_profiles.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. TESTES DO UNCERTAINTY SAMPLER (EIXO 4.1)
    # --------------------------------------------------------------------------
    def test_uncertainty_calculation(self):
        """Valida a fórmula de incerteza: Incerteza(x) = 1.0 - 2 * |P - 0.5|."""
        # P = 0.50 -> Incerteza máxima (1.00)
        self.assertAlmostEqual(UncertaintySampler.calculate_uncertainty(0.50), 1.00, places=3)
        # P = 0.00 ou 1.00 -> Incerteza mínima (0.00)
        self.assertAlmostEqual(UncertaintySampler.calculate_uncertainty(0.00), 0.00, places=3)
        self.assertAlmostEqual(UncertaintySampler.calculate_uncertainty(1.00), 0.00, places=3)
        # P = 0.55 -> Incerteza 0.90
        self.assertAlmostEqual(UncertaintySampler.calculate_uncertainty(0.55), 0.90, places=3)
        # P = 0.45 -> Incerteza 0.90
        self.assertAlmostEqual(UncertaintySampler.calculate_uncertainty(0.45), 0.90, places=3)

    def test_uncertainty_filtering_and_enqueueing(self):
        """Valida que apenas lances na faixa de incerteza (45% a 65%) são enfileirados."""
        sampler = UncertaintySampler(queue_file=self.queue_file)

        # Lance com certeza alta (P = 0.92) -> Não deve enfileirar
        res_high = sampler.evaluate_and_enqueue(
            strike_data={"strike_type": "Men"}, confidence=0.92
        )
        self.assertIsNone(res_high)
        self.assertEqual(len(sampler.get_queue()), 0)

        # Lance com incerteza alta (P = 0.52) -> Deve enfileirar com triage LLM
        res_unc = sampler.evaluate_and_enqueue(
            strike_data={"strike_type": "Men", "sub_scores": {"target_impact": 0.50, "fumikomi_sync": 0.48}},
            confidence=0.52,
            profile_name="normal"
        )
        self.assertIsNotNone(res_unc)
        self.assertEqual(len(sampler.get_queue()), 1)
        self.assertEqual(res_unc["status"], "pending_curation")
        self.assertIn("llm_triage", res_unc)
        self.assertIsNotNone(res_unc["llm_triage"])

        # Resolver item da fila
        item_id = res_unc["id"]
        ok = sampler.resolve_item(item_id, label_approved=True, reviewer_dan=7, notes="Ippon confirmado por Sensei 7º Dan")
        self.assertTrue(ok)
        self.assertEqual(len(sampler.get_queue(status="pending_curation")), 0)
        self.assertEqual(len(sampler.get_queue(status="curated")), 1)

    # --------------------------------------------------------------------------
    # 2. TESTES DO GOLDEN BENCHMARK & REGRESSION PREVENTION (EIXO 4.2)
    # --------------------------------------------------------------------------
    def test_golden_benchmark_evaluation(self):
        """Valida a avaliação do conjunto padrão-ouro e métricas F1 / Acurácia."""
        gb = GoldenBenchmark(dataset_path=self.golden_file)
        samples = gb.load_samples()
        self.assertGreaterEqual(len(samples), 9)

        normal_profile = DEFAULT_CALIBRATION_PROFILES["normal"]
        metrics = gb.evaluate_profile(normal_profile)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertGreater(metrics["accuracy"], 0.70)
        self.assertGreater(metrics["f1"], 0.70)

    def test_golden_benchmark_blocks_regression(self):
        """Valida que recalibrações defeituosas que causem regressão no Golden Benchmark são bloqueadas."""
        gb = GoldenBenchmark(dataset_path=self.golden_file)
        current_profile = DEFAULT_CALIBRATION_PROFILES["normal"]

        # Perfil degradado (exigência absurda de 99% que zera os verdadeiros positivos)
        degraded_profile = {
            "min_total_score": 0.99,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.95, "fumikomi_sync": 0.95, "posture": 0.95, "zanshin": 0.95}
        }

        passed, report = gb.validate_no_regression(degraded_profile, current_profile, tolerance=0.03)
        self.assertFalse(passed)
        self.assertIsNotNone(report.get("block_reason"))

        # Perfil válido ligeiramente ajustado -> Deve passar
        valid_profile = dict(current_profile)
        valid_profile["min_total_score"] = 0.64
        passed_ok, report_ok = gb.validate_no_regression(valid_profile, current_profile, tolerance=0.03)
        self.assertTrue(passed_ok)

    # --------------------------------------------------------------------------
    # 3. TESTES DO CONSENSO MULTI-ÁRBITRO & DIVERGÊNCIA (EIXO 4.3)
    # --------------------------------------------------------------------------
    def test_multi_judge_consensus_unanimity(self):
        """Valida consenso unânime de 3 árbitros (3 bandeiras brancas)."""
        reviews = [
            {"dan": 5, "verdict": "IPPON"},
            {"dan": 6, "verdict": "IPPON"},
            {"dan": "shinpan", "verdict": "IPPON"},
        ]
        cons = MultiJudgeConsensus.consolidate_reviews(reviews)
        self.assertTrue(cons["is_ippon"])
        self.assertEqual(cons["divergence_degree"], 0.0)
        self.assertEqual(cons["majority_ratio"], "3/3")
        self.assertEqual(cons["training_weight"], 1.0)

    def test_multi_judge_consensus_split_decision_2_of_3(self):
        """Valida a regra oficial da FIK (2 de 3 bandeiras com grau de divergência e atenuação de treino)."""
        reviews = [
            {"dan": 4, "verdict": "IPPON"},
            {"dan": 5, "verdict": "IPPON"},
            {"dan": 3, "verdict": "NO_POINT"},
        ]
        cons = MultiJudgeConsensus.consolidate_reviews(reviews)
        self.assertTrue(cons["is_ippon"])
        self.assertGreater(cons["divergence_degree"], 0.0)
        self.assertEqual(cons["majority_ratio"], "2/3")
        self.assertLess(cons["training_weight"], 1.0)

    # --------------------------------------------------------------------------
    # 4. TESTES DE DECAIMENTO TEMPORAL DE REVISORES (EIXO 4.4)
    # --------------------------------------------------------------------------
    def test_reviewer_trust_decay(self):
        """Valida que inatividade temporal e histórico de divergência atenuam o peso do árbitro."""
        mgr = ReviewerTrustManager(storage_file=self.trust_file, half_life_days=100.0)

        # Revisor recém ativo
        now = time.time()
        mgr.record_activity("sensei_yamada", dan_level=7, agreed_with_consensus=True, timestamp=now)
        w_active = mgr.get_effective_weight("sensei_yamada", base_dan=7, current_timestamp=now)
        self.assertAlmostEqual(w_active, 7.0, places=1)

        # Revisor inativo após 100 dias (1 meia-vida -> peso cai para ~50%)
        past_ts = now - (100.0 * 86400.0)
        mgr.record_activity("sensei_inativo", dan_level=6, agreed_with_consensus=True, timestamp=past_ts)
        w_inactive = mgr.get_effective_weight("sensei_inativo", base_dan=6, current_timestamp=now)
        self.assertAlmostEqual(w_inactive, 3.0, delta=0.3)

    # --------------------------------------------------------------------------
    # 5. TESTES DO ASSISTENTE LLM (OFFLINE / MOTOR ESPECIALISTA FIK)
    # --------------------------------------------------------------------------
    def test_llm_assistant_uncertain_strike_analysis(self):
        """Valida a geração de parecer técnico estruturado pelo motor de Kendo."""
        assistant = KendoLLMAssistant() # Sem API key = motor especialista offline garantido
        strike_data = {
            "strike_type": "Men",
            "scores": {"target_impact": 0.85, "fumikomi_sync": 0.30, "posture": 0.70, "zanshin": 0.60},
            "sub_scores": {"target_impact": 0.85, "fumikomi_sync": 0.30, "posture": 0.70, "zanshin": 0.60},
            "hasuji_score": 0.85
        }
        res = assistant.analyze_uncertain_strike(strike_data, profile_name="normal", confidence=0.51)

        self.assertIn("verdict", res)
        self.assertIn("primary_deficiency", res)
        self.assertIn("fik_rule_rationale", res)
        self.assertIn("coaching_advice", res)
        # O ponto mais fraco é Fumikomi (0.30)
        self.assertEqual(res["primary_deficiency"], "KI_KEN_TAI_ICHI")
        self.assertIn("Fumikomi", res["fik_rule_rationale"])

    def test_llm_assistant_movement_labeling(self):
        """Valida a rotulagem cinemática assistida para captura de treinos e vídeos."""
        assistant = KendoLLMAssistant()
        kinematic_summary = {
            "cadence_cpm": 55.0,
            "strike_count": 12,
            "peak_wrist_speed": 1.45,
            "spine_tilt_deg": 6.2
        }
        labeled = assistant.assisted_movement_labeling(kinematic_summary, context_hint="Treino de repetições")
        self.assertIn("detected_modality", labeled)
        self.assertIn("execution_quality", labeled)
        self.assertIn("action_spotting_tags", labeled)
        self.assertEqual(labeled["detected_modality"], "Kirikaeshi")

    # --------------------------------------------------------------------------
    # 6. TESTE DE INTEGRAÇÃO FEEDBACK MANAGER COM EIXO 4
    # --------------------------------------------------------------------------
    def test_feedback_manager_integration_with_eixo4(self):
        """Valida que o FeedbackManager interage com o Golden Benchmark e Uncertainty Sampler."""
        mgr = FeedbackManager(
            dataset_path=self.feedback_file,
            history_path=self.history_file,
            profiles_path=self.profiles_file
        )

        # 1. Salvar feedback incerto (52%) -> Deve gerar item na fila de curadoria ativa
        mgr.save_feedback(
            video_name="match_eixo4.mp4",
            profile_key="normal",
            event_id="ev_unc_1",
            label="TP",
            total_score=52.0,
            sub_scores={"target_impact": 0.52, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.50}
        )
        queue = mgr.get_active_learning_queue()
        self.assertGreaterEqual(len(queue), 1)
        self.assertEqual(queue[0]["status"], "pending_curation")

        # 2. Avaliação de métricas no Golden Benchmark
        gb_metrics = mgr.get_golden_benchmark_metrics("normal")
        self.assertIn("accuracy", gb_metrics)
        self.assertIn("f1", gb_metrics)


if __name__ == "__main__":
    unittest.main()
