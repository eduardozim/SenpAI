"""
Suíte de Testes Automatizados para o Eixo 1:
Otimização Matemática dos Pesos, Platt Scaling, Concept Drift e Ponderação por Golpe.
Validação formal dos requisitos de REFERENCIA_MOTOR_CALIBRACAO_E_APRENDIZADO.md.
"""

import unittest
import os
import json
import tempfile
import time
import numpy as np

from src.engine.mathematical_calibrator import (
    ProbabilisticPlattCalibrator,
    ConceptDriftDetector,
    BayesianCalibrationOptimizer,
    DEFAULT_WEIGHTS_BY_STRIKE_TYPE,
    SHINPAN_REVIEWER_WEIGHT,
    KYU_ANON_WEIGHT,
    ASYMMETRIC_FP_COST,
    ASYMMETRIC_FN_COST
)
from src.engine.calibrator import CalibrationEngine
from src.engine.feedback_manager import FeedbackManager, DEFAULT_CALIBRATION_PROFILES


class TestMathematicalCalibrationEixo1(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.profiles_file = os.path.join(self.temp_dir.name, "test_profiles.json")
        self.feedback_file = os.path.join(self.temp_dir.name, "test_feedback.json")
        self.history_file = os.path.join(self.temp_dir.name, "test_history.json")

        # Salva um perfil de teste completo
        with open(self.profiles_file, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CALIBRATION_PROFILES, f, indent=2)

    def tearDown(self):
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # 1. TESTES DO CLASSIFICADOR PROBABILÍSTICO (PLATT SCALING - EIXO 1.2)
    # --------------------------------------------------------------------------
    def test_platt_scaling_probabilities(self):
        """Valida que a curva sigmoidal de Platt gera probabilidades estritamente crescentes em [0, 1]."""
        platt = ProbabilisticPlattCalibrator(a_param=12.0, b_param=-8.0)
        p_low = platt.predict_proba(0.20)
        p_mid = platt.predict_proba(0.667)
        p_high = platt.predict_proba(0.95)

        self.assertGreaterEqual(p_low, 0.0)
        self.assertLessEqual(p_high, 1.0)
        self.assertLess(p_low, p_mid)
        self.assertLess(p_mid, p_high)
        # s = 0.667 com A=12, B=-8 resulta em z ≈ 0 -> P ≈ 0.50
        self.assertAlmostEqual(p_mid, 0.50, delta=0.05)

    def test_platt_scaling_fit_and_serialization(self):
        """Valida o ajuste por máxima verossimilhança e serialização dict."""
        platt = ProbabilisticPlattCalibrator()
        # Amostras sintéticas onde scores >= 0.70 são positivos
        scores = [0.45, 0.52, 0.58, 0.62, 0.72, 0.78, 0.85, 0.92]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        weights = [1.0] * len(scores)

        fit_res = platt.fit(scores, labels, weights)
        self.assertTrue(fit_res.get("fitted"))
        self.assertGreater(platt.predict_proba(0.85), platt.predict_proba(0.50))

        # Teste de serialização
        p_dict = platt.to_dict()
        self.assertIn("a", p_dict)
        self.assertIn("b", p_dict)
        platt_restored = ProbabilisticPlattCalibrator.from_dict(p_dict)
        self.assertAlmostEqual(platt.a, platt_restored.a, places=3)
        self.assertAlmostEqual(platt.b, platt_restored.b, places=3)

    # --------------------------------------------------------------------------
    # 2. TESTES DE CONCEPT DRIFT E DERIVA TEMPORAL (EIXO 1.3)
    # --------------------------------------------------------------------------
    def test_concept_drift_stable_distribution(self):
        """Valida que distribuições idênticas não acionam alerta de Concept Drift (p-valor alto)."""
        detector = ConceptDriftDetector(p_value_threshold=0.05, min_samples_required=15)
        np.random.seed(42)
        base = list(np.random.normal(0.70, 0.08, 30))
        recent = list(np.random.normal(0.70, 0.08, 30))

        res = detector.detect_drift(recent, base)
        self.assertFalse(res["drift_detected"])
        self.assertEqual(res["status"], "stable")
        self.assertGreater(res["p_value"], 0.05)

    def test_concept_drift_detected_on_shift(self):
        """Valida que uma mudança acentuada na distribuição de scores dispara alerta (KS test bilateral)."""
        detector = ConceptDriftDetector(p_value_threshold=0.05, min_samples_required=15)
        np.random.seed(42)
        base = list(np.random.normal(0.80, 0.04, 30))
        recent = list(np.random.normal(0.50, 0.04, 30))  # Queda substancial nos scores

        res = detector.detect_drift(recent, base)
        self.assertTrue(res["drift_detected"])
        self.assertEqual(res["status"], "drift_alert")
        self.assertLess(res["p_value"], 0.01)
        self.assertLess(res["mean_shift"], -0.20)

    def test_temporal_exponential_decay(self):
        """Valida a fórmula de decaimento exponencial W(t) = exp(-lambda * delta_t)."""
        optimizer = BayesianCalibrationOptimizer(half_life_days=30.0)
        now = time.time()

        # Feedback de hoje: sem decaimento (peso 1.0)
        w_now = optimizer.calculate_decay_weight(now, now=now)
        self.assertAlmostEqual(w_now, 1.0, places=3)

        # Feedback de 30 dias atrás: peso reduzido para ~0.50 (half-life)
        ts_30_days_ago = now - (30 * 86400)
        w_30d = optimizer.calculate_decay_weight(ts_30_days_ago, now=now)
        self.assertAlmostEqual(w_30d, 0.50, delta=0.03)

        # Feedback de 60 dias atrás: peso reduzido para ~0.25
        ts_60_days_ago = now - (60 * 86400)
        w_60d = optimizer.calculate_decay_weight(ts_60_days_ago, now=now)
        self.assertAlmostEqual(w_60d, 0.25, delta=0.03)

    # --------------------------------------------------------------------------
    # 3. TESTES DO OTIMIZADOR BAYESIANO / NUMÉRICO E PERDA ASSIMÉTRICA (EIXO 1.1)
    # --------------------------------------------------------------------------
    def test_reviewer_weight_governance(self):
        """Valida hierarquia de autoridade dos revisores (Shinpan=4.5, Dan=1-8, Kyu=0.8)."""
        optimizer = BayesianCalibrationOptimizer()

        self.assertEqual(optimizer.get_reviewer_weight({"is_shinpan_decision": True}), SHINPAN_REVIEWER_WEIGHT)
        self.assertEqual(optimizer.get_reviewer_weight({"reviewer_dan": "Shinpan Oficial"}), SHINPAN_REVIEWER_WEIGHT)
        self.assertEqual(optimizer.get_reviewer_weight({"reviewer_dan": 7}), 7.0)
        self.assertEqual(optimizer.get_reviewer_weight({"reviewer_dan": "3º Dan"}), 3.0)
        self.assertEqual(optimizer.get_reviewer_weight({"reviewer_dan": "1º Kyu"}), KYU_ANON_WEIGHT)
        self.assertEqual(optimizer.get_reviewer_weight({}), KYU_ANON_WEIGHT)

    def test_asymmetric_loss_penalty(self):
        """Valida que Falsos Positivos têm custo 3x superior aos Falsos Negativos (Shiai FIK)."""
        optimizer = BayesianCalibrationOptimizer(asymmetric_fp_cost=3.0, asymmetric_fn_cost=1.0)
        weights = {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15}
        sub_th = {"target_impact": 0.50, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.50}

        # Cenário 1: 1 FP isolado
        fp_sample = [{
            "y": 0, "target_impact": 0.90, "fumikomi_sync": 0.90, "posture": 0.90, "zanshin": 0.90,
            "weight": 1.0
        }]
        loss_fp, _ = optimizer.evaluate_loss(weights, 0.65, sub_th, fp_sample)

        # Cenário 2: 1 FN isolado
        fn_sample = [{
            "y": 1, "target_impact": 0.40, "fumikomi_sync": 0.40, "posture": 0.40, "zanshin": 0.40,
            "weight": 1.0
        }]
        loss_fn, _ = optimizer.evaluate_loss(weights, 0.65, sub_th, fn_sample)

        # Perda de FP deve ser exatamente 3x a perda de FN
        self.assertAlmostEqual(loss_fp, 3.0 * loss_fn, places=3)

    def test_optimization_constraints_and_bounds(self):
        """Valida que a otimização respeita soma(weights)=1.0, cada peso >= 0.10 e bounds dos limiares."""
        optimizer = BayesianCalibrationOptimizer()
        current_cfg = {
            "name": "Normal Test",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        # Conjunto sintético de feedbacks com marcações de TP e FP
        feedbacks = [
            {"label": "TP", "sub_scores": {"target_impact": 85, "fumikomi_sync": 80, "posture": 80, "zanshin": 75}, "reviewer_dan": 5},
            {"label": "TP", "sub_scores": {"target_impact": 90, "fumikomi_sync": 85, "posture": 75, "zanshin": 80}, "reviewer_dan": 6},
            {"label": "FP", "sub_scores": {"target_impact": 65, "fumikomi_sync": 50, "posture": 40, "zanshin": 30}, "reviewer_dan": 4},
            {"label": "FP", "sub_scores": {"target_impact": 55, "fumikomi_sync": 45, "posture": 45, "zanshin": 25}, "is_shinpan_decision": True},
            {"label": "FN", "sub_scores": {"target_impact": 75, "fumikomi_sync": 70, "posture": 70, "zanshin": 65}, "reviewer_dan": 3}
        ]

        new_cfg, summary = optimizer.optimize(current_cfg, feedbacks)
        self.assertIn("weights", new_cfg)
        opt_weights = new_cfg["weights"]

        # Restrição 1: soma dos pesos deve ser 1.0 (com tolerância de arredondamento)
        total_w = sum(opt_weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=2)

        # Restrição 2: cada peso individual >= 0.10
        for k, v in opt_weights.items():
            self.assertGreaterEqual(v, 0.099, f"Peso {k} inferior ao mínimo de 0.10: {v}")

        # Restrição 3: min_total_score entre 0.50 e 0.90
        self.assertGreaterEqual(new_cfg["min_total_score"], 0.50)
        self.assertLessEqual(new_cfg["min_total_score"], 0.90)

        # Restrição 4: sub_thresholds entre 0.20 e 0.85
        for sk, sv in new_cfg["sub_thresholds"].items():
            self.assertGreaterEqual(sv, 0.20)
            self.assertLessEqual(sv, 0.85)

    # --------------------------------------------------------------------------
    # 4. TESTES DE PESOS POR TIPO DE GOLPE (EIXO 1.4) E CALIBRATION ENGINE
    # --------------------------------------------------------------------------
    def test_weights_by_strike_type_routing(self):
        """Valida que evaluate_strike utiliza os pesos corretos de cada Waza (Men, Kote, Do, Tsuki)."""
        engine = CalibrationEngine(config_path=self.profiles_file, profile_name="normal")

        # Avaliação de um Men
        eval_men = engine.evaluate_strike(
            target_score=0.80, fumikomi_score=0.75, posture_score=0.70, zanshin_score=0.60,
            strike_type="MEN"
        )
        self.assertEqual(eval_men["strike_type"], "MEN")
        self.assertIn("probability", eval_men)
        self.assertIn("probability_pct", eval_men)
        self.assertGreaterEqual(eval_men["probability"], 0.0)
        self.assertLessEqual(eval_men["probability"], 1.0)

        # Avaliação de Tsuki com mesmo score tem peso diferente (Hasuji/colinearidade crítico no alvo)
        eval_tsuki = engine.evaluate_strike(
            target_score=0.80, fumikomi_score=0.75, posture_score=0.70, zanshin_score=0.60,
            strike_type="TSUKI"
        )
        self.assertEqual(eval_tsuki["strike_type"], "TSUKI")
        # No Tsuki, target_impact tem peso 0.50 vs 0.35 no Men
        self.assertNotEqual(eval_men["weights_used"]["target_impact"], eval_tsuki["weights_used"]["target_impact"])

        # Compatibilidade com chamadas sem strike_type (fallback suave para MEN)
        eval_legacy = engine.evaluate_strike(0.80, 0.75, 0.70, 0.60)
        self.assertIn("total_score", eval_legacy)
        self.assertIn("is_valid", eval_legacy)

    # --------------------------------------------------------------------------
    # 5. TESTES DE INTEGRAÇÃO COM FEEDBACK MANAGER
    # --------------------------------------------------------------------------
    def test_feedback_manager_optimize_integration(self):
        """Valida que optimize_profile_config do FeedbackManager aciona a otimização matemática com sucesso."""
        f_mgr = FeedbackManager(
            dataset_path=self.feedback_file,
            history_path=self.history_file,
            profiles_path=self.profiles_file
        )

        # Injeta feedbacks de treino
        now = time.time()
        test_feedbacks = [
            {"id": "fb1", "profile_key": "normal", "label": "TP", "sub_scores": {"target_impact": 88, "fumikomi_sync": 84, "posture": 80, "zanshin": 76}, "reviewer_dan": 6, "timestamp": now},
            {"id": "fb2", "profile_key": "normal", "label": "TP", "sub_scores": {"target_impact": 92, "fumikomi_sync": 86, "posture": 78, "zanshin": 82}, "reviewer_dan": 7, "timestamp": now},
            {"id": "fb3", "profile_key": "normal", "label": "FP", "sub_scores": {"target_impact": 62, "fumikomi_sync": 48, "posture": 40, "zanshin": 28}, "is_shinpan_decision": True, "timestamp": now},
            {"id": "fb4", "profile_key": "normal", "label": "FN", "sub_scores": {"target_impact": 78, "fumikomi_sync": 74, "posture": 72, "zanshin": 68}, "reviewer_dan": 5, "timestamp": now}
        ]
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(test_feedbacks, f)

        cur_cfg = DEFAULT_CALIBRATION_PROFILES["normal"]
        new_cfg, stats = f_mgr.optimize_profile_config("normal", cur_cfg)

        self.assertEqual(stats["status"], "success")
        self.assertIn("optimization_method", stats)
        self.assertIn("platt_scaling", new_cfg)
        self.assertIn("weights_by_strike_type", new_cfg)
        self.assertIn("last_calibrated_at", new_cfg)


if __name__ == "__main__":
    unittest.main()
