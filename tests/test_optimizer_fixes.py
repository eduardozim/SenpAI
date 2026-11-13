"""
Testes das correções do otimizador de pesos e limiares (Pendência I4 → P5.1 a P5.4).
"""

import unittest

from src.engine.calibrator import decide_strike, CalibrationEngine, FIK_STRICT_ELIMINATORY_PILLARS, LENIENT_ELIMINATORY_PILLARS, DEFAULT_ELIMINATORY_PILLARS
from src.engine.mathematical_calibrator import BayesianCalibrationOptimizer, DEFAULT_WEIGHTS_BY_STRIKE_TYPE

BASE = {
    "name": "Teste",
    "min_total_score": 0.74,
    "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
    "sub_thresholds": {"target_impact": 0.50, "fumikomi_sync": 0.40, "posture": 0.40, "zanshin": 0.35, "hasuji": 0.40},
}


def _fb(label, t, f, p, z, video="v1.mp4", hasuji=None, strike="MEN"):
    sub = {"target_impact": t, "fumikomi_sync": f, "posture": p, "zanshin": z}
    if hasuji is not None:
        sub["hasuji"] = hasuji
    return {"label": label, "sub_scores": sub, "video_name": video, "strike_type": strike, "reviewer_dan": 5}


class TestSameDecisionAsProduction(unittest.TestCase):
    def test_engine_and_shared_decision_agree(self):
        engine = CalibrationEngine(profile_name="normal")
        for args in [(0.9, 0.8, 0.7, 0.6, "MEN", 0.9), (0.55, 0.2, 0.9, 0.1, "KOTE", None), (0.35, 0.9, 0.9, 0.9, "DO", 0.8)]:
            t, f, p, z, st, h = args
            ev = engine.evaluate_strike(t, f, p, z, strike_type=st, hasuji_score=h)
            d = decide_strike(t, f, p, z, config=engine.active_config, strike_type=st, hasuji_score=h)
            self.assertEqual(ev["is_valid"], d["is_valid"])
            self.assertAlmostEqual(ev["total_score"], round(d["total_score"] * 100, 1), places=1)

    def test_optimizer_loss_counts_production_decisions(self):
        opt = BayesianCalibrationOptimizer()
        cfg = dict(BASE, min_total_score=0.50, pilares_eliminatorios=LENIENT_ELIMINATORY_PILLARS)
        samples = opt.preprocess_feedbacks([
            _fb("TP", 90, 85, 80, 75, hasuji=90),
            _fb("FP", 92, 10, 90, 10),   # sem fumikomi/zanshin: na regra permissiva vira ponto → FP
        ])
        _, metrics = opt.config_loss(cfg, samples)
        self.assertEqual((metrics["tp_weight"] > 0, metrics["fp_weight"] > 0), (True, True))

    def test_eliminatory_pillars_option(self):
        lenient = dict(BASE, min_total_score=0.5, pilares_eliminatorios=LENIENT_ELIMINATORY_PILLARS)
        strict = dict(BASE, min_total_score=0.5)  # padrão
        args = (0.9, 0.9, 0.9, 0.1)  # sem zanshin
        self.assertEqual(DEFAULT_ELIMINATORY_PILLARS, FIK_STRICT_ELIMINATORY_PILLARS)  # decisão de 05/11/2026
        self.assertTrue(decide_strike(*args, config=lenient)["is_valid"])
        self.assertFalse(decide_strike(*args, config=strict)["is_valid"])
        self.assertIn("SEM_ZANSHIN", decide_strike(*args, config=lenient)["failed"])


class TestOptimizerFixes(unittest.TestCase):
    def test_fn_without_scores_is_discarded_not_invented(self):
        opt = BayesianCalibrationOptimizer()
        samples = opt.preprocess_feedbacks([_fb("TP", 90, 85, 80, 75), {"label": "FN", "sub_scores": {}, "video_name": "v1.mp4"}])
        self.assertEqual(len(samples), 1)
        self.assertEqual(opt.last_skipped_without_scores, 1)

    def test_weights_go_where_production_reads_them(self):
        opt = BayesianCalibrationOptimizer()
        new_cfg, summary = opt.optimize(dict(BASE), [_fb("TP", 90, 85, 80, 75), _fb("FP", 40, 30, 40, 20)])
        self.assertEqual(summary["method"], "differential_evolution")
        for st in DEFAULT_WEIGHTS_BY_STRIKE_TYPE:
            self.assertEqual(new_cfg["weights_by_strike_type"][st], new_cfg["weights"])
            self.assertAlmostEqual(sum(new_cfg["weights"].values()), 1.0, places=2)

    def test_learns_from_data_and_reduces_loss(self):
        """Pontos reais com nota ~0,66 estavam sendo recusados pelo mínimo de 0,74: o otimizador baixa o mínimo."""
        opt = BayesianCalibrationOptimizer()
        feedbacks = []
        for v in range(4):
            video = f"luta_{v}.mp4"
            feedbacks += [_fb("TP", 72, 66, 64, 60, video), _fb("TP", 70, 64, 66, 62, video),
                          _fb("FP", 45, 40, 45, 30, video), _fb("FP", 35, 60, 60, 50, video)]
        new_cfg, summary = opt.optimize(dict(BASE), feedbacks)
        self.assertLess(summary["final_loss"], summary["initial_loss"])
        self.assertTrue(summary["validacao"]["aceito"])
        for fb in feedbacks:
            s = fb["sub_scores"]
            d = decide_strike(s["target_impact"] / 100, s["fumikomi_sync"] / 100, s["posture"] / 100, s["zanshin"] / 100, config=new_cfg)
            self.assertEqual(d["is_valid"], fb["label"] == "TP")

    def test_no_worst_fp_rule(self):
        """Um FP já recusado pela decisão de produção não empurra o limiar (a antiga regra fazia "pior FP + 0,02")."""
        opt = BayesianCalibrationOptimizer()
        new_cfg, _ = opt.optimize(dict(BASE, min_total_score=0.65), [_fb("FP", 62, 50, 50, 45, strike="KOTE") | {"total_score": 68.0}])
        self.assertLess(new_cfg["min_total_score"], 0.70)
        self.assertFalse(decide_strike(0.62, 0.50, 0.50, 0.45, config=new_cfg, strike_type="KOTE")["is_valid"])

    def test_rejects_when_validation_videos_get_worse(self):
        """Nos vídeos de treino a nota mínima "deveria" cair; nos de validação isso cria falsos pontos → mantém a atual."""
        opt = BayesianCalibrationOptimizer(l2_reg_weight=0.0)
        train_videos = [f"t{i}.mp4" for i in range(6)]
        val_candidates = [f"x{i}.mp4" for i in range(6)]
        feedbacks = []
        for v in train_videos:
            feedbacks.append(_fb("TP", 66, 60, 60, 60, v))
        for v in val_candidates:
            feedbacks.append(_fb("FP", 66, 60, 60, 60, v))
        train, val = opt.split_by_video(opt.preprocess_feedbacks(feedbacks))
        self.assertTrue(val)
        self.assertFalse({s["video"] for s in train} & {s["video"] for s in val})
        cfg, summary = opt.optimize(dict(BASE), feedbacks)
        if summary.get("status") == "rejected_validation":
            self.assertEqual(cfg["min_total_score"], BASE["min_total_score"])
            self.assertFalse(summary["validacao"]["aceito"])
        else:
            self.assertLessEqual(summary["validacao"]["perda_nova"], summary["validacao"]["perda_atual"])


if __name__ == "__main__":
    unittest.main()
