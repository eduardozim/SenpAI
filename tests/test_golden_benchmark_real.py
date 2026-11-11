"""
Testes do Golden Benchmark com lutas reais (Pendência I5) e da fonte única de métricas.
O golden é o GoldenBenchmark de src/engine/active_learning.py; as métricas vêm de src/engine/weak_labels.py.
"""

import os
import json
import tempfile
import unittest
from unittest import mock

from src.engine import weak_labels as wl
from src.engine.active_learning import GoldenBenchmark, MIN_REAL_SAMPLES_FOR_GATE, REAL_MATCH_CATEGORY, is_golden_video
from src.analytics.event_spotter import StrikeEvent
from src.utils import feature_cache

COMPLETE_ENTRY = {"aka": "A", "shiro": "S", "pontos_completos": True,
                  "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:01.3"}]}


def _ev(frame, label, valid, kenshi="KENSHI_AKA", sub=None):
    return {"impact_frame": frame, "attacker_id": kenshi, "weak_label": label, "predicted_valid": valid,
            "time_sec": frame / 30.0, "strike_type": "MEN",
            "sub_scores": sub if sub is not None else {"target_impact": 80.0, "fumikomi_sync": 70.0, "posture": 60.0, "zanshin": 50.0, "hasuji": 90.0}}


def _report(items, complete=True, points=None):
    points = points if points is not None else [
        {"index": 0, "kenshi_id": "KENSHI_AKA", "technique": "MEN", "time_sec": 1.0, "is_strike": True},
        {"index": 1, "kenshi_id": "KENSHI_SHIRO", "technique": "KOTE", "time_sec": 9.0, "is_strike": True},
    ]
    return {
        "video_name": "luta.mp4",
        "pontos_completos": complete,
        "official_points": points,
        "events": {"items": items, "official_strike_points": sum(1 for p in points if p["is_strike"])},
        "candidates": {"items": [{}] * 5, "counts": {wl.POSITIVE: 2}},
    }


def _golden(tmp):
    return GoldenBenchmark(dataset_path=os.path.join(tmp, "golden_dataset.json"))


class TestUnifiedMetrics(unittest.TestCase):
    def test_precision_recall_f1(self):
        self.assertEqual(wl.precision_recall_f1(3, 1, 1), {"precisao": 0.75, "recall": 0.75, "f1": 0.75})
        self.assertEqual(wl.precision_recall_f1(0, 2, 2), {"precisao": 0.0, "recall": 0.0, "f1": 0.0})
        self.assertEqual(wl.precision_recall_f1(0, 0, 0), {"precisao": None, "recall": None, "f1": None})

    def test_match_metrics_counts_ambiguous_valid_as_false_point_when_complete(self):
        items = [_ev(30, wl.POSITIVE, True), _ev(40, wl.AMBIGUOUS, True, kenshi="KENSHI_AKA"), _ev(90, wl.NEG_NO_POINT, False)]
        m = wl.match_metrics(_report(items))
        self.assertEqual((m["acertos"], m["falsos_pontos"], m["perdidos"]), (1, 1, 1))
        self.assertEqual(m["placar_oficial"], {"aka": 1, "shiro": 1})
        self.assertEqual(m["placar_senpai"], {"aka": 2, "shiro": 0})
        self.assertEqual(m["erro_placar"], 2)
        self.assertFalse(m["placar_exato"])
        self.assertEqual((m["precisao"], m["recall"]), (0.5, 0.5))

    def test_decision_accuracy(self):
        items = [_ev(30, wl.POSITIVE, True),        # deu ponto num ponto oficial: certo
                 _ev(40, wl.AMBIGUOUS, True),       # gabarito completo: ponto duplicado, errado
                 _ev(90, wl.NEG_NO_POINT, False),   # não deu ponto num não-ponto: certo
                 _ev(120, wl.NEG_OUTSIDE, True)]    # deu ponto fora da luta: errado
        m = wl.match_metrics(_report(items))
        self.assertEqual((m["decisoes_avaliadas"], m["decisoes_corretas"], m["acuracia"]), (4, 2, 0.5))

        incomplete = wl.match_metrics(_report(items, complete=False))
        self.assertEqual(incomplete["decisoes_avaliadas"], 3)  # ambíguo não conta sem gabarito completo
        self.assertIsNone(incomplete["acuracia"])

        t = wl.aggregate_match_metrics([m, incomplete])
        self.assertEqual((t["decisoes_avaliadas"], t["acuracia"]), (4, 0.5))

    def test_match_metrics_incomplete_has_no_precision(self):
        items = [_ev(30, wl.POSITIVE, True), _ev(40, wl.AMBIGUOUS, True), _ev(80, wl.UNKNOWN, True)]
        m = wl.match_metrics(_report(items, complete=False))
        self.assertEqual(m["falsos_pontos"], 0)  # ambíguo e desconhecido não contam sem gabarito completo
        self.assertIsNone(m["precisao"])
        self.assertIsNone(m["f1"])
        self.assertEqual(m["recall"], 0.5)

    def test_aggregate(self):
        rows = [wl.match_metrics(_report([_ev(30, wl.POSITIVE, True), _ev(40, wl.AMBIGUOUS, True)])),
                wl.match_metrics(_report([_ev(30, wl.POSITIVE, True), _ev(270, wl.POSITIVE, True, kenshi="KENSHI_SHIRO")])),
                wl.match_metrics(_report([_ev(30, wl.POSITIVE, True)], complete=False))]
        t = wl.aggregate_match_metrics(rows)
        self.assertEqual((t["lutas"], t["lutas_completas"], t["placares_exatos"]), (3, 2, 1))
        self.assertEqual(t["precisao"], 0.75)        # só lutas completas: 3 acertos, 1 falso
        self.assertEqual(t["recall"], round(4 / 6, 4))  # todas: 4 acertos de 6 pontos


class TestGoldenMatches(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gb = _golden(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_freeze_requires_a_match_with_valid_points(self):
        """Verdade = revisão: basta ser luta (gabarito incompleto pode); descartado ou tempo inválido não entra."""
        with self.assertRaises(ValueError):
            self.gb.freeze_match("luta.mp4", {"pontos_completos": True, "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "x"}]})
        with self.assertRaises(ValueError):
            self.gb.freeze_match("luta.mp4", {"tipo": "descartar", "pontos": []})
        with self.assertRaises(ValueError):
            self.gb.freeze_match("luta.mp4", {"tipo": "kakari_geiko", "pontos": []})
        self.assertFalse(self.gb.is_golden_video("luta.mp4"))
        self.gb.freeze_match("luta.mp4", {"pontos": [], "pontos_completos": False})
        self.assertTrue(self.gb.is_golden_video("luta.mp4"))

    def test_frozen_copy_is_independent_of_gabarito(self):
        entry = json.loads(json.dumps(COMPLETE_ENTRY))
        self.gb.freeze_match("/pasta/luta.mp4", entry)
        self.assertTrue(self.gb.is_golden_video("luta.mp4"))
        self.assertFalse(self.gb.differs_from_gabarito("luta.mp4", entry))

        entry["pontos"].append({"kenshi": "SHIRO", "tecnica": "KOTE", "tempo": "00:20"})
        self.assertTrue(self.gb.differs_from_gabarito("luta.mp4", entry))
        self.assertEqual(len(self.gb.load_matches()["luta.mp4"]["pontos"]), 1)

        self.assertTrue(self.gb.unfreeze_match("luta.mp4"))
        self.assertFalse(self.gb.unfreeze_match("luta.mp4"))

    def test_report_to_samples(self):
        items = [_ev(30, wl.POSITIVE, True), _ev(40, wl.NEG_CORRECTION, False), _ev(50, wl.UNKNOWN, False),
                 _ev(55, wl.AMBIGUOUS, False), _ev(60, wl.NEG_OUTSIDE, False, sub={})]
        samples = GoldenBenchmark.report_to_samples(_report(items))
        self.assertEqual([s["ground_truth"] for s in samples], [True, False])
        self.assertEqual(samples[0]["category"], REAL_MATCH_CATEGORY)
        self.assertEqual(samples[0]["metrics"]["target_impact"], 0.8)
        self.assertEqual(samples[0]["metrics"]["hasuji_score"], 0.9)


class TestGoldenRegressionGate(unittest.TestCase):
    def test_gate_switches_from_synthetic_to_real_in_the_same_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            gb = _golden(tmp)
            self.assertEqual(len(gb.load_samples()), 9)
            self.assertEqual(gb.samples_source()["fonte"], "sintetico")

            sample = {"id": "x", "category": REAL_MATCH_CATEGORY, "strike_type": "Men", "ground_truth": True,
                      "metrics": {"target_impact": 0.9, "fumikomi_sync": 0.9, "posture": 0.9, "zanshin": 0.9}}
            gb.replace_real_samples([dict(sample, id=f"x{i}") for i in range(MIN_REAL_SAMPLES_FOR_GATE - 1)])
            self.assertEqual(len(gb.load_samples()), 9)  # poucas reais: mantém sintéticas
            self.assertEqual(gb.samples_source()["golpes_reais"], MIN_REAL_SAMPLES_FOR_GATE - 1)

            gb.replace_real_samples([dict(sample, id=f"y{i}", ground_truth=(i % 2 == 0)) for i in range(MIN_REAL_SAMPLES_FOR_GATE)])
            self.assertEqual(len(gb.load_samples()), MIN_REAL_SAMPLES_FOR_GATE)
            self.assertEqual(gb.samples_source()["fonte"], "real")
            self.assertEqual(gb.evaluate_profile({})["source"], "real")

            # As sintéticas continuam no arquivo, como reserva
            with open(gb.dataset_path, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), 9 + MIN_REAL_SAMPLES_FOR_GATE)

    def test_evaluate_profile_follows_eliminatory_pillars_option(self):
        """A trava do golden usa a mesma decisão da análise: a opção pilares_eliminatorios vale nos dois."""
        from src.engine.calibrator import FIK_STRICT_ELIMINATORY_PILLARS, LENIENT_ELIMINATORY_PILLARS
        from src.engine.feedback_manager import DEFAULT_CALIBRATION_PROFILES
        with tempfile.TemporaryDirectory() as tmp:
            gb = _golden(tmp)
            lenient = dict(DEFAULT_CALIBRATION_PROFILES["normal"], min_total_score=0.64, pilares_eliminatorios=LENIENT_ELIMINATORY_PILLARS)
            strict = dict(lenient, pilares_eliminatorios=FIK_STRICT_ELIMINATORY_PILLARS)
            self.assertGreater(gb.evaluate_profile(lenient)["fp"], gb.evaluate_profile(strict)["fp"])

    def test_evaluate_profile_uses_unified_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = _golden(tmp).evaluate_profile({"min_total_score": 0.99, "sub_thresholds": {}})
            self.assertEqual((m["tp"], m["precision"], m["recall"], m["f1"]), (0, 0.0, 0.0, 0.0))


class TestGoldenNeverTrains(unittest.TestCase):
    def setUp(self):
        GoldenBenchmark().unfreeze_match("luta.mp4")

    tearDown = setUp

    def test_golden_match_never_goes_to_feedback_or_curation(self):
        GoldenBenchmark().freeze_match("luta.mp4", COMPLETE_ENTRY)
        self.assertTrue(is_golden_video("luta.mp4"))
        report = _report([_ev(30, wl.POSITIVE, True), _ev(40, wl.AMBIGUOUS, True)])
        report["video_path"] = "luta.mp4"
        fm, sampler = mock.Mock(), mock.Mock()
        self.assertTrue(wl.export_to_feedback(report, feedback_manager=fm).get("golden"))
        self.assertEqual(wl.enqueue_ambiguous(report, sampler=sampler), 0)
        fm.save_feedback.assert_not_called()
        sampler.evaluate_and_enqueue.assert_not_called()

    def test_default_golden_lives_in_test_sandbox(self):
        self.assertNotIn(os.path.normpath("data/benchmark_golden"), os.path.normpath(GoldenBenchmark().dataset_path))


class TestGoldenRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.utils.demo_generator import generate_demo_kendo_video
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "demo_golden.mp4")
        generate_demo_kendo_video(cls.video, duration_sec=3, fps=30)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        feature_cache.clear_feature_cache()

    def test_run_matches_on_frozen_match(self):
        from src.pipeline import SenpAIPipeline
        gb = _golden(os.path.join(self.tmp.name, "golden"))
        gb.freeze_match("demo_golden.mp4", COMPLETE_ENTRY)
        gb.freeze_match("sumiu.mp4", COMPLETE_ENTRY)

        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        fake = StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0)
        from src.engine.label_bank import LabelBank
        bank = LabelBank()
        review_id = None
        try:
            with mock.patch.object(pipeline.event_spotter, "detect_strikes", return_value=[fake]):
                # Só o ponto dos shinpan: nada medido, e o golpe vira pedido de revisão
                before = gb.run_matches(pipeline, self.tmp.name)
                review_id = bank.add_label(origin="curadoria", video_name="demo_golden.mp4", claim="ponto", value=True,
                                           time_sec=1.33, kenshi=None, ref="r1", reviewer_weight=5.0)
                res = gb.run_matches(pipeline, self.tmp.name)
        finally:
            labels = bank.load()
            labels.pop(review_id, None)
            bank._save(labels)

        self.assertEqual(before["total"]["lutas"], 0)
        self.assertEqual(before["sem_revisao"], ["demo_golden.mp4"])
        self.assertTrue(any("shinpan deram ponto" in r["motivo"] for r in before["pedidos_revisao"]))
        self.assertEqual(res["videos_ausentes"], ["sumiu.mp4"])
        self.assertEqual(res["falhas"], [])
        self.assertEqual(res["total"]["lutas"], 1)
        self.assertEqual(res["total"]["validos_revisados"], 1)
        self.assertEqual(res["total"]["candidatos_que_acham_valido"], 1)
        self.assertEqual(res["total"]["shinpan_concordam"], 1)     # shinpan e revisão deram ponto no mesmo golpe
        self.assertEqual(len(gb.load_history()), 1)
        self.assertEqual(gb.samples_source()["golpes_reais"], res["golpes_reais"])

        # A medição mostrada no painel é exatamente a última rodada gravada
        from src.utils.golden_benchmark_panel import latest_measurement
        last = latest_measurement(gb)
        self.assertEqual(last["total"], res["total"])
        self.assertIn("acuracia", last["total"])


if __name__ == "__main__":
    unittest.main()
