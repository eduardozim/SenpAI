"""
A validade do golpe vem só das revisões (kenshi e senseis, pesadas pelo dan), nunca da decisão dos shinpan.
O gabarito e as bandeiras ficam como contexto ("rotulo_shinpan") e viram pedidos de revisão.
"""

import os
import tempfile
import unittest
from unittest import mock

from src.engine import weak_labels as wl
from src.engine import label_bank as lb


def _item(t, shinpan, valid=False, kenshi="KENSHI_AKA"):
    return {"time_sec": t, "impact_frame": int(t * 30), "attacker_id": kenshi, "weak_label": shinpan,
            "predicted_valid": valid, "strike_type": "MEN", "matched_point": 0 if shinpan == wl.POSITIVE else None,
            "sub_scores": {"target_impact": 80.0}}


def _report(items, complete=True, missed=None):
    return {"video_name": "luta.mp4", "pontos_completos": complete,
            "official_points": [{"index": 0, "kenshi_id": "KENSHI_AKA", "technique": "KOTE", "time_sec": 10.0, "is_strike": True}],
            "events": {"items": [dict(i) for i in items], "missed_points": missed or [], "official_strike_points": 1},
            "candidates": {"items": [dict(i) for i in items], "counts": {}}}


def _review(t, value, dan, origin="revisao", claim="ponto"):
    return {"id": f"{origin}{t}{dan}", "origin": origin, "video_name": "luta.mp4", "claim": claim, "value": value,
            "time_sec": t, "kenshi": None, "reviewer_weight": dan, "confidence": 1.0}


class TestShinpanIsNotALabel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bank = lb.LabelBank(os.path.join(self.tmp.name, "bank.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_without_review_nothing_is_labeled(self):
        report = _report([_item(10.0, wl.POSITIVE, True), _item(20.0, wl.NEG_NO_POINT, True)])
        lb.relabel_by_reviews(report, [], self.bank)
        self.assertEqual({e["weak_label"] for e in report["events"]["items"]}, {wl.UNKNOWN})
        self.assertEqual([e["rotulo_shinpan"] for e in report["events"]["items"]], [wl.POSITIVE, wl.NEG_NO_POINT])

    def test_review_overrules_shinpan_and_agreement_is_counted(self):
        """Os shinpan deram ponto em 10 s; um 5º dan diz que não vale. Em 20 s os shinpan não deram; um 3º dan diz que vale."""
        report = _report([_item(10.0, wl.POSITIVE, True), _item(20.0, wl.NEG_NO_POINT, False), _item(30.0, wl.POSITIVE, True)])
        lb.relabel_by_reviews(report, [_review(10.0, False, 5), _review(20.0, True, 3), _review(30.0, True, 2)], self.bank)
        labels = [e["weak_label"] for e in report["events"]["items"]]
        self.assertEqual(labels, [wl.NEG_CORRECTION, wl.POSITIVE, wl.POSITIVE])
        self.assertEqual(report["events"]["items"][0]["autoridade_rotulo"], 5.0)
        self.assertEqual(report["concordancia_shinpan"], {"comparados": 3, "concordam": 1})

    def test_higher_dan_wins_and_close_reviews_conflict(self):
        report = _report([_item(10.0, wl.POSITIVE, True)])
        lb.relabel_by_reviews(report, [_review(10.0, True, 2), _review(10.1, False, 6)], self.bank)
        self.assertEqual(report["events"]["items"][0]["weak_label"], wl.NEG_CORRECTION)
        report = _report([_item(10.0, wl.POSITIVE, True)])
        conflicts = lb.relabel_by_reviews(report, [_review(10.0, True, 4), _review(10.1, False, 4, origin="curadoria")], self.bank)
        self.assertEqual(len(conflicts), 1)

    def test_corrections_never_come_from_shinpan_origins(self):
        self.bank.add_labels([dict(_review(10.0, True, 1, origin="gabarito"), id=None),
                              dict(_review(14.0, False, 3), id=None)])
        with self.assertRaises(ValueError):          # origem "bandeiras" não existe mais
            self.bank.add_labels([dict(_review(12.0, True, 1, origin="bandeiras"), id=None)])
        self.assertEqual([c["origin"] for c in self.bank.corrections_for("luta.mp4")], ["revisao"])

    def test_review_requests_point_to_what_shinpan_decided(self):
        missed = [{"index": 1, "kenshi_id": "KENSHI_SHIRO", "technique": "DO", "time_sec": 40.0, "is_strike": True}]
        report = _report([_item(10.0, wl.POSITIVE, False), _item(20.0, wl.NEG_NO_POINT, True),
                          _item(25.0, wl.NEG_NO_POINT, False), _item(30.0, wl.POSITIVE, True)], missed=missed)
        lb.relabel_by_reviews(report, [_review(30.0, True, 3)], self.bank)
        req = lb.review_requests(report)
        self.assertEqual(sorted(r["time_sec"] for r in req), [10.0, 20.0, 40.0])   # 25 s: ninguém deu ponto; 30 s: já revisado
        self.assertEqual(req[0]["time_sec"], 10.0)                                 # ponto dos shinpan primeiro
        self.assertEqual(next(r for r in req if r["time_sec"] == 40.0)["strike_type"], "DO")

    def test_technique_of_the_official_point_still_counts(self):
        """A técnica do ponto é a técnica tentada (não um juízo de validade): vale mesmo se a revisão invalidar o golpe."""
        report = _report([_item(10.0, wl.POSITIVE, True)])
        lb.relabel_by_reviews(report, [_review(10.0, False, 5)], self.bank)
        m = wl.review_metrics(report)
        self.assertEqual((m["tecnica_avaliada"], m["tecnica_correta"]), (1, 0))   # detectou MEN, o ponto foi KOTE
        report = _report([_item(10.0, wl.POSITIVE, True)])
        lb.relabel_by_reviews(report, [_review(10.0, False, 5, claim="golpe")], self.bank)
        self.assertEqual(wl.review_metrics(report)["tecnica_avaliada"], 0)       # "não é golpe": sem técnica


class TestReviewMetrics(unittest.TestCase):
    def test_golden_truth_is_the_review(self):
        bank = lb.LabelBank(os.path.join(tempfile.gettempdir(), "nao_usado_bank.json"))
        report = _report([_item(10.0, wl.POSITIVE, True),       # shinpan e revisão: válido; SenpAI deu → acerto
                          _item(20.0, wl.POSITIVE, True),       # shinpan deram; revisão diz inválido → falso ponto
                          _item(30.0, wl.NEG_NO_POINT, False),  # shinpan não deram; revisão diz válido → perdido
                          _item(40.0, wl.NEG_NO_POINT, True)])  # sem revisão → fora da conta
        reviews = [_review(10.0, True, 4), _review(20.0, False, 4), _review(30.0, True, 4), _review(50.0, True, 4)]
        lb.relabel_by_reviews(report, reviews, bank)
        m = wl.review_metrics(report)
        self.assertEqual((m["acertos"], m["falsos_pontos"], m["perdidos"]), (1, 1, 2))   # 50 s: válido e não detectado
        self.assertEqual(m["golpes_revisados"], 3)
        self.assertEqual(m["precisao"], 0.5)
        self.assertEqual((m["shinpan_comparados"], m["shinpan_concordam"]), (3, 1))
        tot = wl.aggregate_review_metrics([m])
        self.assertEqual(tot["f1"], wl.precision_recall_f1(1, 1, 2)["f1"])
        self.assertAlmostEqual(tot["concordancia_shinpan"], round(1 / 3, 4))


class TestFeedbackOptimizerIgnoresShinpanLabels(unittest.TestCase):
    def test_default_feedback_skips_official_score_labels_and_golden(self):
        from src.engine.feedback_manager import FeedbackManager
        fm = FeedbackManager.__new__(FeedbackManager)
        data = [{"label": "TP", "decision_category": "rotulo_fraco", "video_name": "a.mp4"},
                {"label": "TP", "decision_category": "IPPON", "video_name": "golden.mp4"},
                {"label": "FP", "decision_category": "INVALID_STRIKE", "video_name": "b.mp4", "profile_key": "normal"},
                {"label": "FP", "decision_category": "INVALID_STRIKE", "video_name": "c.mp4", "profile_key": "rigido"}]
        with mock.patch.object(FeedbackManager, "load_feedback", return_value=data),              mock.patch("src.engine.active_learning.is_golden_video", side_effect=lambda n: n == "golden.mp4"):
            kept = fm.review_feedback("normal")
        self.assertEqual([d["video_name"] for d in kept], ["b.mp4"])


class TestCycleSendsShinpanPointsToCuration(unittest.TestCase):
    def test_review_requests_become_shinpan_doubts_with_limit(self):
        from src.engine import training_cycle as tc
        sampler = mock.Mock()
        reqs = [{"video_path": "v.mp4", "time_sec": float(i), "kenshi": "KENSHI_AKA", "strike_type": "MEN",
                 "sub_scores": {}, "total_score": None, "motivo": "Os shinpan deram ponto", "prioridade": 1.0} for i in range(5)]
        with mock.patch("src.engine.active_learning.UncertaintySampler", return_value=sampler):
            self.assertEqual(tc.enqueue_review_requests(reqs, limit=3), 3)
        kw = sampler.enqueue_doubt.call_args.kwargs
        self.assertEqual((kw["doubt_type"], kw["kenshi"]), ("shinpan", "AKA"))
        self.assertIn("3 de 4", tc.shinpan_agreement_text([{"concordancia_shinpan": {"comparados": 4, "concordam": 3}}]))
        self.assertEqual(tc.shinpan_agreement_text([{}]), "")


if __name__ == "__main__":
    unittest.main()
