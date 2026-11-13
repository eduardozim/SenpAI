"""
Referência dos golpes validados: distribuição real dos pilares, pesada pela autoridade do rótulo.
"""

import os
import tempfile
import unittest
from unittest import mock

from src.analytics import strike_reference as sr
from src.engine import weak_labels as wl


def _ev(score, label=wl.POSITIVE, tech="MEN", auth=3.0, origin="gabarito", **extra):
    scores = {"target_impact": score, "fumikomi_sync": score, "posture": score, "zanshin": score}
    scores.update(extra)
    return {"weak_label": label, "strike_type": tech, "sub_scores": scores, "autoridade_rotulo": auth,
            "origem_rotulo": origin, "matched_point": None}


def _training(n_videos=4, per_video=6, base=0.6, tech="MEN"):
    reports = []
    for v in range(n_videos):
        items = [_ev(base + 0.01 * i + 0.002 * v, tech=tech) for i in range(per_video)]
        items.append(_ev(0.1, label=wl.NEG_NO_POINT, tech=tech))       # inválido: não entra
        reports.append({"video_name": f"luta{v}.mp4", "official_points": [], "events": {"items": items}})
    return {"reports": reports}


class TestBuild(unittest.TestCase):
    def test_only_validated_strikes_and_minimum(self):
        ref = sr.build_reference(_training(n_videos=4, per_video=6))
        men = ref["tecnicas"]["MEN"]
        self.assertEqual((men["golpes"], men["videos"], men["medido"]), (24, 4, True))
        self.assertGreater(men["pilares"]["zanshin"]["p5"], 0.5)        # o golpe inválido (0.1) ficou de fora
        self.assertFalse(ref["tecnicas"]["KOTE"]["medido"])
        few = sr.build_reference(_training(n_videos=1, per_video=25))   # muitos golpes, mas de um vídeo só
        self.assertFalse(few["tecnicas"]["MEN"]["medido"])

    def test_technique_comes_from_correction_then_official_point(self):
        item = _ev(0.7, tech="MEN")
        item["matched_point"] = 0
        report = {"official_points": [{"index": 0, "technique": "KOTE"}]}
        self.assertEqual(sr._technique(item, report), "KOTE")
        item["tecnica_corrigida"] = "DO"
        self.assertEqual(sr._technique(item, report), "DO")

    def test_higher_authority_pulls_the_distribution(self):
        """Um golpe confirmado por um 8º dan pesa mais que um lido das bandeiras."""
        low = sr._weighted_quantile([(0.2, 1.5), (0.9, 8.0)], 50)
        self.assertEqual(low, 0.9)
        self.assertEqual(sr._weighted_quantile([(0.2, 8.0), (0.9, 1.5)], 50), 0.2)


class TestUse(unittest.TestCase):
    def setUp(self):
        self.ref = sr.build_reference(_training(n_videos=4, per_video=6, base=0.6))

    def test_percentiles_and_description(self):
        c = sr.compare("men", {"target_impact": 0.9, "fumikomi_sync": 0.62, "posture": 0.5, "zanshin": 66}, self.ref)
        self.assertTrue(c["medido"])
        self.assertEqual(c["percentis"]["target_impact"], 100.0)
        self.assertEqual(c["percentis"]["posture"], 0.0)
        text = sr.describe("MEN", {"target_impact": 0.9, "posture": 0.5, "zanshin": 0.66}, self.ref)
        self.assertIn("24 MEN validados", text)
        self.assertIn("Ponto mais fraco: postura", text)

    def test_not_measured_never_shows_numbers(self):
        text = sr.describe("KOTE", {"target_impact": 0.9}, self.ref)
        self.assertIn("ainda não medida", text)
        self.assertIn(f"0 de {sr.MIN_SAMPLES}", text)
        self.assertIsNone(sr.outlier_reason("KOTE", {"target_impact": 0.1, "posture": 0.1, "zanshin": 0.1}, True, self.ref))

    def test_outliers(self):
        typical = {"target_impact": 0.65, "fumikomi_sync": 0.65, "posture": 0.65, "zanshin": 0.65}
        self.assertIsNone(sr.outlier_reason("MEN", typical, True, self.ref))
        weak_zanshin = dict(typical, zanshin=0.3)
        reason, unc = sr.outlier_reason("MEN", weak_zanshin, True, self.ref)
        self.assertIn("zanshin", reason)
        self.assertGreater(unc, 0.5)
        strong = {k: 0.8 for k in typical}
        reason, _ = sr.outlier_reason("MEN", strong, False, self.ref)
        self.assertIn("deixou passar", reason)
        self.assertIsNone(sr.outlier_reason("MEN", {k: 0.61 for k in typical}, False, self.ref))   # abaixo da mediana

    def test_save_load_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ref.json")
            sr.save_reference(self.ref, path)
            self.assertEqual(sr.load_reference(path)["tecnicas"]["MEN"]["golpes"], 24)
            with mock.patch.object(sr, "get_reference_path", return_value=path):
                self.assertEqual(sr.cached_reference()["tecnicas"]["MEN"]["golpes"], 24)
            with mock.patch.object(sr, "get_reference_path", return_value=os.path.join(tmp, "nao_existe.json")):
                self.assertEqual(sr.cached_reference(), {})


class TestCycleStep(unittest.TestCase):
    def test_step_reuses_calibration_reports(self):
        from src.engine import training_cycle as tc
        tc._last_training = _training(n_videos=2, per_video=3)
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(sr, "get_reference_path", return_value=os.path.join(tmp, "r.json")), \
             mock.patch("src.engine.label_bank.build_training_reports") as rebuild:
            res = tc.step_reference(mock.Mock(), "pasta")
            rebuild.assert_not_called()
            self.assertEqual(res["status"], "pulado")       # 6 golpes de 2 vídeos: abaixo do mínimo
            self.assertIn("MEN 6 golpe(s)/2 vídeo(s)", res["detalhe"])
            self.assertTrue(os.path.isfile(os.path.join(tmp, "r.json")))
        self.assertIsNone(tc._last_training)


class TestPipelineDoubts(unittest.TestCase):
    def test_outlier_strike_becomes_padrao_doubt_with_scores(self):
        from src.pipeline import SenpAIPipeline
        ref = sr.build_reference(_training(n_videos=4, per_video=6, base=0.6))
        events = [{"event_info": {"impact_frame": 300, "type": "MEN", "attacker_id": "KENSHI_AKA"},
                   "evaluation": {"is_valid": True, "total_score": 70.0,
                                  "sub_scores": {"target_impact": 0.65, "fumikomi_sync": 0.65, "posture": 0.65, "zanshin": 0.2}}},
                  {"event_info": {"impact_frame": 600, "type": "MEN", "attacker_id": "KENSHI_SHIRO"},
                   "evaluation": {"is_valid": True, "total_score": 70.0,
                                  "sub_scores": {"target_impact": 0.65, "fumikomi_sync": 0.65, "posture": 0.65, "zanshin": 0.65}}}]
        fake = mock.Mock()
        fake.calibrator.profile_name = "normal"
        sampler = mock.Mock()
        with mock.patch.object(sr, "cached_reference", return_value=ref), \
             mock.patch("src.engine.active_learning.UncertaintySampler", return_value=sampler), \
             mock.patch("src.utils.strike_clip_manager.extract_strike_subclip", return_value=None):
            SenpAIPipeline._enqueue_extra_doubts(fake, "luta.mp4", 30.0, [], [], events, None)
        calls = [c.kwargs for c in sampler.enqueue_doubt.call_args_list]
        self.assertEqual([c["doubt_type"] for c in calls], ["padrao"])
        self.assertEqual(calls[0]["strike_data"]["sub_scores"]["zanshin"], 0.2)
        self.assertIn("zanshin", calls[0]["reason"])


if __name__ == "__main__":
    unittest.main()
