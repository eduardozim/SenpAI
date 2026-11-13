"""
Testes do Classificador de Técnica — MEN, KOTE, DO, TSUKI (Pendência I9).
"""

import os
import math
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.analytics import technique_classifier as tcl
from src.analytics.event_spotter import StrikeEvent
from src.engine import weak_labels as wl
from src.utils import model_registry

# Oponente parado em x=600: cabeça em y=160, garganta ~185, punhos (kamae) em y=260, quadril em y=300
DEF_CX = 600
TARGETS = {
    "MEN": (DEF_CX - 10, 150),     # topo da cabeça
    "KOTE": (DEF_CX - 40, 262),    # punho direito do oponente
    "DO": (DEF_CX + 20, 280),      # lateral do tronco
    "TSUKI": (DEF_CX - 25, 188),   # garganta
}


def _pose(cx, wrist, tip=None, ears_y=160):
    base = {
        "LEFT_SHOULDER": (cx - 20, 200), "RIGHT_SHOULDER": (cx + 20, 200),
        "LEFT_HIP": (cx - 15, 300), "RIGHT_HIP": (cx + 15, 300),
        "LEFT_EAR": (cx - 8, ears_y), "RIGHT_EAR": (cx + 8, ears_y),
        "LEFT_WRIST": (wrist[0] - 4, wrist[1]), "RIGHT_WRIST": (wrist[0] + 4, wrist[1]),
    }
    pose = {k: {"px": x, "py": y, "x": x / 1000, "y": y / 1000, "visibility": 0.9} for k, (x, y) in base.items()}
    if tip is not None:
        pose["SHINAI"] = {"detected": True, "tip_px": tip}
    return pose


def _strike(technique, jitter=0.0, n=40, impact=25):
    """Atacante em x=420 terminando o golpe no alvo da técnica (ponta do shinai no alvo)."""
    tx, ty = TARGETS[technique]
    tx, ty = tx + jitter, ty + jitter
    atk, dfd = [], []
    for i in range(n):
        if technique == "TSUKI":
            wrist = (430 + min(i, impact) * 2, 240)                       # estocada horizontal
        elif technique == "DO":
            wrist = (430 + min(i, impact) * 2, 150 + min(i, impact) * 4)  # corte diagonal
        else:
            wrist = (440, 120 + min(i, impact) * 5) if i > impact - 8 else (440, 90)  # furikaburi e descida
        tip = (tx, ty) if i >= impact else (wrist[0] + 80, wrist[1] - 40)
        atk.append(_pose(420 + min(i, impact) * 2, wrist, tip=tip))
        dfd.append(_pose(DEF_CX, (DEF_CX - 40, 260)))
    return atk, dfd, impact


def _dataset(per_class=6, videos=3, classes=("MEN", "KOTE", "DO", "TSUKI")):
    rows, y, groups = [], [], []
    rng = np.random.default_rng(1)
    for v in range(videos):
        for c in classes:
            for _ in range(per_class // videos + 1):
                atk, dfd, t = _strike(c, jitter=float(rng.normal(0, 4)))
                rows.append(tcl.extract_technique_features(atk, dfd, t, 30.0))
                y.append(c)
                groups.append(f"luta_{v}.mp4")
    return {"rows": rows, "y": y, "groups": groups, "skipped": []}


class TestFeatures(unittest.TestCase):
    def test_features_complete_and_finite(self):
        atk, dfd, t = _strike("MEN")
        f = tcl.extract_technique_features(atk, dfd, t, 30.0)
        self.assertEqual(list(f), tcl.FEATURE_NAMES)
        self.assertTrue(all(math.isfinite(v) for v in f.values()))
        self.assertEqual(f["usa_ponta_shinai"], 1.0)

    def test_target_relative_to_opponent(self):
        men = tcl.extract_technique_features(*_strike("MEN"), 30.0)
        kote = tcl.extract_technique_features(*_strike("KOTE"), 30.0)
        self.assertLess(men["alvo_dist_cabeca"], kote["alvo_dist_cabeca"])
        self.assertLess(kote["alvo_dist_punhos"], men["alvo_dist_punhos"])

    def test_missing_defender_does_not_crash(self):
        atk, _, t = _strike("DO")
        f = tcl.extract_technique_features(atk, [], t, 30.0)
        self.assertTrue(all(math.isfinite(v) for v in f.values()))


class TestModel(unittest.TestCase):
    def test_softmax_learns_the_four_techniques(self):
        ds = _dataset()
        X = tcl.features_matrix(ds["rows"])
        params = tcl.fit_softmax(X, ds["y"], tcl.TECHNIQUES)
        pred = [tcl.TECHNIQUES[i] for i in np.argmax(tcl.predict_proba(params, X), axis=1)]
        acc = sum(p == t for p, t in zip(pred, ds["y"])) / len(pred)
        self.assertGreaterEqual(acc, 0.95)

    def test_train_validates_by_video_and_beats_majority_guess(self):
        model = tcl.train_model(_dataset())
        v = model["validacao_por_video"]
        self.assertGreater(v["acuracia"], v["chute_mais_comum"])
        self.assertEqual(sorted(model["params"]["classes"]), sorted(tcl.TECHNIQUES))

    def test_rare_technique_is_left_out_instead_of_guessed(self):
        ds = _dataset(classes=("MEN", "KOTE"))
        atk, dfd, t = _strike("TSUKI")
        ds["rows"].append(tcl.extract_technique_features(atk, dfd, t, 30.0))
        ds["y"].append("TSUKI")
        ds["groups"].append("luta_0.mp4")
        model = tcl.train_model(ds)
        self.assertEqual(model["params"]["classes"], ["MEN", "KOTE"])

    def test_insufficient_data(self):
        with self.assertRaises(ValueError):
            tcl.train_model(_dataset(per_class=2, videos=1, classes=("MEN",)))

    def test_save_activate_predict(self):
        model = tcl.train_model(_dataset())
        with tempfile.TemporaryDirectory() as tmp:
            path = tcl.save_model(model, tmp)
            model_registry.set_active_model(tcl.MODEL_FAMILY, path, "teste", tmp)
            clf = tcl.TechniqueClassifier.load_active(tmp)
            atk, dfd, t = _strike("KOTE")
            pred = clf.predict(atk, dfd, t, 30.0)
            self.assertEqual(pred["tecnica"], "KOTE")
            self.assertAlmostEqual(sum(pred["probabilidades"].values()), 1.0, places=3)


class TestTechniqueMetric(unittest.TestCase):
    def test_agreement_only_on_hits_with_known_official_technique(self):
        points = [{"index": 0, "technique": "KOTE"}, {"index": 1, "technique": "HANSOKU"}, {"index": 2, "technique": "Men"}]
        events = [
            {"predicted_valid": True, "weak_label": wl.POSITIVE, "matched_point": 0, "strike_type": "KOTE"},
            {"predicted_valid": True, "weak_label": wl.POSITIVE, "matched_point": 2, "strike_type": "DO"},
            {"predicted_valid": True, "weak_label": wl.POSITIVE, "matched_point": 1, "strike_type": "MEN"},   # hansoku: sem técnica
            {"predicted_valid": False, "weak_label": wl.POSITIVE, "matched_point": 0, "strike_type": "KOTE"},  # não deu ponto
        ]
        self.assertEqual(wl.technique_agreement(events, points), (2, 1))
        self.assertEqual(wl.normalize_technique("Men"), "MEN")
        self.assertIsNone(wl.normalize_technique("HANSOKU"))


class _StubTechnique:
    path = "stub.json"

    def predict(self, *args):
        return {"tecnica": "TSUKI", "probabilidade": 0.8, "probabilidades": {"MEN": 0.2, "TSUKI": 0.8}}


class TestPipelineIntegration(unittest.TestCase):
    def test_model_overrides_rule_based_technique(self):
        from src.pipeline import SenpAIPipeline
        from src.utils.demo_generator import generate_demo_kendo_video
        with tempfile.TemporaryDirectory() as tmp:
            video = os.path.join(tmp, "demo_tecnica.mp4")
            generate_demo_kendo_video(video, duration_sec=3, fps=30)
            pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
            self.assertIsNone(pipeline.technique_classifier)
            fake = StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0)
            with mock.patch.object(pipeline.event_spotter, "detect_strikes", return_value=[fake]):
                plain = pipeline.process_video(video, record_learning_side_effects=False)
                pipeline.technique_classifier = _StubTechnique()
                fake2 = StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0)
                pipeline.event_spotter.detect_strikes.return_value = [fake2]
                learned = pipeline.process_video(video, record_learning_side_effects=False)
        self.assertEqual(plain["events"][0]["event_info"]["technique_source"], "regra_fixa")
        info = learned["events"][0]["event_info"]
        self.assertEqual((info["type"], info["technique_source"], info["technique_rule_based"]), ("TSUKI", "modelo", "MEN"))


class TestPromotion(unittest.TestCase):
    def _run(self, current, challenger):
        model = tcl.train_model(_dataset())
        with tempfile.TemporaryDirectory() as tmp:
            path = tcl.save_model(model, tmp)
            with mock.patch.object(tcl, "evaluate_on_golden", side_effect=[{"total": current}, {"total": challenger}]), \
                 mock.patch("src.utils.model_registry.set_active_model") as activate:
                res = tcl.promote_if_better(mock.Mock(), tmp, path)
        return res["promovido"], activate.called

    def test_promotes_when_technique_improves(self):
        self.assertEqual(self._run({"acuracia_tecnica": 0.4, "f1": 0.5, "tecnica_avaliada": 10},
                                   {"acuracia_tecnica": 0.7, "f1": 0.5, "tecnica_avaliada": 10}), (True, True))

    def test_never_promotes_if_points_f1_drops(self):
        self.assertEqual(self._run({"acuracia_tecnica": 0.4, "f1": 0.6, "tecnica_avaliada": 10},
                                   {"acuracia_tecnica": 0.9, "f1": 0.5, "tecnica_avaliada": 10}), (False, False))

    def test_no_promotion_without_golden_techniques(self):
        self.assertEqual(self._run({"acuracia_tecnica": None, "f1": None, "tecnica_avaliada": 0},
                                   {"acuracia_tecnica": None, "f1": None, "tecnica_avaliada": 0}), (False, False))


class TestDataset(unittest.TestCase):
    def test_excludes_automatic_and_golden_matches(self):
        from src.engine.active_learning import GoldenBenchmark
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = os.path.join(tmp, "gabarito.json")
            point = [{"kenshi": "AKA", "tecnica": "KOTE", "tempo": "00:01"}]
            wl.save_match_labels({"versao": 1, "videos": {
                "manual.mp4": {"pontos_completos": True, "pontos": point},
                "auto.mp4": {"pontos_completos": True, "pontos": point, "origem": "automatico_bandeiras"},
                "golden.mp4": {"pontos_completos": True, "pontos": point},
            }}, labels_path)
            for n in ("manual.mp4", "auto.mp4", "golden.mp4"):
                open(os.path.join(tmp, n), "w").close()
            GoldenBenchmark().freeze_match("golden.mp4", {"pontos_completos": True, "pontos": point})
            atk, dfd, t = _strike("KOTE")
            def fake_report(pipe, video, entry, **kw):
                return {"video_name": os.path.basename(video), "video_path": video,
                        "official_points": wl.parse_official_points(entry),
                        "candidates": {"items": [{"weak_label": wl.POSITIVE, "matched_point": 0, "attacker_id": "KENSHI_AKA", "impact_frame": t, "time_sec": t / 30.0},
                                                 {"weak_label": wl.NEG_NO_POINT, "matched_point": None, "attacker_id": "KENSHI_AKA", "impact_frame": 5, "time_sec": 5 / 30.0}]},
                        "events": {"items": []}}
            pipeline = mock.Mock()
            pipeline.extract_match_features.return_value = {"aka_history": atk, "shiro_history": dfd, "fps": 30.0}
            pipeline.combatant_tracker.interpolate_missing_poses.side_effect = lambda h, max_gap=10: h
            try:
                with mock.patch.object(wl, "weak_label_video", side_effect=fake_report) as wlv:
                    ds = tcl.build_dataset(pipeline, tmp, labels_path=labels_path)
            finally:
                GoldenBenchmark().unfreeze_match("golden.mp4")
        self.assertEqual(wlv.call_count, 2)  # manual e automática são reanalisadas; a golden não
        self.assertEqual((ds["y"], ds["groups"]), (["KOTE"], ["manual.mp4"]))
        self.assertEqual({s["video_name"] for s in ds["skipped"]}, {"auto.mp4", "golden.mp4"})


if __name__ == "__main__":
    unittest.main()
