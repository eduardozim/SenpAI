"""
Testes do Detector em Duas Etapas — filtro aprendido de golpes (Pendência I3).
"""

import os
import json
import math
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.analytics import strike_classifier as sc
from src.analytics.event_spotter import StrikeEvent

JOINTS = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "LEFT_WRIST", "RIGHT_WRIST",
          "LEFT_EAR", "RIGHT_EAR", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL"]


def _pose(cx, wrist_y, scale=1.0, ankle_lift=0.0):
    """Kenshi simplificado em pixels: tronco de 100 px (× scale), pulsos na altura indicada."""
    base = {
        "LEFT_SHOULDER": (cx - 20, 200), "RIGHT_SHOULDER": (cx + 20, 200),
        "LEFT_HIP": (cx - 15, 300), "RIGHT_HIP": (cx + 15, 300),
        "LEFT_EAR": (cx - 8, 160), "RIGHT_EAR": (cx + 8, 160),
        "LEFT_WRIST": (cx - 5, wrist_y), "RIGHT_WRIST": (cx + 5, wrist_y),
        "LEFT_ANKLE": (cx - 15, 480 - ankle_lift), "RIGHT_ANKLE": (cx + 15, 480),
        "LEFT_HEEL": (cx - 18, 485 - ankle_lift), "RIGHT_HEEL": (cx + 12, 485),
    }
    return {k: {"px": x * scale, "py": y * scale, "x": x / 1000, "y": y / 1000, "visibility": 0.9} for k, (x, y) in base.items()}


def _strike_history(n=60, impact=30, scale=1.0, real=True):
    """Golpe real: furikaburi (pulsos sobem), corte rápido, fumikomi e avanço. Falso: pulsos parados."""
    hist = []
    for i in range(n):
        if real:
            if i < impact - 10:
                wy, cx = 230, 300
            elif i < impact - 4:
                wy, cx = 230 - (i - (impact - 10)) * 25, 300 + (i - (impact - 10)) * 3
            elif i <= impact:
                wy, cx = 80 + (i - (impact - 4)) * 40, 320 + (i - (impact - 4)) * 6
            else:
                wy, cx = 240, 350 + (i - impact) * 4
            lift = 40 if impact - 3 <= i <= impact - 1 else 0
        else:
            wy, cx, lift = 230 + (i % 2) * 3, 300, 0
        hist.append(_pose(cx, wy, scale=scale, ankle_lift=lift))
    return hist


def _defender(n=60, scale=1.0):
    return [_pose(420, 230, scale=scale) for _ in range(n)]


class TestFeatures(unittest.TestCase):
    def test_all_features_present_and_finite(self):
        f = sc.extract_candidate_features(_strike_history(), _defender(), 30, 30.0)
        self.assertEqual(list(f.keys()), sc.FEATURE_NAMES)
        self.assertTrue(all(math.isfinite(v) for v in f.values()))

    def test_real_strike_differs_from_static_noise(self):
        real = sc.extract_candidate_features(_strike_history(real=True), _defender(), 30, 30.0)
        fake = sc.extract_candidate_features(_strike_history(real=False), _defender(), 30, 30.0)
        self.assertGreater(real["pulso_dir_vel_pico"], fake["pulso_dir_vel_pico"] * 5)
        self.assertGreater(real["elevacao_maxima"], fake["elevacao_maxima"])
        self.assertGreater(real["fumikomi_tornozelo"], fake["fumikomi_tornozelo"])

    def test_camera_distance_does_not_change_features(self):
        near = sc.extract_candidate_features(_strike_history(scale=2.0), _defender(scale=2.0), 30, 30.0)
        far = sc.extract_candidate_features(_strike_history(scale=1.0), _defender(scale=1.0), 30, 30.0)
        for k in sc.FEATURE_NAMES:
            self.assertAlmostEqual(near[k], far[k], places=6, msg=k)

    def test_missing_poses_do_not_crash(self):
        hist = _strike_history()
        hist[28:33] = [None] * 5
        f = sc.extract_candidate_features(hist, [], 30, 30.0)
        self.assertGreater(f["frames_sem_pose"], 0.0)
        self.assertTrue(all(math.isfinite(v) for v in f.values()))


def _synthetic_dataset(videos=4, per_video=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, y, groups = [], [], []
    for v in range(videos):
        for i in range(per_video):
            real = i % 3 == 0
            jitter = int(rng.integers(-2, 3))
            rows.append(sc.extract_candidate_features(_strike_history(real=real), _defender(), 30 + jitter, 30.0))
            y.append(1 if real else 0)
            groups.append(f"video_{v}.mp4")
    return {"rows": rows, "y": y, "groups": groups, "skipped": []}


class TestModel(unittest.TestCase):
    def test_logistic_separates_strikes(self):
        ds = _synthetic_dataset()
        X, y = sc.features_matrix(ds["rows"]), np.array(ds["y"], float)
        p = sc.predict_proba(sc.fit_logistic(X, y), X)
        self.assertGreater(p[y == 1].min(), p[y == 0].max())

    def test_cross_validation_is_grouped_by_video(self):
        ds = _synthetic_dataset()
        X, y = sc.features_matrix(ds["rows"]), np.array(ds["y"], float)
        calls = []
        real_fit = sc.fit_logistic

        def spy(Xt, yt, l2=1.0):
            calls.append(len(yt))
            return real_fit(Xt, yt, l2=l2)

        with mock.patch.object(sc, "fit_logistic", side_effect=spy):
            oof = sc.group_cross_validate(X, y, ds["groups"])
        self.assertEqual(calls, [36, 36, 36, 36])  # cada vídeo fica de fora do treino da sua previsão
        self.assertFalse(np.isnan(oof).any())

    def test_train_model_requires_enough_data(self):
        ds = _synthetic_dataset(videos=1, per_video=6)
        with self.assertRaises(ValueError):
            sc.train_model(ds)

    def test_train_save_load_and_activate(self):
        model = sc.train_model(_synthetic_dataset())
        self.assertEqual(model["features"], sc.FEATURE_NAMES)
        self.assertEqual(model["dados"]["positivos"], 16)
        self.assertGreater(model["validacao_por_video"]["com_filtro"]["f1"], model["validacao_por_video"]["sem_filtro"]["f1"])
        with tempfile.TemporaryDirectory() as tmp:
            path = sc.save_model(model, tmp)
            self.assertIsNone(sc.StrikeFilter.load_active(tmp))
            sc.set_active_model(path, "teste", tmp)
            flt = sc.StrikeFilter.load_active(tmp)
            self.assertIsNotNone(flt)
            p_real = flt.probability(_strike_history(real=True), _defender(), 30, 30.0)
            p_fake = flt.probability(_strike_history(real=False), _defender(), 30, 30.0)
            self.assertTrue(flt.keep(p_real))
            self.assertFalse(flt.keep(p_fake))
            sc.set_active_model(None, "teste", tmp)
            self.assertIsNone(sc.StrikeFilter.load_active(tmp))

    def test_model_with_other_features_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"schema": sc.MODEL_SCHEMA_VERSION, "features": ["outra"]}, f)
            self.assertIsNone(sc.load_model(path))


class _StubFilter:
    def __init__(self, keep_all, path="stub.json"):
        self.keep_all = keep_all
        self.path = path

    def probability(self, *args):
        return 0.9 if self.keep_all else 0.1

    def keep(self, p):
        return p >= 0.5


class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.utils.demo_generator import generate_demo_kendo_video
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "demo_filtro.mp4")
        generate_demo_kendo_video(cls.video, duration_sec=3, fps=30)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, pipeline, **kw):
        fake = StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0)
        with mock.patch.object(pipeline.event_spotter, "detect_strikes", return_value=[fake]):
            return pipeline.process_video(self.video, record_learning_side_effects=False, **kw)

    def test_filter_off_by_default_in_tests(self):
        from src.pipeline import SenpAIPipeline
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        self.assertIsNone(pipeline.strike_filter)
        res = self._run(pipeline)
        self.assertFalse(res["strike_filter"]["ativo"])
        self.assertEqual(res["events_detected_count"], 1)

    def test_filter_drops_and_keeps_candidates(self):
        from src.pipeline import SenpAIPipeline
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")

        pipeline.strike_filter = _StubFilter(keep_all=False)
        dropped = self._run(pipeline)
        self.assertEqual(dropped["events_detected_count"], 0)
        self.assertEqual(dropped["strike_filter"], {"ativo": True, "modelo": "stub.json", "candidatos": 1, "mantidos": 0})

        pipeline.strike_filter = _StubFilter(keep_all=True)
        kept = self._run(pipeline)
        self.assertEqual(kept["events_detected_count"], 1)
        self.assertEqual(kept["events"][0]["event_info"]["strike_filter_probability"], 0.9)

        bypass = self._run(pipeline, use_strike_filter=False)
        self.assertFalse(bypass["strike_filter"]["ativo"])


class TestBuildDataset(unittest.TestCase):
    def test_dataset_from_gabarito_excludes_golden(self):
        from src.pipeline import SenpAIPipeline
        from src.engine import weak_labels as wl
        from src.engine.active_learning import GoldenBenchmark
        from src.utils.demo_generator import generate_demo_kendo_video

        entry = {"pontos_completos": True, "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:01.3"}]}
        gb = GoldenBenchmark()
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("treino.mp4", "golden.mp4"):
                generate_demo_kendo_video(os.path.join(tmp, name), duration_sec=3, fps=30)
            labels_path = os.path.join(tmp, "gabarito.json")
            wl.save_match_labels({"versao": 1, "videos": {"treino.mp4": entry, "golden.mp4": entry,
                                                         "vazio.mp4": {"pontos": [], "pontos_completos": False}}}, labels_path)
            gb.freeze_match("golden.mp4", entry)
            try:
                pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
                strikes = [StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0),
                           StrikeEvent("MEN", start_frame=75, impact_frame=86, end_frame=89, fps=30.0)]  # 2,87 s: longe do ponto
                with mock.patch.object(pipeline.event_spotter, "detect_strikes", return_value=strikes):
                    ds = sc.build_dataset(pipeline, tmp, labels_path=labels_path)
            finally:
                gb.unfreeze_match("golden.mp4")

        self.assertEqual(set(ds["groups"]), {"treino.mp4"})
        self.assertIn({"video_name": "golden.mp4", "motivo": "golden"}, ds["skipped"])
        self.assertEqual(sum(ds["y"]), 1)            # o golpe em 1,33 s casa com o ponto oficial
        self.assertGreaterEqual(len(ds["y"]), 2)     # o outro vira negativo (gabarito completo)
        self.assertEqual(len(ds["rows"][0]), len(sc.FEATURE_NAMES))


class TestPromotion(unittest.TestCase):
    def test_no_promotion_without_golden_matches(self):
        model = sc.train_model(_synthetic_dataset())
        with tempfile.TemporaryDirectory() as tmp:
            path = sc.save_model(model, tmp)
            empty = {"lutas": [], "total": {"f1": None}}
            with mock.patch.object(sc, "evaluate_on_golden", return_value=empty), \
                 mock.patch.object(sc, "set_active_model") as activate:
                res = sc.promote_if_better(mock.Mock(), tmp, path)
            self.assertFalse(res["promovido"])
            activate.assert_not_called()

    def test_promotes_only_when_golden_f1_improves(self):
        model = sc.train_model(_synthetic_dataset())
        with tempfile.TemporaryDirectory() as tmp:
            path = sc.save_model(model, tmp)
            current = {"lutas": [{}], "total": {"f1": 0.30}}
            better = {"lutas": [{}], "total": {"f1": 0.55}}
            worse = {"lutas": [{}], "total": {"f1": 0.20}}
            for challenger, expected in ((better, True), (worse, False)):
                with mock.patch.object(sc, "evaluate_on_golden", side_effect=[current, challenger]), \
                     mock.patch.object(sc, "set_active_model") as activate:
                    res = sc.promote_if_better(mock.Mock(), tmp, path)
                self.assertEqual(res["promovido"], expected)
                self.assertEqual(activate.called, expected)


if __name__ == "__main__":
    unittest.main()
