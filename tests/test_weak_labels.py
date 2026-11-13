"""
Testes Unitários da Rotulagem Fraca a partir do Placar Oficial (Pendência I2).
"""

import os
import json
import tempfile
import unittest
from unittest import mock

from src.engine import weak_labels as wl
from src.analytics.event_spotter import StrikeEvent
from src.utils import feature_cache


def _point(idx, kenshi, t, technique="MEN"):
    return {"index": idx, "kenshi_id": kenshi, "technique": technique, "time_sec": t, "is_strike": True}


def _det(frame, kenshi="KENSHI_AKA", action_class=None):
    return {"impact_frame": frame, "attacker_id": kenshi, "action_class": action_class}


class TestWeakLabelParsing(unittest.TestCase):
    def test_parse_time_formats(self):
        self.assertEqual(wl.parse_time_to_seconds(12), 12.0)
        self.assertEqual(wl.parse_time_to_seconds("42"), 42.0)
        self.assertEqual(wl.parse_time_to_seconds("01:23"), 83.0)
        self.assertAlmostEqual(wl.parse_time_to_seconds("01:23,5"), 83.5)
        self.assertEqual(wl.parse_time_to_seconds("1:00:00"), 3600.0)
        with self.assertRaises(ValueError):
            wl.parse_time_to_seconds("1m20")

    def test_normalize_kenshi(self):
        self.assertEqual(wl.normalize_kenshi("aka"), "KENSHI_AKA")
        self.assertEqual(wl.normalize_kenshi("Branco"), "KENSHI_SHIRO")
        self.assertIsNone(wl.normalize_kenshi(""))

    def test_hansoku_point_has_no_strike_to_match(self):
        points = wl.parse_official_points({"pontos": [
            {"kenshi": "SHIRO", "tecnica": "HANSOKU"},
            {"kenshi": "AKA", "tecnica": "kote", "tempo": "00:30"}
        ]})
        self.assertFalse(points[0]["is_strike"])
        self.assertIsNone(points[0]["time_sec"])
        self.assertEqual(points[1]["technique"], "KOTE")
        self.assertEqual(points[1]["time_sec"], 30.0)

    def test_template_keeps_filled_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gabarito.json")
            wl.make_labels_template(["/x/luta1.mp4"], path)
            data = wl.load_match_labels(path)
            data["videos"]["luta1.mp4"]["pontos"] = [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:10"}]
            wl.save_match_labels(data, path)

            _, added = wl.make_labels_template(["/x/luta1.mp4", "/x/luta2.mp4"], path)
            data = wl.load_match_labels(path)
            self.assertEqual(added, 1)
            self.assertEqual(len(data["videos"]["luta1.mp4"]["pontos"]), 1)
            self.assertFalse(wl.is_entry_filled(data["videos"]["luta2.mp4"]))


class TestLabelDetections(unittest.TestCase):
    FPS = 30.0

    def _run(self, dets, points, complete=True, start=0, end=10_000):
        return wl.label_detections(dets, points, self.FPS, start, end, complete, tolerance_sec=1.5)

    def test_match_is_one_to_one_and_closest(self):
        res = self._run([_det(290), _det(300), _det(900)], [_point(0, "KENSHI_AKA", 10.0)])
        labels = [i["weak_label"] for i in res["items"]]
        self.assertEqual(labels, [wl.AMBIGUOUS, wl.POSITIVE, wl.NEG_NO_POINT])

    def test_kenshi_must_match(self):
        res = self._run([_det(300, "KENSHI_SHIRO")], [_point(0, "KENSHI_AKA", 10.0)])
        self.assertEqual(res["items"][0]["weak_label"], wl.AMBIGUOUS)
        self.assertEqual(len(res["missed_points"]), 1)
        self.assertEqual(res["official_strike_points"], 1)

    def test_outside_match_window_is_negative(self):
        res = self._run([_det(30), _det(3000)], [], start=100, end=2000)
        self.assertEqual([i["weak_label"] for i in res["items"]], [wl.NEG_OUTSIDE, wl.NEG_OUTSIDE])

    def test_tsubazeriai_negative_even_with_incomplete_points(self):
        res = self._run([_det(600, action_class="TSUBAZERIAI"), _det(900)], [_point(0, "KENSHI_AKA", 10.0)], complete=False)
        self.assertEqual([i["weak_label"] for i in res["items"]], [wl.NEG_TSUBAZERIAI, wl.UNKNOWN])
        self.assertEqual(res["counts"], {wl.NEG_TSUBAZERIAI: 1, wl.UNKNOWN: 1})

    def test_point_near_tsubazeriai_still_positive(self):
        """Hiki-waza saindo do tsubazeriai: se há ponto oficial, o casamento vence."""
        res = self._run([_det(300, action_class="TSUBAZERIAI")], [_point(0, "KENSHI_AKA", 10.0)])
        self.assertEqual(res["items"][0]["weak_label"], wl.POSITIVE)


class TestExportToFeedback(unittest.TestCase):
    def test_only_confident_labels_are_saved(self):
        report = {"video_name": "luta.mp4", "events": {"items": [
            {"impact_frame": 300, "attacker_id": "KENSHI_AKA", "weak_label": wl.POSITIVE, "reason": "r", "time_sec": 10.0, "sub_scores": {"target_impact": 80.0}, "total_score": 70.0, "strike_type": "MEN"},
            {"impact_frame": 900, "attacker_id": "KENSHI_AKA", "weak_label": wl.NEG_NO_POINT, "reason": "r", "time_sec": 30.0},
            {"impact_frame": 290, "attacker_id": "KENSHI_AKA", "weak_label": wl.AMBIGUOUS, "reason": "r", "time_sec": 9.7},
            {"impact_frame": 1200, "attacker_id": "KENSHI_AKA", "weak_label": wl.UNKNOWN, "reason": "r", "time_sec": 40.0},
        ]}}
        fm = mock.Mock()
        saved = wl.export_to_feedback(report, feedback_manager=fm)
        self.assertEqual(saved, {"TP": 1, "FP": 1})
        labels = [c.kwargs["label"] for c in fm.save_feedback.call_args_list]
        self.assertEqual(labels, ["TP", "FP"])
        for c in fm.save_feedback.call_args_list:
            self.assertFalse(c.kwargs["enqueue_uncertain"])
            self.assertTrue(c.kwargs["notes"].startswith(wl.WEAK_LABEL_NOTE))


class TestWeakLabelVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.utils.demo_generator import generate_demo_kendo_video
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.temp_dir.name, "demo_weak_labels.mp4")
        generate_demo_kendo_video(cls.video, duration_sec=3, fps=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()
        feature_cache.clear_feature_cache()

    def test_end_to_end_on_cached_features(self):
        from src.pipeline import SenpAIPipeline
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        fake = StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0)
        entry = {"pontos_completos": True, "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:01.3"}]}
        with mock.patch.object(pipeline.event_spotter, "detect_strikes", return_value=[fake]):
            report = wl.weak_label_video(pipeline, self.video, entry)

        self.assertGreaterEqual(report["candidates"]["counts"].get(wl.POSITIVE, 0), 1)
        self.assertEqual(report["events"]["counts"].get(wl.POSITIVE, 0), 1)
        self.assertEqual(report["metricas"]["placar_oficial"], {"aka": 1, "shiro": 0})
        self.assertEqual(report["candidates"]["official_strike_points"], 1)

        path = wl.save_report(report)
        self.assertNotIn(os.path.normpath(wl.DEFAULT_REPORTS_DIR), os.path.normpath(path))
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["video_name"], "demo_weak_labels.mp4")

        summary = wl.aggregate_match_metrics([report["metricas"]])
        self.assertEqual(summary["lutas"], 1)
        self.assertEqual(summary["recall_candidatos"], 1.0)


if __name__ == "__main__":
    unittest.main()
