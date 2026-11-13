"""
Testes da Pendência P6: a Fila de Curadoria Ativa recebe as notas reais do golpe.
"""

import os
import tempfile
import unittest
from unittest import mock

from src.pipeline import SenpAIPipeline
from src.analytics.event_spotter import StrikeEvent
from src.utils.demo_generator import generate_demo_kendo_video

SUB_SCORES = {"target_impact": 72.0, "fumikomi_sync": 55.0, "posture": 61.0, "zanshin": 40.0, "hasuji": 80.0}


class TestCurationQueueSubScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video = os.path.join(cls.tmp.name, "demo_curadoria.mp4")
        generate_demo_kendo_video(cls.video, duration_sec=3, fps=30)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_uncertain_strike_is_enqueued_with_real_sub_scores(self):
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        fake_strike = StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0)
        real_evaluate = pipeline.calibrator.evaluate_strike

        def uncertain_evaluate(*args, **kwargs):
            # Avaliação real, com nota e sub-notas forçadas para a faixa de incerteza (45%–65%)
            evaluation = real_evaluate(*args, **kwargs)
            evaluation.update({"total_score": 55.0, "sub_scores": dict(SUB_SCORES)})
            return evaluation

        with mock.patch.object(pipeline.event_spotter, "detect_strikes", return_value=[fake_strike]), \
             mock.patch.object(pipeline.calibrator, "evaluate_strike", side_effect=uncertain_evaluate), \
             mock.patch("src.utils.strike_clip_manager.extract_strike_subclip", return_value=None), \
             mock.patch("src.engine.active_learning.UncertaintySampler.evaluate_and_enqueue") as enqueue:
            res = pipeline.process_video(self.video, record_learning_side_effects=True)

        self.assertIsNotNone(res)
        enqueue.assert_called()
        strike_data = enqueue.call_args.kwargs["strike_data"]
        self.assertEqual(strike_data["sub_scores"], SUB_SCORES)
        self.assertEqual(strike_data["scores"], SUB_SCORES)
        self.assertEqual(strike_data["total_score"], 55.0)

    def test_curation_decision_saves_real_scores_to_feedback(self):
        from src.engine.feedback_manager import FeedbackManager
        fm = FeedbackManager()
        item = {"id": "unc_teste_p6", "profile_name": "normal", "strike_type": "MEN", "confidence": 0.5,
                "video_source": "luta.mp4", "timestamp": "00:01.333",
                "strike_data": {"sub_scores": dict(SUB_SCORES), "total_score": 55.0}}
        with mock.patch.object(fm.uncertainty_sampler, "get_queue", return_value=[item]), \
             mock.patch.object(fm.uncertainty_sampler, "resolve_item", return_value=False), \
             mock.patch.object(fm, "save_feedback") as save:
            fm.resolve_active_learning_item("unc_teste_p6", decision="IPPON", reviewer_dan=5, trigger_retrain=False)
        save.assert_called_once()
        self.assertEqual(save.call_args.kwargs["sub_scores"], SUB_SCORES)
        self.assertEqual(save.call_args.kwargs["total_score"], 55.0)
        self.assertEqual(save.call_args.kwargs["label"], "TP")


if __name__ == "__main__":
    unittest.main()
