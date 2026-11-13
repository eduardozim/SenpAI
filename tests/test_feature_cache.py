"""
Testes Unitários do Cache Persistente de Features de Vídeo (Pendência I1).
Garante que a reanálise de um vídeo já processado reaproveita as poses do disco,
produz o mesmo resultado da extração completa e não grava efeitos colaterais de aprendizado.
"""

import unittest
import os
import tempfile
from unittest import mock

from src.pipeline import SenpAIPipeline
from src.analytics.event_spotter import StrikeEvent
from src.utils import feature_cache
from src.utils.demo_generator import generate_demo_kendo_video


class TestFeatureCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_video = os.path.join(cls.temp_dir.name, "demo_feature_cache.mp4")
        generate_demo_kendo_video(cls.test_video, duration_sec=3, fps=30)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def setUp(self):
        feature_cache.clear_feature_cache()

    def tearDown(self):
        feature_cache.clear_feature_cache()

    def test_cache_disabled_by_default_in_test_mode(self):
        """Em modo de teste o cache fica desligado por padrão e vive na sandbox isolada."""
        self.assertFalse(feature_cache.is_feature_cache_enabled_by_default())
        self.assertNotIn(os.path.normpath(feature_cache.DEFAULT_FEATURE_CACHE_DIR), os.path.normpath(feature_cache.get_feature_cache_dir()))

    def test_fingerprint_independent_of_file_name(self):
        """Renomear ou mover o vídeo mantém a mesma impressão digital de conteúdo."""
        copy_path = os.path.join(self.temp_dir.name, "outro_nome.mp4")
        with open(self.test_video, "rb") as src, open(copy_path, "wb") as dst:
            dst.write(src.read())
        self.assertEqual(
            feature_cache.compute_video_fingerprint(self.test_video),
            feature_cache.compute_video_fingerprint(copy_path)
        )

    def test_cache_key_changes_with_extraction_params(self):
        """Parâmetros que mudam a extração geram chaves diferentes."""
        base = feature_cache.build_cache_key(self.test_video, "yolov26")
        self.assertNotEqual(base, feature_cache.build_cache_key(self.test_video, "yolov8"))
        self.assertNotEqual(base, feature_cache.build_cache_key(self.test_video, "yolov26", frame_step=2))
        self.assertNotEqual(base, feature_cache.build_cache_key(self.test_video, "yolov26", invert_combatants=True))

    def test_second_run_reuses_cache_with_same_result(self):
        """A segunda análise lê as poses do cache, sem rodar a inferência, e chega ao mesmo resultado."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        first = pipeline.process_video(self.test_video, use_feature_cache=True)
        self.assertIsNotNone(first)
        self.assertFalse(first["feature_cache_hit"])
        self.assertEqual(feature_cache.get_feature_cache_stats()["entries"], 1)

        with mock.patch.object(pipeline.pose_detector, "process_frame_candidates_batch") as mocked_infer:
            second = pipeline.process_video(self.test_video, use_feature_cache=True)
            mocked_infer.assert_not_called()

        self.assertTrue(second["feature_cache_hit"])
        self.assertEqual(first["events_detected_count"], second["events_detected_count"])
        self.assertEqual(first["scoreboard"]["aka_score"], second["scoreboard"]["aka_score"])
        self.assertEqual(first["scoreboard"]["shiro_score"], second["scoreboard"]["shiro_score"])
        self.assertEqual(first["sonkyo_analysis"]["match_start_frame"], second["sonkyo_analysis"]["match_start_frame"])
        self.assertEqual(first["plane_filtering"]["flag_decision"], second["plane_filtering"]["flag_decision"])

    def test_training_mode_reuses_cache(self):
        """Modo Treinamento (I13): a 2ª análise reaproveita pose e re-identificação, com o mesmo resultado."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        first = pipeline.process_video(self.test_video, analysis_mode="training", use_feature_cache=True)
        self.assertFalse(first["feature_cache_hit"])
        first_tracks = sorted(pipeline.training_multi_tracker.tracks)
        key = feature_cache.build_cache_key(self.test_video, pipeline.vision_model, mode="training")
        self.assertNotEqual(key, feature_cache.build_cache_key(self.test_video, pipeline.vision_model))
        self.assertIsNotNone(feature_cache.load_training_features(key))
        with mock.patch.object(pipeline.pose_detector, "process_frame_candidates_batch") as mocked_infer:
            second = pipeline.process_video(self.test_video, analysis_mode="training", use_feature_cache=True)
            mocked_infer.assert_not_called()
        self.assertTrue(second["feature_cache_hit"])
        self.assertEqual(first["events_detected_count"], second["events_detected_count"])
        self.assertEqual(first["training_analysis"]["modality_key"], second["training_analysis"]["modality_key"])
        self.assertEqual(len(first["training_analysis"]["sub_sessions"]), len(second["training_analysis"]["sub_sessions"]))
        self.assertEqual(sorted(pipeline.training_multi_tracker.tracks), first_tracks)

    def test_corrupted_cache_file_is_regenerated(self):
        """Um arquivo de cache corrompido é ignorado e a extração completa é refeita."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        key = feature_cache.build_cache_key(self.test_video, pipeline.vision_model)
        with open(os.path.join(feature_cache.get_feature_cache_dir(), f"{key}.pkl.gz"), "wb") as f:
            f.write(b"lixo")

        result = pipeline.process_video(self.test_video, use_feature_cache=True)
        self.assertIsNotNone(result)
        self.assertFalse(result["feature_cache_hit"])
        self.assertIsNotNone(feature_cache.load_match_features(key))

    def test_extract_match_features_for_batch_precompute(self):
        """Pré-processamento em lote grava o cache e a chamada seguinte o reaproveita."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        first = pipeline.extract_match_features(self.test_video)
        second = pipeline.extract_match_features(self.test_video)
        self.assertFalse(first["from_cache"])
        self.assertTrue(second["from_cache"])
        self.assertEqual(len(first["aka_history"]), len(second["aka_history"]))

    def test_side_effects_can_be_disabled_for_replays(self):
        """Com record_learning_side_effects=False, a reanálise não grava baselines do Kenshi."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        if pipeline.kinesthetic_manager is None:
            self.skipTest("KinestheticProfileManager indisponível neste ambiente.")
        # O vídeo demo não gera golpes; injeta um evento para exercitar o caminho de avaliação
        fake_strike = StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0)
        with mock.patch.object(pipeline.event_spotter, "detect_strikes", return_value=[fake_strike]), \
             mock.patch.object(pipeline.kinesthetic_manager, "record_strike_event", return_value={}) as mocked_record:
            res = pipeline.process_video(self.test_video, use_feature_cache=True, record_learning_side_effects=False)
            self.assertGreaterEqual(res["events_detected_count"], 1)
            mocked_record.assert_not_called()

            pipeline.process_video(self.test_video, use_feature_cache=True, record_learning_side_effects=True)
            mocked_record.assert_called()


if __name__ == "__main__":
    unittest.main()
