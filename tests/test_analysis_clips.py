"""
Cortes de 10 s dos golpes em dúvida das análises (vão para data/videos e alimentam as bandeiras) e
mineração só de vídeos novos de luta.
"""

import os
import json
import tempfile
import unittest
from unittest import mock

from src.utils import video_library as vl
from src.vision import flag_detector as fd
from src.engine import weak_labels as wl


def _fake_extractor(calls):
    def ext(src, out, target_time_sec, window_before_sec, window_after_sec, strike_type):
        calls.append((target_time_sec, window_before_sec, window_after_sec))
        open(out, "wb").close()
        return out
    return ext


class TestAnalysisClips(unittest.TestCase):
    def test_clip_saved_once_per_strike_neighbourhood(self):
        with tempfile.TemporaryDirectory() as tmp:
            vdir = os.path.join(tmp, "videos")
            os.makedirs(vdir)
            src = os.path.join(tmp, "upload_1793_Luta A x B.mp4")
            open(src, "wb").close()
            calls = []
            ext = _fake_extractor(calls)
            first = vl.save_analysis_clip(src, 42.0, vdir, ext)
            self.assertEqual(os.path.basename(first), "analise_Luta_A_x_B_42.0s.mp4")
            self.assertEqual(calls[0], (42.0, 5.0, 5.0))                 # 5 s antes e 5 s depois
            self.assertIsNone(vl.save_analysis_clip(src, 47.5, vdir, ext))   # golpe próximo: não repete
            self.assertIsNotNone(vl.save_analysis_clip(src, 60.0, vdir, ext))
            self.assertIsNone(vl.save_analysis_clip(os.path.join(vdir, "internet_x.mp4"), 3.0, vdir, ext))  # já na pasta
            self.assertIsNone(vl.save_analysis_clip(os.path.join(tmp, "nao_existe.mp4"), 3.0, vdir, ext))
            self.assertTrue(vl.is_analysis_clip(first) and vl.is_dataset_video(first))
            self.assertFalse(vl.is_dataset_video("Minha luta.mp4"))

    def test_doubt_from_analysis_saves_clip_but_modality_doubt_does_not(self):
        from src.engine.active_learning import UncertaintySampler
        from src.engine.feedback_manager import FeedbackManager
        FeedbackManager().clear_active_learning_queue()
        saved = []
        with mock.patch("src.utils.video_library.save_analysis_clip",
                        side_effect=lambda src, t: saved.append((src, t)) or "data/videos/analise_x_10.0s.mp4"):
            s = UncertaintySampler()
            e = s.enqueue_doubt("pontuacao", "data/tmp/upload_1_luta.mp4", {"timestamp_seconds": 10.0}, 0.6, "nota incerta")
            s.enqueue_doubt("modalidade", "data/videos/internet_y.mp4", {"timestamp_seconds": 5.0}, 0.6, "modalidade")
        self.assertEqual(saved, [("data/tmp/upload_1_luta.mp4", 10.0)])
        self.assertEqual(e["corte_treino"], "data/videos/analise_x_10.0s.mp4")
        FeedbackManager().clear_active_learning_queue()


class TestMiningCandidates(unittest.TestCase):
    def test_only_new_fight_videos_are_mined(self):
        with tempfile.TemporaryDirectory() as tmp:
            fd.save_dataset({"a": {"video_name": "velho.mp4", "rotulo": None}}, tmp, mined=["vazio.mp4"])
            labels = os.path.join(tmp, "g.json")
            wl.save_match_labels({"versao": 1, "videos": {"treino.mp4": {"tipo": "kakari_geiko", "ignorar": True},
                                                          "lixo.mp4": {"tipo": "descartar", "ignorar": True},
                                                          "luta.mp4": {"tipo": "luta"}}}, labels)
            videos = [f"/v/{n}" for n in ("velho.mp4", "vazio.mp4", "treino.mp4", "lixo.mp4", "luta.mp4",
                                          "novo.mp4", "analise_x_10.0s.mp4")]
            todo, skipped = fd.mining_candidates(videos, tmp, labels)
            self.assertEqual([os.path.basename(v) for v in todo], ["luta.mp4", "novo.mp4", "analise_x_10.0s.mp4"])
            self.assertEqual(skipped, {"ja_minerados": 2, "nao_luta": 2})
            # O registro de minerados sobrevive a salvar rótulos
            fd.set_labels({"a": "nenhuma"}, tmp)
            self.assertIn("vazio.mp4", fd.mined_videos(tmp))
            with open(os.path.join(tmp, "labels.json"), encoding="utf-8") as f:
                self.assertIn("videos_minerados", json.load(f))


if __name__ == "__main__":
    unittest.main()
