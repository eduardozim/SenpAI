"""
Testes do Classificador Luta × 14 Modalidades (Pendência I14).
Poses sintéticas: suburi (sozinho, braços acima da cabeça em ritmo regular), ashi-sabaki (sozinho, kamae
parado e pés andando) e luta (dupla perto, golpes irregulares).
"""

import math
import random
import tempfile
import unittest
from unittest import mock

from src.analytics import modality_classifier as mcl
from src.analytics.training_analyzer import TrainingAnalyzer, modality_confidence_label, MODEL_METHOD
from src.utils import model_registry

FPS = 30.0


def _pose(cx, hip_y, wrist_y, ankle_dx=0.0):
    pts = {
        "LEFT_SHOULDER": (cx - 20, hip_y - 100), "RIGHT_SHOULDER": (cx + 20, hip_y - 100),
        "LEFT_HIP": (cx - 15, hip_y), "RIGHT_HIP": (cx + 15, hip_y),
        "LEFT_EAR": (cx - 8, hip_y - 140), "RIGHT_EAR": (cx + 8, hip_y - 140),
        "LEFT_WRIST": (cx - 4, wrist_y), "RIGHT_WRIST": (cx + 4, wrist_y),
        "LEFT_ANKLE": (cx - 15 + ankle_dx, hip_y + 120), "RIGHT_ANKLE": (cx + 15 - ankle_dx, hip_y + 120),
    }
    return {k: {"px": x, "py": y, "x": x / 1000, "y": y / 1000, "visibility": 0.9} for k, (x, y) in pts.items()}


def _suburi(n, rng):
    """Sozinho, no lugar; os punhos sobem acima da cabeça e descem a cada ~1 s."""
    period = 30 + rng.randint(-2, 2)
    return [[_pose(400 + rng.gauss(0, 1), 300, 200 - 90 * max(0.0, math.sin(2 * math.pi * i / period)) + rng.gauss(0, 2))
             for i in range(n)]]


def _ashi(n, rng):
    """Sozinho, punhos em kamae; o corpo anda para frente e para trás com passos."""
    return [[_pose(400 + 60 * math.sin(2 * math.pi * i / 45) + rng.gauss(0, 1), 300, 220 + rng.gauss(0, 2),
                   ankle_dx=10 * math.sin(2 * math.pi * i / 15)) for i in range(n)]]


def _luta(n, rng):
    """Dupla a ~1,5 tronco; golpes rápidos em momentos irregulares."""
    out = []
    for side in (-1, 1):
        hits = sorted(rng.sample(range(n), max(1, n // 50)))
        h = []
        for i in range(n):
            near = any(0 <= i - t < 6 for t in hits)
            h.append(_pose(400 + side * 70 + 15 * math.sin(i / 20.0 + side), 300,
                           (120 if near else 220) + rng.gauss(0, 2), ankle_dx=4 * math.sin(i / 7.0)))
        out.append(h)
    return out


GENERATORS = {"suburi": _suburi, "ashi_sabaki": _ashi, mcl.LUTA: _luta}


def _dataset(videos_per_class=3, seconds=40, seed=0):
    rng = random.Random(seed)
    data = {"rows": [], "embeddings": [], "y": [], "groups": [], "skipped": [], "encoder": None}
    for cls, gen in GENERATORS.items():
        for v in range(videos_per_class):
            hist = gen(int(seconds * FPS), rng)
            for a, b in mcl.video_windows(len(hist[0]), FPS):
                data["rows"].append(mcl.window_features(hist, a, b, FPS))
                data["embeddings"].append(None)
                data["y"].append(cls)
                data["groups"].append(f"{cls}_{v}.mp4")
    return data


class TestModalityFeatures(unittest.TestCase):
    def test_features_separate_solo_and_pair(self):
        rng = random.Random(1)
        solo = mcl.window_features(_suburi(300, rng), 0, 300, FPS)
        pair = mcl.window_features(_luta(300, rng), 0, 300, FPS)
        self.assertEqual(solo["praticante_solo"], 1.0)
        self.assertEqual(pair["praticantes_ativos"], 2.0)
        self.assertGreater(pair["tempo_com_parceiro"], 0.9)
        self.assertGreater(solo["bracos_acima_cabeca"], 0.2)
        self.assertGreater(solo["golpes_por_minuto"], 20)

    def test_windows_respect_label_range(self):
        wins = mcl.video_windows(3000, FPS, start_s=10.0, end_s=40.0)
        self.assertEqual(wins[0][0], 300)
        self.assertLessEqual(wins[-1][1], 1200)
        self.assertEqual(mcl.video_windows(50, FPS), [])   # trecho curto demais (< 3 s)

    def test_empty_window_has_zero_features(self):
        f = mcl.window_features([[None] * 300], 0, 300, FPS)
        self.assertEqual(f["praticantes_ativos"], 0.0)
        self.assertTrue(all(v == 0.0 for v in f.values()))


class TestModalityTraining(unittest.TestCase):
    def test_train_validates_by_video_and_beats_baseline(self):
        model = mcl.train_model(_dataset())
        v = model["validacao_por_video"]
        self.assertEqual(sorted(model["params"]["classes"]), sorted(GENERATORS))
        self.assertGreater(v["acuracia"], v["chute_mais_comum"])
        self.assertGreaterEqual(v["acuracia"], 0.8)
        self.assertEqual(model["variantes_validadas"], {"sem_encoder": v["acuracia"]})

    def test_insufficient_data_raises(self):
        data = _dataset(videos_per_class=1)
        keep = [i for i, c in enumerate(data["y"]) if c == "suburi"]
        only_one = dict(data, **{k: [data[k][i] for i in keep] for k in ("rows", "embeddings", "y", "groups")})
        with self.assertRaises(ValueError):
            mcl.train_model(only_one)

    def test_promotion_and_prediction(self):
        with tempfile.TemporaryDirectory() as d:
            model = mcl.train_model(_dataset())
            path = mcl.save_model(model, d)
            self.assertTrue(mcl.promote_if_better(path, d)["promovido"])
            # O mesmo modelo de novo não supera o atual
            again = mcl.save_model(dict(model, trained_at=model["trained_at"] + "b"), d)
            self.assertFalse(mcl.promote_if_better(again, d)["promovido"])
            clf = mcl.ModalityClassifier.load_active(d)
            self.assertIsNotNone(clf)
            rng = random.Random(9)
            hist = _suburi(900, rng)[0]
            ashi = _ashi(900, rng)[0]
            pred = clf.predict_segments([hist + ashi], 1800, FPS)
            classes = [s["classe"] for s in pred["segmentos"]]
            self.assertEqual(classes[0], "suburi")
            self.assertEqual(classes[-1], "ashi_sabaki")
            for s in pred["segmentos"]:
                self.assertTrue(0.0 < s["probabilidade"] <= 1.0)

    def test_rejects_model_not_better_than_majority(self):
        with tempfile.TemporaryDirectory() as d:
            model = mcl.train_model(_dataset())
            model["validacao_por_video"] = {"acuracia": 0.30, "chute_mais_comum": 0.34}
            path = mcl.save_model(model, d)
            res = mcl.promote_if_better(path, d)
            self.assertFalse(res["promovido"])
            self.assertIn("chute", res["motivo"])
            self.assertIsNone(model_registry.get_active_model_path(mcl.MODEL_FAMILY, d))

    def test_build_dataset_uses_bank_labels_and_skips_golden(self):
        rng = random.Random(3)
        tracks = {"a.mp4": (_suburi(1200, rng), FPS, 1200), "b.mp4": (_luta(1200, rng), FPS, 1200)}
        labels = [
            {"video_name": "a.mp4", "value": "suburi", "time_sec": 0.0, "fim_s": 20.0},
            {"video_name": "b.mp4", "value": mcl.LUTA, "time_sec": 0.0, "fim_s": None},
            {"video_name": "g.mp4", "value": mcl.LUTA, "time_sec": 0.0, "fim_s": None},
        ]
        with tempfile.TemporaryDirectory() as d:
            for n in ("a.mp4", "b.mp4", "g.mp4"):
                open(f"{d}/{n}", "wb").close()
            with mock.patch("src.engine.label_bank.LabelBank.modality_labels", return_value=labels), \
                 mock.patch("src.engine.active_learning.is_golden_video", side_effect=lambda n: n == "g.mp4"), \
                 mock.patch("src.analytics.pose_encoder.PoseEncoder.load_active", return_value=None), \
                 mock.patch.object(mcl, "load_tracks", side_effect=lambda p, v, prefer_match: tracks[v.replace("\\", "/").split("/")[-1]]):
                data = mcl.build_dataset(mock.Mock(), d)
        self.assertEqual(data["skipped"], [{"video_name": "g.mp4", "motivo": "golden"}])
        self.assertEqual(data["y"].count("suburi"), 3)       # 20 s → janelas 0-10, 5-15, 10-20
        self.assertEqual(data["y"].count(mcl.LUTA), 7)       # 40 s inteiros
        self.assertEqual(set(data["groups"]), {"a.mp4", "b.mp4"})


class TestSuggestions(unittest.TestCase):
    def test_threshold_is_measured_on_validation(self):
        probs = [0.95] * 12 + [0.6] * 10
        correct = [True] * 12 + [False] * 5 + [True] * 5
        thr = mcl.confidence_threshold(probs, correct)
        self.assertEqual((thr["valor"], thr["acerto"], thr["trechos"]), (0.95, 1.0, 12))
        self.assertIsNone(mcl.confidence_threshold([0.9] * 5, [True] * 5))          # poucos trechos
        self.assertIsNone(mcl.confidence_threshold([0.9] * 20, [True, False] * 10))  # nunca chega a 90%
        model = mcl.train_model(_dataset())
        self.assertIn("limiar_confianca", model)

    def _run(self, gab_videos, title=None, thr=0.8, prob=0.9, cls="suburi"):
        from src.engine import weak_labels as wl
        from src.engine.active_learning import UncertaintySampler
        clf = mock.Mock()
        clf.path = "modelo_x.json"
        clf.model = {"limiar_confianca": None if thr is None else {"valor": thr}}
        clf.predict_segments.return_value = {"dominante": cls, "probabilidade": prob,
                                             "segmentos": [{"classe": cls, "inicio_s": 0.0, "fim_s": 30.0, "probabilidade": prob}]}
        tmp = tempfile.mkdtemp()
        labels = f"{tmp}/g.json"
        wl.save_match_labels({"versao": 1, "videos": gab_videos}, labels)
        videos = [f"{tmp}/{n}" for n in ("novo.mp4", "feito.mp4")]
        for v in videos:
            open(v, "wb").close()
        from src.engine.feedback_manager import FeedbackManager
        FeedbackManager().clear_active_learning_queue()
        mcl._save_suggestions({})
        with mock.patch.object(mcl, "_title_modality", return_value=title):
            res = mcl.suggest_modalities(mock.Mock(), videos, classifier=clf, labels_path=labels,
                                         tracks_loader=lambda v: ([[None]], 30.0, 900), make_clip=False)
        doubts = [it for it in UncertaintySampler().get_queue() if it.get("tipo_duvida") == "modalidade"]
        return res, mcl.load_suggestions(), doubts

    def test_confident_suggestion_needs_no_doubt_and_skips_classified(self):
        res, sug, doubts = self._run({"feito.mp4": {"tipo": "kirikaeshi", "ignorar": True}})
        self.assertEqual(list(sug), ["novo.mp4"])
        self.assertEqual((sug["novo.mp4"]["tipo"], sug["novo.mp4"]["confianca"]), ("suburi", "alta"))
        self.assertEqual((res["sugeridos"], res["alta"], res["duvidas"]), (1, 1, 0))
        self.assertEqual(doubts, [])

    def test_low_confidence_or_title_disagreement_becomes_doubt(self):
        _, sug, doubts = self._run({}, prob=0.6)
        self.assertEqual(sug["novo.mp4"]["confianca"], "baixa")
        self.assertEqual(len([d for d in doubts if d["video_source"].endswith("novo.mp4")]), 1)
        _, _, doubts = self._run({}, title="kakari_geiko", prob=0.95)
        d = [d for d in doubts if d["video_source"].endswith("novo.mp4")][0]
        self.assertIn("título", d["motivo"])
        self.assertEqual(d["opcoes"]["sugestao"], "suburi")

    def test_without_measured_threshold_everything_goes_to_curation(self):
        res, sug, _ = self._run({}, thr=None, prob=0.99)
        self.assertEqual(sug["novo.mp4"]["confianca"], "baixa")
        self.assertIn("ainda não medido", res["detalhe"])

    def test_answering_doubt_writes_gabarito_type(self):
        from src.engine import weak_labels as wl
        from src.engine.feedback_manager import FeedbackManager
        _, _, doubts = self._run({}, prob=0.5)
        d = [d for d in doubts if d["video_source"].endswith("novo.mp4")][0]
        labels = d["video_source"].rsplit("/", 1)[0] + "/g.json"
        self.assertTrue(FeedbackManager().resolve_modality_doubt(d["id"], "ji_geiko", labels_path=labels))
        entry = wl.load_match_labels(labels)["videos"]["novo.mp4"]
        self.assertEqual((entry["tipo"], entry["ignorar"]), ("ji_geiko", True))
        with self.assertRaises(ValueError):
            FeedbackManager().resolve_modality_doubt(d["id"], "inexistente", labels_path=labels)


class TestModalityIntegration(unittest.TestCase):
    def test_confidence_label_is_honest(self):
        self.assertEqual(modality_confidence_label("AUTO_DETECTED", 0.92), "confiança não medida (regra fixa)")
        self.assertEqual(modality_confidence_label("MANUAL_SELECT", 1.0), "definida pelo usuário")
        self.assertIn("73%", modality_confidence_label(MODEL_METHOD, 0.73))

    def test_analyzer_uses_model_segments(self):
        rng = random.Random(5)
        hist = _suburi(900, rng)[0] + _ashi(900, rng)[0]
        pred = {"segmentos": [
            {"classe": "suburi", "inicio_s": 0.0, "fim_s": 30.0, "probabilidade": 0.81},
            {"classe": mcl.LUTA, "inicio_s": 30.0, "fim_s": 35.0, "probabilidade": 0.55},
            {"classe": "ashi_sabaki", "inicio_s": 35.0, "fim_s": 60.0, "probabilidade": 0.66},
        ], "dominante": "suburi", "probabilidade": 0.81}
        res = TrainingAnalyzer().analyze_multi_session({"KENSHI_01": hist}, fps=FPS, model_prediction=pred).to_dict()
        self.assertEqual(res["detection_method"], MODEL_METHOD)
        self.assertEqual(res["modality_key"], "suburi")
        self.assertAlmostEqual(res["detection_confidence"], 0.81, places=2)
        subs = res["sub_sessions"]
        self.assertEqual([s["modality_key"] for s in subs], ["suburi", "ashi_sabaki"])   # trecho de luta fica fora
        self.assertTrue(all(s["detection_method"] == MODEL_METHOD for s in subs))

    def test_user_correction_wins_over_model(self):
        rng = random.Random(6)
        hist = _suburi(900, rng)[0]
        pred = {"segmentos": [{"classe": "suburi", "inicio_s": 0.0, "fim_s": 30.0, "probabilidade": 0.9}]}
        res = TrainingAnalyzer().analyze_multi_session({"KENSHI_01": hist}, fps=FPS, modality_override="kirikaeshi",
                                                       model_prediction=pred).to_dict()
        self.assertEqual(res["modality_key"], "kirikaeshi")
        self.assertEqual(res["detection_method"], "MANUAL_SELECT")

    def test_pipeline_check_warns_on_mode_mismatch(self):
        from src.pipeline import SenpAIPipeline
        pipe = SenpAIPipeline.__new__(SenpAIPipeline)
        pipe.modality_classifier = None
        self.assertIsNone(pipe._check_modality([[]], FPS, 0, chosen_mode="luta"))
        fake = mock.Mock(path="m.json")
        fake.predict_segments.return_value = {"segmentos": [], "dominante": "suburi", "probabilidade": 0.7}
        pipe.modality_classifier = fake
        chk = pipe._check_modality([[None]], FPS, 1, chosen_mode="luta")
        self.assertIn("treino", chk["aviso"])
        fake.predict_segments.return_value = {"segmentos": [], "dominante": mcl.LUTA, "probabilidade": 0.8}
        self.assertIn("luta", pipe._check_modality([[None]], FPS, 1, chosen_mode="treino")["aviso"])
        self.assertIsNone(pipe._check_modality([[None]], FPS, 1, chosen_mode="luta")["aviso"])


if __name__ == "__main__":
    unittest.main()
