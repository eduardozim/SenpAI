"""
Testes do Encoder de Movimento Auto-supervisionado (Pendência I10).
"""

import os
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.analytics import pose_encoder as pe
from src.analytics import strike_classifier as sc
from src.engine import training_cycle as tc


def _pose(cx, wy, scale=1.0, tip=True):
    pts = {"LEFT_SHOULDER": (cx - 20, 200), "RIGHT_SHOULDER": (cx + 20, 200), "LEFT_ELBOW": (cx - 25, 240),
           "RIGHT_ELBOW": (cx + 25, 240), "LEFT_WRIST": (cx - 5, wy), "RIGHT_WRIST": (cx + 5, wy),
           "LEFT_HIP": (cx - 15, 300), "RIGHT_HIP": (cx + 15, 300), "LEFT_KNEE": (cx - 15, 390),
           "RIGHT_KNEE": (cx + 15, 390), "LEFT_ANKLE": (cx - 15, 480), "RIGHT_ANKLE": (cx + 15, 480),
           "LEFT_EAR": (cx - 8, 160), "RIGHT_EAR": (cx + 8, 160)}
    pose = {k: {"px": x * scale, "py": y * scale} for k, (x, y) in pts.items()}
    if tip:
        pose["SHINAI"] = {"detected": True, "tip_px": ((cx + 80) * scale, (wy - 30) * scale)}
    return pose


def _history(n=60, scale=1.0, kind=0):
    return [_pose(300 + i * (2 if kind else 0), 230 - (i % 20) * (6 if kind else 1), scale) for i in range(n)]


def _synthetic_windows(n_per_video=300, videos=3, seed=0):
    rng = np.random.default_rng(seed)
    X, groups = [], []
    for v in range(videos):
        for i in range(n_per_video):
            base = rng.normal(0, 1, (1, pe.FRAME_DIM)).astype(np.float32)
            t = np.linspace(0, 1, pe.WINDOW, dtype=np.float32)[:, None]
            w = base + np.sin(t * (1 + i % 7)) * rng.normal(0, 1, (1, pe.FRAME_DIM)).astype(np.float32)
            w[:, -1] = 1.0
            X.append(w)
            groups.append(f"luta_{v}.mp4")
    return np.stack(X), groups


class TestWindows(unittest.TestCase):
    def test_window_shape_and_camera_distance_invariance(self):
        near = pe.window_tensor(_history(scale=2.0), _history(scale=2.0), 30)
        far = pe.window_tensor(_history(scale=1.0), _history(scale=1.0), 30)
        self.assertEqual(near.shape, (pe.WINDOW, pe.FRAME_DIM))
        np.testing.assert_allclose(near, far, atol=1e-5)

    def test_window_rejected_with_too_many_missing_poses(self):
        hist = _history()
        for i in range(10, 40):
            hist[i] = None
        self.assertIsNone(pe.window_tensor(hist, _history(), 30))

    def test_embedding_feature_helpers(self):
        self.assertEqual(len(pe.embedding_feature_names()), pe.EMBED_DIM)
        base = {"a": 1.0}
        self.assertEqual(pe.with_embedding(base, None, [], [], 0), base)
        self.assertEqual(pe.feature_names_for(["a"], True), ["a"] + pe.embedding_feature_names())


class TestTraining(unittest.TestCase):
    def test_train_save_and_embed(self):
        X, groups = _synthetic_windows()
        with tempfile.TemporaryDirectory() as tmp:
            meta = pe.train_encoder(X, groups, epochs=2, batch=128, device="cpu", models_dir=tmp)
            self.assertTrue(os.path.isfile(meta["modelo"]))
            self.assertTrue(os.path.isfile(meta["modelo"].replace(".pt", ".json")))
            self.assertFalse(set(meta["videos_validacao"]) & set(groups[i] for i in range(len(groups)) if groups[i] not in meta["videos_validacao"]))
            r = meta["reconhecimento_do_par"]
            self.assertIsNotNone(r["depois_do_treino"])
            enc = pe.PoseEncoder.from_path(meta["modelo"])
            emb = enc.embed(_history(kind=1), _history(), 30)
            self.assertEqual(emb.shape, (pe.EMBED_DIM,))
            self.assertTrue(np.isfinite(emb).all())

    def test_refuses_too_little_data(self):
        X, groups = _synthetic_windows(n_per_video=50, videos=1)
        with self.assertRaises(ValueError):
            pe.train_encoder(X, groups, epochs=1, device="cpu")


class _FakeEncoder:
    path = "fake_encoder.pt"

    def embed(self, attacker, defender, impact_frame):
        return np.full(pe.EMBED_DIM, 0.1, dtype=np.float32)


class TestClassifierVariants(unittest.TestCase):
    def test_strike_filter_validates_with_and_without_encoder(self):
        from tests.test_strike_classifier import _synthetic_dataset
        ds = _synthetic_dataset()
        ds["rows"] = [pe.with_embedding(r, _FakeEncoder(), [], [], 0) for r in ds["rows"]]
        ds["encoder"] = _FakeEncoder.path
        model = sc.train_model(ds)
        self.assertEqual(set(model["variantes_validadas"]), {"sem_encoder", "com_encoder"})
        # Encoder constante não acrescenta informação: o empate fica com a versão sem encoder
        self.assertIsNone(model["encoder"])
        self.assertEqual(model["features"], sc.FEATURE_NAMES)

    def test_model_using_missing_encoder_is_rejected(self):
        from tests.test_strike_classifier import _synthetic_dataset
        model = sc.train_model(_synthetic_dataset())
        model["features"] = pe.feature_names_for(sc.FEATURE_NAMES, True)
        model["encoder"] = "nao_existe.pt"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(model, f)
            self.assertIsNone(sc.load_model(path))


class TestCycleStep(unittest.TestCase):
    def test_skips_when_cache_barely_grew(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = os.path.join(tmp, "pose_encoder_x.pt")
            open(weights, "w").close()
            with open(weights.replace(".pt", ".json"), "w", encoding="utf-8") as f:
                json.dump({"janelas": {"treino": 900, "validacao": 100}}, f)
            with mock.patch.object(pe, "collect_windows", return_value=(np.zeros((1100, 1, 1)), ["a"] * 1100)), \
                 mock.patch("src.utils.model_registry.get_active_model_path", return_value=weights), \
                 mock.patch.object(pe, "train_encoder") as train:
                res = tc.step_pretrain()
        self.assertEqual(res["status"], "pulado")
        train.assert_not_called()

    def test_activates_only_good_encoder(self):
        for quality, expected in ((0.85, True), (0.3, False)):
            meta = {"modelo": "enc.pt", "reconhecimento_do_par": {"antes_do_treino": 0.2, "depois_do_treino": quality}}
            with mock.patch.object(pe, "collect_windows", return_value=(np.zeros((2000, 1, 1)), ["a"] * 2000)), \
                 mock.patch("src.utils.model_registry.get_active_model_path", return_value=None), \
                 mock.patch.object(pe, "train_encoder", return_value=meta), \
                 mock.patch("src.utils.model_registry.set_active_model") as activate:
                res = tc.step_pretrain()
            self.assertEqual((res["ativado"], activate.called), (expected, expected))


if __name__ == "__main__":
    unittest.main()
