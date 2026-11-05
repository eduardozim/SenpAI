"""
Testes do Detector de Bandeiras dos Shinpan (Pendência I7).
O treino YOLO e a inferência em vídeo exigem GPU/pesos e não rodam aqui; testamos a lógica.
"""

import os
import json
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.vision import flag_detector as fd
from src.engine import weak_labels as wl
from src.utils import model_registry


def _kp(sx=500, sy=300, torso=100):
    """Pose COCO-17 mínima: ombros em (sx±20, sy), quadris um tronco abaixo."""
    kp = np.zeros((17, 2))
    kp[5], kp[6] = (sx - 20, sy), (sx + 20, sy)
    kp[11], kp[12] = (sx - 15, sy + torso), (sx + 15, sy + torso)
    return kp


class TestGeometryAndHint(unittest.TestCase):
    def test_crop_box_above_shoulders(self):
        x0, y0, x1, y1 = fd.crop_box(_kp(), 1920, 1080)
        self.assertLess(y0, 300 - 100)      # sobe acima da cabeça
        self.assertLess(y1, 300 + 100)      # não desce até o quadril
        self.assertLess(x0, 480)
        self.assertGreater(x1, 520)

    def test_crop_box_requires_shoulders_and_hips(self):
        kp = _kp()
        kp[11] = (0, 0)
        self.assertIsNone(fd.crop_box(kp, 1920, 1080))

    def test_color_hint(self):
        crop = np.full((128, 128, 3), 60, np.uint8)
        self.assertIsNone(fd.color_hint(crop))
        crop[20:40, 50:70] = (0, 0, 220)        # bandeira vermelha (BGR)
        self.assertEqual(fd.color_hint(crop), "vermelha")
        crop2 = np.full((128, 128, 3), 60, np.uint8)
        crop2[20:40, 50:70] = (245, 245, 245)
        self.assertEqual(fd.color_hint(crop2), "branca")


def _samples(spec, fps=4.0, duration=12.0):
    """spec: {pessoa: (x, cor, inicio_s, fim_s)} → amostras com a cor erguida entre início e fim."""
    out = []
    for i in range(int(duration * fps)):
        t = i / fps
        people = []
        for x, color, start, end in spec.values():
            cls = color if start <= t < end else "nenhuma"
            people.append(((float(x), 300.0), 100.0, cls))
        out.append((t, people))
    return out


class TestRaiseEvents(unittest.TestCase):
    def test_two_judges_raising_red_is_one_aka_point(self):
        ev = fd.raise_events(_samples({"a": (200, "vermelha", 5.0, 9.0), "b": (900, "vermelha", 5.5, 7.0),
                                       "c": (1600, "branca", 99, 99)}))
        self.assertEqual(len(ev), 1)
        self.assertEqual((ev[0]["cor"], ev[0]["kenshi"], ev[0]["juizes"]), ("vermelha", "AKA", 2))
        self.assertAlmostEqual(ev[0]["bandeira_s"], 5.0)
        self.assertAlmostEqual(ev[0]["golpe_estimado_s"], 4.0)

    def test_single_judge_is_not_a_point(self):
        self.assertEqual(fd.raise_events(_samples({"a": (200, "branca", 5.0, 9.0), "b": (900, "branca", 99, 99)})), [])

    def test_flag_already_up_or_held_is_not_a_new_raise(self):
        # Bandeira erguida desde o início (sem subida) e camiseta branca fixa não viram ponto
        ev = fd.raise_events(_samples({"a": (200, "branca", 0.0, 12.0), "b": (900, "branca", 0.0, 12.0)}))
        self.assertEqual(ev, [])

    def test_two_separate_points(self):
        ev = fd.raise_events(_samples({"a": (200, "vermelha", 2.0, 4.0), "b": (900, "vermelha", 2.2, 4.0),
                                       "c": (200, "nenhuma", 0, 0)}, duration=20.0)
                             + [(t + 20.0, p) for t, p in _samples({"a": (200, "branca", 3.0, 5.0),
                                                                     "b": (900, "branca", 3.4, 5.0)})])
        self.assertEqual([(e["kenshi"], round(e["bandeira_s"])) for e in ev], [("AKA", 2), ("SHIRO", 23)])


class TestDatasetAndRegistry(unittest.TestCase):
    def test_labels_queue_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            fd.save_dataset({
                "a": {"video_name": "v1.mp4", "tempo_s": 5, "arquivo": "x", "pista": None, "rotulo": None},
                "b": {"video_name": "v1.mp4", "tempo_s": 9, "arquivo": "x", "pista": "vermelha", "rotulo": None},
                "c": {"video_name": "v2.mp4", "tempo_s": 1, "arquivo": "x", "pista": None, "rotulo": "nenhuma"},
            }, tmp)
            self.assertEqual([q["id"] for q in fd.next_to_label(dataset_dir=tmp)], ["b", "a"])  # pista primeiro
            self.assertEqual(fd.set_labels({"b": "vermelha", "a": "pular", "zzz": "vermelha", "c": "invalido"}, tmp), 2)
            st = fd.dataset_stats(tmp)
            self.assertEqual((st["vermelha"], st["nenhuma"], st["pular"], st["sem_rotulo"]), (1, 1, 1, 0))
            self.assertIn("vermelha: 1/15", fd.check_enough_labels(st))

    def test_split_by_video_never_mixes_videos(self):
        items = {f"{v}_{i}": {"video_name": f"{v}.mp4", "rotulo": "nenhuma"} for v in "abcd" for i in range(5)}
        train, val = fd.split_by_video(items)
        self.assertTrue(train and val)
        self.assertFalse({items[k]["video_name"] for k in train} & {items[k]["video_name"] for k in val})

    def test_shared_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.pt")
            open(path, "w").close()
            self.assertIsNone(model_registry.get_active_model_path("x", tmp))
            model_registry.set_active_model("x", path, "teste", tmp)
            self.assertEqual(model_registry.get_active_model_path("x", tmp), path)
            model_registry.set_active_model("x", None, "teste", tmp)
            self.assertIsNone(model_registry.get_active_model_path("x", tmp))


class TestEvaluateAndApply(unittest.TestCase):
    def _gabarito(self, tmp):
        labels_path = os.path.join(tmp, "gabarito.json")
        manual = {"pontos_completos": True, "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:11"}]}
        wl.save_match_labels({"versao": 1, "videos": {
            "m1.mp4": manual, "m2.mp4": manual, "m3.mp4": manual,
            "nova.mp4": {"pontos": [], "pontos_completos": False},
        }}, labels_path)
        for n in ("m1.mp4", "m2.mp4", "m3.mp4", "nova.mp4"):
            open(os.path.join(tmp, n), "w").close()
        return labels_path

    def test_match_events_window(self):
        entry = {"pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:11"}]}
        ok = [{"kenshi": "AKA", "bandeira_s": 12.0}]
        wrong_color = [{"kenshi": "SHIRO", "bandeira_s": 12.0}]
        too_late = [{"kenshi": "AKA", "bandeira_s": 16.0}]
        self.assertEqual(fd.match_events_to_points(ok, entry), {"tp": 1, "fp": 0, "fn": 0})
        self.assertEqual(fd.match_events_to_points(wrong_color, entry), {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(fd.match_events_to_points(too_late, entry), {"tp": 0, "fp": 1, "fn": 1})

    def test_apply_only_when_detector_proved_reliable(self):
        good = lambda v: {"eventos": [{"kenshi": "AKA", "cor": "vermelha", "bandeira_s": 12.0, "golpe_estimado_s": 11.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = self._gabarito(tmp)
            ev = fd.evaluate_against_gabarito(tmp, labels_path, detector=good)
            self.assertEqual((ev["total"]["videos"], ev["total"]["f1"]), (3, 1.0))
            res = fd.apply_to_gabarito(tmp, ev, labels_path, detector=good)
            self.assertTrue(res["aplicado"])
            data = wl.load_match_labels(labels_path)["videos"]
            self.assertEqual(data["nova.mp4"]["origem"], fd.AUTO_ORIGIN)
            self.assertEqual(data["nova.mp4"]["pontos"][0]["kenshi"], "AKA")
            self.assertNotIn("origem", data["m1.mp4"])  # gabarito manual nunca é sobrescrito

    def test_apply_refused_when_detector_is_weak(self):
        bad = lambda v: {"eventos": [{"kenshi": "SHIRO", "cor": "branca", "bandeira_s": 40.0, "golpe_estimado_s": 39.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = self._gabarito(tmp)
            ev = fd.evaluate_against_gabarito(tmp, labels_path, detector=bad)
            res = fd.apply_to_gabarito(tmp, ev, labels_path, detector=bad)
            self.assertFalse(res["aplicado"])
            self.assertFalse(wl.is_entry_filled(wl.load_match_labels(labels_path)["videos"]["nova.mp4"]))

    def test_golden_rejects_automatic_gabarito(self):
        from src.engine.active_learning import GoldenBenchmark
        auto = {"pontos_completos": True, "origem": fd.AUTO_ORIGIN,
                "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:11"}]}
        with tempfile.TemporaryDirectory() as tmp:
            gb = GoldenBenchmark(dataset_path=os.path.join(tmp, "golden_dataset.json"))
            with self.assertRaises(ValueError):
                gb.freeze_match("auto.mp4", auto)
            gb.freeze_match("manual.mp4", dict(auto, origem="manual"))
            self.assertTrue(gb.is_golden_video("manual.mp4"))


if __name__ == "__main__":
    unittest.main()
