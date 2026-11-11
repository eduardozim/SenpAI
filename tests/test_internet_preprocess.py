"""
Pré-processamento das lutas baixadas da internet: análise uma vez por vídeo, dúvidas na curadoria, e a
resposta da curadoria volta ao treino como revisão (com o dan).
"""

import os
import json
import shutil
import tempfile
import unittest
from unittest import mock

from src.engine import weak_labels as wl
from src.engine import internet_collector as ic
from src.engine import internet_preprocess as ip
from src.engine.label_bank import LabelBank


def _touch(folder, name):
    open(os.path.join(folder, name), "wb").close()


class _FakePipeline:
    """Cada análise cria uma dúvida na fila do vídeo analisado (como o pipeline real faz com enqueue_doubts=True)."""

    def __init__(self, queue):
        self.queue, self.calls = queue, []

    def process_video(self, video, **kw):
        self.calls.append((os.path.basename(video), kw))
        self.queue.append({"status": "pending_curation", "video_source": video.replace("\\", "/")})
        return {"events": [{}, {}, {}]}


class TestInternetPreprocess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.vdir = self.tmp.name
        self.labels = os.path.join(self.vdir, "gabarito.json")
        for n in ("internet_a.mp4", "internet_b.mp4", "internet_treino.mp4", "internet_lixo.mp4",
                  "internet_semtitulo.mp4", "minha_luta.mp4"):
            _touch(self.vdir, n)
        wl.save_match_labels({"versao": 1, "videos": {
            "internet_b.mp4": {"tipo": "luta"},
            "internet_lixo.mp4": {"tipo": "descartar", "ignorar": True},
            "internet_treino.mp4": {"tipo": "kirikaeshi", "ignorar": True},
        }}, self.labels)
        shutil.rmtree(ic.get_root_dir(), ignore_errors=True)
        catalog = {"a": {"arquivo": "internet_a.mp4", "modalidade_titulo": "luta"},
                   "s": {"arquivo": "internet_semtitulo.mp4", "modalidade_titulo": None}}
        self.cat = mock.patch.object(ic, "load_catalog", return_value=catalog)
        self.cat.start()
        self.queue = []
        self.q = mock.patch.object(ip, "_pending_doubts", side_effect=lambda: list(self.queue))
        self.q.start()

    def tearDown(self):
        self.q.stop()
        self.cat.stop()
        shutil.rmtree(ic.get_root_dir(), ignore_errors=True)
        self.tmp.cleanup()

    def test_only_internet_fights_are_selected(self):
        names = [os.path.basename(v) for v in ip.internet_fights(self.vdir, self.labels)]
        self.assertEqual(names, ["internet_a.mp4", "internet_b.mp4"])   # título "luta" ou tipo luta no gabarito

    def test_each_fight_is_analysed_once_and_doubts_are_counted(self):
        pipe = _FakePipeline(self.queue)
        res = ip.preprocess_internet_videos(pipe, self.vdir, self.labels)
        self.assertEqual((res["processados"], res["duvidas"], res["restantes"]), (["internet_a.mp4", "internet_b.mp4"], 2, 0))
        kw = pipe.calls[0][1]
        self.assertEqual((kw["record_learning_side_effects"], kw["enqueue_doubts"], kw["use_feature_cache"]), (False, True, True))
        again = ip.preprocess_internet_videos(pipe, self.vdir, self.labels)
        self.assertEqual(again["processados"], [])
        st = ip.status(self.vdir, self.labels)
        self.assertEqual((st["lutas"], st["preprocessadas"], st["pendentes"], st["golpes_analisados"], st["duvidas_abertas"]),
                         (2, 2, 0, 6, 2))

    def test_stops_when_curation_queue_is_almost_full(self):
        from src.engine.active_learning import MAX_QUEUE_ITEMS
        self.queue.extend({"status": "pending_curation", "video_source": "x.mp4"} for _ in range(MAX_QUEUE_ITEMS - ip.QUEUE_ROOM_RESERVE))
        res = ip.preprocess_internet_videos(_FakePipeline(self.queue), self.vdir, self.labels)
        self.assertEqual((res["processados"], res["restantes"]), ([], 2))
        self.assertIn("fila de curadoria quase cheia", res["motivo_parada"])

    def test_failure_is_reported_and_video_stays_pending(self):
        pipe = mock.Mock()
        pipe.process_video.side_effect = RuntimeError("vídeo corrompido")
        res = ip.preprocess_internet_videos(pipe, self.vdir, self.labels, limit=1)
        self.assertEqual(res["falhas"][0]["video_name"], "internet_a.mp4")
        self.assertEqual(res["restantes"], 2)


    def test_second_process_waits_for_the_first(self):
        os.makedirs(ic.get_root_dir(), exist_ok=True)
        with open(os.path.join(ic.get_root_dir(), ip.LOCK_FILE), "w", encoding="utf-8") as f:
            json.dump({"pid": os.getppid()}, f)                 # outro processo vivo
        res = ip.preprocess_internet_videos(_FakePipeline(self.queue), self.vdir, self.labels)
        self.assertEqual(res["processados"], [])
        self.assertIn("já há um pré-processamento rodando", res["motivo_parada"])
        with open(os.path.join(ic.get_root_dir(), ip.LOCK_FILE), "w", encoding="utf-8") as f:
            json.dump({"pid": 999999999}, f)                    # processo que já morreu: trava velha
        res = ip.preprocess_internet_videos(_FakePipeline(self.queue), self.vdir, self.labels)
        self.assertEqual(len(res["processados"]), 2)
        self.assertFalse(os.path.exists(os.path.join(ic.get_root_dir(), ip.LOCK_FILE)))


class TestCurationAnswerTrainsTheModel(unittest.TestCase):
    def test_curation_decision_on_internet_video_becomes_a_review_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = LabelBank(os.path.join(tmp, "bank.json"))
            fb = os.path.join(tmp, "fb.json")
            labels = os.path.join(tmp, "g.json")
            wl.save_match_labels({"versao": 1, "videos": {}}, labels)
            with open(fb, "w", encoding="utf-8") as f:
                json.dump([{"id": "1", "id_event": "unc_1", "video_name": "data/videos/internet_a.mp4", "label": "TP",
                            "timestamp": "00:12.500", "reviewer_dan": 4, "decision_category": "IPPON",
                            "notes": "Homologado via Curadoria Ativa", "sub_scores": {"target_impact": 80}}], f)
            bank.sync_from_sources(labels_path=labels, feedback_path=fb)
            corr = bank.corrections_for("internet_a.mp4")
            self.assertEqual([(c["origin"], c["value"], c["time_sec"], c["reviewer_weight"]) for c in corr],
                             [("curadoria", True, 12.5, 4.0)])
            self.assertIn("internet_a.mp4", bank.labeled_videos(labels))   # entra no próximo ciclo de treino


class TestCycleStep(unittest.TestCase):
    def test_step_reports_skip_and_summary(self):
        from src.engine import training_cycle as tc
        empty = {"processados": [], "duvidas": 0, "falhas": [], "restantes": 0, "motivo_parada": None}
        with mock.patch.object(ip, "preprocess_internet_videos", return_value=empty):
            self.assertEqual(tc.step_preprocess(mock.Mock(), "pasta")["status"], "pulado")
        done = dict(empty, processados=["internet_a.mp4"], duvidas=3, restantes=5)
        with mock.patch.object(ip, "preprocess_internet_videos", return_value=done):
            res = tc.step_preprocess(mock.Mock(), "pasta")
        self.assertEqual((res["status"], res["duvidas"], res["restantes"]), ("ok", 3, 5))
        self.assertIn("3 dúvida(s)", res["detalhe"])


if __name__ == "__main__":
    unittest.main()
