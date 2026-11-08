"""
Testes do Banco Único de Rótulos (Pendência I11).
"""

import os
import json
import tempfile
import unittest
from unittest import mock

from src.engine import weak_labels as wl
from src.engine.label_bank import (LabelBank, canonical_video_name, parse_time, build_training_reports,
                                   optimizer_feedback, ORIGIN_BASE_AUTHORITY)
from src.analytics.event_spotter import StrikeEvent


def _item(t, label, kenshi="KENSHI_AKA", **extra):
    return dict({"time_sec": t, "impact_frame": int(t * 30), "attacker_id": kenshi, "weak_label": label, "reason": "x"}, **extra)


def _fb(label, ts, video="luta.mp4", dan=5, **extra):
    return dict({"id": f"{video}_{ts}_{label}", "id_event": "ev", "video_name": video, "label": label, "timestamp": ts,
                 "reviewer_dan": dan, "sub_scores": {"target_impact": 80, "fumikomi_sync": 70, "posture": 70, "zanshin": 60},
                 "strike_type": "MEN", "notes": "", "decision_category": ""}, **extra)


class TestBasics(unittest.TestCase):
    def test_names_and_times(self):
        self.assertEqual(canonical_video_name("D:/x/upload_1793816081_Luta A.mp4"), "Luta A.mp4")
        self.assertEqual(parse_time("00:13.600"), 13.6)
        self.assertEqual(parse_time("13.6s"), 13.6)
        self.assertEqual(parse_time("1:02:03"), 3723.0)
        self.assertIsNone(parse_time("abc"))

    def test_add_validate_stats_and_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = LabelBank(os.path.join(tmp, "b.json"))
            with self.assertRaises(ValueError):
                bank.add_label(origin="boato", video_name="a.mp4", claim="ponto", value=True)
            bank.add_label(origin="revisao", video_name="upload_1_a.mp4", claim="ponto", value=True, time_sec=3.0)
            bank.add_label(origin="internet", video_name="b.mp4", claim="tecnica", value="KOTE", time_sec=9.0, confidence=0.5)
            st = bank.stats()
            self.assertEqual((st["total"], st["videos"]), (2, 2))
            self.assertEqual(bank.labels("a.mp4")[0]["video_name"], "a.mp4")
            self.assertEqual(bank.remove_origin("internet"), 1)

    def test_authority_hierarchy(self):
        shinpan = {"origin": "revisao", "reviewer_weight": 4.5}
        internet = {"origin": "internet", "reviewer_weight": 1.0}
        self.assertGreater(LabelBank.authority(shinpan), LabelBank.authority(internet))
        self.assertEqual(LabelBank.authority({"origin": "gabarito"}), ORIGIN_BASE_AUTHORITY["gabarito"])


class TestSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bank = LabelBank(os.path.join(self.tmp.name, "bank.json"))
        self.labels_path = os.path.join(self.tmp.name, "gabarito.json")
        self.fb_path = os.path.join(self.tmp.name, "feedback.json")
        wl.save_match_labels({"versao": 1, "videos": {
            "luta.mp4": {"pontos_completos": True, "pontos": [{"kenshi": "AKA", "tecnica": "KOTE", "tempo": "00:05"},
                                                             {"kenshi": "SHIRO", "tecnica": "HANSOKU"}]},
            "auto.mp4": {"pontos_completos": True, "origem": "automatico_bandeiras", "pontos": [{"kenshi": "SHIRO", "tecnica": "MEN", "tempo": "00:20"}]},
            "vazia.mp4": {"pontos": [], "pontos_completos": False},
        }}, self.labels_path)
        with open(self.fb_path, "w", encoding="utf-8") as f:
            json.dump([
                _fb("TP", "00:30.000"),
                _fb("FP", "00:40.000", dan="shinpan", is_shinpan_decision=True),
                _fb("FP", "00:45.000", decision_category="NO_STRIKE", id_event="unc_1", notes="Curadoria Ativa"),
                _fb("FP", "00:50.000", decision_category="rotulo_fraco"),   # derivado do gabarito: ignorado
                _fb("TP", "", ),                                            # sem tempo: ignorado
            ], f)

    def tearDown(self):
        self.tmp.cleanup()

    def _sync(self):
        return self.bank.sync_from_sources(labels_path=self.labels_path, feedback_path=self.fb_path)

    def test_sync_reads_every_source_once(self):
        counts = self._sync()
        self.assertEqual((counts["ignorados_derivados"], counts["ignorados_sem_tempo"]), (1, 1))
        st = self.bank.stats()
        # gabarito: ponto + técnica + "modalidade = luta"; bandeiras: ponto + "modalidade = luta"
        self.assertEqual(st["por_origem"], {"gabarito": 3, "bandeiras": 2, "revisao": 2, "curadoria": 1})
        self.assertEqual(sorted(x["video_name"] for x in self.bank.modality_labels() if x["value"] == "luta"), ["auto.mp4", "luta.mp4"])
        tech = self.bank.labels("luta.mp4", claim="tecnica")
        self.assertEqual([t["value"] for t in tech], ["KOTE"])
        golpe = self.bank.labels("luta.mp4", claim="golpe")
        self.assertEqual((golpe[0]["value"], golpe[0]["origin"]), (False, "curadoria"))
        shinpan_fp = [x for x in self.bank.labels("luta.mp4", claim="ponto") if x["time_sec"] == 40.0][0]
        self.assertEqual(shinpan_fp["reviewer_weight"], 4.5)

    def test_sync_is_idempotent_and_follows_source_removals(self):
        self._sync()
        self._sync()
        self.assertEqual(self.bank.stats()["total"], 8)
        with open(self.fb_path, "w", encoding="utf-8") as f:
            json.dump([_fb("TP", "00:30.000")], f)
        self._sync()
        self.assertEqual(self.bank.stats()["por_origem"].get("curadoria", 0), 0)
        self.assertEqual(self.bank.stats()["por_origem"]["revisao"], 1)

    def test_direct_labels_survive_sync(self):
        self.bank.add_label(origin="revisao", video_name="luta.mp4", claim="ponto", value=False, time_sec=60.0, ref="direto")
        self._sync()
        self.assertTrue(any(x.get("ref") == "direto" for x in self.bank.labels("luta.mp4")))


class TestTrainingModeCorrections(unittest.TestCase):
    def test_segments_become_modality_labels_and_latest_edit_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = LabelBank(os.path.join(tmp, "b.json"))
            n = bank.record_training_segments("upload_3_Treino.mp4", [
                {"start_time": "00:00", "end_time": "03:00", "modality_key": "kirikaeshi"},
                {"start_time": "03:00", "end_time": "05:30", "modality_key": "waza_geiko"},
                {"start_time": "", "end_time": "06:00", "modality_key": "kihon"},   # sem início: ignorado
            ], reviewer_weight=5)
            self.assertEqual(n, 2)
            labels = bank.modality_labels()
            self.assertEqual([(x["video_name"], x["value"], x["time_sec"], x["fim_s"]) for x in labels],
                             [("Treino.mp4", "kirikaeshi", 0.0, 180.0), ("Treino.mp4", "waza_geiko", 180.0, 330.0)])
            # Nova edição substitui a anterior
            bank.record_training_segments("Treino.mp4", [{"start_time": "00:00", "end_time": "05:30", "modality_key": "suburi"}])
            self.assertEqual([x["value"] for x in bank.modality_labels()], ["suburi"])

    def test_video_type_in_gabarito_becomes_modality(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = LabelBank(os.path.join(tmp, "b.json"))
            labels_path = os.path.join(tmp, "g.json")
            fb_path = os.path.join(tmp, "f.json")
            with open(fb_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            wl.save_match_labels({"versao": 1, "videos": {
                "internet_kk.mp4": {"tipo": "kakari_geiko", "ignorar": True, "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:03"}]},
                "exame.mp4": {"tipo": "shinsa", "ignorar": True, "pontos": []},
                "sg.mp4": {"tipo": "shiai_geiko", "pontos": [], "pontos_completos": False},
                "lixo.mp4": {"tipo": "descartar", "ignorar": True, "pontos": []},
                "antigo.mp4": {"ignorar": True, "pontos": []},
            }}, labels_path)
            # O título do vídeo da internet tinha sugerido outra modalidade: a escolha no gabarito vence
            bank.add_label(origin="internet", video_name="internet_kk.mp4", claim="modalidade", value="kirikaeshi",
                           time_sec=0.0, ref="titulo", confidence=0.5)
            bank.add_label(origin="internet", video_name="sg2.mp4", claim="modalidade", value="shiai_geiko", time_sec=0.0, ref="titulo")
            bank.sync_from_sources(labels_path=labels_path, feedback_path=fb_path)
            got = {x["video_name"]: x["value"] for x in bank.modality_labels()}
            self.assertEqual(got, {"internet_kk.mp4": "kakari_geiko", "exame.mp4": "shinsa", "sg.mp4": "luta", "sg2.mp4": "luta"})
            self.assertEqual(bank.labels("internet_kk.mp4", claim="ponto"), [])   # treino: pontos não entram
        self.assertEqual(wl.entry_video_type({"ignorar": True}), wl.VIDEO_TYPE_DISCARD)
        self.assertEqual(wl.entry_video_type({"tipo": "shiai_geiko"}), wl.VIDEO_TYPE_LUTA)
        self.assertTrue(wl.is_entry_ignored({"tipo": "ji_geiko"}))
        self.assertFalse(wl.is_entry_discarded({"tipo": "ji_geiko"}))
        self.assertFalse(wl.is_entry_ignored({"pontos": []}))

    def test_whole_video_modality(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = LabelBank(os.path.join(tmp, "b.json"))
            self.assertEqual(bank.record_training_segments("t.mp4", [], whole_video_modality="ashi_sabaki"), 1)
            self.assertEqual(bank.modality_labels()[0]["value"], "ashi_sabaki")
            self.assertEqual(bank.record_training_segments("t.mp4", []), 0)
            self.assertEqual(bank.modality_labels(), [])


class TestApplyCorrections(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bank = LabelBank(os.path.join(self.tmp.name, "bank.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_correction_labels_unknown_and_technique(self):
        items = [_item(10.0, wl.UNKNOWN), _item(20.0, wl.UNKNOWN)]
        corr = [{"id": "c1", "origin": "revisao", "video_name": "v", "claim": "ponto", "value": True, "time_sec": 10.4, "reviewer_weight": 5},
                {"id": "c2", "origin": "curadoria", "video_name": "v", "claim": "golpe", "value": False, "time_sec": 19.5, "reviewer_weight": 3},
                {"id": "c3", "origin": "revisao", "video_name": "v", "claim": "tecnica", "value": "DO", "time_sec": 10.0, "reviewer_weight": 5}]
        out, conflicts = self.bank.apply_corrections(items, corr)
        self.assertEqual([o["weak_label"] for o in out], [wl.POSITIVE, wl.NEG_CORRECTION])
        self.assertEqual(out[0]["tecnica_corrigida"], "DO")
        self.assertEqual(conflicts, [])

    def test_strong_reviewer_overrides_gabarito_weak_one_becomes_conflict(self):
        items = [_item(10.0, wl.POSITIVE)]
        weak = [{"id": "w", "origin": "revisao", "video_name": "v", "claim": "ponto", "value": False, "time_sec": 10.0, "reviewer_weight": 4.5}]
        strong = [{"id": "s", "origin": "revisao", "video_name": "v", "claim": "ponto", "value": False, "time_sec": 10.0, "reviewer_weight": 8}]
        out, conflicts = self.bank.apply_corrections(items, weak)
        self.assertEqual((out[0]["weak_label"], len(conflicts)), (wl.POSITIVE, 1))
        out, conflicts = self.bank.apply_corrections(items, strong)
        self.assertEqual((out[0]["weak_label"], len(conflicts)), (wl.NEG_CORRECTION, 0))

    def test_kenshi_must_match_when_given(self):
        items = [_item(10.0, wl.UNKNOWN, kenshi="KENSHI_SHIRO")]
        corr = [{"id": "c", "origin": "revisao", "video_name": "v", "claim": "ponto", "value": True, "time_sec": 10.0, "kenshi": "AKA", "reviewer_weight": 5}]
        out, _ = self.bank.apply_corrections(items, corr)
        self.assertEqual(out[0]["weak_label"], wl.UNKNOWN)


class TestTrainingReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.utils.demo_generator import generate_demo_kendo_video
        cls.tmp = tempfile.TemporaryDirectory()
        for n in ("so_correcao.mp4", "treino.mp4"):
            generate_demo_kendo_video(os.path.join(cls.tmp.name, n), duration_sec=3, fps=30)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_video_with_only_corrections_teaches_all_models(self):
        """Uma correção feita na revisão (sem gabarito) vira exemplo do filtro, da técnica e da calibração."""
        from src.pipeline import SenpAIPipeline
        labels_path = os.path.join(self.tmp.name, "gabarito.json")
        wl.save_match_labels({"versao": 1, "videos": {"treino.mp4": {"ignorar": True, "pontos": []}}}, labels_path)
        bank = LabelBank(os.path.join(self.tmp.name, "bank.json"))
        bank.add_label(origin="revisao", video_name="upload_9_so_correcao.mp4", claim="ponto", value=True, time_sec=40 / 30, kenshi="AKA", reviewer_weight=5)
        bank.add_label(origin="revisao", video_name="so_correcao.mp4", claim="tecnica", value="KOTE", time_sec=40 / 30, kenshi="AKA", reviewer_weight=5)
        bank.add_label(origin="revisao", video_name="treino.mp4", claim="ponto", value=True, time_sec=1.0, reviewer_weight=5)  # vídeo ignorado

        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        fake = StrikeEvent("MEN", start_frame=30, impact_frame=40, end_frame=60, fps=30.0)
        with mock.patch.object(pipeline.event_spotter, "detect_strikes", return_value=[fake]):
            training = build_training_reports(pipeline, self.tmp.name, labels_path=labels_path, bank=bank)

        self.assertEqual([r["video_name"] for r in training["reports"]], ["so_correcao.mp4"])
        report = training["reports"][0]
        cands = [c for c in report["candidates"]["items"] if c["weak_label"] == wl.POSITIVE]
        self.assertTrue(cands)
        self.assertEqual(cands[0]["tecnica_corrigida"], "KOTE")
        self.assertEqual(cands[0]["origem_rotulo"], "revisao")
        samples = optimizer_feedback(training, bank=bank)
        self.assertTrue(any(s["label"] == "TP" and s["origem"] == "revisao" for s in samples))

    def test_optimizer_accepts_bank_samples(self):
        from src.engine.feedback_manager import FeedbackManager, DEFAULT_CALIBRATION_PROFILES
        fm = FeedbackManager()
        samples = [{"label": "TP", "sub_scores": {"target_impact": 90, "fumikomi_sync": 85, "posture": 85, "zanshin": 80},
                    "video_name": "a.mp4", "reviewer_dan": 6},
                   {"label": "FP", "sub_scores": {"target_impact": 40, "fumikomi_sync": 30, "posture": 40, "zanshin": 20},
                    "video_name": "b.mp4", "reviewer_dan": 6}]
        with mock.patch.object(fm, "load_feedback", return_value=[]) as load:
            _, stats = fm.optimize_profile_config("normal", DEFAULT_CALIBRATION_PROFILES["normal"], feedback_list=samples)
        load.assert_not_called()
        self.assertNotEqual(stats["status"], "no_data")


if __name__ == "__main__":
    unittest.main()
