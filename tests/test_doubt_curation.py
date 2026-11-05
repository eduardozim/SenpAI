"""
Testes da Curadoria de Todas as Dúvidas (Pendência I12).
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from src.engine.active_learning import UncertaintySampler, DOUBT_TYPES
from src.engine.feedback_manager import FeedbackManager
from src.engine.label_bank import LabelBank


def _sd(t, **extra):
    return dict({"strike_type": "MEN", "timestamp_seconds": t, "impact_frame": int(t * 30), "fps": 30.0}, **extra)


class _Base(unittest.TestCase):
    def setUp(self):
        self.sampler = UncertaintySampler()
        self.sampler.clear_queue(delete_clips=False)
        self.bank = LabelBank()
        if os.path.exists(self.bank.path):
            os.remove(self.bank.path)

    def tearDown(self):
        self.sampler.clear_queue(delete_clips=False)
        if os.path.exists(self.bank.path):
            os.remove(self.bank.path)


class TestGenericQueue(_Base):
    def test_types_priority_order_and_dedupe(self):
        with self.assertRaises(ValueError):
            self.sampler.enqueue_doubt("palpite", "luta.mp4", _sd(1.0), 0.5, "x")
        low = self.sampler.enqueue_doubt("tecnica", "luta.mp4", _sd(10.0), uncertainty=0.5, impact=0.6, reason="técnica")
        high = self.sampler.enqueue_doubt("regra", "luta.mp4", _sd(20.0, kenshi="AKA"), uncertainty=0.9, impact=1.0, reason="yame")
        self.assertEqual((low["prioridade"], high["prioridade"]), (0.3, 0.9))
        self.assertEqual(high["kenshi"], "AKA")
        self.assertEqual(high["tempo_golpe_s"], 20.0)
        # Mesma dúvida (vídeo, tipo e instante ±1 s) não entra de novo
        self.assertIsNone(self.sampler.enqueue_doubt("regra", "luta.mp4", _sd(20.4), 0.9, "yame de novo"))
        # Outro tipo no mesmo instante entra
        self.assertIsNotNone(self.sampler.enqueue_doubt("conflito", "luta.mp4", _sd(20.4), 0.95, "conflito"))
        order = [it["tipo_duvida"] for it in self.sampler.get_queue(status="pending_curation")]
        self.assertEqual(order, ["conflito", "regra", "tecnica"])

    def test_score_doubts_keep_working_with_impact(self):
        valid = self.sampler.evaluate_and_enqueue(_sd(5.0, is_valid=True), confidence=0.55, video_source="luta.mp4", enrich_with_llm=False)
        invalid = self.sampler.evaluate_and_enqueue(_sd(9.0, is_valid=False), confidence=0.55, video_source="luta.mp4", enrich_with_llm=False)
        self.assertEqual(valid["tipo_duvida"], "pontuacao")
        self.assertGreater(valid["prioridade"], invalid["prioridade"])
        self.assertIsNone(self.sampler.evaluate_and_enqueue(_sd(30.0), confidence=0.95, video_source="luta.mp4", enrich_with_llm=False))
        self.assertIn("pontuacao", DOUBT_TYPES)


class TestDecisionsReachTheBank(_Base):
    def test_point_decision_carries_strike_time_to_bank(self):
        """Antes, o 'timestamp' gravado era a hora da fila e o banco descartava o rótulo."""
        fm = FeedbackManager()
        item = self.sampler.evaluate_and_enqueue(_sd(13.6, is_valid=True, sub_scores={"target_impact": 80, "fumikomi_sync": 70, "posture": 70, "zanshin": 60}),
                                                 confidence=0.5, video_source="luta.mp4", enrich_with_llm=False)
        with mock.patch.object(fm, "save_feedback", wraps=fm.save_feedback) as save:
            fm.resolve_active_learning_item(item["id"], decision="INVALID_STRIKE", reviewer_dan=6, trigger_retrain=False)
        self.assertEqual(save.call_args.kwargs["timestamp"], "00:13.600")
        counts = self.bank.sync_from_sources()
        self.assertEqual(counts["ignorados_sem_tempo"], 0)
        labels = self.bank.labels("luta.mp4", claim="ponto")
        self.assertEqual((labels[0]["origin"], labels[0]["value"], labels[0]["time_sec"]), ("curadoria", False, 13.6))

    def test_technique_decision_goes_to_bank(self):
        fm = FeedbackManager()
        item = self.sampler.enqueue_doubt("tecnica", "upload_7_luta.mp4", _sd(8.0, kenshi="SHIRO"), 0.6, "técnica incerta")
        self.assertTrue(fm.resolve_technique_doubt(item["id"], "kote", reviewer_dan=7))
        label = self.bank.labels("luta.mp4", claim="tecnica")[0]
        self.assertEqual((label["value"], label["origin"], label["kenshi"], label["time_sec"]), ("KOTE", "curadoria", "SHIRO", 8.0))
        self.assertEqual(label["reviewer_weight"], 7.0)
        self.assertEqual(self.sampler.get_queue(status="pending_curation"), [])
        with self.assertRaises(ValueError):
            fm.resolve_technique_doubt("qualquer", "chute")


class TestPipelineDoubts(_Base):
    def test_filter_technique_and_rule_doubts(self):
        from src.pipeline import SenpAIPipeline
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")

        def ev(frame, kenshi="KENSHI_AKA", type_="MEN"):
            return SimpleNamespace(impact_frame=frame, attacker_id=kenshi, type=type_)

        def analyzed(frame, kenshi, valid=True):
            return {"event_info": {"impact_frame": frame, "attacker_id": kenshi, "type": "MEN"}, "evaluation": {"is_valid": valid}}

        events = [analyzed(300, "KENSHI_AKA"), analyzed(312, "KENSHI_SHIRO"),   # 0,4 s: ai-uchi
                  analyzed(900, "KENSHI_AKA"), analyzed(960, "KENSHI_AKA"),     # 2 s depois: após o yame
                  analyzed(1500, "KENSHI_SHIRO", valid=False)]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video = f.name
        try:
            with mock.patch("src.utils.strike_clip_manager.extract_strike_subclip", return_value=None):
                added = pipeline._enqueue_extra_doubts(
                    video, 30.0, filter_doubts=[(ev(600), 0.52)],
                    technique_doubts=[(ev(700, type_="DO"), {"tecnica": "DO", "probabilidade": 0.41, "probabilidades": {"DO": 0.41, "MEN": 0.39}})],
                    analyzed_events=events, filter_threshold=0.5)
        finally:
            os.remove(video)
        queue = self.sampler.get_queue(status="pending_curation")
        kinds = sorted(it["tipo_duvida"] for it in queue)
        self.assertEqual(added, 4)
        self.assertEqual(kinds, ["golpe_filtro", "regra", "regra", "tecnica"])
        reasons = " ".join(it["motivo"] for it in queue)
        self.assertIn("ai-uchi", reasons)
        self.assertIn("após o yame", reasons)


class TestConflictsGoToCuration(_Base):
    def test_enqueue_conflicts(self):
        from src.engine.training_cycle import enqueue_conflicts
        n = enqueue_conflicts([{"video_name": "luta.mp4", "time_sec": 12.0, "atual": "POSITIVO", "correcao": "NEGATIVO_CORRECAO", "origem": "revisao", "id": "x"}], "pasta")
        self.assertEqual(n, 1)
        item = self.sampler.get_queue(status="pending_curation")[0]
        self.assertEqual(item["tipo_duvida"], "conflito")
        self.assertIn("não é ponto", item["motivo"])
        self.assertEqual(enqueue_conflicts([], "pasta"), 0)


if __name__ == "__main__":
    unittest.main()
