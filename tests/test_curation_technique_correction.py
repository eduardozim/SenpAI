"""
Testes da correção de golpe na Fila de Curadoria Ativa: o árbitro escolhe qual golpe houve (pré-marcado com o
identificado pelo sistema). Golpe trocado → feedback com o golpe corrigido, rótulo "tecnica" no Banco Único e
notas dos pilares fora do feedback (foram medidas para o golpe errado).
Só usa mocks: nada é gravado em data/ nem em config/.
"""

import unittest
from unittest import mock

SUB_SCORES = {"target_impact": 30.0, "fumikomi_sync": 70.0, "posture": 65.0, "zanshin": 50.0}


def _item(strike_type="TSUKI"):
    return {"id": "unc_tec_1", "profile_name": "normal", "strike_type": strike_type, "confidence": 0.5,
            "video_source": "luta.mp4", "tempo_golpe_s": 12.5,
            "strike_data": {"sub_scores": dict(SUB_SCORES), "total_score": 52.0, "timestamp_seconds": 12.5}}


class TestCurationTechniqueCorrection(unittest.TestCase):
    def _resolve(self, item, **kwargs):
        from src.engine.feedback_manager import FeedbackManager
        fm = FeedbackManager()
        with mock.patch.object(fm.uncertainty_sampler, "get_queue", return_value=[item]), \
             mock.patch.object(fm.uncertainty_sampler, "resolve_item", return_value=True) as resolve, \
             mock.patch.object(fm, "save_feedback") as save, \
             mock.patch("src.engine.label_bank.LabelBank.add_label") as add_label:
            res = fm.resolve_active_learning_item(item["id"], reviewer_dan=5, trigger_retrain=False, **kwargs)
        return res, save, add_label, resolve

    def test_corrected_technique_on_invalid_strike(self):
        res, save, add_label, _ = self._resolve(_item("TSUKI"), decision="INVALID_STRIKE", technique="MEN")
        save.assert_called_once()
        kw = save.call_args.kwargs
        self.assertEqual(kw["strike_type"], "MEN")
        self.assertEqual(kw["label"], "FP")
        self.assertEqual(kw["sub_scores"], {}, "Notas medidas para o golpe errado não entram na calibração")
        self.assertIn("TSUKI → MEN", kw["notes"])
        add_label.assert_called_once()
        lb = add_label.call_args.kwargs
        self.assertEqual((lb["origin"], lb["claim"], lb["value"], lb["time_sec"]), ("curadoria", "tecnica", "MEN", 12.5))
        self.assertTrue(res["technique_corrected"])
        self.assertEqual(res["technique"], "MEN")

    def test_corrected_technique_on_ippon(self):
        res, save, add_label, _ = self._resolve(_item("TSUKI"), decision="IPPON", technique="kote")
        self.assertEqual(save.call_args.kwargs["strike_type"], "KOTE")
        self.assertEqual(save.call_args.kwargs["label"], "TP")
        self.assertEqual(add_label.call_args.kwargs["value"], "KOTE")
        self.assertTrue(res["technique_corrected"])

    def test_same_technique_keeps_scores_and_adds_no_technique_label(self):
        res, save, add_label, _ = self._resolve(_item("MEN"), decision="IPPON", technique="MEN")
        self.assertEqual(save.call_args.kwargs["strike_type"], "MEN")
        self.assertEqual(save.call_args.kwargs["sub_scores"], SUB_SCORES)
        add_label.assert_not_called()
        self.assertFalse(res["technique_corrected"])

    def test_without_technique_behaves_as_before(self):
        res, save, add_label, _ = self._resolve(_item("TSUKI"), decision="IPPON")
        self.assertEqual(save.call_args.kwargs["strike_type"], "TSUKI")
        self.assertEqual(save.call_args.kwargs["sub_scores"], SUB_SCORES)
        add_label.assert_not_called()

    def test_no_strike_ignores_technique(self):
        res, save, add_label, _ = self._resolve(_item("TSUKI"), decision="NO_STRIKE", technique="MEN")
        self.assertEqual(save.call_args.kwargs["strike_type"], "TSUKI")
        add_label.assert_not_called()
        self.assertFalse(res["technique_corrected"])

    def test_invalid_technique_is_rejected_before_resolving(self):
        from src.engine.feedback_manager import FeedbackManager
        fm = FeedbackManager()
        with mock.patch.object(fm.uncertainty_sampler, "get_queue", return_value=[_item()]), \
             mock.patch.object(fm.uncertainty_sampler, "resolve_item") as resolve, \
             mock.patch.object(fm, "save_feedback") as save:
            with self.assertRaises(ValueError):
                fm.resolve_active_learning_item("unc_tec_1", decision="IPPON", technique="SHOMEN?", trigger_retrain=False)
        save.assert_not_called()
        resolve.assert_not_called()

    def test_auto_trainer_passes_technique_through(self):
        from src.engine.auto_trainer import auto_trainer
        with mock.patch.object(auto_trainer.feedback_mgr, "resolve_active_learning_item", return_value={}) as r:
            auto_trainer.resolve_active_learning_item("unc_tec_1", False, 5, "x", trigger_retrain=False,
                                                      cleanup_clip=True, decision="INVALID_STRIKE", technique="MEN")
        self.assertEqual(r.call_args.kwargs["technique"], "MEN")


if __name__ == "__main__":
    unittest.main()
