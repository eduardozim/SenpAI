"""
Testes do Ciclo de Treino Real (Pendência I8).
"""

import os
import time
import tempfile
import unittest
from unittest import mock

from src.engine import training_cycle as tc


def _ok(**extra):
    return dict({"status": "ok", "detalhe": "ok"}, **extra)


class TestCycle(unittest.TestCase):
    def test_runs_all_steps_records_history_and_survives_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist = os.path.join(tmp, "ciclos.jsonl")
            seen = []
            with mock.patch.object(tc, "step_cache", return_value=_ok()),                  mock.patch.object(tc, "step_preprocess", return_value=_ok()), \
                 mock.patch.object(tc, "step_labels", side_effect=RuntimeError("sem GPU")), \
                 mock.patch.object(tc, "step_pretrain", return_value=_ok()), \
                 mock.patch.object(tc, "step_strike_filter", return_value=_ok(promovido=True)), \
                 mock.patch.object(tc, "step_technique", return_value=_ok(promovido=False)),                  mock.patch.object(tc, "step_modality", return_value=_ok(promovido=False)), \
                 mock.patch.object(tc, "step_suggest_modality", return_value=_ok()), \
                 mock.patch.object(tc, "step_calibration", return_value=_ok()), \
                 mock.patch.object(tc, "step_reference", return_value=_ok()), \
                 mock.patch.object(tc, "step_measure", return_value=_ok(total={"f1": 0.5})):
                rec = tc.run_training_cycle(tmp, pipeline=mock.Mock(), history_path=hist,
                                            progress=lambda k, n: seen.append(k))
            self.assertEqual(seen, ["cache", "preprocessamento", "rotulos", "pretreino", "filtro", "tecnica", "modalidade",
                                    "sugestao", "calibracao", "referencia", "medicao"])
            self.assertEqual(rec["etapas"]["rotulos"]["status"], "falhou")
            self.assertIn("sem GPU", rec["etapas"]["rotulos"]["detalhe"])
            self.assertEqual(rec["etapas"]["medicao"]["status"], "ok")   # seguiu depois da falha
            self.assertTrue(rec["promoveu_modelo"])
            self.assertEqual(len(tc.load_cycle_history(hist)), 1)

    def test_cache_step_skips_ignored_videos(self):
        from src.engine import weak_labels as wl
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = os.path.join(tmp, "gabarito.json")
            wl.save_match_labels({"versao": 1, "videos": {
                "luta.mp4": {"pontos": [], "pontos_completos": False},
                "Treino 1.mp4": {"pontos": [], "ignorar": True},
            }}, labels_path)
            videos = [os.path.join(tmp, "luta.mp4"), os.path.join(tmp, "Treino 1.mp4")]
            pipeline = mock.Mock(vision_model="yolov26")
            with mock.patch("src.utils.feature_cache.build_cache_key", return_value="k"), \
                 mock.patch("src.utils.feature_cache.load_match_features", return_value=None):
                res = tc.step_cache(pipeline, videos, labels_path=labels_path)
            pipeline.extract_match_features.assert_called_once_with(videos[0])
            self.assertEqual((res["novos"], res["ignorados"]), (1, 1))
            self.assertIn("Não aproveitável", res["detalhe"])   # "não é luta" antigo = descartado

    def test_ignored_entry_is_never_filled_golden_or_flag_target(self):
        from src.engine import weak_labels as wl
        from src.engine.active_learning import GoldenBenchmark
        ignored = {"ignorar": True, "pontos_completos": True, "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:05"}]}
        self.assertTrue(wl.is_entry_ignored(ignored))
        self.assertFalse(wl.is_entry_filled(ignored))
        with tempfile.TemporaryDirectory() as tmp:
            gb = GoldenBenchmark(dataset_path=os.path.join(tmp, "golden_dataset.json"))
            with self.assertRaises(ValueError):
                gb.freeze_match("treino.mp4", ignored)

    def test_strike_filter_step_skipped_without_data(self):
        pipeline = mock.Mock(strike_filter="anterior")
        with mock.patch("src.analytics.strike_classifier.build_dataset", return_value={"rows": [], "y": [], "groups": [], "skipped": []}):
            res = tc.step_strike_filter(pipeline, "pasta")
        self.assertEqual(res["status"], "pulado")
        self.assertIn("Dados insuficientes", res["detalhe"])
        self.assertEqual(pipeline.strike_filter, "anterior")  # nada muda sem promoção

    def test_strike_filter_step_reloads_filter_after_promotion(self):
        pipeline = mock.Mock(strike_filter=None)
        model = {"validacao_por_video": {"com_filtro": {"f1": 0.7}}}
        with mock.patch("src.analytics.strike_classifier.build_dataset", return_value={}), \
             mock.patch("src.analytics.strike_classifier.train_model", return_value=model), \
             mock.patch("src.analytics.strike_classifier.save_model", return_value="m.json"), \
             mock.patch("src.analytics.strike_classifier.promote_if_better", return_value={"promovido": True, "motivo": "F1 0.3 → 0.6"}), \
             mock.patch("src.analytics.strike_classifier.StrikeFilter.load_active", return_value="novo_filtro"):
            res = tc.step_strike_filter(pipeline, "pasta")
        self.assertTrue(res["promovido"])
        self.assertEqual(pipeline.strike_filter, "novo_filtro")

    def test_measure_step_skipped_without_golden(self):
        with mock.patch("src.engine.active_learning.GoldenBenchmark.load_matches", return_value={}):
            res = tc.step_measure(mock.Mock(), "pasta")
        self.assertEqual(res["status"], "pulado")


class TestBackgroundThread(unittest.TestCase):
    def test_single_cycle_at_a_time(self):
        seen_detail = []
        def slow_cycle(videos_dir, device_preference="gpu", progress=None, detail=None):
            progress("cache", tc.STEP_NAMES["cache"])
            detail("vídeo 1 de 2 sem poses no cache: a.mp4")
            seen_detail.append(tc.get_cycle_state()["detalhe"])
            time.sleep(0.3)
            return {"id": "x"}

        with mock.patch.object(tc, "run_training_cycle", side_effect=slow_cycle):
            self.assertTrue(tc.start_cycle_thread("pasta"))
            self.assertFalse(tc.start_cycle_thread("pasta"))
            for _ in range(50):
                if not tc.get_cycle_state()["rodando"]:
                    break
                time.sleep(0.05)
        state = tc.get_cycle_state()
        self.assertFalse(state["rodando"])
        self.assertEqual(state["resultado"], {"id": "x"})
        self.assertEqual(seen_detail, ["vídeo 1 de 2 sem poses no cache: a.mp4"])

    def test_running_text_shows_step_detail_and_time(self):
        import datetime
        from src.utils.training_cycle_panel import running_text
        ago = (datetime.datetime.now() - datetime.timedelta(minutes=3)).isoformat(timespec="seconds")
        text = running_text({"etapa": "Cache de poses (etapa 1 de 9)", "detalhe": "vídeo 3 de 24 sem poses no cache: x.mp4",
                             "etapa_inicio": ago, "inicio": ago})
        self.assertIn("etapa 1 de 9", text)
        self.assertIn("vídeo 3 de 24", text)
        self.assertIn("3 min", text)


class TestCacheAndReadiness(unittest.TestCase):
    def test_cache_includes_training_videos_and_skips_discarded(self):
        from src.engine import weak_labels as wl
        with tempfile.TemporaryDirectory() as tmp:
            labels = os.path.join(tmp, "g.json")
            wl.save_match_labels({"versao": 1, "videos": {"treino.mp4": {"tipo": "kakari_geiko", "ignorar": True},
                                                          "lixo.mp4": {"tipo": "descartar", "ignorar": True}}}, labels)
            pipe = mock.Mock(vision_model="m")
            details = []
            tc._detail_cb = details.append
            try:
                with mock.patch("src.utils.feature_cache.load_match_features", return_value=None),                      mock.patch("src.utils.feature_cache.build_cache_key", return_value="k"):
                    res = tc.step_cache(pipe, ["/v/luta.mp4", "/v/treino.mp4", "/v/lixo.mp4"], labels)
            finally:
                tc._detail_cb = None
            extracted = [c.args[0] for c in pipe.extract_match_features.call_args_list]
            self.assertEqual(extracted, ["/v/luta.mp4", "/v/treino.mp4"])
            self.assertEqual(res["ignorados"], 1)
            self.assertEqual(details[-1], "vídeo 2 de 2 sem poses no cache: treino.mp4")

    def test_readiness_lists_what_is_still_unclassified(self):
        from src.engine import weak_labels as wl
        with tempfile.TemporaryDirectory() as tmp:
            for n in ("a.mp4", "b.mp4", "c.mp4", "analise_x_1.0s.mp4"):
                open(os.path.join(tmp, n), "wb").close()
            labels = os.path.join(tmp, "g.json")
            wl.save_match_labels({"versao": 1, "videos": {"a.mp4": {"tipo": "ji_geiko", "ignorar": True},
                                                          "b.mp4": {"tipo": "luta", "pontos": []}}}, labels)
            with mock.patch("src.engine.active_learning.UncertaintySampler.get_queue", return_value=[]):
                r = tc.readiness(tmp, labels)
        got = {it["chave"]: it["quantidade"] for it in r["pendencias"]}
        self.assertEqual(got, {"videos": 1, "lutas": 1})
        self.assertFalse(r["pronto"])
        self.assertEqual(r["videos"], 3)   # o corte de análise não conta


def _panel_app():
    from src.utils.training_cycle_panel import render_training_cycle_panel
    render_training_cycle_panel(device_preference="cpu")


class TestPanel(unittest.TestCase):
    def test_panel_shows_last_cycle(self):
        from streamlit.testing.v1 import AppTest
        from src.utils.training_cycle_panel import cycles_table
        record = {"iniciado_em": "2026-11-04T20:00:00", "promoveu_modelo": False, "etapas": {
            "cache": _ok(), "bandeiras": {"status": "pulado", "detalhe": "Etapa antiga, já removida."},
            "filtro": {"status": "pulado", "detalhe": "Dados insuficientes"}, "medicao": {"status": "pulado", "detalhe": "sem golden"}}}
        table = cycles_table([record])
        self.assertEqual(table.iloc[0][tc.STEP_NAMES["filtro"]], "⏭️")
        self.assertNotIn("bandeiras", tc.STEP_NAMES)                     # ciclos antigos com a etapa ainda abrem
        self.assertEqual(table.iloc[0]["F1 no golden"], "—")

        with mock.patch.object(tc, "load_cycle_history", return_value=[record]):
            at = AppTest.from_function(_panel_app, default_timeout=30)
            at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any("Rodar ciclo agora" in b.label for b in at.button))
        self.assertTrue(any("Dados insuficientes" in m.value for m in at.markdown))


if __name__ == "__main__":
    unittest.main()
