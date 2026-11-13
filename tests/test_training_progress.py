"""
Testes do histórico de evolução por nome (src/analytics/training_progress.py). Usa arquivo temporário:
nada é gravado em data/.
"""

import os
import json
import tempfile
import unittest

from src.analytics import training_progress as tp
from src.analytics.training_analyzer import TrainingAnalyzer, KendokaTrainingProfile
from src.analytics.training_feedback import diagnosis_markdown
from src.analytics.training_narrative import template_narrative, narrative_facts, unsupported_numbers
from tests.test_training_feedback import _pillars


def _profile(kendoka_id="KENSHI_01", **over):
    p = _pillars(**over)
    s, i, e = TrainingAnalyzer().generate_pedagogical_feedback(p, "kihon", "x")
    return KendokaTrainingProfile(kendoka_id, "Kenshi 01", pillars=p, strengths=s, improvements=i,
                                  recommended_exercises=e).to_dict()


class TestTrainingProgress(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "progress.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _attach(self, prof, session, name, date):
        return tp.attach_progress(prof, session_key=session, name=name, default_name="Kenshi 01",
                                  video=f"{session}.mp4", session_date=date, path=self.path)

    def test_default_names_are_not_recorded(self):
        for name in ("", "Kenshi 01", "Kendoca Shiro (Esquerda)", "Kendoka Praticante", "Praticante #2"):
            evo = self._attach(_profile(), "s1", name, "2026-11-01T10:00:00")
            self.assertTrue(evo.get("sem_nome"), name)
        self.assertEqual(tp.load_progress(self.path)["sessoes"], [])

    def test_first_session_then_comparison_with_goals(self):
        first = _profile()          # verticalidade 60 → meta 70; Ki-Ken-Tai-Ichi 70 → meta 80
        self.assertTrue(self._attach(first, "s1", "Eduardo Zim", "2026-11-01T10:00:00").get("primeira_sessao"))
        better = _profile(movimentacao_submetrics={"verticalidade_coluna": 74.0, "nivelamento_ombros": 85.0,
                                                   "alinhamento_base_pes": 75.0, "amplitude_furikaburi": 88.0})
        evo = self._attach(better, "s2", "  eduardo   ZIM ", "2026-11-08T10:00:00")   # mesmo nome, outra grafia
        self.assertEqual(evo["anterior"]["data"], "2026-11-01")
        self.assertTrue(evo["anterior"]["mesma_modalidade"])
        vert = next(m for m in evo["metricas"] if m["metrica"] == "verticalidade_coluna")
        self.assertEqual((vert["antes"], vert["agora"], vert["delta"]), (60.0, 74.0, 14.0))
        goals = {g["label"].split(" (")[0]: g["atingida"] for g in evo["metas_anteriores"]}
        self.assertTrue(goals["Verticalidade da coluna"])
        self.assertFalse(goals["Sincronismo Ki-Ken-Tai-Ichi"])
        self.assertIs(better["diagnosis"]["evolucao"], evo)
        self.assertEqual(evo["sessoes_no_historico"], 2)

    def test_reviewing_an_older_session_compares_with_the_one_before_it(self):
        self._attach(_profile(), "s1", "Ana", "2026-11-01T10:00:00")
        self._attach(_profile(), "s3", "Ana", "2026-11-15T10:00:00")
        evo = self._attach(_profile(), "s2", "Ana", "2026-11-08T10:00:00")
        self.assertEqual(evo["anterior"]["data"], "2026-11-01")

    def test_rename_moves_the_record_and_rerender_does_not_rewrite(self):
        prof = _profile()
        self._attach(prof, "s1", "Ana", "2026-11-01T10:00:00")
        mtime = os.path.getmtime(self.path)
        self._attach(prof, "s1", "Ana", "2026-11-01T10:00:00")          # nova renderização da tela
        self.assertEqual(os.path.getmtime(self.path), mtime, "Sem mudança, o arquivo não é regravado")
        self._attach(prof, "s1", "Beatriz", "2026-11-01T10:00:00")      # renomeado
        names = [s["nome"] for s in tp.load_progress(self.path)["sessoes"]]
        self.assertEqual(names, ["Beatriz"])
        self._attach(prof, "s1", "Kenshi 01", "2026-11-01T10:00:00")    # voltou ao nome padrão
        self.assertEqual(tp.load_progress(self.path)["sessoes"], [])

    def test_other_modality_is_used_only_without_same_modality(self):
        self._attach(_profile(), "s1", "Ana", "2026-11-01T10:00:00")
        data = tp.load_progress(self.path)
        data["sessoes"][0]["modalidade"] = "suburi"
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        evo = self._attach(_profile(), "s2", "Ana", "2026-11-08T10:00:00")
        self.assertFalse(evo["anterior"]["mesma_modalidade"])

    def test_evolution_in_markdown_and_template_narrative(self):
        self._attach(_profile(), "s1", "Ana", "2026-11-01T10:00:00")
        prof = _profile(movimentacao_submetrics={"verticalidade_coluna": 74.0, "nivelamento_ombros": 85.0,
                                                 "alinhamento_base_pes": 75.0, "amplitude_furikaburi": 88.0})
        self._attach(prof, "s2", "Ana", "2026-11-08T10:00:00")
        diag = prof["diagnosis"]
        md = "\n".join(diagnosis_markdown(diag))
        self.assertIn("### 📈 Evolução", md)
        self.assertIn("Verticalidade da coluna (Shisei) de 60% para 74%", md)
        txt = template_narrative(diag)
        self.assertIn("Metas da sessão anterior atingidas: 1 de 3", txt)
        self.assertEqual(unsupported_numbers(txt, narrative_facts(diag)), [],
                         "A evolução vai nos fatos da IA, então os números dela são permitidos")

    def test_live_report_is_recorded_by_kendoka_name(self):
        from tests.test_training_live_manager import _run_live_session
        _, rep = _run_live_session("kihon")
        evo = tp.attach_progress(rep, session_key=rep["session_id"], name="Ana", default_name="Kendoka Praticante",
                                 source="ao_vivo", session_date=rep["created_at"], path=self.path)
        self.assertTrue(evo.get("primeira_sessao"))
        rec = tp.load_progress(self.path)["sessoes"][0]
        self.assertEqual((rec["origem"], rec["modalidade"], rec["repeticoes"]), ("ao_vivo", "kihon", 6))
        self.assertIn("kikentai_sincronismo", rec["metricas"])


if __name__ == "__main__":
    unittest.main()
