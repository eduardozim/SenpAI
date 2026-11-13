"""
Testes do resumo do sensei (src/analytics/training_narrative.py): texto do modelo, texto da IA só com fatos
do diagnóstico, rejeição de números inventados e privacidade (o nome do kenshi não vai para a IA).
Usa um LLM falso: nenhuma chamada de rede.
"""

import json
import unittest
from unittest import mock

from src.analytics import training_narrative as tn
from src.analytics.training_feedback import build_diagnosis, diagnosis_markdown
from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA, KendokaTrainingProfile, TrainingAnalyzer
from tests.test_training_feedback import _pillars


class FakeLLM:
    provider = "fake"

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def _call_remote_llm(self, prompt):
        self.prompts.append(prompt)
        return self.response


GOOD_TEXT = (
    "O ponto que mais pede atenção agora é a verticalidade da coluna: ficou em 60%, e a referência é 80%. "
    "O tronco se inclina à frente nos golpes, e no kendo o golpe sai do quadril com a coluna ereta.\n\n"
    "Seus pontos fortes foram a cadência da modalidade e a amplitude do furikaburi.\n\n"
    "Comece pelo Suburi diante do espelho, 3 séries de 30 Shōmen-uchi lentos, 2 vezes ao dia. "
    "Na próxima sessão, a meta é chegar a 70% ou mais."
)


def _diag():
    return build_diagnosis(_pillars(), "kihon", TRAINING_MODALITIES_METADATA["kihon"])


class TestTemplateNarrative(unittest.TestCase):
    def test_template_mentions_urgent_point_measured_cause_exercise_and_goal(self):
        txt = tn.template_narrative(_diag())
        self.assertIn("verticalidade da coluna (Shisei): 60%, quando a referência é 80%", txt)
        self.assertIn("Suburi diante do espelho", txt)
        self.assertIn("de 60% para 70% ou mais", txt)
        self.assertEqual(tn.unsupported_numbers(txt, tn.narrative_facts(_diag())), [],
                         "O texto do modelo também só usa números do diagnóstico")

    def test_template_without_findings(self):
        d = _diag()
        d["achados"], d["metas"] = [], []
        self.assertIn("dentro da referência", tn.template_narrative(d))


class TestAiNarrative(unittest.TestCase):
    def test_accepts_text_with_only_known_numbers(self):
        llm = FakeLLM({"texto": GOOD_TEXT})
        nar = tn.ai_narrative(_diag(), llm=llm)
        self.assertEqual(nar["fonte"], "ia")
        self.assertEqual(nar["texto"], GOOD_TEXT)
        self.assertIn("Use SOMENTE os fatos", llm.prompts[0])

    def test_rejects_invented_numbers(self):
        llm = FakeLLM({"texto": GOOD_TEXT.replace("ficou em 60%", "ficou em 47%")})
        nar = tn.ai_narrative(_diag(), llm=llm)
        self.assertEqual(nar["fonte"], "modelo")
        self.assertIn("47", nar["aviso"])

    def test_rejects_empty_and_short(self):
        self.assertEqual(tn.ai_narrative(_diag(), llm=FakeLLM({}))["fonte"], "modelo")
        self.assertEqual(tn.ai_narrative(_diag(), llm=FakeLLM({"texto": "Bom treino."}))["fonte"], "modelo")

    def test_without_ai_keeps_template_with_reason(self):
        with mock.patch("src.engine.web_research.get_llm", return_value=None):
            nar = tn.ai_narrative(_diag())
        self.assertEqual(nar["fonte"], "modelo")
        self.assertIn("Variáveis do Sistema", nar["aviso"])

    def test_kenshi_name_is_not_sent(self):
        p = _pillars()
        s, i, e = TrainingAnalyzer().generate_pedagogical_feedback(p, "kihon", "Eduardo Zimermann")
        prof = KendokaTrainingProfile("K1", "Kenshi 01", custom_name="Eduardo Zimermann", pillars=p,
                                      strengths=s, improvements=i, recommended_exercises=e).to_dict()
        llm = FakeLLM({"texto": GOOD_TEXT})
        tn.ai_narrative(prof["diagnosis"], llm=llm)
        self.assertNotIn("Eduardo", llm.prompts[0])
        self.assertNotIn("Zimermann", llm.prompts[0])


class TestNarrativeInReports(unittest.TestCase):
    def test_markdown_shows_ai_text_when_present(self):
        d = _diag()
        self.assertIn("gerado a partir das medidas", "\n".join(diagnosis_markdown(d)))
        d["narrativa"] = {"texto": GOOD_TEXT, "fonte": "ia"}
        md = "\n".join(diagnosis_markdown(d))
        self.assertIn("escrito pela IA", md)
        self.assertIn("O ponto que mais pede atenção", md)

    def test_pdf_with_narrative(self):
        from src.utils.training_pdf_generator import generate_individual_kendoka_pdf_report
        p = _pillars()
        s, i, e = TrainingAnalyzer().generate_pedagogical_feedback(p, "kihon", "Edu")
        prof = KendokaTrainingProfile("K1", "Kenshi 01", pillars=p, strengths=s, improvements=i,
                                      recommended_exercises=e).to_dict()
        prof["diagnosis"]["narrativa"] = {"texto": GOOD_TEXT, "fonte": "ia"}
        pdf = generate_individual_kendoka_pdf_report(prof, TRAINING_MODALITIES_METADATA["kihon"], "Edu")
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
