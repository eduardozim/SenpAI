"""
Testes do diagnóstico detalhado do modo de treinamento (src/analytics/training_feedback.py):
medição real do Ki-Ken-Tai-Ichi, gravidade e "onde" de cada ponto de atenção, escolha de exercícios do
catálogo, plano e metas, mesclagem de perfis e laudo em PDF. Não grava nada em data/ nem em config/.
"""

import unittest

from src.analytics import training_feedback as tf
from src.analytics.training_analyzer import (
    TrainingAnalyzer, TrainingPillarMetrics, KendokaTrainingProfile, TRAINING_MODALITIES_METADATA,
    merge_kendoka_profiles_dict,
)

FPS = 30.0


def _pose(foot_x: float, wrist_y: float = 0.30, tilt: float = 0.0) -> dict:
    return {
        "NOSE": {"x": 0.50, "y": 0.20},
        "LEFT_SHOULDER": {"x": 0.45 + tilt, "y": 0.30}, "RIGHT_SHOULDER": {"x": 0.55 + tilt, "y": 0.30},
        "LEFT_HIP": {"x": 0.45, "y": 0.55}, "RIGHT_HIP": {"x": 0.55, "y": 0.55},
        "LEFT_ANKLE": {"x": 0.45, "y": 0.90}, "RIGHT_ANKLE": {"x": foot_x, "y": 0.90},
        "RIGHT_FOOT_INDEX": {"x": foot_x + 0.02, "y": 0.92},
        "RIGHT_WRIST": {"x": 0.50, "y": wrist_y},
    }


def _history_with_fumikomi(n: int, impact: int, landing_delay_frames: int) -> list:
    """Pé direito parado, avança rápido por 4 quadros e pousa `landing_delay_frames` depois do impacto."""
    land = impact + landing_delay_frames
    hist = []
    for f in range(n):
        if f <= land - 4:
            x = 0.55
        elif f <= land:
            x = 0.55 + 0.06 * (f - (land - 4))
        else:
            x = 0.55 + 0.24
        hist.append(_pose(x))
    return hist


class TestKikentaiMeasurement(unittest.TestCase):
    def test_foot_landing_after_impact_is_late(self):
        hist = _history_with_fumikomi(60, impact=30, landing_delay_frames=3)
        res = tf.measure_kikentai(hist, 30, FPS)
        self.assertIsNotNone(res)
        offset_ms, score = res
        self.assertAlmostEqual(offset_ms, 100.0, delta=40.0)
        self.assertLess(score, 80.0)

    def test_foot_landing_with_impact_is_synchronous(self):
        hist = _history_with_fumikomi(60, impact=30, landing_delay_frames=0)
        offset_ms, score = tf.measure_kikentai(hist, 30, FPS)
        self.assertLessEqual(abs(offset_ms), 40.0)
        self.assertGreaterEqual(score, 90.0)

    def test_still_feet_are_not_measured(self):
        hist = [_pose(0.55) for _ in range(60)]
        self.assertIsNone(tf.measure_kikentai(hist, 30, FPS))

    def test_missing_feet_are_not_measured(self):
        hist = [{k: v for k, v in _pose(0.55).items() if "ANKLE" not in k and "FOOT" not in k} for _ in range(60)]
        self.assertIsNone(tf.measure_kikentai(hist, 30, FPS))


class TestPillarMetricsKikentai(unittest.TestCase):
    def setUp(self):
        self.analyzer = TrainingAnalyzer()

    def test_suburi_does_not_invent_kikentai(self):
        hist = [_pose(0.55) for _ in range(300)]
        strikes = [{"impact_frame": f} for f in (40, 100, 160, 220)]
        p = self.analyzer.calculate_pillar_metrics(hist, strikes, "suburi", fps=FPS)
        self.assertNotIn("kikentai_sincronismo", p.precisao_submetrics)
        self.assertEqual(p.nao_medidos[0]["metrica"], "kikentai_sincronismo")
        self.assertIn("não se aplica", p.nao_medidos[0]["motivo"])
        self.assertEqual(len(p.rep_details), 4)
        self.assertEqual(p.rep_details[1]["tempo_s"], round(100 / FPS, 2))

    def test_kihon_measures_kikentai_from_feet(self):
        hist = []
        strikes = []
        for k in range(4):           # 4 golpes, pé chegando 3 quadros depois de cada impacto
            block = _history_with_fumikomi(60, impact=30, landing_delay_frames=3)
            strikes.append({"impact_frame": len(hist) + 30})
            hist.extend(block)
        p = self.analyzer.calculate_pillar_metrics(hist, strikes, "kihon", fps=FPS)
        self.assertIn("kikentai_sincronismo", p.precisao_submetrics)
        self.assertLess(p.precisao_submetrics["kikentai_sincronismo"], 80.0)
        self.assertTrue(all(r.get("kikentai_ms", 0) > 30 for r in p.rep_details))
        self.assertEqual(p.nao_medidos, [])


def _pillars(**over) -> TrainingPillarMetrics:
    reps = [{"rep": i + 1, "tempo_s": 5.0 + i * 3.0, "postura": 58.0 + i * 3, "centro": 85.0, "alvo": 80.0,
             "furikaburi": 85.0, "kikentai_ms": 90.0, "kikentai": 60.0} for i in range(8)]
    kw = dict(
        movimentacao_score=70, precisao_score=70, constancia_score=80,
        movimentacao_submetrics={"verticalidade_coluna": 60.0, "nivelamento_ombros": 85.0,
                                 "alinhamento_base_pes": 75.0, "amplitude_furikaburi": 88.0},
        precisao_submetrics={"trajetoria_alvo": 80.0, "kikentai_sincronismo": 70.0, "controle_linha_centro": 86.0},
        constancia_submetrics={"regularidade_ritmo": 85.0, "resistencia_fadiga": 82.0, "adequacao_cadencia": 95.0},
        cadence_cpm=20, total_repetitions=8, rep_details=reps,
    )
    kw.update(over)
    return TrainingPillarMetrics(**kw)


class TestDiagnosis(unittest.TestCase):
    def test_findings_are_ordered_by_severity_with_where_and_measured_cause(self):
        d = tf.build_diagnosis(_pillars(), "kihon", TRAINING_MODALITIES_METADATA["kihon"])
        sev = [f["gravidade"] for f in d["achados"]]
        self.assertEqual(sev, sorted(sev, key=lambda s: tf.SEVERITY_RANK[s]))
        first = d["achados"][0]
        self.assertEqual((first["metrica"], first["gravidade"]), ("verticalidade_coluna", "prioritaria"))
        self.assertEqual(first["momentos"][0]["tempo_s"], 5.0, "A pior repetição de postura é a primeira")
        kkt = next(f for f in d["achados"] if f["metrica"] == "kikentai_sincronismo")
        self.assertTrue(kkt["causas"][0].startswith("Medido: em média o pé chega 90 ms depois"))
        self.assertIn("pé 90 ms depois", kkt["onde"])

    def test_exercises_cover_weaknesses_and_strength_within_limit(self):
        d = tf.build_diagnosis(_pillars(), "kihon", TRAINING_MODALITIES_METADATA["kihon"])
        ex = d["exercicios"]
        self.assertLessEqual(len(ex), tf.MAX_EXERCISES)
        self.assertEqual(ex[0]["tipo"], "corretivo")
        self.assertIn("Verticalidade", ex[0]["motivo"])
        self.assertIn("2 vezes ao dia", ex[0]["dose"], "Gravidade prioritária usa a dose maior")
        self.assertTrue(any(e["tipo"] == "reforco" for e in ex))
        self.assertEqual(len({e["id"] for e in ex}), len(ex))
        for e in ex:
            for k in ("name", "target", "prescription", "passos", "como_medir"):
                self.assertIn(k, e)

    def test_solo_modality_gets_only_solo_exercises(self):
        p = _pillars(constancia_submetrics={"regularidade_ritmo": 60.0, "resistencia_fadiga": 55.0, "adequacao_cadencia": 60.0})
        d = tf.build_diagnosis(p, "suburi", TRAINING_MODALITIES_METADATA["suburi"])
        self.assertTrue(d["exercicios"])
        self.assertTrue(all(e["formato"] == "solo" for e in d["exercicios"]))

    def test_metronome_dose_names_the_modality_rhythm(self):
        p = _pillars(constancia_submetrics={"regularidade_ritmo": 60.0, "resistencia_fadiga": 82.0, "adequacao_cadencia": 95.0})
        d = tf.build_diagnosis(p, "suburi", TRAINING_MODALITIES_METADATA["suburi"])
        metro = next(e for e in d["exercicios"] if e["id"] == "suburi_metronomo")
        self.assertIn("25–60 golpes por minuto", metro["dose"])

    def test_goals_and_plan(self):
        d = tf.build_diagnosis(_pillars(), "kihon", TRAINING_MODALITIES_METADATA["kihon"])
        self.assertEqual(d["metas"][0]["meta"], 70.0, "Meta gradual: +10 pontos, sem passar da referência")
        self.assertTrue(any("3 sessões por semana" in p for p in d["plano"]))

    def test_all_good_gives_maintenance(self):
        p = _pillars(movimentacao_submetrics={"verticalidade_coluna": 90.0, "nivelamento_ombros": 90.0,
                                              "alinhamento_base_pes": 90.0, "amplitude_furikaburi": 90.0},
                     precisao_submetrics={"trajetoria_alvo": 90.0, "controle_linha_centro": 90.0},
                     rep_details=[])
        d = tf.build_diagnosis(p, "suburi", TRAINING_MODALITIES_METADATA["suburi"])
        self.assertEqual(d["achados"], [])
        self.assertTrue(d["exercicios"])

    def test_pedagogical_feedback_keeps_list_format_and_attaches_diagnosis(self):
        p = _pillars()
        strengths, improvements, exercises = TrainingAnalyzer().generate_pedagogical_feedback(p, "kihon", "Edu")
        self.assertTrue(improvements[0].startswith("[PRIORITÁRIO]"))
        self.assertTrue(strengths and exercises)
        prof = KendokaTrainingProfile("K1", "Kenshi 01", pillars=p, strengths=strengths, improvements=improvements,
                                      recommended_exercises=exercises)
        self.assertEqual(prof.to_dict()["diagnosis"]["modality_key"], "kihon")
        md = prof.generate_individual_report_markdown(TRAINING_MODALITIES_METADATA["kihon"])
        for section in ("Pontos de Atenção", "Por que importa", "Plano da Semana", "Metas para a Próxima Sessão"):
            self.assertIn(section, md)


class TestMergeAndPdf(unittest.TestCase):
    def _profile(self, pillars: TrainingPillarMetrics) -> dict:
        s, i, e = TrainingAnalyzer().generate_pedagogical_feedback(pillars, "kihon", "Edu")
        return KendokaTrainingProfile("K1", "Kenshi 01", pillars=pillars, strengths=s, improvements=i,
                                      recommended_exercises=e).to_dict()

    def test_merge_does_not_average_unmeasured_kikentai_with_zero(self):
        p1 = self._profile(_pillars())
        p2 = self._profile(_pillars(precisao_submetrics={"trajetoria_alvo": 80.0, "controle_linha_centro": 86.0},
                                    nao_medidos=[{"metrica": "kikentai_sincronismo", "label": "x", "motivo": "y"}]))
        merged = merge_kendoka_profiles_dict(p1, p2, "Edu")
        self.assertEqual(merged["pillars"]["precisao_submetrics"]["kikentai_sincronismo"], 70.0)
        self.assertEqual(merged["pillars"]["nao_medidos"], [])
        self.assertEqual(len(merged["pillars"]["rep_details"]), 16)
        self.assertEqual(merged["diagnosis"]["repeticoes_analisadas"], 16)
        self.assertEqual(merged["improvements"][0], tf.legacy_lists(merged["diagnosis"])[1][0])

    def test_pdf_with_detailed_diagnosis(self):
        from src.utils.training_pdf_generator import generate_individual_kendoka_pdf_report
        prof = self._profile(_pillars())
        pdf = generate_individual_kendoka_pdf_report(prof, TRAINING_MODALITIES_METADATA["kihon"], "Edu")
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 3000)


class TestCatalog(unittest.TestCase):
    def test_catalog_is_consistent(self):
        catalog = tf.load_exercise_catalog()
        self.assertGreaterEqual(len(catalog), 10)
        ids = [e["id"] for e in catalog]
        self.assertEqual(len(ids), len(set(ids)))
        for e in catalog:
            for m in e.get("corrige", []) + e.get("reforca", []):
                self.assertIn(m, tf.METRICS, f"{e['id']}: métrica desconhecida {m}")
            self.assertEqual(set(e["dose"]), {"leve", "moderada", "prioritaria"}, e["id"])
            self.assertIn(e["formato"], ("solo", "dupla", "bogu"))
        for metric in tf.METRICS:
            self.assertTrue(any(metric in e.get("corrige", []) for e in catalog), f"Sem exercício para {metric}")


class TestCatalogEditing(unittest.TestCase):
    """Validação e gravação usadas pela aba Configurações > Catálogo de Exercícios (sempre num arquivo temporário)."""

    def _valid(self, **over):
        ex = {"id": "teste", "nome": "Teste", "formato": "solo", "nivel": "todos",
              "corrige": ["controle_linha_centro"], "reforca": [],
              "dose": {"leve": "1x10", "moderada": "2x10", "prioritaria": "3x10"}}
        ex.update(over)
        return ex

    def test_validation_messages(self):
        self.assertEqual(tf.validate_exercise(self._valid()), [])
        self.assertTrue(any("Prioritária" in e for e in tf.validate_exercise(self._valid(dose={"leve": "x", "moderada": "y"}))))
        self.assertTrue(any("Formato" in e for e in tf.validate_exercise(self._valid(formato="kata"))))
        self.assertTrue(any("desconhecida" in e for e in tf.validate_exercise(self._valid(corrige=["kiai"]))))
        self.assertTrue(any("nunca será recomendado" in e for e in tf.validate_exercise(self._valid(corrige=[]))))

    def test_save_validates_and_keeps_document_fields(self):
        import json, os, tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cat.json")
            tf.save_exercise_catalog([self._valid()], path=path)
            doc = json.load(open(path, encoding="utf-8"))
            self.assertEqual([e["id"] for e in doc["exercicios"]], ["teste"])
            self.assertIn("_sobre", doc)
            with self.assertRaises(ValueError):
                tf.save_exercise_catalog([self._valid(), self._valid()], path=path)   # id repetido
            with self.assertRaises(ValueError):
                tf.save_exercise_catalog([self._valid(nome="")], path=path)
            self.assertEqual(len(json.load(open(path, encoding="utf-8"))["exercicios"]), 1, "Inválido não grava")

    def test_coverage_and_slug(self):
        cov = {c["metrica"]: c for c in tf.catalog_coverage([self._valid()])}
        self.assertEqual(cov["controle_linha_centro"]["corrige"], 1)
        self.assertEqual(cov["verticalidade_coluna"]["corrige"], 0)
        self.assertEqual(tf.slugify("Kote-Men Rápido!"), "kote_men_rapido")


if __name__ == "__main__":
    unittest.main()
