"""
Suíte de Testes Automatizados para o Eixo 6:
Modelagem do Estilo Individual do Kenshi & Warm Start de Perfis.
"""

import os
import tempfile
import unittest
from src.analytics.kenshi_style_model import (
    MetricDistribution,
    KinestheticBaselineModel,
    KinestheticProfileManager,
    ProfileWarmStartManager,
)
from src.analytics.training_analyzer import (
    KendokaTrainingProfile,
    TrainingPillarMetrics,
)
from src.engine.feedback_manager import FeedbackManager


class TestKinestheticBaselineModel(unittest.TestCase):
    """Testes para o Modelo Cinestésico Individual e Distribuição Welford."""

    def test_welford_metric_distribution(self):
        dist = MetricDistribution()
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        for v in values:
            dist.update(v)

        self.assertEqual(dist.count, 5)
        self.assertAlmostEqual(dist.mean, 14.0, places=3)
        self.assertGreater(dist.std_dev, 2.5)

        data = dist.to_dict()
        restored = MetricDistribution.from_dict(data)
        self.assertEqual(restored.count, 5)
        self.assertAlmostEqual(restored.mean, 14.0, places=3)

    def test_baseline_learning_and_strike_evaluation(self):
        model = KinestheticBaselineModel(kenshi_id="KENSHI_TEST", display_name="Sensei Tanaka")

        # Alimentar baseline com 5 golpes com postura média de 8.0° e fumikomi de 10ms
        for _ in range(5):
            model.update_from_strike(
                strike_type="MEN",
                spine_tilt_deg=8.0,
                fumikomi_offset_ms=10.0,
                elbow_extension_deg=165.0
            )

        self.assertEqual(model.strikes_count, 5)
        self.assertAlmostEqual(model.metrics["spine_tilt_strike_deg"].mean, 8.0, places=2)

        # Avaliar um golpe com postura descompensada (+10° a mais que a média)
        eval_res = model.evaluate_strike_against_baseline(
            strike_type="MEN",
            observed_spine_tilt_deg=18.0,
            observed_fumikomi_ms=10.0,
            observed_elbow_extension_deg=165.0
        )

        self.assertTrue(eval_res["has_sufficient_history"])
        self.assertIn("spine_tilt", eval_res["deviations"])
        self.assertAlmostEqual(eval_res["deviations"]["spine_tilt"]["diff"], 10.0, places=1)
        self.assertTrue(len(eval_res["insights"]) >= 1)
        self.assertIn("inclinação", eval_res["insights"][0].lower())

    def test_training_session_baseline_update(self):
        model = KinestheticBaselineModel(kenshi_id="KENSHI_SOLO", display_name="Kendoca Solo")
        model.update_from_training_session(
            cadence_cpm=45.0,
            resting_posture_angle=3.5,
            mean_spine_tilt=6.0,
            mean_fumikomi_ms=12.0
        )

        self.assertEqual(model.sessions_count, 1)
        self.assertAlmostEqual(model.metrics["cadence_cpm"].mean, 45.0, places=1)
        self.assertAlmostEqual(model.metrics["posture_resting_angle_deg"].mean, 3.5, places=1)


class TestKinestheticProfileManager(unittest.TestCase):
    """Testes para o Gerenciador Central de Perfis Cinestésicos."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.storage_file = os.path.join(self.tmp_dir.name, "test_kenshi_baselines.json")
        self.manager = KinestheticProfileManager(storage_path=self.storage_file)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_get_or_create_and_persistence(self):
        prof = self.manager.get_or_create_profile("KENSHI_AKA", display_name="Kenshi Aka (Vermelho)")
        self.assertEqual(prof.kenshi_id, "KENSHI_AKA")
        self.assertEqual(prof.display_name, "Kenshi Aka (Vermelho)")
        self.assertTrue(os.path.exists(self.storage_file))

        # Registrar golpes
        self.manager.record_strike_event(
            kenshi_id="KENSHI_AKA",
            strike_type="KOTE",
            spine_tilt_deg=7.5,
            fumikomi_offset_ms=15.0,
            elbow_extension_deg=150.0
        )

        # Recarregar do disco e verificar integridade
        new_mgr = KinestheticProfileManager(storage_path=self.storage_file)
        self.assertIn("KENSHI_AKA", new_mgr.profiles)
        restored_prof = new_mgr.profiles["KENSHI_AKA"]
        self.assertEqual(restored_prof.strikes_count, 1)

    def test_list_profiles(self):
        self.manager.get_or_create_profile("KENSHI_1", display_name="Atleta Um")
        self.manager.get_or_create_profile("KENSHI_2", display_name="Atleta Dois")
        profiles_list = self.manager.list_profiles()
        self.assertEqual(len(profiles_list), 2)
        ids = [p["kenshi_id"] for p in profiles_list]
        self.assertIn("KENSHI_1", ids)
        self.assertIn("KENSHI_2", ids)


class TestProfileWarmStart(unittest.TestCase):
    """Testes para a Derivação de Perfis com Warm Start (Eixo 6.2)."""

    def setUp(self):
        self.base_profile = {
            "name": "Treino Geral / Keiko (Normal)",
            "min_total_score": 0.70,
            "weights": {
                "target_impact": 0.40,
                "fumikomi_sync": 0.25,
                "posture": 0.20,
                "zanshin": 0.15,
            },
            "sub_thresholds": {
                "target_impact": 0.60,
                "fumikomi_sync": 0.50,
                "posture": 0.50,
                "zanshin": 0.45,
            },
            "weights_by_strike_type": {
                "MEN": {"target_impact": 0.35, "fumikomi_sync": 0.30, "posture": 0.20, "zanshin": 0.15},
                "KOTE": {"target_impact": 0.45, "fumikomi_sync": 0.25, "posture": 0.18, "zanshin": 0.12},
                "DO": {"target_impact": 0.45, "fumikomi_sync": 0.15, "posture": 0.20, "zanshin": 0.20},
                "TSUKI": {"target_impact": 0.50, "fumikomi_sync": 0.20, "posture": 0.15, "zanshin": 0.15},
            }
        }

    def test_derive_more_strict(self):
        derived = ProfileWarmStartManager.derive_profile(
            source_profile_config=self.base_profile,
            new_profile_key="campeonato_rigido",
            direction="more_strict",
            adjustment_factor=1.10,
            new_profile_name="Campeonato Super Rígido"
        )

        self.assertEqual(derived["name"], "Campeonato Super Rígido")
        self.assertTrue(derived["warm_started"])
        self.assertEqual(derived["warm_start_direction"], "more_strict")
        # 0.70 * 1.10 = 0.77
        self.assertAlmostEqual(derived["min_total_score"], 0.77, places=2)
        # Sub-threshold target: 0.60 * 1.10 = 0.66
        self.assertAlmostEqual(derived["sub_thresholds"]["target_impact"], 0.66, places=2)
        # Pesos herdados intactos
        self.assertEqual(derived["weights"]["target_impact"], 0.40)
        self.assertEqual(derived["weights_by_strike_type"]["MEN"]["target_impact"], 0.35)

    def test_derive_more_permissive(self):
        derived = ProfileWarmStartManager.derive_profile(
            source_profile_config=self.base_profile,
            new_profile_key="iniciantes_dojo",
            direction="more_permissive",
            adjustment_factor=1.15
        )

        self.assertTrue(derived["warm_started"])
        self.assertEqual(derived["warm_start_direction"], "more_permissive")
        # Limiar deve ter sido reduzido
        self.assertLess(derived["min_total_score"], 0.70)
        self.assertLess(derived["sub_thresholds"]["target_impact"], 0.60)

    def test_feedback_manager_warm_start_integration(self):
        tmp_dir = tempfile.TemporaryDirectory()
        profiles_file = os.path.join(tmp_dir.name, "test_profiles.json")
        fb_mgr = FeedbackManager(profiles_path=profiles_file)

        derived = fb_mgr.derive_profile_warm_start(
            source_profile_key="normal",
            new_profile_key="torneio_teste",
            direction="more_strict",
            factor=1.08,
            new_name="Torneio de Teste"
        )

        self.assertIn("torneio_teste", fb_mgr.load_profiles())
        self.assertEqual(derived["name"], "Torneio de Teste")
        self.assertTrue(derived["warm_started"])

        lineage = fb_mgr.list_profiles_with_lineage()
        keys = [p["key"] for p in lineage]
        self.assertIn("torneio_teste", keys)
        tmp_dir.cleanup()


class TestKendokaReportWithBaseline(unittest.TestCase):
    """Testes para inclusão da Seção 3 de Baseline Cinestésico no Relatório."""

    def test_report_includes_kinesthetic_section(self):
        pillars = TrainingPillarMetrics(
            movimentacao_score=85.0,
            precisao_score=80.0,
            constancia_score=82.0,
            movimentacao_submetrics={"verticalidade_coluna": 88.0, "nivelamento_ombros": 85.0, "alinhamento_base_pes": 80.0, "amplitude_furikaburi": 85.0},
            precisao_submetrics={"trajetoria_alvo": 80.0, "kikentai_sincronismo": 82.0, "controle_linha_centro": 80.0},
            constancia_submetrics={"regularidade_ritmo": 84.0, "resistencia_fadiga": 80.0, "adequacao_cadencia": 82.0},
            cadence_cpm=42.0,
            cadence_std_dev_seconds=0.35,
            total_repetitions=30
        )

        profile = KendokaTrainingProfile(
            kendoka_id="KENSHI_SHIRO",
            default_name="Kendoca Shiro",
            custom_name="Eduardo Zimermann",
            pillars=pillars,
            kinesthetic_insights=["Postura com 2.1° a mais de inclinação do que seu baseline habitual."]
        )

        md = profile.generate_individual_report_markdown()
        self.assertIn("## 3. Análise Comparativa com o Baseline Cinestésico Individual (Eixo 6)", md)
        self.assertIn("Postura com 2.1° a mais de inclinação", md)
        self.assertIn("Eduardo Zimermann", md)


if __name__ == "__main__":
    unittest.main()
