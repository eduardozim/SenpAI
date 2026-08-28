"""
Testes Automatizados para o Módulo de Treinamento & Aprendizado de Kendo (SenpAI).
Cobre:
- Identificação e integridade das 10 modalidades oficiais de treinamento de Kendo.
- Classificação automática e override de modalidade.
- Cálculo e coerência dos 3 Pilares Fundamentais (Forma, Precisão, Constância).
- Rastreamento e nomeação individual de Kendocas.
- Geração de diagnósticos pedagógicos, pontos de melhoria e exercícios práticos recomendados.
"""

import unittest
from typing import Dict, List, Any

from src.analytics.training_analyzer import (
    TrainingAnalyzer,
    TRAINING_MODALITIES_METADATA,
    TrainingPillarMetrics,
    KendokaTrainingProfile,
    TrainingSessionResult
)
from src.analytics.event_spotter import StrikeEvent


class TestTrainingModesAndPedagogy(unittest.TestCase):
    """Conjunto de testes para validação do motor analítico de Treinamento."""

    def setUp(self):
        self.analyzer = TrainingAnalyzer()

    def _create_mock_pose(self, wrist_y: float = 0.4, nose_x: float = 0.5, heel_diff: float = 0.05, tilt: float = 0.02) -> Dict[str, Any]:
        """Gera pose sintética com pontos chave para testes."""
        return {
            "NOSE": {"x": nose_x, "y": 0.3, "z": 0.0, "visibility": 0.95},
            "LEFT_SHOULDER": {"x": 0.45, "y": 0.4, "z": 0.0, "visibility": 0.95},
            "RIGHT_SHOULDER": {"x": 0.55, "y": 0.4, "z": 0.0, "visibility": 0.95},
            "LEFT_HIP": {"x": 0.45 + tilt, "y": 0.6, "z": 0.0, "visibility": 0.95},
            "RIGHT_HIP": {"x": 0.55 + tilt, "y": 0.6, "z": 0.0, "visibility": 0.95},
            "LEFT_WRIST": {"x": 0.48, "y": wrist_y, "z": 0.0, "visibility": 0.95},
            "RIGHT_WRIST": {"x": 0.52, "y": wrist_y, "z": 0.0, "visibility": 0.95},
            "LEFT_ANKLE": {"x": 0.46, "y": 0.85 + heel_diff, "z": 0.0, "visibility": 0.95},
            "RIGHT_ANKLE": {"x": 0.54, "y": 0.85, "z": 0.0, "visibility": 0.95}
        }

    def test_metadata_and_10_modalities_integrity(self):
        """Verifica a presença e consistência das 10 modalidades oficiais de treinamento de Kendo."""
        expected_modalities = [
            "ashi_sabaki",
            "suburi",
            "kihon",
            "kirikaeshi",
            "uchikomi_geiko",
            "kakari_geiko",
            "waza_geiko",
            "oji_waza_geiko",
            "ji_geiko",
            "shiai_geiko"
        ]

        self.assertEqual(len(TRAINING_MODALITIES_METADATA), 10)
        for mod_key in expected_modalities:
            self.assertIn(mod_key, TRAINING_MODALITIES_METADATA)
            meta = TRAINING_MODALITIES_METADATA[mod_key]
            self.assertIn("name", meta)
            self.assertIn("japanese", meta)
            self.assertIn("category", meta)
            self.assertIn("description", meta)
            self.assertIn("focus_areas", meta)
            self.assertGreaterEqual(len(meta["focus_areas"]), 2)

    def test_modality_auto_detection_suburi_solo(self):
        """Valida que praticante solo executando cortes contínuos é classificado como Suburi."""
        # 60 frames com cortes no ar
        pose_hist = [self._create_mock_pose(wrist_y=0.2 if i % 10 < 5 else 0.5) for i in range(60)]
        strikes = [
            StrikeEvent(strike_type="MEN", start_frame=5, impact_frame=10, end_frame=15, fps=30.0, attacker_id="KENSHI_SOLO"),
            StrikeEvent(strike_type="MEN", start_frame=15, impact_frame=20, end_frame=25, fps=30.0, attacker_id="KENSHI_SOLO"),
            StrikeEvent(strike_type="MEN", start_frame=25, impact_frame=30, end_frame=35, fps=30.0, attacker_id="KENSHI_SOLO")
        ]

        mod_key, conf, _ = self.analyzer.detect_training_modality(pose_hist, secondary_history=[], detected_strikes=strikes, fps=30.0)
        self.assertEqual(mod_key, "suburi")
        self.assertGreaterEqual(conf, 0.80)

    def test_modality_auto_detection_kirikaeshi_pair(self):
        """Valida que dupla com alta cadência e múltiplos golpes alternados é classificada como Kirikaeshi."""
        # 120 frames (4 segundos) com 10 golpes -> 150 CPM
        pose_hist_p1 = [self._create_mock_pose() for _ in range(120)]
        pose_hist_p2 = [self._create_mock_pose(nose_x=0.7) for _ in range(120)]
        strikes = [
            StrikeEvent(strike_type="MEN", start_frame=i*12-2, impact_frame=i*12, end_frame=i*12+2, fps=30.0, attacker_id="KENSHI_SHIRO")
            for i in range(1, 10)
        ]

        mod_key, conf, _ = self.analyzer.detect_training_modality(pose_hist_p1, secondary_history=pose_hist_p2, detected_strikes=strikes, fps=30.0)
        self.assertEqual(mod_key, "kirikaeshi")
        self.assertGreaterEqual(conf, 0.85)

    def test_pillar_metrics_calculation(self):
        """Verifica a precisão dos cálculos numéricos dos 3 Pilares (Forma, Precisão, Constância)."""
        pose_hist = [self._create_mock_pose(tilt=0.01, heel_diff=0.02) for _ in range(90)]
        strikes = [
            StrikeEvent(strike_type="MEN", start_frame=15, impact_frame=20, end_frame=25, fps=30.0, attacker_id="KENSHI_SHIRO"),
            StrikeEvent(strike_type="MEN", start_frame=45, impact_frame=50, end_frame=55, fps=30.0, attacker_id="KENSHI_SHIRO"),
            StrikeEvent(strike_type="MEN", start_frame=75, impact_frame=80, end_frame=85, fps=30.0, attacker_id="KENSHI_SHIRO")
        ]

        pillars = self.analyzer.calculate_pillar_metrics(pose_hist, strikes, "suburi", fps=30.0)

        # Validação dos intervalos e estrutura
        self.assertIsInstance(pillars, TrainingPillarMetrics)
        self.assertGreaterEqual(pillars.forma, 0.0)
        self.assertLessEqual(pillars.forma, 100.0)
        self.assertGreaterEqual(pillars.precisao, 0.0)
        self.assertLessEqual(pillars.precisao, 100.0)
        self.assertGreaterEqual(pillars.constancia, 0.0)
        self.assertLessEqual(pillars.constancia, 100.0)
        self.assertGreaterEqual(pillars.overall_score, 0.0)
        self.assertLessEqual(pillars.overall_score, 100.0)

        # Validação das sub-métricas
        self.assertIn("verticalidade_coluna", pillars.forma_submetrics)
        self.assertIn("alinhamento_base_pes", pillars.forma_submetrics)
        self.assertIn("trajetoria_alvo", pillars.precisao_submetrics)
        self.assertIn("regularidade_ritmo", pillars.constancia_submetrics)
        self.assertEqual(pillars.total_repetitions, 3)

    def test_pedagogical_feedback_and_exercises(self):
        """Valida que diagnósticos pedagógicos geram pontos fortes, correções e exercícios adequados."""
        pillars = TrainingPillarMetrics(
            forma_score=88.0,
            precisao_score=65.0,
            constancia_score=70.0,
            forma_submetrics={"verticalidade_coluna": 85.0, "nivelamento_ombros": 85.0, "alinhamento_base_pes": 82.0, "amplitude_furikaburi": 80.0},
            precisao_submetrics={"trajetoria_alvo": 60.0, "kikentai_sincronismo": 62.0, "controle_linha_centro": 65.0},
            constancia_submetrics={"regularidade_ritmo": 72.0, "resistencia_fadiga": 68.0, "adequacao_cadencia": 75.0},
            cadence_cpm=40.0,
            cadence_std_dev_seconds=0.35,
            total_repetitions=5
        )

        strengths, improvements, exercises = self.analyzer.generate_pedagogical_feedback(pillars, "suburi", "Eduardo")

        self.assertGreater(len(strengths), 0)
        self.assertGreater(len(improvements), 0)
        self.assertGreater(len(exercises), 0)

        # Exercício deve conter nome, alvo e prescrição
        first_ex = exercises[0]
        self.assertIn("name", first_ex)
        self.assertIn("target", first_ex)
        self.assertIn("prescription", first_ex)

    def test_kendoka_profile_custom_naming(self):
        """Valida rastreamento de perfil e suporte à nomeação interativa do Kendoca."""
        profile = KendokaTrainingProfile(
            kendoka_id="KENSHI_SHIRO",
            default_name="Kendoca Shiro (Esquerda)",
            custom_name="Eduardo Zimermann",
            role="Kakarite"
        )

        self.assertEqual(profile.custom_name, "Eduardo Zimermann")
        self.assertIn("Eduardo Zimermann", profile.display_name)
        
        # Teste de alteração de nome
        profile.set_custom_name("Sensei Tanaka")
        self.assertEqual(profile.custom_name, "Sensei Tanaka")

    def test_full_session_analysis_and_serialization(self):
        """Valida a execução completa de analyze_session e serialização to_dict()."""
        pose_hist_p1 = [self._create_mock_pose() for _ in range(60)]
        pose_hist_p2 = [self._create_mock_pose(nose_x=0.7) for _ in range(60)]
        strikes = [
            StrikeEvent(strike_type="MEN", start_frame=10, impact_frame=15, end_frame=20, fps=30.0, attacker_id="KENSHI_SHIRO"),
            StrikeEvent(strike_type="KOTE", start_frame=30, impact_frame=35, end_frame=40, fps=30.0, attacker_id="KENSHI_AKA")
        ]

        custom_names = {"KENSHI_SHIRO": "Eduardo Zimermann", "KENSHI_AKA": "Parceiro Keiko"}

        result = self.analyzer.analyze_session(
            primary_history=pose_hist_p1,
            secondary_history=pose_hist_p2,
            detected_strikes=strikes,
            modality_override="uchikomi_geiko",
            fps=30.0,
            custom_kendoka_names=custom_names
        )

        self.assertIsInstance(result, TrainingSessionResult)
        self.assertEqual(result.modality_key, "uchikomi_geiko")
        self.assertEqual(result.detection_method, "MANUAL_SELECT")
        self.assertTrue(result.is_manual_override)
        self.assertEqual(len(result.kendokas), 2)
        self.assertEqual(result.kendokas[0].custom_name, "Eduardo Zimermann")
        self.assertEqual(result.kendokas[1].custom_name, "Parceiro Keiko")

        # Serialização JSON-friendly
        res_dict = result.to_dict()
        self.assertIn("modality_key", res_dict)
        self.assertIn("kendokas", res_dict)
        self.assertEqual(len(res_dict["kendokas"]), 2)
        self.assertIn("pillars", res_dict["kendokas"][0])
        self.assertIn("recommended_exercises", res_dict["kendokas"][0])


if __name__ == "__main__":
    unittest.main()
