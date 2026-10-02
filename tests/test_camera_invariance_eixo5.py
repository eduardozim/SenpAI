"""
Testes Unitários e de Integração para o Eixo 5:
Invariância de Câmera, Normalização Espacial e Profundidade Monocular.
Cobre:
1. Estimativa do Vetor de Combate e Ângulo de Filmagem (Frontal, Oblíquo, Lateral).
2. Estimativa de Profundidade Monocular 3D (Pseudo-Keypoints 3D a partir de 2D).
3. Cálculo de Distância Maai Tridimensional Euclidiana (X, Y, Z).
4. Diagnóstico Automático de Qualidade do Ângulo de Filmagem e Fatores de Compensação.
5. Compensação Geométrica de Perspectiva no Avaliador Postural (Shisei).
6. Integração no SenpAIPipeline e Diagnóstico Técnico.
"""

import unittest
import math
import numpy as np
from typing import Dict, Any

from src.analytics.camera_invariance import (
    CombatVectorEstimator,
    MonocularDepthEstimator,
    CameraQualityDiagnostic,
    CameraInvarianceEngine
)
from src.analytics.biomechanics import BiomechanicsAnalyzer


class TestCameraInvarianceEixo5(unittest.TestCase):

    def setUp(self):
        # Kenshis em visão lateral clássica (Aka à esquerda, Shiro à direita, mesma linha Z)
        self.aka_lateral = {
            "RIGHT_SHOULDER": {"x": 0.35, "y": 0.40, "z": 0.0},
            "LEFT_SHOULDER": {"x": 0.33, "y": 0.40, "z": 0.0},
            "RIGHT_HIP": {"x": 0.35, "y": 0.65, "z": 0.0},
            "LEFT_HIP": {"x": 0.33, "y": 0.65, "z": 0.0},
            "RIGHT_WRIST": {"x": 0.45, "y": 0.35, "z": 0.0}
        }
        self.shiro_lateral = {
            "RIGHT_SHOULDER": {"x": 0.65, "y": 0.40, "z": 0.0},
            "LEFT_SHOULDER": {"x": 0.67, "y": 0.40, "z": 0.0},
            "RIGHT_HIP": {"x": 0.65, "y": 0.65, "z": 0.0},
            "LEFT_HIP": {"x": 0.67, "y": 0.65, "z": 0.0},
            "RIGHT_WRIST": {"x": 0.55, "y": 0.45, "z": 0.0}
        }

        # Kenshis em visão frontal (alinhados em profundidade Z, pouca separação em X)
        self.aka_frontal = {
            "RIGHT_SHOULDER": {"x": 0.48, "y": 0.50, "z": -0.25},
            "LEFT_SHOULDER": {"x": 0.52, "y": 0.50, "z": -0.25},
            "RIGHT_HIP": {"x": 0.48, "y": 0.70, "z": -0.25},
            "LEFT_HIP": {"x": 0.52, "y": 0.70, "z": -0.25}
        }
        self.shiro_frontal = {
            "RIGHT_SHOULDER": {"x": 0.49, "y": 0.35, "z": 0.35},
            "LEFT_SHOULDER": {"x": 0.51, "y": 0.35, "z": 0.35},
            "RIGHT_HIP": {"x": 0.49, "y": 0.50, "z": 0.35},
            "LEFT_HIP": {"x": 0.51, "y": 0.50, "z": 0.35}
        }

    # ==========================================================================
    # 1. ESTIMATIVA DO VETOR DE COMBATE E ÂNGULO DE PONTO DE VISTA (EIXO 5.1)
    # ==========================================================================
    def test_estimate_camera_angle_lateral(self):
        """Valida que combatentes dispostos horizontalmente são classificados como LATERAL (60°-90°)."""
        res = CombatVectorEstimator.estimate_camera_angle(self.aka_lateral, self.shiro_lateral)
        self.assertEqual(res["camera_category"], "LATERAL")
        self.assertGreaterEqual(res["estimated_angle_deg"], 60.0)

    def test_estimate_camera_angle_frontal(self):
        """Valida que combatentes com grande profundidade e pouco delta X são classificados como FRONTAL (0°-30°)."""
        res = CombatVectorEstimator.estimate_camera_angle(self.aka_frontal, self.shiro_frontal)
        self.assertEqual(res["camera_category"], "FRONTAL")
        self.assertLess(res["estimated_angle_deg"], 35.0)

    def test_estimate_camera_angle_solo_fallback(self):
        """Valida estimativa solo resiliente quando apenas um praticante está em cena."""
        res = CombatVectorEstimator.estimate_camera_angle(self.aka_lateral, None)
        self.assertIn("estimated_angle_deg", res)
        self.assertIn(res["camera_category"], ["FRONTAL", "OBLIQUO", "LATERAL"])

    # ==========================================================================
    # 2. ESTIMATIVA DE PROFUNDIDADE MONOCULAR E PSEUDO-KEYPOINTS 3D (EIXO 5.2)
    # ==========================================================================
    def test_reconstruct_3d_landmarks(self):
        """Valida que landmarks 2D ganham coordenadas Z e preservam visibilidade."""
        landmarks_3d = MonocularDepthEstimator.reconstruct_3d_landmarks(self.aka_lateral, camera_angle_deg=45.0)
        self.assertIn("RIGHT_HIP", landmarks_3d)
        self.assertIn("z", landmarks_3d["RIGHT_HIP"])
        self.assertIn("RIGHT_WRIST", landmarks_3d)
        # O pulso que avança no corte deve ter Z projetado em profundidade
        self.assertIsNotNone(landmarks_3d["RIGHT_WRIST"]["z"])

    def test_calculate_3d_maai(self):
        """Valida que a distância 3D incorpora a separação Z além do plano XY."""
        atk_3d = MonocularDepthEstimator.reconstruct_3d_landmarks(self.aka_lateral, 65.0)
        def_3d = MonocularDepthEstimator.reconstruct_3d_landmarks(self.shiro_lateral, 65.0)

        dist_3d = MonocularDepthEstimator.calculate_3d_maai(atk_3d, def_3d)
        self.assertGreater(dist_3d, 0.20)
        self.assertLess(dist_3d, 0.60)

    # ==========================================================================
    # 3. DIAGNÓSTICO AUTOMÁTICO DE QUALIDADE DO ÂNGULO DE FILMAGEM (EIXO 5.3)
    # ==========================================================================
    def test_diagnose_camera_angle_lateral_vs_frontal(self):
        """Valida notas de confiabilidade diferenciadas por ângulo."""
        # Ângulo lateral (75°): postura e Fumikomi devem ter alta confiabilidade
        diag_lateral = CameraQualityDiagnostic.diagnose_camera_angle(75.0)
        self.assertGreaterEqual(diag_lateral["criterion_reliabilities"]["posture_shisei"], 80.0)
        self.assertGreaterEqual(diag_lateral["criterion_reliabilities"]["fumikomi"], 80.0)
        self.assertGreaterEqual(diag_lateral["overall_quality_score"], 80.0)

        # Ângulo frontal (20°): Tsuki deve ter boa confiabilidade, postura menor
        diag_frontal = CameraQualityDiagnostic.diagnose_camera_angle(20.0)
        self.assertGreaterEqual(diag_frontal["criterion_reliabilities"]["tsuki_depth"], 80.0)
        self.assertLess(diag_frontal["criterion_reliabilities"]["posture_shisei"], 65.0)
        self.assertTrue(len(diag_frontal["diagnostic_feedback"]) > 0)

    # ==========================================================================
    # 4. COMPENSAÇÃO GEOMÉTRICA DE PERSPECTIVA NO AVALIADOR POSTURAL (EIXO 5.1)
    # ==========================================================================
    def test_biomechanics_posture_perspective_compensation(self):
        """Valida que o avaliador de postura compensa o ângulo de visão da câmera."""
        analyzer = BiomechanicsAnalyzer()

        # Tronco com inclinação leve observada em 2D
        inclined_lm = {
            "RIGHT_SHOULDER": {"x": 0.45, "y": 0.35},
            "RIGHT_HIP": {"x": 0.40, "y": 0.65}
        }

        # Em visão lateral direta (90°): a inclinação observada é real
        score_lateral = analyzer.evaluate_posture(inclined_lm, camera_angle_deg=90.0)

        # Em visão frontal comprimida (25°): a inclinação aparente é amplificada geometricamente
        score_frontal = analyzer.evaluate_posture(inclined_lm, camera_angle_deg=25.0)

        # A inclinação corrigida em visão frontal penaliza mais que a leitura lateral crua
        self.assertNotEqual(score_lateral, score_frontal)
        self.assertLessEqual(score_frontal, score_lateral)

    # ==========================================================================
    # 5. TESTE DO MOTOR COMPLETO DE INVARIÂNCIA DE CÂMERA
    # ==========================================================================
    def test_camera_invariance_engine_complete(self):
        """Valida o processamento conjunto de invariância espacial do motor."""
        engine = CameraInvarianceEngine()
        result = engine.process_frame_spatial_invariance(self.aka_lateral, self.shiro_lateral)

        self.assertIn("angle_info", result)
        self.assertIn("quality_diagnostic", result)
        self.assertIn("maai_3d", result)
        self.assertIn("aka_3d_landmarks", result)
        self.assertIn("shiro_3d_landmarks", result)

        self.assertEqual(result["angle_info"]["camera_category"], "LATERAL")
        self.assertGreater(result["maai_3d"], 0.20)
        self.assertGreater(result["quality_diagnostic"]["overall_quality_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
