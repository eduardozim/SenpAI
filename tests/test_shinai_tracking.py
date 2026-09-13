"""
Testes Automatizados para Rastreamento do Shinai e Imunidade a Inversões de Identidade.
"""

import unittest
import numpy as np
import cv2
from typing import Dict, Any

from src.vision.shinai_tracker import ShinaiTracker
from src.vision.combatant_tracker import CombatantTracker


class TestShinaiTrackingAndAntiSwap(unittest.TestCase):
    def setUp(self):
        self.shinai_tracker = ShinaiTracker()
        self.tracker = CombatantTracker()

    def _create_mock_landmarks(self, center_x: float, center_y: float = 0.50, facing_right: bool = True) -> Dict[str, Any]:
        """Cria landmarks simulados de um Kenshi com mãos e antebraços orientados."""
        wrist_dx = 0.08 if facing_right else -0.08
        elbow_dx = 0.02 if facing_right else -0.02
        return {
            "NOSE": {"x": center_x, "y": center_y - 0.25, "z": 0.0, "visibility": 0.9, "px": int(center_x * 640), "py": int((center_y - 0.25) * 480)},
            "LEFT_SHOULDER": {"x": center_x - 0.05, "y": center_y - 0.12, "z": 0.0, "visibility": 0.9, "px": int((center_x - 0.05) * 640), "py": int((center_y - 0.12) * 480)},
            "RIGHT_SHOULDER": {"x": center_x + 0.05, "y": center_y - 0.12, "z": 0.0, "visibility": 0.9, "px": int((center_x + 0.05) * 640), "py": int((center_y - 0.12) * 480)},
            "LEFT_ELBOW": {"x": center_x + elbow_dx - 0.02, "y": center_y, "z": 0.0, "visibility": 0.9, "px": int((center_x + elbow_dx - 0.02) * 640), "py": int(center_y * 480)},
            "RIGHT_ELBOW": {"x": center_x + elbow_dx, "y": center_y, "z": 0.0, "visibility": 0.9, "px": int((center_x + elbow_dx) * 640), "py": int(center_y * 480)},
            "LEFT_WRIST": {"x": center_x + wrist_dx - 0.02, "y": center_y, "z": 0.0, "visibility": 0.9, "px": int((center_x + wrist_dx - 0.02) * 640), "py": int(center_y * 480)},
            "RIGHT_WRIST": {"x": center_x + wrist_dx, "y": center_y, "z": 0.0, "visibility": 0.9, "px": int((center_x + wrist_dx) * 640), "py": int(center_y * 480)},
            "LEFT_HIP": {"x": center_x - 0.04, "y": center_y + 0.15, "z": 0.0, "visibility": 0.9, "px": int((center_x - 0.04) * 640), "py": int((center_y + 0.15) * 480)},
            "RIGHT_HIP": {"x": center_x + 0.04, "y": center_y + 0.15, "z": 0.0, "visibility": 0.9, "px": int((center_x + 0.04) * 640), "py": int((center_y + 0.15) * 480)},
            "LEFT_ANKLE": {"x": center_x - 0.04, "y": 0.85, "z": 0.0, "visibility": 0.9, "px": int((center_x - 0.04) * 640), "py": int(0.85 * 480)},
            "RIGHT_ANKLE": {"x": center_x + 0.04, "y": 0.85, "z": 0.0, "visibility": 0.9, "px": int((center_x + 0.04) * 640), "py": int(0.85 * 480)},
        }

    def test_shinai_kinematic_fallback(self):
        """Valida que o Shinai é estimado kinematicamente quando frame não possui imagem."""
        lm = self._create_mock_landmarks(0.35, 0.50, facing_right=True)
        res = self.shinai_tracker.track_shinai(frame=None, landmarks=lm)
        self.assertIsNotNone(res)
        self.assertTrue(res["detected"])
        self.assertEqual(res["facing"], "RIGHT")
        self.assertGreater(res["tip_norm"][0], res["base_norm"][0])

    def test_shinai_facing_direction(self):
        """Valida a detecção de orientação do corpo e espada (facing direction)."""
        lm_right = self._create_mock_landmarks(0.30, 0.50, facing_right=True)
        res_r = self.shinai_tracker.track_shinai(frame=None, landmarks=lm_right)
        self.assertEqual(res_r["facing"], "RIGHT")

        lm_left = self._create_mock_landmarks(0.70, 0.50, facing_right=False)
        res_l = self.shinai_tracker.track_shinai(frame=None, landmarks=lm_left)
        self.assertEqual(res_l["facing"], "LEFT")

    def test_anti_swap_resilience_against_color_noise(self):
        """
        TESTE CRÍTICO: Valida que uma vez travado o combate (LOCKED_COMBAT),
        ruídos de cor na imagem NÃO provocam inversão espúria de lados entre Shiro e Aka.
        """
        tracker = CombatantTracker(lock_tracks=True)
        c_left = self._create_mock_landmarks(0.30, 0.50, facing_right=True)
        c_right = self._create_mock_landmarks(0.70, 0.50, facing_right=False)

        frame = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)

        # 1. Inicializar combate: Shiro à esquerda, Aka à direita
        aka_0, shiro_0, _ = tracker.associate_and_filter([c_left, c_right], frame=frame)
        self.assertEqual(tracker.tracking_state, "LOCKED_COMBAT")
        self.assertAlmostEqual(tracker.shiro.last_center_x, 0.30, places=1)
        self.assertAlmostEqual(tracker.aka.last_center_x, 0.70, places=1)

        # 2. Simular 30 frames onde o lutador da ESQUERDA (Shiro) sofre ruído de cor moderado no fundo/piso (score ~0.15)
        # No código antigo com limiar de 0.15 e delta de 2.0, isso invertia no primeiro frame!
        # Com o filtro de contraste diferencial (>= 0.25), ruídos moderados não causam inversão.
        noisy_frame = frame.copy()
        cv2.rectangle(noisy_frame, (185, 220), (192, 228), (20, 20, 220), -1) # Reflexo leve no Shiro

        for f in range(30):
            aka_lm, shiro_lm, _ = tracker.associate_and_filter([c_left, c_right], frame=noisy_frame)
            # Shiro deve permanecer na esquerda (c_left) e Aka na direita (c_right)
            self.assertEqual(aka_lm["RIGHT_ANKLE"]["x"], c_right["RIGHT_ANKLE"]["x"], f"Inversão indevida no frame {f}!")
            self.assertEqual(shiro_lm["RIGHT_ANKLE"]["x"], c_left["RIGHT_ANKLE"]["x"], f"Inversão indevida no frame {f}!")

    def test_background_official_discarded(self):
        """Valida que pessoas no fundo da quadra com pés acima de y=0.58 são descartadas como BACKGROUND."""
        tracker = CombatantTracker()
        c_left = self._create_mock_landmarks(0.35, 0.50, facing_right=True)
        c_right = self._create_mock_landmarks(0.65, 0.50, facing_right=False)
        # Calibrar com os lutadores
        tracker.calibrate_main_plane([c_left, c_right])

        # Árbitro/espectador ao fundo na mesa (ground_y = 0.52)
        c_bg = self._create_mock_landmarks(0.82, 0.40, facing_right=False)
        c_bg["RIGHT_ANKLE"]["y"] = 0.52
        c_bg["LEFT_ANKLE"]["y"] = 0.52

        p_type, _, _ = tracker.classify_plane(c_bg)
        self.assertEqual(p_type, "BACKGROUND")

    def test_shinai_attached_to_landmarks(self):
        """Valida que o Shinai é anexado aos dicionários de landmarks de Aka e Shiro."""
        tracker = CombatantTracker()
        c_left = self._create_mock_landmarks(0.30, 0.50, facing_right=True)
        c_right = self._create_mock_landmarks(0.70, 0.50, facing_right=False)

        aka_lm, shiro_lm, _ = tracker.associate_and_filter([c_left, c_right])
        self.assertIn("SHINAI", aka_lm)
        self.assertIn("SHINAI", shiro_lm)
        self.assertTrue(aka_lm["SHINAI"]["detected"])
        self.assertTrue(shiro_lm["SHINAI"]["detected"])
        self.assertEqual(shiro_lm["SHINAI"]["facing"], "RIGHT")
        self.assertEqual(aka_lm["SHINAI"]["facing"], "LEFT")

    def test_shinai_visual_hough_unpacking_robustness(self):
        """Valida que o desempacotamento de linhas do HoughLinesP é robusto a arrays 1D/2D."""
        tracker = ShinaiTracker()
        lm = self._create_mock_landmarks(0.50, 0.50, facing_right=True)

        # Criar frame sintético com linha de cor de bambu (BGR ~ 120, 190, 220 -> HSV ~ 21, 116, 220)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        wrist_x = lm["RIGHT_WRIST"]["px"]
        wrist_y = lm["RIGHT_WRIST"]["py"]
        cv2.line(frame, (wrist_x, wrist_y), (wrist_x + 80, wrist_y - 40), (120, 190, 220), 4)

        res = tracker.track_shinai(frame=frame, landmarks=lm, expected_facing="RIGHT")
        self.assertIsNotNone(res)
        self.assertTrue(res["detected"])
        self.assertIn("tip_norm", res)


if __name__ == "__main__":
    unittest.main()
