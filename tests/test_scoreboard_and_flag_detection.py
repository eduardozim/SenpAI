import unittest
import numpy as np
import cv2
from typing import Dict, Any

from src.vision.combatant_tracker import CombatantTracker
from src.pipeline import ShinpanAIPipeline


class TestScoreboardAndFlagDetection(unittest.TestCase):
    def setUp(self):
        self.tracker = CombatantTracker()

    def _create_mock_landmarks(self, center_x: float, center_y: float = 0.5) -> Dict[str, Any]:
        """Cria landmarks simulados com ombros e quadril em torno de center_x."""
        return {
            "LEFT_SHOULDER": {"x": center_x - 0.05, "y": center_y - 0.15, "z": 0.0, "visibility": 0.9},
            "RIGHT_SHOULDER": {"x": center_x + 0.05, "y": center_y - 0.15, "z": 0.0, "visibility": 0.9},
            "LEFT_HIP": {"x": center_x - 0.04, "y": center_y + 0.15, "z": 0.0, "visibility": 0.9},
            "RIGHT_HIP": {"x": center_x + 0.04, "y": center_y + 0.15, "z": 0.0, "visibility": 0.9},
            "LEFT_ANKLE": {"x": center_x - 0.04, "y": 0.90, "z": 0.0, "visibility": 0.9},
            "RIGHT_ANKLE": {"x": center_x + 0.04, "y": 0.90, "z": 0.0, "visibility": 0.9},
        }

    def test_detect_red_flag_score_on_navy_kendogi(self):
        """Valida que a fita vermelha é detectada mesmo com Keikogi azul-marinho."""
        frame = np.full((480, 640, 3), (30, 20, 10), dtype=np.uint8) # Fundo/Keikogi escuro
        lm = self._create_mock_landmarks(0.3, 0.5)

        # Sem fita vermelha
        score_empty = CombatantTracker.detect_red_flag_score(frame, lm)
        self.assertAlmostEqual(score_empty, 0.0, places=2)

        # Adicionar fita vermelha (BGR: 20, 20, 220) na região dorsal
        cv2.rectangle(frame, (160, 200), (220, 260), (20, 20, 220), -1)
        score_with_flag = CombatantTracker.detect_red_flag_score(frame, lm)
        self.assertGreater(score_with_flag, 0.5)

    def test_detect_red_flag_score_on_white_kendogi(self):
        """Valida que a fita vermelha é detectada com precisão sobre Keikogi branco."""
        frame = np.full((480, 640, 3), (240, 240, 240), dtype=np.uint8) # Keikogi branco
        lm = self._create_mock_landmarks(0.7, 0.5)

        # Sem fita vermelha (apenas uniforme branco)
        score_white_only = CombatantTracker.detect_red_flag_score(frame, lm)
        self.assertAlmostEqual(score_white_only, 0.0, places=2)

        # Adicionar fita vermelha sobre o Keikogi branco
        cv2.rectangle(frame, (420, 200), (480, 260), (20, 20, 220), -1)
        score_with_flag = CombatantTracker.detect_red_flag_score(frame, lm)
        self.assertGreater(score_with_flag, 0.5)

    def test_flag_detection_identifies_right_kenshi_as_aka(self):
        """
        Valida que se o lutador à direita tiver a flag vermelha (câmera invertida),
        ele é corretamente identificado como Aka e o da esquerda como Shiro.
        """
        frame = np.full((480, 640, 3), (40, 40, 40), dtype=np.uint8)
        c_left = self._create_mock_landmarks(0.30, 0.5)
        c_right = self._create_mock_landmarks(0.70, 0.5)

        # Pintar flag vermelha no lutador da DIREITA (x ~ 0.70 -> pixels 420-470)
        cv2.rectangle(frame, (430, 200), (470, 250), (20, 20, 220), -1)

        tracker = CombatantTracker()
        for _ in range(5):
            aka_lm, shiro_lm, _ = tracker.associate_and_filter([c_left, c_right], frame=frame)

        # Lutador da direita deve ser associado ao Aka
        self.assertIsNotNone(aka_lm)
        self.assertIsNotNone(shiro_lm)
        self.assertEqual(aka_lm, c_right)
        self.assertEqual(shiro_lm, c_left)
        self.assertIn("RIGHT_IS_AKA", tracker.flag_decision)

    def test_manual_inversion_of_combatants(self):
        """Valida que a opção de inversão manual inverte Aka e Shiro."""
        c_left = self._create_mock_landmarks(0.30, 0.5)
        c_right = self._create_mock_landmarks(0.70, 0.5)

        tracker_inverted = CombatantTracker(invert_assignment=True)
        aka_lm, shiro_lm, _ = tracker_inverted.associate_and_filter([c_left, c_right])

        # Com inversão manual e sem flag, o da direita torna-se Aka e o da esquerda Shiro
        self.assertEqual(aka_lm, c_right)
        self.assertEqual(shiro_lm, c_left)

    def test_scoreboard_result_calculation(self):
        """Valida cálculo de placar oficial (Ippon), vencedor e empate."""
        import tempfile
        import os
        from src.utils.demo_generator import generate_demo_kendo_video

        with tempfile.TemporaryDirectory() as tmpdir:
            test_vid = os.path.join(tmpdir, "demo_kendo_scoreboard.mp4")
            generate_demo_kendo_video(test_vid, duration_sec=2, fps=30)

            pipeline = ShinpanAIPipeline()
            res = pipeline.process_video(test_vid)
            self.assertIsNotNone(res)
            self.assertIn("scoreboard", res)
            sb = res["scoreboard"]
            self.assertIn("aka_score", sb)
            self.assertIn("shiro_score", sb)
            self.assertIn("winner", sb)
            self.assertIn("result_description", sb)
            self.assertIn("flag_detection", sb)


if __name__ == "__main__":
    unittest.main()
