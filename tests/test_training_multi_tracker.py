"""
Testes unitários dedicados ao novo motor de rastreamento de múltiplos Kenshis (TrainingMultiKenshiTracker)
e do sistema de re-identificação (KenshiReID).
"""

import unittest
import numpy as np
from src.vision.kenshi_reid import (
    KenshiReIDExtractor,
    KenshiAppearanceSignature,
    KenshiAppearanceSnapshot
)
from src.vision.training_multi_tracker import TrainingMultiKenshiTracker


def _create_mock_landmarks(x_center: float, height: float = 0.5, is_facing_front: bool = True):
    """Gera landmarks anatômicos simulados para testes."""
    nose_vis = 0.9 if is_facing_front else 0.1
    y_top = 0.3
    return {
        "NOSE": {"x": x_center, "y": y_top, "z": 0.0, "visibility": nose_vis, "px": int(x_center * 640), "py": int(y_top * 480)},
        "LEFT_EAR": {"x": x_center - 0.03, "y": y_top + 0.01, "z": 0.0, "visibility": 0.8, "px": 100, "py": 100},
        "RIGHT_EAR": {"x": x_center + 0.03, "y": y_top + 0.01, "z": 0.0, "visibility": 0.8, "px": 100, "py": 100},
        "LEFT_SHOULDER": {"x": x_center - 0.06, "y": y_top + 0.10, "z": 0.0, "visibility": 0.9, "px": 100, "py": 100},
        "RIGHT_SHOULDER": {"x": x_center + 0.06, "y": y_top + 0.10, "z": 0.0, "visibility": 0.9, "px": 100, "py": 100},
        "LEFT_HIP": {"x": x_center - 0.04, "y": y_top + 0.25, "z": 0.0, "visibility": 0.9, "px": 100, "py": 100},
        "RIGHT_HIP": {"x": x_center + 0.04, "y": y_top + 0.25, "z": 0.0, "visibility": 0.9, "px": 100, "py": 100},
        "LEFT_ANKLE": {"x": x_center - 0.04, "y": y_top + height, "z": 0.0, "visibility": 0.9, "px": 100, "py": 100},
        "RIGHT_ANKLE": {"x": x_center + 0.04, "y": y_top + height, "z": 0.0, "visibility": 0.9, "px": 100, "py": 100},
    }


class TestKenshiReIDAndMultiTracker(unittest.TestCase):

    def test_orientation_and_biometrics(self):
        """Valida detecção de orientação anatômica e extração de biometria invariante."""
        lm_front = _create_mock_landmarks(0.5, height=0.55, is_facing_front=True)
        lm_back = _create_mock_landmarks(0.5, height=0.55, is_facing_front=False)

        ori_front = KenshiReIDExtractor.determine_orientation(lm_front)
        ori_back = KenshiReIDExtractor.determine_orientation(lm_back)

        self.assertEqual(ori_front, "FRONT")
        self.assertEqual(ori_back, "BACK")

        bio = KenshiReIDExtractor.extract_biometrics(lm_front)
        self.assertIsNotNone(bio)
        self.assertEqual(len(bio), 4)

    def test_multi_kenshi_tracking_more_than_two(self):
        """Valida que o rastreador de treino rastreia N kenshis simultâneos (ex: 4 kenshis)."""
        tracker = TrainingMultiKenshiTracker(max_lost_frames=10)

        # 4 Kenshis posicionados em X = 0.2, 0.4, 0.6, 0.8
        candidates = [
            _create_mock_landmarks(0.2, height=0.50),
            _create_mock_landmarks(0.4, height=0.52),
            _create_mock_landmarks(0.6, height=0.54),
            _create_mock_landmarks(0.8, height=0.56)
        ]

        frame_dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        # Processar 5 frames com os 4 kenshis
        for _ in range(5):
            res = tracker.process_frame(candidates, frame=frame_dummy)

        self.assertEqual(len(tracker.tracks), 4)
        for k_id, track in tracker.tracks.items():
            self.assertEqual(track.status, "ACTIVE")
            self.assertEqual(len(track.history), 5)

    def test_kenshi_exits_and_returns_reidentification(self):
        """
        Cenário Crítico de Treino:
        - Kenshi 1 e Kenshi 2 estão no treino.
        - Kenshi 1 sai do enquadramento da câmera por 20 frames (entra em DORMANT).
        - Kenshi 1 retorna à câmera com as mesmas características biométricas.
        - O rastreador deve identificá-lo como o mesmo KENSHI_01 (Re-ID) e reativá-lo,
          em vez de criar um KENSHI_03 espúrio.
        """
        tracker = TrainingMultiKenshiTracker(max_lost_frames=8, reid_similarity_threshold=0.55)
        frame_dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        k1_lm = _create_mock_landmarks(0.3, height=0.45)
        k2_lm = _create_mock_landmarks(0.7, height=0.60)

        # 1. Ambos presentes nos primeiros 10 frames
        for _ in range(10):
            tracker.process_frame([k1_lm, k2_lm], frame=frame_dummy)

        self.assertEqual(len(tracker.tracks), 2)
        self.assertIn("KENSHI_01", tracker.tracks)
        self.assertIn("KENSHI_02", tracker.tracks)

        # 2. Kenshi 1 sai do quadro. Apenas Kenshi 2 fica visível por 15 frames
        for _ in range(15):
            tracker.process_frame([k2_lm], frame=frame_dummy)

        # Kenshi 1 deve ter passado para DORMANT
        self.assertEqual(tracker.tracks["KENSHI_01"].status, "DORMANT")
        self.assertEqual(tracker.tracks["KENSHI_02"].status, "ACTIVE")

        # 3. Kenshi 1 retorna ao quadro em X = 0.32 com a mesma biometria (altura 0.45)
        k1_return_lm = _create_mock_landmarks(0.32, height=0.45)
        for _ in range(5):
            tracker.process_frame([k1_return_lm, k2_lm], frame=frame_dummy)

        # Deve continuar com exatamente 2 Kenshis (KENSHI_01 reativado, sem criar KENSHI_03)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(tracker.tracks["KENSHI_01"].status, "ACTIVE")
        self.assertGreaterEqual(tracker.tracks["KENSHI_01"].reentry_count, 1)
        self.assertGreaterEqual(tracker.total_reidentifications, 1)

    def test_draw_overlay(self):
        """Valida que draw_overlay renderiza os Kenshis e o banner de sessão sem exceções."""
        tracker = TrainingMultiKenshiTracker(max_lost_frames=8)
        frame_dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        k1_lm = _create_mock_landmarks(0.3, height=0.45)
        k2_lm = _create_mock_landmarks(0.7, height=0.60)
        res = tracker.process_frame([k1_lm, k2_lm], frame=frame_dummy)

        drawn = tracker.draw_overlay(frame_dummy, res, custom_names={"KENSHI_01": "Miyamoto"})
        self.assertIsNotNone(drawn)
        self.assertEqual(drawn.shape, frame_dummy.shape)
        # Verifica se o frame desenhado foi modificado (não mais todo preto)
        self.assertTrue(np.any(drawn > 0))


if __name__ == "__main__":
    unittest.main()

