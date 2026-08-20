"""
Testes Unitários para o Motor de Fusão e Consenso Multi-Câmeras (MultiCameraFusionEngine).
Valida a regra fundamental:
"A definição de haver ou não o golpe deve ser tomado com base no conjunto das imagens das câmeras.
Quanto mais câmeras, mais necessária a confirmação em imagens/frames da realização da técnica."
"""

import unittest
import numpy as np
from typing import Dict, Any, List

from src.analytics.multi_camera_fusion import (
    MultiCameraFusionEngine,
    CameraFrameEvidence,
    MultiCameraStrikeEvaluation
)


def create_synthetic_pose(wrist_y: float, wrist_x: float = 0.50, shoulder_y: float = 0.40) -> Dict[str, Any]:
    """Cria landmarks sintéticos para simulação de posições do corpo e do pulso."""
    return {
        "NOSE": {"x": 0.50, "y": 0.25, "z": 0.0, "visibility": 0.9},
        "RIGHT_SHOULDER": {"x": 0.55, "y": shoulder_y, "z": 0.0, "visibility": 0.9},
        "LEFT_SHOULDER": {"x": 0.45, "y": shoulder_y, "z": 0.0, "visibility": 0.9},
        "RIGHT_ELBOW": {"x": 0.56, "y": shoulder_y + 0.10, "z": 0.0, "visibility": 0.9},
        "RIGHT_WRIST": {"x": wrist_x, "y": wrist_y, "z": 0.0, "visibility": 0.9},
        "LEFT_WRIST": {"x": 0.48, "y": wrist_y + 0.02, "z": 0.0, "visibility": 0.9},
        "RIGHT_HIP": {"x": 0.53, "y": 0.65, "z": 0.0, "visibility": 0.9},
        "LEFT_HIP": {"x": 0.47, "y": 0.65, "z": 0.0, "visibility": 0.9},
        "RIGHT_KNEE": {"x": 0.54, "y": 0.78, "z": 0.0, "visibility": 0.9},
        "LEFT_KNEE": {"x": 0.46, "y": 0.78, "z": 0.0, "visibility": 0.9},
        "RIGHT_ANKLE": {"x": 0.52, "y": 0.90, "z": 0.0, "visibility": 0.9},
        "LEFT_ANKLE": {"x": 0.48, "y": 0.90, "z": 0.0, "visibility": 0.9},
        "RIGHT_FOOT_INDEX": {"x": 0.52, "y": 0.90, "z": 0.0, "visibility": 0.9}
    }


def generate_strike_history(impact_frame: int = 25, total_frames: int = 40, is_active_strike: bool = True) -> List[Dict[str, Any]]:
    """Gera histórico temporal de poses simulando um golpe de Men ou repouso."""
    history = []
    for f in range(total_frames):
        if not is_active_strike:
            # Posição estacionária de Kamae (sem aceleração brusca)
            history.append(create_synthetic_pose(wrist_y=0.55))
        else:
            if f < impact_frame - 8:
                # Kamae inicial
                history.append(create_synthetic_pose(wrist_y=0.55))
            elif f < impact_frame:
                # Elevação e descida rápida (aceleração do golpe)
                progress = (f - (impact_frame - 8)) / 8.0
                wrist_y = 0.55 - progress * 0.35  # Mãos sobem para Men
                history.append(create_synthetic_pose(wrist_y=wrist_y))
            elif f == impact_frame:
                # Ponto de impacto de Men (mãos na altura do topo do alvo)
                history.append(create_synthetic_pose(wrist_y=0.25))
            else:
                # Zanshin e estabilização
                history.append(create_synthetic_pose(wrist_y=0.35))
    return history


class TestMultiCameraFusion(unittest.TestCase):
    def setUp(self):
        self.engine_normal = MultiCameraFusionEngine(profile_name="normal")
        self.engine_rigid = MultiCameraFusionEngine(profile_name="rigido")
        self.engine_permissive = MultiCameraFusionEngine(profile_name="permissivo")

    def test_quorum_calculation_scaling_with_camera_count(self):
        """Valida que quanto mais câmeras, maior é a exigência de quórum e confirmação em frames."""
        # 1 Câmera: 1/1
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(1, "normal"), 1)
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(1, "rigido"), 1)

        # 2 Câmeras: 2/2 (Ambas devem confirmar)
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(2, "normal"), 2)
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(2, "rigido"), 2)
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(2, "permissivo"), 2)

        # 3 Câmeras: 2/3 no normal/permissivo, 3/3 no rígido
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(3, "normal"), 2)
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(3, "rigido"), 3)

        # 4 Câmeras: 3/4 no normal/permissivo, 4/4 no rígido
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(4, "normal"), 3)
        self.assertEqual(MultiCameraFusionEngine.calculate_required_quorum(4, "rigido"), 4)

        # Verificação da propriedade de monotonicidade (quorum nunca decresce com mais câmeras)
        q1 = MultiCameraFusionEngine.calculate_required_quorum(1, "normal")
        q2 = MultiCameraFusionEngine.calculate_required_quorum(2, "normal")
        q3 = MultiCameraFusionEngine.calculate_required_quorum(3, "normal")
        q4 = MultiCameraFusionEngine.calculate_required_quorum(4, "normal")
        self.assertTrue(q1 <= q2 <= q3 <= q4)

    def test_single_camera_evaluation(self):
        """Valida a avaliação monocular com 1 câmera."""
        history_cam1 = generate_strike_history(impact_frame=25, total_frames=40, is_active_strike=True)
        eval_res = self.engine_normal.evaluate_multi_camera_strike(
            camera_histories=[history_cam1],
            camera_labels=["Câmera Lateral"],
            reference_cam_idx=0,
            impact_frame=25,
            technique="MEN",
            fps=30.0
        )
        self.assertEqual(eval_res.num_active_cameras, 1)
        self.assertEqual(eval_res.num_confirming_cameras, 1)
        self.assertEqual(eval_res.required_quorum, 1)
        self.assertTrue(eval_res.is_strike_confirmed)
        self.assertEqual(eval_res.decision_status, "CONFIRMED_MULTICAM")

    def test_two_cameras_rejects_single_angle_false_positive(self):
        """Valida que com 2 câmeras, um golpe visto em apenas 1 ângulo é rejeitado por falta de confirmação cruzada."""
        history_cam1 = generate_strike_history(impact_frame=25, total_frames=40, is_active_strike=True)
        history_cam2_stationary = generate_strike_history(impact_frame=25, total_frames=40, is_active_strike=False)

        eval_res = self.engine_normal.evaluate_multi_camera_strike(
            camera_histories=[history_cam1, history_cam2_stationary],
            camera_labels=["Câmera 1 (Lateral Esq)", "Câmera 2 (Lateral Dir)"],
            reference_cam_idx=0,
            impact_frame=25,
            technique="MEN",
            fps=30.0
        )

        self.assertEqual(eval_res.num_active_cameras, 2)
        self.assertEqual(eval_res.num_confirming_cameras, 1) # Apenas Cam 1 confirmou
        self.assertEqual(eval_res.required_quorum, 2)
        self.assertFalse(eval_res.is_strike_confirmed)
        self.assertEqual(eval_res.decision_status, "REJECTED_SINGLE_ANGLE")
        self.assertIn("REJEITADO (VISÃO UNILATERAL)", eval_res.summary_text)

    def test_two_cameras_successful_consensus(self):
        """Valida que com 2 câmeras, quando ambos os ângulos confirmam a realização da técnica, o golpe é homologado."""
        history_cam1 = generate_strike_history(impact_frame=25, total_frames=40, is_active_strike=True)
        history_cam2 = generate_strike_history(impact_frame=26, total_frames=40, is_active_strike=True)

        eval_res = self.engine_normal.evaluate_multi_camera_strike(
            camera_histories=[history_cam1, history_cam2],
            camera_labels=["Câmera 1 (Lateral Esq)", "Câmera 2 (Lateral Dir)"],
            reference_cam_idx=0,
            impact_frame=25,
            technique="MEN",
            fps=30.0
        )

        self.assertEqual(eval_res.num_active_cameras, 2)
        self.assertEqual(eval_res.num_confirming_cameras, 2)
        self.assertTrue(eval_res.is_strike_confirmed)
        self.assertEqual(eval_res.decision_status, "CONFIRMED_MULTICAM")
        self.assertGreater(eval_res.joint_score, 50.0)

    def test_three_cameras_quorum_and_consensus(self):
        """Valida que com 3 câmeras, a confirmação em 2 câmeras aprova no modo normal e reprova no modo rígido."""
        history_cam1 = generate_strike_history(impact_frame=25, is_active_strike=True)
        history_cam2 = generate_strike_history(impact_frame=25, is_active_strike=True)
        history_cam3_occluded = generate_strike_history(impact_frame=25, is_active_strike=False)

        # Modo Normal: Quórum 2/3 -> Aprovado
        eval_normal = self.engine_normal.evaluate_multi_camera_strike(
            camera_histories=[history_cam1, history_cam2, history_cam3_occluded],
            camera_labels=["Cam 1", "Cam 2", "Cam 3"],
            reference_cam_idx=0,
            impact_frame=25,
            technique="MEN",
            fps=30.0
        )
        self.assertEqual(eval_normal.num_confirming_cameras, 2)
        self.assertEqual(eval_normal.required_quorum, 2)
        self.assertTrue(eval_normal.is_strike_confirmed)

        # Modo Rígido: Quórum 3/3 -> Reprovado
        eval_rigid = self.engine_rigid.evaluate_multi_camera_strike(
            camera_histories=[history_cam1, history_cam2, history_cam3_occluded],
            camera_labels=["Cam 1", "Cam 2", "Cam 3"],
            reference_cam_idx=0,
            impact_frame=25,
            technique="MEN",
            fps=30.0
        )
        self.assertEqual(eval_rigid.num_confirming_cameras, 2)
        self.assertEqual(eval_rigid.required_quorum, 3)
        self.assertFalse(eval_rigid.is_strike_confirmed)
        self.assertEqual(eval_rigid.decision_status, "REJECTED_INSUFFICIENT_CONSENSUS")

    def test_four_cameras_scaling_and_rejection_on_low_quorum(self):
        """Valida que com 4 câmeras, se apenas 1 ou 2 câmeras confirmarem, o golpe é estritamente rejeitado."""
        h_active1 = generate_strike_history(impact_frame=25, is_active_strike=True)
        h_active2 = generate_strike_history(impact_frame=26, is_active_strike=True)
        h_idle3 = generate_strike_history(impact_frame=25, is_active_strike=False)
        h_idle4 = generate_strike_history(impact_frame=25, is_active_strike=False)

        eval_res = self.engine_normal.evaluate_multi_camera_strike(
            camera_histories=[h_active1, h_active2, h_idle3, h_idle4],
            camera_labels=["Canto 1", "Canto 2", "Canto 3", "Canto 4"],
            reference_cam_idx=0,
            impact_frame=25,
            technique="MEN",
            fps=30.0
        )
        # Quórum para 4 câmeras é 3
        self.assertEqual(eval_res.num_active_cameras, 4)
        self.assertEqual(eval_res.num_confirming_cameras, 2)
        self.assertEqual(eval_res.required_quorum, 3)
        self.assertFalse(eval_res.is_strike_confirmed)

        # Agora com 3 câmeras ativas confirmando -> Aprovado
        h_active3 = generate_strike_history(impact_frame=25, is_active_strike=True)
        eval_approved = self.engine_normal.evaluate_multi_camera_strike(
            camera_histories=[h_active1, h_active2, h_active3, h_idle4],
            camera_labels=["Canto 1", "Canto 2", "Canto 3", "Canto 4"],
            reference_cam_idx=0,
            impact_frame=25,
            technique="MEN",
            fps=30.0
        )
        self.assertEqual(eval_approved.num_confirming_cameras, 3)
        self.assertTrue(eval_approved.is_strike_confirmed)
        self.assertEqual(eval_approved.decision_status, "CONFIRMED_MULTICAM")

    def test_temporal_sync_window_alignment(self):
        """Valida que picos de impacto ligeiramente defasados no tempo (ex: 2 frames) são sincronizados pela janela temporal."""
        # Cam 1 impacta no frame 22, Cam 2 no frame 25 (diferença de 3 frames = ~100ms)
        h1 = generate_strike_history(impact_frame=22, total_frames=40, is_active_strike=True)
        h2 = generate_strike_history(impact_frame=25, total_frames=40, is_active_strike=True)

        eval_res = self.engine_normal.evaluate_multi_camera_strike(
            camera_histories=[h1, h2],
            camera_labels=["Cam A", "Cam B"],
            reference_cam_idx=0,
            impact_frame=22,
            technique="MEN",
            fps=30.0
        )
        self.assertTrue(eval_res.is_strike_confirmed)
        self.assertEqual(eval_res.num_confirming_cameras, 2)

    def test_evidence_and_serialization(self):
        """Verifica a integridade e serialização dos dados em formato dict."""
        h1 = generate_strike_history(impact_frame=20, is_active_strike=True)
        h2 = generate_strike_history(impact_frame=20, is_active_strike=True)

        eval_res = self.engine_normal.evaluate_multi_camera_strike(
            camera_histories=[h1, h2],
            camera_labels=["Cam 1", "Cam 2"],
            reference_cam_idx=0,
            impact_frame=20,
            technique="MEN",
            fps=30.0
        )
        d = eval_res.to_dict()
        self.assertEqual(d["technique"], "MEN")
        self.assertEqual(d["num_active_cameras"], 2)
        self.assertEqual(d["num_confirming_cameras"], 2)
        self.assertEqual(d["required_quorum"], 2)
        self.assertIn("consensus_pct", d)
        self.assertEqual(len(d["camera_evidences"]), 2)
        self.assertTrue(d["camera_evidences"][0]["is_confirmed"])


if __name__ == "__main__":
    unittest.main()
