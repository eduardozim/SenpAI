"""
Testes unitários para o gerenciador de Treinamento em Tempo Real (LiveTrainingSessionManager).
"""

import unittest
import sys
import os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.analytics.training_live_manager import LiveTrainingSessionManager
from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA


class TestLiveTrainingSessionManager(unittest.TestCase):
    def setUp(self):
        self.manager = LiveTrainingSessionManager(
            modality_override="suburi",
            kendoka_name="Test Kendoka",
            target_dan=3
        )

    def test_initialization(self):
        self.assertEqual(self.manager.kendoka_name, "Test Kendoka")
        self.assertEqual(self.manager.current_modality_key, "suburi")
        self.assertFalse(self.manager.is_auto_detected)
        self.assertEqual(self.manager.rep_count, 0)
        self.assertEqual(len(self.manager.rep_history), 0)

    def test_process_empty_frame(self):
        res = self.manager.process_live_frame(live_pose_histories=[[]], fps=30.0)
        self.assertIn("modality_name", res)
        self.assertEqual(res["rep_count"], 0)
        self.assertIn("movement_score", res)
        self.assertIn("precision_score", res)
        self.assertIn("constancy_score", res)

    def test_furikaburi_and_strike_detection(self):
        # Simular quadro de Furikaburi (punho acima dos ombros)
        pose_furikaburi = {
            "RIGHT_WRIST": {"x": 0.5, "y": 0.2, "z": 0.0, "visibility": 0.9},
            "LEFT_WRIST": {"x": 0.5, "y": 0.2, "z": 0.0, "visibility": 0.9},
            "RIGHT_SHOULDER": {"x": 0.52, "y": 0.45, "z": 0.0, "visibility": 0.9},
            "LEFT_SHOULDER": {"x": 0.48, "y": 0.45, "z": 0.0, "visibility": 0.9},
            "RIGHT_HIP": {"x": 0.51, "y": 0.70, "z": 0.0, "visibility": 0.9},
            "LEFT_HIP": {"x": 0.49, "y": 0.70, "z": 0.0, "visibility": 0.9},
        }

        # 1. Passo com Furikaburi
        res1 = self.manager.process_live_frame([[pose_furikaburi]], fps=30.0)
        self.assertEqual(self.manager._strike_phase, "FURIKABURI")
        self.assertEqual(self.manager.rep_count, 0)

        # 2. Simular quadro de Uchi (punho descendo rapidamente para altura dos ombros)
        pose_strike = {
            "RIGHT_WRIST": {"x": 0.5, "y": 0.44, "z": 0.0, "visibility": 0.9},
            "LEFT_WRIST": {"x": 0.5, "y": 0.44, "z": 0.0, "visibility": 0.9},
            "RIGHT_SHOULDER": {"x": 0.52, "y": 0.45, "z": 0.0, "visibility": 0.9},
            "LEFT_SHOULDER": {"x": 0.48, "y": 0.45, "z": 0.0, "visibility": 0.9},
            "RIGHT_HIP": {"x": 0.51, "y": 0.70, "z": 0.0, "visibility": 0.9},
            "LEFT_HIP": {"x": 0.49, "y": 0.70, "z": 0.0, "visibility": 0.9},
        }

        time.sleep(0.4)  # Garantir intervalo mínimo
        res2 = self.manager.process_live_frame([[pose_strike]], fps=30.0)
        self.assertTrue(res2["new_rep_detected"])
        self.assertEqual(self.manager.rep_count, 1)
        self.assertEqual(len(self.manager.rep_history), 1)

    def test_render_live_hud_html(self):
        html_out = self.manager.render_live_hud_html()
        self.assertIn("REPETIÇÕES", html_out)
        self.assertIn("CADÊNCIA", html_out)
        self.assertIn("Movimentação", html_out)
        self.assertIn("Precisão", html_out)
        self.assertIn("Constância", html_out)

    def test_generate_final_session_report(self):
        rep = self.manager.generate_final_session_report()
        self.assertIn("markdown_report", rep)
        self.assertIn("duration_seconds", rep)
        self.assertIn("average_movement", rep)
        self.assertIn("average_precision", rep)
        self.assertIn("average_constancy", rep)
        self.assertIn("strengths", rep)
        self.assertIn("prescribed_drills", rep)


if __name__ == "__main__":
    unittest.main()


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t


def _live_pose(wrist_y: float, foot_x: float) -> dict:
    return {
        "NOSE": {"x": 0.50, "y": 0.20},
        "RIGHT_WRIST": {"x": 0.52, "y": wrist_y}, "LEFT_WRIST": {"x": 0.50, "y": wrist_y},
        "RIGHT_SHOULDER": {"x": 0.55, "y": 0.30}, "LEFT_SHOULDER": {"x": 0.45, "y": 0.30},
        "RIGHT_HIP": {"x": 0.55, "y": 0.55}, "LEFT_HIP": {"x": 0.45, "y": 0.55},
        "LEFT_ANKLE": {"x": 0.45, "y": 0.90}, "RIGHT_ANKLE": {"x": foot_x, "y": 0.90},
        "RIGHT_FOOT_INDEX": {"x": foot_x + 0.02, "y": 0.92},
    }


def _run_live_session(modality: str, strikes: int = 6, landing_delay: int = 3):
    """Sessão simulada a 30 fps: kamae, furikaburi, impacto; o pé pousa `landing_delay` quadros depois do impacto."""
    from unittest import mock
    clock = _Clock()
    with mock.patch("time.time", new=lambda: clock.t):
        mgr = LiveTrainingSessionManager(modality_override=modality, kendoka_name="Edu")
        history, foot_x = [], 0.55
        for _ in range(strikes):
            for f in range(45):
                wrist = 0.15 if 10 <= f < 20 else (0.31 if f == 20 else 0.35)
                land = 20 + landing_delay
                if land - 4 < f <= land:
                    foot_x += 0.06
                history.append(_live_pose(wrist, foot_x))
                mgr.process_live_frame([history], fps=30.0)
                clock.t += 1.0 / 30.0
        for _ in range(15):                       # quadros finais para medir o último golpe
            history.append(_live_pose(0.35, foot_x))
            mgr.process_live_frame([history], fps=30.0)
            clock.t += 1.0 / 30.0
        report = mgr.generate_final_session_report()
    return mgr, report


class TestLiveDiagnosis(unittest.TestCase):
    def test_kihon_measures_kikentai_per_rep_and_builds_diagnosis(self):
        mgr, report = _run_live_session("kihon")
        self.assertEqual(mgr.rep_count, 6)
        self.assertEqual(len(report["rep_details"]), 6)
        offsets = [r.get("kikentai_ms") for r in report["rep_details"]]
        self.assertTrue(all(o is not None and 60 <= o <= 140 for o in offsets), offsets)
        self.assertIn("kikentai_sincronismo", [f["metrica"] for f in report["diagnosis"]["achados"]])
        kkt = next(f for f in report["diagnosis"]["achados"] if f["metrica"] == "kikentai_sincronismo")
        self.assertTrue(kkt["causas"][0].startswith("Medido: em média o pé chega"))
        self.assertIn("o pé chegou", mgr.latest_feedback)     # alerta durante a sessão
        self.assertTrue(report["recommended_exercises"])
        self.assertEqual(len(report["prescribed_drills"]), len(report["recommended_exercises"]))
        for section in ("Pontos de Atenção", "Plano da Semana", "Metas para a Próxima Sessão"):
            self.assertIn(section, report["markdown_report"])

    def test_synchronous_feet_give_good_kikentai(self):
        _, report = _run_live_session("kihon", landing_delay=0)
        metrics = [f["metrica"] for f in report["diagnosis"]["achados"]]
        self.assertNotIn("kikentai_sincronismo", metrics)
        self.assertIn("kikentai_sincronismo", [s["metrica"] for s in report["diagnosis"]["pontos_fortes"]])

    def test_suburi_marks_kikentai_as_not_applicable(self):
        _, report = _run_live_session("suburi")
        nm = report["diagnosis"]["nao_medidos"]
        self.assertEqual(nm[0]["metrica"], "kikentai_sincronismo")
        self.assertIn("não se aplica", nm[0]["motivo"])
        self.assertTrue(all(e["formato"] == "solo" for e in report["recommended_exercises"]))

    def test_session_without_reps_does_not_invent_precision(self):
        mgr = LiveTrainingSessionManager(modality_override="kihon")
        report = mgr.generate_final_session_report()
        self.assertEqual(report["total_reps"], 0)
        self.assertNotIn("trajetoria_alvo", [f["metrica"] for f in report["diagnosis"]["achados"]])
        self.assertIn("trajetoria_alvo", [n["metrica"] for n in report["diagnosis"]["nao_medidos"]])
