"""
Testes Automatizados para Detecção de Sonkyō, Bounding de Luta, Rastreamento dos 2 Kenshi e Filtragem de Planos.
"""

import os
import unittest
import numpy as np

from src.analytics.sonkyo_detector import SonkyoDetector, SonkyoInterval
from src.vision.combatant_tracker import CombatantTracker, CombatantProfile
from src.analytics.event_spotter import EventSpotter, StrikeEvent
from src.pipeline import ShinpanaiPipeline
from src.utils.demo_generator import generate_demo_kendo_video

class TestSonkyoAndPlaneFiltering(unittest.TestCase):
    def setUp(self):
        self.sonkyo_detector = SonkyoDetector()
        self.tracker = CombatantTracker()
        self.spotter = EventSpotter()

    def _create_synthetic_standing_pose(self, center_x: float = 0.50) -> dict:
        """Cria pose de Kenshi em pé (Kamae normal)."""
        return {
            "NOSE": {"x": center_x, "y": 0.25, "z": 0.0, "visibility": 0.9, "px": int(center_x*640), "py": int(0.25*480)},
            "RIGHT_SHOULDER": {"x": center_x + 0.05, "y": 0.38, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.05)*640), "py": int(0.38*480)},
            "LEFT_SHOULDER": {"x": center_x - 0.05, "y": 0.38, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.05)*640), "py": int(0.38*480)},
            "RIGHT_HIP": {"x": center_x + 0.04, "y": 0.62, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.04)*640), "py": int(0.62*480)},
            "LEFT_HIP": {"x": center_x - 0.04, "y": 0.62, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.04)*640), "py": int(0.62*480)},
            "RIGHT_KNEE": {"x": center_x + 0.04, "y": 0.76, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.04)*640), "py": int(0.76*480)},
            "LEFT_KNEE": {"x": center_x - 0.04, "y": 0.76, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.04)*640), "py": int(0.76*480)},
            "RIGHT_ANKLE": {"x": center_x + 0.04, "y": 0.90, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.04)*640), "py": int(0.90*480)},
            "LEFT_ANKLE": {"x": center_x - 0.04, "y": 0.90, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.04)*640), "py": int(0.90*480)},
            "RIGHT_WRIST": {"x": center_x + 0.08, "y": 0.50, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.08)*640), "py": int(0.50*480)},
            "LEFT_WRIST": {"x": center_x - 0.02, "y": 0.52, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.02)*640), "py": int(0.52*480)}
        }

    def _create_synthetic_sonkyo_pose(self, center_x: float = 0.50) -> dict:
        """Cria pose de Kenshi em Sonkyō (agachamento ritualístico com quadril rebaixado)."""
        return {
            "NOSE": {"x": center_x, "y": 0.38, "z": 0.0, "visibility": 0.9, "px": int(center_x*640), "py": int(0.38*480)},
            "RIGHT_SHOULDER": {"x": center_x + 0.05, "y": 0.48, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.05)*640), "py": int(0.48*480)},
            "LEFT_SHOULDER": {"x": center_x - 0.05, "y": 0.48, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.05)*640), "py": int(0.48*480)},
            "RIGHT_HIP": {"x": center_x + 0.04, "y": 0.78, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.04)*640), "py": int(0.78*480)},
            "LEFT_HIP": {"x": center_x - 0.04, "y": 0.78, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.04)*640), "py": int(0.78*480)},
            "RIGHT_KNEE": {"x": center_x + 0.08, "y": 0.84, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.08)*640), "py": int(0.84*480)},
            "LEFT_KNEE": {"x": center_x - 0.08, "y": 0.84, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.08)*640), "py": int(0.84*480)},
            "RIGHT_ANKLE": {"x": center_x + 0.03, "y": 0.90, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.03)*640), "py": int(0.90*480)},
            "LEFT_ANKLE": {"x": center_x - 0.03, "y": 0.90, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.03)*640), "py": int(0.90*480)},
            "RIGHT_WRIST": {"x": center_x + 0.02, "y": 0.65, "z": 0.0, "visibility": 0.9, "px": int((center_x+0.02)*640), "py": int(0.65*480)},
            "LEFT_WRIST": {"x": center_x - 0.02, "y": 0.65, "z": 0.0, "visibility": 0.9, "px": int((center_x-0.02)*640), "py": int(0.65*480)}
        }

    def test_sonkyo_pose_evaluation(self):
        standing = self._create_synthetic_standing_pose()
        is_s_stand, conf_stand, metrics_stand = self.sonkyo_detector.evaluate_sonkyo_pose(standing)
        self.assertFalse(is_s_stand, "Pose em pé não deve ser classificada como Sonkyō.")
        self.assertGreater(metrics_stand["hip_ratio"], 0.40)

        sonkyo = self._create_synthetic_sonkyo_pose()
        is_s_sonkyo, conf_sonkyo, metrics_sonkyo = self.sonkyo_detector.evaluate_sonkyo_pose(sonkyo)
        self.assertTrue(is_s_sonkyo, "Pose agachada deve ser classificada como Sonkyō.")
        self.assertLessEqual(metrics_sonkyo["hip_ratio"], 0.45)
        self.assertGreaterEqual(conf_sonkyo, 0.55)

    def test_match_boundaries_detection(self):
        # 100 frames: 0..20 Sonkyō Inicial, 25..80 Combate em pé, 85..100 Sonkyō Final
        timeline = []
        for i in range(100):
            if i <= 20:
                timeline.append(self._create_synthetic_sonkyo_pose())
            elif i < 85:
                timeline.append(self._create_synthetic_standing_pose())
            else:
                timeline.append(self._create_synthetic_sonkyo_pose())

        res = self.sonkyo_detector.detect_match_boundaries(timeline, fps=30.0)
        self.assertTrue(res["is_bounded"])
        self.assertTrue(res["has_initial_sonkyo"])
        self.assertTrue(res["has_final_sonkyo"])
        self.assertGreaterEqual(res["match_start_frame"], 20)
        self.assertLessEqual(res["match_end_frame"], 85)
        self.assertGreater(res["effective_combat_duration_seconds"], 1.5)

    def test_filter_strikes_outside_sonkyo(self):
        # Criar histórico de poses com movimento contínuo e 3 picos nítidos (frames 10, 50 e 95)
        pose_history = []
        for f in range(110):
            p = self._create_synthetic_standing_pose()
            # Pequeno ruído/movimento natural contínuo
            base_y = 0.50 + 0.008 * np.sin(f * 0.3)
            p["RIGHT_WRIST"]["y"] = base_y
            
            # Pico no frame 10 (durante Sonkyō inicial)
            if f == 9:
                p["RIGHT_WRIST"]["y"] = 0.20
            elif f == 10:
                p["RIGHT_WRIST"]["y"] = 0.60
            # Pico no frame 50 (durante luta ativa)
            elif f == 49:
                p["RIGHT_WRIST"]["y"] = 0.20
            elif f == 50:
                p["RIGHT_WRIST"]["y"] = 0.60
            # Pico no frame 95 (durante Sonkyō final)
            elif f == 94:
                p["RIGHT_WRIST"]["y"] = 0.20
            elif f == 95:
                p["RIGHT_WRIST"]["y"] = 0.60
            pose_history.append(p)

        # Sem filtro de Sonkyō
        all_strikes = self.spotter.detect_strikes(pose_history, fps=30.0, filter_out_of_bounds=False)
        self.assertGreaterEqual(len(all_strikes), 2)

        # Com filtro de Sonkyō ativo: Início=25, Fim=85
        bounded_strikes = self.spotter.detect_strikes(
            pose_history,
            fps=30.0,
            start_bound_frame=25,
            end_bound_frame=85,
            filter_out_of_bounds=True
        )
        self.assertEqual(len(bounded_strikes), 1, "Apenas o golpe ocorrido entre os Sonkyōs deve ser registrado.")
        self.assertEqual(bounded_strikes[0].impact_frame, 50)
        self.assertTrue(bounded_strikes[0].is_within_sonkyo_bounds)

    def test_plane_classification_background_and_foreground(self):
        # Calibrar plano principal com Kenshi normal
        main_kenshi = self._create_synthetic_standing_pose(center_x=0.40)
        self.tracker.calibrate_main_plane([main_kenshi])

        # 1. Candidato no Plano Principal
        p_main, scale_m, _ = self.tracker.classify_plane(main_kenshi)
        self.assertEqual(p_main, "MAIN_PLANE")
        self.assertAlmostEqual(scale_m, 1.0, delta=0.15)

        # 2. Candidato em Segundo Plano (Background / Outra luta ao fundo)
        bg_kenshi = {k: {"x": v["x"] * 0.4 + 0.3, "y": v["y"] * 0.4 + 0.2, "z": 0.0, "visibility": 0.8, "px": int((v["x"]*0.4+0.3)*640), "py": int((v["y"]*0.4+0.2)*480)} for k, v in main_kenshi.items()}
        p_bg, scale_bg, reason_bg = self.tracker.classify_plane(bg_kenshi)
        self.assertEqual(p_bg, "BACKGROUND")
        self.assertLess(scale_bg, 0.65)
        self.assertIn("Segundo Plano", reason_bg)

        # 3. Candidato em Primeiro Plano Excessivo (Pessoa passando colada na câmera)
        fg_occluder = {k: {"x": v["x"] * 1.6 - 0.2, "y": v["y"] * 1.6 - 0.3, "z": 0.0, "visibility": 0.9, "px": int(100), "py": int(100)} for k, v in main_kenshi.items()}
        p_fg, scale_fg, reason_fg = self.tracker.classify_plane(fg_occluder)
        self.assertEqual(p_fg, "FOREGROUND_OCCLUDER")
        self.assertGreater(scale_fg, 1.40)
        self.assertIn("câmera", reason_fg)

    def test_two_combatants_association(self):
        aka_pose = self._create_synthetic_standing_pose(center_x=0.35)
        shiro_pose = self._create_synthetic_standing_pose(center_x=0.65)
        bg_pose = {k: {"x": v["x"] * 0.3 + 0.3, "y": v["y"] * 0.3 + 0.2, "z": 0.0, "visibility": 0.8, "px": int((v["x"]*0.3+0.3)*640), "py": int((v["y"]*0.3+0.2)*480)} for k, v in aka_pose.items()}

        aka_res, shiro_res, disc = self.tracker.associate_and_filter([aka_pose, shiro_pose, bg_pose])
        self.assertIsNotNone(aka_res)
        self.assertIsNotNone(shiro_res)
        self.assertEqual(len(disc), 1)
        self.assertEqual(disc[0]["plane_type"], "BACKGROUND")

    def test_sonkyo_with_hakama_occlusions(self):
        """Testa resiliência da detecção de Sonkyō mesmo quando os joelhos/tornozelos estão oclusos pelo Hakama."""
        occluded_sonkyo = self._create_synthetic_sonkyo_pose()
        # Simular oclusão das pernas pelo Hakama
        del occluded_sonkyo["RIGHT_KNEE"]
        del occluded_sonkyo["LEFT_KNEE"]
        
        is_s, conf, metrics = self.sonkyo_detector.evaluate_sonkyo_pose(occluded_sonkyo)
        self.assertTrue(is_s, "Sonkyō com pernas oclusas pelo Hakama deve ser detectado corretamente.")
        self.assertGreaterEqual(conf, 0.48)
        self.assertGreaterEqual(metrics["torso_ratio"], 0.60)

    def test_sonkyo_temporal_gap_bridging(self):
        """Testa preenchimento de falhas temporárias (dropouts) no rastreamento durante Sonkyō."""
        timeline = []
        for i in range(80):
            if 10 <= i <= 30:
                # Simular queda pontual de rastreamento nos frames 18, 19 e 20
                if i in [18, 19, 20]:
                    timeline.append(None)
                else:
                    timeline.append(self._create_synthetic_sonkyo_pose())
            else:
                timeline.append(self._create_synthetic_standing_pose())

        res = self.sonkyo_detector.detect_match_boundaries(timeline, fps=30.0)
        self.assertTrue(res["has_initial_sonkyo"])
        # O intervalo deve ter sido unificado através do fechamento morfológico
        init_s = res["initial_sonkyo"]
        self.assertLessEqual(init_s["start_frame"], 12)
        self.assertGreaterEqual(init_s["end_frame"], 28)

    def test_pipeline_integration_with_sonkyo(self):
        # Gerar vídeo sintético de teste
        test_vid = "test_sonkyo_pipeline_match.mp4"
        out_vid = "test_sonkyo_pipeline_annotated.mp4"
        generate_demo_kendo_video(test_vid, duration_sec=3, fps=30)

        try:
            pipeline = ShinpanaiPipeline(calibration_profile="normal", device_preference="cpu")
            result = pipeline.process_video(video_path=test_vid, output_video_path=out_vid)

            self.assertIn("sonkyo_analysis", result)
            self.assertIn("plane_filtering", result)
            self.assertIn("effective_combat_duration_seconds", result)
            self.assertIsInstance(result["events"], list)
            self.assertTrue(os.path.exists(out_vid))
            self.assertGreater(os.path.getsize(out_vid), 1000)

            sonkyo = result["sonkyo_analysis"]
            self.assertIn("match_start_frame", sonkyo)
            self.assertIn("match_end_frame", sonkyo)
            self.assertIn("status_message", sonkyo)

            planes = result["plane_filtering"]
            self.assertIn("discarded_background_count", planes)
            self.assertIn("discarded_foreground_count", planes)
        finally:
            if os.path.exists(test_vid):
                os.remove(test_vid)
            if os.path.exists(out_vid):
                os.remove(out_vid)

if __name__ == "__main__":
    unittest.main()
