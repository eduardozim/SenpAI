"""
Testes Automatizados para Detecção de Sonkyō, Bounding de Luta, Rastreamento dos 2 Kenshi e Filtragem de Planos.
"""

import os
import unittest
import numpy as np

from src.analytics.sonkyo_detector import SonkyoDetector, SonkyoInterval
from src.vision.combatant_tracker import CombatantTracker, CombatantProfile
from src.analytics.event_spotter import EventSpotter, StrikeEvent
from src.pipeline import SenpAIPipeline
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
        """Valida a classificação da postura de Sonkyō verificando compressão vertical, rebaixamento de quadril e coluna."""
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
        """Testa a identificação e delimitação temporal automática da luta entre o Sonkyō inicial e final."""
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
        """Verifica a filtragem e descarte de golpes ocorridos fora da janela regulamentar de combate delimitada por Sonkyō."""
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
        """Valida a classificação geométrica de planos descartando segundo plano (fundo) e oclusão em primeiro plano."""
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
        """Testa o rastreamento e associação contínua dos dois Kenshi principais (Aka e Shiro), descartando oponentes secundários."""
        left_pose = self._create_synthetic_standing_pose(center_x=0.35)
        right_pose = self._create_synthetic_standing_pose(center_x=0.65)
        bg_pose = {k: {"x": v["x"] * 0.3 + 0.3, "y": v["y"] * 0.3 + 0.2, "z": 0.0, "visibility": 0.8, "px": int((v["x"]*0.3+0.3)*640), "py": int((v["y"]*0.3+0.2)*480)} for k, v in left_pose.items()}

        aka_res, shiro_res, disc = self.tracker.associate_and_filter([left_pose, right_pose, bg_pose])
        self.assertIsNotNone(aka_res)
        self.assertIsNotNone(shiro_res)
        self.assertEqual(aka_res, right_pose, "Lutador à direita deve ser associado ao Aka.")
        self.assertEqual(shiro_res, left_pose, "Lutador à esquerda deve ser associado ao Shiro.")
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
        """Valida a integração end-to-end do pipeline de avaliação com detecção de Sonkyō, filtragem de planos e vídeo anotado."""
        # Gerar vídeo sintético de teste
        test_vid = "test_sonkyo_pipeline_match.mp4"
        out_vid = "test_sonkyo_pipeline_annotated.mp4"
        generate_demo_kendo_video(test_vid, duration_sec=3, fps=30)

        try:
            pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
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

    def test_timestamp_to_frame_conversions(self):
        """Testa a conversão precisa de múltiplos formatos de timestamp (MM:SS.mmm, segundos) para número de frame."""
        from src.analytics.sonkyo_detector import SonkyoInterval
        self.assertEqual(SonkyoInterval.timestamp_to_frame("00:01.000", fps=30.0), 30)
        self.assertEqual(SonkyoInterval.timestamp_to_frame("00:02.500", fps=30.0), 75)
        self.assertEqual(SonkyoInterval.timestamp_to_frame("01:00.000", fps=30.0), 1800)
        self.assertEqual(SonkyoInterval.timestamp_to_frame("5.0s", fps=30.0), 150)
        self.assertEqual(SonkyoInterval.timestamp_to_frame("", fps=30.0), 0)

    def test_sonkyo_learning_and_profile_persistence(self):
        """Verifica a extração e persistência do aprendizado contínuo de Sonkyō a partir de anotações do árbitro."""
        test_profile_path = "config/test_sonkyo_learned_profile.json"
        if os.path.exists(test_profile_path):
            os.remove(test_profile_path)

        try:
            detector = SonkyoDetector(learned_profile_path=test_profile_path)
            self.assertEqual(detector.learned_profile["samples_count"], 0)

            # Criar histórico de poses (10 frames de Sonkyō e 20 em pé)
            timeline = [self._create_synthetic_sonkyo_pose() for _ in range(10)] + [self._create_synthetic_standing_pose() for _ in range(20)]
            
            # Aprender a partir do intervalo [0, 9]
            learn_res = detector.learn_from_annotation(timeline, start_frame=0, end_frame=9, fps=30.0, interval_type="INITIAL")
            self.assertEqual(learn_res["status"], "success")
            self.assertEqual(detector.learned_profile["samples_count"], 1)
            self.assertTrue(os.path.exists(test_profile_path))

            # Recarregar detector com o perfil salvo
            reloaded_detector = SonkyoDetector(learned_profile_path=test_profile_path)
            stats = reloaded_detector.get_learned_stats()
            self.assertEqual(stats["samples_count"], 1)
            self.assertGreater(stats["exemplars_count"], 0)

            # Resetar perfil
            reloaded_detector.reset_learned_profile()
            self.assertEqual(reloaded_detector.learned_profile["samples_count"], 0)
        finally:
            if os.path.exists(test_profile_path):
                os.remove(test_profile_path)

    def test_detect_match_boundaries_with_overrides(self):
        """Valida a aplicação de overrides manuais de limites de Sonkyō e recálculo dinâmico dos limites de luta."""
        test_profile_path = "config/test_sonkyo_override_profile.json"
        if os.path.exists(test_profile_path):
            os.remove(test_profile_path)

        try:
            detector = SonkyoDetector(learned_profile_path=test_profile_path)
            timeline = [self._create_synthetic_standing_pose() for _ in range(100)]
            
            # Forçar overrides manuais do árbitro
            init_override = {"start_timestamp": "00:00.200", "end_timestamp": "00:00.800"}
            final_override = {"start_timestamp": "00:02.500", "end_timestamp": "00:03.000"}

            res = detector.detect_match_boundaries(
                timeline,
                fps=30.0,
                initial_sonkyo_override=init_override,
                final_sonkyo_override=final_override
            )

            self.assertTrue(res["has_initial_sonkyo"])
            self.assertTrue(res["has_final_sonkyo"])
            self.assertEqual(res["match_start_frame"], 26) # end_frame=24 + 2
            self.assertEqual(res["match_end_frame"], 73)   # start_frame=75 - 2
            self.assertIn("Limites de Sonkyō atualizados e aprendidos", res["status_message"])
            self.assertGreaterEqual(res["learned_samples_total"], 2)
        finally:
            if os.path.exists(test_profile_path):
                os.remove(test_profile_path)

    def test_pipeline_reprocess_with_sonkyo_overrides(self):
        """Testa o reprocessamento de combate pelo pipeline aplicando overrides manuais de Sonkyō e aprendizado adaptativo."""
        test_vid = "test_reprocess_sonkyo_match.mp4"
        out_vid = "test_reprocess_sonkyo_annotated.mp4"
        generate_demo_kendo_video(test_vid, duration_sec=4, fps=30)

        try:
            pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
            
            init_ov = {"start_timestamp": "00:00.000", "end_timestamp": "00:01.000"}
            fin_ov = {"start_timestamp": "00:03.000", "end_timestamp": "00:03.900"}

            result = pipeline.process_video(
                video_path=test_vid,
                output_video_path=out_vid,
                initial_sonkyo_override=init_ov,
                final_sonkyo_override=fin_ov
            )

            self.assertIsNotNone(result)
            sonkyo = result["sonkyo_analysis"]
            self.assertTrue(sonkyo["has_initial_sonkyo"])
            self.assertTrue(sonkyo["has_final_sonkyo"])
            self.assertEqual(sonkyo["match_start_frame"], 32)
            self.assertEqual(sonkyo["match_end_frame"], 88)
        finally:
            if os.path.exists(test_vid):
                os.remove(test_vid)
            if os.path.exists(out_vid):
                os.remove(out_vid)

    def test_default_fallback_sonkyo_when_not_detected(self):
        """Testa se momentos padrão no início e término do vídeo são incluídos quando o Sonkyō não é detectado."""
        detector = SonkyoDetector()
        # Linha do tempo com 90 frames sem nenhum Sonkyō (apenas pessoas em pé)
        timeline = [self._create_synthetic_standing_pose() for _ in range(90)]
        
        res = detector.detect_match_boundaries(timeline, fps=30.0)
        self.assertTrue(res["is_bounded"])
        self.assertTrue(res["has_initial_sonkyo"])
        self.assertTrue(res["has_final_sonkyo"])
        
        # Verificar Sonkyō Inicial no início do vídeo
        init_s = res["initial_sonkyo"]
        self.assertIsNotNone(init_s)
        self.assertEqual(init_s["start_frame"], 0)
        self.assertFalse(init_s["is_detected"])
        self.assertEqual(init_s["start_timestamp"], "00:00.000")
        
        # Verificar Sonkyō Final no final do vídeo
        fin_s = res["final_sonkyo"]
        self.assertIsNotNone(fin_s)
        self.assertEqual(fin_s["end_frame"], 89)
        self.assertFalse(fin_s["is_detected"])
        self.assertEqual(fin_s["end_timestamp"], "00:02.966")
        
        self.assertIn("início e término", res["status_message"].lower())


if __name__ == "__main__":
    unittest.main()


