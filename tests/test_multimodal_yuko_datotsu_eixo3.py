"""
Testes Unitários e de Integração para o Eixo 3:
Reconhecimento Multimodal de Golpes Válidos (Yuko-Datotsu).
Cobre:
1. Interação Atacante <-> Defensor (Colisão Shinai-Bogu e Eliminação de Ku-totsu).
2. Hasuji - O Ângulo da Lâmina (5° Pilar da Avaliação Biomecânica).
3. Detecção de Seme (Pressão e Intenção Pré-Golpe no Chushin-sen).
4. Detecção de Oji-waza e Debana (Contrataques e Inversão Dinâmica de Papéis).
5. Fusão Multimodal com Faixa de Áudio (Kiai e Estalo Datotsu-on com Sincronismo <= 40ms).
6. Modelo Temporal de Sequência de Poses (Action Spotting TCN com 10 Classes de Kendo).
7. Integração no Calibrador e no SenpAIPipeline.
"""

import unittest
import numpy as np
from typing import Dict, Any, List

from src.analytics.multimodal_yuko_datotsu import (
    TargetImpactEvaluator,
    HasujiEvaluator,
    SemeDetector,
    CounterattackDetector,
    AudioKiaiFusion,
    TemporalActionSpotter,
    MultimodalYukoDatotsuEngine
)
from src.analytics.biomechanics import BiomechanicsAnalyzer
from src.engine.calibrator import CalibrationEngine
from src.pipeline import SenpAIPipeline


class TestMultimodalYukoDatotsuEixo3(unittest.TestCase):

    def setUp(self):
        # Landmarks sintéticos padrão para Atacante e Defensor
        self.attacker_lm = {
            "RIGHT_WRIST": {"x": 0.48, "y": 0.30, "z": 0.0},
            "LEFT_WRIST": {"x": 0.46, "y": 0.32, "z": 0.0},
            "RIGHT_ELBOW": {"x": 0.42, "y": 0.38, "z": 0.0},
            "RIGHT_SHOULDER": {"x": 0.38, "y": 0.42, "z": 0.0},
            "RIGHT_HIP": {"x": 0.38, "y": 0.65, "z": 0.0},
            "LEFT_HIP": {"x": 0.36, "y": 0.65, "z": 0.0},
            "NOSE": {"x": 0.39, "y": 0.30, "z": 0.0}
        }
        self.defender_lm = {
            "NOSE": {"x": 0.62, "y": 0.28, "z": 0.0},
            "RIGHT_WRIST": {"x": 0.58, "y": 0.46, "z": 0.0},
            "LEFT_WRIST": {"x": 0.59, "y": 0.46, "z": 0.0},
            "RIGHT_SHOULDER": {"x": 0.62, "y": 0.38, "z": 0.0},
            "RIGHT_HIP": {"x": 0.62, "y": 0.65, "z": 0.0},
            "LEFT_HIP": {"x": 0.64, "y": 0.65, "z": 0.0}
        }

    # ==========================================================================
    # 1. TESTES DE INTERAÇÃO ATACANTE <-> DEFENSOR & KU-TOTSU (EIXO 3.1)
    # ==========================================================================
    def test_target_impact_valid_men_collision(self):
        """Valida impacto de Men próximo ao topo do Bogu do defensor em distância válida."""
        shinai_data = {"tip_norm": (0.62, 0.25), "angle_deg": 90.0}
        res = TargetImpactEvaluator.evaluate_target_collision(
            strike_type="MEN",
            attacker_landmarks=self.attacker_lm,
            defender_landmarks=self.defender_lm,
            shinai_data=shinai_data
        )
        self.assertFalse(res["is_ku_totsu"], "Golpe dentro da distância não deve ser Ku-totsu.")
        self.assertTrue(res["target_hit"], "Kensen incidente na cabeça do defensor deve validar impacto.")
        self.assertGreaterEqual(res["collision_score"], 0.70)

    def test_target_impact_ku_totsu_rejection(self):
        """Valida que golpe desferido fora do alcance (distância excessiva) é rejeitado como Ku-totsu."""
        far_defender = {
            "NOSE": {"x": 0.95, "y": 0.28, "z": 0.0},
            "RIGHT_HIP": {"x": 0.95, "y": 0.65, "z": 0.0},
            "LEFT_HIP": {"x": 0.96, "y": 0.65, "z": 0.0}
        }
        res = TargetImpactEvaluator.evaluate_target_collision(
            strike_type="MEN",
            attacker_landmarks=self.attacker_lm,
            defender_landmarks=far_defender
        )
        self.assertTrue(res["is_ku_totsu"], "Distância excessiva deve ser classificada como Ku-totsu.")
        self.assertFalse(res["target_hit"])
        self.assertLessEqual(res["collision_score"], 0.25)

    def test_target_impact_kote_and_do(self):
        """Valida impacto específico em Kote (antebraço) e Do (flanco)."""
        # Kote
        kote_res = TargetImpactEvaluator.evaluate_target_collision(
            strike_type="KOTE",
            attacker_landmarks=self.attacker_lm,
            defender_landmarks=self.defender_lm,
            shinai_data={"tip_norm": (0.58, 0.46)}
        )
        self.assertFalse(kote_res["is_ku_totsu"])
        self.assertGreaterEqual(kote_res["collision_score"], 0.70)

        # Do
        do_res = TargetImpactEvaluator.evaluate_target_collision(
            strike_type="DO",
            attacker_landmarks=self.attacker_lm,
            defender_landmarks=self.defender_lm,
            shinai_data={"tip_norm": (0.62, 0.62)}
        )
        self.assertFalse(do_res["is_ku_totsu"])
        self.assertGreaterEqual(do_res["collision_score"], 0.65)

    # ==========================================================================
    # 2. TESTES DE HASUJI - O ÂNGULO DA LÂMINA (EIXO 3.2)
    # ==========================================================================
    def test_hasuji_evaluation_men(self):
        """Valida que Men vertical (+-15°) é aceito e angulações inclinadas (>25°) são rejeitadas."""
        # Ângulo ideal 90° (vertical)
        res_ideal = HasujiEvaluator.evaluate_hasuji("MEN", blade_angle_deg=90.0)
        self.assertTrue(res_ideal["is_hasuji_valid"])
        self.assertGreaterEqual(res_ideal["hasuji_score"], 0.95)

        # Desvio aceitável (80° -> 10° de desvio <= 15°)
        res_ok = HasujiEvaluator.evaluate_hasuji("MEN", blade_angle_deg=80.0)
        self.assertTrue(res_ok["is_hasuji_valid"])
        self.assertGreaterEqual(res_ok["hasuji_score"], 0.80)

        # Desvio excessivo (55° -> 35° de desvio > 25° de cutoff - corte de chapa)
        res_bad = HasujiEvaluator.evaluate_hasuji("MEN", blade_angle_deg=55.0)
        self.assertFalse(res_bad["is_hasuji_valid"])
        self.assertLess(res_bad["hasuji_score"], 0.40)

    def test_hasuji_evaluation_do_and_tsuki(self):
        """Valida tolerâncias de Hasuji para Do (diagonal ~45°) e Tsuki (horizontal ~0°)."""
        res_do = HasujiEvaluator.evaluate_hasuji("DO", blade_angle_deg=45.0)
        self.assertTrue(res_do["is_hasuji_valid"])
        self.assertGreaterEqual(res_do["hasuji_score"], 0.95)

        res_tsuki = HasujiEvaluator.evaluate_hasuji("TSUKI", blade_angle_deg=2.0)
        self.assertTrue(res_tsuki["is_hasuji_valid"])
        self.assertGreaterEqual(res_tsuki["hasuji_score"], 0.95)

    # ==========================================================================
    # 3. TESTES DE DETECÇÃO DE SEME (EIXO 3.3)
    # ==========================================================================
    def test_seme_detection_advancing(self):
        """Valida que avanço com tronco ereto em direção ao oponente gera Seme positivo."""
        history = []
        # Simula 25 frames de avanço da esquerda para a direita (0.32 -> 0.40)
        for i in range(25):
            x_pos = 0.32 + (i / 25.0) * 0.08
            history.append({
                "RIGHT_HIP": {"x": x_pos, "y": 0.65},
                "RIGHT_SHOULDER": {"x": x_pos, "y": 0.40}  # Perfeitamente ereto (tilt ~0°)
            })
        def_hist = [{"RIGHT_HIP": {"x": 0.65, "y": 0.65}}] * 25

        seme_res = SemeDetector.evaluate_seme(history, def_hist, impact_frame=24)
        self.assertTrue(seme_res["is_seme_present"])
        self.assertIn("SEME_FORTE", seme_res["status"])
        self.assertGreaterEqual(seme_res["seme_score"], 0.85)

    def test_seme_detection_retreating(self):
        """Valida que golpe desferido em recuo desordenado é penalizado no Seme."""
        history = []
        # Simula recuo para longe do oponente (0.40 -> 0.30)
        for i in range(25):
            x_pos = 0.40 - (i / 25.0) * 0.10
            history.append({
                "RIGHT_HIP": {"x": x_pos, "y": 0.65},
                "RIGHT_SHOULDER": {"x": x_pos - 0.05, "y": 0.40}
            })
        def_hist = [{"RIGHT_HIP": {"x": 0.65, "y": 0.65}}] * 25

        seme_res = SemeDetector.evaluate_seme(history, def_hist, impact_frame=24)
        self.assertFalse(seme_res["is_seme_present"])
        self.assertIn("SEM_SEME_RECUO", seme_res["status"])
        self.assertLessEqual(seme_res["seme_score"], 0.50)

    # ==========================================================================
    # 4. TESTES DE OJI-WAZA E DEBANA (EIXO 3.4)
    # ==========================================================================
    def test_counterattack_detection_debana(self):
        """Valida identificação de Debana quando o oponente acelera os braços no instante prévio."""
        atk_hist = [self.attacker_lm] * 20
        # Oponente levantando as mãos rapidamente antes do impacto (Furikaburi)
        def_hist = []
        for i in range(20):
            # Mãos subindo de 0.50 para 0.25 (y diminui para cima)
            wy = 0.50 - (i / 20.0) * 0.25
            def_hist.append({
                "RIGHT_WRIST": {"x": 0.60, "y": wy},
                "LEFT_WRIST": {"x": 0.60, "y": wy}
            })

        counter_res = CounterattackDetector.detect_counterattack(atk_hist, def_hist, impact_frame=19)
        self.assertTrue(counter_res["is_counterattack"])
        self.assertEqual(counter_res["technique_category"], "OJI_WAZA")
        self.assertIn(counter_res["counterattack_type"], ["DEBANA_WAZA", "KAESHI_OU_NUKI_WAZA"])

    def test_direct_attack_shikake(self):
        """Valida que ataque de iniciativa direta sem movimento prévio do oponente é classificado como Shikake."""
        atk_hist = [self.attacker_lm] * 20
        def_hist = [self.defender_lm] * 20  # Oponente estático em guarda Kamae

        counter_res = CounterattackDetector.detect_counterattack(atk_hist, def_hist, impact_frame=19)
        self.assertFalse(counter_res["is_counterattack"])
        self.assertEqual(counter_res["technique_category"], "SHIKAKE_WAZA")

    # ==========================================================================
    # 5. TESTES DE FUSÃO MULTIMODAL DE ÁUDIO (EIXO 3.5)
    # ==========================================================================
    def test_audio_fusion_nonexistent_file_graceful_fallback(self):
        """Valida que arquivo sem áudio ou ausente não quebra a execução e faz fallback seguro."""
        res = AudioKiaiFusion.analyze_audio_events("non_existent_audio_video.mp4", impact_timestamp_sec=1.2)
        self.assertFalse(res["audio_present"])
        self.assertTrue(res["is_sync_valid"], "Vídeo silencioso não deve penalizar a sincronia.")

    # ==========================================================================
    # 6. TESTES DO CLASSIFICADOR TEMPORAL TCN DE 10 CLASSES (EIXO 3.6)
    # ==========================================================================
    def test_temporal_action_spotter_classes(self):
        """Valida supressão de falso disparo em Tsubazeriai e classificação correta de golpes."""
        spotter = TemporalActionSpotter()

        # Caso 1: Tsubazeriai (distância muito curta, baixa velocidade dos pulsos)
        clinch_seq = [self.attacker_lm] * 30
        res_tsuba = spotter.classify_sequence(clinch_seq, strike_type_hint="MEN", maai_distance=0.10)
        self.assertEqual(res_tsuba["predicted_class"], "TSUBAZERIAI")
        self.assertTrue(res_tsuba["suppress_false_trigger"], "Tsubazeriai deve ser suprimido de disparo de Ippon.")

        # Caso 2: Golpe Men com velocidade explosiva
        fast_seq = []
        for i in range(30):
            # Aceleração rápida na descida do corte
            wy = 0.20 + (i / 30.0) * 0.35
            fast_seq.append({
                "RIGHT_WRIST": {"x": 0.50, "y": wy},
                "LEFT_WRIST": {"x": 0.50, "y": wy}
            })
        res_men = spotter.classify_sequence(fast_seq, strike_type_hint="MEN", maai_distance=0.35)
        self.assertEqual(res_men["predicted_class"], "MEN_ATTACK")
        self.assertTrue(res_men["is_valid_strike_action"])

    # ==========================================================================
    # 7. TESTE DE INTEGRAÇÃO NO CALIBRADOR COM O 5° PILAR (HASUJI)
    # ==========================================================================
    def test_calibrator_5th_pillar_hasuji_integration(self):
        """Valida que o Calibrador integra hasuji_score como 5° pilar e rejeita Hasuji incorreto."""
        calibrator = CalibrationEngine(profile_name="normal")

        # Golpe com excelentes Ki-Ken-Tai-Ichi e Hasuji perfeito
        res_good = calibrator.evaluate_strike(
            target_score=0.90,
            fumikomi_score=0.85,
            posture_score=0.88,
            zanshin_score=0.80,
            strike_type="MEN",
            hasuji_score=0.95
        )
        self.assertTrue(res_good["is_valid"])
        self.assertIn("hasuji", res_good["sub_scores"])
        self.assertEqual(res_good["sub_scores"]["hasuji"], 95.0)

        # Golpe que passa nos 4 pilares mas falha severamente no Hasuji (corte com as costas da espada)
        res_bad_hasuji = calibrator.evaluate_strike(
            target_score=0.90,
            fumikomi_score=0.85,
            posture_score=0.88,
            zanshin_score=0.80,
            strike_type="MEN",
            hasuji_score=0.20  # Hasuji inaceitável (< 0.40)
        )
        self.assertFalse(res_bad_hasuji["is_valid"])
        self.assertIn("HASUJI_INCORRETO", res_bad_hasuji["failed_subcriteria"])

    def test_calibrator_ku_totsu_rejection(self):
        """Valida que o Calibrador invalida ponto com bandeira is_ku_totsu ativa."""
        calibrator = CalibrationEngine(profile_name="normal")
        res = calibrator.evaluate_strike(
            target_score=0.90,
            fumikomi_score=0.85,
            posture_score=0.85,
            zanshin_score=0.80,
            strike_type="MEN",
            is_ku_totsu=True
        )
        self.assertFalse(res["is_valid"])
        self.assertIn("KU_TOTSU_VAZIO", res["failed_subcriteria"])

    # ==========================================================================
    # 8. TESTE DO MOTOR COMPLETO MULTIMODAL YUKO-DATOTSU ENGINE
    # ==========================================================================
    def test_multimodal_engine_complete_evaluation(self):
        """Valida execução de ponta a ponta do MultimodalYukoDatotsuEngine."""
        engine = MultimodalYukoDatotsuEngine()

        history_atk = [self.attacker_lm] * 30
        history_def = [self.defender_lm] * 30

        result = engine.evaluate_complete_strike(
            strike_type="MEN",
            attacker_history=history_atk,
            defender_history=history_def,
            impact_frame=20,
            shinai_data={"tip_norm": (0.62, 0.28), "angle_deg": 88.0}
        )

        self.assertIn("multimodal_valid", result)
        self.assertIn("target_collision", result)
        self.assertIn("hasuji", result)
        self.assertIn("seme", result)
        self.assertIn("counterattack", result)
        self.assertIn("audio", result)
        self.assertIn("action_spotting", result)
        self.assertFalse(result["is_ku_totsu"])
        self.assertGreater(result["hasuji_score"], 0.70)


if __name__ == "__main__":
    unittest.main()
