"""
Testes unitários para o gerador de relatórios em PDF de Treinamento de Kendo (SenpAI).
"""

import unittest
from typing import Dict, Any

from src.utils.training_pdf_generator import (
    generate_training_pdf_report,
    generate_individual_kendoka_pdf_report,
    _clean_text_for_pdf
)
from src.analytics.training_analyzer import (
    TRAINING_MODALITIES_METADATA,
    TrainingAnalyzer,
    recalculate_blocks_from_segments,
    sync_kendoka_names_in_training_data,
    merge_kendokas_in_training_data
)
from src.utils.training_package_manager import (
    export_training_session_package,
    import_training_session_package
)


class TestTrainingPDFGenerator(unittest.TestCase):
    def setUp(self):
        self.sample_res = {
            "duration_seconds": 65.4,
            "processing_fps": 30.0,
            "training_analysis": {
                "modality": "kirikaeshi",
                "detection_method": "AUTO_DETECTED",
                "detection_confidence": 0.96,
                "description": "Sequência clássica de golpes contínuos: Shomen inicial, cortes Sayu-men alternados, Taiatari e Shomen final.",
                "justification": "Cadência e sincronismo bilateral Sayu-men detectados com alta confiança.",
                "learned_principles": [
                    "Sincronismo Ki-Ken-Tai-Ichi",
                    "Cortes em ângulo de 45 graus no Sayu-men",
                    "Trabalho de pés Suri-ashi contínuo"
                ],
                "web_sources": [
                    {"title": "Manual de Fundamentos da All Japan Kendo Federation (AJKF)", "type": "Oficial AJKF"},
                    {"title": "Normas de Arbitragem da International Kendo Federation (FIK)", "type": "Oficial FIK"}
                ],
                "kendokas": [
                    {
                        "kendoka_id": "KENSHI_01",
                        "default_name": "Kendoca 1",
                        "custom_name": "Eduardo Zimermann",
                        "role": "Kakarite (Atacante)",
                        "presence_seconds": 62.0,
                        "reentry_count": 0,
                        "pillars": {
                            "overall_score": 87.5,
                            "movimentacao": 84.0,
                            "precisao": 91.0,
                            "constancia": 88.0,
                            "cadence_cpm": 60.0,
                            "cadence_std_dev_seconds": 0.15,
                            "total_repetitions": 40,
                            "movimentacao_submetrics": {
                                "verticalidade_coluna": 92.0,
                                "nivelamento_ombros": 85.0,
                                "alinhamento_base_pes": 80.0,
                                "amplitude_furikaburi": 82.0
                            },
                            "precisao_submetrics": {
                                "trajetoria_alvo": 92.0,
                                "kikentai_sincronismo": 88.0,
                                "controle_linha_centro": 90.0,
                                "hasuji_angle_consistency": 86.0
                            },
                            "constancia_submetrics": {
                                "regularidade_ritmo": 89.0,
                                "resistencia_fadiga": 86.0
                            }
                        },
                        "strengths": [
                            "Excelente verticalidade e postura (Shisei)",
                            "Ataque Shomen central vigoroso com Fumikomi limpo"
                        ],
                        "improvements": [
                            "Evitar elevação do calcanhar esquerdo durante o recuo",
                            "Relaxar ombros para ampliar a fluidez no Sayu-men"
                        ],
                        "recommended_exercises": [
                            {
                                "name": "Hayasuburi com ênfase em respiração diafragmática",
                                "target": "Constância e Relaxamento",
                                "prescription": "3 séries de 50 repetições mantendo cadência uniforme."
                            }
                        ]
                    },
                    {
                        "kendoka_id": "KENSHI_02",
                        "default_name": "Kendoca 2",
                        "custom_name": "Sensei Kenji",
                        "role": "Motodachi (Receptor)",
                        "presence_seconds": 65.0,
                        "reentry_count": 0,
                        "pillars": {
                            "overall_score": 94.0,
                            "movimentacao": 95.0,
                            "precisao": 93.0,
                            "constancia": 94.0,
                            "cadence_cpm": 60.0,
                            "cadence_std_dev_seconds": 0.10,
                            "total_repetitions": 40,
                            "movimentacao_submetrics": {
                                "verticalidade_coluna": 97.0,
                                "nivelamento_ombros": 95.0,
                                "alinhamento_base_pes": 94.0,
                                "amplitude_furikaburi": 95.0
                            },
                            "precisao_submetrics": {
                                "trajetoria_alvo": 94.0,
                                "kikentai_sincronismo": 93.0,
                                "controle_linha_centro": 92.0,
                                "hasuji_angle_consistency": 92.0
                            },
                            "constancia_submetrics": {
                                "regularidade_ritmo": 95.0,
                                "resistencia_fadiga": 92.0
                            }
                        },
                        "strengths": [
                            "Recepção canônica com ângulo perfeito de espada",
                            "Estabilidade irrepreensível de Kamae"
                        ],
                        "improvements": [],
                        "recommended_exercises": []
                    }
                ]
            }
        }

    def test_clean_text_for_pdf(self):
        """Verifica higienização de texto, remoção de caracteres incompatíveis e conversão de emojis."""
        raw = "🥋 Treino de Kendo ⏱️ com cadência e acentuação: Ação, Pontuação & Árvore <tag>"
        cleaned = _clean_text_for_pdf(raw)
        self.assertIn("[Kendo]", cleaned)
        self.assertIn("[Tempo]", cleaned)
        self.assertIn("Ação", cleaned)
        self.assertIn("Pontuação", cleaned)
        self.assertIn("&amp;", cleaned)
        self.assertIn("&lt;tag&gt;", cleaned)

    def test_generate_consolidated_pdf_report(self):
        """Valida que o relatório PDF consolidado é gerado com cabeçalho válido do formato PDF."""
        pdf_bytes = generate_training_pdf_report(
            self.sample_res,
            kendoka_names={"KENSHI_01": "Eduardo Zimermann", "KENSHI_02": "Sensei Kenji"},
            modality_key="kirikaeshi"
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # Assinatura mágica do arquivo PDF
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_generate_individual_pdf_report(self):
        """Valida geração do PDF individual de um kendoca."""
        k_prof = self.sample_res["training_analysis"]["kendokas"][0]
        meta = TRAINING_MODALITIES_METADATA["kirikaeshi"]
        pdf_bytes = generate_individual_kendoka_pdf_report(k_prof, meta, "Eduardo Zimermann")
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_generate_pdf_with_minimal_data(self):
        """Valida resiliência do gerador quando faltam campos opcionais."""
        minimal_res = {
            "duration_seconds": 15.0,
            "training_analysis": {
                "modality": "kihon",
                "kendokas": []
            }
        }
        pdf_bytes = generate_training_pdf_report(minimal_res)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 500)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_generate_pdf_with_multi_session_modality_blocks(self):
        """Valida geração de PDF consolidado e individual com múltiplas sessões e blocos concentrados por modalidade."""
        multi_res = dict(self.sample_res)
        multi_res["duration_seconds"] = 300.0
        train_copy = dict(self.sample_res["training_analysis"])
        train_copy["sub_sessions"] = [
            {
                "segment_index": 1,
                "start_seconds": 0.0,
                "end_seconds": 180.0,
                "start_timestamp": "00:00",
                "end_timestamp": "03:00",
                "duration_seconds": 180.0,
                "modality_key": "kirikaeshi",
                "modality_name": "Kirikaeshi (切り返し)",
                "modality_category": "Exercício Combinado Fundamental",
                "detection_confidence": 0.94,
                "detection_method": "AUTO_DETECTED",
                "total_repetitions": 40,
                "kendokas": train_copy["kendokas"]
            },
            {
                "segment_index": 2,
                "start_seconds": 180.0,
                "end_seconds": 300.0,
                "start_timestamp": "03:00",
                "end_timestamp": "05:00",
                "duration_seconds": 120.0,
                "modality_key": "waza_geiko",
                "modality_name": "Waza-geiko (技稽古)",
                "modality_category": "Treinamento Técnico Tático",
                "detection_confidence": 0.91,
                "detection_method": "AUTO_DETECTED",
                "total_repetitions": 15,
                "kendokas": train_copy["kendokas"]
            }
        ]
        train_copy["modality_blocks"] = [
            {
                "block_index": 1,
                "modality_key": "kirikaeshi",
                "modality_name": "Kirikaeshi (切り返し)",
                "modality_category": "Exercício Combinado Fundamental",
                "intervals": [{"segment_index": 1, "start_timestamp": "00:00", "end_timestamp": "03:00", "duration_seconds": 180.0, "repetitions": 40}],
                "intervals_label": "00:00 a 03:00",
                "session_count": 1,
                "total_duration_seconds": 180.0,
                "total_repetitions": 40,
                "kendokas": train_copy["kendokas"]
            },
            {
                "block_index": 2,
                "modality_key": "waza_geiko",
                "modality_name": "Waza-geiko (技稽古)",
                "modality_category": "Treinamento Técnico Tático",
                "intervals": [{"segment_index": 2, "start_timestamp": "03:00", "end_timestamp": "05:00", "duration_seconds": 120.0, "repetitions": 15}],
                "intervals_label": "03:00 a 05:00",
                "session_count": 1,
                "total_duration_seconds": 120.0,
                "total_repetitions": 15,
                "kendokas": train_copy["kendokas"]
            }
        ]
        multi_res["training_analysis"] = train_copy

        pdf_bytes = generate_training_pdf_report(
            multi_res,
            kendoka_names={"KENSHI_01": "Eduardo Zimermann", "KENSHI_02": "Sensei Kenji"},
            modality_key="kirikaeshi"
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1500)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

        indiv_pdf = generate_individual_kendoka_pdf_report(
            kendoka_profile=train_copy["kendokas"][0],
            modality_meta=TRAINING_MODALITIES_METADATA["kirikaeshi"],
            custom_name="Eduardo Zimermann",
            modality_blocks=train_copy["modality_blocks"],
            sub_sessions=train_copy["sub_sessions"]
        )
        self.assertIsInstance(indiv_pdf, bytes)
        self.assertGreater(len(indiv_pdf), 1200)
        self.assertTrue(indiv_pdf.startswith(b"%PDF-"))

    def test_sequence_segmentation_and_modality_blocks_concentration(self):
        """
        Valida a identificação de múltiplas sessões na mesma sequência (ex: 00:00 a 03:00 Kirikaeshi,
        03:00 a 05:00 Waza-geiko) e a concentração de blocos por modalidade no relatório.
        """
        analyzer = TrainingAnalyzer()
        fps = 10.0
        total_frames = 3000  # 300s (05:00)

        def make_pose(x_val: float, wrist_y: float = 0.40) -> Dict[str, Any]:
            return {
                "NOSE": {"x": x_val, "y": 0.25},
                "LEFT_SHOULDER": {"x": x_val - 0.05, "y": 0.35},
                "RIGHT_SHOULDER": {"x": x_val + 0.05, "y": 0.35},
                "LEFT_HIP": {"x": x_val - 0.04, "y": 0.60},
                "RIGHT_HIP": {"x": x_val + 0.04, "y": 0.60},
                "LEFT_WRIST": {"x": x_val, "y": wrist_y},
                "RIGHT_WRIST": {"x": x_val, "y": wrist_y},
                "LEFT_ANKLE": {"x": x_val - 0.04, "y": 0.88},
                "RIGHT_ANKLE": {"x": x_val + 0.04, "y": 0.89},
            }

        primary_hist = [make_pose(0.35) for _ in range(total_frames)]
        secondary_hist = [make_pose(0.65) for _ in range(total_frames)]

        # Segmentos explícitos: 00:00-02:00 Kirikaeshi, 02:00-04:00 Waza-geiko, 04:00-05:00 Kirikaeshi
        segments_override = [
            {"start_time": "00:00", "end_time": "02:00", "modality_key": "kirikaeshi"},
            {"start_time": "02:00", "end_time": "04:00", "modality_key": "waza_geiko"},
            {"start_time": "04:00", "end_time": "05:00", "modality_key": "kirikaeshi"},
        ]

        res = analyzer.analyze_session(
            primary_history=primary_hist,
            secondary_history=secondary_hist,
            detected_strikes=[],
            fps=fps,
            session_segments_override=segments_override
        )
        res_dict = res.to_dict()

        self.assertEqual(len(res_dict["sub_sessions"]), 3)
        self.assertEqual(res_dict["sub_sessions"][0]["modality_key"], "kirikaeshi")
        self.assertEqual(res_dict["sub_sessions"][0]["start_timestamp"], "00:00")
        self.assertEqual(res_dict["sub_sessions"][0]["end_timestamp"], "02:00")
        self.assertEqual(res_dict["sub_sessions"][1]["modality_key"], "waza_geiko")
        self.assertEqual(res_dict["sub_sessions"][1]["start_timestamp"], "02:00")
        self.assertEqual(res_dict["sub_sessions"][1]["end_timestamp"], "04:00")
        self.assertEqual(res_dict["sub_sessions"][2]["modality_key"], "kirikaeshi")
        self.assertEqual(res_dict["sub_sessions"][2]["start_timestamp"], "04:00")
        self.assertEqual(res_dict["sub_sessions"][2]["end_timestamp"], "05:00")

        # Os 2 trechos de Kirikaeshi devem estar concentrados em um único bloco de modalidade!
        self.assertEqual(len(res_dict["modality_blocks"]), 2)
        b_kiri = res_dict["modality_blocks"][0]
        b_waza = res_dict["modality_blocks"][1]
        self.assertEqual(b_kiri["modality_key"], "kirikaeshi")
        self.assertEqual(b_kiri["session_count"], 2)
        self.assertEqual(b_kiri["intervals_label"], "00:00 a 02:00, 04:00 a 05:00")
        self.assertAlmostEqual(b_kiri["total_duration_seconds"], 180.0, places=1)

        self.assertEqual(b_waza["modality_key"], "waza_geiko")
        self.assertEqual(b_waza["session_count"], 1)
        self.assertEqual(b_waza["intervals_label"], "02:00 a 04:00")
        self.assertAlmostEqual(b_waza["total_duration_seconds"], 120.0, places=1)

        # Relatório Markdown individual deve conter a seção separada por bloco de modalidade
        md_report = res.kendokas[0].generate_individual_report_markdown(
            TRAINING_MODALITIES_METADATA["kirikaeshi"],
            modality_blocks=res_dict["modality_blocks"]
        )
        self.assertIn("Avaliação Separada por Bloco de Modalidade de Treinamento", md_report)
        self.assertIn("00:00 a 02:00, 04:00 a 05:00", md_report)
        self.assertIn("02:00 a 04:00", md_report)

        # Recálculo interativo pós-análise (00:00 a 03:00 Kirikaeshi, 03:00 a 05:00 Waza-geiko)
        recalculate_blocks_from_segments(
            res_dict,
            segments_override=[
                {"start_time": "00:00", "end_time": "03:00", "modality_key": "kirikaeshi"},
                {"start_time": "03:00", "end_time": "05:00", "modality_key": "waza_geiko"},
            ],
            fps=fps
        )
        self.assertEqual(len(res_dict["sub_sessions"]), 2)
        self.assertEqual(res_dict["modality_blocks"][0]["intervals_label"], "00:00 a 03:00")
        self.assertEqual(res_dict["modality_blocks"][1]["intervals_label"], "03:00 a 05:00")

        # Sincronização de nomes em kendokas, sub_sessions e modality_blocks
        sync_kendoka_names_in_training_data(res_dict, {"KENSHI_1": "Eduardo Sensei"})
        self.assertEqual(res_dict["kendokas"][0]["custom_name"], "Eduardo Sensei")
        self.assertEqual(res_dict["sub_sessions"][0]["kendokas"][0]["custom_name"], "Eduardo Sensei")
        self.assertEqual(res_dict["modality_blocks"][0]["kendokas"][0]["custom_name"], "Eduardo Sensei")

        # Valida analyze_multi_session com golpes detectados (garante ausência de KeyError: 'event_index')
        dummy_strike = {"impact_frame": 50, "timestamp": "00:05", "type": "MEN"}
        multi_res = analyzer.analyze_multi_session(
            kenshi_histories={"KENSHI_01": primary_hist, "KENSHI_02": secondary_hist},
            detected_strikes_by_kenshi={"KENSHI_01": [dummy_strike], "KENSHI_02": []},
            fps=fps,
            session_segments_override=[
                {"start_time": "00:00", "end_time": "03:00", "modality_key": "kirikaeshi"},
                {"start_time": "03:00", "end_time": "05:00", "modality_key": "waza_geiko"},
            ]
        )
        self.assertEqual(len(multi_res.sub_sessions), 2)
        self.assertEqual(len(multi_res.modality_blocks), 2)


if __name__ == "__main__":
    unittest.main()
