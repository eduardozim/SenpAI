"""
Testes Automatizados para Exportação e Importação de Pacotes de Treinamento (training_package_manager.py).
Valida o salvamento completo de resultados de análise de treino (vídeo analisado + dados do relatório),
exportação em pacote zip e importação para continuidade da análise no Modo de Treinamento & Aprendizado.
"""

import io
import os
import json
import zipfile
import tempfile
import unittest
import numpy as np

from src.utils.training_package_manager import (
    export_training_session_package,
    import_training_session_package,
    is_valid_training_package,
    sanitize_for_json,
    PACKAGE_IDENTIFIER,
    PACKAGE_VERSION
)
from src.analytics.training_analyzer import (
    TrainingAnalyzer,
    merge_kendoka_profiles_dict,
    differentiate_kendoka_names,
    recalculate_blocks_from_segments,
    sync_kendoka_names_in_training_data,
    merge_kendokas_in_training_data
)


class TestTrainingPackageManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.extract_dir = os.path.join(self.temp_dir.name, "extracted")

        # Cria arquivos fictícios de vídeo original e anotado
        self.dummy_original_video = os.path.join(self.temp_dir.name, "dummy_original.mp4")
        with open(self.dummy_original_video, "wb") as f:
            f.write(b"ORIGINAL_VIDEO_BINARY_DATA_TEST_12345")

        self.dummy_annotated_video = os.path.join(self.temp_dir.name, "annotated_match.mp4")
        with open(self.dummy_annotated_video, "wb") as f:
            f.write(b"ANNOTATED_VIDEO_BINARY_DATA_TEST_67890")

        # Dados completos de teste de análise de treinamento
        self.sample_analysis_result = {
            "duration_seconds": 45.5,
            "total_frames": 1365,
            "fps": 30.0,
            "processing_time_seconds": 12.34,
            "processing_fps": 110.6,
            "video_path": self.dummy_original_video,
            "training_analysis": {
                "modality_key": "suburi",
                "modality_name": "Suburi (素振り)",
                "detection_confidence": 0.94,
                "detection_method": "AUTO_DETECTED",
                "is_manual_override": False,
                "duration_seconds": 45.5,
                "total_frames_analyzed": 1365,
                "kendokas": [
                    {
                        "kendoka_id": "KENSHI_1",
                        "default_name": "Kendoca 1",
                        "custom_name": "Musashi Miyamoto",
                        "role": "Praticante Principal",
                        "pillars": {
                            "overall_score": np.float64(88.5),
                            "movimentacao": np.float64(90.0),
                            "precisao": np.float64(86.0),
                            "constancia": np.float64(89.5),
                            "cadence_cpm": np.float64(34.2),
                            "total_repetitions": np.int64(26),
                            "cadence_std_dev_seconds": np.float64(0.18),
                            "movimentacao_submetrics": {
                                "verticalidade_coluna": 92.0,
                                "nivelamento_ombros": 88.0,
                                "alinhamento_base_pes": 90.0,
                                "amplitude_furikaburi": 90.0
                            },
                            "precisao_submetrics": {
                                "trajetoria_alvo": 87.0,
                                "kikentai_sincronismo": 85.0,
                                "controle_linha_centro": 86.0
                            },
                            "constancia_submetrics": {
                                "regularidade_ritmo": 91.0,
                                "resistencia_fadiga": 88.0,
                                "adequacao_cadencia": 89.5
                            }
                        },
                        "strengths": ["Coluna perfeitamente vertical", "Ritmo consistente"],
                        "improvements": ["Ajustar elevação do calcanhar esquerdo"],
                        "recommended_exercises": [
                            {"name": "Jōge-buri com Pausa", "target": "Ombros", "prescription": "50 repetições diárias"}
                        ]
                    }
                ],
                "session_summary_text": "Treino de Suburi executado com alta consistência.",
                "justification": "Movimentos de corte repetidos no ar identificados.",
                "learned_principles": ["Tenouchi firme no momento do corte", "Coluna ereta (Shisei)"],
                "web_sources": [{"title": "Guia Oficial de Suburi", "url": "https://kendo.org/suburi", "type": "Web"}]
            }
        }

        self.kendoka_names = {"KENSHI_1": "Musashi Miyamoto"}
        self.session_reviews = {"review_1": {"notes": "Excelente postura", "reviewer_dan": 4}}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sanitize_for_json(self):
        """Valida que tipos complexos do NumPy e estruturas são convertidos adequadamente para tipos nativos."""
        raw_data = {
            "score": np.float64(85.5),
            "reps": np.int64(30),
            "is_valid": np.bool_(True),
            "array": np.array([1, 2, 3]),
            "nested": {
                "nan_val": float("nan"),
                "inf_val": float("inf")
            }
        }
        sanitized = sanitize_for_json(raw_data)
        # Deve ser serializável em JSON sem erros
        json_str = json.dumps(sanitized)
        self.assertIn('"score": 85.5', json_str)
        self.assertIn('"reps": 30', json_str)
        self.assertIn('"is_valid": true', json_str)
        self.assertIn('"array": [1, 2, 3]', json_str)
        self.assertEqual(sanitized["nested"]["nan_val"], 0.0)
        self.assertEqual(sanitized["nested"]["inf_val"], 0.0)

    def test_export_package_as_bytes(self):
        """Valida a exportação da sessão de treinamento em memória (bytes) contendo vídeo e relatório."""
        zip_bytes = export_training_session_package(
            analysis_result=self.sample_analysis_result,
            video_path=self.dummy_original_video,
            annotated_video_path=self.dummy_annotated_video,
            kendoka_names=self.kendoka_names,
            invert_aka_shiro=True,
            session_reviews=self.session_reviews
        )

        self.assertIsInstance(zip_bytes, bytes)
        self.assertGreater(len(zip_bytes), 0)

        # Abre o ZIP a partir dos bytes e confere a integridade
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            namelist = zf.namelist()
            self.assertIn("training_session.json", namelist)
            self.assertIn("analyzed_video.mp4", namelist)
            self.assertIn("original_video.mp4", namelist)

            # Confere o conteúdo do JSON
            data = json.loads(zf.read("training_session.json").decode("utf-8"))
            self.assertEqual(data["metadata"]["package_type"], PACKAGE_IDENTIFIER)
            self.assertEqual(data["metadata"]["modality_key"], "suburi")
            self.assertEqual(data["training_kendoka_names"]["KENSHI_1"], "Musashi Miyamoto")
            self.assertTrue(data["invert_aka_shiro"])

    def test_export_package_to_file(self):
        """Valida a exportação direta para um arquivo em disco."""
        target_zip = os.path.join(self.temp_dir.name, "export_test.zip")
        out_path = export_training_session_package(
            analysis_result=self.sample_analysis_result,
            video_path=self.dummy_original_video,
            annotated_video_path=self.dummy_annotated_video,
            kendoka_names=self.kendoka_names,
            destination=target_zip
        )

        self.assertEqual(out_path, os.path.abspath(target_zip))
        self.assertTrue(os.path.isfile(target_zip))

    def test_is_valid_training_package(self):
        """Valida a função de verificação rápida de pacotes de treinamento."""
        zip_bytes = export_training_session_package(
            analysis_result=self.sample_analysis_result,
            video_path=self.dummy_original_video,
            annotated_video_path=self.dummy_annotated_video
        )

        is_valid, msg = is_valid_training_package(zip_bytes)
        self.assertTrue(is_valid, msg)

        # Arquivo ZIP arbitrário sem o JSON esperado
        bad_buf = io.BytesIO()
        with zipfile.ZipFile(bad_buf, "w") as bzf:
            bzf.writestr("outra_coisa.txt", "teste")
        is_valid_bad, msg_bad = is_valid_training_package(bad_buf.getvalue())
        self.assertFalse(is_valid_bad)
        self.assertIn("ausente", msg_bad)

        # Bytes corrompidos
        is_valid_corrupt, _ = is_valid_training_package(b"NOT_A_ZIP_FILE")
        self.assertFalse(is_valid_corrupt)

    def test_import_training_package_full_flow(self):
        """Valida o fluxo completo de exportação e subsequente importação para continuidade da análise."""
        zip_bytes = export_training_session_package(
            analysis_result=self.sample_analysis_result,
            video_path=self.dummy_original_video,
            annotated_video_path=self.dummy_annotated_video,
            kendoka_names=self.kendoka_names,
            invert_aka_shiro=True,
            session_reviews=self.session_reviews
        )

        # Importa para a pasta de extração temporária
        imported = import_training_session_package(
            package_source=zip_bytes,
            extract_dir=self.extract_dir
        )

        self.assertTrue(imported["success"])
        self.assertTrue(os.path.isfile(imported["video_file_path"]))
        self.assertTrue(os.path.isfile(imported["annotated_output"]))
        self.assertEqual(imported["training_kendoka_names"]["KENSHI_1"], "Musashi Miyamoto")
        self.assertTrue(imported["invert_aka_shiro"])
        self.assertEqual(imported["session_reviews"], self.session_reviews)

        # Valida que o resultado analítico foi restaurado com fidelidade
        restored_res = imported["analysis_result"]
        self.assertEqual(restored_res["training_analysis"]["modality_key"], "suburi")
        self.assertEqual(restored_res["total_frames"], 1365)
        # O caminho do vídeo interno do resultado da análise deve apontar para o novo arquivo extraído
        self.assertEqual(restored_res["video_path"], imported["video_file_path"])

        # Confere o conteúdo binário dos vídeos extraídos
        with open(imported["video_file_path"], "rb") as f:
            self.assertEqual(f.read(), b"ORIGINAL_VIDEO_BINARY_DATA_TEST_12345")
        with open(imported["annotated_output"], "rb") as f:
            self.assertEqual(f.read(), b"ANNOTATED_VIDEO_BINARY_DATA_TEST_67890")

    def test_import_with_only_annotated_video(self):
        """Valida que quando apenas o vídeo analisado for incluído, a importação resolve ambos os caminhos com sucesso."""
        zip_bytes = export_training_session_package(
            analysis_result=self.sample_analysis_result,
            video_path=None,
            annotated_video_path=self.dummy_annotated_video,
            kendoka_names=self.kendoka_names
        )

        imported = import_training_session_package(
            package_source=zip_bytes,
            extract_dir=self.extract_dir
        )

        self.assertTrue(imported["success"])
        self.assertTrue(os.path.isfile(imported["video_file_path"]))
        self.assertTrue(os.path.isfile(imported["annotated_output"]))
        self.assertEqual(imported["video_file_path"], imported["annotated_output"])

    def test_import_zip_slip_protection(self):
        """Valida que tentativas de Directory Traversal (Zip Slip) são bloqueadas na extração."""
        malicious_buf = io.BytesIO()
        with zipfile.ZipFile(malicious_buf, "w") as zf:
            pkg_data = {
                "metadata": {"package_type": PACKAGE_IDENTIFIER, "format_version": PACKAGE_VERSION},
                "analysis_result": sanitize_for_json(self.sample_analysis_result)
            }
            zf.writestr("training_session.json", json.dumps(pkg_data))
            zf.writestr("../malicious.txt", "conteudo perigoso")
            zf.writestr("analyzed_video.mp4", b"VIDEO_DATA")

        # Extrai em uma pasta segura
        imported = import_training_session_package(
            package_source=malicious_buf.getvalue(),
            extract_dir=self.extract_dir
        )
        self.assertTrue(imported["success"])
        # O arquivo ../malicious.txt NÃO deve ter sido gravado fora da pasta
        outside_path = os.path.join(self.temp_dir.name, "malicious.txt")
        self.assertFalse(os.path.exists(outside_path))

    def test_workflow_continue_analysis_after_import(self):
        """
        Valida o fluxo contínuo de trabalho do usuário:
        1. Executa análise inicial e exporta o pacote com o vídeo analisado e o relatório.
        2. Importa o pacote mais tarde para dar continuidade à análise.
        3. Realiza novas edições durante a continuação da análise (novo nome de Kendoka, inversão de lados e nova nota).
        4. Re-exporta o pacote atualizado.
        5. Re-importa e comprova que todas as alterações da continuação da análise foram mantidas.
        """
        # 1. Export inicial
        pkg1_bytes = export_training_session_package(
            analysis_result=self.sample_analysis_result,
            video_path=self.dummy_original_video,
            annotated_video_path=self.dummy_annotated_video,
            kendoka_names={"KENSHI_1": "Kenshi Inicial"},
            invert_aka_shiro=False
        )

        # 2. Import para continuar
        extract_dir1 = os.path.join(self.temp_dir.name, "session_continue_1")
        imported1 = import_training_session_package(pkg1_bytes, extract_dir=extract_dir1)
        self.assertEqual(imported1["training_kendoka_names"]["KENSHI_1"], "Kenshi Inicial")
        self.assertFalse(imported1["invert_aka_shiro"])

        # 3. Usuário continua a análise: renomeia kendoka, inverte os lados e adiciona revisão
        updated_res = imported1["analysis_result"]
        updated_k_names = {"KENSHI_1": "Sensei Kojiro"}
        updated_invert = True
        updated_reviews = {"review_pos_analise": {"observacao": "Correção de Kamae aplicada", "grau": 5}}

        # 4. Re-exporta a análise continuada
        pkg2_bytes = export_training_session_package(
            analysis_result=updated_res,
            video_path=imported1["video_file_path"],
            annotated_video_path=imported1["annotated_output"],
            kendoka_names=updated_k_names,
            invert_aka_shiro=updated_invert,
            session_reviews=updated_reviews
        )

        # 5. Re-importa para validar a persistência total das alterações de continuação
        extract_dir2 = os.path.join(self.temp_dir.name, "session_continue_2")
        imported2 = import_training_session_package(pkg2_bytes, extract_dir=extract_dir2)

        self.assertTrue(imported2["success"])
        self.assertEqual(imported2["training_kendoka_names"]["KENSHI_1"], "Sensei Kojiro")
        self.assertTrue(imported2["invert_aka_shiro"])
        self.assertIn("review_pos_analise", imported2["session_reviews"])
        self.assertEqual(imported2["session_reviews"]["review_pos_analise"]["grau"], 5)
        self.assertTrue(os.path.isfile(imported2["video_file_path"]))
        self.assertTrue(os.path.isfile(imported2["annotated_output"]))

    def test_kendoka_custom_name_preservation_across_all_id_variants(self):
        """
        Valida que o nome alterado do Kendoca é salvo no pacote (.zip) e restaurado
        com sucesso em todas as variações de identificador (KENSHI_01, KENSHI_1)
        e diretamente no perfil do kendoka em analysis_result['training_analysis']['kendokas'].
        """
        res_copy = sanitize_for_json(self.sample_analysis_result)
        res_copy["training_analysis"]["kendokas"][0]["kendoka_id"] = "KENSHI_01"
        res_copy["training_analysis"]["kendokas"][0]["default_name"] = "Kenshi 01"
        res_copy["training_analysis"]["kendokas"][0]["custom_name"] = "Kenshi 01"

        altered_name = "Miyamoto Musashi Sensei"
        pkg_bytes = export_training_session_package(
            analysis_result=res_copy,
            video_path=self.dummy_original_video,
            annotated_video_path=self.dummy_annotated_video,
            kendoka_names={"KENSHI_01": altered_name}
        )

        extract_dir = os.path.join(self.temp_dir.name, "session_altered_name")
        imported = import_training_session_package(pkg_bytes, extract_dir=extract_dir)

        # Verifica dicionário de nomes
        self.assertEqual(imported["training_kendoka_names"].get("KENSHI_01"), altered_name)
        self.assertEqual(imported["training_kendoka_names"].get("KENSHI_1"), altered_name)

        # Verifica perfil interno no resultado analítico
        imported_kendokas = imported["analysis_result"]["training_analysis"]["kendokas"]
        self.assertEqual(len(imported_kendokas), 1)
        self.assertEqual(imported_kendokas[0]["custom_name"], altered_name)

    def test_merge_kendoka_profiles_dict(self):
        """Valida a consolidação de dados e recalibração dos 3 Pilares ao mesclar perfis com mesmo nome."""
        p1 = {
            "kendoka_id": "KENSHI_01",
            "default_name": "Kendoca 1",
            "custom_name": "Eduardo",
            "presence_seconds": 20.0,
            "presence_frames": 600,
            "reentry_count": 1,
            "repetition_timeline": [
                {"repetition": 1, "timestamp_seconds": 2.0, "technique": "MEN"},
                {"repetition": 2, "timestamp_seconds": 6.0, "technique": "MEN"}
            ],
            "pillars": {
                "overall_score": 80.0,
                "movimentacao": 80.0,
                "precisao": 80.0,
                "constancia": 80.0,
                "cadence_cpm": 25.0,
                "total_repetitions": 2,
                "cadence_std_dev_seconds": 0.20
            },
            "strengths": ["Boa postura inicial"],
            "improvements": ["Ajustar pé esquerdo"],
            "recommended_exercises": [{"name": "Suburi Rápido", "prescription": "30x"}]
        }

        p2 = {
            "kendoka_id": "KENSHI_05",
            "default_name": "Kendoca 5",
            "custom_name": "Eduardo",
            "presence_seconds": 30.0,
            "presence_frames": 900,
            "reentry_count": 0,
            "repetition_timeline": [
                {"repetition": 1, "timestamp_seconds": 12.0, "technique": "MEN"},
                {"repetition": 2, "timestamp_seconds": 16.0, "technique": "MEN"},
                {"repetition": 3, "timestamp_seconds": 20.0, "technique": "KOTE"}
            ],
            "pillars": {
                "overall_score": 90.0,
                "movimentacao": 90.0,
                "precisao": 90.0,
                "constancia": 90.0,
                "cadence_cpm": 30.0,
                "total_repetitions": 3,
                "cadence_std_dev_seconds": 0.15
            },
            "strengths": ["Kiai forte"],
            "improvements": ["Zanshin mais firme"],
            "recommended_exercises": [{"name": "Kirikaeshi", "prescription": "2x"}]
        }

        merged = merge_kendoka_profiles_dict(p1, p2, "Eduardo Zim")

        self.assertEqual(merged["kendoka_id"], "KENSHI_01")
        self.assertEqual(merged["custom_name"], "Eduardo Zim")
        self.assertEqual(merged["presence_seconds"], 50.0)
        self.assertEqual(merged["presence_frames"], 1500)
        self.assertEqual(merged["reentry_count"], 2)  # 1 + 0 + 1
        self.assertEqual(len(merged["repetition_timeline"]), 5)
        self.assertEqual(merged["pillars"]["total_repetitions"], 5)
        # Verifica timeline reindexada
        self.assertEqual(merged["repetition_timeline"][0]["repetition"], 1)
        self.assertEqual(merged["repetition_timeline"][4]["repetition"], 5)
        # Média ponderada dos pilares (pesos 2 e 3): (80*2 + 90*3)/5 = 86.0
        self.assertAlmostEqual(merged["pillars"]["movimentacao"], 86.0, places=1)
        self.assertAlmostEqual(merged["pillars"]["precisao"], 86.0, places=1)
        self.assertAlmostEqual(merged["pillars"]["constancia"], 86.0, places=1)
        # Exercícios e forças consolidados
        self.assertEqual(len(merged["strengths"]), 2)
        self.assertEqual(len(merged["improvements"]), 2)
        self.assertEqual(len(merged["recommended_exercises"]), 2)

    def test_differentiate_kendoka_names(self):
        """Valida a adição de sufixos numéricos sequenciais quando o usuário opta por não mesclar."""
        kendokas = [
            {"kendoka_id": "KENSHI_01", "custom_name": "Eduardo"},
            {"kendoka_id": "KENSHI_05", "custom_name": "Eduardo"}
        ]
        n1, n2 = differentiate_kendoka_names(kendokas, "KENSHI_01", "KENSHI_05", "Eduardo")
        self.assertEqual(n1, "Eduardo (1)")
        self.assertEqual(n2, "Eduardo (2)")

        # Caso já existam praticantes com (1) e (2)
        kendokas_3 = [
            {"kendoka_id": "KENSHI_01", "custom_name": "Eduardo (1)"},
            {"kendoka_id": "KENSHI_02", "custom_name": "Eduardo (2)"},
            {"kendoka_id": "KENSHI_05", "custom_name": "Eduardo"}
        ]
        n1_c, n3 = differentiate_kendoka_names(kendokas_3, "KENSHI_01", "KENSHI_05", "Eduardo")
        self.assertEqual(n1_c, "Eduardo (1)")
        self.assertEqual(n3, "Eduardo (3)")

    def test_no_false_collision_when_kendokas_sorted_by_presence(self):
        """
        Regressão: Quando os praticantes estão ordenados por tempo de presença (ex.: KENSHI_01 no idx 0,
        KENSHI_03 no idx 1, KENSHI_04 no idx 2), sync_kendoka_names_in_training_data NÃO deve
        sobrescrever o nome de KENSHI_04 (idx 2 -> idx+1=3) com o nome de KENSHI_03.
        """
        train_data = {
            "kendokas": [
                {"kendoka_id": "KENSHI_01", "default_name": "Kenshi 01", "custom_name": "Kenshi 01"},
                {"kendoka_id": "KENSHI_03", "default_name": "Kenshi 03", "custom_name": "Kenshi 03"},
                {"kendoka_id": "KENSHI_04", "default_name": "Kenshi 04", "custom_name": "Kenshi 04"},
            ],
            "sub_sessions": [],
            "modality_blocks": [],
        }
        custom_map = {}
        sync_kendoka_names_in_training_data(train_data, custom_map)

        self.assertEqual(train_data["kendokas"][0]["custom_name"], "Kenshi 01")
        self.assertEqual(train_data["kendokas"][1]["custom_name"], "Kenshi 03")
        self.assertEqual(train_data["kendokas"][2]["custom_name"], "Kenshi 04")
        self.assertEqual(custom_map.get("KENSHI_03"), "Kenshi 03")
        self.assertEqual(custom_map.get("KENSHI_04"), "Kenshi 04")


if __name__ == "__main__":
    unittest.main()


