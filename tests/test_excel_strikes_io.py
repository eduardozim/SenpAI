"""
Testes Automatizados para Exportação e Importação de Golpes via Excel (.xlsx) e Treinamento Adaptativo.
"""

import io
import os
import unittest
import pandas as pd
import openpyxl

from src.utils.excel_strikes_manager import (
    export_strikes_to_excel,
    import_strikes_from_excel,
    apply_imported_strikes_to_session_reviews,
    execute_training_from_imported_strikes
)
from src.engine.feedback_manager import FeedbackManager


class TestExcelStrikesIO(unittest.TestCase):
    def setUp(self):
        self.test_dataset = "data/test_excel_feedback.json"
        self.test_history = "data/test_excel_history.json"
        self._cleanup_test_files()
        self.feedback_mgr = FeedbackManager(
            dataset_path=self.test_dataset,
            history_path=self.test_history
        )

    def tearDown(self):
        self._cleanup_test_files()

    def _cleanup_test_files(self):
        for p in [self.test_dataset, self.test_history]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def _get_sample_strikes(self):
        return [
            {
                "event_id": "event_1_frame_60",
                "source": "AI_DETECTED",
                "timestamp": "00:02.000",
                "time_sec": 2.0,
                "orig_is_valid": True,
                "attacker_id": "KENSHI_AKA",
                "attacker_label": "Kenshi Aka (Vermelho)",
                "review": {
                    "event_id": "event_1_frame_60",
                    "strike_type": "MEN",
                    "timestamp": "00:02.000",
                    "attacker_id": "KENSHI_AKA",
                    "attacker_name": "Kenshi Aka (Vermelho)",
                    "is_valid_ippon": True,
                    "category": "VALID_IPPON",
                    "label": "TP",
                    "total_score": 82.5,
                    "sub_scores": {
                        "target_impact": 85.0,
                        "fumikomi_sync": 80.0,
                        "posture": 80.0,
                        "zanshin": 85.0
                    },
                    "notes": "Golpe limpo com bom fumikomi"
                },
                "raw_event": {
                    "evaluation": {
                        "is_valid": True,
                        "total_score": 82.5,
                        "sub_scores": {
                            "target_impact": 85.0,
                            "fumikomi_sync": 80.0,
                            "posture": 80.0,
                            "zanshin": 85.0
                        }
                    }
                }
            },
            {
                "event_id": "event_2_frame_150",
                "source": "AI_DETECTED",
                "timestamp": "00:05.000",
                "time_sec": 5.0,
                "orig_is_valid": False,
                "attacker_id": "KENSHI_SHIRO",
                "attacker_label": "Kenshi Shiro (Branco)",
                "review": {
                    "event_id": "event_2_frame_150",
                    "strike_type": "KOTE",
                    "timestamp": "00:05.000",
                    "attacker_id": "KENSHI_SHIRO",
                    "attacker_name": "Kenshi Shiro (Branco)",
                    "is_valid_ippon": False,
                    "category": "INVALID_HIT",
                    "label": "FP",
                    "total_score": 55.0,
                    "sub_scores": {
                        "target_impact": 50.0,
                        "fumikomi_sync": 60.0,
                        "posture": 60.0,
                        "zanshin": 50.0
                    },
                    "notes": "Fumikomi atrasado"
                },
                "raw_event": {
                    "evaluation": {
                        "is_valid": False,
                        "total_score": 55.0,
                        "sub_scores": {
                            "target_impact": 50.0,
                            "fumikomi_sync": 60.0,
                            "posture": 60.0,
                            "zanshin": 50.0
                        }
                    }
                }
            }
        ]

    def test_export_strikes_to_excel_structure(self):
        """Valida que a exportação gera um arquivo .xlsx válido com abas corretas, metadados e cabeçalhos adequados."""
        sample_strikes = self._get_sample_strikes()
        excel_bytes = export_strikes_to_excel(
            sample_strikes,
            video_name="shiai_test.mp4",
            streaming_url="https://www.youtube.com/watch?v=sample123",
            reviewer_dan=5
        )

        self.assertIsInstance(excel_bytes, bytes)
        self.assertGreater(len(excel_bytes), 1000)

        # Carregar via openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        sheet_names = wb.sheetnames
        self.assertIn("Golpes Detectados", sheet_names)
        self.assertIn("Metadados & Vídeo", sheet_names)
        self.assertIn("Instruções & Dicionário", sheet_names)

        # Ler via pandas
        df_strikes = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Golpes Detectados")
        self.assertEqual(len(df_strikes), 2)
        self.assertIn("ID_Evento", df_strikes.columns)
        self.assertIn("Golpe", df_strikes.columns)
        self.assertIn("Ippon_Valido", df_strikes.columns)
        self.assertIn("Dan_Revisor", df_strikes.columns)
        self.assertIn("Link_Streaming", df_strikes.columns)
        self.assertEqual(df_strikes.iloc[0]["Golpe"], "MEN")
        self.assertEqual(df_strikes.iloc[0]["Ippon_Valido"], "SIM")
        self.assertEqual(df_strikes.iloc[1]["Golpe"], "KOTE")
        self.assertEqual(df_strikes.iloc[1]["Ippon_Valido"], "NÃO")
        self.assertEqual(df_strikes.iloc[0]["Dan_Revisor"], "5º Dan (Godan)")
        self.assertEqual(df_strikes.iloc[0]["Link_Streaming"], "https://www.youtube.com/watch?v=sample123")

        # Verificar aba de metadados
        df_meta = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Metadados & Vídeo")
        self.assertGreater(len(df_meta), 4)
        meta_dict = dict(zip(df_meta["Metadado"], df_meta["Valor"]))
        self.assertEqual(meta_dict.get("Link de Streaming"), "https://www.youtube.com/watch?v=sample123")
        self.assertEqual(meta_dict.get("Graduação Dan do Revisor"), "5º Dan (Godan)")

    def test_import_strikes_with_edits(self):
        """Valida que modificações na planilha (ex: correção de golpe e status de Ippon) são importadas com precisão."""
        sample_strikes = self._get_sample_strikes()
        excel_bytes = export_strikes_to_excel(sample_strikes)

        # Simular edição do usuário na planilha
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Golpes Detectados")
        # Altera golpe 2 de KOTE para DO e valida como SIM (Ippon Válido)
        df.loc[1, "Golpe"] = "DO"
        df.loc[1, "Ippon_Valido"] = "SIM"
        df.loc[1, "Observacoes"] = "Correção pelo árbitro: Do perfeito"

        edited_bytes = io.BytesIO()
        with pd.ExcelWriter(edited_bytes, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Golpes Detectados", index=False)
        edited_bytes.seek(0)

        imported_items, summary = import_strikes_from_excel(edited_bytes)

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["total_rows"], 2)
        self.assertEqual(summary["valid_ippons"], 2)
        self.assertEqual(summary["invalid_hits"], 0)

        # O segundo golpe deve ser DO e ter is_valid_ippon=True
        self.assertEqual(imported_items[1]["strike_type"], "DO")
        self.assertTrue(imported_items[1]["is_valid_ippon"])
        self.assertEqual(imported_items[1]["label"], "TP")
        self.assertEqual(imported_items[1]["notes"], "Correção pelo árbitro: Do perfeito")

    def test_import_strikes_with_new_included_strike(self):
        """Valida que adicionar uma nova linha sem ID_Evento na planilha é identificado como novo golpe incluído."""
        sample_strikes = self._get_sample_strikes()
        excel_bytes = export_strikes_to_excel(sample_strikes)

        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Golpes Detectados")
        # Adicionar uma terceira linha (golpe perdido não detectado pela IA)
        new_row = {
            "ID_Evento": "", # em branco
            "Timestamp": "00:08.300",
            "Tempo_Segundos": 8.3,
            "Golpe": "TSUKI",
            "Lutador_ID": "KENSHI_AKA",
            "Lutador_Nome": "Kenshi Aka (Vermelho)",
            "Ippon_Valido": "SIM",
            "Categoria_Decisao": "VALID_IPPON",
            "Rotulo_Avaliacao": "TP",
            "Pontuacao_Total": 90.0,
            "Impacto_Alvo": 92.0,
            "Fumikomi_Sync": 88.0,
            "Postura": 90.0,
            "Zanshin": 90.0,
            "Origem": "INCLUIDO_MANUAL",
            "Observacoes": "Tsuki certeiro incluído manualmente"
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        edited_bytes = io.BytesIO()
        with pd.ExcelWriter(edited_bytes, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Golpes Detectados", index=False)
        edited_bytes.seek(0)

        imported_items, summary = import_strikes_from_excel(
            edited_bytes,
            existing_events=sample_strikes
        )

        self.assertEqual(summary["total_rows"], 3)
        self.assertEqual(summary["new_strikes_count"], 1)
        self.assertEqual(summary["valid_ippons"], 2)

        # O terceiro item deve ter id gerado e is_included=True
        third_item = imported_items[2]
        self.assertTrue(third_item["is_included"])
        self.assertEqual(third_item["strike_type"], "TSUKI")
        self.assertEqual(third_item["timestamp"], "00:08.300")

    def test_apply_imported_strikes_to_session_reviews(self):
        """Valida que a atualização do session_reviews reflete os dados importados mantendo consistência."""
        sample_strikes = self._get_sample_strikes()
        excel_bytes = export_strikes_to_excel(sample_strikes)
        imported_items, _ = import_strikes_from_excel(io.BytesIO(excel_bytes))

        session_revs = {}
        updated_revs = apply_imported_strikes_to_session_reviews(imported_items, session_revs)

        self.assertEqual(len(updated_revs), 2)
        self.assertIn("event_1_frame_60", updated_revs)
        self.assertTrue(updated_revs["event_1_frame_60"]["is_confirmed"])
        self.assertTrue(updated_revs["event_1_frame_60"]["is_edited"])
        self.assertEqual(updated_revs["event_1_frame_60"]["strike_type"], "MEN")

    def test_execute_training_from_imported_strikes(self):
        """Valida o retreinamento do modelo com governança por Dan a partir de dados importados do Excel."""
        sample_strikes = self._get_sample_strikes()
        excel_bytes = export_strikes_to_excel(sample_strikes)

        # Modifica o golpe 1 para FP (Falso Positivo) com pontuação alta para forçar adaptação de corte
        df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name="Golpes Detectados")
        df.loc[0, "Ippon_Valido"] = "NÃO"
        df.loc[0, "Rotulo_Avaliacao"] = "FP"
        df.loc[0, "Pontuacao_Total"] = 72.0

        edited_bytes = io.BytesIO()
        with pd.ExcelWriter(edited_bytes, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Golpes Detectados", index=False)
        edited_bytes.seek(0)

        imported_items, _ = import_strikes_from_excel(edited_bytes)

        base_cfg = {
            "name": "Treino Geral / Keiko (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        train_res = execute_training_from_imported_strikes(
            imported_strikes=imported_items,
            video_name="test_match.mp4",
            profile_key="normal",
            reviewer_dan=6, # 6º Dan (Rokudan)
            current_profile_config=base_cfg,
            feedback_mgr=self.feedback_mgr
        )

        self.assertEqual(train_res["status"], "success")
        self.assertEqual(train_res["items_count"], 2)
        self.assertEqual(train_res["reviewer_dan"], 6)
        self.assertEqual(train_res["reviewer_dan_name"], "6º Dan (Rokudan)")

        # Valida que o feedback foi persistido
        fb_saved = self.feedback_mgr.load_feedback()
        self.assertEqual(len(fb_saved), 2)

    def test_export_and_import_with_streaming_url_and_reviewer_dan(self):
        """Valida que o link de streaming e a graduação Dan são preservados na exportação e extraídos na importação."""
        sample_strikes = self._get_sample_strikes()
        test_url = "https://www.youtube.com/watch?v=kendo_finals_2026"
        test_dan = 7 # 7º Dan (Nanadan)

        excel_bytes = export_strikes_to_excel(
            sample_strikes,
            video_name="all_japan_championship.mp4",
            streaming_url=test_url,
            reviewer_dan=test_dan
        )

        imported_items, summary = import_strikes_from_excel(io.BytesIO(excel_bytes))

        self.assertEqual(summary["status"], "success")
        self.assertEqual(summary["streaming_url"], test_url)
        self.assertEqual(summary["reviewer_dan"], 7)
        self.assertEqual(summary["reviewer_dan_name"], "7º Dan (Nanadan)")

        # Checar que os itens individuais herdaram os metadados
        self.assertEqual(len(imported_items), 2)
        for itm in imported_items:
            self.assertEqual(itm["dan_revisor"], 7)
            self.assertEqual(itm["streaming_url"], test_url)

    def test_export_and_import_with_shinpan_reviewer(self):
        """Valida que a Decisão dos Shinpans é preservada no Excel e executa retreinamento com peso calibrado 4.5."""
        sample_strikes = self._get_sample_strikes()
        excel_bytes = export_strikes_to_excel(
            sample_strikes,
            video_name="shiai_tokyo.mp4",
            reviewer_dan="shinpan"
        )

        imported_items, summary = import_strikes_from_excel(io.BytesIO(excel_bytes))
        self.assertEqual(summary["reviewer_dan"], "shinpan")
        self.assertEqual(summary["reviewer_dan_name"], "Decisão dos Shinpans")

        base_cfg = {
            "name": "Shiai",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }
        train_res = execute_training_from_imported_strikes(
            imported_strikes=imported_items,
            video_name="shiai_tokyo.mp4",
            profile_key="shiai",
            reviewer_dan="shinpan",
            current_profile_config=base_cfg,
            feedback_mgr=self.feedback_mgr
        )
        self.assertEqual(train_res["status"], "success")
        self.assertEqual(train_res["reviewer_dan"], "shinpan")
        self.assertEqual(train_res["reviewer_dan_name"], "Decisão dos Shinpans")
        self.assertTrue(train_res.get("is_shinpan_decision"))

        metrics = self.feedback_mgr.get_training_metrics()
        self.assertEqual(metrics["shinpan_trainings_count"], 1)
        self.assertEqual(metrics["average_dan_level"], 0.0)


if __name__ == "__main__":
    unittest.main()
