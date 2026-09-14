"""
Testes Automatizados para Edição por Dan, Retreinamento e Governança de Treinamento no SenpAI.
"""

import os
import json
import unittest
from src.engine.feedback_manager import (
    FeedbackManager,
    DAN_NAMES,
    DuplicateShinpanReviewError,
    normalize_video_identifier
)

class TestDanTrainingGovernance(unittest.TestCase):
    def setUp(self):
        self.test_dataset_path = "data/test_feedback_dataset.json"
        self.test_history_path = "data/test_training_history.json"
        self.test_profiles_path = "config/test_calibration_profiles.json"
        self.test_shinpan_registry_path = "data/test_shinpan_reviewed_videos.json"

        for p in [self.test_dataset_path, self.test_history_path, self.test_profiles_path, self.test_shinpan_registry_path]:
            if os.path.exists(p):
                os.remove(p)

        self.mgr = FeedbackManager(
            dataset_path=self.test_dataset_path,
            history_path=self.test_history_path,
            profiles_path=self.test_profiles_path,
            shinpan_registry_path=self.test_shinpan_registry_path
        )

    def tearDown(self):
        for p in [self.test_dataset_path, self.test_history_path, self.test_profiles_path, self.test_shinpan_registry_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_save_review_session_with_dan(self):
        """Valida o salvamento de sessão de revisão com Dan, calculando métricas e histórico de auditoria."""
        items = [
            {
                "event_id": "ev_1",
                "label": "TP",
                "strike_type": "MEN",
                "timestamp": "00:01.200",
                "total_score": 82.0,
                "sub_scores": {"target_impact": 85.0, "fumikomi_sync": 80.0, "posture": 80.0, "zanshin": 80.0},
                "is_confirmed": True
            },
            {
                "event_id": "ev_2",
                "label": "FP",
                "strike_type": "KOTE",
                "timestamp": "00:04.500",
                "total_score": 68.0,
                "sub_scores": {"target_impact": 60.0, "fumikomi_sync": 50.0, "posture": 50.0, "zanshin": 45.0},
                "is_edited": True,
                "notes": "Alvo incorreto"
            }
        ]

        current_config = {
            "name": "Treino Geral (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        new_config, record = self.mgr.save_review_session(
            video_name="match_kendo.mp4",
            profile_key="normal",
            reviewer_dan=4,  # Yondan (4º Dan)
            review_items=items,
            current_profile_config=current_config
        )

        self.assertEqual(record["reviewer_dan"], 4)
        self.assertEqual(record["reviewer_dan_name"], "4º Dan (Yondan)")
        self.assertEqual(record["items_count"], 2)

        data = self.mgr.load_feedback()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["reviewer_dan"], 4)
        self.assertEqual(data[0]["reviewer_dan_name"], "4º Dan (Yondan)")
        self.assertIn("review_date", data[0])

    def test_get_training_metrics_dan_distribution(self):
        """Verifica o cálculo de métricas de governança: total de sessões, nível médio Dan e distribuição 1º ao 8º Dan."""
        current_config = {
            "name": "Normal",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        # Sessão 1: 3º Dan (Sandan)
        self.mgr.save_review_session("m1.mp4", "normal", 3, [{"event_id": "e1", "label": "TP", "strike_type": "MEN"}], current_config)
        # Sessão 2: 5º Dan (Godan)
        self.mgr.save_review_session("m2.mp4", "normal", 5, [{"event_id": "e2", "label": "TP", "strike_type": "KOTE"}], current_config)

        metrics = self.mgr.get_training_metrics()

        self.assertEqual(metrics["total_trainings_count"], 2)
        self.assertEqual(metrics["human_trainings_count"], 2)
        self.assertEqual(metrics["auto_trainings_count"], 0)
        self.assertEqual(metrics["average_dan_level"], 4.0)  # (3 + 5) / 2 = 4.0
        self.assertIn("4.0º Dan", metrics["average_dan_label"])

        dan_dist = metrics["dan_distribution"]
        self.assertEqual(len(dan_dist), 10)  # 1º ao 8º Dan + ⚖️ Shinpans + 🤖 IA
        
        dan3_entry = next(item for item in dan_dist if item["Dan"] == "3º Dan")
        dan5_entry = next(item for item in dan_dist if item["Dan"] == "5º Dan")
        shinpan_entry = next(item for item in dan_dist if "Shinpans" in item["Dan"])
        auto_entry = next(item for item in dan_dist if "IA" in item["Dan"])
        
        self.assertEqual(dan3_entry["Quantidade Treinamentos"], 1)
        self.assertEqual(dan5_entry["Quantidade Treinamentos"], 1)
        self.assertEqual(shinpan_entry["Quantidade Treinamentos"], 0)
        self.assertEqual(auto_entry["Quantidade Treinamentos"], 0)

        # Adicionar uma sessão de treinamento automático de IA e verificar que não altera o Dan médio humano
        auto_session = {
            "id": "auto_train_test_1",
            "timestamp": "2026-08-30T12:00:00",
            "video_name": "AI_Auto_Trainer_general_all",
            "reviewer_dan": 0,
            "reviewer_dan_name": "Treinamento Automático por IA (Web & Vídeo)",
            "is_auto_training": True,
            "optimization_summary": {"mode": "auto_training_ai"}
        }
        hist = self.mgr.load_history()
        hist.append(auto_session)
        with open(self.test_history_path, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2, ensure_ascii=False)

        metrics_with_auto = self.mgr.get_training_metrics()
        self.assertEqual(metrics_with_auto["total_trainings_count"], 3)
        self.assertEqual(metrics_with_auto["human_trainings_count"], 2)
        self.assertEqual(metrics_with_auto["auto_trainings_count"], 1)
        self.assertEqual(metrics_with_auto["average_dan_level"], 4.0)  # Continua 4.0, não poluído pela IA
        auto_entry_post = next(item for item in metrics_with_auto["dan_distribution"] if "IA" in item["Dan"])
        self.assertEqual(auto_entry_post["Quantidade Treinamentos"], 1)

    def test_export_and_import_training_package(self):
        """Testa a exportação e importação de pacotes de treinamento JSON com preservação de datas e graduação Dan."""
        current_config = {
            "name": "Normal",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        self.mgr.save_review_session("m1.mp4", "normal", 6, [{"event_id": "e1", "label": "TP", "strike_type": "MEN", "timestamp": "00:01.000"}], current_config)

        pkg = self.mgr.export_training_package()
        self.assertEqual(pkg["system_name"], "SenpAI")
        self.assertIn("exported_at", pkg)
        self.assertEqual(len(pkg["review_items"]), 1)
        self.assertEqual(pkg["review_items"][0]["reviewer_dan"], 6)

        # Resetar gerenciador e importar o pacote
        self.mgr.reset_all_training_data()
        self.assertEqual(self.mgr.get_training_metrics()["total_trainings_count"], 0)

        import_res = self.mgr.import_training_package(pkg)
        self.assertEqual(import_res["status"], "success")
        self.assertEqual(import_res["imported_items_count"], 1)

        metrics_post = self.mgr.get_training_metrics()
        self.assertEqual(metrics_post["total_trainings_count"], 1)
        self.assertEqual(metrics_post["average_dan_level"], 6.0)

    def test_reset_all_training_data(self):
        """Valida o reset completo do histórico de treinamento e dataset restaurando o sistema ao estágio inicial."""
        self.mgr.save_feedback("v.mp4", "normal", "ev1", "TP", strike_type="MEN", reviewer_dan=2)
        self.mgr.reset_all_training_data()

        self.assertEqual(len(self.mgr.load_feedback()), 0)
        self.assertEqual(len(self.mgr.load_history()), 0)

    def test_import_raw_list_json(self):
        """Valida a importação de listas JSON brutas de feedbacks e recalibração automática do Dan médio."""
        raw_list = [
            {
                "id": "item_1",
                "video_name": "match_test.mp4",
                "profile_key": "normal",
                "label": "TP",
                "strike_type": "MEN",
                "reviewer_dan": 5
            }
        ]
        res = self.mgr.import_training_package(raw_list)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["new_items_added"], 1)
        self.assertEqual(self.mgr.get_training_metrics()["total_trainings_count"], 1)
        self.assertEqual(self.mgr.get_training_metrics()["average_dan_level"], 5.0)

    def test_get_training_storage_info(self):
        """Valida o cálculo do espaço em disco ocupado pelos dados de treinamento, modelos e configurações."""
        # Salvar um item para garantir que os arquivos tenham conteúdo
        self.mgr.save_feedback("match_storage.mp4", "normal", "ev_st", "TP", strike_type="MEN", reviewer_dan=4)

        storage_info = self.mgr.get_training_storage_info()

        self.assertIn("total_bytes", storage_info)
        self.assertIn("total_formatted", storage_info)
        self.assertIn("categories", storage_info)
        self.assertIn("files", storage_info)

        self.assertGreater(storage_info["total_bytes"], 0)
        self.assertTrue(any(unit in storage_info["total_formatted"] for unit in ["B", "KB", "MB", "GB"]))

        categories = storage_info["categories"]
        self.assertIn("datasets", categories)
        self.assertIn("models", categories)
        self.assertIn("knowledge_config", categories)

        self.assertGreater(categories["datasets"]["bytes"], 0)
        self.assertIn("B", categories["datasets"]["formatted"])

        # Verificar se as métricas de governança também incluem storage_info
        metrics = self.mgr.get_training_metrics()
        self.assertIn("storage_info", metrics)
        self.assertEqual(metrics["storage_info"]["total_bytes"], storage_info["total_bytes"])

    def test_shinpan_review_session_and_governance_metrics(self):
        """Valida que a Decisão dos Shinpans é tratada à parte da graduação Dan, com peso constante 4.5 e métricas dedicadas."""
        current_config = {
            "name": "Normal",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        # 1. Salvar sessão por Decisão dos Shinpans
        shinpan_items = [
            {
                "event_id": "sh_1",
                "label": "TP",
                "category": "VALID_IPPON",
                "decision_category": "VALID_IPPON",
                "strike_type": "MEN",
                "timestamp": "00:02.100",
                "total_score": 85.0,
                "sub_scores": {"target_impact": 90.0, "fumikomi_sync": 80.0, "posture": 80.0, "zanshin": 75.0},
                "is_confirmed": False,
                "is_edited": True
            }
        ]
        new_cfg, record = self.mgr.save_review_session(
            video_name="shiai_finals.mp4",
            profile_key="normal",
            reviewer_dan="shinpan",
            review_items=shinpan_items,
            current_profile_config=current_config
        )

        self.assertEqual(record["reviewer_dan"], "shinpan")
        self.assertEqual(record["reviewer_dan_name"], "Decisão dos Shinpans")
        self.assertTrue(record.get("is_shinpan_decision"))
        self.assertAlmostEqual(sum(new_cfg["weights"].values()), 1.0, places=3)

        # 2. Métricas de governança com apenas Shinpans
        m1 = self.mgr.get_training_metrics()
        self.assertEqual(m1["total_trainings_count"], 1)
        self.assertEqual(m1["shinpan_trainings_count"], 1)
        self.assertEqual(m1["human_trainings_count"], 0)
        self.assertEqual(m1["average_dan_level"], 0.0)  # Shinpans tratados à parte, Dan médio não poluído

        # 3. Adicionar sessão de treinador 4º Dan (Yondan)
        coach_items = [
            {
                "event_id": "co_1",
                "label": "TP",
                "category": "VALID_IPPON",
                "decision_category": "VALID_IPPON",
                "strike_type": "KOTE",
                "timestamp": "00:03.500",
                "total_score": 80.0,
                "sub_scores": {"target_impact": 80.0, "fumikomi_sync": 80.0, "posture": 80.0, "zanshin": 80.0}
            }
        ]
        self.mgr.save_review_session(
            video_name="shiai_finals.mp4",
            profile_key="normal",
            reviewer_dan=4,
            review_items=coach_items,
            current_profile_config=new_cfg
        )

        # 4. Métricas consolidadas: Shinpans + 4º Dan
        m2 = self.mgr.get_training_metrics()
        self.assertEqual(m2["total_trainings_count"], 2)
        self.assertEqual(m2["shinpan_trainings_count"], 1)
        self.assertEqual(m2["human_trainings_count"], 1)
        self.assertEqual(m2["average_dan_level"], 4.0)  # Permanece puramente 4.0

        # Verificar tabela de distribuição
        dan_dist = m2["dan_distribution"]
        shinpan_row = next(r for r in dan_dist if "Shinpans" in r["Dan"])
        dan4_row = next(r for r in dan_dist if r["Dan"] == "4º Dan")
        self.assertEqual(shinpan_row["Quantidade Treinamentos"], 1)
        self.assertEqual(dan4_row["Quantidade Treinamentos"], 1)

        # 5. Exportação e Importação de pacote mantendo flag Shinpan
        pkg = self.mgr.export_training_package()
        self.mgr.reset_all_training_data()
        self.assertEqual(self.mgr.get_training_metrics()["total_trainings_count"], 0)

        imp_res = self.mgr.import_training_package(pkg)
        self.assertEqual(imp_res["status"], "success")
        m3 = self.mgr.get_training_metrics()
        self.assertEqual(m3["total_trainings_count"], 2)
        self.assertEqual(m3["shinpan_trainings_count"], 1)
        self.assertEqual(m3["average_dan_level"], 4.0)

    def test_shinpan_ui_state_transitions(self):
        """Valida as 5 regras de transição de estado da UI e listagem de golpes:
        1. Habilitar edição desmarcado: lista todos os golpes de IA.
        2. Habilitar edição marcado e Dan selecionado (não Shinpan): lista golpes de IA.
        3. Habilitar edição marcado e Decisão dos Shinpans selecionado: lista limpa apenas com Ippons dos Shinpans.
        4. Habilitar edição desmarcado depois de Shinpans selecionado: volta a listar golpes de IA.
        5. Selecionar um Dan após Shinpans selecionado: volta a listar golpes de IA.
        """
        ai_events = [
            {"event_info": {"type": "MEN", "timestamp": "00:02.100", "impact_frame": 63}, "evaluation": {"is_valid": True}},
            {"event_info": {"type": "KOTE", "timestamp": "00:05.400", "impact_frame": 162}, "evaluation": {"is_valid": False}},
            {"event_info": {"type": "DO", "timestamp": "00:08.800", "impact_frame": 264}, "evaluation": {"is_valid": True}}
        ]

        session_revs = {
            "shinpan_1": {
                "event_id": "shinpan_1",
                "label": "TP",
                "category": "VALID_IPPON",
                "is_valid_ippon": True,
                "reviewer_dan": "shinpan",
                "timestamp": "00:02.100",
                "strike_type": "MEN"
            },
            "dan_inc_1": {
                "event_id": "dan_inc_1",
                "label": "TP",
                "category": "VALID_IPPON",
                "is_valid_ippon": True,
                "is_included": True,
                "reviewer_dan": 4,
                "timestamp": "00:04.000",
                "strike_type": "MEN"
            }
        }

        def build_strikes(enable_editing, selected_dan):
            combined = []
            if enable_editing and selected_dan == "shinpan":
                for fn_k, fn_v in session_revs.items():
                    is_fn_ippon = fn_v.get("is_valid_ippon", fn_v.get("category") == "VALID_IPPON" or fn_v.get("label") == "TP")
                    if is_fn_ippon and fn_v.get("reviewer_dan") == "shinpan":
                        combined.append({"event_id": fn_k, "source": "SHINPAN_IPPON"})
            else:
                for idx_raw, ev_data in enumerate(ai_events):
                    ev = ev_data["event_info"]
                    event_id_str = f"event_{idx_raw+1}_frame_{ev['impact_frame']}"
                    combined.append({"event_id": event_id_str, "source": "AI_DETECTED"})
                if enable_editing:
                    for fn_k, fn_v in session_revs.items():
                        if fn_v.get("is_included") and fn_v.get("reviewer_dan") != "shinpan":
                            combined.append({"event_id": fn_k, "source": "INCLUDED"})
            return combined

        # Caso 1: Habilitar edição desmarcado -> Lista completa de golpes identificados pela IA (3 golpes)
        c1 = build_strikes(enable_editing=False, selected_dan=3)
        self.assertEqual(len(c1), 3)
        self.assertTrue(all(s["source"] == "AI_DETECTED" for s in c1))

        # Caso 2: Habilitar edição marcado e Dan selecionado (ex: 4º Dan) -> Golpes IA + inclusão Dan (4 golpes)
        c2 = build_strikes(enable_editing=True, selected_dan=4)
        self.assertEqual(len(c2), 4)
        self.assertEqual(len([s for s in c2 if s["source"] == "AI_DETECTED"]), 3)
        self.assertEqual(len([s for s in c2 if s["source"] == "INCLUDED"]), 1)

        # Caso 3: Habilitar edição marcado e Decisão dos Shinpans selecionado -> Apenas Ippons dos Shinpans (1 golpe)
        c3 = build_strikes(enable_editing=True, selected_dan="shinpan")
        self.assertEqual(len(c3), 1)
        self.assertEqual(c3[0]["source"], "SHINPAN_IPPON")
        self.assertEqual(c3[0]["event_id"], "shinpan_1")

        # Caso 3b: Decisão dos Shinpans sem nenhum Ippon cadastrado -> Lista completamente limpa (0 golpes)
        old_revs = session_revs.copy()
        session_revs.clear()
        c3b = build_strikes(enable_editing=True, selected_dan="shinpan")
        self.assertEqual(len(c3b), 0)
        session_revs.update(old_revs)

        # Caso 4: Habilitar edição desmarcado depois de Shinpans selecionado (selected_dan ainda "shinpan" no state)
        # -> Volta a apresentar todos os golpes identificados pela IA (3 golpes)
        c4 = build_strikes(enable_editing=False, selected_dan="shinpan")
        self.assertEqual(len(c4), 3)
        self.assertTrue(all(s["source"] == "AI_DETECTED" for s in c4))

        # Caso 5: Selecionar um Dan (ex: 3º Dan) após Decisão dos Shinpans -> Volta a apresentar golpes identificados pela IA (4 golpes com inclusão do Dan)
        c5 = build_strikes(enable_editing=True, selected_dan=3)
        self.assertEqual(len(c5), 4)
        self.assertEqual(len([s for s in c5 if s["source"] == "AI_DETECTED"]), 3)
        self.assertFalse(any(s["source"] == "SHINPAN_IPPON" for s in c5))

    def test_normalize_video_identifier(self):
        """Valida que URLs de diferentes formatos do YouTube e nomes de arquivos normalizam para o mesmo ID canônico."""
        # Formatos YouTube
        self.assertEqual(normalize_video_identifier("https://www.youtube.com/watch?v=kendo_match_123"), "youtube:kendo_match_123")
        self.assertEqual(normalize_video_identifier("https://youtu.be/kendo_match_123"), "youtube:kendo_match_123")
        self.assertEqual(normalize_video_identifier("https://www.youtube.com/shorts/kendo_match_123?feature=share"), "youtube:kendo_match_123")
        self.assertEqual(normalize_video_identifier("https://www.youtube.com/live/kendo_match_123"), "youtube:kendo_match_123")
        self.assertEqual(normalize_video_identifier("https://www.youtube.com/embed/kendo_match_123"), "youtube:kendo_match_123")

        # URL genérica HTTP/HTTPS
        self.assertEqual(normalize_video_identifier("https://example.com/videos/shiai_final.mp4?token=abc"), "url:https://example.com/videos/shiai_final.mp4")

        # Upload local (remoção de prefixo temporário 'upload_timestamp_')
        self.assertEqual(normalize_video_identifier("upload_1740000000_shiai_tokyo.mp4"), "file:shiai_tokyo.mp4")
        self.assertEqual(normalize_video_identifier("shiai_tokyo.mp4"), "file:shiai_tokyo.mp4")
        self.assertEqual(normalize_video_identifier("C:/temp/uploads/upload_9999_final.mp4"), "file:final.mp4")

        # Entradas vazias
        self.assertEqual(normalize_video_identifier(None), "")
        self.assertEqual(normalize_video_identifier(""), "")

    def test_shinpan_video_link_registration_and_blocking_duplicates(self):
        """Valida que uma Decisão dos Shinpans registra o link do vídeo e bloqueia uma 2ª entrada com o mesmo link."""
        video_url = "https://www.youtube.com/watch?v=kendo_championship_final_2024"
        video_name = "Final Masculina Kendo 2024"
        items = [
            {
                "event_id": "sh_1",
                "label": "TP",
                "strike_type": "MEN",
                "timestamp": "01:23.450",
                "total_score": 100.0,
                "is_valid_ippon": True,
                "reviewer_dan": "shinpan"
            }
        ]
        cfg = {
            "name": "Treino Geral (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        # Antes de salvar: vídeo não consta como registrado
        is_reviewed, _ = self.mgr.is_video_reviewed_by_shinpan(video_url=video_url, video_name=video_name)
        self.assertFalse(is_reviewed)

        # 1ª Entrada: Decisão dos Shinpans deve ser salva com sucesso
        new_cfg, rec = self.mgr.save_review_session(
            video_name=video_name,
            profile_key="normal",
            reviewer_dan="shinpan",
            review_items=items,
            current_profile_config=cfg,
            video_url=video_url
        )
        self.assertEqual(rec["reviewer_dan"], "shinpan")
        self.assertEqual(rec["items_count"], 1)

        # Após salvar: vídeo deve estar registrado na lista de vídeos dos Shinpans
        is_reviewed_after, reg_info = self.mgr.is_video_reviewed_by_shinpan(video_url=video_url, video_name=video_name)
        self.assertTrue(is_reviewed_after)
        self.assertIsNotNone(reg_info)
        self.assertEqual(reg_info["video_identifier"], "youtube:kendo_championship_final_2024")

        # 2ª Entrada com a mesma URL completa: DEVE ser BLOQUEADA (levantar DuplicateShinpanReviewError)
        with self.assertRaises(DuplicateShinpanReviewError) as ctx_exact:
            self.mgr.save_review_session(
                video_name=video_name,
                profile_key="normal",
                reviewer_dan="shinpan",
                review_items=items,
                current_profile_config=new_cfg,
                video_url=video_url
            )
        self.assertIn("Entrada duplicada bloqueada", str(ctx_exact.exception))

        # 2ª Entrada com link encurtado (youtu.be) apontando para o mesmo vídeo: TAMBÉM DEVE ser BLOQUEADA!
        alternate_url = "https://youtu.be/kendo_championship_final_2024"
        with self.assertRaises(DuplicateShinpanReviewError) as ctx_alt:
            self.mgr.save_review_session(
                video_name="Outro Titulo Mas Mesmo Video",
                profile_key="normal",
                reviewer_dan="shinpan",
                review_items=items,
                current_profile_config=new_cfg,
                video_url=alternate_url
            )
        self.assertIn("Entrada duplicada bloqueada", str(ctx_alt.exception))

    def test_dan_review_unrestricted_duplicate_video_links(self):
        """Valida que revisões por DAN (1º ao 8º Dan) NÃO possuem restrição para entradas duplicadas com o mesmo link."""
        video_url = "https://www.youtube.com/watch?v=training_session_dan_study"
        video_name = "Estudo Técnico de Ippon"
        cfg = {
            "name": "Treino Geral (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }
        items_1 = [{"event_id": "e1", "label": "TP", "strike_type": "MEN", "reviewer_dan": 3}]
        items_2 = [{"event_id": "e2", "label": "FP", "strike_type": "KOTE", "reviewer_dan": 6}]
        items_3 = [{"event_id": "e3", "label": "TP", "strike_type": "DO", "reviewer_dan": 7}]

        # 1ª Entrada com 3º Dan
        cfg_1, rec_1 = self.mgr.save_review_session(
            video_name=video_name,
            profile_key="normal",
            reviewer_dan=3,
            review_items=items_1,
            current_profile_config=cfg,
            video_url=video_url
        )
        self.assertEqual(rec_1["reviewer_dan"], 3)

        # 2ª Entrada com o mesmo link de vídeo com 6º Dan -> PERMITIDA SEM RESTRIÇÃO
        cfg_2, rec_2 = self.mgr.save_review_session(
            video_name=video_name,
            profile_key="normal",
            reviewer_dan=6,
            review_items=items_2,
            current_profile_config=cfg_1,
            video_url=video_url
        )
        self.assertEqual(rec_2["reviewer_dan"], 6)

        # 3ª Entrada com o mesmo link de vídeo com 7º Dan -> PERMITIDA SEM RESTRIÇÃO
        cfg_3, rec_3 = self.mgr.save_review_session(
            video_name=video_name,
            profile_key="normal",
            reviewer_dan=7,
            review_items=items_3,
            current_profile_config=cfg_2,
            video_url=video_url
        )
        self.assertEqual(rec_3["reviewer_dan"], 7)

        # Histórico deve conter as 3 sessões do mesmo link
        history = self.mgr.load_history()
        self.assertEqual(len(history), 3)
        self.assertTrue(all(h["video_url"] == video_url for h in history))

        # O vídeo NÃO deve estar registrado na lista de vídeos dos Shinpans
        is_sh, _ = self.mgr.is_video_reviewed_by_shinpan(video_url=video_url)
        self.assertFalse(is_sh)

    def test_shinpan_and_dan_interoperability_on_same_video(self):
        """Valida que um vídeo avaliado por Dan pode receber 1 Decisão dos Shinpans, e depois continuar recebendo avaliações por Dan livremente."""
        video_url = "https://www.youtube.com/watch?v=interop_kendo_match"
        video_name = "Combate de Exemplo Interoperabilidade"
        cfg = {
            "name": "Treino Geral (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }

        # 1. Avaliação prévia por 4º Dan
        cfg, _ = self.mgr.save_review_session(
            video_name=video_name,
            profile_key="normal",
            reviewer_dan=4,
            review_items=[{"event_id": "dan_ev", "label": "TP", "strike_type": "MEN", "reviewer_dan": 4}],
            current_profile_config=cfg,
            video_url=video_url
        )

        # 2. Primeira Decisão dos Shinpans para este vídeo -> PERMITIDA
        cfg, sh_rec = self.mgr.save_review_session(
            video_name=video_name,
            profile_key="normal",
            reviewer_dan="shinpan",
            review_items=[{"event_id": "sh_ev", "label": "TP", "strike_type": "MEN", "reviewer_dan": "shinpan"}],
            current_profile_config=cfg,
            video_url=video_url
        )
        self.assertEqual(sh_rec["reviewer_dan"], "shinpan")

        # 3. Segunda Decisão dos Shinpans para este vídeo -> BLOQUEADA!
        with self.assertRaises(DuplicateShinpanReviewError):
            self.mgr.save_review_session(
                video_name=video_name,
                profile_key="normal",
                reviewer_dan="shinpan",
                review_items=[{"event_id": "sh_ev2", "label": "TP", "strike_type": "KOTE", "reviewer_dan": "shinpan"}],
                current_profile_config=cfg,
                video_url=video_url
            )

        # 4. Avaliação posterior por 8º Dan no mesmo vídeo -> PERMITIDA SEM QUALQUER RESTRIÇÃO!
        cfg, dan8_rec = self.mgr.save_review_session(
            video_name=video_name,
            profile_key="normal",
            reviewer_dan=8,
            review_items=[{"event_id": "dan8_ev", "label": "TP", "strike_type": "TSUKI", "reviewer_dan": 8}],
            current_profile_config=cfg,
            video_url=video_url
        )
        self.assertEqual(dan8_rec["reviewer_dan"], 8)

    def test_reset_clears_shinpan_video_registry(self):
        """Valida que reset_all_training_data limpa o registro de vídeos dos Shinpans."""
        video_url = "https://www.youtube.com/watch?v=video_to_reset"
        cfg = {
            "name": "Treino Geral (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }
        self.mgr.save_review_session(
            video_name="Vídeo Reset",
            profile_key="normal",
            reviewer_dan="shinpan",
            review_items=[{"event_id": "sh_1", "label": "TP", "strike_type": "MEN", "reviewer_dan": "shinpan"}],
            current_profile_config=cfg,
            video_url=video_url
        )
        self.assertTrue(self.mgr.is_video_reviewed_by_shinpan(video_url=video_url)[0])

        # Executa o reset do treinamento
        self.mgr.reset_all_training_data()

        # O registro deve estar limpo
        self.assertFalse(self.mgr.is_video_reviewed_by_shinpan(video_url=video_url)[0])
        self.assertEqual(len(self.mgr.load_shinpan_reviewed_videos()), 0)

    def test_export_and_import_package_with_shinpan_registry(self):
        """Valida que export_training_package e import_training_package persistem e restauram a lista de vídeos dos Shinpans."""
        video_url = "https://www.youtube.com/watch?v=package_video_test"
        cfg = {
            "name": "Treino Geral (Normal)",
            "min_total_score": 0.65,
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }
        self.mgr.save_review_session(
            video_name="Vídeo Pacote",
            profile_key="normal",
            reviewer_dan="shinpan",
            review_items=[{"event_id": "sh_1", "label": "TP", "strike_type": "MEN", "reviewer_dan": "shinpan"}],
            current_profile_config=cfg,
            video_url=video_url
        )

        pkg = self.mgr.export_training_package()
        self.assertIn("shinpan_reviewed_videos", pkg)
        self.assertEqual(len(pkg["shinpan_reviewed_videos"]), 1)
        self.assertEqual(pkg["shinpan_reviewed_videos"][0]["video_identifier"], "youtube:package_video_test")

        # Limpa o estado
        self.mgr.reset_all_training_data()
        self.assertFalse(self.mgr.is_video_reviewed_by_shinpan(video_url=video_url)[0])

        # Importa o pacote de volta
        imp_summary = self.mgr.import_training_package(pkg)
        self.assertEqual(imp_summary["status"], "success")

        # O vídeo deve estar registrado novamente
        is_re_reviewed, rec = self.mgr.is_video_reviewed_by_shinpan(video_url=video_url)
        self.assertTrue(is_re_reviewed)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["video_identifier"], "youtube:package_video_test")

if __name__ == "__main__":
    unittest.main()

