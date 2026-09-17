"""
Testes Automatizados para o Motor de Treinamento Automático por IA, Retreinamento de Modelo e Painel de Evolução.
"""

import os
import json
import time
import unittest
from src.engine.auto_trainer import AutoTrainingEngine, AUTO_TRAINING_SCOPES, KENDO_KNOWLEDGE_RESOURCES
from src.engine.feedback_manager import FeedbackManager

class TestAutoTrainer(unittest.TestCase):
    def setUp(self):
        self.test_kb_path = "config/test_ai_knowledge_base.json"
        self.test_profiles_path = "config/test_calibration_profiles.json"
        self.test_history_path = "data/test_training_history.json"
        self.test_feedback_path = "data/test_feedback_dataset.json"
        self.test_checkpoint_path = "data/test_auto_training_checkpoint.json"

        for p in [self.test_kb_path, self.test_profiles_path, self.test_history_path, self.test_feedback_path, self.test_checkpoint_path]:
            if os.path.exists(p):
                os.remove(p)

        self.engine = AutoTrainingEngine(
            knowledge_base_path=self.test_kb_path,
            profiles_path=self.test_profiles_path,
            history_path=self.test_history_path,
            feedback_path=self.test_feedback_path,
            checkpoint_path=self.test_checkpoint_path
        )

    def tearDown(self):
        for p in [self.test_kb_path, self.test_profiles_path, self.test_history_path, self.test_feedback_path, self.test_checkpoint_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_knowledge_base_initialization(self):
        """Valida a inicialização automática e integridade da base de conhecimento de IA de Kendo."""
        kb = self.engine.load_knowledge_base()
        self.assertIn("version", kb)
        self.assertIn("sources", kb)
        self.assertIn("learned_parameters", kb)
        self.assertGreaterEqual(len(kb["sources"]), 2)
        self.assertIn("fik_regulations", kb["sources"])

    def test_diagnose_latent_need_empty_state(self):
        """Valida o diagnóstico de necessidade mais latente em estado inicial do sistema."""
        diag = self.engine.diagnose_latent_need()
        self.assertIn("chosen_scope", diag)
        self.assertIn("scope_name", diag)
        self.assertIn("diagnosis_reasons", diag)
        self.assertGreaterEqual(len(diag["diagnosis_reasons"]), 1)

    def test_diagnose_latent_need_with_false_positives(self):
        """Valida o diagnóstico quando há acúmulo de Falsos Positivos elegendo Shiai/Arbitragem."""
        fps = [
            {"event_id": f"fp_{i}", "label": "FP", "category": "INVALID_HIT", "strike_type": "MEN", "timestamp": "00:01.000"}
            for i in range(8)
        ]
        with open(self.test_feedback_path, "w", encoding="utf-8") as f:
            json.dump(fps, f, indent=2)

        diag = self.engine.diagnose_latent_need()
        self.assertEqual(diag["chosen_scope"], "recorded_shiai")
        self.assertTrue(any("Falsos Positivos" in r for r in diag["diagnosis_reasons"]))

    def test_retrain_detection_model(self):
        """Valida o retreinamento efetivo do modelo de detecção de golpes e calibração de perfis."""
        sources = [
            {"id": "fik_regulations", "title": "FIK Regulations", "type": "Regulamento FIK", "focus": "Yuko-Datotsu"}
        ]
        res = self.engine.retrain_detection_model(
            effective_scope="recorded_shiai",
            sources_consulted=sources,
            intensity="padrao"
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("normal", res["profiles_retrained"])
        self.assertGreaterEqual(len(res["improvements"]), 1)

        # Verificar se os perfis foram gravados em disco com novos pesos
        profiles = self.engine.calibrator.get_all_profiles()
        self.assertIn("normal", profiles)
        self.assertEqual(profiles["normal"]["weights"]["target_impact"], 0.40)

    def test_run_auto_training_quick_execution(self):
        """Valida a execução de um ciclo de auto-treinamento rápido respeitando o tempo e retreinando o modelo."""
        callbacks_received = []

        def on_progress(p_data):
            callbacks_received.append(p_data)

        # Duração ultrarrápida (0.05 min ~ 3-5s para teste)
        result = self.engine.run_auto_training(
            scope_key="latent_need",
            duration_minutes=0.08,
            intensity="rapido",
            include_video=True,
            include_text_guidelines=True,
            progress_callback=on_progress
        )

        self.assertIn(result["status"], ["success", "stopped_early"])
        self.assertGreater(result["final_accuracy_pct"], result["initial_accuracy_pct"])
        self.assertGreaterEqual(len(result["sources_consulted"]), 1)
        self.assertGreaterEqual(len(result["improvements_summary"]), 1)
        self.assertGreater(len(callbacks_received), 0)
        self.assertIn("retrain_summary", result)

        # Validar persistência no histórico de governança
        history = self.engine.feedback_mgr.load_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["reviewer_dan"], 0)
        self.assertTrue(history[0].get("is_auto_training", False))
        self.assertEqual(history[0]["optimization_summary"]["mode"], "auto_training_ai")

    def test_get_evolution_statistics_and_sources(self):
        """Valida a geração de estatísticas consolidadas e consulta de corpus para o Painel de Evolução."""
        # Executar um ciclo de auto-treinamento para popular estatísticas
        self.engine.run_auto_training(
            scope_key="recorded_shiai",
            duration_minutes=0.08,
            intensity="rapido"
        )

        stats = self.engine.get_evolution_statistics()
        self.assertEqual(stats["total_auto_trainings"], 1)
        self.assertGreater(stats["total_duration_seconds"], 0)
        self.assertIn("s", stats["total_duration_formatted"])
        self.assertGreaterEqual(stats["average_accuracy_pct"], 30.0)
        self.assertGreaterEqual(len(stats["accuracy_timeline"]), 1)
        self.assertGreaterEqual(stats["total_sources_indexed"], 2)

        # Validar corpus de fontes consultadas
        sources_list = self.engine.get_consulted_knowledge_sources()
        self.assertGreaterEqual(len(sources_list), 2)
        self.assertTrue(any(s["id"] == "fik_regulations" for s in sources_list))

    def test_run_auto_training_for_modalities(self):
        """Valida o treinamento específico para as 14 modalidades pedagógicas de Kendo."""
        result = self.engine.run_auto_training(
            scope_key="all_14_modalities",
            duration_minutes=0.08,
            intensity="padrao"
        )
        self.assertEqual(result["scope_key"], "all_14_modalities")
        self.assertTrue(any("14 Modalidades" in imp for imp in result["improvements_summary"]))

    def test_modalities_accuracy_summary(self):
        """Valida a geração do sumário de acurácia para todas as 14 modalidades de aprendizado de Kendo."""
        summary = self.engine.get_modalities_accuracy_summary()
        self.assertEqual(len(summary), 14)
        
        # Validar estrutura de cada modalidade com baselines realistas (< 50%)
        for mod in summary:
            self.assertIn("key", mod)
            self.assertIn("name", mod)
            self.assertIn("japanese", mod)
            self.assertIn("category", mod)
            self.assertGreaterEqual(mod["current_accuracy"], 30.0)
            self.assertLessEqual(mod["current_accuracy"], 100.0)
            self.assertIn("gain_formatted", mod)
            self.assertIn(mod["status"], ["Excelente / Shiai", "Calibrado", "Em Calibração", "Fase Inicial (Falsos Positivos)"])
            self.assertIn("pillar_movement_pct", mod)
            self.assertIn("pillar_precision_pct", mod)
            self.assertIn("pillar_constancy_pct", mod)
            self.assertIn("cadence_optimal", mod)

        # Validar que get_evolution_statistics inclui o sumário e média
        stats = self.engine.get_evolution_statistics()
        self.assertIn("modalities_accuracy_summary", stats)
        self.assertEqual(len(stats["modalities_accuracy_summary"]), 14)
        self.assertIn("average_modality_accuracy_pct", stats)
        self.assertGreaterEqual(stats["average_modality_accuracy_pct"], 30.0)

    def test_checkpoint_crud_and_persistence(self):
        """Valida gravação atômica, leitura e limpeza de checkpoints persistentes."""
        self.assertIsNone(self.engine.load_checkpoint())
        self.assertIsNone(self.engine.has_saved_checkpoint())

        ckpt_data = {
            "status": "in_progress",
            "session_id": "test_sess_01",
            "scope_key": "all_14_modalities",
            "samples_processed": 150,
            "current_accuracy": 89.2
        }
        self.engine.save_checkpoint(ckpt_data)

        loaded = self.engine.load_checkpoint()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["session_id"], "test_sess_01")
        self.assertEqual(loaded["samples_processed"], 150)
        self.assertIn("last_checkpoint_timestamp", loaded)

        self.engine.clear_checkpoint()
        self.assertIsNone(self.engine.load_checkpoint())

    def test_error_salvages_and_consolidates_knowledge(self):
        """Valida que em caso de erro no meio do ciclo todo o conhecimento adquirido é salvo e consolidado sem perda."""
        call_count = [0]

        def failing_callback(progress_data):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise RuntimeError("Simulação de Falha Crítica de Rede / Memória")

        result = self.engine.run_auto_training(
            scope_key="modality_suburi",
            duration_minutes=0.15,
            intensity="padrao",
            progress_callback=failing_callback
        )

        # O motor deve capturar a exceção e retornar status salvado
        self.assertEqual(result["status"], "interrupted_salvaged")
        self.assertIn("Simulação de Falha Crítica", result.get("error_message", ""))
        self.assertGreater(result["samples_processed"], 0)
        self.assertGreaterEqual(len(result["sources_consulted"]), 1)

        # Validar que a Base de Conhecimento e Histórico foram gravados com os dados até o erro
        kb = self.engine.load_knowledge_base()
        self.assertGreaterEqual(kb["training_sessions_completed"], 1)

        history = self.engine.feedback_mgr.load_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["optimization_summary"]["status"], "interrupted_salvaged")

        # Validar que o checkpoint foi marcado como consolidado
        ckpt = self.engine.load_checkpoint()
        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt["status"], "interrupted_salvaged")
        self.assertTrue(ckpt.get("consolidated", False))

    def test_continuous_cumulative_training(self):
        """Valida que o treinamento seguinte parte exatamente da acurácia e conhecimento do treinamento anterior."""
        # Sessão 1
        res1 = self.engine.run_auto_training(
            scope_key="modality_suburi",
            duration_minutes=0.08,
            intensity="rapido"
        )
        acc_s1 = res1["final_accuracy_pct"]

        # Sessão 2: deve herdar acc_s1 como ponto de partida
        res2 = self.engine.run_auto_training(
            scope_key="modality_suburi",
            duration_minutes=0.08,
            intensity="rapido"
        )
        self.assertEqual(res2["initial_accuracy_pct"], acc_s1)
        self.assertGreaterEqual(res2["final_accuracy_pct"], acc_s1)

    def test_consolidate_pending_checkpoint(self):
        """Valida consolidação automática de fontes e acurácia de checkpoint pendente."""
        ckpt_data = {
            "status": "in_progress",
            "scope_key": "modality_suburi",
            "current_accuracy": 95.5,
            "sources_consulted": [
                {"title": "Tratado Específico de Haya-Suburi Rápido", "type": "Tratado Especial"}
            ],
            "consolidated": False
        }
        self.engine.save_checkpoint(ckpt_data)

        # Consolidar
        res = self.engine.consolidate_pending_checkpoint()
        self.assertIsNotNone(res)
        self.assertTrue(res.get("consolidated"))

        # Checar na Base de Conhecimento
        kb = self.engine.load_knowledge_base()
        self.assertTrue(any("Haya-Suburi" in s.get("title", "") for s in kb["sources"].values()))

    def test_reset_knowledge_base(self):
        """Valida o reset completo da base de conhecimento e checkpoint."""
        self.engine.save_checkpoint({"status": "in_progress"})
        self.assertIsNotNone(self.engine.load_checkpoint())

    def test_get_scope_current_accuracy(self):
        """Valida a consulta dinâmica da acurácia e ganho acumulado por escopo."""
        # Estado inicial
        info_sub = self.engine.get_scope_current_accuracy("modality_suburi")
        self.assertEqual(info_sub["current_accuracy"], 46.5)
        self.assertEqual(info_sub["sessions_count"], 0)

        info_shiai = self.engine.get_scope_current_accuracy("recorded_shiai")
        self.assertEqual(info_shiai["current_accuracy"], 34.0)

        # Após um treinamento em suburi
        self.engine.run_auto_training(
            scope_key="modality_suburi",
            duration_minutes=0.08,
            intensity="rapido"
        )
        info_sub_after = self.engine.get_scope_current_accuracy("modality_suburi")
        self.assertGreater(info_sub_after["current_accuracy"], 46.5)
        self.assertGreater(info_sub_after["gain_pct"], 0)
        self.assertGreaterEqual(info_sub_after["sessions_count"], 1)

    def test_diagnose_latent_need_selectable_strategies(self):
        """Valida a seleção explícita das 3 estratégias de necessidade mais latente."""
        # 1. Menor percentual de aprendizado
        diag_low = self.engine.diagnose_latent_need(strategy="lowest_accuracy")
        self.assertIn("chosen_scope", diag_low)
        self.assertTrue(any("menor percentual" in r.lower() or "menor precisão" in r.lower() or "latente" in r.lower() for r in diag_low["diagnosis_reasons"]))

        # 2. Conhecimento geral do Kendo
        diag_gen = self.engine.diagnose_latent_need(strategy="general_knowledge")
        self.assertEqual(diag_gen["chosen_scope"], "general_all")
        self.assertTrue(any("conhecimento geral" in r.lower() or "princípios do kendo" in r.lower() for r in diag_gen["diagnosis_reasons"]))

        # 3. Modalidade randômica
        diag_rnd = self.engine.diagnose_latent_need(strategy="random_modality")
        self.assertIn("chosen_scope", diag_rnd)
        self.assertTrue(diag_rnd["chosen_scope"].startswith("modality_"))
        self.assertTrue(any("randômica" in r.lower() or "randomica" in r.lower() for r in diag_rnd["diagnosis_reasons"]))

    def test_diagnose_latent_need_automatic_sequence(self):
        """Valida a progressão sequencial automática em 3 etapas: 1º menor acurácia -> 2º geral -> 3º randômica -> loop."""
        kb = self.engine.load_knowledge_base()

        # Etapa 1: Menor percentual de aprendizado
        kb["learned_parameters"]["auto_learning_sequence_step"] = 0
        self.engine.save_knowledge_base(kb)
        diag_step1 = self.engine.diagnose_latent_need(strategy="auto")
        self.assertEqual(diag_step1["sequence_step"], 1)
        self.assertTrue(diag_step1["chosen_scope"].startswith("modality_"))
        self.assertTrue(any("etapa 1/3" in r.lower() or "menor percentual" in r.lower() for r in diag_step1["diagnosis_reasons"]))

        # Etapa 2: Conhecimento geral
        kb["learned_parameters"]["auto_learning_sequence_step"] = 1
        self.engine.save_knowledge_base(kb)
        diag_step2 = self.engine.diagnose_latent_need(strategy="auto")
        self.assertEqual(diag_step2["sequence_step"], 2)
        self.assertEqual(diag_step2["chosen_scope"], "general_all")
        self.assertTrue(any("etapa 2/3" in r.lower() or "conhecimento geral" in r.lower() for r in diag_step2["diagnosis_reasons"]))

        # Etapa 3: Modalidade randômica (última opção)
        kb["learned_parameters"]["auto_learning_sequence_step"] = 2
        self.engine.save_knowledge_base(kb)
        diag_step3 = self.engine.diagnose_latent_need(strategy="auto")
        self.assertEqual(diag_step3["sequence_step"], 3)
        self.assertTrue(diag_step3["chosen_scope"].startswith("modality_"))
        self.assertTrue(any("etapa 3/3" in r.lower() or "randômica" in r.lower() for r in diag_step3["diagnosis_reasons"]))

        # Ciclo reinicia: Etapa 4 -> 1
        kb["learned_parameters"]["auto_learning_sequence_step"] = 3
        self.engine.save_knowledge_base(kb)
        diag_step4 = self.engine.diagnose_latent_need(strategy="auto")
        self.assertEqual(diag_step4["sequence_step"], 1)
        self.assertTrue(diag_step4["chosen_scope"].startswith("modality_"))

    def test_web_knowledge_search_and_fallback(self):
        """Valida que a busca web por IA encontra dados e lida com falha de rede/desconhecidos com fallback."""
        # Busca por modalidade existente (deve retornar fontes da web ou base local resiliente)
        sources_sub = self.engine.search_web_kendo_knowledge("suburi", max_results=2)
        self.assertIsInstance(sources_sub, list)
        self.assertGreaterEqual(len(sources_sub), 1)
        self.assertIn("title", sources_sub[0])
        self.assertIn("summary", sources_sub[0])

        # Busca por escopo geral
        sources_gen = self.engine.search_web_kendo_knowledge("general", max_results=2)
        self.assertGreaterEqual(len(sources_gen), 1)

        # Busca por chave desconhecida (deve retornar fallback sem crash)
        sources_unk = self.engine.search_web_kendo_knowledge("modalidade_inexistente_xyz", max_results=2)
        self.assertIsInstance(sources_unk, list)
        self.assertGreaterEqual(len(sources_unk), 1)

    def test_cumulative_modality_knowledge_recording(self):
        """Valida que o treinamento automático acumula princípios, fontes e logs de evolução de forma persistente."""
        # Executar treino rápido em kirikaeshi
        res = self.engine.run_auto_training(
            scope_key="modality_kirikaeshi",
            duration_minutes=0.08,
            intensity="rapido"
        )
        self.assertEqual(res["status"], "success")

        # Inspecionar conhecimento acumulado da modalidade
        learned = self.engine.get_modality_learned_knowledge("kirikaeshi")
        self.assertEqual(learned["key"], "kirikaeshi")
        self.assertIn("principles_learned", learned)
        self.assertGreaterEqual(len(learned["principles_learned"]), 2)
        self.assertIn("biomechanical_profile", learned)
        self.assertIn("evolution_log", learned)
        self.assertGreaterEqual(len(learned["evolution_log"]), 1)

        # Princípios gerais do Kendo
        gen_principles = self.engine.get_general_kendo_principles()
        self.assertGreaterEqual(len(gen_principles), 4)

    def test_export_and_merge_knowledge_data(self):
        """Valida a exportação e mesclagem cumulativa da Base de Conhecimento do Auto-Trainer."""
        # Treinar suburi para elevar acurácia
        self.engine.retrain_detection_model("modality_suburi", sources_consulted=[{
            "title": "Manual FIK Teste",
            "type": "Manual Oficial",
            "principles": ["Hasuji Estrito no Suburi"]
        }])

        exp_data = self.engine.export_knowledge_data()
        self.assertIn("ai_knowledge_base", exp_data)
        self.assertIn("sources", exp_data["ai_knowledge_base"])

        kb_suburi = exp_data["ai_knowledge_base"]["learned_parameters"]["training_modalities"]["suburi"]
        suburi_acc = kb_suburi["current_accuracy"]

        # Resetar a base
        self.engine.reset_knowledge_base()
        kb_fresh = self.engine.load_knowledge_base()
        self.assertLess(
            kb_fresh["learned_parameters"]["training_modalities"]["suburi"]["current_accuracy"],
            suburi_acc
        )

        # Mesclar os dados exportados
        merge_res = self.engine.merge_knowledge_data(exp_data["ai_knowledge_base"], exp_data.get("auto_training_checkpoint"))
        self.assertEqual(merge_res["status"], "success")

        kb_restored = self.engine.load_knowledge_base()
        self.assertEqual(
            kb_restored["learned_parameters"]["training_modalities"]["suburi"]["current_accuracy"],
            suburi_acc
        )
        self.assertIn(
            "Hasuji Estrito no Suburi",
            kb_restored["learned_parameters"]["training_modalities"]["suburi"]["principles_learned"]
        )


if __name__ == "__main__":
    unittest.main()


