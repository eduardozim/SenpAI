"""
Testes Unitários para o Eixo 2: Conversão da Pesquisa Web em Parâmetros Físicos Acionáveis.
Cobre:
1. Classificação hierárquica de fontes (Tier 1 a Tier 5).
2. Resolução de conflitos de parâmetros por autoridade e conservadorismo.
3. Pipeline de extração estruturada de restrições em JSON Schema.
4. Mineração empírica de vídeos de referência oficial com percentis (p25, p50, p75, p90).
5. Injeção de Priors Bayesianos no otimizador de calibração matemática.
"""

import os
import json
import unittest
import numpy as np

from src.engine.actionable_research import (
    SourceAuthorityTier,
    SourceHierarchyResolver,
    PhysicalConstraintExtractor,
    EmpiricalDistributionLearner,
    BayesianPriorInjector,
    DEFAULT_PHYSICAL_CONSTRAINTS
)
from src.engine.mathematical_calibrator import BayesianCalibrationOptimizer
from src.engine.llm_assistant import KendoLLMAssistant
from src.engine.auto_trainer import AutoTrainingEngine, AutoTrainer


class TestActionableResearchEixo2(unittest.TestCase):

    def setUp(self):
        self.resolver = SourceHierarchyResolver()
        self.extractor = PhysicalConstraintExtractor(resolver=self.resolver)
        self.test_emp_path = "tests/test_data_empirical_distributions.json"
        self.emp_learner = EmpiricalDistributionLearner(storage_path=self.test_emp_path)
        self.test_kb_path = "tests/test_ai_knowledge_base_eixo2.json"
        self.prior_injector = BayesianPriorInjector(knowledge_base_path=self.test_kb_path)

    def tearDown(self):
        for path in [self.test_emp_path, self.test_kb_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    # --------------------------------------------------------------------------
    # 1. HIERARQUIA DE FONTES (EIXO 2.3)
    # --------------------------------------------------------------------------
    def test_source_hierarchy_classification(self):
        """Verifica a classificação correta das 5 camadas de autoridade."""
        # Tier 1: FIK Official
        tier_fik = self.resolver.classify_source({
            "title": "FIK Official Rulebook - The Regulations of Kendo Shiai and Shinpan",
            "url": "https://www.kendo-fik.org/regulations"
        })
        self.assertEqual(tier_fik, SourceAuthorityTier.LEVEL_1_FIK_OFFICIAL)
        self.assertEqual(tier_fik.priority, 1)
        self.assertEqual(tier_fik.weight, 1.00)

        # Tier 2: AJKF / ZNKR Handbook
        tier_ajkf = self.resolver.classify_source({
            "title": "AJKF Referee Handbook (All Japan Kendo Federation)",
            "url": "https://www.kendo.or.jp/knowledge/rules/"
        })
        self.assertEqual(tier_ajkf, SourceAuthorityTier.LEVEL_2_AJKF_HANDBOOK)
        self.assertEqual(tier_ajkf.priority, 2)
        self.assertEqual(tier_ajkf.weight, 0.85)

        # Tier 3: Literatura especializada
        tier_lit = self.resolver.classify_source({
            "title": "Manual Técnico de Kendo e Arbitragem Contemporânea",
            "type": "Manual Técnico"
        })
        self.assertEqual(tier_lit, SourceAuthorityTier.LEVEL_3_SPECIALIZED_LITERATURE)
        self.assertEqual(tier_lit.priority, 3)

        # Tier 4: Artigos acadêmicos / Biomecânica
        tier_acad = self.resolver.classify_source({
            "title": "Kinematic & Kinetic Analysis of Elite Kendo Strikes (Sports Biomechanics Journal)",
            "url": "https://sports-biomechanics.org/kendo/kinematics-fumikomi"
        })
        self.assertEqual(tier_acad, SourceAuthorityTier.LEVEL_4_ACADEMIC_PAPERS)
        self.assertEqual(tier_acad.priority, 4)

        # Tier 5: Fóruns / Blogs / Redes
        tier_blog = self.resolver.classify_source({
            "title": "Discussão de Kendo no Reddit / Fórum Aberto",
            "url": "https://reddit.com/r/kendo/comments/123"
        })
        self.assertEqual(tier_blog, SourceAuthorityTier.LEVEL_5_BLOGS_FORUMS)
        self.assertEqual(tier_blog.priority, 5)
        self.assertEqual(tier_blog.weight, 0.00)

    # --------------------------------------------------------------------------
    # 2. RESOLUÇÃO DE CONFLITOS E CONSERVADORISMO (EIXO 2.3)
    # --------------------------------------------------------------------------
    def test_conflict_resolution_authority_wins(self):
        """Verifica que a autoridade maior (menor priority) prevalece incondicionalmente."""
        src_fik = {
            "title": "FIK Official Rulebook",
            "authority_tier": 1
        }
        src_paper = {
            "title": "Sports Biomechanics Journal",
            "authority_tier": 4
        }

        # FIK exige postura máxima de 8.5°, artigo sugere 12.0°
        winner_val, winner_src, reason = self.resolver.resolve_constraint_conflict(
            param_name="spine_tilt_max_deg",
            val_existing=12.0,
            source_existing=src_paper,
            val_candidate=8.5,
            source_candidate=src_fik
        )
        self.assertEqual(winner_val, 8.5)
        self.assertEqual(winner_src["title"], "FIK Official Rulebook")
        self.assertIn("Autoridade superior prevaleceu", reason)

    def test_conflict_resolution_conservative_tie_break(self):
        """Verifica que em empate de autoridade, o critério mais rigoroso (conservador) prevalece."""
        src_ajkf_1 = {"title": "AJKF Manual A", "authority_tier": 2}
        src_ajkf_2 = {"title": "AJKF Manual B", "authority_tier": 2}

        # Para parâmetro 'max' (tilt): menor valor é mais conservador
        winner_val, winner_src, reason = self.resolver.resolve_constraint_conflict(
            param_name="spine_tilt_max_deg",
            val_existing=9.5,
            source_existing=src_ajkf_1,
            val_candidate=8.5,
            source_candidate=src_ajkf_2
        )
        self.assertEqual(winner_val, 8.5)
        self.assertEqual(winner_src["title"], "AJKF Manual B")
        self.assertIn("Critério conservador", reason)

        # Para parâmetro 'min' (tempo zanshin): maior valor é mais conservador (mais exigente)
        winner_val_z, winner_src_z, _ = self.resolver.resolve_constraint_conflict(
            param_name="zanshin_duration_min_sec",
            val_existing=0.70,
            source_existing=src_ajkf_1,
            val_candidate=0.85,
            source_candidate=src_ajkf_2
        )
        self.assertEqual(winner_val_z, 0.85)
        self.assertEqual(winner_src_z["title"], "AJKF Manual B")

    def test_conflict_resolution_discards_blogs(self):
        """Verifica que fontes de fórum/blog (Tier 5) são descartadas imediatamente."""
        src_official = {"title": "AJKF Handbook", "authority_tier": 2}
        src_blog = {"title": "Blog de Kendo", "authority_tier": 5}

        winner_val, winner_src, reason = self.resolver.resolve_constraint_conflict(
            param_name="fumikomi_sync_ms",
            val_existing=30.0,
            source_existing=src_official,
            val_candidate=90.0,
            source_candidate=src_blog
        )
        self.assertEqual(winner_val, 30.0)
        self.assertEqual(winner_src["title"], "AJKF Handbook")
        self.assertIn("Prior descartado", reason)

    # --------------------------------------------------------------------------
    # 3. EXTRAÇÃO ESTRUTURADA EM JSON SCHEMA (EIXO 2.1)
    # --------------------------------------------------------------------------
    def test_physical_constraint_extraction(self):
        """Verifica a extração estruturada de constraints biomecânicas a partir de metadados/texto."""
        source_meta = {
            "title": "FIK Official Refereeing Guidelines for Men Strikes",
            "summary": "O golpe de Men exige postura ereta com inclinação de coluna < 8.5° e sincronismo de Fumikomi em 35ms. Zanshin com duração mínima de 0.80s e desvio de Hasuji menor que 12°.",
            "principles": ["Ki-Ken-Tai-Ichi simultâneo", "Shisei ereto"]
        }

        extracted = self.extractor.extract_from_source(source_meta)

        self.assertIn("concept", extracted)
        self.assertIn("constraints", extracted)
        c = extracted["constraints"]

        self.assertIn("spine_tilt_max_deg", c)
        self.assertAlmostEqual(c["spine_tilt_max_deg"], 8.5, places=1)
        self.assertIn("fumikomi_hand_foot_window_ms", c)
        self.assertIn("zanshin_duration_min_sec", c)
        self.assertAlmostEqual(c["zanshin_duration_min_sec"], 0.80, places=1)
        self.assertIn("hasuji_max_deviation_deg", c)
        self.assertAlmostEqual(c["hasuji_max_deviation_deg"], 12.0, places=1)

    def test_llm_assistant_extract_physical_constraints(self):
        """Verifica que o KendoLLMAssistant expõe o método de extração paramétrica com fallback offline."""
        llm = KendoLLMAssistant()
        sample_text = "Manual da FIK: Golpe de Kote com Tenouchi firme, coluna ereta até 9.0 graus e Zanshin de 0.75 segundos."

        res = llm.extract_physical_constraints(sample_text, source_title="FIK Kote Regulations")

        self.assertIsInstance(res, dict)
        self.assertIn("constraints", res)
        constraints = res["constraints"]
        self.assertIn("spine_tilt_max_deg", constraints)

    # --------------------------------------------------------------------------
    # 4. MINERAÇÃO DE VÍDEOS DE REFERÊNCIA OFICIAL (EIXO 2.2)
    # --------------------------------------------------------------------------
    def test_empirical_distribution_calculation(self):
        """Verifica o cálculo de estatísticas descritivas e percentis (p25, p50, p75, p90)."""
        values = [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
        dist = self.emp_learner.compute_distribution(values)

        self.assertAlmostEqual(dist["mean"], float(np.mean(values)), places=3)
        self.assertAlmostEqual(dist["p50"], 0.85, places=2)  # mediana
        self.assertEqual(dist["sample_count"], 9)
        self.assertGreater(dist["p90"], dist["p75"])
        self.assertGreater(dist["p75"], dist["p50"])
        self.assertGreater(dist["p50"], dist["p25"])

    def test_mine_from_confirmed_clips(self):
        """Verifica a mineração a partir de clipes oficiais confirmados por árbitros (flags >= 2)."""
        mock_clips = [
            {
                "strike_type": "MEN",
                "label": "IPPON",
                "flags_confirmed": 3,
                "sub_scores": {"target_impact": 0.88, "fumikomi_sync": 0.84, "posture": 0.86, "zanshin": 0.82, "hasuji": 0.85}
            },
            {
                "strike_type": "MEN",
                "label": "IPPON",
                "flags_confirmed": 2,
                "sub_scores": {"target_impact": 0.82, "fumikomi_sync": 0.80, "posture": 0.82, "zanshin": 0.78, "hasuji": 0.80}
            },
            {
                "strike_type": "KOTE",
                "label": "VALID",
                "flags_confirmed": 3,
                "sub_scores": {"target_impact": 0.85, "fumikomi_sync": 0.78, "posture": 0.80, "zanshin": 0.75, "hasuji": 0.82}
            },
            {
                # Lance inválido (0 bandeiras) -> deve ser ignorado
                "strike_type": "MEN",
                "label": "NO_IPPON",
                "flags_confirmed": 0,
                "sub_scores": {"target_impact": 0.40, "fumikomi_sync": 0.35, "posture": 0.50, "zanshin": 0.30, "hasuji": 0.45}
            }
        ]

        results = self.emp_learner.mine_from_confirmed_clips(mock_clips)

        self.assertIn("metadata", results)
        self.assertIn("distributions_by_strike", results)
        dists = results["distributions_by_strike"]

        self.assertIn("MEN", dists)
        self.assertIn("KOTE", dists)
        self.assertIn("DO", dists)
        self.assertIn("TSUKI", dists)

        men_target = dists["MEN"]["target_impact"]
        self.assertGreater(men_target["mean"], 0.75)
        self.assertIn("p50", men_target)
        self.assertIn("p90", men_target)

    # --------------------------------------------------------------------------
    # 5. INJEÇÃO DE PRIORS BAYESIANOS NO MOTOR DE OTIMIZAÇÃO (EIXO 2.1 & 2.3)
    # --------------------------------------------------------------------------
    def test_bayesian_prior_injection_into_optimizer(self):
        """Verifica que as fronteiras físicas são respeitadas pelo otimizador de calibração."""
        priors = self.prior_injector.derive_optimizer_bounds_and_priors(strike_type="MEN")

        self.assertEqual(priors["strike_type"], "MEN")
        self.assertIn("sub_threshold_lower_bounds", priors)
        lower_bounds = priors["sub_threshold_lower_bounds"]

        # A exigência mínima da FIK para postura e fumikomi impõe limites inferiores rígidos
        self.assertGreaterEqual(lower_bounds["posture"], 0.50)
        self.assertGreaterEqual(lower_bounds["fumikomi_sync"], 0.45)

        # Execução do otimizador de pesos com injeção de priors
        optimizer = BayesianCalibrationOptimizer()
        cfg = {
            "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
            "min_total_score": 0.65,
            "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
        }
        feedbacks = [
            {"label": "TP", "sub_scores": {"target_impact": 0.85, "fumikomi_sync": 0.80, "posture": 0.80, "zanshin": 0.75}, "strike_type": "MEN"},
            {"label": "FP", "sub_scores": {"target_impact": 0.45, "fumikomi_sync": 0.40, "posture": 0.35, "zanshin": 0.30}, "strike_type": "MEN"},
        ]

        opt_cfg, metrics = optimizer.optimize(cfg, feedbacks, strike_type="MEN")

        # Verifica que os sub_thresholds otimizados não foram rebaixados além das fronteiras físicas
        sub_th = opt_cfg["sub_thresholds"]
        self.assertGreaterEqual(sub_th["posture"], lower_bounds["posture"])
        self.assertGreaterEqual(sub_th["fumikomi_sync"], lower_bounds["fumikomi_sync"])
        self.assertGreaterEqual(opt_cfg["min_total_score"], priors["min_total_score_floor"])

    def test_auto_trainer_integration_eixo2(self):
        """Verifica a integração completa dos métodos do Eixo 2 no AutoTrainer."""
        trainer = AutoTrainer(knowledge_base_path=self.test_kb_path)

        constraints = trainer.get_physical_constraints()
        self.assertIsInstance(constraints, dict)
        self.assertIn("men_strike_biomechanics", constraints)

        emp_dists = trainer.get_empirical_reference_distributions()
        self.assertIsInstance(emp_dists, dict)

        # Teste de mineração de clipes oficiais
        mined = trainer.mine_official_video_clips([])
        self.assertIn("distributions_by_strike", mined)


if __name__ == "__main__":
    unittest.main()
