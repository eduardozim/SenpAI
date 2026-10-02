"""
Motor de Calibração de Sensibilidade e Avaliação Técnica.
Gerencia perfis de calibração (Rígido, Normal, Permissivo, Shiai),
ponderação especializada por tipo de golpe (Men, Kote, Do, Tsuki),
classificador probabilístico calibrado (Platt Scaling) e detecção de Concept Drift.
Implementação alinhada ao Eixo 1 de REFERENCIA_MOTOR_CALIBRACAO_E_APRENDIZADO.md.
"""

import json
import os
from typing import Dict, Any, List, Optional

from src.engine.mathematical_calibrator import (
    ProbabilisticPlattCalibrator,
    ConceptDriftDetector,
    DEFAULT_WEIGHTS_BY_STRIKE_TYPE
)


class CalibrationEngine:
    def __init__(self, config_path: str = "config/calibration_profiles.json", profile_name: str = "normal"):
        self.config_path = config_path
        self.drift_detector = ConceptDriftDetector()
        self.profiles = self._load_profiles()
        self.set_profile(profile_name)

    def _load_profiles(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # Fallback inline default profile
            return {
                "normal": {
                    "name": "Treino Geral (Normal)",
                    "min_total_score": 0.65,
                    "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
                    "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45},
                    "weights_by_strike_type": json.loads(json.dumps(DEFAULT_WEIGHTS_BY_STRIKE_TYPE)),
                    "platt_scaling": {"a": 12.0, "b": -8.0, "is_fitted": False}
                }
            }

    def get_all_profiles(self) -> Dict[str, Any]:
        """Retorna todos os perfis de calibração carregados."""
        self.profiles = self._load_profiles()
        return self.profiles

    def set_profile(self, profile_name: str):
        if profile_name in self.profiles:
            self.current_profile_key = profile_name
            self.active_config = self.profiles[profile_name].copy()
        else:
            self.current_profile_key = "normal"
            self.active_config = self.profiles.get("normal", {}).copy()

        # Atualiza classificador probabilístico de Platt Scaling
        platt_data = self.active_config.get("platt_scaling", {})
        if platt_data:
            self.platt_calibrator = ProbabilisticPlattCalibrator.from_dict(platt_data)
        else:
            min_tot = self.active_config.get("min_total_score", 0.65)
            self.platt_calibrator = ProbabilisticPlattCalibrator(a_param=12.0, b_param=-12.0 * min_tot)

    def get_weights_for_strike(self, strike_type: Optional[str] = None) -> Dict[str, float]:
        """
        Retorna os pesos especializados para o tipo de golpe (MEN, KOTE, DO, TSUKI).
        Se não fornecido ou não cadastrado, faz fallback para os pesos globais do perfil.
        """
        global_weights = self.active_config.get(
            "weights",
            {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15}
        )

        if not strike_type:
            return global_weights

        st_clean = str(strike_type).upper().strip()
        by_strike = self.active_config.get("weights_by_strike_type", {})
        if st_clean in by_strike:
            return by_strike[st_clean]
        elif st_clean in DEFAULT_WEIGHTS_BY_STRIKE_TYPE:
            return DEFAULT_WEIGHTS_BY_STRIKE_TYPE[st_clean]

        return global_weights

    def update_custom_settings(
        self,
        min_total_score: float,
        weight_target: float,
        weight_fumikomi: float,
        weight_posture: float,
        weight_zanshin: float,
        strike_type: Optional[str] = None
    ):
        """Permite ajuste fino dos sliders de calibração, opcionalmente por tipo de golpe."""
        total_w = weight_target + weight_fumikomi + weight_posture + weight_zanshin
        if total_w == 0:
            total_w = 1.0

        normalized_w = {
            "target_impact": round(weight_target / total_w, 3),
            "fumikomi_sync": round(weight_fumikomi / total_w, 3),
            "posture": round(weight_posture / total_w, 3),
            "zanshin": round(weight_zanshin / total_w, 3)
        }
        diff = round(1.0 - sum(normalized_w.values()), 3)
        normalized_w["target_impact"] = round(normalized_w["target_impact"] + diff, 3)

        self.active_config["min_total_score"] = round(min_total_score, 2)

        if strike_type:
            st_clean = str(strike_type).upper().strip()
            if "weights_by_strike_type" not in self.active_config:
                self.active_config["weights_by_strike_type"] = json.loads(json.dumps(DEFAULT_WEIGHTS_BY_STRIKE_TYPE))
            self.active_config["weights_by_strike_type"][st_clean] = normalized_w
        else:
            self.active_config["weights"] = normalized_w

        # Atualiza o interceptor de Platt para centrar em 50% de probabilidade no novo min_total_score
        self.platt_calibrator.b = -self.platt_calibrator.a * min_total_score

    def update_and_save_profile(self, profile_key: str, updated_config: Dict[str, Any]):
        """Atualiza a configuração do perfil selecionado e persiste no JSON de perfis."""
        self.profiles[profile_key] = updated_config
        if profile_key == self.current_profile_key:
            self.active_config = updated_config.copy()
            platt_data = self.active_config.get("platt_scaling", {})
            if platt_data:
                self.platt_calibrator = ProbabilisticPlattCalibrator.from_dict(platt_data)

        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)

    def evaluate_strike(
        self,
        target_score: float,
        fumikomi_score: float,
        posture_score: float,
        zanshin_score: float,
        strike_type: str = "MEN",
        hasuji_score: Optional[float] = None,
        seme_score: Optional[float] = None,
        is_ku_totsu: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Aplica os pesos da calibração ativa (com especialização por strike_type),
        determina a validade do ponto (Yuko-datotsu) e calcula a probabilidade calibrada (Platt Scaling).
        Suporta o 5° Pilar Hasuji e eliminação de Ku-totsu (Eixo 3.1 e 3.2).
        """
        norm_strike = str(strike_type).upper().strip() if strike_type else "MEN"
        weights = self.get_weights_for_strike(norm_strike)
        min_total = self.active_config.get("min_total_score", 0.65)
        sub_thresholds = self.active_config.get("sub_thresholds", {})

        # Se hasuji_score foi fornecido (5° Pilar do Eixo 3.2), combinamos proporcionalmente
        # mantendo compatibilidade direta com modelos de 4 pilares
        base_score = (
            target_score * weights["target_impact"] +
            fumikomi_score * weights["fumikomi_sync"] +
            posture_score * weights["posture"] +
            zanshin_score * weights["zanshin"]
        )

        if hasuji_score is not None:
            # 5° Pilar: Hasuji modula com peso balanceado de 15% reescalonando a base em 85%
            total_score = (base_score * 0.85) + (float(hasuji_score) * 0.15)
        else:
            total_score = base_score

        # Probabilidade Calibrada de Ippon via Platt Scaling (Eixo 1.2)
        prob = self.platt_calibrator.predict_proba(total_score)
        prob_pct = round(prob * 100.0, 1)

        # Checar se atendeu ao escore mínimo global e aos sub-requisitos mínimos
        is_valid = total_score >= min_total

        # Se um sub-requisito crítico falhou acentuadamente, invalida o ponto mesmo que o total passe
        failed_subcriteria = []
        if is_ku_totsu:
            is_valid = False
            failed_subcriteria.append("KU_TOTSU_VAZIO")

        if target_score < sub_thresholds.get("target_impact", 0.40):
            is_valid = False
            failed_subcriteria.append("ALVO_FORA")
        if fumikomi_score < sub_thresholds.get("fumikomi_sync", 0.30):
            failed_subcriteria.append("SEM_FUMIKOMI")
        if posture_score < sub_thresholds.get("posture", 0.30):
            failed_subcriteria.append("POSTURA_INCLINADA")
        if zanshin_score < sub_thresholds.get("zanshin", 0.20):
            failed_subcriteria.append("SEM_ZANSHIN")
        hasuji_thresh = sub_thresholds.get("hasuji", 0.40)
        if hasuji_score is not None and hasuji_score < hasuji_thresh:
            is_valid = False
            failed_subcriteria.append("HASUJI_INCORRETO")

        sub_scores_dict = {
            "target_impact": round(target_score * 100, 1),
            "fumikomi_sync": round(fumikomi_score * 100, 1),
            "posture": round(posture_score * 100, 1),
            "zanshin": round(zanshin_score * 100, 1)
        }
        if hasuji_score is not None:
            sub_scores_dict["hasuji"] = round(hasuji_score * 100, 1)
        if seme_score is not None:
            sub_scores_dict["seme"] = round(seme_score * 100, 1)

        return {
            "is_valid": is_valid,
            "total_score": round(total_score * 100, 1),
            "min_required": round(min_total * 100, 1),
            "probability": round(prob, 4),
            "probability_pct": prob_pct,
            "probability_label": f"{prob_pct}%",
            "strike_type": norm_strike,
            "profile_used": self.active_config.get("name", "Custom"),
            "weights_used": weights,
            "sub_scores": sub_scores_dict,
            "failed_subcriteria": failed_subcriteria,
            "hasuji_score": round(hasuji_score, 3) if hasuji_score is not None else None,
            "seme_score": round(seme_score, 3) if seme_score is not None else None,
            "is_ku_totsu": is_ku_totsu
        }

    def check_concept_drift(
        self,
        recent_scores: List[float],
        baseline_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Executa o teste Kolmogorov-Smirnov (scipy.stats.ks_2samp) para avaliar desvios
        estatísticos entre os lances recentes e a base histórica de calibração (Eixo 1.3).
        """
        if baseline_scores is None:
            # Baseline padrão sintetizada a partir dos limiares do perfil ativo
            min_tot = self.active_config.get("min_total_score", 0.65)
            # Amostra gaussiana representativa ao redor do limiar de aprovação
            baseline_scores = [round(min_tot + offset, 3) for offset in [-0.15, -0.10, -0.05, 0.0, 0.04, 0.08, 0.12, 0.15, 0.18, 0.20] * 3]

        return self.drift_detector.detect_drift(recent_scores, baseline_scores)
