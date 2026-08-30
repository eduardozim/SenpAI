"""
Motor de Calibração de Sensibilidade e Avaliação Técnica.
Gerencia perfis de calibração (Rígido, Normal, Permissivo) e calcula a pontuação final ponderada.
"""

import json
import os
from typing import Dict, Any

class CalibrationEngine:
    def __init__(self, config_path: str = "config/calibration_profiles.json", profile_name: str = "normal"):
        self.config_path = config_path
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
                    "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
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
            self.active_config = self.profiles["normal"].copy()

    def update_custom_settings(
        self,
        min_total_score: float,
        weight_target: float,
        weight_fumikomi: float,
        weight_posture: float,
        weight_zanshin: float
    ):
        """Permite ajuste fino dos sliders de calibração."""
        total_w = weight_target + weight_fumikomi + weight_posture + weight_zanshin
        if total_w == 0:
            total_w = 1.0

        self.active_config["min_total_score"] = min_total_score
        self.active_config["weights"] = {
            "target_impact": weight_target / total_w,
            "fumikomi_sync": weight_fumikomi / total_w,
            "posture": weight_posture / total_w,
            "zanshin": weight_zanshin / total_w
        }

    def update_and_save_profile(self, profile_key: str, updated_config: Dict[str, Any]):
        """Atualiza a configuração do perfil selecionado e persiste no JSON de perfis."""
        self.profiles[profile_key] = updated_config
        if profile_key == self.current_profile_key:
            self.active_config = updated_config.copy()
            
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.profiles, f, indent=2, ensure_ascii=False)

    def evaluate_strike(
        self,
        target_score: float,
        fumikomi_score: float,
        posture_score: float,
        zanshin_score: float
    ) -> Dict[str, Any]:
        """
        Aplica os pesos da calibração ativa e determina a validade do ponto (Yuko-datotsu).
        """
        weights = self.active_config["weights"]
        min_total = self.active_config["min_total_score"]
        sub_thresholds = self.active_config.get("sub_thresholds", {})

        total_score = (
            target_score * weights["target_impact"] +
            fumikomi_score * weights["fumikomi_sync"] +
            posture_score * weights["posture"] +
            zanshin_score * weights["zanshin"]
        )

        # Checar se atendeu ao escore mínimo global e aos sub-requisitos mínimos
        is_valid = total_score >= min_total
        
        # Se um sub-requisito crítico falhou acentuadamente, invalida o ponto mesmo que o total passe
        failed_subcriteria = []
        if target_score < sub_thresholds.get("target_impact", 0.40):
            is_valid = False
            failed_subcriteria.append("ALVO_FORA")
        if fumikomi_score < sub_thresholds.get("fumikomi_sync", 0.30):
            failed_subcriteria.append("SEM_FUMIKOMI")
        if posture_score < sub_thresholds.get("posture", 0.30):
            failed_subcriteria.append("POSTURA_INCLINADA")
        if zanshin_score < sub_thresholds.get("zanshin", 0.20):
            failed_subcriteria.append("SEM_ZANSHIN")

        return {
            "is_valid": is_valid,
            "total_score": round(total_score * 100, 1),
            "min_required": round(min_total * 100, 1),
            "profile_used": self.active_config.get("name", "Custom"),
            "sub_scores": {
                "target_impact": round(target_score * 100, 1),
                "fumikomi_sync": round(fumikomi_score * 100, 1),
                "posture": round(posture_score * 100, 1),
                "zanshin": round(zanshin_score * 100, 1)
            },
            "failed_subcriteria": failed_subcriteria
        }
