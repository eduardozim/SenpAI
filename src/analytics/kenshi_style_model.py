"""
Módulo de Modelagem do Estilo Individual do Kenshi (Eixo 6).
Implementa o Perfil Cinestésico Individual (Kinesthetic Baseline) e
mecanismos de avaliação baseada em desvio do próprio baseline.

Referência: Eixo 6 do REFERENCIA_MOTOR_CALIBRACAO_E_APRENDIZADO.MD
"""

import os
import json
import math
import time
from typing import Dict, Any, List, Optional, Tuple


class MetricDistribution:
    """Distribuição estatística contínua de uma métrica biomecânica (média e desvio padrão via Welford)."""

    def __init__(self, count: int = 0, mean: float = 0.0, m2: float = 0.0):
        self.count = int(count)
        self.mean = float(mean)
        self.m2 = float(m2)

    @property
    def std_dev(self) -> float:
        if self.count < 2:
            return 1.0  # desvio neutro inicial
        return float(math.sqrt(max(1e-6, self.m2 / (self.count - 1))))

    def update(self, value: float) -> None:
        """Atualização incremental online de média e variância pelo algoritmo de Welford."""
        val = float(value)
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.m2 += delta * delta2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "m2": round(self.m2, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricDistribution":
        if not data:
            return cls()
        return cls(
            count=data.get("count", 0),
            mean=data.get("mean", 0.0),
            m2=data.get("m2", 0.0),
        )


class KinestheticBaselineModel:
    """
    Modelo Cinestésico Individual de um praticante de Kendo (Eixo 6.1).
    Aprende o baseline biomecânico do atleta (postura em repouso, Fumikomi,
    extensão de braço, cadência e inclinação) e mede desvios em relação a si próprio.
    """

    def __init__(
        self,
        kenshi_id: str,
        display_name: str,
        created_at: Optional[str] = None,
        last_updated_at: Optional[str] = None,
        sessions_count: int = 0,
        strikes_count: int = 0,
        metrics: Optional[Dict[str, MetricDistribution]] = None,
        strike_extensions: Optional[Dict[str, MetricDistribution]] = None,
    ):
        self.kenshi_id = str(kenshi_id).strip()
        self.display_name = str(display_name or kenshi_id).strip()
        self.created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.last_updated_at = last_updated_at or self.created_at
        self.sessions_count = int(sessions_count)
        self.strikes_count = int(strikes_count)

        # Métricas universais do praticante
        self.metrics: Dict[str, MetricDistribution] = metrics or {
            "posture_resting_angle_deg": MetricDistribution(),  # Chudan Shisei
            "fumikomi_window_ms": MetricDistribution(),         # Sincronismo pé/corte
            "cadence_cpm": MetricDistribution(),                # Cadência de treino
            "spine_tilt_strike_deg": MetricDistribution(),      # Inclinação no corte
            "zanshin_duration_s": MetricDistribution(),         # Duração do Zanshin
        }

        # Extensão típica de braço por golpe (MEN, KOTE, DO, TSUKI)
        self.strike_extensions: Dict[str, MetricDistribution] = strike_extensions or {
            "MEN": MetricDistribution(),
            "KOTE": MetricDistribution(),
            "DO": MetricDistribution(),
            "TSUKI": MetricDistribution(),
        }

    def update_from_strike(
        self,
        strike_type: str,
        spine_tilt_deg: Optional[float] = None,
        fumikomi_offset_ms: Optional[float] = None,
        elbow_extension_deg: Optional[float] = None,
        zanshin_duration_s: Optional[float] = None,
    ) -> None:
        """Registra telemetria de um golpe individual executado pelo praticante."""
        self.strikes_count += 1
        self.last_updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if spine_tilt_deg is not None:
            self.metrics["spine_tilt_strike_deg"].update(spine_tilt_deg)

        if fumikomi_offset_ms is not None:
            self.metrics["fumikomi_window_ms"].update(fumikomi_offset_ms)

        if zanshin_duration_s is not None:
            self.metrics["zanshin_duration_s"].update(zanshin_duration_s)

        st_upper = (strike_type or "MEN").upper()
        if elbow_extension_deg is not None:
            if st_upper not in self.strike_extensions:
                self.strike_extensions[st_upper] = MetricDistribution()
            self.strike_extensions[st_upper].update(elbow_extension_deg)

    def update_from_training_session(
        self,
        cadence_cpm: Optional[float] = None,
        resting_posture_angle: Optional[float] = None,
        mean_spine_tilt: Optional[float] = None,
        mean_fumikomi_ms: Optional[float] = None,
    ) -> None:
        """Registra métricas agregadas de uma sessão de treinamento (Keiko/Dojo)."""
        self.sessions_count += 1
        self.last_updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if cadence_cpm is not None and cadence_cpm > 0:
            self.metrics["cadence_cpm"].update(cadence_cpm)

        if resting_posture_angle is not None:
            self.metrics["posture_resting_angle_deg"].update(resting_posture_angle)

        if mean_spine_tilt is not None:
            self.metrics["spine_tilt_strike_deg"].update(mean_spine_tilt)

        if mean_fumikomi_ms is not None:
            self.metrics["fumikomi_window_ms"].update(mean_fumikomi_ms)

    def evaluate_strike_against_baseline(
        self,
        strike_type: str,
        observed_spine_tilt_deg: Optional[float] = None,
        observed_fumikomi_ms: Optional[float] = None,
        observed_elbow_extension_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Avalia o golpe medindo o desvio do próprio baseline cinestésico do praticante.
        Gera diagnósticos personalizados humanizados.
        """
        evaluations: Dict[str, Any] = {
            "kenshi_id": self.kenshi_id,
            "display_name": self.display_name,
            "has_sufficient_history": self.strikes_count >= 3 or self.sessions_count >= 1,
            "deviations": {},
            "insights": [],
        }

        # 1. Inclinação da Coluna (Spine Tilt)
        if observed_spine_tilt_deg is not None and self.metrics["spine_tilt_strike_deg"].count >= 2:
            base_mean = self.metrics["spine_tilt_strike_deg"].mean
            base_std = self.metrics["spine_tilt_strike_deg"].std_dev
            diff = observed_spine_tilt_deg - base_mean
            z_score = diff / base_std if base_std > 0 else 0.0

            evaluations["deviations"]["spine_tilt"] = {
                "observed": round(observed_spine_tilt_deg, 2),
                "baseline_mean": round(base_mean, 2),
                "diff": round(diff, 2),
                "z_score": round(z_score, 2),
            }

            if diff > 5.0 and z_score > 1.2:
                evaluations["insights"].append(
                    f"Executou com {diff:.1f}° a mais de inclinação do que sua média habitual ({observed_spine_tilt_deg:.1f}° vs {base_mean:.1f}° habitual)."
                )
            elif diff < -5.0 and z_score < -1.2:
                evaluations["insights"].append(
                    f"Excelente postura mais ereta: {abs(diff):.1f}° menos inclinada que sua média habitual."
                )

        # 2. Sincronismo do Fumikomi
        if observed_fumikomi_ms is not None and self.metrics["fumikomi_window_ms"].count >= 2:
            base_mean_fumi = self.metrics["fumikomi_window_ms"].mean
            base_std_fumi = self.metrics["fumikomi_window_ms"].std_dev
            diff_fumi = observed_fumikomi_ms - base_mean_fumi
            z_fumi = diff_fumi / base_std_fumi if base_std_fumi > 0 else 0.0

            evaluations["deviations"]["fumikomi_offset"] = {
                "observed_ms": round(observed_fumikomi_ms, 1),
                "baseline_mean_ms": round(base_mean_fumi, 1),
                "diff_ms": round(diff_fumi, 1),
                "z_score": round(z_fumi, 2),
            }

            if abs(diff_fumi) > 25.0 and abs(z_fumi) > 1.5:
                direction = "atrasado" if diff_fumi > 0 else "adiantado"
                evaluations["insights"].append(
                    f"Fumikomi esteve {abs(diff_fumi):.0f}ms mais {direction} que sua janela habitual ({observed_fumikomi_ms:.0f}ms vs {base_mean_fumi:.0f}ms)."
                )

        # 3. Extensão de Braço por Tipo de Golpe
        st_upper = (strike_type or "MEN").upper()
        if observed_elbow_extension_deg is not None and st_upper in self.strike_extensions:
            ext_dist = self.strike_extensions[st_upper]
            if ext_dist.count >= 2:
                diff_ext = observed_elbow_extension_deg - ext_dist.mean
                z_ext = diff_ext / ext_dist.std_dev if ext_dist.std_dev > 0 else 0.0

                evaluations["deviations"]["elbow_extension"] = {
                    "strike_type": st_upper,
                    "observed_deg": round(observed_elbow_extension_deg, 1),
                    "baseline_mean_deg": round(ext_dist.mean, 1),
                    "diff_deg": round(diff_ext, 1),
                    "z_score": round(z_ext, 2),
                }

                if diff_ext < -10.0 and z_ext < -1.5:
                    evaluations["insights"].append(
                        f"Extensão do braço no {st_upper} esteve {abs(diff_ext):.1f}° mais encolhida que o habitual ({observed_elbow_extension_deg:.1f}° vs {ext_dist.mean:.1f}°)."
                    )

        return evaluations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kenshi_id": self.kenshi_id,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "sessions_count": self.sessions_count,
            "strikes_count": self.strikes_count,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "strike_extensions": {k: v.to_dict() for k, v in self.strike_extensions.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KinestheticBaselineModel":
        raw_metrics = data.get("metrics", {})
        metrics = {k: MetricDistribution.from_dict(v) for k, v in raw_metrics.items()}
        for k in ["posture_resting_angle_deg", "fumikomi_window_ms", "cadence_cpm", "spine_tilt_strike_deg", "zanshin_duration_s"]:
            if k not in metrics:
                metrics[k] = MetricDistribution()

        raw_ext = data.get("strike_extensions", {})
        strike_extensions = {k: MetricDistribution.from_dict(v) for k, v in raw_ext.items()}
        for k in ["MEN", "KOTE", "DO", "TSUKI"]:
            if k not in strike_extensions:
                strike_extensions[k] = MetricDistribution()

        return cls(
            kenshi_id=data.get("kenshi_id", "KENSHI_DEFAULT"),
            display_name=data.get("display_name", data.get("kenshi_id", "Kenshi")),
            created_at=data.get("created_at"),
            last_updated_at=data.get("last_updated_at"),
            sessions_count=data.get("sessions_count", 0),
            strikes_count=data.get("strikes_count", 0),
            metrics=metrics,
            strike_extensions=strike_extensions,
        )


class KinestheticProfileManager:
    """
    Gerenciador Central de Perfis Cinestésicos Individuais de Praticantes (Eixo 6).
    Persiste e gerencia o conhecimento biomecânico individual em `data/kenshi_baselines.json`.
    """

    def __init__(self, storage_path: str = "data/kenshi_baselines.json"):
        self.storage_path = storage_path
        self.profiles: Dict[str, KinestheticBaselineModel] = {}
        self.load_baselines()

    def get_or_create_profile(self, kenshi_id: str, display_name: Optional[str] = None) -> KinestheticBaselineModel:
        """Recupera um perfil existente ou cria um novo para o Kendoca."""
        k_id = str(kenshi_id or "KENSHI_SOLO").strip()
        if k_id not in self.profiles:
            disp = display_name or k_id
            self.profiles[k_id] = KinestheticBaselineModel(kenshi_id=k_id, display_name=disp)
            self.save_baselines()
        else:
            if display_name and display_name.strip() and display_name != self.profiles[k_id].display_name:
                self.profiles[k_id].display_name = display_name.strip()
        return self.profiles[k_id]

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Lista todos os perfis cinestésicos cadastrados com resumo."""
        result = []
        for k_id, p in self.profiles.items():
            result.append({
                "kenshi_id": k_id,
                "display_name": p.display_name,
                "sessions_count": p.sessions_count,
                "strikes_count": p.strikes_count,
                "last_updated_at": p.last_updated_at,
                "has_baseline": (p.strikes_count >= 3 or p.sessions_count >= 1),
                "mean_spine_tilt": round(p.metrics["spine_tilt_strike_deg"].mean, 1) if p.metrics["spine_tilt_strike_deg"].count > 0 else None,
                "mean_cadence_cpm": round(p.metrics["cadence_cpm"].mean, 1) if p.metrics["cadence_cpm"].count > 0 else None,
                "mean_fumikomi_ms": round(p.metrics["fumikomi_window_ms"].mean, 1) if p.metrics["fumikomi_window_ms"].count > 0 else None,
            })
        return result

    def record_strike_event(
        self,
        kenshi_id: str,
        strike_type: str,
        spine_tilt_deg: Optional[float] = None,
        fumikomi_offset_ms: Optional[float] = None,
        elbow_extension_deg: Optional[float] = None,
        display_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Registra um golpe e retorna avaliação baseada em desvio do baseline."""
        profile = self.get_or_create_profile(kenshi_id, display_name=display_name)
        # Primeiro avalia contra o baseline existente antes de incorporar o golpe atual
        eval_result = profile.evaluate_strike_against_baseline(
            strike_type=strike_type,
            observed_spine_tilt_deg=spine_tilt_deg,
            observed_fumikomi_ms=fumikomi_offset_ms,
            observed_elbow_extension_deg=elbow_extension_deg,
        )
        # Atualiza o baseline com o novo golpe
        profile.update_from_strike(
            strike_type=strike_type,
            spine_tilt_deg=spine_tilt_deg,
            fumikomi_offset_ms=fumikomi_offset_ms,
            elbow_extension_deg=elbow_extension_deg,
        )
        self.save_baselines()
        return eval_result

    def record_training_session(
        self,
        kenshi_id: str,
        display_name: Optional[str] = None,
        cadence_cpm: Optional[float] = None,
        resting_posture_angle: Optional[float] = None,
        mean_spine_tilt: Optional[float] = None,
        mean_fumikomi_ms: Optional[float] = None,
    ) -> KinestheticBaselineModel:
        """Registra dados agregados de uma sessão de treinamento e persiste."""
        profile = self.get_or_create_profile(kenshi_id, display_name=display_name)
        profile.update_from_training_session(
            cadence_cpm=cadence_cpm,
            resting_posture_angle=resting_posture_angle,
            mean_spine_tilt=mean_spine_tilt,
            mean_fumikomi_ms=mean_fumikomi_ms,
        )
        self.save_baselines()
        return profile

    def save_baselines(self) -> bool:
        """Persiste todos os baselines cinestésicos no arquivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            payload = {
                "schema_version": "1.0",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "profiles": {k_id: p.to_dict() for k_id, p in self.profiles.items()},
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def load_baselines(self) -> bool:
        """Carrega os baselines salvos do disco."""
        if not os.path.exists(self.storage_path):
            return False
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_profiles = data.get("profiles", {})
            self.profiles = {k_id: KinestheticBaselineModel.from_dict(v) for k_id, v in raw_profiles.items()}
            return True
        except Exception:
            return False


class ProfileWarmStartManager:
    """
    Gerenciador de Transferência de Conhecimento entre Perfis (Warm Start - Eixo 6.2).
    Permite derivar novos perfis a partir de perfis calibrados existentes, herdando
    pesos de Ki-Ken-Tai-Ichi, Platt Scaling e acelerando a convergência matemática.
    """

    @staticmethod
    def derive_profile(
        source_profile_config: Dict[str, Any],
        new_profile_key: str,
        direction: str = "more_strict",
        adjustment_factor: float = 1.05,
        new_profile_name: Optional[str] = None,
        new_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deriva um novo perfil a partir de um perfil pai (Warm Start).
        
        Args:
            source_profile_config: Dicionário de configuração do perfil base.
            new_profile_key: Identificador da nova chave (ex: 'campeonato_rigido').
            direction: 'more_strict' (eleva exigências), 'more_permissive' (reduz exigências) ou 'neutral'.
            adjustment_factor: Fator multiplicativo para limiares (default: 1.05 = +5%).
            new_profile_name: Nome legível para a UI.
            new_description: Descrição da finalidade do perfil derivado.
            
        Returns:
            Dict com a nova configuração pronta para persistência em calibration_profiles.json.
        """
        import copy
        derived = copy.deepcopy(source_profile_config)

        factor = max(1.01, float(adjustment_factor))
        if direction == "more_strict":
            mult = factor
        elif direction == "more_permissive":
            mult = 1.0 / factor
        else:
            mult = 1.0

        # 1. Ajuste do Limiar Mínimo Global (clamped em [0.40, 0.92])
        cur_min_total = float(derived.get("min_total_score", 0.70))
        derived["min_total_score"] = round(min(0.92, max(0.40, cur_min_total * mult)), 2)

        # 2. Ajuste dos Sub-Limiares por Critério (clamped em [0.20, 0.90])
        cur_sub = derived.get("sub_thresholds", {})
        new_sub = {}
        for k, v in cur_sub.items():
            new_sub[k] = round(min(0.90, max(0.20, float(v) * mult)), 2)
        derived["sub_thresholds"] = new_sub

        # 3. Herança estrita de Pesos de Ki-Ken-Tai-Ichi (Global e por Golpe)
        # Os pesos são mantidos exatamente iguais aos do perfil pai para Warm Start bayesiano
        derived["weights"] = copy.deepcopy(source_profile_config.get("weights", {
            "target_impact": 0.40,
            "fumikomi_sync": 0.25,
            "posture": 0.20,
            "zanshin": 0.15,
        }))

        if "weights_by_strike_type" in source_profile_config:
            derived["weights_by_strike_type"] = copy.deepcopy(source_profile_config["weights_by_strike_type"])

        # 4. Metadados de Linhagem e Warm Start
        derived["name"] = new_profile_name or f"Perfil Derivado ({new_profile_key})"
        derived["description"] = new_description or f"Perfil derivado via Warm Start (Eixo 6.2) com ajuste '{direction}'."
        derived["derived_from"] = source_profile_config.get("name", "Perfil Base")
        derived["warm_started"] = True
        derived["warm_start_direction"] = direction
        derived["derived_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return derived
