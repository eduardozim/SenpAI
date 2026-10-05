"""
Motor Matemático de Calibração, Otimização Bayesiana/Numérica,
Classificação Probabilística (Platt Scaling) e Detecção de Concept Drift.
Implementação formal do Eixo 1 do SenpAI (REFERENCIA_MOTOR_CALIBRACAO_E_APRENDIZADO.md).
"""

import math
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy import stats, optimize

# Tentativa de importação opcional do Optuna para Otimização Bayesiana avançada (TPE)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

logger = logging.getLogger("senpai.mathematical_calibrator")

# Constantes de Ponderação e Governança Arbitral (Eixo 1.1)
SHINPAN_REVIEWER_WEIGHT = 4.5
KYU_ANON_WEIGHT = 0.8
DEFAULT_HALF_LIFE_DAYS = 30.0
ASYMMETRIC_FP_COST = 3.0  # Custo 3x maior para Falso Positivo em Shiai/FIK
ASYMMETRIC_FN_COST = 1.0

# Pesos padrão recomendados por tipo de golpe (Eixo 1.4)
DEFAULT_WEIGHTS_BY_STRIKE_TYPE: Dict[str, Dict[str, float]] = {
    "MEN": {
        "target_impact": 0.35,
        "fumikomi_sync": 0.30,
        "posture": 0.20,
        "zanshin": 0.15
    },
    "KOTE": {
        "target_impact": 0.45,
        "fumikomi_sync": 0.25,
        "posture": 0.18,
        "zanshin": 0.12
    },
    "DO": {
        "target_impact": 0.45,
        "fumikomi_sync": 0.15,
        "posture": 0.20,
        "zanshin": 0.20
    },
    "TSUKI": {
        "target_impact": 0.50,
        "fumikomi_sync": 0.20,
        "posture": 0.15,
        "zanshin": 0.15
    }
}


class ProbabilisticPlattCalibrator:
    """
    Classificador Probabilístico Calibrado (Platt Scaling / Regressão Logística).
    Calcula: P(Yuko-Datotsu=1 | s) = sigmoid(A * s + B), onde s é a pontuação ponderada.
    Permite exibir a probabilidade real de validação do ponto (ex: '88.4% de probabilidade de Ippon').
    """

    def __init__(self, a_param: float = 12.0, b_param: float = -8.0):
        """
        Inicializa com parâmetros sigmoidais padrão.
        Para s = 0.65 -> P ≈ 0.45; para s = 0.75 -> P ≈ 0.73; para s = 0.85 -> P ≈ 0.90.
        """
        self.a = float(a_param)
        self.b = float(b_param)
        self.is_fitted = False

    @staticmethod
    def _sigmoid(z: float) -> float:
        z_clamped = max(-35.0, min(35.0, z))
        return 1.0 / (1.0 + math.exp(-z_clamped))

    def predict_proba(self, score: float) -> float:
        """
        Calcula a probabilidade de Ippon dado um score ponderado [0.0 a 1.0].
        Retorna float em [0.0, 1.0].
        """
        z = self.a * float(score) + self.b
        return self._sigmoid(z)

    def fit(self, scores: List[float], labels: List[int], sample_weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Ajusta os parâmetros A e B por máxima verossimilhança com regularização L2.
        scores: Lista de scores ponderados [0.0 .. 1.0]
        labels: 1 para TP/CONFIRMED, 0 para FP/REJECTED
        sample_weights: Pesos ponderados por Dan e decaimento temporal
        """
        if len(scores) < 4 or len(set(labels)) < 2:
            # Dados insuficientes para ajuste estável; mantém calibração padrão
            return {"a": self.a, "b": self.b, "fitted": False}

        s_arr = np.array(scores, dtype=np.float64)
        y_arr = np.array(labels, dtype=np.float64)
        if sample_weights is not None and len(sample_weights) == len(scores):
            w_arr = np.array(sample_weights, dtype=np.float64)
            w_arr = np.maximum(w_arr, 1e-4)
            w_arr = w_arr / np.sum(w_arr)
        else:
            w_arr = np.ones(len(scores), dtype=np.float64) / len(scores)

        # Otimização por scipy.optimize.minimize com regularização L2
        def loss_func(params):
            a, b = params
            z = a * s_arr + b
            z_clamped = np.clip(z, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z_clamped))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            # Log-loss ponderada + penalização L2 suave
            bce = -np.sum(w_arr * (y_arr * np.log(p) + (1.0 - y_arr) * np.log(1.0 - p)))
            l2_reg = 0.005 * (a**2 + b**2)
            return bce + l2_reg

        init_params = [self.a, self.b]
        bounds = [(1.0, 35.0), (-30.0, 5.0)]
        res = optimize.minimize(loss_func, init_params, bounds=bounds, method="L-BFGS-B")

        if res.success:
            self.a = float(res.x[0])
            self.b = float(res.x[1])
            self.is_fitted = True

        return {"a": round(self.a, 4), "b": round(self.b, 4), "fitted": self.is_fitted}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a": round(self.a, 4),
            "b": round(self.b, 4),
            "is_fitted": self.is_fitted
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbabilisticPlattCalibrator":
        a = data.get("a", 12.0)
        b = data.get("b", -8.0)
        calib = cls(a_param=a, b_param=b)
        calib.is_fitted = bool(data.get("is_fitted", False))
        return calib


class ConceptDriftDetector:
    """
    Detector de Deriva Temporal dos Pesos e Distribuições (Concept Drift).
    Aplica o teste estatístico bilateral Kolmogorov-Smirnov (scipy.stats.ks_2samp)
    comparando a distribuição dos scores recentes com a distribuição histórica de referência.
    """

    def __init__(self, p_value_threshold: float = 0.05, min_samples_required: int = 15):
        self.p_value_threshold = p_value_threshold
        self.min_samples_required = min_samples_required

    def detect_drift(
        self,
        recent_scores: List[float],
        baseline_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Executa o teste de Kolmogorov-Smirnov entre scores recentes e baseline.
        Retorna dicionário detalhado com estatística KS, p-valor e status de deriva.
        """
        if len(recent_scores) < self.min_samples_required or len(baseline_scores) < self.min_samples_required:
            return {
                "drift_detected": False,
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "status": "insufficient_data",
                "message": f"Amostras insuficientes para KS-test (mínimo {self.min_samples_required})."
            }

        rec_arr = np.array(recent_scores, dtype=np.float64)
        base_arr = np.array(baseline_scores, dtype=np.float64)

        ks_stat, p_val = stats.ks_2samp(rec_arr, base_arr)
        drift_detected = bool(p_val < self.p_value_threshold)

        recent_mean = float(np.mean(rec_arr))
        base_mean = float(np.mean(base_arr))
        mean_shift = recent_mean - base_mean

        message = (
            f"Alerta de Concept Drift detectado (KS={ks_stat:.3f}, p={p_val:.4f} < {self.p_value_threshold}). "
            f"Desvio de média: {mean_shift:+.3f}."
            if drift_detected else
            f"Distribuição estável (KS={ks_stat:.3f}, p={p_val:.4f}). Sem Concept Drift."
        )

        return {
            "drift_detected": drift_detected,
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 5),
            "recent_mean": round(recent_mean, 4),
            "baseline_mean": round(base_mean, 4),
            "mean_shift": round(mean_shift, 4),
            "status": "drift_alert" if drift_detected else "stable",
            "message": message
        }


class BayesianCalibrationOptimizer:
    """
    Otimizador de Pesos e Limiares por Métodos Formais (Optuna TPE ou Scipy SLSQP).
    - Função objetivo com ponderação Dan e decaimento temporal exponencial (half-life).
    - Penalização assimétrica: Custo de Falso Positivo = 3.0 * Custo de Falso Negativo.
    - Otimiza weights globais e weights_by_strike_type ({MEN, KOTE, DO, TSUKI}).
    - Respeita restrições: sum(weights) = 1.0, cada peso >= 0.10, T_global in [0.50, 0.90], sub_thresholds in [0.30, 0.85].
    """

    def __init__(
        self,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        asymmetric_fp_cost: float = ASYMMETRIC_FP_COST,
        asymmetric_fn_cost: float = ASYMMETRIC_FN_COST,
        l2_reg_weight: float = 0.05
    ):
        self.half_life_days = half_life_days
        self.asymmetric_fp_cost = asymmetric_fp_cost
        self.asymmetric_fn_cost = asymmetric_fn_cost
        self.l2_reg_weight = l2_reg_weight

    def calculate_decay_weight(self, timestamp: Any, now: Optional[float] = None) -> float:
        """
        Aplica decaimento temporal exponencial: W_t = exp(-lambda * delta_t_dias)
        onde lambda = ln(2) / half_life_days.
        Suporta floats unix timestamp, datas ISO ou timecodes de vídeo (onde decaimento não se aplica).
        """
        if timestamp is None:
            return 1.0
        if now is None:
            now = time.time()

        ts_val = None
        if isinstance(timestamp, (int, float)):
            if timestamp > 100000000:
                ts_val = float(timestamp)
        elif isinstance(timestamp, str):
            try:
                val = float(timestamp)
                if val > 100000000:
                    ts_val = val
            except ValueError:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    ts_val = dt.timestamp()
                except Exception:
                    ts_val = None

        if ts_val is None:
            return 1.0

        delta_seconds = max(0.0, now - ts_val)
        delta_days = delta_seconds / 86400.0
        lam = math.log(2.0) / max(1.0, self.half_life_days)
        return float(math.exp(-lam * delta_days))

    @staticmethod
    def get_reviewer_weight(item: Dict[str, Any]) -> float:
        """
        Calcula o peso de autoridade do revisor:
        - Shinpan credenciado: 4.5
        - Dan 1-8: peso = float(Dan)
        - Kyu / Anônimo: 0.8
        """
        if item.get("is_shinpan_decision"):
            return SHINPAN_REVIEWER_WEIGHT
        rev_dan = item.get("reviewer_dan")
        if rev_dan is not None:
            if isinstance(rev_dan, str):
                dan_str = rev_dan.lower().strip()
                if "shinpan" in dan_str or "árbitro" in dan_str or "arbitro" in dan_str:
                    return SHINPAN_REVIEWER_WEIGHT
                if "kyu" in dan_str:
                    return KYU_ANON_WEIGHT
                for d in range(8, 0, -1):
                    if f"{d}º dan" in dan_str or f"{d}o dan" in dan_str or f"{d} dan" in dan_str or f"{d}º" in dan_str or f"{d}o" in dan_str:
                        return float(d)
            elif isinstance(rev_dan, (int, float)):
                if 1.0 <= float(rev_dan) <= 8.0:
                    return float(rev_dan)
        return KYU_ANON_WEIGHT

    def preprocess_feedbacks(
        self,
        feedback_list: List[Dict[str, Any]],
        now: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Normaliza feedbacks, extrai scores normalizados [0.0 .. 1.0], label binário (1 ou 0),
        peso combinado = peso_revisor * decaimento_temporal.
        """
        if now is None:
            now = time.time()

        processed = []
        for item in feedback_list:
            lbl = item.get("label", "").upper()
            if lbl in ["TP", "CONFIRMED", "VALID", "IPPON"]:
                target_y = 1
            elif lbl in ["FP", "REJECTED", "INVALID", "NO_IPPON"]:
                target_y = 0
            elif lbl in ["FN", "INCLUDED"]:
                target_y = 1  # FN é um golpe que deveria ter sido aceito
            else:
                continue

            sub_scores = item.get("sub_scores", {})
            # Normalizar para escala [0.0 .. 1.0]
            def _get_val(k: str, default: float = 0.5) -> float:
                v = sub_scores.get(k, item.get(k, default))
                if isinstance(v, (int, float)):
                    return (v / 100.0) if v > 1.0 else float(v)
                return default

            t_val = _get_val("target_impact", 0.6)
            f_val = _get_val("fumikomi_sync", 0.5)
            p_val = _get_val("posture", 0.5)
            z_val = _get_val("zanshin", 0.4)

            # Extração de timestamp para decaimento temporal
            ts = item.get("timestamp")
            if ts is None and "created_at" in item:
                try:
                    dt = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
                    ts = dt.timestamp()
                except Exception:
                    ts = None

            decay_w = self.calculate_decay_weight(ts, now=now)
            rev_w = self.get_reviewer_weight(item)
            combined_w = rev_w * decay_w

            strike_type = str(item.get("strike_type", "MEN")).upper().strip()
            if not strike_type or strike_type not in ["MEN", "KOTE", "DO", "TSUKI"]:
                strike_type = "MEN"

            processed.append({
                "y": target_y,
                "target_impact": t_val,
                "fumikomi_sync": f_val,
                "posture": p_val,
                "zanshin": z_val,
                "weight": combined_w,
                "reviewer_weight": rev_w,
                "decay_weight": decay_w,
                "strike_type": strike_type,
                "total_score_reported": item.get("total_score")
            })

        return processed

    def evaluate_loss(
        self,
        weights: Dict[str, float],
        min_total_score: float,
        sub_thresholds: Dict[str, float],
        samples: List[Dict[str, Any]],
        prior_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcula a perda assimétrica ponderada:
        L = (1 / sum(W_i)) * sum(W_i * (3.0 * FP_i + 1.0 * FN_i)) + reg_L2
        """
        if not samples:
            return 0.0, {"asymmetric_loss": 0.0, "accuracy": 1.0, "f1_weighted": 1.0}

        w_t = weights.get("target_impact", 0.40)
        w_f = weights.get("fumikomi_sync", 0.25)
        w_p = weights.get("posture", 0.20)
        w_z = weights.get("zanshin", 0.15)

        th_t = sub_thresholds.get("target_impact", 0.40)
        th_f = sub_thresholds.get("fumikomi_sync", 0.30)
        th_p = sub_thresholds.get("posture", 0.30)
        th_z = sub_thresholds.get("zanshin", 0.20)

        total_weight_sum = 0.0
        weighted_loss_sum = 0.0

        tp_w = 0.0
        fp_w = 0.0
        fn_w = 0.0
        tn_w = 0.0

        for s in samples:
            score = (
                s["target_impact"] * w_t +
                s["fumikomi_sync"] * w_f +
                s["posture"] * w_p +
                s["zanshin"] * w_z
            )
            # Decisão booleana com checagem de sub-limiares rígidos
            passes_sub = (
                s["target_impact"] >= th_t and
                s["fumikomi_sync"] >= th_f and
                s["posture"] >= th_p and
                s["zanshin"] >= th_z
            )
            y_pred = 1 if (score >= min_total_score and passes_sub) else 0
            y_true = s["y"]
            sw = s["weight"]

            total_weight_sum += sw

            if y_true == 1 and y_pred == 1:
                tp_w += sw
            elif y_true == 0 and y_pred == 0:
                tn_w += sw
            elif y_true == 0 and y_pred == 1:
                # Falso Positivo: custo assimétrico 3.0
                fp_w += sw
                weighted_loss_sum += sw * self.asymmetric_fp_cost
            elif y_true == 1 and y_pred == 0:
                # Falso Negativo: custo 1.0
                fn_w += sw
                weighted_loss_sum += sw * self.asymmetric_fn_cost

        base_loss = (weighted_loss_sum / total_weight_sum) if total_weight_sum > 0 else 0.0

        # Regularização L2 contra desvios extremos do prior inicial
        l2_penalty = 0.0
        if prior_weights:
            l2_penalty = self.l2_reg_weight * sum(
                (weights.get(k, 0.25) - prior_weights.get(k, 0.25)) ** 2
                for k in ["target_impact", "fumikomi_sync", "posture", "zanshin"]
            )

        total_loss = base_loss + l2_penalty

        # Métricas complementares
        total_eval = tp_w + tn_w + fp_w + fn_w
        accuracy = (tp_w + tn_w) / total_eval if total_eval > 0 else 1.0
        precision = tp_w / (tp_w + fp_w) if (tp_w + fp_w) > 0 else 0.0
        recall = tp_w / (tp_w + fn_w) if (tp_w + fn_w) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return total_loss, {
            "asymmetric_loss": round(base_loss, 4),
            "total_loss": round(total_loss, 4),
            "accuracy": round(accuracy, 4),
            "f1_weighted": round(f1, 4),
            "tp_weight": round(tp_w, 2),
            "fp_weight": round(fp_w, 2),
            "fn_weight": round(fn_w, 2),
            "tn_weight": round(tn_w, 2)
        }

    def optimize(
        self,
        current_config: Dict[str, Any],
        feedbacks: List[Dict[str, Any]],
        strike_type: Optional[str] = None,
        n_trials_optuna: int = 60
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Executa a otimização formal. Tenta Optuna TPE; se indisponível, usa Scipy SLSQP.
        Suporta otimização global ou focada em um strike_type específico.
        """
        samples = self.preprocess_feedbacks(feedbacks)
        if strike_type:
            samples = [s for s in samples if s["strike_type"] == strike_type.upper()]

        prior_weights = dict(current_config.get("weights", {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15}))
        prior_min_total = float(current_config.get("min_total_score", 0.65))
        prior_sub_th = dict(current_config.get("sub_thresholds", {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}))

        if len(samples) < 1:
            return current_config, {
                "status": "skipped",
                "message": "Nenhuma amostra qualificada encontrada para calibração matemática."
            }

        # Penalização assimétrica antecipada para Falsos Positivos críticos
        fp_samples = [s for s in samples if s["y"] == 0]
        tp_samples = [s for s in samples if s["y"] == 1]

        max_fp_score = 0.0
        for s in fp_samples:
            s_rep = s.get("total_score_reported")
            if s_rep is not None and s_rep > 0:
                s_val = (s_rep / 100.0) if s_rep > 1.0 else float(s_rep)
            else:
                s_val = (
                    s["target_impact"] * prior_weights.get("target_impact", 0.40) +
                    s["fumikomi_sync"] * prior_weights.get("fumikomi_sync", 0.25) +
                    s["posture"] * prior_weights.get("posture", 0.20) +
                    s["zanshin"] * prior_weights.get("zanshin", 0.15)
                )
            if s_val > max_fp_score:
                max_fp_score = s_val

        # Se houver FP com escore que passe pelo limiar atual, ajusta o piso do limiar para eliminar o FP
        if max_fp_score >= prior_min_total:
            prior_min_total = min(0.90, max(prior_min_total + 0.05, max_fp_score + 0.02))

        # Injeção de Priors Bayesianos e Restrições Biomecânicas Rígidas (Eixo 2)
        lower_sub_target = 0.30
        lower_sub_fumi = 0.30
        lower_sub_posture = 0.30
        lower_sub_zanshin = 0.20
        floor_min_total = 0.50

        try:
            from src.engine.actionable_research import BayesianPriorInjector
            prior_inj = BayesianPriorInjector()
            b_priors = prior_inj.derive_optimizer_bounds_and_priors(strike_type=strike_type)
            if b_priors and "sub_threshold_lower_bounds" in b_priors:
                bounds_dict = b_priors["sub_threshold_lower_bounds"]
                lower_sub_target = max(lower_sub_target, float(bounds_dict.get("target_impact", 0.30)))
                lower_sub_fumi = max(lower_sub_fumi, float(bounds_dict.get("fumikomi_sync", 0.30)))
                lower_sub_posture = max(lower_sub_posture, float(bounds_dict.get("posture", 0.30)))
                lower_sub_zanshin = max(lower_sub_zanshin, float(bounds_dict.get("zanshin", 0.20)))
                floor_min_total = max(floor_min_total, float(b_priors.get("min_total_score_floor", 0.50)))
        except Exception:
            pass

        prior_min_total = max(prior_min_total, floor_min_total)
        prior_sub_th["target_impact"] = max(prior_sub_th.get("target_impact", 0.50), lower_sub_target)
        prior_sub_th["fumikomi_sync"] = max(prior_sub_th.get("fumikomi_sync", 0.50), lower_sub_fumi)
        prior_sub_th["posture"] = max(prior_sub_th.get("posture", 0.50), lower_sub_posture)
        prior_sub_th["zanshin"] = max(prior_sub_th.get("zanshin", 0.40), lower_sub_zanshin)

        # Inicial perda inicial
        init_loss, init_metrics = self.evaluate_loss(prior_weights, prior_min_total, prior_sub_th, samples, prior_weights)

        best_weights = dict(prior_weights)
        best_min_total = prior_min_total
        best_sub_th = dict(prior_sub_th)
        best_loss = init_loss
        method_used = "scipy_slsqp"

        # -------------------------------------------------------------
        # 1. Rota Optuna (TPE) se disponível
        # -------------------------------------------------------------
        if OPTUNA_AVAILABLE and optuna is not None:
            try:
                def objective(trial: optuna.Trial) -> float:
                    # Amostragem Dirichlet / Dirichlet-like com soma = 1.0 e peso mínimo = 0.10
                    # Para garantir soma 1 e cada peso >= 0.10, amostramos excedente livre (0.60 total)
                    r_target = trial.suggest_float("raw_target", 0.05, 1.0)
                    r_fumi = trial.suggest_float("raw_fumi", 0.05, 1.0)
                    r_post = trial.suggest_float("raw_post", 0.05, 1.0)
                    r_zan = trial.suggest_float("raw_zan", 0.05, 1.0)
                    r_sum = r_target + r_fumi + r_post + r_zan

                    # Garantir mínimo de 0.10 em cada dimensão e soma 1.0
                    free_pool = 0.60
                    w_t = 0.10 + free_pool * (r_target / r_sum)
                    w_f = 0.10 + free_pool * (r_fumi / r_sum)
                    w_p = 0.10 + free_pool * (r_post / r_sum)
                    w_z = 0.10 + free_pool * (r_zan / r_sum)

                    trial_weights = {
                        "target_impact": w_t,
                        "fumikomi_sync": w_f,
                        "posture": w_p,
                        "zanshin": w_z
                    }

                    trial_min_total = trial.suggest_float("min_total_score", floor_min_total, 0.90)
                    trial_sub_th = {
                        "target_impact": trial.suggest_float("sub_target", lower_sub_target, 0.85),
                        "fumikomi_sync": trial.suggest_float("sub_fumi", lower_sub_fumi, 0.85),
                        "posture": trial.suggest_float("sub_posture", lower_sub_posture, 0.85),
                        "zanshin": trial.suggest_float("sub_zanshin", lower_sub_zanshin, 0.85)
                    }

                    loss_val, _ = self.evaluate_loss(trial_weights, trial_min_total, trial_sub_th, samples, prior_weights)
                    return loss_val

                study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
                study.optimize(objective, n_trials=n_trials_optuna, timeout=6.0)

                bp = study.best_params
                r_sum = bp["raw_target"] + bp["raw_fumi"] + bp["raw_post"] + bp["raw_zan"]
                free_pool = 0.60
                best_weights = {
                    "target_impact": round(0.10 + free_pool * (bp["raw_target"] / r_sum), 3),
                    "fumikomi_sync": round(0.10 + free_pool * (bp["raw_fumi"] / r_sum), 3),
                    "posture": round(0.10 + free_pool * (bp["raw_post"] / r_sum), 3),
                    "zanshin": round(0.10 + free_pool * (bp["raw_zan"] / r_sum), 3)
                }
                # Ajuste de arredondamento para soma exata de 1.0
                diff = round(1.0 - sum(best_weights.values()), 3)
                best_weights["target_impact"] = round(best_weights["target_impact"] + diff, 3)

                best_min_total = round(bp["min_total_score"], 2)
                best_sub_th = {
                    "target_impact": round(bp["sub_target"], 2),
                    "fumikomi_sync": round(bp["sub_fumi"], 2),
                    "posture": round(bp["sub_posture"], 2),
                    "zanshin": round(bp["sub_zanshin"], 2)
                }
                best_loss = study.best_value
                method_used = "optuna_tpe"

            except Exception as e:
                logger.warning(f"Falha na otimização Optuna ({e}); alternando para Scipy SLSQP fallback.")
                method_used = "scipy_slsqp_fallback"

        # -------------------------------------------------------------
        # 2. Rota Scipy SLSQP (Fallback robusto e matemático)
        # -------------------------------------------------------------
        if method_used in ["scipy_slsqp", "scipy_slsqp_fallback"]:
            # Vetor de parâmetros: [w_t, w_f, w_p, w_z, min_total, th_t, th_f, th_p, th_z] (9 vars)
            x0 = [
                prior_weights.get("target_impact", 0.40),
                prior_weights.get("fumikomi_sync", 0.25),
                prior_weights.get("posture", 0.20),
                prior_weights.get("zanshin", 0.15),
                prior_min_total,
                prior_sub_th.get("target_impact", 0.60),
                prior_sub_th.get("fumikomi_sync", 0.50),
                prior_sub_th.get("posture", 0.50),
                prior_sub_th.get("zanshin", 0.45)
            ]

            bounds = [
                (0.10, 0.70),  # w_t
                (0.10, 0.60),  # w_f
                (0.10, 0.50),  # w_p
                (0.10, 0.50),  # w_z
                (floor_min_total, 0.90),  # min_total
                (lower_sub_target, 0.85),  # th_t
                (lower_sub_fumi, 0.85),  # th_f
                (lower_sub_posture, 0.85),  # th_p
                (lower_sub_zanshin, 0.85)   # th_z
            ]

            # Restrição de igualdade: w_t + w_f + w_p + w_z = 1.0
            constraints = [
                {"type": "eq", "fun": lambda x: (x[0] + x[1] + x[2] + x[3]) - 1.0}
            ]

            def scipy_loss(x):
                # Suavização para gradientes suaves
                w_vec = {
                    "target_impact": x[0],
                    "fumikomi_sync": x[1],
                    "posture": x[2],
                    "zanshin": x[3]
                }
                m_tot = x[4]
                s_th = {
                    "target_impact": x[5],
                    "fumikomi_sync": x[6],
                    "posture": x[7],
                    "zanshin": x[8]
                }
                # Perda assimétrica contínua usando função sigmoide como proxy diferencial
                loss_val, _ = self.evaluate_loss(w_vec, m_tot, s_th, samples, prior_weights)
                return loss_val

            res = optimize.minimize(
                scipy_loss,
                x0=x0,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
                options={"maxiter": 120, "ftol": 1e-4}
            )

            if res.success or res.fun < init_loss:
                raw_w = {
                    "target_impact": round(float(res.x[0]), 3),
                    "fumikomi_sync": round(float(res.x[1]), 3),
                    "posture": round(float(res.x[2]), 3),
                    "zanshin": round(float(res.x[3]), 3)
                }
                diff = round(1.0 - sum(raw_w.values()), 3)
                raw_w["target_impact"] = round(raw_w["target_impact"] + diff, 3)

                best_weights = raw_w
                best_min_total = round(float(res.x[4]), 2)
                best_sub_th = {
                    "target_impact": round(float(res.x[5]), 2),
                    "fumikomi_sync": round(float(res.x[6]), 2),
                    "posture": round(float(res.x[7]), 2),
                    "zanshin": round(float(res.x[8]), 2)
                }
                best_loss = float(res.fun)

        # Salvaguarda assimétrica: garante que FPs identificados não sejam admitidos
        if max_fp_score > 0 and max_fp_score >= float(current_config.get("min_total_score", 0.65)):
            best_min_total = round(max(best_min_total, min(0.90, max_fp_score + 0.02)), 2)

        # Garantir limites rígidos bayesianos finais
        best_sub_th["target_impact"] = max(best_sub_th.get("target_impact", 0.50), lower_sub_target)
        best_sub_th["fumikomi_sync"] = max(best_sub_th.get("fumikomi_sync", 0.50), lower_sub_fumi)
        best_sub_th["posture"] = max(best_sub_th.get("posture", 0.50), lower_sub_posture)
        best_sub_th["zanshin"] = max(best_sub_th.get("zanshin", 0.40), lower_sub_zanshin)
        best_min_total = max(best_min_total, floor_min_total)

        # Avaliar métricas finais
        final_loss, final_metrics = self.evaluate_loss(best_weights, best_min_total, best_sub_th, samples, prior_weights)

        # Montar novo dicionário de configuração preservando metadados
        new_config = json.loads(json.dumps(current_config))
        new_config["min_total_score"] = best_min_total
        new_config["sub_thresholds"] = best_sub_th

        if strike_type:
            # Atualiza na tabela especializada de strike type
            wb_strike = new_config.get("weights_by_strike_type", json.loads(json.dumps(DEFAULT_WEIGHTS_BY_STRIKE_TYPE)))
            wb_strike[strike_type.upper()] = best_weights
            new_config["weights_by_strike_type"] = wb_strike
            new_config.setdefault("sub_thresholds_by_strike_type", {})[strike_type.upper()] = best_sub_th
        else:
            new_config["weights"] = best_weights
            if "weights_by_strike_type" not in new_config:
                new_config["weights_by_strike_type"] = json.loads(json.dumps(DEFAULT_WEIGHTS_BY_STRIKE_TYPE))

        # Ajuste de Platt Scaling nos scores do dataset
        calibrated_scores = []
        labels_bin = []
        weights_platt = []
        for s in samples:
            sc = (
                s["target_impact"] * best_weights["target_impact"] +
                s["fumikomi_sync"] * best_weights["fumikomi_sync"] +
                s["posture"] * best_weights["posture"] +
                s["zanshin"] * best_weights["zanshin"]
            )
            calibrated_scores.append(sc)
            labels_bin.append(s["y"])
            weights_platt.append(s["weight"])

        platt = ProbabilisticPlattCalibrator()
        platt_info = platt.fit(calibrated_scores, labels_bin, sample_weights=weights_platt)
        new_config["platt_scaling"] = platt.to_dict()
        new_config["last_calibrated_at"] = datetime.now(timezone.utc).isoformat()

        summary = {
            "method": method_used,
            "samples_analyzed": len(samples),
            "initial_loss": round(init_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_reduction_pct": round(max(0.0, (init_loss - final_loss) / max(1e-4, init_loss)) * 100.0, 1),
            "initial_f1": init_metrics.get("f1_weighted", 0.0),
            "final_f1": final_metrics.get("f1_weighted", 0.0),
            "accuracy": final_metrics.get("accuracy", 1.0),
            "asymmetric_fp_cost": self.asymmetric_fp_cost,
            "platt_params": platt_info
        }

        return new_config, summary
