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
SCORE_KEYS = ("target_impact", "fumikomi_sync", "posture", "zanshin")

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
    Otimizador de Pesos e Limiares (busca global sem gradiente: scipy differential_evolution).
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
        Normaliza feedbacks: notas reais em [0.0 .. 1.0] (incluindo hasuji quando houver), rótulo binário,
        peso = peso_revisor * decaimento_temporal e o vídeo de origem (para validar separando por vídeo).
        Feedbacks SEM as notas do golpe (ex.: FN de golpe não detectado) são descartados — nunca recebem
        notas inventadas (Pendência P5.3). A contagem fica em self.last_skipped_without_scores.
        """
        if now is None:
            now = time.time()

        processed = []
        skipped = 0
        for item in feedback_list:
            lbl = str(item.get("label", "")).upper()
            if lbl in ["TP", "CONFIRMED", "VALID", "IPPON", "FN", "INCLUDED"]:
                target_y = 1
            elif lbl in ["FP", "REJECTED", "INVALID", "NO_IPPON"]:
                target_y = 0
            else:
                continue

            sub_scores = item.get("sub_scores") or {}
            if not all(isinstance(sub_scores.get(k), (int, float)) for k in SCORE_KEYS):
                skipped += 1
                continue

            def _norm(v: float) -> float:
                v = float(v)
                return v / 100.0 if v > 1.0 else v

            hasuji = sub_scores.get("hasuji", sub_scores.get("hasuji_score"))

            ts = item.get("timestamp")
            if not isinstance(ts, (int, float)) and "review_date" in item:
                ts = item.get("review_date")
            if ts is None and "created_at" in item:
                ts = item.get("created_at")

            decay_w = self.calculate_decay_weight(ts, now=now)
            rev_w = self.get_reviewer_weight(item)

            strike_type = str(item.get("strike_type", "MEN")).upper().strip()
            if strike_type not in ["MEN", "KOTE", "DO", "TSUKI"]:
                strike_type = "MEN"

            processed.append({
                "y": target_y,
                "target_impact": _norm(sub_scores["target_impact"]),
                "fumikomi_sync": _norm(sub_scores["fumikomi_sync"]),
                "posture": _norm(sub_scores["posture"]),
                "zanshin": _norm(sub_scores["zanshin"]),
                "hasuji": _norm(hasuji) if isinstance(hasuji, (int, float)) else None,
                "weight": rev_w * decay_w,
                "reviewer_weight": rev_w,
                "decay_weight": decay_w,
                "strike_type": strike_type,
                "video": str(item.get("video_name") or ""),
            })

        self.last_skipped_without_scores = skipped
        return processed

    @staticmethod
    def _config_with_weights(base_config: Dict[str, Any], weights: Dict[str, float], min_total: float,
                             sub_thresholds: Dict[str, float], strike_type: Optional[str] = None) -> Dict[str, Any]:
        """Configuração candidata: os pesos vão para onde a análise de luta lê (weights_by_strike_type)."""
        cfg = json.loads(json.dumps(base_config))
        cfg["min_total_score"] = min_total
        cfg["sub_thresholds"] = dict(sub_thresholds)
        by_strike = cfg.get("weights_by_strike_type") or json.loads(json.dumps(DEFAULT_WEIGHTS_BY_STRIKE_TYPE))
        targets = [strike_type.upper()] if strike_type else list(DEFAULT_WEIGHTS_BY_STRIKE_TYPE)
        for st in targets:
            by_strike[st] = dict(weights)
        cfg["weights_by_strike_type"] = by_strike
        if not strike_type:
            cfg["weights"] = dict(weights)
        return cfg

    def config_loss(self, config: Dict[str, Any], samples: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Perda assimétrica ponderada com a MESMA decisão da análise de luta (calibrator.decide_strike):
        L = sum(W_i * (3.0 * FP_i + 1.0 * FN_i)) / sum(W_i)
        """
        from src.engine.calibrator import decide_strike
        from src.engine.weak_labels import precision_recall_f1
        tp_w = fp_w = fn_w = tn_w = 0.0
        loss_sum = 0.0
        for s in samples:
            y_pred = decide_strike(
                s["target_impact"], s["fumikomi_sync"], s["posture"], s["zanshin"],
                config=config, strike_type=s.get("strike_type", "MEN"), hasuji_score=s.get("hasuji")
            )["is_valid"]
            sw = float(s.get("weight", 1.0))
            if s["y"] == 1 and y_pred:
                tp_w += sw
            elif s["y"] == 0 and not y_pred:
                tn_w += sw
            elif s["y"] == 0 and y_pred:
                fp_w += sw
                loss_sum += sw * self.asymmetric_fp_cost
            else:
                fn_w += sw
                loss_sum += sw * self.asymmetric_fn_cost
        total = tp_w + tn_w + fp_w + fn_w
        prf = precision_recall_f1(tp_w, fp_w, fn_w)
        return (loss_sum / total if total > 0 else 0.0), {
            "asymmetric_loss": round(loss_sum / total, 4) if total > 0 else 0.0,
            "accuracy": round((tp_w + tn_w) / total, 4) if total > 0 else 1.0,
            "f1_weighted": prf["f1"] or 0.0,
            "precisao": prf["precisao"], "recall": prf["recall"],
            "tp_weight": round(tp_w, 2), "fp_weight": round(fp_w, 2),
            "fn_weight": round(fn_w, 2), "tn_weight": round(tn_w, 2),
        }

    def evaluate_loss(
        self,
        weights: Dict[str, float],
        min_total_score: float,
        sub_thresholds: Dict[str, float],
        samples: List[Dict[str, Any]],
        prior_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Perda de um conjunto de pesos/limiares aplicado a todos os tipos de golpe (+ regularização L2)."""
        if not samples:
            return 0.0, {"asymmetric_loss": 0.0, "accuracy": 1.0, "f1_weighted": 1.0}
        cfg = self._config_with_weights({}, weights, min_total_score, sub_thresholds)
        base_loss, metrics = self.config_loss(cfg, samples)
        l2_penalty = 0.0
        if prior_weights:
            l2_penalty = self.l2_reg_weight * sum(
                (weights.get(k, 0.25) - prior_weights.get(k, 0.25)) ** 2 for k in SCORE_KEYS
            )
        metrics["total_loss"] = round(base_loss + l2_penalty, 4)
        return base_loss + l2_penalty, metrics

    @staticmethod
    def split_by_video(samples: List[Dict[str, Any]], val_fraction: float = 0.25, seed: int = 7) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Treino e validação por vídeo (nenhum vídeo nos dois lados). Com menos de 2 vídeos, sem validação."""
        videos = sorted({s.get("video", "") for s in samples})
        if len(videos) < 2:
            return samples, []
        rng = np.random.default_rng(seed)
        rng.shuffle(videos)
        n_val = max(1, int(round(len(videos) * val_fraction)))
        val_videos = set(videos[:n_val])
        return ([s for s in samples if s.get("video", "") not in val_videos],
                [s for s in samples if s.get("video", "") in val_videos])

    def optimize(
        self,
        current_config: Dict[str, Any],
        feedbacks: List[Dict[str, Any]],
        strike_type: Optional[str] = None,
        n_trials_optuna: int = 60
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Recalibra pesos, nota mínima e os limiares que de fato invalidam um golpe (alvo e hasuji),
        com a mesma decisão da análise de luta.
        - Busca global sem gradiente (scipy differential_evolution) sobre a perda real em degraus.
        - Regularização L2 em direção à configuração atual (mudanças pequenas, a menos que os dados peçam).
        - Treina em parte dos vídeos e só aceita se não piorar a perda nos vídeos de validação.
        - Sem a antiga regra "pior FP + 0,02" (P5.4) e sem notas inventadas para FN (P5.3).
        n_trials_optuna é mantido só por compatibilidade de assinatura.
        """
        from src.engine.calibrator import resolve_weights

        samples = self.preprocess_feedbacks(feedbacks)
        skipped = getattr(self, "last_skipped_without_scores", 0)
        if strike_type:
            samples = [s for s in samples if s["strike_type"] == strike_type.upper()]
        if not samples:
            return current_config, {
                "status": "skipped",
                "message": "Nenhuma amostra com as notas do golpe para calibrar.",
                "descartados_sem_notas": skipped,
            }

        prior_weights = dict(resolve_weights(current_config, strike_type) if strike_type else
                             current_config.get("weights", {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15}))
        prior_min_total = float(current_config.get("min_total_score", 0.65))
        prior_sub_th = dict(current_config.get("sub_thresholds", {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}))
        prior_hasuji = float(prior_sub_th.get("hasuji", 0.40))

        # Restrições biomecânicas (Eixo 2): pisos para todos os sub-limiares e para a nota mínima
        floors = {"target_impact": 0.30, "fumikomi_sync": 0.30, "posture": 0.30, "zanshin": 0.20}
        floor_min_total = 0.50
        try:
            from src.engine.actionable_research import BayesianPriorInjector
            b_priors = BayesianPriorInjector().derive_optimizer_bounds_and_priors(strike_type=strike_type)
            if b_priors and "sub_threshold_lower_bounds" in b_priors:
                for k in floors:
                    floors[k] = max(floors[k], float(b_priors["sub_threshold_lower_bounds"].get(k, floors[k])))
                floor_min_total = max(floor_min_total, float(b_priors.get("min_total_score_floor", 0.50)))
        except Exception:
            pass
        floors = {k: min(v, 0.85) for k, v in floors.items()}
        floor_min_total = min(floor_min_total, 0.90)
        lower_target = floors["target_impact"]
        for k in ("fumikomi_sync", "posture", "zanshin"):
            prior_sub_th[k] = max(float(prior_sub_th.get(k, floors[k])), floors[k])

        train, val = self.split_by_video(samples)
        free_pool = 0.60

        def decode(x) -> Tuple[Dict[str, float], float, Dict[str, float]]:
            raw = np.maximum(np.array(x[:4], dtype=float), 1e-6)
            w = 0.10 + free_pool * raw / raw.sum()
            weights = {k: round(float(v), 3) for k, v in zip(SCORE_KEYS, w)}
            weights["target_impact"] = round(weights["target_impact"] + round(1.0 - sum(weights.values()), 3), 3)
            sub_th = dict(prior_sub_th)
            sub_th["target_impact"] = round(float(x[5]), 2)
            sub_th["hasuji"] = round(float(x[6]), 2)
            return weights, round(float(x[4]), 2), sub_th

        def objective(x) -> float:
            weights, min_total, sub_th = decode(x)
            cfg = self._config_with_weights(current_config, weights, min_total, sub_th, strike_type)
            loss, _ = self.config_loss(cfg, train)
            reg = sum((weights[k] - prior_weights.get(k, 0.25)) ** 2 for k in SCORE_KEYS)
            reg += (min_total - prior_min_total) ** 2 + (sub_th["target_impact"] - prior_sub_th.get("target_impact", 0.40)) ** 2
            reg += (sub_th["hasuji"] - prior_hasuji) ** 2
            return loss + self.l2_reg_weight * reg

        bounds = [(0.05, 1.0)] * 4 + [(floor_min_total, 0.90), (lower_target, 0.85), (0.20, 0.85)]
        prior_raw = [max(0.05, min(1.0, (prior_weights.get(k, 0.25) - 0.10) / free_pool + 0.05)) for k in SCORE_KEYS]
        x0 = prior_raw + [min(max(prior_min_total, floor_min_total), 0.90),
                          min(max(prior_sub_th.get("target_impact", 0.40), lower_target), 0.85),
                          min(max(prior_hasuji, 0.20), 0.85)]
        # SciPy recusa x0 exatamente no limite: prende o ponto inicial um pouco para dentro
        x0 = [float(np.clip(v, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))) for v, (lo, hi) in zip(x0, bounds)]
        res = optimize.differential_evolution(objective, bounds, x0=x0, seed=42, maxiter=60, popsize=12,
                                              tol=1e-7, polish=False, updating="immediate")
        best_weights, best_min_total, best_sub_th = decode(res.x)
        candidate = self._config_with_weights(current_config, best_weights, best_min_total, best_sub_th, strike_type)

        init_loss, init_metrics = self.config_loss(current_config, train)
        final_loss, final_metrics = self.config_loss(candidate, train)

        validation = None
        accepted = True
        if val:
            cur_val, cur_val_m = self.config_loss(current_config, val)
            new_val, new_val_m = self.config_loss(candidate, val)
            accepted = new_val <= cur_val
            validation = {
                "videos": sorted({s.get("video", "") for s in val}), "amostras": len(val),
                "perda_atual": round(cur_val, 4), "perda_nova": round(new_val, 4),
                "f1_atual": cur_val_m["f1_weighted"], "f1_novo": new_val_m["f1_weighted"], "aceito": accepted,
            }

        if not accepted:
            return current_config, {
                "status": "rejected_validation", "method": "differential_evolution",
                "message": "A nova calibração piorou nos vídeos de validação; a atual foi mantida.",
                "samples_analyzed": len(samples), "descartados_sem_notas": skipped, "validacao": validation,
                "initial_loss": round(init_loss, 4), "final_loss": round(final_loss, 4), "loss_reduction_pct": 0.0,
                "initial_f1": init_metrics.get("f1_weighted", 0.0), "final_f1": init_metrics.get("f1_weighted", 0.0),
                "accuracy": init_metrics.get("accuracy", 1.0), "asymmetric_fp_cost": self.asymmetric_fp_cost,
            }

        # Platt Scaling sobre as notas totais calculadas pela decisão de produção
        from src.engine.calibrator import decide_strike
        totals = [decide_strike(s["target_impact"], s["fumikomi_sync"], s["posture"], s["zanshin"], config=candidate,
                                strike_type=s["strike_type"], hasuji_score=s.get("hasuji"))["total_score"] for s in samples]
        platt = ProbabilisticPlattCalibrator()
        platt_info = platt.fit(totals, [s["y"] for s in samples], sample_weights=[s["weight"] for s in samples])
        candidate["platt_scaling"] = platt.to_dict()
        candidate["last_calibrated_at"] = datetime.now(timezone.utc).isoformat()

        return candidate, {
            "method": "differential_evolution",
            "samples_analyzed": len(samples),
            "descartados_sem_notas": skipped,
            "amostras_treino": len(train),
            "validacao": validation,
            "initial_loss": round(init_loss, 4),
            "final_loss": round(final_loss, 4),
            "loss_reduction_pct": round(max(0.0, (init_loss - final_loss) / max(1e-4, init_loss)) * 100.0, 1),
            "initial_f1": init_metrics.get("f1_weighted", 0.0),
            "final_f1": final_metrics.get("f1_weighted", 0.0),
            "accuracy": final_metrics.get("accuracy", 1.0),
            "asymmetric_fp_cost": self.asymmetric_fp_cost,
            "platt_params": platt_info,
        }
