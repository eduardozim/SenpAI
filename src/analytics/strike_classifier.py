"""
Detector de Golpes em Duas Etapas — etapa 2: filtro aprendido (Pendência I3).

Etapa 1 (EventSpotter, inalterado): propõe candidatos a golpe pela velocidade do pulso.
Etapa 2 (este módulo): um classificador leve decide, para cada candidato, se ele é um golpe que
vira ponto, olhando a janela de poses ao redor do impacto (os dois pulsos, desaceleração no impacto,
amplitude do corte, maai entre os kenshi, aproximação, fumikomi pelo tornozelo, ponta do shinai...).

- Aprende com os golpes REVISADOS (kenshi e senseis, pesados pelo dan), a partir do cache de poses (I1):
  POSITIVO = 1, negativos = 0; golpes sem revisão ficam de fora. A decisão dos shinpan (placar oficial e
  bandeiras) não é rótulo: eles erram, e o modelo aprenderia o viés deles.
- Lutas do Golden Benchmark nunca entram no treino: servem só para decidir a promoção.
- Validação cruzada agrupada por vídeo (nenhum vídeo aparece no treino e na validação ao mesmo tempo).
- O modelo é uma regressão logística em numpy, salvo em JSON legível (pesos, médias, limiar).
- Um modelo só fica ativo na análise de luta se melhorar o F1 no Golden Benchmark (promote_if_better).

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.analytics.strike_classifier train "<pasta de vídeos>"
    python -m src.analytics.strike_classifier evaluate "<pasta de vídeos>" [--model <arquivo>] [--promote]
    python -m src.analytics.strike_classifier status | deactivate
"""

import os
import json
import math
import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import model_registry

MODEL_SCHEMA_VERSION = 1
MIN_TRAINING_POSITIVES = 5
MIN_TRAINING_NEGATIVES = 5

FEATURE_NAMES = [
    "pulso_dir_vel_pico", "pulso_esq_vel_pico", "pulsos_sincronia", "desaceleracao_impacto",
    "elevacao_maxima", "amplitude_corte", "maai", "aproximacao", "avanco_quadril",
    "fumikomi_tornozelo", "oponente_vel_pulso", "visibilidade_pulsos", "frames_sem_pose",
    "shinai_vel_ponta", "shinai_detectado", "zanshin_deslocamento",
]

WINDOW_BEFORE = 15
WINDOW_AFTER = 15


# ------------------------------------------------------------------------------
# Features
# ------------------------------------------------------------------------------

def _pt(pose: Optional[Dict[str, Any]], name: str) -> Optional[Tuple[float, float]]:
    """Coordenada em pixels de uma articulação (ou None)."""
    if not pose:
        return None
    lm = pose.get(name)
    if not isinstance(lm, dict):
        return None
    if "px" in lm and "py" in lm:
        return float(lm["px"]), float(lm["py"])
    if "x" in lm and "y" in lm:
        return float(lm["x"]), float(lm["y"])
    return None


def _mid(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return a or b
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _dist(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _torso(pose) -> Optional[float]:
    """Comprimento do tronco (ombros → quadris), a régua que torna as features independentes da câmera."""
    sh = _mid(_pt(pose, "LEFT_SHOULDER"), _pt(pose, "RIGHT_SHOULDER"))
    hp = _mid(_pt(pose, "LEFT_HIP"), _pt(pose, "RIGHT_HIP"))
    d = _dist(sh, hp)
    return d if d and d > 1e-6 else None


def _shinai_tip(pose) -> Optional[Tuple[float, float]]:
    sh = pose.get("SHINAI") if pose else None
    if isinstance(sh, dict) and sh.get("detected"):
        tip = sh.get("tip_px") or sh.get("tip_norm")
        if tip:
            return float(tip[0]), float(tip[1])
    return None


def _speeds(history, getter, start: int, end: int, fps: float) -> List[Optional[float]]:
    """Velocidade (unidades/s) quadro a quadro de um ponto, entre start e end (inclusive)."""
    out = []
    for i in range(start, end + 1):
        if 0 < i < len(history):
            a, b = getter(history[i - 1]), getter(history[i])
            d = _dist(a, b)
            out.append(None if d is None else d * fps)
        else:
            out.append(None)
    return out


def _max(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def extract_candidate_features(
    attacker_history: Sequence[Optional[Dict[str, Any]]],
    defender_history: Sequence[Optional[Dict[str, Any]]],
    impact_frame: int,
    fps: float
) -> Dict[str, float]:
    """Features numéricas de um candidato. Valores ausentes viram 0 (com indicadores de ausência)."""
    fps = float(fps) if fps and fps > 0 else 30.0
    n = len(attacker_history)
    t = max(0, min(int(impact_frame), n - 1)) if n else 0
    s, e = max(0, t - WINDOW_BEFORE), min(n - 1, t + WINDOW_AFTER)
    window = [attacker_history[i] for i in range(s, e + 1)] if n else []

    torso = _mean([_torso(p) for p in window]) or 1.0
    scale = lambda v: None if v is None else v / torso  # noqa: E731

    rw = _speeds(attacker_history, lambda p: _pt(p, "RIGHT_WRIST"), t - 5, t + 2, fps)
    lw = _speeds(attacker_history, lambda p: _pt(p, "LEFT_WRIST"), t - 5, t + 2, fps)
    r_peak, l_peak = scale(_max(rw)), scale(_max(lw))
    sync = (min(r_peak, l_peak) / max(r_peak, l_peak)) if r_peak and l_peak and max(r_peak, l_peak) > 0 else None

    wrists = lambda p: _mid(_pt(p, "RIGHT_WRIST"), _pt(p, "LEFT_WRIST"))  # noqa: E731
    at_impact = _mean(_speeds(attacker_history, wrists, t - 1, t + 1, fps))
    after = _mean(_speeds(attacker_history, wrists, t + 3, t + 8, fps))
    decel = (after / at_impact) if at_impact and after is not None and at_impact > 0 else None

    # Elevação (furikaburi) e amplitude do corte: em imagem, y cresce para baixo
    raise_vals, wrist_y_at_t = [], None
    for i in range(max(0, t - 12), t + 1):
        p = attacker_history[i] if i < n else None
        w, head = wrists(p), _mid(_pt(p, "LEFT_EAR"), _pt(p, "RIGHT_EAR"))
        if w is not None and head is not None:
            raise_vals.append((head[1] - w[1]) / torso)
        if i == t and w is not None:
            wrist_y_at_t = w[1]
    max_raise = max(raise_vals) if raise_vals else None
    top_y = None
    for i in range(max(0, t - 12), t + 1):
        w = wrists(attacker_history[i]) if i < n else None
        if w is not None:
            top_y = w[1] if top_y is None else min(top_y, w[1])
    amplitude = ((wrist_y_at_t - top_y) / torso) if wrist_y_at_t is not None and top_y is not None else None

    hips = lambda p: _mid(_pt(p, "LEFT_HIP"), _pt(p, "RIGHT_HIP"))  # noqa: E731
    d_at = lambda i: _dist(hips(attacker_history[i]), hips(defender_history[i])) if 0 <= i < min(n, len(defender_history)) else None  # noqa: E731
    maai_t = d_at(t)
    maai_before = d_at(max(0, t - 10))
    approach = ((maai_t - maai_before) / torso * fps / 10.0) if maai_t is not None and maai_before is not None else None

    hip_adv = scale(_mean(_speeds(attacker_history, hips, t - 8, t, fps)))

    ankle_speed = []
    for name in ("LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL"):
        ankle_speed.append(_max(_speeds(attacker_history, lambda p, nm=name: _pt(p, nm), t - 6, t + 6, fps)))
    fumikomi = scale(_max(ankle_speed))

    opp = scale(_max(_speeds(defender_history, wrists, t - 3, t + 3, fps))) if defender_history else None

    vis = []
    for p in window:
        for name in ("RIGHT_WRIST", "LEFT_WRIST"):
            lm = p.get(name) if p else None
            if isinstance(lm, dict) and "visibility" in lm:
                vis.append(float(lm["visibility"]))
    missing = (sum(1 for p in window if not p) / len(window)) if window else 1.0

    tip_speed = scale(_max(_speeds(attacker_history, _shinai_tip, t - 5, t + 2, fps)))
    shinai_seen = (sum(1 for p in window if _shinai_tip(p) is not None) / len(window)) if window else 0.0

    z_end = hips(attacker_history[min(n - 1, t + 15)]) if n else None
    zanshin = scale(_dist(hips(attacker_history[t]) if n else None, z_end))

    raw = {
        "pulso_dir_vel_pico": r_peak, "pulso_esq_vel_pico": l_peak, "pulsos_sincronia": sync,
        "desaceleracao_impacto": decel, "elevacao_maxima": max_raise, "amplitude_corte": amplitude,
        "maai": scale(maai_t), "aproximacao": approach, "avanco_quadril": hip_adv,
        "fumikomi_tornozelo": fumikomi, "oponente_vel_pulso": opp,
        "visibilidade_pulsos": _mean(vis), "frames_sem_pose": missing,
        "shinai_vel_ponta": tip_speed, "shinai_detectado": shinai_seen, "zanshin_deslocamento": zanshin,
    }
    feats = {}
    for k in FEATURE_NAMES:
        v = raw.get(k)
        feats[k] = 0.0 if v is None or not math.isfinite(v) else float(v)
    return feats


def features_matrix(rows: List[Dict[str, float]], names: Optional[List[str]] = None) -> np.ndarray:
    names = names or FEATURE_NAMES
    return np.array([[r[k] for k in names] for r in rows], dtype=float).reshape(-1, len(names))


# ------------------------------------------------------------------------------
# Modelo (regressão logística L2, balanceada por classe)
# ------------------------------------------------------------------------------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> Dict[str, Any]:
    """Newton/IRLS com pesos de classe balanceados e padronização das features."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Z = (X - mean) / std
    n, d = Z.shape
    pos = max(1, int(y.sum()))
    neg = max(1, n - pos)
    sw = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg))
    A = np.hstack([Z, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0  # sem penalidade no viés
    for _ in range(iters):
        p = _sigmoid(A @ w)
        grad = A.T @ (sw * (p - y)) + reg * w
        H = (A * (sw * p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return {"mean": mean.tolist(), "std": std.tolist(), "weights": w[:-1].tolist(), "bias": float(w[-1])}


def predict_proba(params: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(params["mean"])) / np.array(params["std"])
    return _sigmoid(Z @ np.array(params["weights"]) + params["bias"])


def best_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, Dict[str, Optional[float]]]:
    """Limiar que maximiza o F1 (fórmula única do SenpAI) nas previsões fora da amostra."""
    from src.engine.weak_labels import precision_recall_f1
    best_t, best = 0.5, {"precisao": None, "recall": None, "f1": None}
    for thr in np.unique(np.round(p, 4)):
        pred = p >= thr
        prf = precision_recall_f1(int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum()))
        if (prf["f1"] or 0) > (best["f1"] or 0):
            best_t, best = float(thr), prf
    return best_t, best


def group_cross_validate(X: np.ndarray, y: np.ndarray, groups: List[str], l2: float = 1.0) -> np.ndarray:
    """Probabilidades fora da amostra, deixando um vídeo de fora por vez."""
    oof = np.full(len(y), np.nan)
    g = np.array(groups)
    for vid in sorted(set(groups)):
        test = g == vid
        train = ~test
        if len(set(y[train].tolist())) < 2:
            continue
        params = fit_logistic(X[train], y[train], l2=l2)
        oof[test] = predict_proba(params, X[test])
    return oof


# ------------------------------------------------------------------------------
# Registro de modelos
# ------------------------------------------------------------------------------

MODEL_FAMILY = "strike_filter"


def get_models_dir() -> str:
    return model_registry.get_models_dir(MODEL_FAMILY)


def save_model(model: Dict[str, Any], models_dir: Optional[str] = None) -> str:
    d = models_dir or get_models_dir()
    path = os.path.join(d, f"strike_filter_{model['trained_at'].replace(':', '').replace('-', '')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    return path


def load_model(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
        if model.get("schema") != MODEL_SCHEMA_VERSION:
            return None
        from src.analytics.pose_encoder import feature_names_for
        uses_encoder = bool(model.get("encoder"))
        if model.get("features") != feature_names_for(FEATURE_NAMES, uses_encoder):
            return None
        if uses_encoder and not os.path.isfile(model["encoder"]):
            return None
        return model
    except Exception:
        return None


def get_active_model_path(models_dir: Optional[str] = None) -> Optional[str]:
    return model_registry.get_active_model_path(MODEL_FAMILY, models_dir)


def set_active_model(path: Optional[str], reason: str = "", models_dir: Optional[str] = None) -> None:
    model_registry.set_active_model(MODEL_FAMILY, path, reason, models_dir)


class StrikeFilter:
    """Etapa 2 aplicada na análise de luta: probabilidade de cada candidato ser golpe que vira ponto."""

    def __init__(self, model: Dict[str, Any], path: Optional[str] = None):
        from src.analytics.pose_encoder import PoseEncoder
        self.model = model
        self.path = path
        self.threshold = float(model["threshold"])
        self.encoder = PoseEncoder.from_path(model.get("encoder"))

    @classmethod
    def load_active(cls, models_dir: Optional[str] = None) -> Optional["StrikeFilter"]:
        path = get_active_model_path(models_dir)
        model = load_model(path) if path else None
        return cls(model, path) if model else None

    def probability(self, attacker_history, defender_history, impact_frame: int, fps: float) -> float:
        from src.analytics.pose_encoder import with_embedding
        feats = extract_candidate_features(attacker_history, defender_history, impact_frame, fps)
        feats = with_embedding(feats, self.encoder, attacker_history, defender_history, impact_frame)
        return float(predict_proba(self.model["params"], features_matrix([feats], self.model["features"]))[0])

    def keep(self, probability: float) -> bool:
        return probability >= self.threshold


# ------------------------------------------------------------------------------
# Dataset, treino e promoção
# ------------------------------------------------------------------------------



def _target(label: str) -> Optional[int]:
    from src.engine import weak_labels as wl
    if label == wl.POSITIVE:
        return 1
    if label in wl.NEGATIVE_LABELS:
        return 0
    return None


def build_dataset(pipeline, videos_dir: str, labels_path: Optional[str] = None,
                  exclude_golden: bool = True) -> Dict[str, Any]:
    """
    Candidatos rotulados pelo BANCO ÚNICO (I11): só as revisões (análise, Excel, curadoria), pesadas pelo dan.
    A decisão dos shinpan (gabarito e bandeiras) não rotula: só gera pedidos de revisão.
    Lutas do Golden Benchmark ficam de fora.
    """
    from src.engine.label_bank import build_training_reports
    from src.analytics.pose_encoder import PoseEncoder, with_embedding
    encoder = PoseEncoder.load_active()
    training = build_training_reports(pipeline, videos_dir, labels_path=labels_path, exclude_golden=exclude_golden)
    rows, y, groups = [], [], []
    for report in training["reports"]:
        targets = [(c, _target(c["weak_label"])) for c in report["candidates"]["items"]]
        targets = [(c, t) for c, t in targets if t is not None]
        if not targets:
            continue
        invert = bool(report.get("inverter_aka_shiro", False))
        features = pipeline.extract_match_features(report["video_path"], invert_combatants=invert)
        tracker = pipeline.combatant_tracker
        hist = {
            "KENSHI_AKA": tracker.interpolate_missing_poses(features["aka_history"], max_gap=10),
            "KENSHI_SHIRO": tracker.interpolate_missing_poses(features["shiro_history"], max_gap=10),
        }
        for cand, target in targets:
            atk = cand["attacker_id"]
            dfd = "KENSHI_SHIRO" if atk == "KENSHI_AKA" else "KENSHI_AKA"
            base = extract_candidate_features(hist[atk], hist[dfd], cand["impact_frame"], features["fps"])
            rows.append(with_embedding(base, encoder, hist[atk], hist[dfd], cand["impact_frame"]))
            y.append(target)
            groups.append(report["video_name"])
    return {"rows": rows, "y": y, "groups": groups, "skipped": training["skipped"],
            "encoder": encoder.path if encoder else None, "conflitos": training["conflitos"]}


def train_model(dataset: Dict[str, Any], l2: float = 1.0) -> Dict[str, Any]:
    """Treina, valida por vídeo e devolve o modelo (ainda não ativo)."""
    from src.engine.weak_labels import precision_recall_f1
    from src.analytics.pose_encoder import feature_names_for
    y = np.array(dataset["y"], dtype=float)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos < MIN_TRAINING_POSITIVES or neg < MIN_TRAINING_NEGATIVES:
        raise ValueError(f"Dados insuficientes para treinar: {pos} positivos e {neg} negativos "
                         f"(mínimo {MIN_TRAINING_POSITIVES} e {MIN_TRAINING_NEGATIVES}). Revise golpes na análise ou responda a curadoria (dúvidas \"Decisão dos shinpan a revisar\").")

    def _cv(names):
        X = features_matrix(dataset["rows"], names)
        oof = group_cross_validate(X, y, dataset["groups"], l2=l2)
        valid = ~np.isnan(oof)
        if valid.sum() and len(set(y[valid].tolist())) == 2:
            thr, prf = best_threshold(y[valid], oof[valid])
            return X, thr, prf
        return X, 0.5, {"precisao": None, "recall": None, "f1": None}

    # Com encoder de pose (I10) disponível, testa as duas versões e fica com a que validar melhor por vídeo
    variants = {"sem_encoder": feature_names_for(FEATURE_NAMES, False)}
    if dataset.get("encoder"):
        variants["com_encoder"] = feature_names_for(FEATURE_NAMES, True)
    results = {k: _cv(names) for k, names in variants.items()}
    chosen = max(results, key=lambda k: (results[k][2]["f1"] or 0.0, k == "sem_encoder"))
    X, threshold, cv = results[chosen]
    params = fit_logistic(X, y, l2=l2)
    baseline = precision_recall_f1(pos, neg, 0)  # sem filtro: todo candidato vira golpe
    return {
        "schema": MODEL_SCHEMA_VERSION,
        "tipo": "regressao_logistica_l2",
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "features": variants[chosen],
        "encoder": dataset.get("encoder") if chosen == "com_encoder" else None,
        "variantes_validadas": {k: v[2]["f1"] for k, v in results.items()},
        "params": params,
        "threshold": threshold,
        "l2": l2,
        "dados": {"candidatos": len(y), "positivos": pos, "negativos": neg, "videos": sorted(set(dataset["groups"]))},
        "validacao_por_video": {"sem_filtro": baseline, "com_filtro": cv},
    }


def evaluate_on_golden(pipeline, videos_dir: str, model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Benchmark por luta (Golden) com o filtro indicado, sem gravar histórico nem amostras."""
    from src.engine.active_learning import GoldenBenchmark
    previous = getattr(pipeline, "strike_filter", None)
    pipeline.strike_filter = StrikeFilter(model) if model else None
    try:
        return GoldenBenchmark().run_matches(pipeline, videos_dir, record_history=False, update_samples=False)
    finally:
        pipeline.strike_filter = previous


def promote_if_better(pipeline, videos_dir: str, model_path: str) -> Dict[str, Any]:
    """Ativa o modelo só se o F1 no Golden Benchmark ficar acima do sistema atual."""
    model = load_model(model_path)
    if model is None:
        raise ValueError("Modelo inválido ou de outra versão de features.")
    active = StrikeFilter.load_active()
    current = evaluate_on_golden(pipeline, videos_dir, active.model if active else None)
    challenger = evaluate_on_golden(pipeline, videos_dir, model)
    cur_f1 = current["total"].get("f1")
    new_f1 = challenger["total"].get("f1")
    if not challenger["lutas"] or new_f1 is None:
        decision, reason = False, "Sem lutas golden completas para comparar: congele lutas antes de promover."
    elif cur_f1 is None or new_f1 > cur_f1:
        decision, reason = True, f"F1 no golden {('—' if cur_f1 is None else f'{cur_f1:.3f}')} → {new_f1:.3f}."
    else:
        decision, reason = False, f"Não melhorou o golden (F1 {cur_f1:.3f} → {new_f1:.3f})."
    if decision:
        set_active_model(model_path, reason)
    return {"promovido": decision, "motivo": reason, "atual": current["total"], "desafiante": challenger["total"]}


# ------------------------------------------------------------------------------
# Linha de comando
# ------------------------------------------------------------------------------

def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Filtro aprendido de golpes (etapa 2 do detector).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_tr = sub.add_parser("train", help="Treina com as lutas do gabarito (exceto golden).")
    p_tr.add_argument("videos_dir")
    p_tr.add_argument("--l2", type=float, default=1.0)
    p_tr.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p_ev = sub.add_parser("evaluate", help="Compara com o sistema atual no golden.")
    p_ev.add_argument("videos_dir")
    p_ev.add_argument("--model", help="Arquivo do modelo (padrão: o mais recente).")
    p_ev.add_argument("--promote", action="store_true", help="Ativa se melhorar o F1 no golden.")
    p_ev.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    sub.add_parser("status", help="Mostra o modelo ativo e os treinados.")
    sub.add_parser("deactivate", help="Desliga o filtro (volta ao detector de 1 etapa).")
    args = parser.parse_args(argv)

    if args.command == "status":
        active = get_active_model_path()
        print(f"Ativo: {active or 'nenhum (detector de 1 etapa)'}")
        for name in sorted(os.listdir(get_models_dir())):
            if name.startswith("strike_filter_"):
                m = load_model(os.path.join(get_models_dir(), name))
                if m:
                    cv = m["validacao_por_video"]["com_filtro"]
                    print(f"  {name}: {m['dados']['candidatos']} candidatos, {len(m['dados']['videos'])} vídeos · "
                          f"validação F1 {_fmt(cv['f1'])} (precisão {_fmt(cv['precisao'])}, recall {_fmt(cv['recall'])})")
        return 0
    if args.command == "deactivate":
        set_active_model(None, "Desativado pela linha de comando.")
        print("Filtro desativado.")
        return 0

    from src.pipeline import SenpAIPipeline
    pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=args.device)
    pipeline.strike_filter = None  # o dataset é sempre montado com os candidatos brutos

    if args.command == "train":
        ds = build_dataset(pipeline, args.videos_dir)
        for s in ds["skipped"]:
            print(f"  fora do treino: {s['video_name']} ({s['motivo']})")
        try:
            model = train_model(ds, l2=args.l2)
        except ValueError as e:
            print(e)
            return 1
        path = save_model(model)
        v = model["validacao_por_video"]
        print(f"Modelo: {path}\n  {model['dados']['candidatos']} candidatos ({model['dados']['positivos']} positivos) em "
              f"{len(model['dados']['videos'])} vídeos\n  validação por vídeo: sem filtro F1 {_fmt(v['sem_filtro']['f1'])} → "
              f"com filtro F1 {_fmt(v['com_filtro']['f1'])} (precisão {_fmt(v['com_filtro']['precisao'])}, "
              f"recall {_fmt(v['com_filtro']['recall'])}) · limiar {model['threshold']:.3f}\n"
              f"  Ainda não está ativo: rode 'evaluate --promote' para comparar no golden.")
        return 0

    path = args.model
    if not path:
        files = sorted(n for n in os.listdir(get_models_dir()) if n.startswith("strike_filter_"))
        if not files:
            print("Nenhum modelo treinado.")
            return 1
        path = os.path.join(get_models_dir(), files[-1])
    if args.promote:
        res = promote_if_better(pipeline, args.videos_dir, path)
        print(f"{'PROMOVIDO' if res['promovido'] else 'NÃO promovido'}: {res['motivo']}")
        return 0
    model = load_model(path)
    for label, m in (("atual", (StrikeFilter.load_active() or None)), ("desafiante", StrikeFilter(model) if model else None)):
        r = evaluate_on_golden(pipeline, args.videos_dir, m.model if m else None)["total"]
        print(f"{label}: precisão {_fmt(r.get('precisao'))} · recall {_fmt(r.get('recall'))} · F1 {_fmt(r.get('f1'))} · "
              f"acurácia {_fmt(r.get('acuracia'))} ({r['lutas']} lutas golden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
