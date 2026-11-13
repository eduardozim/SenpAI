"""
Classificador de Técnica — MEN, KOTE, DO, TSUKI (Pendência I9).

Substitui a regra fixa "altura da mão, senão MEN" do EventSpotter por um modelo que aprende
onde o golpe termina EM RELAÇÃO AO CORPO DO OPONENTE (cabeça, punho, lateral do tronco, garganta)
e a direção do movimento (corte vertical, diagonal ou estocada).

- Rótulos: a técnica oficial de cada ponto do gabarito (I2), herdada pelo golpe que casou com ele.
  Pontos de gabarito automático antigo (origem "automatico") não entram: a técnica deles é desconhecida.
- Lutas do Golden Benchmark nunca entram no treino.
- Validação cruzada deixando um vídeo de fora por vez.
- Modelo: regressão logística multinomial (softmax) L2 em numpy, salvo em JSON legível,
  no registro único de modelos (família "technique_classifier").
- Promoção: só se a acurácia de técnica no golden melhorar sem piorar o F1 dos pontos.

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.analytics.technique_classifier train "<pasta de vídeos>"
    python -m src.analytics.technique_classifier evaluate "<pasta de vídeos>" [--promote]
    python -m src.analytics.technique_classifier status | deactivate
"""

import os
import json
import math
import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analytics.strike_classifier import _pt, _mid, _dist, _torso, _shinai_tip, _speeds, _mean, _max
from src.utils import model_registry

MODEL_FAMILY = "technique_classifier"
MODEL_SCHEMA_VERSION = 1
TECHNIQUES = ["MEN", "KOTE", "DO", "TSUKI"]
MIN_PER_CLASS = 5
MIN_CLASSES = 2

FEATURE_NAMES = [
    "alvo_dy_cabeca", "alvo_dx_cabeca", "alvo_dist_cabeca",
    "alvo_dy_punhos", "alvo_dist_punhos",
    "alvo_dy_garganta", "alvo_dist_garganta",
    "alvo_dy_quadril", "alvo_dx_tronco", "alvo_dist_tronco",
    "pulsos_altura_propria", "elevacao_maxima", "descida",
    "mov_horizontal", "mov_vertical", "razao_horizontal", "usa_ponta_shinai",
]


def _target_point(pose):
    """Onde o golpe "chega": a ponta do shinai, ou o meio dos pulsos se o shinai não foi visto."""
    tip = _shinai_tip(pose)
    if tip is not None:
        return tip, 1.0
    return _mid(_pt(pose, "RIGHT_WRIST"), _pt(pose, "LEFT_WRIST")), 0.0


def extract_technique_features(
    attacker_history: Sequence[Optional[Dict[str, Any]]],
    defender_history: Sequence[Optional[Dict[str, Any]]],
    impact_frame: int,
    fps: float
) -> Dict[str, float]:
    fps = float(fps) if fps and fps > 0 else 30.0
    n = len(attacker_history)
    t = max(0, min(int(impact_frame), n - 1)) if n else 0
    atk = attacker_history[t] if n else None
    dfd = defender_history[t] if t < len(defender_history) else None

    torso_a = _torso(atk) or 1.0
    torso_d = _torso(dfd) or torso_a
    target, uses_tip = _target_point(atk)

    head = _mid(_pt(dfd, "LEFT_EAR"), _pt(dfd, "RIGHT_EAR")) or _pt(dfd, "NOSE")
    d_wrists = _mid(_pt(dfd, "RIGHT_WRIST"), _pt(dfd, "LEFT_WRIST"))
    d_sh = _mid(_pt(dfd, "LEFT_SHOULDER"), _pt(dfd, "RIGHT_SHOULDER"))
    d_hip = _mid(_pt(dfd, "LEFT_HIP"), _pt(dfd, "RIGHT_HIP"))
    throat = (d_sh[0], d_sh[1] - 0.15 * torso_d) if d_sh else None
    trunk = _mid(d_sh, d_hip)

    def rel(a, b, axis):
        return None if a is None or b is None else (a[axis] - b[axis]) / torso_d

    def dist(a, b):
        d = _dist(a, b)
        return None if d is None else d / torso_d

    wrists_a = lambda p: _mid(_pt(p, "RIGHT_WRIST"), _pt(p, "LEFT_WRIST"))  # noqa: E731
    sh_a = _mid(_pt(atk, "LEFT_SHOULDER"), _pt(atk, "RIGHT_SHOULDER"))
    w_t = wrists_a(atk)
    own_height = ((sh_a[1] - w_t[1]) / torso_a) if sh_a and w_t else None

    ys = [wrists_a(attacker_history[i]) for i in range(max(0, t - 12), t + 1)]
    ys = [p[1] for p in ys if p is not None]
    head_a = _mid(_pt(atk, "LEFT_EAR"), _pt(atk, "RIGHT_EAR"))
    max_raise = ((head_a[1] - min(ys)) / torso_a) if ys and head_a else None
    drop = ((w_t[1] - min(ys)) / torso_a) if ys and w_t else None

    start = wrists_a(attacker_history[max(0, t - 6)]) if n else None
    mov_h = (abs(w_t[0] - start[0]) / torso_a) if w_t and start else None
    mov_v = (abs(w_t[1] - start[1]) / torso_a) if w_t and start else None
    ratio = (mov_h / (mov_h + mov_v)) if mov_h is not None and mov_v is not None and (mov_h + mov_v) > 1e-6 else None

    raw = {
        "alvo_dy_cabeca": rel(target, head, 1), "alvo_dx_cabeca": rel(target, head, 0), "alvo_dist_cabeca": dist(target, head),
        "alvo_dy_punhos": rel(target, d_wrists, 1), "alvo_dist_punhos": dist(target, d_wrists),
        "alvo_dy_garganta": rel(target, throat, 1), "alvo_dist_garganta": dist(target, throat),
        "alvo_dy_quadril": rel(target, d_hip, 1), "alvo_dx_tronco": rel(target, trunk, 0), "alvo_dist_tronco": dist(target, trunk),
        "pulsos_altura_propria": own_height, "elevacao_maxima": max_raise, "descida": drop,
        "mov_horizontal": mov_h, "mov_vertical": mov_v, "razao_horizontal": ratio, "usa_ponta_shinai": uses_tip,
    }
    return {k: (0.0 if raw.get(k) is None or not math.isfinite(raw[k]) else float(raw[k])) for k in FEATURE_NAMES}


def features_matrix(rows: List[Dict[str, float]], names: Optional[List[str]] = None) -> np.ndarray:
    names = names or FEATURE_NAMES
    return np.array([[r[k] for k in names] for r in rows], dtype=float).reshape(-1, len(names))


# ------------------------------------------------------------------------------
# Modelo multinomial
# ------------------------------------------------------------------------------

def _softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def fit_softmax(X: np.ndarray, y: List[str], classes: List[str], l2: float = 1.0, iters: int = 600, lr: float = 0.5) -> Dict[str, Any]:
    """Regressão logística multinomial com padronização, classes balanceadas e L2 (gradiente descendente)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Z = (X - mean) / std
    n, d = Z.shape
    k = len(classes)
    Y = np.zeros((n, k))
    idx = {c: i for i, c in enumerate(classes)}
    for r, label in enumerate(y):
        Y[r, idx[label]] = 1.0
    counts = Y.sum(axis=0)
    sw = (n / (k * np.maximum(counts, 1.0)))[np.argmax(Y, axis=1)]
    W = np.zeros((d, k))
    b = np.zeros(k)
    for _ in range(iters):
        P = _softmax(Z @ W + b)
        G = (P - Y) * sw[:, None] / n
        W -= lr * (Z.T @ G + l2 * W / n)
        b -= lr * G.sum(axis=0)
    return {"classes": classes, "mean": mean.tolist(), "std": std.tolist(), "W": W.tolist(), "b": b.tolist()}


def predict_proba(params: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    Z = (X - np.array(params["mean"])) / np.array(params["std"])
    return _softmax(Z @ np.array(params["W"]) + np.array(params["b"]))


def group_cross_validate(X: np.ndarray, y: List[str], groups: List[str], classes: List[str], l2: float = 1.0) -> List[Optional[str]]:
    """Previsões fora da amostra, deixando um vídeo de fora por vez."""
    return [p for p, _ in group_cross_validate_proba(X, y, groups, classes, l2=l2)]


def group_cross_validate_proba(X: np.ndarray, y: List[str], groups: List[str], classes: List[str],
                               l2: float = 1.0) -> List[Tuple[Optional[str], Optional[float]]]:
    """Como group_cross_validate, com a probabilidade da classe prevista (para medir o limiar de confiança)."""
    preds: List[Tuple[Optional[str], Optional[float]]] = [(None, None)] * len(y)
    g = np.array(groups)
    ya = np.array(y)
    for vid in sorted(set(groups)):
        test = g == vid
        train = ~test
        if len(set(ya[train].tolist())) < 2:
            continue
        train_classes = [c for c in classes if c in set(ya[train].tolist())]
        params = fit_softmax(X[train], ya[train].tolist(), train_classes, l2=l2)
        P = predict_proba(params, X[test])
        for i, row in zip(np.where(test)[0], P):
            preds[i] = (train_classes[int(np.argmax(row))], float(np.max(row)))
    return preds


# ------------------------------------------------------------------------------
# Registro, aplicação, dataset, treino e promoção
# ------------------------------------------------------------------------------

def get_models_dir() -> str:
    return model_registry.get_models_dir(MODEL_FAMILY)


def save_model(model: Dict[str, Any], models_dir: Optional[str] = None) -> str:
    path = os.path.join(models_dir or get_models_dir(), f"technique_{model['trained_at'].replace(':', '').replace('-', '')}.json")
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


class TechniqueClassifier:
    """Aplicado na análise de luta: técnica de cada golpe pelo modelo, com as probabilidades."""

    def __init__(self, model: Dict[str, Any], path: Optional[str] = None):
        from src.analytics.pose_encoder import PoseEncoder
        self.model = model
        self.path = path
        self.encoder = PoseEncoder.from_path(model.get("encoder"))

    @classmethod
    def load_active(cls, models_dir: Optional[str] = None) -> Optional["TechniqueClassifier"]:
        path = model_registry.get_active_model_path(MODEL_FAMILY, models_dir)
        model = load_model(path) if path else None
        return cls(model, path) if model else None

    def predict(self, attacker_history, defender_history, impact_frame: int, fps: float) -> Dict[str, Any]:
        from src.analytics.pose_encoder import with_embedding
        feats = extract_technique_features(attacker_history, defender_history, impact_frame, fps)
        feats = with_embedding(feats, self.encoder, attacker_history, defender_history, impact_frame)
        p = predict_proba(self.model["params"], features_matrix([feats], self.model["features"]))[0]
        classes = self.model["params"]["classes"]
        best = int(np.argmax(p))
        return {"tecnica": classes[best], "probabilidade": round(float(p[best]), 4),
                "probabilidades": {c: round(float(v), 4) for c, v in zip(classes, p)}}


def build_dataset(pipeline, videos_dir: str, labels_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Golpes com técnica conhecida, pelo BANCO ÚNICO (I11): a técnica informada numa revisão ou a do ponto oficial
    que o golpe casou no tempo (gabarito manual; é a técnica tentada, não um juízo de validade). Gabarito automático antigo não tem
    técnica conhecida; lutas golden ficam de fora.
    """
    from src.engine import weak_labels as wl
    from src.engine.label_bank import build_training_reports
    from src.analytics.pose_encoder import PoseEncoder, with_embedding
    encoder = PoseEncoder.load_active()
    training = build_training_reports(pipeline, videos_dir, labels_path=labels_path)
    rows, y, groups = [], [], []
    skipped = list(training["skipped"])
    for report in training["reports"]:
        official = {} if report.get("origem_gabarito") == "automatico" else \
            {p["index"]: wl.normalize_technique(p.get("technique")) for p in report["official_points"]}
        if report.get("origem_gabarito") == "automatico":
            skipped.append({"video_name": report["video_name"], "motivo": "técnica desconhecida (pontos automáticos)"})
        examples = []
        for cand in report["candidates"]["items"]:
            # Técnica tentada, válida ou não: a correção do revisor ou a do ponto oficial que o golpe casou.
            # Golpe que a revisão disse que nem é golpe fica de fora.
            if cand.get("nao_e_golpe"):
                continue
            tech = wl.normalize_technique(cand.get("tecnica_corrigida")) or official.get(cand.get("matched_point"))
            if tech:
                examples.append((cand, tech))
        if not examples:
            continue
        invert = bool(report.get("inverter_aka_shiro", False))
        features = pipeline.extract_match_features(report["video_path"], invert_combatants=invert)
        tracker = pipeline.combatant_tracker
        hist = {
            "KENSHI_AKA": tracker.interpolate_missing_poses(features["aka_history"], max_gap=10),
            "KENSHI_SHIRO": tracker.interpolate_missing_poses(features["shiro_history"], max_gap=10),
        }
        for cand, tech in examples:
            atk = cand["attacker_id"]
            dfd = "KENSHI_SHIRO" if atk == "KENSHI_AKA" else "KENSHI_AKA"
            base = extract_technique_features(hist[atk], hist[dfd], cand["impact_frame"], features["fps"])
            rows.append(with_embedding(base, encoder, hist[atk], hist[dfd], cand["impact_frame"]))
            y.append(tech)
            groups.append(report["video_name"])
    return {"rows": rows, "y": y, "groups": groups, "skipped": skipped, "encoder": encoder.path if encoder else None}


def train_model(dataset: Dict[str, Any], l2: float = 1.0) -> Dict[str, Any]:
    counts = {c: dataset["y"].count(c) for c in TECHNIQUES}
    classes = [c for c in TECHNIQUES if counts[c] >= MIN_PER_CLASS]
    if len(classes) < MIN_CLASSES:
        raise ValueError(f"Dados insuficientes para treinar: {counts} (mínimo {MIN_PER_CLASS} golpes em pelo menos "
                         f"{MIN_CLASSES} técnicas). Preencha mais lutas no gabarito, com a técnica de cada ponto.")
    from src.analytics.pose_encoder import feature_names_for
    keep = [i for i, label in enumerate(dataset["y"]) if label in classes]
    rows = [dataset["rows"][i] for i in keep]
    y = [dataset["y"][i] for i in keep]
    groups = [dataset["groups"][i] for i in keep]

    def _cv(names):
        X = features_matrix(rows, names)
        oof = group_cross_validate(X, y, groups, classes, l2=l2)
        judged = [(p, t) for p, t in zip(oof, y) if p is not None]
        return X, (round(sum(1 for p, t in judged if p == t) / len(judged), 4) if judged else None)

    # Com encoder de pose (I10) disponível, testa as duas versões e fica com a que validar melhor por vídeo
    variants = {"sem_encoder": feature_names_for(FEATURE_NAMES, False)}
    if dataset.get("encoder"):
        variants["com_encoder"] = feature_names_for(FEATURE_NAMES, True)
    results = {k: _cv(names) for k, names in variants.items()}
    chosen = max(results, key=lambda k: (results[k][1] or 0.0, k == "sem_encoder"))
    X, cv_acc = results[chosen]
    baseline = round(max(counts[c] for c in classes) / len(y), 4)  # sempre chutar a técnica mais comum
    return {
        "schema": MODEL_SCHEMA_VERSION,
        "tipo": "regressao_logistica_multinomial_l2",
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "features": variants[chosen],
        "encoder": dataset.get("encoder") if chosen == "com_encoder" else None,
        "variantes_validadas": {k: v[1] for k, v in results.items()},
        "params": fit_softmax(X, y, classes, l2=l2),
        "dados": {"golpes": len(y), "por_tecnica": {c: counts[c] for c in classes}, "videos": sorted(set(groups))},
        "validacao_por_video": {"acuracia": cv_acc, "chute_mais_comum": baseline},
    }


def evaluate_on_golden(pipeline, videos_dir: str, model: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    from src.engine.active_learning import GoldenBenchmark
    previous = getattr(pipeline, "technique_classifier", None)
    pipeline.technique_classifier = TechniqueClassifier(model) if model else None
    try:
        return GoldenBenchmark().run_matches(pipeline, videos_dir, record_history=False, update_samples=False)
    finally:
        pipeline.technique_classifier = previous


def promote_if_better(pipeline, videos_dir: str, model_path: str) -> Dict[str, Any]:
    """Ativa o modelo só se a acurácia de técnica no golden subir sem piorar o F1 dos pontos."""
    model = load_model(model_path)
    if model is None:
        raise ValueError("Modelo inválido ou de outra versão de features.")
    active = TechniqueClassifier.load_active()
    current = evaluate_on_golden(pipeline, videos_dir, active.model if active else None)["total"]
    challenger = evaluate_on_golden(pipeline, videos_dir, model)["total"]
    cur, new = current.get("acuracia_tecnica"), challenger.get("acuracia_tecnica")
    f1_cur, f1_new = current.get("f1") or 0.0, challenger.get("f1") or 0.0
    if not challenger.get("tecnica_avaliada"):
        ok, reason = False, "Sem golpes com técnica oficial nas lutas golden: congele lutas com gabarito antes de promover."
    elif f1_new < f1_cur:
        ok, reason = False, f"Piorou o F1 dos pontos no golden ({f1_cur:.3f} → {f1_new:.3f})."
    elif cur is None or new > cur:
        ok, reason = True, f"Acurácia de técnica no golden {('—' if cur is None else f'{cur:.3f}')} → {new:.3f}."
    else:
        ok, reason = False, f"Não melhorou a técnica no golden ({cur:.3f} → {new:.3f})."
    if ok:
        model_registry.set_active_model(MODEL_FAMILY, model_path, reason)
    return {"promovido": ok, "motivo": reason, "atual": current, "desafiante": challenger}


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Classificador de técnica (MEN, KOTE, DO, TSUKI).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_tr = sub.add_parser("train")
    p_tr.add_argument("videos_dir")
    p_tr.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("videos_dir")
    p_ev.add_argument("--model")
    p_ev.add_argument("--promote", action="store_true")
    p_ev.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    sub.add_parser("status")
    sub.add_parser("deactivate")
    args = parser.parse_args(argv)

    if args.command == "status":
        print(f"Ativo: {model_registry.get_active_model_path(MODEL_FAMILY) or 'nenhum (regra fixa do EventSpotter)'}")
        return 0
    if args.command == "deactivate":
        model_registry.set_active_model(MODEL_FAMILY, None, "Desativado pela linha de comando.")
        return 0
    from src.pipeline import SenpAIPipeline
    pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=args.device)
    pipeline.technique_classifier = None
    if args.command == "train":
        try:
            model = train_model(build_dataset(pipeline, args.videos_dir))
        except ValueError as e:
            print(e)
            return 1
        path = save_model(model)
        v = model["validacao_por_video"]
        print(f"Modelo: {path}\n  {model['dados']['golpes']} golpes {model['dados']['por_tecnica']} · validação por vídeo: "
              f"acurácia {_fmt(v['acuracia'])} (chutar a mais comum daria {_fmt(v['chute_mais_comum'])})")
        return 0
    path = args.model or max((os.path.join(get_models_dir(), f) for f in os.listdir(get_models_dir()) if f.startswith("technique_")), default=None)
    if not path:
        print("Nenhum modelo treinado.")
        return 1
    if args.promote:
        res = promote_if_better(pipeline, args.videos_dir, path)
        print(f"{'PROMOVIDO' if res['promovido'] else 'NÃO promovido'}: {res['motivo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
