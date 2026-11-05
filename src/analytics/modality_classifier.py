"""
Classificador Luta × 14 Modalidades de Treino (Pendência I14).

Substitui a regra fixa de `TrainingAnalyzer.detect_training_modality` (que devolvia confianças escritas
à mão, como 92% ou 88%) por um modelo aprendido. Ele classifica cada TRECHO de ~10 s de vídeo como
"luta" ou uma das 14 modalidades (suburi, ashi_sabaki, kirikaeshi...), olhando o grupo de praticantes:
quantos estão ativos, se estão em dupla, cadência e ritmo dos golpes, trabalho de pés, braços acima da
cabeça, pausas e, opcionalmente, o encoder de movimento (I10).

- Rótulos: o Banco Único (I11), com as correções de trechos do Modo Treinamento (I13) e "luta" para
  cada luta com gabarito.
- Validação deixando um vídeo de fora por vez; modelo softmax compartilhado com o classificador de técnica (I9).
- Promoção só se a validação superar o chute da classe mais comum e o modelo atual.
- Sem modelo ativo, a interface mostra "confiança não medida".

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.analytics.modality_classifier train "<pasta de vídeos>" [--promote]
    python -m src.analytics.modality_classifier status | deactivate
"""

import os
import json
import math
import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.analytics.technique_classifier import fit_softmax, predict_proba, group_cross_validate
from src.utils import model_registry

MODEL_FAMILY = "modality_classifier"
SCHEMA_VERSION = 1
LUTA = "luta"
WINDOW_SEC = 10.0
STRIDE_SEC = 5.0
MIN_WINDOWS_PER_CLASS = 5
MIN_CLASSES = 2

FEATURE_NAMES = [
    "praticantes_ativos", "praticante_solo", "pulso_vel_media", "pulso_vel_p90", "tornozelo_vel_media",
    "quadril_vel_media", "razao_pulso_tornozelo", "bracos_acima_cabeca", "parado", "golpes_por_minuto",
    "ritmo_irregular", "distancia_parceiro", "tempo_com_parceiro",
]


# ------------------------------------------------------------------------------
# Features de um trecho
# ------------------------------------------------------------------------------

def _xy(pose, name):
    lm = pose.get(name) if pose else None
    if isinstance(lm, dict):
        if "px" in lm:
            return float(lm["px"]), float(lm["py"])
        if "x" in lm:
            return float(lm["x"]), float(lm["y"])
    return None


def _mid(a, b):
    if a is None or b is None:
        return a or b
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _series(history, start, end, getter):
    return [getter(history[i]) if 0 <= i < len(history) and history[i] else None for i in range(start, end)]


def _speed(points, fps, scale):
    out = []
    for a, b in zip(points, points[1:]):
        out.append(None if a is None or b is None else math.hypot(b[0] - a[0], b[1] - a[1]) * fps / scale)
    return [v for v in out if v is not None]


def window_features(histories: Sequence[Sequence[Optional[dict]]], start: int, end: int, fps: float) -> Dict[str, float]:
    """Features do grupo de praticantes no trecho [start, end) (quadros)."""
    fps = float(fps) if fps and fps > 0 else 30.0
    n = max(1, end - start)
    active = []
    for h in histories:
        present = sum(1 for i in range(start, end) if 0 <= i < len(h) and h[i])
        if present / n >= 0.5:
            active.append(h)
    feats = {k: 0.0 for k in FEATURE_NAMES}
    feats["praticantes_ativos"] = float(min(len(active), 6))
    feats["praticante_solo"] = 1.0 if len(active) == 1 else 0.0
    if not active:
        return feats

    per = {k: [] for k in ("w", "w90", "a", "h", "raised", "still", "spm", "irregular")}
    hips_by_track = []
    torsos = []
    for h in active:
        sh = _series(h, start, end, lambda p: _mid(_xy(p, "LEFT_SHOULDER"), _xy(p, "RIGHT_SHOULDER")))
        hp = _series(h, start, end, lambda p: _mid(_xy(p, "LEFT_HIP"), _xy(p, "RIGHT_HIP")))
        torso_vals = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(sh, hp) if a and b]
        torso = float(np.median(torso_vals)) if torso_vals else 0.0
        if torso < 1e-6:
            continue
        torsos.append(torso)
        hips_by_track.append(hp)
        wr = _series(h, start, end, lambda p: _mid(_xy(p, "LEFT_WRIST"), _xy(p, "RIGHT_WRIST")))
        an = _series(h, start, end, lambda p: _mid(_xy(p, "LEFT_ANKLE"), _xy(p, "RIGHT_ANKLE")))
        head = _series(h, start, end, lambda p: _mid(_xy(p, "LEFT_EAR"), _xy(p, "RIGHT_EAR")))
        ws, as_, hs = _speed(wr, fps, torso), _speed(an, fps, torso), _speed(hp, fps, torso)
        if ws:
            per["w"].append(float(np.mean(ws)))
            per["w90"].append(float(np.percentile(ws, 90)))
            thr = float(np.mean(ws) + 1.5 * np.std(ws))
            peaks = [i for i in range(1, len(ws) - 1) if ws[i] > thr and ws[i] >= ws[i - 1] and ws[i] >= ws[i + 1]]
            merged = []
            for i in peaks:
                if not merged or i - merged[-1] > 0.3 * fps:
                    merged.append(i)
            per["spm"].append(len(merged) / (n / fps) * 60.0)
            gaps = np.diff(merged) if len(merged) > 2 else []
            per["irregular"].append(float(np.std(gaps) / np.mean(gaps)) if len(gaps) and np.mean(gaps) > 0 else 0.0)
        if as_:
            per["a"].append(float(np.mean(as_)))
        if hs:
            per["h"].append(float(np.mean(hs)))
        raised = [1.0 if w and hd and w[1] < hd[1] else 0.0 for w, hd in zip(wr, head) if w and hd]
        if raised:
            per["raised"].append(float(np.mean(raised)))
        if ws and hs:
            still = [1.0 if a < 0.3 and b < 0.3 else 0.0 for a, b in zip(ws, hs)]
            per["still"].append(float(np.mean(still)))

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    feats["pulso_vel_media"] = mean(per["w"])
    feats["pulso_vel_p90"] = mean(per["w90"])
    feats["tornozelo_vel_media"] = mean(per["a"])
    feats["quadril_vel_media"] = mean(per["h"])
    feats["razao_pulso_tornozelo"] = feats["pulso_vel_media"] / max(feats["tornozelo_vel_media"], 1e-3)
    feats["bracos_acima_cabeca"] = mean(per["raised"])
    feats["parado"] = mean(per["still"])
    feats["golpes_por_minuto"] = mean(per["spm"])
    feats["ritmo_irregular"] = mean(per["irregular"])

    # Dupla: distância ao parceiro mais próximo (em troncos) e fração do tempo com alguém perto
    if len(hips_by_track) >= 2:
        torso = float(np.median(torsos))
        nearest, close = [], []
        for t in range(n):
            pts = [hp[t] for hp in hips_by_track if t < len(hp) and hp[t]]
            if len(pts) < 2:
                continue
            for i, a in enumerate(pts):
                d = min(math.hypot(a[0] - b[0], a[1] - b[1]) for j, b in enumerate(pts) if j != i) / torso
                nearest.append(d)
                close.append(1.0 if d < 2.5 else 0.0)
        feats["distancia_parceiro"] = mean(nearest)
        feats["tempo_com_parceiro"] = mean(close)
    return {k: (0.0 if not math.isfinite(v) else float(v)) for k, v in feats.items()}


def window_embedding(encoder, histories, start: int, end: int, fps: float) -> Optional[np.ndarray]:
    """Média do encoder de movimento (I10) a cada 1 s do trecho, para os 2 praticantes mais presentes."""
    if encoder is None:
        return None
    ranked = sorted(histories, key=lambda h: -sum(1 for i in range(start, end) if 0 <= i < len(h) and h[i]))[:2]
    if not ranked:
        return None
    step = max(1, int(fps))
    vecs = []
    for idx, h in enumerate(ranked):
        other = ranked[1 - idx] if len(ranked) == 2 else []
        for c in range(start, end, step):
            vecs.append(encoder.embed(h, other, c))
    return np.mean(vecs, axis=0) if vecs else None


def video_windows(n_frames: int, fps: float, start_s: float = 0.0, end_s: Optional[float] = None) -> List[Tuple[int, int]]:
    fps = float(fps) if fps and fps > 0 else 30.0
    a = int(start_s * fps)
    b = min(n_frames, int(end_s * fps)) if end_s is not None else n_frames
    w, s = int(WINDOW_SEC * fps), int(STRIDE_SEC * fps)
    if b - a < w:
        return [(a, b)] if b - a >= int(3 * fps) else []
    return [(i, i + w) for i in range(a, b - w + 1, s)]


# ------------------------------------------------------------------------------
# Poses de um vídeo (do cache de luta ou do Modo Treinamento)
# ------------------------------------------------------------------------------

def load_tracks(pipeline, video_path: str, prefer_match: bool) -> Optional[Tuple[List[list], float, int]]:
    """Históricos de pose dos praticantes, fps e nº de quadros. Usa os caches (I1/I13); extrai se faltar."""
    from src.utils import feature_cache
    key_t = feature_cache.build_cache_key(video_path, pipeline.vision_model, mode="training")
    cached_t = feature_cache.load_training_features(key_t)
    if cached_t is not None and not prefer_match:
        return [t["history"] for t in cached_t["tracks"].values()], cached_t["fps"], cached_t["total_frames"]
    try:
        f = pipeline.extract_match_features(video_path)
        return [f["aka_history"], f["shiro_history"]], f["fps"], f["total_frames"]
    except Exception:
        pass
    if cached_t is None:
        res = pipeline.process_video(video_path, analysis_mode="training", use_feature_cache=True, record_learning_side_effects=False)
        if res is None:
            return None
        cached_t = feature_cache.load_training_features(key_t)
    if cached_t is None:
        return None
    return [t["history"] for t in cached_t["tracks"].values()], cached_t["fps"], cached_t["total_frames"]


# ------------------------------------------------------------------------------
# Dataset, treino, registro e predição
# ------------------------------------------------------------------------------

def build_dataset(pipeline, videos_dir: str) -> Dict[str, Any]:
    """Trechos rotulados pelo Banco Único (I11/I13): correções de modalidade e "luta" das lutas com gabarito."""
    from src.engine.label_bank import LabelBank
    from src.engine.active_learning import is_golden_video
    from src.analytics.pose_encoder import PoseEncoder
    encoder = PoseEncoder.load_active()
    labels = LabelBank().modality_labels()
    rows, embs, y, groups, skipped = [], [], [], [], []
    by_video: Dict[str, List[Dict[str, Any]]] = {}
    for lb in labels:
        by_video.setdefault(lb["video_name"], []).append(lb)
    for name, lbs in sorted(by_video.items()):
        if is_golden_video(name):
            skipped.append({"video_name": name, "motivo": "golden"})
            continue
        from src.engine.internet_collector import resolve_video_path
        video = resolve_video_path(videos_dir, name)  # inclui os vídeos da coleta (I15)
        if not os.path.isfile(video):
            skipped.append({"video_name": name, "motivo": "vídeo ausente"})
            continue
        loaded = load_tracks(pipeline, video, prefer_match=all(lb["value"] == LUTA for lb in lbs))
        if loaded is None:
            skipped.append({"video_name": name, "motivo": "sem poses"})
            continue
        histories, fps, n_frames = loaded
        for lb in lbs:
            for a, b in video_windows(n_frames, fps, lb.get("time_sec") or 0.0, lb.get("fim_s")):
                rows.append(window_features(histories, a, b, fps))
                emb = window_embedding(encoder, histories, a, b, fps)
                embs.append(None if emb is None else emb.tolist())
                y.append(lb["value"])
                groups.append(name)
    return {"rows": rows, "embeddings": embs, "y": y, "groups": groups, "skipped": skipped,
            "encoder": encoder.path if encoder else None}


def _matrix(dataset: Dict[str, Any], idx: List[int], with_encoder: bool) -> np.ndarray:
    X = np.array([[dataset["rows"][i][k] for k in FEATURE_NAMES] for i in idx], dtype=float).reshape(-1, len(FEATURE_NAMES))
    if with_encoder:
        E = np.array([dataset["embeddings"][i] for i in idx], dtype=float)
        X = np.hstack([X, E])
    return X


def train_model(dataset: Dict[str, Any], l2: float = 1.0) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for c in dataset["y"]:
        counts[c] = counts.get(c, 0) + 1
    classes = sorted(c for c, n in counts.items() if n >= MIN_WINDOWS_PER_CLASS)
    if len(classes) < MIN_CLASSES or len(set(dataset["groups"])) < 2:
        raise ValueError(f"Dados insuficientes para treinar: {counts or 'nenhum trecho rotulado'} "
                         f"(mínimo {MIN_WINDOWS_PER_CLASS} trechos de 10 s em pelo menos {MIN_CLASSES} classes, de 2 vídeos). "
                         "Corrija modalidades no Modo Treinamento ou preencha o gabarito de lutas.")
    keep = [i for i, c in enumerate(dataset["y"]) if c in classes]
    y = [dataset["y"][i] for i in keep]
    groups = [dataset["groups"][i] for i in keep]
    has_emb = bool(dataset.get("encoder")) and all(dataset["embeddings"][i] is not None for i in keep)

    def _cv(with_encoder):
        X = _matrix(dataset, keep, with_encoder)
        oof = group_cross_validate(X, y, groups, classes, l2=l2)
        judged = [(p, t) for p, t in zip(oof, y) if p is not None]
        return X, (round(sum(1 for p, t in judged if p == t) / len(judged), 4) if judged else None)

    variants = {"sem_encoder": _cv(False)}
    if has_emb:
        variants["com_encoder"] = _cv(True)
    chosen = max(variants, key=lambda k: (variants[k][1] or 0.0, k == "sem_encoder"))
    X, cv_acc = variants[chosen]
    baseline = round(max(counts[c] for c in classes) / len(y), 4)
    return {
        "schema": SCHEMA_VERSION, "tipo": "regressao_logistica_multinomial_l2",
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "features": FEATURE_NAMES, "encoder": dataset.get("encoder") if chosen == "com_encoder" else None,
        "params": fit_softmax(X, y, classes, l2=l2),
        "dados": {"trechos": len(y), "por_classe": {c: counts[c] for c in classes}, "videos": sorted(set(groups))},
        "validacao_por_video": {"acuracia": cv_acc, "chute_mais_comum": baseline},
        "variantes_validadas": {k: v[1] for k, v in variants.items()},
    }


def save_model(model: Dict[str, Any], models_dir: Optional[str] = None) -> str:
    d = models_dir or model_registry.get_models_dir(MODEL_FAMILY)
    path = os.path.join(d, f"modality_{model['trained_at'].replace(':', '').replace('-', '')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    return path


def load_model(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            model = json.load(f)
        if model.get("schema") != SCHEMA_VERSION or model.get("features") != FEATURE_NAMES:
            return None
        if model.get("encoder") and not os.path.isfile(model["encoder"]):
            return None
        return model
    except Exception:
        return None


def promote_if_better(model_path: str, models_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Ativa se a validação por vídeo superar o chute da classe mais comum e o modelo atual.
    (Não há golden de modalidade ainda: a comparação usa a validação por vídeo registrada em cada modelo.)
    """
    model = load_model(model_path)
    if model is None:
        raise ValueError("Modelo inválido.")
    v = model["validacao_por_video"]
    current = load_model(model_registry.get_active_model_path(MODEL_FAMILY, models_dir))
    cur_acc = current["validacao_por_video"]["acuracia"] if current else None
    acc = v["acuracia"] or 0.0
    if acc <= v["chute_mais_comum"]:
        ok, reason = False, f"Não supera o chute da classe mais comum ({acc:.2f} ≤ {v['chute_mais_comum']:.2f})."
    elif cur_acc is not None and acc <= cur_acc:
        ok, reason = False, f"Não supera o modelo atual ({acc:.2f} ≤ {cur_acc:.2f})."
    else:
        ok, reason = True, f"Validação por vídeo {('—' if cur_acc is None else f'{cur_acc:.2f}')} → {acc:.2f} (chute: {v['chute_mais_comum']:.2f})."
    if ok:
        model_registry.set_active_model(MODEL_FAMILY, model_path, reason, models_dir)
    return {"promovido": ok, "motivo": reason}


class ModalityClassifier:
    """Aplica o modelo: trechos de 10 s → luta ou modalidade, com a probabilidade medida do modelo."""

    def __init__(self, model: Dict[str, Any], path: Optional[str] = None):
        from src.analytics.pose_encoder import PoseEncoder
        self.model = model
        self.path = path
        self.encoder = PoseEncoder.from_path(model.get("encoder"))

    @classmethod
    def load_active(cls, models_dir: Optional[str] = None) -> Optional["ModalityClassifier"]:
        path = model_registry.get_active_model_path(MODEL_FAMILY, models_dir)
        model = load_model(path)
        return cls(model, path) if model else None

    def predict_windows(self, histories, n_frames: int, fps: float) -> List[Dict[str, Any]]:
        out = []
        classes = self.model["params"]["classes"]
        for a, b in video_windows(n_frames, fps):
            f = window_features(histories, a, b, fps)
            x = np.array([[f[k] for k in FEATURE_NAMES]], dtype=float)
            if self.model.get("encoder"):
                emb = window_embedding(self.encoder, histories, a, b, fps)
                x = np.hstack([x, np.array([emb if emb is not None else np.zeros(32)])])
            p = predict_proba(self.model["params"], x)[0]
            out.append({"inicio_s": a / fps, "fim_s": b / fps, "classe": classes[int(np.argmax(p))],
                        "probabilidade": float(np.max(p))})
        return out

    def predict_segments(self, histories, n_frames: int, fps: float) -> Dict[str, Any]:
        """Trechos consecutivos da mesma classe (suavizados por maioria de 3) e a classe dominante do vídeo."""
        wins = self.predict_windows(histories, n_frames, fps)
        if not wins:
            return {"segmentos": [], "dominante": None, "probabilidade": None}
        labels = [w["classe"] for w in wins]
        smooth = []
        for i in range(len(labels)):
            neigh = labels[max(0, i - 1):i + 2]
            smooth.append(max(set(neigh), key=neigh.count))
        segments = []
        for w, c in zip(wins, smooth):
            if segments and segments[-1]["classe"] == c:
                segments[-1]["fim_s"] = w["fim_s"]
                segments[-1]["_probs"].append(w["probabilidade"])
            else:
                segments.append({"classe": c, "inicio_s": w["inicio_s"], "fim_s": w["fim_s"], "_probs": [w["probabilidade"]]})
        for s in segments:
            s["probabilidade"] = round(float(np.mean(s.pop("_probs"))), 4)
        dur: Dict[str, float] = {}
        for s in segments:
            dur[s["classe"]] = dur.get(s["classe"], 0.0) + s["fim_s"] - s["inicio_s"]
        dominant = max(dur, key=dur.get)
        dom_prob = float(np.mean([s["probabilidade"] for s in segments if s["classe"] == dominant]))
        return {"segmentos": segments, "dominante": dominant, "probabilidade": round(dom_prob, 4)}


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Classificador luta × 14 modalidades.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_tr = sub.add_parser("train")
    p_tr.add_argument("videos_dir")
    p_tr.add_argument("--promote", action="store_true")
    p_tr.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    sub.add_parser("status")
    sub.add_parser("deactivate")
    args = parser.parse_args(argv)
    if args.command == "status":
        print(f"Ativo: {model_registry.get_active_model_path(MODEL_FAMILY) or 'nenhum (regra fixa; confiança não medida)'}")
        return 0
    if args.command == "deactivate":
        model_registry.set_active_model(MODEL_FAMILY, None, "Desativado pela linha de comando.")
        return 0
    from src.pipeline import SenpAIPipeline
    pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=args.device)
    try:
        model = train_model(build_dataset(pipeline, args.videos_dir))
    except ValueError as e:
        print(e)
        return 1
    path = save_model(model)
    v = model["validacao_por_video"]
    print(f"Modelo: {path}\n  {model['dados']['trechos']} trechos {model['dados']['por_classe']} · validação por vídeo {v['acuracia']} "
          f"(chute {v['chute_mais_comum']})")
    if args.promote:
        print(promote_if_better(path)["motivo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
