"""
Rotulagem Fraca a partir do Placar Oficial (Pendência I2 — Rotulagem barata e em escala).

Em vez de revisar cada detecção à mão, o usuário informa só os pontos oficiais de cada luta
(kenshi, técnica e o instante aproximado no vídeo) num gabarito JSON. A partir do cache de
features (I1), o SenpAI reanalisa o vídeo em segundos e rotula automaticamente cada detecção:

- POSITIVO              casa com um ponto oficial (mesmo kenshi, dentro da tolerância de tempo)
- NEGATIVO_FORA_DA_LUTA antes do sonkyō inicial ou depois do final
- NEGATIVO_TSUBAZERIAI  disparo classificado como tsubazeriai e sem ponto oficial perto
- NEGATIVO_SEM_PONTO    nenhum ponto oficial perto (só quando o gabarito tem todos os pontos)
- AMBIGUO               perto de um ponto oficial que já casou com outra detecção, ou com o kenshi
                        trocado; vai para revisão humana (fila de curadoria ativa)
- DESCONHECIDO          gabarito incompleto e sem ponto perto; não vira rótulo

Pontos oficiais que nenhuma detecção encontrou são registrados como "perdidos" (recall do detector).
As métricas (precision_recall_f1, match_metrics, aggregate_match_metrics) são a fonte única usada
também pelo GoldenBenchmark (src/engine/active_learning.py).

Rótulos gerados em dois níveis:
- candidatos: todos os disparos brutos do EventSpotter (sem corte do sonkyō), base do filtro da I3;
- eventos: o que a análise de luta avaliou, que vira TP/FP no feedback_dataset (com sub-notas reais).

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.engine.weak_labels template "<pasta de vídeos>"     # cria/completa o gabarito
    python -m src.engine.weak_labels run "<pasta de vídeos>" [--save-feedback] [--enqueue-ambiguous]
    python -m src.engine.weak_labels golden list | freeze "<vídeo>" | unfreeze "<vídeo>" | run "<pasta de vídeos>"
"""

import os
import re
import json
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir
from src.utils.file_utils import safe_write_json, safe_load_json

DEFAULT_LABELS_PATH = os.path.join("data", "match_labels", "gabarito_lutas.json")
DEFAULT_REPORTS_DIR = os.path.join("data", "weak_labels")
DEFAULT_TOLERANCE_SEC = 1.5
WEAK_LABEL_NOTE = "Rótulo fraco (placar oficial)"

KENSHI_IDS = {"AKA": "KENSHI_AKA", "SHIRO": "KENSHI_SHIRO"}
# Pontos que não nascem de um golpe (não há detecção a casar)
NON_STRIKE_TECHNIQUES = {"HANSOKU", "FUSEN", "HANTEI"}

POSITIVE = "POSITIVO"
NEG_OUTSIDE = "NEGATIVO_FORA_DA_LUTA"
NEG_TSUBAZERIAI = "NEGATIVO_TSUBAZERIAI"
NEG_NO_POINT = "NEGATIVO_SEM_PONTO"
AMBIGUOUS = "AMBIGUO"
UNKNOWN = "DESCONHECIDO"
NEG_CORRECTION = "NEGATIVO_CORRECAO"   # um revisor/curador disse que não é ponto (banco de rótulos, I11)
NEGATIVE_LABELS = {NEG_OUTSIDE, NEG_TSUBAZERIAI, NEG_NO_POINT, NEG_CORRECTION}


# ------------------------------------------------------------------------------
# Gabarito
# ------------------------------------------------------------------------------

def get_reports_dir() -> str:
    """Pasta dos relatórios de rotulagem; em modo de teste usa a sandbox isolada."""
    reports_dir = os.path.join(get_test_sandbox_dir(), "weak_labels") if is_test_mode() else DEFAULT_REPORTS_DIR
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir


def parse_time_to_seconds(value: Any) -> float:
    """Aceita segundos (número) ou texto 'ss', 'mm:ss', 'mm:ss.mmm' ou 'hh:mm:ss'."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace(",", ".")
    if not text:
        raise ValueError("tempo vazio")
    parts = text.split(":")
    if len(parts) > 3 or not all(re.fullmatch(r"\d+(\.\d+)?", p) for p in parts):
        raise ValueError(f"tempo inválido: '{value}'")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60.0 + float(p)
    return seconds


def normalize_kenshi(value: Any) -> Optional[str]:
    """'AKA', 'aka', 'vermelho', 'KENSHI_AKA' → 'KENSHI_AKA'; vazio/desconhecido → None."""
    text = str(value or "").strip().upper()
    if text in ("AKA", "KENSHI_AKA", "VERMELHO", "RED"):
        return KENSHI_IDS["AKA"]
    if text in ("SHIRO", "KENSHI_SHIRO", "BRANCO", "WHITE"):
        return KENSHI_IDS["SHIRO"]
    return None


def _empty_entry() -> Dict[str, Any]:
    return {
        "aka": "",
        "shiro": "",
        "pontos_completos": False,
        "pontos": [],
        "observacoes": "Preencha os pontos oficiais: {\"kenshi\": \"AKA|SHIRO\", \"tecnica\": \"MEN|KOTE|DO|TSUKI|HANSOKU\", \"tempo\": \"mm:ss\"}. "
                       "Marque pontos_completos=true quando todos os pontos da luta estiverem listados."
    }


def _labels_file(path: str) -> str:
    """Em modo de teste, o gabarito padrão vai para a sandbox (nunca o gabarito real do usuário)."""
    from src.utils.test_isolation import get_safe_data_path
    return get_safe_data_path(path, "test_gabarito_lutas.json")


def load_match_labels(path: str = DEFAULT_LABELS_PATH) -> Dict[str, Any]:
    """Carrega o gabarito ({"videos": {nome_do_arquivo: entrada}}); retorna estrutura vazia se não existir."""
    path = _labels_file(path)
    data = safe_load_json(path, default={"versao": 1, "videos": {}})
    if not isinstance(data, dict):
        return {"versao": 1, "videos": {}}
    data.setdefault("videos", {})
    return data


def save_match_labels(data: Dict[str, Any], path: str = DEFAULT_LABELS_PATH) -> str:
    path = _labels_file(path)
    safe_write_json(path, data, indent=2)
    return path


def make_labels_template(video_paths: List[str], path: str = DEFAULT_LABELS_PATH) -> Tuple[str, int]:
    """Cria ou completa o gabarito com uma entrada vazia por vídeo; nunca apaga o que já foi preenchido."""
    data = load_match_labels(path)
    added = 0
    for video in video_paths:
        name = os.path.basename(video)
        if name not in data["videos"]:
            data["videos"][name] = _empty_entry()
            added += 1
    save_match_labels(data, path)
    return path, added


def parse_official_points(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normaliza os pontos de uma entrada do gabarito. Pontos sem golpe (hansoku etc.) ficam marcados."""
    points = []
    for idx, raw in enumerate(entry.get("pontos", []) or []):
        technique = str(raw.get("tecnica", "") or "").strip().upper()
        point = {
            "index": idx,
            "kenshi_id": normalize_kenshi(raw.get("kenshi")),
            "technique": technique or None,
            "time_sec": None,
            "is_strike": technique not in NON_STRIKE_TECHNIQUES
        }
        if point["is_strike"]:
            point["time_sec"] = parse_time_to_seconds(raw.get("tempo"))
        points.append(point)
    return points


def is_entry_ignored(entry: Optional[Dict[str, Any]]) -> bool:
    """Vídeo marcado como "não é luta" (ex.: gravação de treino): fica fora do gabarito, do cache e do treino."""
    return bool((entry or {}).get("ignorar"))


def is_entry_filled(entry: Dict[str, Any]) -> bool:
    """Uma luta só é rotulável se tiver pontos ou se o gabarito disser que acabou sem pontos (e não for ignorada)."""
    if is_entry_ignored(entry):
        return False
    return bool(entry.get("pontos")) or bool(entry.get("pontos_completos"))


# ------------------------------------------------------------------------------
# Casamento detecção × ponto oficial
# ------------------------------------------------------------------------------

def label_detections(
    detections: List[Dict[str, Any]],
    official_points: List[Dict[str, Any]],
    fps: float,
    match_start_frame: int,
    match_end_frame: int,
    points_complete: bool,
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC
) -> Dict[str, Any]:
    """
    Rotula cada detecção ({impact_frame, attacker_id, action_class?}) contra os pontos oficiais.
    O casamento é um-para-um, guloso pela menor diferença de tempo, e exige o mesmo kenshi
    quando o ponto oficial informa quem pontuou.
    """
    fps = float(fps) if fps and fps > 0 else 30.0
    strike_points = [p for p in official_points if p.get("is_strike") and p.get("time_sec") is not None]
    times = [float(d["impact_frame"]) / fps for d in detections]

    pairs = []
    for d_idx, det in enumerate(detections):
        for p in strike_points:
            dt = abs(times[d_idx] - p["time_sec"])
            if dt <= tolerance_sec and (p["kenshi_id"] is None or p["kenshi_id"] == det.get("attacker_id")):
                pairs.append((dt, d_idx, p["index"]))
    pairs.sort()

    det_match: Dict[int, int] = {}
    used_points = set()
    for dt, d_idx, p_idx in pairs:
        if d_idx in det_match or p_idx in used_points:
            continue
        det_match[d_idx] = p_idx
        used_points.add(p_idx)

    labeled = []
    for d_idx, det in enumerate(detections):
        t = times[d_idx]
        frame = int(det["impact_frame"])
        near_point = any(abs(t - p["time_sec"]) <= tolerance_sec for p in strike_points)
        if d_idx in det_match:
            label, reason = POSITIVE, f"casou com o ponto oficial #{det_match[d_idx] + 1}"
        elif frame < match_start_frame or frame > match_end_frame:
            label, reason = NEG_OUTSIDE, "fora da janela entre sonkyōs"
        elif near_point:
            label, reason = AMBIGUOUS, "perto de um ponto oficial já casado ou de outro kenshi"
        elif str(det.get("action_class") or "").upper() == "TSUBAZERIAI":
            label, reason = NEG_TSUBAZERIAI, "tsubazeriai sem ponto oficial perto"
        elif points_complete:
            label, reason = NEG_NO_POINT, "sem ponto oficial perto"
        else:
            label, reason = UNKNOWN, "gabarito incompleto"
        item = dict(det)
        item.update({
            "time_sec": round(t, 3),
            "weak_label": label,
            "reason": reason,
            "matched_point": det_match.get(d_idx)
        })
        labeled.append(item)

    missed = [p for p in strike_points if p["index"] not in used_points]
    counts: Dict[str, int] = {}
    for item in labeled:
        counts[item["weak_label"]] = counts.get(item["weak_label"], 0) + 1
    return {
        "items": labeled,
        "missed_points": missed,
        "counts": counts,
        "official_strike_points": len(strike_points)
    }


# ------------------------------------------------------------------------------
# Métricas (fonte única: rotulagem, golden benchmark e trava do otimizador usam estas funções)
# ------------------------------------------------------------------------------

def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    """Fórmula única de precisão, recall e F1 do SenpAI. None quando a métrica é indefinida (0/0)."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    rnd = lambda v: None if v is None else round(v, 4)
    return {"precisao": rnd(precision), "recall": rnd(recall), "f1": rnd(f1)}


def is_false_point(item: Dict[str, Any], points_complete: bool) -> bool:
    """
    Ponto dado pelo SenpAI que não é oficial. Com gabarito completo, ambíguo também conta
    (é ponto duplicado ou atribuído ao kenshi errado); com gabarito incompleto, só os negativos certos.
    """
    if not item.get("predicted_valid") or item["weak_label"] == POSITIVE:
        return False
    return item["weak_label"] in NEGATIVE_LABELS or (points_complete and item["weak_label"] == AMBIGUOUS)


def decision_is_known(item: Dict[str, Any], points_complete: bool) -> bool:
    """O rótulo diz com certeza se o golpe deveria virar ponto?"""
    label = item["weak_label"]
    return label == POSITIVE or label in NEGATIVE_LABELS or (points_complete and label == AMBIGUOUS)


def decision_is_correct(item: Dict[str, Any]) -> bool:
    """Acertou a decisão: deu ponto num golpe oficial, ou não deu ponto num golpe que não era ponto."""
    return bool(item.get("predicted_valid")) == (item["weak_label"] == POSITIVE)


def normalize_technique(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    for t in ("MEN", "KOTE", "DO", "TSUKI"):
        if text == t or text.startswith(t + " ") or text.startswith(t + "_"):
            return t
    return None


def technique_agreement(events: List[Dict[str, Any]], points: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Nos golpes que acertaram um ponto oficial: quantos têm técnica oficial conhecida e quantos acertaram a técnica."""
    by_index = {p["index"]: normalize_technique(p.get("technique")) for p in points}
    judged = correct = 0
    for e in events:
        if not e.get("predicted_valid") or e["weak_label"] != POSITIVE or e.get("matched_point") is None:
            continue
        official = by_index.get(e["matched_point"])
        if official is None:
            continue
        judged += 1
        correct += int(normalize_technique(e.get("strike_type")) == official)
    return judged, correct


def match_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Métrica única por luta.
    - Pontos do placar: acertos, falsos pontos, perdidos → precisão, recall e F1.
    - Decisões: cada golpe avaliado com rótulo conhecido → acurácia (ponto/não-ponto correto).
    """
    points = report["official_points"]
    complete = bool(report["pontos_completos"])
    events = report["events"]["items"]
    official = {k: sum(1 for p in points if p["kenshi_id"] == k) for k in KENSHI_IDS.values()}
    predicted = {k: sum(1 for e in events if e["attacker_id"] == k and e.get("predicted_valid")) for k in KENSHI_IDS.values()}
    hits = sum(1 for e in events if e.get("predicted_valid") and e["weak_label"] == POSITIVE)
    false_points = sum(1 for e in events if is_false_point(e, complete))
    strike_points = report["events"]["official_strike_points"]
    missed = strike_points - hits
    prf = precision_recall_f1(hits, false_points, missed)
    if not complete:
        prf["precisao"] = prf["f1"] = None  # falsos pontos desconhecidos sem gabarito completo
    score_error = sum(abs(official[k] - predicted[k]) for k in KENSHI_IDS.values())
    judged = [e for e in events if decision_is_known(e, complete)]
    correct = sum(1 for e in judged if decision_is_correct(e))
    tech_judged, tech_correct = technique_agreement(events, points)
    return {
        "video_name": report["video_name"],
        "pontos_completos": complete,
        "placar_oficial": {"aka": official["KENSHI_AKA"], "shiro": official["KENSHI_SHIRO"]},
        "placar_senpai": {"aka": predicted["KENSHI_AKA"], "shiro": predicted["KENSHI_SHIRO"]},
        "placar_exato": complete and score_error == 0,
        "erro_placar": score_error,
        "pontos_oficiais": strike_points,
        "acertos": hits,
        "falsos_pontos": false_points,
        "perdidos": missed,
        "candidatos": len(report["candidates"]["items"]),
        "candidatos_que_acham_ponto": report["candidates"]["counts"].get(POSITIVE, 0),
        "decisoes_avaliadas": len(judged),
        "decisoes_corretas": correct,
        "acuracia": round(correct / len(judged), 4) if judged and complete else None,
        "tecnica_avaliada": tech_judged,
        "tecnica_correta": tech_correct,
        "acuracia_tecnica": round(tech_correct / tech_judged, 4) if tech_judged else None,
        **prf,
    }


def aggregate_match_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Soma das lutas. Recall usa todas; acurácia, precisão e F1 só as lutas com gabarito completo."""
    keys = ("pontos_oficiais", "acertos", "falsos_pontos", "perdidos", "candidatos", "candidatos_que_acham_ponto", "erro_placar")
    tot: Dict[str, Any] = {k: sum(r[k] for r in rows) for k in keys}
    complete = [r for r in rows if r["pontos_completos"]]
    tot["lutas"] = len(rows)
    tot["lutas_completas"] = len(complete)
    tot["placares_exatos"] = sum(1 for r in rows if r["placar_exato"])
    full = precision_recall_f1(sum(r["acertos"] for r in complete), sum(r["falsos_pontos"] for r in complete), sum(r["perdidos"] for r in complete))
    tot["precisao"], tot["f1"] = full["precisao"], full["f1"]
    tot["recall"] = precision_recall_f1(tot["acertos"], 0, tot["perdidos"])["recall"]
    tech = sum(r.get("tecnica_avaliada", 0) for r in rows)
    tot["tecnica_avaliada"] = tech
    tot["acuracia_tecnica"] = round(sum(r.get("tecnica_correta", 0) for r in rows) / tech, 4) if tech else None
    judged = sum(r.get("decisoes_avaliadas", 0) for r in complete)
    tot["decisoes_avaliadas"] = judged
    tot["acuracia"] = round(sum(r.get("decisoes_corretas", 0) for r in complete) / judged, 4) if judged else None
    tot["recall_candidatos"] = round(tot["candidatos_que_acham_ponto"] / tot["pontos_oficiais"], 4) if tot["pontos_oficiais"] else None
    return tot


# ------------------------------------------------------------------------------
# Execução sobre um vídeo
# ------------------------------------------------------------------------------

def _raw_candidates(pipeline, features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Todos os disparos do EventSpotter, para os dois kenshi, sem corte de sonkyō nem deduplicação."""
    fps = features["fps"]
    tracker = pipeline.combatant_tracker
    candidates = []
    for kenshi_id, key in (("KENSHI_AKA", "aka_history"), ("KENSHI_SHIRO", "shiro_history")):
        history = tracker.interpolate_missing_poses(features[key], max_gap=10)
        # Mesmo critério da análise de luta: Shiro só é varrido com pelo menos 15 poses
        if kenshi_id == "KENSHI_SHIRO" and len([p for p in history if p]) < 15:
            continue
        for ev in pipeline.event_spotter.detect_strikes(
            history, fps=fps, start_bound_frame=0, end_bound_frame=None,
            attacker_id=kenshi_id, attacker_name=kenshi_id, filter_out_of_bounds=False
        ):
            candidates.append({
                "impact_frame": ev.impact_frame,
                "start_frame": ev.start_frame,
                "end_frame": ev.end_frame,
                "attacker_id": kenshi_id,
                "strike_type": ev.type
            })
    candidates.sort(key=lambda c: c["impact_frame"])
    return candidates


def _analyzed_events(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    events = []
    for ev in result.get("events", []):
        info = ev.get("event_info", {})
        evaluation = ev.get("evaluation", {})
        events.append({
            "impact_frame": int(info.get("impact_frame", 0)),
            "attacker_id": info.get("attacker_id"),
            "strike_type": info.get("type") or evaluation.get("strike_type"),
            "timestamp": info.get("timestamp"),
            "action_class": info.get("action_spotting_class"),
            "predicted_valid": bool(evaluation.get("is_valid", False)),
            "total_score": evaluation.get("total_score", 0.0),
            "sub_scores": evaluation.get("sub_scores", {})
        })
    return events


def weak_label_video(
    pipeline,
    video_path: str,
    entry: Dict[str, Any],
    tolerance_sec: float = DEFAULT_TOLERANCE_SEC,
    invert_combatants: bool = False
) -> Dict[str, Any]:
    """Reanalisa o vídeo (via cache de features) e rotula candidatos e eventos contra o gabarito."""
    points = parse_official_points(entry)
    points_complete = bool(entry.get("pontos_completos"))

    features = pipeline.extract_match_features(video_path, invert_combatants=invert_combatants)
    result = pipeline.process_video(
        video_path,
        invert_combatants=invert_combatants,
        use_feature_cache=True,
        record_learning_side_effects=False
    )
    if result is None:
        raise RuntimeError("análise interrompida")

    sonkyo = result.get("sonkyo_analysis", {})
    start_f = int(sonkyo.get("match_start_frame", 0))
    end_f = int(sonkyo.get("match_end_frame", features["total_frames"] - 1))
    fps = features["fps"]

    candidates = label_detections(_raw_candidates(pipeline, features), points, fps, start_f, end_f, points_complete, tolerance_sec)
    events = label_detections(_analyzed_events(result), points, fps, start_f, end_f, points_complete, tolerance_sec)

    report = {
        "video_name": os.path.basename(video_path),
        "video_path": video_path,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cache_key": features.get("cache_key"),
        "fps": fps,
        "match_window_frames": [start_f, end_f],
        "pontos_completos": points_complete,
        "tolerance_sec": tolerance_sec,
        "official_points": points,
        "candidates": candidates,
        "events": events,
    }
    report["metricas"] = match_metrics(report)
    return report


def save_report(report: Dict[str, Any]) -> str:
    stem = os.path.splitext(report["video_name"])[0]
    safe = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")[:80] or "video"
    path = os.path.join(get_reports_dir(), f"{safe}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def export_to_feedback(report: Dict[str, Any], feedback_manager=None, profile_key: str = "normal") -> Dict[str, int]:
    """
    Grava os eventos rotulados no feedback_dataset: POSITIVO → TP, negativos → FP.
    Ambíguos e desconhecidos ficam de fora. Pontos perdidos (FN) não são gravados até a I4,
    porque hoje o otimizador inventa sub-notas para FN sem sub_scores (P5.3).
    """
    from src.engine.active_learning import is_golden_video
    if is_golden_video(report["video_name"]):
        log_event("WARNING", f"Luta do golden benchmark não vai para o feedback: {report['video_name']}", "weak_labels")
        return {"TP": 0, "FP": 0, "golden": True}
    if feedback_manager is None:
        from src.engine.feedback_manager import FeedbackManager
        feedback_manager = FeedbackManager()
    saved = {"TP": 0, "FP": 0}
    for ev in report["events"]["items"]:
        if ev["weak_label"] == POSITIVE:
            label = "TP"
        elif ev["weak_label"] in NEGATIVE_LABELS:
            label = "FP"
        else:
            continue
        feedback_manager.save_feedback(
            video_name=report["video_name"],
            profile_key=profile_key,
            event_id=f"weak_{ev['attacker_id']}_{ev['impact_frame']}",
            label=label,
            sub_scores=ev.get("sub_scores") or {},
            total_score=float(ev.get("total_score") or 0.0),
            strike_type=str(ev.get("strike_type") or "MEN").upper(),
            timestamp=ev.get("timestamp") or f"{ev['time_sec']:.3f}s",
            notes=f"{WEAK_LABEL_NOTE}: {ev['weak_label']} — {ev['reason']}",
            decision_category="rotulo_fraco",
            enqueue_uncertain=False
        )
        saved[label] += 1
    return saved


def enqueue_ambiguous(report: Dict[str, Any], sampler=None, profile_key: str = "normal") -> int:
    """Envia os eventos ambíguos para a fila de curadoria ativa, com o corte do vídeo quando possível."""
    from src.engine.active_learning import is_golden_video
    ambiguous = [e for e in report["events"]["items"] if e["weak_label"] == AMBIGUOUS]
    if not ambiguous or is_golden_video(report["video_name"]):
        return 0  # luta golden nunca alimenta a curadoria (que vira treino)
    if sampler is None:
        from src.engine.active_learning import UncertaintySampler
        sampler = UncertaintySampler()
    from src.utils.strike_clip_manager import extract_strike_subclip, DEFAULT_CLIPS_DIR
    count = 0
    for ev in ambiguous:
        clip = None
        try:
            clip_out = os.path.join(DEFAULT_CLIPS_DIR, f"unc_weak_{int(time.time() * 1000)}_{ev['impact_frame']}_corte.mp4")
            clip = extract_strike_subclip(report["video_path"], clip_out, target_time_sec=ev["time_sec"])
        except Exception as e:
            log_event("WARNING", f"Corte do lance ambíguo falhou: {e}", "weak_labels")
        enqueued = sampler.evaluate_and_enqueue(
            strike_data={
                "strike_type": ev.get("strike_type"),
                "scores": ev.get("sub_scores", {}),
                "sub_scores": ev.get("sub_scores", {}),
                "total_score": ev.get("total_score", 50.0),
                "timestamp_seconds": ev["time_sec"],
                "timestamp": ev.get("timestamp"),
                "impact_frame": ev["impact_frame"],
                "notes": f"{WEAK_LABEL_NOTE}: {ev['reason']}"
            },
            confidence=0.5,  # força a faixa de incerteza: quem decide é o revisor
            video_source=report["video_path"],
            profile_name=profile_key,
            enrich_with_llm=False,
            clip_path=clip
        )
        if enqueued:
            count += 1
    return count


# ------------------------------------------------------------------------------
# Linha de comando
# ------------------------------------------------------------------------------

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    from src.utils.feature_cache import list_videos

    parser = argparse.ArgumentParser(description="Rotulagem fraca de lutas a partir do placar oficial.")
    parser.add_argument("--labels", default=DEFAULT_LABELS_PATH, help="Arquivo do gabarito (JSON).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_tpl = sub.add_parser("template", help="Cria/completa o gabarito com os vídeos da pasta.")
    p_tpl.add_argument("target")
    p_run = sub.add_parser("run", help="Rotula os vídeos que já têm pontos no gabarito.")
    p_run.add_argument("target")
    p_run.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_SEC)
    p_run.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p_run.add_argument("--save-feedback", action="store_true", help="Grava TP/FP no feedback_dataset.")
    p_run.add_argument("--enqueue-ambiguous", action="store_true", help="Envia ambíguos à fila de curadoria.")
    p_gold = sub.add_parser("golden", help="Golden benchmark (GoldenBenchmark): lutas congeladas, fora do treino.")
    gsub = p_gold.add_subparsers(dest="golden_command", required=True)
    gsub.add_parser("list", help="Lista as lutas do golden e a fonte das amostras da trava.")
    g_fr = gsub.add_parser("freeze", help="Congela lutas do gabarito no golden.")
    g_fr.add_argument("videos", nargs="+", help="Nome do arquivo de vídeo, como no gabarito.")
    g_un = gsub.add_parser("unfreeze", help="Remove lutas do golden.")
    g_un.add_argument("videos", nargs="+")
    g_run = gsub.add_parser("run", help="Roda o benchmark nas lutas golden.")
    g_run.add_argument("target", help="Pasta dos vídeos.")
    g_run.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_SEC)
    g_run.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = parser.parse_args(argv)

    if args.command == "golden":
        return _golden_cli(args)

    videos = list_videos(args.target)
    if args.command == "template":
        path, added = make_labels_template(videos, args.labels)
        print(f"Gabarito: {os.path.abspath(path)} ({added} vídeos novos)")
        return 0

    labels = load_match_labels(args.labels)
    todo = [(v, labels["videos"][os.path.basename(v)]) for v in videos
            if os.path.basename(v) in labels["videos"] and is_entry_filled(labels["videos"][os.path.basename(v)])]
    if not todo:
        print("Nenhum vídeo da pasta tem pontos preenchidos no gabarito.")
        return 1

    from src.pipeline import SenpAIPipeline
    pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=args.device)
    reports, failures = [], 0
    for idx, (video, entry) in enumerate(todo, start=1):
        t0 = time.time()
        try:
            report = weak_label_video(pipeline, video, entry, tolerance_sec=args.tolerance,
                                      invert_combatants=bool(entry.get("inverter_aka_shiro", False)))
            path = save_report(report)
            extra = ""
            if args.save_feedback:
                fb = export_to_feedback(report)
                extra += " · luta do golden: fora do feedback" if fb.get("golden") else f" · feedback {fb}"
            if args.enqueue_ambiguous:
                extra += f" · {enqueue_ambiguous(report)} na fila"
            s = report["metricas"]
            print(f"[{idx}/{len(todo)}] {time.time() - t0:.1f}s — {report['video_name']}\n"
                  f"    placar oficial {s['placar_oficial']} × SenpAI {s['placar_senpai']} · "
                  f"candidatos {report['candidates']['counts']} · eventos {report['events']['counts']}{extra}\n"
                  f"    relatório: {path}")
            reports.append(report)
        except Exception as e:
            failures += 1
            print(f"[{idx}/{len(todo)}] FALHA — {os.path.basename(video)}: {e}")

    if reports:
        print("Resumo:", json.dumps(aggregate_match_metrics([r["metricas"] for r in reports]), ensure_ascii=False))
    return 1 if failures else 0


def _fmt_pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def _golden_cli(args) -> int:
    from src.engine.active_learning import GoldenBenchmark
    gb = GoldenBenchmark()

    if args.golden_command == "list":
        matches = gb.load_matches()
        src_info = gb.samples_source()
        print(f"{len(matches)} luta(s) no golden · trava usando amostras {src_info['fonte']} "
              f"({src_info['golpes_reais']} golpes reais; mínimo {src_info['minimo_reais']}).")
        for name, e in sorted(matches.items()):
            print(f"  - {name} ({len(e.get('pontos') or [])} pontos, congelada em {e.get('congelado_em')})")
        return 0

    if args.golden_command == "freeze":
        labels = load_match_labels(args.labels)["videos"]
        status = 0
        for name in args.videos:
            try:
                gb.freeze_match(name, labels.get(os.path.basename(name), {}))
                print(f"Congelada: {name}")
            except ValueError as e:
                status = 1
                print(f"Não congelada: {name}: {e}")
        return status

    if args.golden_command == "unfreeze":
        for name in args.videos:
            print(("Removida: " if gb.unfreeze_match(name) else "Não estava no golden: ") + name)
        return 0

    from src.pipeline import SenpAIPipeline
    pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=args.device)
    res = gb.run_matches(pipeline, args.target, tolerance_sec=args.tolerance,
                         progress=lambda i, n, name: print(f"[{i}/{n}] {name}"))
    for r in res["lutas"]:
        print(f"  {r['video_name']}: oficial {r['placar_oficial']} × SenpAI {r['placar_senpai']} · "
              f"acertos {r['acertos']} · falsos {r['falsos_pontos']} · perdidos {r['perdidos']}")
    t = res["total"]
    print(f"TOTAL {t['lutas']} lutas · placares exatos {t['placares_exatos']} · precisão {_fmt_pct(t['precisao'])} · "
          f"recall {_fmt_pct(t['recall'])} · F1 {_fmt_pct(t['f1'])} · detector bruto acha {_fmt_pct(t['recall_candidatos'])} · "
          f"{res['golpes_reais']} golpes reais")
    for name in res["videos_ausentes"]:
        print(f"  ausente na pasta: {name}")
    for f in res["falhas"]:
        print(f"  FALHA {f['video_name']}: {f['erro']}")
    return 1 if res["falhas"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
