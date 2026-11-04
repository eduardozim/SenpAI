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

Rótulos gerados em dois níveis:
- candidatos: todos os disparos brutos do EventSpotter (sem corte do sonkyō), base do filtro da I3;
- eventos: o que a análise de luta avaliou, que vira TP/FP no feedback_dataset (com sub-notas reais).

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.engine.weak_labels template "<pasta de vídeos>"     # cria/completa o gabarito
    python -m src.engine.weak_labels run "<pasta de vídeos>" [--save-feedback] [--enqueue-ambiguous]
"""

import os
import re
import json
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir

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
NEGATIVE_LABELS = {NEG_OUTSIDE, NEG_TSUBAZERIAI, NEG_NO_POINT}


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


def load_match_labels(path: str = DEFAULT_LABELS_PATH) -> Dict[str, Any]:
    """Carrega o gabarito ({"videos": {nome_do_arquivo: entrada}}); retorna estrutura vazia se não existir."""
    if not os.path.isfile(path):
        return {"versao": 1, "videos": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("videos", {})
    return data


def save_match_labels(data: Dict[str, Any], path: str = DEFAULT_LABELS_PATH) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
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


def is_entry_filled(entry: Dict[str, Any]) -> bool:
    """Uma luta só é rotulável se tiver pontos ou se o gabarito disser que acabou sem pontos."""
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
    tp = counts.get(POSITIVE, 0)
    known_neg = sum(counts.get(k, 0) for k in NEGATIVE_LABELS)
    return {
        "items": labeled,
        "missed_points": missed,
        "counts": counts,
        "official_strike_points": len(strike_points),
        "recall": round(tp / len(strike_points), 4) if strike_points else None,
        "precision": round(tp / (tp + known_neg), 4) if points_complete and (tp + known_neg) else None
    }


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


def _system_metrics(events: List[Dict[str, Any]], points: List[Dict[str, Any]], points_complete: bool) -> Dict[str, Any]:
    """Placar previsto × oficial e precisão/recall de quem o SenpAI deu como ponto."""
    official = {k: sum(1 for p in points if p["kenshi_id"] == k) for k in KENSHI_IDS.values()}
    predicted = {k: sum(1 for e in events if e["attacker_id"] == k and e["predicted_valid"]) for k in KENSHI_IDS.values()}
    valid = [e for e in events if e["predicted_valid"]]
    tp = sum(1 for e in valid if e["weak_label"] == POSITIVE)
    fp = sum(1 for e in valid if e["weak_label"] in NEGATIVE_LABELS)
    strike_points = sum(1 for p in points if p["is_strike"])
    return {
        "placar_oficial": {"aka": official["KENSHI_AKA"], "shiro": official["KENSHI_SHIRO"]},
        "placar_senpai": {"aka": predicted["KENSHI_AKA"], "shiro": predicted["KENSHI_SHIRO"]},
        "pontos_dados_pelo_senpai": len(valid),
        "acertos": tp,
        "precisao": round(tp / (tp + fp), 4) if points_complete and (tp + fp) else None,
        "recall": round(tp / strike_points, 4) if strike_points else None
    }


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

    return {
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
        "system": _system_metrics(events["items"], points, points_complete)
    }


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
    ambiguous = [e for e in report["events"]["items"] if e["weak_label"] == AMBIGUOUS]
    if not ambiguous:
        return 0
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


def summarize_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Totais de todas as lutas rotuladas (base para a métrica honesta da I6)."""
    total = {"lutas": len(reports), "pontos_oficiais": 0, "candidatos": 0, "candidatos_positivos": 0,
             "eventos": 0, "eventos_positivos": 0, "pontos_dados": 0, "acertos": 0, "falsos_pontos": 0, "ambiguos": 0}
    for r in reports:
        total["pontos_oficiais"] += r["candidates"]["official_strike_points"]
        total["candidatos"] += len(r["candidates"]["items"])
        total["candidatos_positivos"] += r["candidates"]["counts"].get(POSITIVE, 0)
        total["eventos"] += len(r["events"]["items"])
        total["eventos_positivos"] += r["events"]["counts"].get(POSITIVE, 0)
        total["ambiguos"] += r["events"]["counts"].get(AMBIGUOUS, 0)
        valid = [e for e in r["events"]["items"] if e["predicted_valid"]]
        total["pontos_dados"] += len(valid)
        total["acertos"] += sum(1 for e in valid if e["weak_label"] == POSITIVE)
        total["falsos_pontos"] += sum(1 for e in valid if e["weak_label"] in NEGATIVE_LABELS)
    p = total["pontos_oficiais"]
    total["recall_candidatos"] = round(total["candidatos_positivos"] / p, 4) if p else None
    total["recall_senpai"] = round(total["acertos"] / p, 4) if p else None
    judged = total["acertos"] + total["falsos_pontos"]
    total["precisao_senpai"] = round(total["acertos"] / judged, 4) if judged else None
    return total


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
    args = parser.parse_args(argv)

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
                extra += f" · feedback {export_to_feedback(report)}"
            if args.enqueue_ambiguous:
                extra += f" · {enqueue_ambiguous(report)} na fila"
            s = report["system"]
            print(f"[{idx}/{len(todo)}] {time.time() - t0:.1f}s — {report['video_name']}\n"
                  f"    placar oficial {s['placar_oficial']} × SenpAI {s['placar_senpai']} · "
                  f"candidatos {report['candidates']['counts']} · eventos {report['events']['counts']}{extra}\n"
                  f"    relatório: {path}")
            reports.append(report)
        except Exception as e:
            failures += 1
            print(f"[{idx}/{len(todo)}] FALHA — {os.path.basename(video)}: {e}")

    if reports:
        print("Resumo:", json.dumps(summarize_reports(reports), ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
