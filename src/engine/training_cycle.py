"""
Ciclo de Treino Real do SenpAI (Pendência I8 — substitui o treinamento simulado por tempo).

O sistema melhora com DADOS, nunca com tempo de execução. Cada ciclo:
1. Cache de poses: extrai as poses dos vídeos ainda não processados (I1).
2. Rótulos automáticos: se o detector de bandeiras provou F1 alto contra os gabaritos manuais,
   grava os pontos lidos das bandeiras nas lutas ainda sem gabarito (I7).
2b. Pré-treino: refaz o encoder de movimento auto-supervisionado quando o cache cresce (I10).
3. Filtro de golpes: treina um modelo DESAFIANTE com os gabaritos (exceto golden) e só o promove
   se ele melhorar o F1 no Golden Benchmark; o anterior fica guardado para voltar (I3).
3b. Técnica: treina um classificador DESAFIANTE (MEN/KOTE/DO/TSUKI) e só o promove se a acurácia
   de técnica no golden subir sem piorar o F1 dos pontos (I9).
4. Medição oficial: roda o Golden Benchmark com o sistema em uso e grava no histórico (I5/I6).

Etapas sem dados suficientes são puladas com o motivo; nada é inventado.
Cada ciclo é registrado em data/training_cycles.jsonl.

Uso via linha de comando (a partir da pasta Dev/), por exemplo num agendador do sistema:
    python -m src.engine.training_cycle run "<pasta de vídeos>" [--device gpu|cpu]
    python -m src.engine.training_cycle history
"""

import os
import json
import threading
import datetime
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir

DEFAULT_HISTORY_PATH = os.path.join("data", "training_cycles.jsonl")
STEP_NAMES = {
    "cache": "Cache de poses",
    "rotulos": "Banco único de rótulos (sincronização)",
    "bandeiras": "Rótulos pelas bandeiras",
    "pretreino": "Pré-treino do encoder de movimento (sem rótulo)",
    "filtro": "Filtro de golpes (desafiante × atual)",
    "tecnica": "Classificador de técnica (desafiante × atual)",
    "modalidade": "Classificador luta × 14 modalidades (desafiante × atual)",
    "sugestao": "Sugestão de modalidade nos vídeos não classificados (dúvidas na curadoria)",
    "calibracao": "Calibração dos perfis (pesos e limiares)",
    "medicao": "Medição no Golden Benchmark",
}


_detail_cb: Optional[Callable[[str], None]] = None


def report_detail(text: str) -> None:
    """Andamento dentro de uma etapa (ex.: "vídeo 3 de 24: nome"), mostrado na tela e na linha de comando."""
    if _detail_cb:
        try:
            _detail_cb(text)
        except Exception:
            pass


def get_history_path() -> str:
    if is_test_mode():
        return os.path.join(get_test_sandbox_dir(), "training_cycles.jsonl")
    os.makedirs(os.path.dirname(DEFAULT_HISTORY_PATH), exist_ok=True)
    return DEFAULT_HISTORY_PATH


def load_cycle_history(path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = path or get_history_path()
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _append_history(record: Dict[str, Any], path: Optional[str] = None) -> None:
    with open(path or get_history_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------------
# Etapas (cada uma devolve {"status": ok|pulado|falhou, "detalhe": str, ...})
# ------------------------------------------------------------------------------

def step_cache(pipeline, videos: List[str], labels_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Poses de todos os vídeos aproveitáveis: lutas (golpes, bandeiras) e treinos/exames (modalidade).
    Só os descartados ficam de fora. Assim a etapa de modalidade não precisa extrair nada depois.
    """
    from src.utils import feature_cache
    from src.engine import weak_labels as wl
    labels = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
    ignored = [v for v in videos if wl.is_entry_discarded(labels.get(os.path.basename(v)))]
    videos = [v for v in videos if v not in ignored]
    pending = [v for v in videos
               if feature_cache.load_match_features(feature_cache.build_cache_key(v, pipeline.vision_model)) is None]
    new, failed = 0, []
    for i, video in enumerate(pending):
        report_detail(f"vídeo {i + 1} de {len(pending)} sem poses no cache: {os.path.basename(video)}")
        try:
            pipeline.extract_match_features(video)
            new += 1
        except Exception as e:
            failed.append(f"{os.path.basename(video)}: {e}")
    detail = f"{new} vídeo(s) novo(s) no cache; {len(videos) - len(pending)} já estavam."
    if ignored:
        detail += (f" {len(ignored)} vídeo(s) que você marcou como \"Não aproveitável\" no Gabarito não entram "
                   "(nenhum vídeo é apagado).")
    if failed:
        detail += f" Falhas: {len(failed)}."
    return {"status": "ok" if not failed else "falhou", "detalhe": detail, "novos": new, "ignorados": len(ignored), "falhas": failed[:5]}


def step_labels() -> Dict[str, Any]:
    """Traz para o banco único (I11) o gabarito e as correções (revisão, Excel, curadoria)."""
    from src.engine.label_bank import LabelBank
    bank = LabelBank()
    counts = bank.sync_from_sources()
    st = bank.stats()
    detail = (f"{st['total']} rótulo(s) de {st['videos']} vídeo(s): " +
              ", ".join(f"{k} {v}" for k, v in sorted(st["por_origem"].items())) if st["total"] else "Banco vazio: preencha o gabarito ou revise golpes.")
    return {"status": "ok" if st["total"] else "pulado", "detalhe": detail, "sincronizados": counts, "banco": st}


def enqueue_conflicts(conflicts: List[Dict[str, Any]], videos_dir: str) -> int:
    """Conflitos de autoridade do banco de rótulos (I11) viram dúvidas na curadoria (I12)."""
    if not conflicts:
        return 0
    from src.engine.active_learning import UncertaintySampler
    labels = {"POSITIVO": "é ponto", "NEGATIVO_CORRECAO": "não é ponto"}
    sampler = UncertaintySampler()
    added = 0
    for c in conflicts:
        reason = (f"Fontes discordam: o rótulo atual diz que {labels.get(c['atual'], c['atual'].lower())}, "
                  f"mas uma correção de {c['origem']} diz que {labels.get(c['correcao'], c['correcao'].lower())}.")
        entry = sampler.enqueue_doubt(
            doubt_type="conflito", video_source=os.path.join(videos_dir, c["video_name"]), uncertainty=0.95, impact=1.0,
            reason=reason, strike_data={"timestamp_seconds": c["time_sec"], "strike_type": "MEN"},
        )
        added += int(entry is not None)
    return added


def step_calibration(pipeline, videos_dir: str) -> Dict[str, Any]:
    """Recalibra os perfis com os golpes rotulados pelo banco (I11 → I4); a trava do golden decide se grava."""
    from src.engine.label_bank import build_training_reports, optimizer_feedback
    from src.engine.feedback_manager import FeedbackManager
    from src.engine.calibrator import CalibrationEngine
    previous = (pipeline.strike_filter, pipeline.technique_classifier)
    pipeline.strike_filter = pipeline.technique_classifier = None  # golpes como a análise base os detecta
    try:
        training = build_training_reports(pipeline, videos_dir)
        samples = optimizer_feedback(training)
    finally:
        pipeline.strike_filter, pipeline.technique_classifier = previous
    queued_conflicts = enqueue_conflicts(training.get("conflitos", []), videos_dir)
    if not samples:
        detail = "Nenhum golpe rotulado com as notas para calibrar."
        if queued_conflicts:
            detail += f" {queued_conflicts} conflito(s) entre fontes enviado(s) à curadoria."
        return {"status": "pulado", "detalhe": detail, "conflitos_na_fila": queued_conflicts}
    fm = FeedbackManager()
    engine = CalibrationEngine()
    results = {}
    for profile in ("normal", "rigido", "permissivo"):
        current = engine.profiles.get(profile)
        if not current:
            continue
        new_cfg, stats = fm.optimize_profile_config(profile, current, feedback_list=samples)
        changed = stats.get("status") == "success" and new_cfg != current
        if changed:
            engine.update_and_save_profile(profile, new_cfg)
        results[profile] = {"gravado": changed, "status": stats.get("status"), "f1": stats.get("f1_score"),
                            "mudancas": stats.get("changes", [])[:3]}
    saved = [p for p, r in results.items() if r["gravado"]]
    detail = f"{len(samples)} golpe(s) rotulado(s); perfis atualizados: {', '.join(saved) or 'nenhum (sem melhora ou bloqueado pelo golden)'}."
    if queued_conflicts:
        detail += f" {queued_conflicts} conflito(s) entre fontes enviado(s) à curadoria."
    return {"status": "ok", "detalhe": detail, "perfis": results, "conflitos_na_fila": queued_conflicts}


def step_flags(videos_dir: str) -> Dict[str, Any]:
    from src.utils import model_registry
    from src.vision import flag_detector as fd
    if not model_registry.get_active_model_path(fd.MODEL_FAMILY):
        return {"status": "pulado", "detalhe": "Nenhum detector de bandeiras ativo (rotule e treine na aba Gabarito de Lutas)."}
    evaluation = fd.evaluate_against_gabarito(videos_dir)
    res = fd.apply_to_gabarito(videos_dir, evaluation)
    t = evaluation["total"]
    if not res.get("aplicado"):
        return {"status": "pulado", "detalhe": res.get("motivo"), "f1": t.get("f1"), "videos": t.get("videos")}
    return {"status": "ok", "detalhe": f"Pontos automáticos gravados em {len(res['lutas'])} luta(s) (F1 do detector {t.get('f1')}).",
            "f1": t.get("f1"), "videos": t.get("videos")}


MIN_ENCODER_QUALITY = 0.5      # reconhecimento do par em lutas não vistas
ENCODER_REFRESH_GROWTH = 1.2   # refaz o pré-treino quando o cache cresce 20%


def step_pretrain(epochs: int = 15) -> Dict[str, Any]:
    from src.analytics import pose_encoder as pe
    from src.utils import model_registry
    X, groups = pe.collect_windows()
    active = model_registry.get_active_model_path(pe.MODEL_FAMILY)
    if active:
        try:
            with open(active.replace(".pt", ".json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            used = meta["janelas"]["treino"] + meta["janelas"]["validacao"]
            if len(X) < used * ENCODER_REFRESH_GROWTH:
                return {"status": "pulado", "detalhe": f"Cache cresceu pouco desde o último pré-treino ({used} → {len(X)} janelas)."}
        except Exception:
            pass
    try:
        meta = pe.train_encoder(X, groups, epochs=epochs)
    except ValueError as e:
        return {"status": "pulado", "detalhe": str(e)}
    quality = meta["reconhecimento_do_par"]["depois_do_treino"] or 0.0
    if quality >= MIN_ENCODER_QUALITY:
        model_registry.set_active_model(pe.MODEL_FAMILY, meta["modelo"], f"Reconhecimento do par {quality:.2f} em lutas não vistas.")
        detail = (f"Encoder ativado ({len(X)} janelas; reconhecimento do par "
                  f"{meta['reconhecimento_do_par']['antes_do_treino']} → {quality}). Os próximos treinos testam com e sem ele.")
    else:
        detail = f"Encoder não ativado: reconhecimento do par {quality} abaixo de {MIN_ENCODER_QUALITY}."
    return {"status": "ok", "detalhe": detail, "ativado": quality >= MIN_ENCODER_QUALITY, "qualidade": quality}


def step_strike_filter(pipeline, videos_dir: str) -> Dict[str, Any]:
    from src.analytics import strike_classifier as sc
    previous = pipeline.strike_filter
    pipeline.strike_filter = None  # o dataset usa sempre os candidatos brutos
    try:
        dataset = sc.build_dataset(pipeline, videos_dir)
        try:
            model = sc.train_model(dataset)
        except ValueError as e:
            return {"status": "pulado", "detalhe": str(e)}
        path = sc.save_model(model)
        promo = sc.promote_if_better(pipeline, videos_dir, path)
    finally:
        pipeline.strike_filter = previous
    if promo["promovido"]:
        pipeline.strike_filter = sc.StrikeFilter.load_active()
    return {"status": "ok", "detalhe": ("Promovido: " if promo["promovido"] else "Mantido o atual: ") + promo["motivo"],
            "modelo": os.path.basename(path), "promovido": promo["promovido"],
            "validacao_por_video": model["validacao_por_video"]["com_filtro"]}


def step_technique(pipeline, videos_dir: str) -> Dict[str, Any]:
    from src.analytics import technique_classifier as tcl
    previous = pipeline.technique_classifier
    pipeline.technique_classifier = None  # o dataset usa sempre a detecção sem o modelo
    try:
        dataset = tcl.build_dataset(pipeline, videos_dir)
        try:
            model = tcl.train_model(dataset)
        except ValueError as e:
            return {"status": "pulado", "detalhe": str(e)}
        path = tcl.save_model(model)
    finally:
        pipeline.technique_classifier = previous
    promo = tcl.promote_if_better(pipeline, videos_dir, path)
    if promo["promovido"]:
        pipeline.technique_classifier = tcl.TechniqueClassifier.load_active()
    return {"status": "ok", "detalhe": ("Promovido: " if promo["promovido"] else "Mantido o atual: ") + promo["motivo"],
            "modelo": os.path.basename(path), "promovido": promo["promovido"],
            "validacao_por_video": model["validacao_por_video"]}


def step_modality(pipeline, videos_dir: str) -> Dict[str, Any]:
    from src.analytics import modality_classifier as mcl
    try:
        model = mcl.train_model(mcl.build_dataset(pipeline, videos_dir))
    except ValueError as e:
        return {"status": "pulado", "detalhe": str(e)}
    path = mcl.save_model(model)
    promo = mcl.promote_if_better(path)
    if promo["promovido"]:
        pipeline.modality_classifier = mcl.ModalityClassifier.load_active()
    return {"status": "ok", "detalhe": ("Promovido: " if promo["promovido"] else "Mantido o atual: ") + promo["motivo"],
            "modelo": os.path.basename(path), "promovido": promo["promovido"],
            "validacao_por_video": model["validacao_por_video"]}


def step_suggest_modality(pipeline, videos: List[str]) -> Dict[str, Any]:
    """O modelo de modalidade ativo sugere o tipo dos vídeos não classificados; incertos viram dúvidas."""
    from src.analytics import modality_classifier as mcl
    return mcl.suggest_modalities(pipeline, videos, classifier=getattr(pipeline, "modality_classifier", None))


def step_measure(pipeline, videos_dir: str) -> Dict[str, Any]:
    from src.engine.active_learning import GoldenBenchmark
    gb = GoldenBenchmark()
    if not gb.load_matches():
        return {"status": "pulado", "detalhe": "Nenhuma luta congelada no golden (Perfis de Calibração > Eixo 4)."}
    res = gb.run_matches(pipeline, videos_dir)
    t = res["total"]
    return {"status": "ok" if res["lutas"] else "pulado",
            "detalhe": f"{t['lutas']} luta(s): acurácia {t.get('acuracia')}, precisão {t.get('precisao')}, recall {t.get('recall')}, F1 {t.get('f1')}.",
            "total": t}


# ------------------------------------------------------------------------------
# Checagem antes de rodar: o que você ainda não classificou
# ------------------------------------------------------------------------------

def readiness(videos_dir: str, labels_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Pendências que o ciclo não consegue aproveitar ainda. Não impede o ciclo: só avisa, para você decidir
    se classifica antes (o treino usa só o que já está rotulado).
    """
    from src.engine import weak_labels as wl
    from src.utils.feature_cache import list_videos
    from src.utils.video_library import is_analysis_clip
    from src.analytics.modality_classifier import is_classified
    videos = [v for v in list_videos(videos_dir) if not is_analysis_clip(v)] if os.path.isdir(videos_dir) else []
    gab = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
    entries = {os.path.basename(v): gab.get(os.path.basename(v)) for v in videos}
    items = []

    def add(key, n, text, where):
        items.append({"chave": key, "quantidade": n, "texto": text, "onde": where})

    add("videos", sum(1 for e in entries.values() if not is_classified(e)),
        "vídeo(s) sem tipo (luta, treino, exame ou descartado)", "Gabarito de Lutas > Tipo do vídeo")
    add("lutas", sum(1 for e in entries.values() if wl.entry_video_type(e) == wl.VIDEO_TYPE_LUTA and (e or {}).get("tipo")
                     and not wl.is_entry_filled(e)),
        "luta(s) sem os pontos preenchidos", "Gabarito de Lutas > pontos da luta")
    try:
        from src.vision import flag_detector as fd
        add("bandeiras", fd.dataset_stats().get("sem_rotulo", 0), "recorte(s) de bandeira sem rótulo",
            "Gabarito de Lutas > Detector de bandeiras")
    except Exception:
        pass
    try:
        from src.engine.active_learning import UncertaintySampler
        add("curadoria", len([q for q in UncertaintySampler().get_queue() if q.get("status") == "pending_curation"]),
            "dúvida(s) abertas na Fila de Curadoria", "Fila de Curadoria")
    except Exception:
        pass
    golden = 0
    try:
        from src.engine.active_learning import GoldenBenchmark
        golden = len(GoldenBenchmark().load_matches())
    except Exception:
        pass
    pending = [it for it in items if it["quantidade"]]
    return {"pendencias": pending, "pronto": not pending, "videos": len(videos), "golden": golden}


# ------------------------------------------------------------------------------
# Ciclo
# ------------------------------------------------------------------------------

def run_training_cycle(
    videos_dir: str,
    pipeline=None,
    device_preference: str = "gpu",
    progress: Optional[Callable[[str, str], None]] = None,
    history_path: Optional[str] = None,
    detail: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Executa as etapas em ordem; uma etapa que falha não impede as seguintes. O ciclo só USA o que já foi
    coletado: a coleta na internet roda apenas pela procura ativa (Base de Conhecimento), quando ligada.
    """
    global _detail_cb
    _detail_cb = detail
    from src.utils.feature_cache import list_videos
    if pipeline is None:
        from src.pipeline import SenpAIPipeline
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=device_preference)
    record: Dict[str, Any] = {
        "id": datetime.datetime.now().strftime("ciclo_%Y%m%dT%H%M%S"),
        "iniciado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "pasta": videos_dir,
        "etapas": {},
    }
    from src.utils.video_library import is_analysis_clip
    # Cortes de 10 s das análises servem às bandeiras e à curadoria; não entram no cache de luta nem nas sugestões
    videos = [v for v in list_videos(videos_dir) if not is_analysis_clip(v)] if os.path.isdir(videos_dir) else []
    steps = [
        ("cache", lambda: step_cache(pipeline, videos)),
        ("bandeiras", lambda: step_flags(videos_dir)),
        ("rotulos", lambda: step_labels()),
        ("pretreino", lambda: step_pretrain()),
        ("filtro", lambda: step_strike_filter(pipeline, videos_dir)),
        ("tecnica", lambda: step_technique(pipeline, videos_dir)),
        ("modalidade", lambda: step_modality(pipeline, videos_dir)),
        ("sugestao", lambda: step_suggest_modality(pipeline, videos)),
        ("calibracao", lambda: step_calibration(pipeline, videos_dir)),
        ("medicao", lambda: step_measure(pipeline, videos_dir)),
    ]
    for n_step, (key, fn) in enumerate(steps, start=1):
        report_detail("")
        if progress:
            progress(key, f"{STEP_NAMES[key]} (etapa {n_step} de {len(steps)})")
        try:
            result = fn()
        except Exception as e:
            result = {"status": "falhou", "detalhe": f"{type(e).__name__}: {e}"}
            log_event("WARNING", f"Ciclo de treino — etapa '{key}' falhou: {e}", "training_cycle")
        record["etapas"][key] = result
    record["terminado_em"] = datetime.datetime.now().isoformat(timespec="seconds")
    record["promoveu_modelo"] = any(record["etapas"].get(k, {}).get("promovido") for k in ("filtro", "tecnica", "modalidade"))
    _append_history(record, history_path)
    log_event("INFO", f"Ciclo de treino {record['id']} concluído: " +
              ", ".join(f"{k}={v['status']}" for k, v in record["etapas"].items()), "training_cycle")
    return record


# ------------------------------------------------------------------------------
# Execução em segundo plano (para a interface)
# ------------------------------------------------------------------------------

_state_lock = threading.Lock()
_state: Dict[str, Any] = {"rodando": False, "etapa": None, "detalhe": "", "inicio": None, "resultado": None, "erro": None}


def get_cycle_state() -> Dict[str, Any]:
    with _state_lock:
        return dict(_state)


def start_cycle_thread(videos_dir: str, device_preference: str = "gpu") -> bool:
    """Inicia um ciclo em segundo plano; devolve False se já houver um rodando."""
    with _state_lock:
        if _state["rodando"]:
            return False
        _state.update({"rodando": True, "etapa": None, "detalhe": "", "resultado": None, "erro": None,
                       "inicio": datetime.datetime.now().isoformat(timespec="seconds")})

    def _progress(key: str, name: str) -> None:
        with _state_lock:
            _state["etapa"] = name
            _state["etapa_inicio"] = datetime.datetime.now().isoformat(timespec="seconds")

    def _detail(text: str) -> None:
        with _state_lock:
            _state["detalhe"] = text

    def _run() -> None:
        try:
            result = run_training_cycle(videos_dir, device_preference=device_preference, progress=_progress, detail=_detail)
            with _state_lock:
                _state["resultado"] = result
        except Exception as e:
            with _state_lock:
                _state["erro"] = f"{type(e).__name__}: {e}"
        finally:
            with _state_lock:
                _state["rodando"] = False
                _state["etapa"] = None

    threading.Thread(target=_run, name="senpai-training-cycle", daemon=True).start()
    return True


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Ciclo de treino real do SenpAI.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Executa um ciclo completo.")
    p_run.add_argument("videos_dir")
    p_run.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    sub.add_parser("history", help="Lista os ciclos executados.")
    args = parser.parse_args(argv)

    if args.command == "history":
        for r in load_cycle_history():
            print(f"{r['iniciado_em']} · " + " · ".join(f"{STEP_NAMES[k]}: {v['status']}" for k, v in r["etapas"].items()))
        return 0
    rec = run_training_cycle(args.videos_dir, device_preference=args.device,
                             progress=lambda k, name: print(f"→ {name}...", flush=True),
                             detail=lambda t: t and print(f"   {t}", flush=True))
    for k, v in rec["etapas"].items():
        print(f"  [{v['status']}] {STEP_NAMES[k]}: {v['detalhe']}")
    return 0 if all(v["status"] != "falhou" for v in rec["etapas"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
