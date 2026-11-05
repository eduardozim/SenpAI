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
    "internet": "Coleta na internet (regras e vídeos)",
    "rotulos": "Banco único de rótulos (sincronização)",
    "bandeiras": "Rótulos pelas bandeiras",
    "pretreino": "Pré-treino do encoder de movimento (sem rótulo)",
    "filtro": "Filtro de golpes (desafiante × atual)",
    "tecnica": "Classificador de técnica (desafiante × atual)",
    "modalidade": "Classificador luta × 14 modalidades (desafiante × atual)",
    "calibracao": "Calibração dos perfis (pesos e limiares)",
    "medicao": "Medição no Golden Benchmark",
}


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
    from src.utils import feature_cache
    from src.engine import weak_labels as wl
    labels = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
    ignored = [v for v in videos if wl.is_entry_ignored(labels.get(os.path.basename(v)))]
    videos = [v for v in videos if v not in ignored]
    new, failed = 0, []
    for video in videos:
        key = feature_cache.build_cache_key(video, pipeline.vision_model)
        if feature_cache.load_match_features(key) is not None:
            continue
        try:
            pipeline.extract_match_features(video)
            new += 1
        except Exception as e:
            failed.append(f"{os.path.basename(video)}: {e}")
    detail = f"{new} vídeo(s) novo(s) no cache; {len(videos) - new - len(failed)} já estavam."
    if ignored:
        detail += f" {len(ignored)} ignorado(s) (não é luta)."
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


def step_internet() -> Dict[str, Any]:
    """Coleta na internet (I15): regulamentos (em cache) e busca/download de vídeos conforme a configuração."""
    from src.engine import internet_collector as ic
    cfg = ic.load_config()
    rules = ic.collect_rules(cfg)
    ok_src = sum(1 for r in rules["fontes"] if r["status"] == "ok")
    vids = ic.collect_videos(cfg)
    detail = (f"Regulamentos: {ok_src}/{len(rules['fontes'])} lidos. Vídeos: {vids['novos_no_catalogo']} novo(s) no catálogo, "
              f"{len(vids['baixados'])} baixado(s), {vids['rotulos']} rótulo(s) e {vids['duvidas']} dúvida(s). "
              + ("" if vids["download_ligado"] else "Download desligado (Base de Conhecimento > Coleta na Internet)."))
    return {"status": "ok", "detalhe": detail, "baixados": vids["baixados"], "rotulos": vids["rotulos"]}


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
# Ciclo
# ------------------------------------------------------------------------------

def run_training_cycle(
    videos_dir: str,
    pipeline=None,
    device_preference: str = "gpu",
    progress: Optional[Callable[[str, str], None]] = None,
    history_path: Optional[str] = None
) -> Dict[str, Any]:
    """Executa as etapas em ordem; uma etapa que falha não impede as seguintes."""
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
    videos = list_videos(videos_dir) if os.path.isdir(videos_dir) else []
    steps = [
        ("cache", lambda: step_cache(pipeline, videos)),
        ("bandeiras", lambda: step_flags(videos_dir)),
        ("internet", lambda: step_internet()),
        ("rotulos", lambda: step_labels()),
        ("pretreino", lambda: step_pretrain()),
        ("filtro", lambda: step_strike_filter(pipeline, videos_dir)),
        ("tecnica", lambda: step_technique(pipeline, videos_dir)),
        ("modalidade", lambda: step_modality(pipeline, videos_dir)),
        ("calibracao", lambda: step_calibration(pipeline, videos_dir)),
        ("medicao", lambda: step_measure(pipeline, videos_dir)),
    ]
    for key, fn in steps:
        if progress:
            progress(key, STEP_NAMES[key])
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
_state: Dict[str, Any] = {"rodando": False, "etapa": None, "resultado": None, "erro": None}


def get_cycle_state() -> Dict[str, Any]:
    with _state_lock:
        return dict(_state)


def start_cycle_thread(videos_dir: str, device_preference: str = "gpu") -> bool:
    """Inicia um ciclo em segundo plano; devolve False se já houver um rodando."""
    with _state_lock:
        if _state["rodando"]:
            return False
        _state.update({"rodando": True, "etapa": None, "resultado": None, "erro": None})

    def _progress(key: str, name: str) -> None:
        with _state_lock:
            _state["etapa"] = name

    def _run() -> None:
        try:
            result = run_training_cycle(videos_dir, device_preference=device_preference, progress=_progress)
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
                             progress=lambda k, name: print(f"→ {name}..."))
    for k, v in rec["etapas"].items():
        print(f"  [{v['status']}] {STEP_NAMES[k]}: {v['detalhe']}")
    return 0 if all(v["status"] != "falhou" for v in rec["etapas"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
