"""
Pré-processamento dos vídeos de luta baixados da internet (Pendência I15 → I12 → I8).

Baixado, um vídeo da internet só tem o rótulo de modalidade pelo título: nenhum golpe dele ensina nada
enquanto ninguém o revisa. Este passo analisa cada luta baixada (com o cache de poses) e manda à Fila de
Curadoria os golpes em que o sistema está em dúvida: nota na faixa de incerteza, filtro de golpes e técnica
indecisos, possível ai-uchi ou golpe após o yame, golpe fora do padrão dos validados. A resposta de quem
revisa (com o dan) vira rótulo no banco único e o próximo ciclo de treino aprende com ela.

- Só lutas (tipo do Gabarito; sem tipo, a modalidade pelo título), nunca descartados nem lutas do golden.
- Cada vídeo é analisado uma vez (registro em data/internet/preprocessados.json).
- A fila de curadoria comporta MAX_QUEUE_ITEMS dúvidas; com ela quase cheia o passo para e o resto espera
  o próximo ciclo (dúvidas que não cabem seriam descartadas sem ninguém ver).
- Não grava o perfil biomecânico dos kenshi (são atletas desconhecidos): só as dúvidas.

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.engine.internet_preprocess run "<pasta de vídeos>" [--device gpu|cpu]
    python -m src.engine.internet_preprocess status "<pasta de vídeos>"
"""

import os
import datetime
import threading
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger_manager import log_event
from src.utils.file_utils import safe_load_json, safe_write_json

REGISTRY_FILE = "preprocessados.json"
LOCK_FILE = "preprocessando.lock"   # um pré-processamento por vez, mesmo entre processos (tela e linha de comando)
INTERNET_PREFIX = "internet_"
QUEUE_ROOM_RESERVE = 20      # para quando sobrarem menos que isto de espaço na fila de curadoria


def _registry_path() -> str:
    from src.engine.internet_collector import get_root_dir
    return os.path.join(get_root_dir(), REGISTRY_FILE)


def load_registry() -> Dict[str, Dict[str, Any]]:
    data = safe_load_json(_registry_path(), default={})
    return data.get("videos", {}) if isinstance(data, dict) else {}


def _save_registry(videos: Dict[str, Dict[str, Any]]) -> None:
    path = _registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safe_write_json(path, {"versao": 1, "videos": videos}, indent=1)


def _lock_path() -> str:
    from src.engine.internet_collector import get_root_dir
    return os.path.join(get_root_dir(), LOCK_FILE)


def running_elsewhere() -> Optional[int]:
    """PID de outro processo que está pré-processando agora (None se ninguém)."""
    import json
    try:
        with open(_lock_path(), "r", encoding="utf-8") as f:
            pid = int(json.load(f).get("pid"))
    except Exception:
        return None
    if pid == os.getpid():
        return None
    try:
        import psutil
        return pid if psutil.pid_exists(pid) else None
    except Exception:
        return pid


def _acquire_lock() -> bool:
    import json
    if running_elsewhere():
        return False
    path = _lock_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "desde": datetime.datetime.now().isoformat(timespec="seconds")}, f)
    return True


def _release_lock() -> None:
    try:
        if not running_elsewhere():
            os.remove(_lock_path())
    except OSError:
        pass


def _is_fight(name: str, gabarito: Dict[str, Any], by_file: Dict[str, Dict[str, Any]]) -> bool:
    from src.engine import weak_labels as wl
    entry = gabarito.get(name)
    if entry and entry.get("tipo"):
        return wl.entry_video_type(entry) == wl.VIDEO_TYPE_LUTA and not wl.is_entry_ignored(entry)
    if entry and wl.is_entry_ignored(entry):
        return False
    return (by_file.get(name) or {}).get("modalidade_titulo") == "luta"


def internet_fights(videos_dir: str, labels_path: Optional[str] = None) -> List[str]:
    """Lutas baixadas da internet que podem ser pré-processadas (caminhos), fora o golden."""
    from src.engine import weak_labels as wl
    from src.engine import internet_collector as ic
    from src.engine.active_learning import is_golden_video
    if not os.path.isdir(videos_dir):
        return []
    gabarito = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
    by_file = {it.get("arquivo"): it for it in ic.load_catalog().values() if it.get("arquivo")}
    out = []
    for name in sorted(os.listdir(videos_dir)):
        if not (name.startswith(INTERNET_PREFIX) and name.lower().endswith(".mp4")):
            continue
        if _is_fight(name, gabarito, by_file) and not is_golden_video(name):
            out.append(os.path.join(videos_dir, name))
    return out


def _pending_doubts() -> List[Dict[str, Any]]:
    from src.engine.active_learning import UncertaintySampler
    return [q for q in UncertaintySampler().get_queue() if q.get("status") == "pending_curation"]


def _doubts_of(name: str, queue: List[Dict[str, Any]]) -> int:
    return sum(1 for q in queue if os.path.basename(str(q.get("video_source") or "").replace("\\", "/")) == name)


def status(videos_dir: str, labels_path: Optional[str] = None) -> Dict[str, Any]:
    """Quantas lutas da internet existem, quantas já foram analisadas e quantas dúvidas delas esperam resposta."""
    fights = [os.path.basename(v) for v in internet_fights(videos_dir, labels_path)]
    reg = load_registry()
    done = [n for n in fights if n in reg]
    queue = _pending_doubts()
    return {
        "rodando_em_outro_processo": running_elsewhere(),
        "lutas": len(fights),
        "preprocessadas": len(done),
        "pendentes": len(fights) - len(done),
        "golpes_analisados": sum(int(reg[n].get("golpes", 0)) for n in done),
        "duvidas_criadas": sum(int(reg[n].get("duvidas", 0)) for n in done),
        "duvidas_abertas": sum(_doubts_of(n, queue) for n in fights),
    }


def preprocess_internet_videos(pipeline, videos_dir: str, labels_path: Optional[str] = None,
                               progress: Optional[Callable[[str], None]] = None,
                               should_stop: Optional[Callable[[], bool]] = None,
                               limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Analisa as lutas da internet ainda não pré-processadas e cria as dúvidas na curadoria.
    Devolve {"processados", "duvidas", "falhas", "restantes", "motivo_parada"}.
    """
    if not _acquire_lock():
        reason = f"já há um pré-processamento rodando (processo {running_elsewhere()})"
        return {"processados": [], "duvidas": 0, "falhas": [], "restantes": None, "motivo_parada": reason}
    try:
        return _preprocess(pipeline, videos_dir, labels_path, progress, should_stop, limit)
    finally:
        _release_lock()


def _preprocess(pipeline, videos_dir, labels_path, progress, should_stop, limit) -> Dict[str, Any]:
    from src.engine.active_learning import MAX_QUEUE_ITEMS
    stop = should_stop or (lambda: False)
    reg = load_registry()
    todo = [v for v in internet_fights(videos_dir, labels_path) if os.path.basename(v) not in reg]
    if limit is not None:
        todo = todo[:limit]
    done, doubts, failures, reason = [], 0, [], None
    for i, video in enumerate(todo, start=1):
        name = os.path.basename(video)
        if stop():
            reason = "interrompido"
            break
        queue = _pending_doubts()
        if len(queue) >= MAX_QUEUE_ITEMS - QUEUE_ROOM_RESERVE:
            reason = (f"fila de curadoria quase cheia ({len(queue)} dúvidas abertas): responda as dúvidas e o "
                      "restante é pré-processado no próximo ciclo")
            break
        if progress:
            progress(f"Analisando luta da internet {i} de {len(todo)}: {name}")
        before = _doubts_of(name, queue)
        try:
            result = pipeline.process_video(video, use_feature_cache=True, record_learning_side_effects=False,
                                            enqueue_doubts=True)
        except Exception as e:
            failures.append({"video_name": name, "erro": f"{type(e).__name__}: {e}"})
            log_event("WARNING", f"Pré-processamento falhou em {name}: {e}", "internet_preprocess")
            continue
        if result is None:
            reason = "análise interrompida"
            break
        added = max(0, _doubts_of(name, _pending_doubts()) - before)
        reg[name] = {"em": datetime.datetime.now().isoformat(timespec="seconds"),
                     "golpes": len(result.get("events", [])), "duvidas": added}
        _save_registry(reg)
        done.append(name)
        doubts += added
    remaining = len([v for v in internet_fights(videos_dir, labels_path) if os.path.basename(v) not in reg])
    log_event("INFO", f"Pré-processamento da internet: {len(done)} luta(s), {doubts} dúvida(s), {remaining} restante(s).",
              "internet_preprocess")
    return {"processados": done, "duvidas": doubts, "falhas": failures, "restantes": remaining, "motivo_parada": reason}


def summary_text(res: Dict[str, Any]) -> str:
    text = f"{len(res['processados'])} luta(s) da internet analisada(s), {res['duvidas']} dúvida(s) enviada(s) à curadoria"
    text += "." if res.get("restantes") is None else f"; {res['restantes']} ainda por analisar."
    if res.get("motivo_parada"):
        text += f" Parou: {res['motivo_parada']}."
    if res.get("falhas"):
        text += f" {len(res['falhas'])} falha(s): " + "; ".join(f"{f['video_name']} ({f['erro']})" for f in res["falhas"][:3])
    return text


# ------------------------------------------------------------------------------
# Execução em segundo plano (botão na Governança de Treinamento)
# ------------------------------------------------------------------------------

_lock = threading.Lock()
_state: Dict[str, Any] = {"rodando": False, "detalhe": "", "inicio": None, "resultado": None, "erro": None}
_stop = threading.Event()


def get_state() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def request_stop() -> None:
    _stop.set()


def start_thread(videos_dir: str, device_preference: str = "gpu") -> bool:
    """Pré-processa em segundo plano; False se já estiver rodando (ou se um ciclo de treino estiver rodando)."""
    from src.engine.training_cycle import get_cycle_state
    with _lock:
        if _state["rodando"] or get_cycle_state().get("rodando"):
            return False
        _state.update({"rodando": True, "detalhe": "", "resultado": None, "erro": None,
                       "inicio": datetime.datetime.now().isoformat(timespec="seconds")})
    _stop.clear()

    def _detail(text: str) -> None:
        with _lock:
            _state["detalhe"] = text

    def _run() -> None:
        try:
            from src.pipeline import SenpAIPipeline
            pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=device_preference)
            res = preprocess_internet_videos(pipeline, videos_dir, progress=_detail, should_stop=_stop.is_set)
            with _lock:
                _state["resultado"] = res
        except Exception as e:
            with _lock:
                _state["erro"] = f"{type(e).__name__}: {e}"
        finally:
            with _lock:
                _state["rodando"] = False

    threading.Thread(target=_run, name="senpai-internet-preprocess", daemon=True).start()
    return True


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Pré-processa as lutas baixadas da internet (dúvidas na curadoria).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("videos_dir")
    run.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    run.add_argument("--limite", type=int, default=None)
    st = sub.add_parser("status")
    st.add_argument("videos_dir")
    args = parser.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(status(args.videos_dir), indent=2, ensure_ascii=False))
        return 0
    from src.pipeline import SenpAIPipeline
    pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=args.device)
    res = preprocess_internet_videos(pipeline, args.videos_dir, progress=print, limit=args.limite)
    print(summary_text(res))
    return 1 if res["falhas"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
