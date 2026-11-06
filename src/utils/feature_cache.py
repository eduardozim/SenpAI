"""
Cache Persistente de Features de Vídeo (Pendência I1 — Treinar por dados, não por tempo).
Guarda em disco o resultado da etapa mais cara do pipeline (inferência de pose + associação
Aka/Shiro), para que cada vídeo seja processado pela rede neural uma única vez.
As etapas seguintes (Sonkyō, detecção de golpes, avaliação e calibração) podem então ser
reexecutadas em segundos sobre o cache, permitindo iterar limiares e detectores rapidamente.

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.utils.feature_cache build "<pasta ou arquivo de vídeo>" [--frame-step N]
    python -m src.utils.feature_cache stats
    python -m src.utils.feature_cache clear
"""

import os
import gzip
import json
import time
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Tuple

from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir

# Incrementar sempre que a saída do PoseDetector ou do CombatantTracker mudar,
# invalidando automaticamente todos os caches gerados pela versão anterior.
FEATURE_CACHE_SCHEMA_VERSION = 1

DEFAULT_FEATURE_CACHE_DIR = os.path.join("data", "feature_cache")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# Bytes lidos no início e no fim do arquivo para a impressão digital do conteúdo
_FINGERPRINT_CHUNK_BYTES = 1024 * 1024


def get_feature_cache_dir() -> str:
    """Retorna a pasta do cache; em modo de teste usa a sandbox isolada."""
    if is_test_mode():
        cache_dir = os.path.join(get_test_sandbox_dir(), "feature_cache")
    else:
        cache_dir = DEFAULT_FEATURE_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def is_feature_cache_enabled_by_default() -> bool:
    """Cache ligado por padrão, exceto em testes automatizados (que precisam exercitar a extração)."""
    return not is_test_mode()


def compute_video_fingerprint(video_path: str) -> str:
    """
    Impressão digital do conteúdo do vídeo (tamanho + SHA-1 do primeiro e do último MB).
    Independe do nome e da pasta do arquivo, então renomear ou mover o vídeo mantém o cache.
    """
    size = os.path.getsize(video_path)
    sha = hashlib.sha1()
    sha.update(str(size).encode("utf-8"))
    with open(video_path, "rb") as f:
        sha.update(f.read(_FINGERPRINT_CHUNK_BYTES))
        if size > 2 * _FINGERPRINT_CHUNK_BYTES:
            f.seek(-_FINGERPRINT_CHUNK_BYTES, os.SEEK_END)
            sha.update(f.read(_FINGERPRINT_CHUNK_BYTES))
    return sha.hexdigest()


def build_cache_key(
    video_path: str,
    vision_model: str,
    frame_step: int = 1,
    invert_combatants: bool = False,
    shiaijo_polygon: Optional[List[Tuple[float, float]]] = None,
    mode: str = "match"
) -> str:
    """
    Chave única combinando o conteúdo do vídeo e todos os parâmetros que alteram a extração.
    mode="training" separa o cache do Modo Treinamento (I13); o modo luta mantém as chaves de sempre.
    """
    params = {
        "schema": FEATURE_CACHE_SCHEMA_VERSION,
        "video": compute_video_fingerprint(video_path),
        "vision_model": str(vision_model or "").lower().strip(),
        "frame_step": max(1, int(frame_step or 1)),
        "invert_combatants": bool(invert_combatants),
        "shiaijo_polygon": [[round(float(x), 4), round(float(y), 4)] for x, y in (shiaijo_polygon or [])]
    }
    if mode and mode != "match":
        params["mode"] = mode
    raw = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:20]


def _cache_file_path(cache_key: str) -> str:
    return os.path.join(get_feature_cache_dir(), f"{cache_key}.pkl.gz")


TRAINING_REQUIRED_KEYS = ("tracks", "summary", "total_reidentifications", "fps", "total_frames")


def load_training_features(cache_key: str) -> Optional[Dict[str, Any]]:
    """Carrega o rastreamento do Modo Treinamento (I13); None se ausente, corrompido ou de outro schema."""
    return load_match_features(cache_key, required=TRAINING_REQUIRED_KEYS)


def load_match_features(cache_key: str, required: Optional[Tuple[str, ...]] = None) -> Optional[Dict[str, Any]]:
    """Carrega as features de um vídeo de luta; retorna None se ausente, corrompido ou de outro schema."""
    path = _cache_file_path(cache_key)
    if not os.path.isfile(path):
        return None
    try:
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("schema") != FEATURE_CACHE_SCHEMA_VERSION:
            return None
        required = required or ("aka_history", "shiro_history", "discarded_per_frame", "fps", "total_frames", "tracker_summary")
        if not all(k in data for k in required):
            return None
        return data
    except Exception as e:
        log_event("WARNING", f"Cache de features ilegível ({os.path.basename(path)}): {e}. Será regenerado.", "feature_cache")
        return None


def save_match_features(cache_key: str, payload: Dict[str, Any]) -> Optional[str]:
    """Grava as features de forma atômica (arquivo temporário + rename). Falhas nunca interrompem a análise."""
    path = _cache_file_path(cache_key)
    tmp_path = f"{path}.tmp"
    try:
        data = dict(payload)
        data["schema"] = FEATURE_CACHE_SCHEMA_VERSION
        data["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with gzip.open(tmp_path, "wb", compresslevel=5) as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        from src.utils.file_utils import safe_replace
        safe_replace(tmp_path, path)
        return path
    except Exception as e:
        log_event("WARNING", f"Falha ao gravar cache de features: {e}", "feature_cache")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None


def get_feature_cache_stats() -> Dict[str, Any]:
    """Resumo do cache: quantidade de vídeos, tamanho total e vídeos de origem."""
    cache_dir = get_feature_cache_dir()
    entries = []
    total_bytes = 0
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".pkl.gz"):
            continue
        path = os.path.join(cache_dir, name)
        size = os.path.getsize(path)
        total_bytes += size
        entries.append({"cache_key": name[:-len(".pkl.gz")], "size_bytes": size})
    return {"cache_dir": os.path.abspath(cache_dir), "entries": len(entries), "total_bytes": total_bytes, "items": entries}


def clear_feature_cache() -> int:
    """Remove todos os arquivos do cache e retorna quantos foram apagados."""
    cache_dir = get_feature_cache_dir()
    removed = 0
    for name in os.listdir(cache_dir):
        if name.endswith(".pkl.gz") or name.endswith(".tmp"):
            os.remove(os.path.join(cache_dir, name))
            removed += 1
    return removed


def list_videos(target: str) -> List[str]:
    """Lista os vídeos de um arquivo ou pasta (sem recursão), ignorando downloads parciais."""
    if os.path.isfile(target):
        return [target]
    return sorted(
        os.path.join(target, n) for n in os.listdir(target)
        if n.lower().endswith(VIDEO_EXTENSIONS) and os.path.isfile(os.path.join(target, n))
    )


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cache de features de vídeo do SenpAI (pose + rastreamento).")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="Extrai e grava as features dos vídeos (pula os já processados).")
    p_build.add_argument("target", help="Arquivo de vídeo ou pasta com vídeos.")
    p_build.add_argument("--frame-step", type=int, default=1)
    p_build.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    sub.add_parser("stats", help="Mostra o tamanho e a quantidade de vídeos em cache.")
    sub.add_parser("clear", help="Apaga todo o cache de features.")
    args = parser.parse_args(argv)

    if args.command == "stats":
        st = get_feature_cache_stats()
        print(f"Pasta: {st['cache_dir']}")
        print(f"Vídeos em cache: {st['entries']} ({st['total_bytes'] / (1024 * 1024):.1f} MB)")
        return 0

    if args.command == "clear":
        print(f"Arquivos removidos: {clear_feature_cache()}")
        return 0

    from src.pipeline import SenpAIPipeline

    videos = list_videos(args.target)
    if not videos:
        print("Nenhum vídeo encontrado.")
        return 1

    pipeline = SenpAIPipeline(calibration_profile="normal", device_preference=args.device)
    failures = 0
    for idx, video in enumerate(videos, start=1):
        t0 = time.time()
        try:
            res = pipeline.extract_match_features(video, frame_step=args.frame_step)
            status = "cache existente" if res["from_cache"] else "extraído"
            print(f"[{idx}/{len(videos)}] {status} em {time.time() - t0:.1f}s — {os.path.basename(video)}")
        except Exception as e:
            failures += 1
            print(f"[{idx}/{len(videos)}] FALHA — {os.path.basename(video)}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
