"""
Controle do Dataset de Treinamento: apagar, exportar e importar (restaurar).

Fonte única do que é "treinamento" no SenpAI (TRAINING_STORES). Os três botões usam esta lista,
então o que é apagado é exatamente o que é exportado e o que a importação restaura.

- apagar (reset_training): remove todos os itens da lista e reescreve os padrões de fábrica
  (perfis de calibração e base de conhecimento). Antes, grava uma cópia de segurança em data/backups/.
- exportar (export_training_zip): .zip com manifest.json (lista de arquivos, tamanhos e SHA-256) e os
  arquivos nos mesmos caminhos. Fica de fora só o que é cache/derivado ou grande demais e recuperável:
  cache de poses, vídeos baixados da internet e pastas runs/ do YOLO.
- importar (import_training_zip): RESTAURA o estado do pacote (não mistura). Valida o manifest e cada
  arquivo, grava uma cópia de segurança do estado atual, apaga e extrai. Pacotes antigos (sem manifest)
  seguem pelo importador legado, que mescla.

O que NÃO é treinamento e não é tocado: vídeos do usuário, cache de poses (data/feature_cache),
configurações do app (config/settings.json, config/internet_sources.json), uploads e logs.
"""

import io
import os
import json
import shutil
import hashlib
import zipfile
import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir

PACKAGE_FORMAT = "senpai-dataset-treinamento"
PACKAGE_VERSION = 1
MANIFEST_NAME = "manifest.json"
BACKUP_DIR = os.path.join("data", "backups")
MAX_BACKUPS = 5

# (caminho relativo à pasta Dev, descrição). Pastas entram inteiras, menos EXPORT_EXCLUDES.
TRAINING_STORES: List[Tuple[str, str]] = [
    ("data/feedback_dataset.json", "Revisões de golpes (Dan / Shinpan)"),
    ("data/training_history.json", "Histórico de treinamento"),
    ("data/shinpan_reviewed_videos.json", "Vídeos revisados por Shinpan"),
    ("data/reviewer_trust_registry.json", "Confiança dos revisores"),
    ("data/label_bank.json", "Banco Único de Rótulos"),
    ("data/modality_suggestions.json", "Sugestões de modalidade do modelo"),
    ("data/match_labels", "Gabarito de lutas"),
    ("data/benchmark_golden", "Golden Benchmark (lutas congeladas e amostras)"),
    ("data/active_learning_queue.json", "Fila de Curadoria"),
    ("data/active_learning_curated_history.json", "Histórico da curadoria"),
    ("data/active_learning_clips", "Clipes da curadoria"),
    ("data/kenshi_baselines.json", "Perfis biomecânicos dos kenshi"),
    ("data/empirical_reference_distributions.json", "Distribuições de referência (formato antigo)"),
    ("data/strike_reference.json", "Referência dos golpes validados"),
    ("data/training_cycles.jsonl", "Histórico dos ciclos de treino"),
    ("data/auto_training_checkpoint.json", "Checkpoint do treinador antigo"),
    ("data/models", "Modelos treinados (filtro, técnica, modalidade, encoder)"),
    ("data/internet", "Coleta na internet (regras, catálogo, sessões, vídeos)"),
    ("config/calibration_profiles.json", "Perfis de calibração"),
    ("config/ai_knowledge_base.json", "Base de conhecimento"),
]

# Fora do .zip (cache, derivado ou grande e recuperável). O apagar remove mesmo assim.
EXPORT_EXCLUDES = ("data/internet/regras_texto/", "data/internet/baixando/", "/runs/")
# Vídeos baixados pela coleta: ficam na pasta única de vídeos (data/videos/internet_*.mp4). O apagar os
# remove; o exportar não os leva (o catálogo guarda os links). Os demais vídeos da pasta são do usuário.
INTERNET_VIDEOS_DIR = "data/videos"
INTERNET_PREFIX = "internet_"
# Vídeos que o próprio sistema traz para o treino: coleta na internet e cortes dos golpes em dúvida das análises
DATASET_VIDEO_PREFIXES = ("internet_", "analise_")


def get_root(root: Optional[str] = None) -> str:
    """Pasta base (Dev/). Em modo de teste, uma pasta da sandbox — nunca os dados reais."""
    if root:
        return root
    if is_test_mode():
        d = os.path.join(get_test_sandbox_dir(), "dataset_root")
        os.makedirs(d, exist_ok=True)
        return d
    return os.getcwd()


def _abs(root: str, rel: str) -> str:
    return os.path.join(root, *rel.split("/"))


def _norm(rel: str) -> str:
    return rel.replace("\\", "/").lstrip("./")


def _is_training_path(rel: str) -> bool:
    rel = _norm(rel)
    return any(rel == p or rel.startswith(p.rstrip("/") + "/") for p, _ in TRAINING_STORES)


def _excluded(rel: str) -> bool:
    rel = "/" + _norm(rel)
    return any(x in rel if x.startswith("/") else rel.startswith("/" + x) for x in EXPORT_EXCLUDES)


def iter_training_files(root: Optional[str] = None, for_export: bool = True) -> Iterable[str]:
    """Caminhos relativos (com /) de todos os arquivos de treinamento existentes."""
    root = get_root(root)
    for rel, _ in TRAINING_STORES:
        p = _abs(root, rel)
        if os.path.isfile(p):
            if not (for_export and _excluded(rel)):
                yield rel
        elif os.path.isdir(p):
            for dirpath, _, files in os.walk(p):
                for name in sorted(files):
                    if name.endswith(".tmp"):
                        continue
                    r = _norm(os.path.relpath(os.path.join(dirpath, name), root))
                    if not (for_export and _excluded(r)):
                        yield r


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(root: Optional[str] = None) -> Dict[str, Any]:
    """Contagens reais do que existe (para a tela e para o manifest)."""
    root = get_root(root)

    def _json(rel, default):
        p = _abs(root, rel)
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    gab = _json("data/match_labels/gabarito_lutas.json", {}).get("videos", {})
    models = 0
    mroot = _abs(root, "data/models")
    if os.path.isdir(mroot):
        models = sum(1 for fam in os.listdir(mroot) if os.path.isfile(os.path.join(mroot, fam, "active.json")))
    cycles = 0
    cp = _abs(root, "data/training_cycles.jsonl")
    if os.path.isfile(cp):
        with open(cp, "r", encoding="utf-8") as f:
            cycles = sum(1 for line in f if line.strip())
    return {
        "revisoes": len(_json("data/feedback_dataset.json", [])),
        "rotulos_no_banco": len(_json("data/label_bank.json", {}).get("rotulos", {})),
        "lutas_com_gabarito": sum(1 for e in gab.values() if e.get("pontos_completos")),
        "lutas_no_golden": len(_json("data/benchmark_golden/golden_matches.json", {}).get("lutas", {})),
        "duvidas_na_fila": len(_json("data/active_learning_queue.json", [])),
        "decisoes_de_curadoria": len(_json("data/active_learning_curated_history.json", [])),
        "modelos_ativos": models,
        "ciclos_de_treino": cycles,
    }


# ------------------------------------------------------------------------------
# Exportar
# ------------------------------------------------------------------------------

def export_training_zip(root: Optional[str] = None) -> bytes:
    root = get_root(root)
    files = list(dict.fromkeys(iter_training_files(root, for_export=True)))
    manifest = {
        "formato": PACKAGE_FORMAT, "versao": PACKAGE_VERSION,
        "criado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "resumo": summarize(root),
        "arquivos": [{"caminho": rel, "bytes": os.path.getsize(_abs(root, rel)), "sha256": _sha256(_abs(root, rel))} for rel in files],
        "fora_do_pacote": list(EXPORT_EXCLUDES),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))
        for rel in files:
            compress = zipfile.ZIP_STORED if rel.lower().endswith((".mp4", ".pt", ".jpg", ".png")) else zipfile.ZIP_DEFLATED
            zf.write(_abs(root, rel), arcname=rel, compress_type=compress)
    log_event("INFO", f"Dataset de treinamento exportado: {len(files)} arquivo(s).", "training_dataset")
    return buf.getvalue()


def _backup(root: str, reason: str) -> Optional[str]:
    """Cópia de segurança antes de apagar ou importar (mantém as últimas MAX_BACKUPS)."""
    if not any(True for _ in iter_training_files(root, for_export=True)):
        return None
    d = _abs(root, BACKUP_DIR)
    os.makedirs(d, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(d, f"treinamento_antes_de_{reason}_{stamp}.zip")
    with open(path, "wb") as f:
        f.write(export_training_zip(root))
    olds = sorted(n for n in os.listdir(d) if n.startswith("treinamento_antes_de_") and n.endswith(".zip"))
    for n in olds[:-MAX_BACKUPS]:
        os.remove(os.path.join(d, n))
    return path


# ------------------------------------------------------------------------------
# Apagar
# ------------------------------------------------------------------------------

def _write_defaults(root: str) -> None:
    from src.engine.feedback_manager import DEFAULT_CALIBRATION_PROFILES
    p = _abs(root, "config/calibration_profiles.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CALIBRATION_PROFILES, f, indent=2, ensure_ascii=False)
    try:
        from src.engine.auto_trainer import AutoTrainingEngine
        AutoTrainingEngine(root=root).reset_knowledge_base()
    except Exception:
        pass


class TrainingBusyError(RuntimeError):
    pass


def _ensure_idle() -> None:
    """Não apaga nem restaura com um ciclo de treino rodando; desliga a procura de vídeos antes."""
    try:
        from src.engine.training_cycle import get_cycle_state
        if get_cycle_state().get("rodando"):
            raise TrainingBusyError("Há um ciclo de treino em andamento. Espere terminar antes de apagar ou importar.")
    except ImportError:
        pass
    try:
        from src.engine.internet_collector import stop_collection_session
        stop_collection_session(wait_sec=30)
    except ImportError:
        pass


def _clear_in_memory_state() -> None:
    """Modelos e estados em memória do processo (o pipeline da tela é recarregado pela própria tela)."""
    try:
        from src.analytics.pose_encoder import PoseEncoder
        PoseEncoder._cache.clear()
    except Exception:
        pass


def _move_to_trash(root: str) -> Tuple[str, List[Tuple[str, str]], int]:
    """
    Move todos os itens da lista para uma lixeira temporária (renomear é instantâneo). Se algo falhar,
    devolve tudo ao lugar e propaga o erro: ou o apagar acontece inteiro, ou nada muda.
    """
    trash = _abs(root, f"data/.lixeira_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    moved: List[Tuple[str, str]] = []
    count = 0
    try:
        for i, (rel, _) in enumerate(TRAINING_STORES):
            src = _abs(root, rel)
            if not os.path.exists(src):
                continue
            count += 1 if os.path.isfile(src) else sum(len(f) for _, _, f in os.walk(src))
            dst = os.path.join(trash, str(i))
            os.makedirs(trash, exist_ok=True)
            os.replace(src, dst)
            moved.append((src, dst))
        vdir = _abs(root, INTERNET_VIDEOS_DIR)
        if os.path.isdir(vdir):
            for name in sorted(os.listdir(vdir)):
                src = os.path.join(vdir, name)
                if name.startswith(DATASET_VIDEO_PREFIXES) and os.path.isfile(src):
                    count += 1
                    dst = os.path.join(trash, "videos", name)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.replace(src, dst)
                    moved.append((src, dst))
    except Exception:
        for src, dst in reversed(moved):
            try:
                os.makedirs(os.path.dirname(src), exist_ok=True)
                os.replace(dst, src)
            except Exception as e:
                log_event("ERROR", f"Falha ao desfazer a remoção de {src}: {e}", "training_dataset")
        shutil.rmtree(trash, ignore_errors=True)
        raise
    return trash, moved, count


def _remove_tree(path: str, attempts: int = 5) -> None:
    """No Windows, uma pasta pode ficar presa por um instante ou conter arquivos somente leitura: tenta de novo."""
    import time
    import stat

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            func(p)
        except Exception:
            pass

    for i in range(attempts):
        try:
            shutil.rmtree(path, onerror=_onerror)
        except TypeError:
            def _onexc(func, p, exc):
                try:
                    os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
                    func(p)
                except Exception:
                    pass
            try:
                shutil.rmtree(path, onexc=_onexc)
            except Exception:
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            shutil.rmtree(path, ignore_errors=True)
        if not os.path.exists(path):
            return
        time.sleep(0.2 * (i + 1))
    log_event("WARNING", f"Não foi possível remover a pasta temporária {path}; ela será removida na próxima vez.", "training_dataset")


def _cleanup_leftovers(root: str) -> None:
    """Remove lixeiras e extrações temporárias que tenham sobrado de operações anteriores."""
    d = _abs(root, "data")
    if os.path.isdir(d):
        for name in os.listdir(d):
            if name.startswith((".lixeira_", ".importando_")):
                _remove_tree(os.path.join(d, name))


def _delete_all(root: str) -> int:
    trash, _, removed = _move_to_trash(root)
    _remove_tree(trash)
    _cleanup_leftovers(root)
    return removed


def _run_to_completion(fn, *args, **kwargs):
    """
    Roda numa thread própria e espera. Se a tela do Streamlit for recarregada no meio (o que interrompe o
    script), a operação continua até o fim em vez de parar pela metade.
    """
    import threading
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 — repassado a quem chamou
            box["error"] = e
            log_event("ERROR", f"{fn.__name__} falhou: {type(e).__name__}: {e}", "training_dataset")

    with _operation_lock:
        t = threading.Thread(target=_target, name=f"senpai-{fn.__name__}")
        t.start()
        t.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


import threading as _threading  # noqa: E402
_operation_lock = _threading.Lock()


def reset_training(root: Optional[str] = None, backup: bool = True) -> Dict[str, Any]:
    """Zera todo o treinamento. Devolve o que havia antes, o que foi removido e onde está a cópia de segurança."""
    return _run_to_completion(_reset_training, get_root(root), backup)


def _reset_training(root: str, backup: bool) -> Dict[str, Any]:
    _ensure_idle()
    before = summarize(root)
    bkp = _backup(root, "apagar") if backup else None
    removed = _delete_all(root)
    _write_defaults(root)
    _clear_in_memory_state()
    after = summarize(root)
    log_event("WARNING", f"TREINAMENTO APAGADO: {removed} arquivo(s) removido(s). Antes: {before}. Cópia: {bkp}.", "training_dataset")
    return {"antes": before, "depois": after, "arquivos_removidos": removed, "copia_de_seguranca": bkp}


# ------------------------------------------------------------------------------
# Importar (restaurar)
# ------------------------------------------------------------------------------

class InvalidPackageError(ValueError):
    pass


def read_manifest(zf: zipfile.ZipFile) -> Optional[Dict[str, Any]]:
    if MANIFEST_NAME not in zf.namelist():
        return None
    manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
    if manifest.get("formato") != PACKAGE_FORMAT:
        raise InvalidPackageError("O manifest não é de um pacote de treinamento do SenpAI.")
    if int(manifest.get("versao", 0)) > PACKAGE_VERSION:
        raise InvalidPackageError(f"Pacote de versão {manifest.get('versao')} é mais novo que este SenpAI ({PACKAGE_VERSION}).")
    return manifest


def validate_package(zf: zipfile.ZipFile, manifest: Dict[str, Any]) -> None:
    names = set(zf.namelist())
    for f in manifest.get("arquivos", []):
        rel = _norm(f["caminho"])
        if ".." in rel.split("/") or os.path.isabs(f["caminho"]) or not _is_training_path(rel):
            raise InvalidPackageError(f"Caminho não permitido no pacote: {f['caminho']}")
        if rel not in names:
            raise InvalidPackageError(f"Arquivo listado no manifest e ausente do pacote: {rel}")
        if hashlib.sha256(zf.read(rel)).hexdigest() != f["sha256"]:
            raise InvalidPackageError(f"Arquivo corrompido no pacote (SHA-256 não confere): {rel}")


def import_training_zip(data: bytes, root: Optional[str] = None, backup: bool = True) -> Dict[str, Any]:
    """Restaura exatamente o estado do pacote. Pacote inválido não altera nada."""
    return _run_to_completion(_import_training_zip, data, get_root(root), backup)


def _import_training_zip(data: bytes, root: str, backup: bool) -> Dict[str, Any]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise InvalidPackageError(f"Arquivo .zip inválido: {e}")
    with zf:
        manifest = read_manifest(zf)
        if manifest is None:
            raise InvalidPackageError("Pacote sem manifest.json (formato antigo).")
        validate_package(zf, manifest)
        _ensure_idle()
        bkp = _backup(root, "importar") if backup else None
        staging = _abs(root, f"data/.importando_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
        try:
            for f in manifest["arquivos"]:
                rel = _norm(f["caminho"])
                dest = os.path.join(staging, *rel.split("/"))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as out:
                    out.write(zf.read(rel))
            trash, moved, _ = _move_to_trash(root)
            try:
                for f in manifest["arquivos"]:
                    rel = _norm(f["caminho"])
                    dest = _abs(root, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    os.replace(os.path.join(staging, *rel.split("/")), dest)
            except Exception:
                for src, dst in reversed(moved):       # volta ao estado anterior
                    if os.path.isdir(src):
                        shutil.rmtree(src, ignore_errors=True)
                    elif os.path.isfile(src):
                        os.remove(src)
                    os.makedirs(os.path.dirname(src), exist_ok=True)
                    os.replace(dst, src)
                raise
            _remove_tree(trash)
        finally:
            _remove_tree(staging)
        if not os.path.isfile(_abs(root, "config/calibration_profiles.json")):
            _write_defaults(root)
    _clear_in_memory_state()
    after = summarize(root)
    log_event("INFO", f"Treinamento restaurado de pacote de {manifest.get('criado_em')}: {len(manifest['arquivos'])} arquivo(s).",
              "training_dataset")
    return {"pacote_criado_em": manifest.get("criado_em"), "arquivos": len(manifest["arquivos"]),
            "resumo_do_pacote": manifest.get("resumo"), "estado_atual": after, "copia_de_seguranca": bkp}


def is_new_format(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return MANIFEST_NAME in zf.namelist()
    except zipfile.BadZipFile:
        return False


SUMMARY_LABELS = {
    "revisoes": "Revisões de golpes", "rotulos_no_banco": "Rótulos no banco", "lutas_com_gabarito": "Lutas com gabarito",
    "lutas_no_golden": "Lutas no golden", "duvidas_na_fila": "Dúvidas na fila", "decisoes_de_curadoria": "Decisões de curadoria",
    "modelos_ativos": "Modelos ativos",
    "ciclos_de_treino": "Ciclos de treino",
}


# ------------------------------------------------------------------------------
# Quanto o sistema já foi treinado: disco e progresso (mesma lista do apagar/exportar)
# ------------------------------------------------------------------------------

# Grupo de cada item da lista, para a tela de disco
STORE_GROUPS = {
    "dataset": ("📂 Dataset de treinamento", "Rótulos, revisões, gabarito, golden, curadoria (com clipes) e coleta"),
    "modelos": ("🧠 Modelos treinados", "Filtro de golpes, técnica, modalidade e encoder de movimento"),
    "conhecimento": ("⚙️ Conhecimento & calibração", "Perfis de calibração e base de conhecimento"),
}

# Apoio: ocupa disco, mas não é treinamento (não é apagado nem exportado)
SUPPORT_ITEMS: List[Tuple[str, str]] = [
    ("data/feature_cache", "Cache de poses (refeito a partir dos vídeos)"),
    ("models", "Modelos base baixados prontos (YOLO pose)"),
]

MODEL_FAMILIES = {
    "strike_filter": "Filtro de golpes", "technique_classifier": "Técnica", "modality_classifier": "Luta × modalidades",
    "pose_encoder": "Encoder de movimento",
}


def _group_of(rel: str) -> str:
    if rel.startswith("data/models"):
        return "modelos"
    if rel.startswith("config/"):
        return "conhecimento"
    return "dataset"


def _size_and_count(path: str, exclude_sub: Optional[str] = None) -> Tuple[int, int]:
    if os.path.isfile(path):
        return os.path.getsize(path), 1
    total = n = 0
    if os.path.isdir(path):
        for dirpath, _, files in os.walk(path):
            if exclude_sub and os.path.normpath(dirpath).startswith(os.path.normpath(exclude_sub)):
                continue
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                    n += 1
                except OSError:
                    pass
    return total, n


def format_bytes(n: int) -> str:
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.2f} {unit}" if unit != "KB" else f"{n / div:.1f} KB"
    return f"{n} B"


def storage_report(root: Optional[str] = None) -> Dict[str, Any]:
    """Disco ocupado pelo treinamento (por item da lista e por grupo) e, à parte, pelo que é só apoio."""
    root = get_root(root)
    items, groups = [], {k: 0 for k in STORE_GROUPS}
    for rel, desc in TRAINING_STORES:
        size, n = _size_and_count(_abs(root, rel))
        g = _group_of(rel)
        groups[g] += size
        items.append({"caminho": rel, "descricao": desc, "grupo": g, "arquivos": n, "bytes": size})
    # Pasta única de vídeos: os da coleta contam como dataset; os do usuário são apoio
    vdir = _abs(root, INTERNET_VIDEOS_DIR)
    inet, user = (0, 0), (0, 0)
    if os.path.isdir(vdir):
        for dirpath, _, files in os.walk(vdir):
            for name in files:
                sz = os.path.getsize(os.path.join(dirpath, name))
                if dirpath == vdir and name.startswith(DATASET_VIDEO_PREFIXES):
                    inet = (inet[0] + sz, inet[1] + 1)
                else:
                    user = (user[0] + sz, user[1] + 1)
    groups["dataset"] += inet[0]
    items.append({"caminho": f"{INTERNET_VIDEOS_DIR}/internet_* e analise_*", "descricao": "Vídeos da coleta na internet e cortes das análises",
                  "grupo": "dataset", "arquivos": inet[1], "bytes": inet[0]})
    support = [{"caminho": INTERNET_VIDEOS_DIR, "descricao": "Seus vídeos de lutas e treinos", "arquivos": user[1], "bytes": user[0]}]
    for rel, desc in SUPPORT_ITEMS:
        size, n = _size_and_count(_abs(root, rel))
        support.append({"caminho": rel, "descricao": desc, "arquivos": n, "bytes": size})
    total = sum(groups.values())
    return {"total_bytes": total, "grupos": groups, "itens": items, "apoio": support,
            "apoio_bytes": sum(s["bytes"] for s in support)}


def training_progress(root: Optional[str] = None) -> Dict[str, Any]:
    """Modelos treinados em uso e a última medição real no golden (None se nunca medido)."""
    root = get_root(root)
    models = {}
    for fam, label in MODEL_FAMILIES.items():
        p = _abs(root, f"data/models/{fam}/active.json")
        active = None
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    a = json.load(f)
                active = {"arquivo": os.path.basename(str(a.get("path") or a.get("model_path") or "")),
                          "ativado_em": a.get("activated_at") or a.get("ativado_em"), "motivo": a.get("reason") or a.get("motivo")}
            except Exception:
                active = {"arquivo": "?", "ativado_em": None, "motivo": None}
        models[fam] = {"nome": label, "ativo": active}
    last = None
    hp = _abs(root, "data/benchmark_golden/benchmark_history.jsonl")
    if os.path.isfile(hp):
        with open(hp, "r", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip()]
        if lines:
            try:
                last = json.loads(lines[-1])
            except Exception:
                last = None
    return {"modelos": models, "modelos_ativos": sum(1 for m in models.values() if m["ativo"]),
            "ultima_medicao_golden": last}
