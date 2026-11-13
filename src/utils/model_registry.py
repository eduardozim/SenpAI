"""
Registro único dos modelos treinados pelo SenpAI.

Cada família de modelo tem uma pasta em data/models/<familia>/ com as versões treinadas e um
ponteiro active.json para a versão em uso. Usado pelo filtro de golpes (I3), técnica, modalidade e
encoder de movimento. Em modo de teste, tudo vive na sandbox isolada.
"""

import os
import json
import datetime
from typing import Any, Dict, Optional

from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir

DEFAULT_MODELS_ROOT = os.path.join("data", "models")
ACTIVE_POINTER_FILE = "active.json"


def get_models_dir(family: str) -> str:
    root = os.path.join(get_test_sandbox_dir(), "models") if is_test_mode() else DEFAULT_MODELS_ROOT
    d = os.path.join(root, family)
    os.makedirs(d, exist_ok=True)
    return d


def get_active_model_path(family: str, models_dir: Optional[str] = None) -> Optional[str]:
    pointer = os.path.join(models_dir or get_models_dir(family), ACTIVE_POINTER_FILE)
    if not os.path.isfile(pointer):
        return None
    try:
        with open(pointer, "r", encoding="utf-8") as f:
            path = json.load(f).get("model_path")
        return path if path and os.path.isfile(path) else None
    except Exception:
        return None


def get_active_info(family: str, models_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    pointer = os.path.join(models_dir or get_models_dir(family), ACTIVE_POINTER_FILE)
    if not os.path.isfile(pointer):
        return None
    try:
        with open(pointer, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def set_active_model(family: str, path: Optional[str], reason: str = "", models_dir: Optional[str] = None) -> None:
    pointer = os.path.join(models_dir or get_models_dir(family), ACTIVE_POINTER_FILE)
    if path is None:
        if os.path.isfile(pointer):
            os.remove(pointer)
        log_event("INFO", f"Modelo '{family}' desativado. {reason}", "model_registry")
        return
    with open(pointer, "w", encoding="utf-8") as f:
        json.dump({"model_path": path, "ativado_em": datetime.datetime.now().isoformat(timespec="seconds"), "motivo": reason},
                  f, indent=2, ensure_ascii=False)
    log_event("INFO", f"Modelo '{family}' ativo: {os.path.basename(path)}. {reason}", "model_registry")
