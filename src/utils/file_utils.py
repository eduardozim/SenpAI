"""
Utilitários de I/O de arquivos com resiliência a travas (file locks) no Windows.
Garante que escritas atômicas com .tmp -> destino não falhem por concorrência de leitura
(ex.: threads do Streamlit renderizando métricas) ou por escaneamento de antivírus (Windows Defender).
"""

import os
import time
import json
import shutil
import threading
from typing import Any, Optional

# Lock global reentrante para sincronização de I/O de dados dentro do processo Python
_GLOBAL_FILE_IO_LOCK = threading.RLock()


def safe_replace(src: str, dst: str, max_retries: int = 25, delay: float = 0.05) -> None:
    """
    Substitui src por dst de forma segura e atômica no Windows.
    Possui retentativas com backoff exponencial para contornar locks temporários
    de leitura mantidos por outras threads ou por antivírus.
    Se a substituição atômica falhar após todas as retentativas (por exemplo, lock
    exclusivo de renomeação), faz fallback para sobrescrita direta de conteúdo (shutil.copyfile).
    """
    for i in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError) as e:
            if i == max_retries - 1:
                # Fallback de emergência no Windows: cópia direta de conteúdo sobre dst
                try:
                    shutil.copyfile(src, dst)
                    try:
                        os.remove(src)
                    except Exception:
                        pass
                    return
                except Exception:
                    raise e
            time.sleep(delay * (1.15 ** i))


def safe_write_json(path: str, data: Any, indent: int = 2) -> None:
    """
    Grava dados em formato JSON em um arquivo temporário e faz a substituição segura para o destino.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with _GLOBAL_FILE_IO_LOCK:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        safe_replace(tmp, path)


def safe_load_json(path: str, default: Any = None) -> Any:
    """
    Carrega dados JSON com retentativas caso o arquivo esteja sendo gravado/substituído
    simultaneamente por outra thread.
    """
    if not os.path.isfile(path):
        return default
    with _GLOBAL_FILE_IO_LOCK:
        for retry in range(5):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, PermissionError):
                if retry == 4:
                    tmp = f"{path}.tmp"
                    if os.path.isfile(tmp):
                        try:
                            with open(tmp, "r", encoding="utf-8") as f_tmp:
                                return json.load(f_tmp)
                        except Exception:
                            pass
                    return default
                time.sleep(0.02 * (retry + 1))
            except Exception:
                return default
        return default

