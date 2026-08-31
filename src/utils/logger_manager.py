"""
Módulo de Gerenciamento de Logs, Alertas e Diagnóstico de Debug do SenpAI.
Captura eventos do sistema, erros, avisos e possibilita visualização e download dos arquivos de log.
"""

import os
import sys
import json
import time
import logging
import datetime
import platform
from collections import deque
from typing import Dict, Any, List, Optional

DEFAULT_LOG_PATH = "logs/senpai_debug.log"
MAX_MEMORY_LOGS = 1000

class RingBufferHandler(logging.Handler):
    """
    Handler de logging customizado que retém as últimas N entradas em memória.
    """
    def __init__(self, capacity: int = MAX_MEMORY_LOGS):
        super().__init__()
        self.buffer: deque = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            entry = {
                "timestamp": datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
                "level": record.levelname,
                "module": record.module,
                "filename": record.filename,
                "lineno": record.lineno,
                "message": record.getMessage(),
                "formatted": msg
            }
            self.buffer.append(entry)
        except Exception:
            self.handleError(record)

_ring_handler: Optional[RingBufferHandler] = None
_logger_initialized = False

def setup_system_logger(log_file: str = DEFAULT_LOG_PATH, level: int = logging.DEBUG) -> logging.Logger:
    """
    Inicializa e configura o logger central do sistema SenpAI com salvamento em arquivo e buffer em memória.
    """
    global _ring_handler, _logger_initialized

    root_logger = logging.getLogger("senpai")
    root_logger.setLevel(level)

    # Evitar duplicar handlers se já inicializado
    if _logger_initialized:
        return root_logger

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Memory RingBuffer handler
    _ring_handler = RingBufferHandler(capacity=MAX_MEMORY_LOGS)
    _ring_handler.setLevel(level)
    _ring_handler.setFormatter(formatter)
    root_logger.addHandler(_ring_handler)

    # Silencia exceção inofensiva do ProactorEventLoop no Windows (WinError 10054) ao fechar/recarregar a aba do navegador
    if sys.platform == "win32":
        try:
            from asyncio.proactor_events import _ProactorBasePipeTransport
            _orig_call_conn_lost = _ProactorBasePipeTransport._call_connection_lost

            def _silenced_call_conn_lost(self, exc):
                try:
                    _orig_call_conn_lost(self, exc)
                except (ConnectionResetError, OSError) as err:
                    if getattr(err, "winerror", None) == 10054 or isinstance(err, ConnectionResetError):
                        pass
                    else:
                        raise

            _ProactorBasePipeTransport._call_connection_lost = _silenced_call_conn_lost
        except Exception:
            pass

    _logger_initialized = True
    root_logger.info(f"Sistema de Log & Debug inicializado. Arquivo: '{log_file}'")

    return root_logger

def get_system_logger() -> logging.Logger:
    if not _logger_initialized:
        return setup_system_logger()
    return logging.getLogger("senpai")

def log_event(level: str, message: str, module_name: str = "app") -> None:
    logger = get_system_logger()
    lvl_upper = level.upper().strip()
    if lvl_upper == "ERROR":
        logger.error(f"[{module_name}] {message}")
    elif lvl_upper == "WARNING" or lvl_upper == "WARN":
        logger.warning(f"[{module_name}] {message}")
    elif lvl_upper == "DEBUG":
        logger.debug(f"[{module_name}] {message}")
    else:
        logger.info(f"[{module_name}] {message}")

def get_memory_logs(max_entries: int = 200, level_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    global _ring_handler
    if not _ring_handler:
        setup_system_logger()

    entries = list(_ring_handler.buffer) if _ring_handler else []

    if level_filter and level_filter != "TODOS":
        lvl_clean = level_filter.upper().strip()
        entries = [e for e in entries if e["level"] == lvl_clean]

    entries.reverse() # Mostrar os mais recentes primeiro
    return entries[:max_entries]

def get_debug_log_file_content(log_file: str = DEFAULT_LOG_PATH) -> str:
    """
    Retorna todo o conteúdo textual do arquivo de log de debug para download.
    """
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Erro ao ler arquivo de log: {e}"
    
    # Se não houver arquivo ainda, gerar a partir do buffer em memória
    entries = get_memory_logs(max_entries=1000)
    lines = [e.get("formatted", f"[{e['timestamp']}] [{e['level']}] {e['message']}") for e in reversed(entries)]
    return "\n".join(lines) if lines else "Nenhum log registrado até o momento."

def get_log_summary() -> Dict[str, int]:
    global _ring_handler
    if not _ring_handler:
        setup_system_logger()

    entries = list(_ring_handler.buffer) if _ring_handler else []

    total = len(entries)
    errors = sum(1 for e in entries if e["level"] == "ERROR")
    warnings = sum(1 for e in entries if e["level"] in ["WARNING", "WARN"])
    info = sum(1 for e in entries if e["level"] == "INFO")
    debug = sum(1 for e in entries if e["level"] == "DEBUG")

    return {
        "total_logs": total,
        "errors_count": errors,
        "warnings_count": warnings,
        "info_count": info,
        "debug_count": debug
    }

def clear_debug_logs(log_file: str = DEFAULT_LOG_PATH) -> None:
    """
    Limpa o arquivo de log no disco e reseta o buffer de memória.
    """
    global _ring_handler
    if _ring_handler:
        _ring_handler.buffer.clear()

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] [logger] Log de debug reiniciado.\n")

    logger = get_system_logger()
    logger.info("Histórico de logs de debug limpo pelo usuário.")

def run_system_diagnostic_check() -> Dict[str, Any]:
    """
    Executa um teste completo de diagnóstico do sistema (Hardware, CUDA, Arquivos e Dependências)
    e grava os alertas no log do sistema.
    """
    logger = get_system_logger()
    logger.info("=== INICIANDO TESTE DE DIAGNÓSTICO DO SISTEMA SENPAI ===")

    diagnostic_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "checks": {}
    }

    # 1. Checar estrutura de diretórios e escrita de arquivos
    try:
        os.makedirs("config", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        test_file = "logs/test_write.tmp"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)

        diagnostic_report["checks"]["filesystem"] = {"status": "OK", "message": "Acesso de leitura/escrita confirmado."}
        logger.info("DIAGNOSTIC: Sistema de arquivos OK.")
    except Exception as ex:
        diagnostic_report["checks"]["filesystem"] = {"status": "ERROR", "message": str(ex)}
        logger.error(f"DIAGNOSTIC ERROR: Falha no sistema de arquivos: {ex}")

    # 2. Checar Hardware e GPU NVIDIA
    try:
        from src.utils.hardware import detect_nvidia_gpu, check_cuda_framework_support
        gpu_info = detect_nvidia_gpu()
        cuda_info = check_cuda_framework_support()

        diagnostic_report["checks"]["hardware"] = {
            "gpu_detected": gpu_info["has_nvidia_gpu"],
            "gpu_name": gpu_info.get("gpu_name", "Nenhum"),
            "torch_cuda": cuda_info.get("torch_cuda", False),
            "onnx_cuda": cuda_info.get("onnx_cuda", False)
        }

        if gpu_info["has_nvidia_gpu"]:
            logger.info(f"DIAGNOSTIC: GPU NVIDIA detectada: {gpu_info['gpu_name']}. PyTorch CUDA: {cuda_info.get('torch_cuda')}")
        else:
            logger.warning("DIAGNOSTIC WARNING: Nenhuma GPU NVIDIA dedicada encontrada. Rodando em modo CPU.")
    except Exception as ex:
        diagnostic_report["checks"]["hardware"] = {"status": "ERROR", "message": str(ex)}
        logger.error(f"DIAGNOSTIC ERROR: Erro na checagem de hardware: {ex}")

    # 3. Checar dependências críticas (OpenCV, MediaPipe, PyTorch/NumPy)
    for lib_name in ["cv2", "mediapipe", "numpy", "torch"]:
        try:
            mod = __import__(lib_name)
            version = getattr(mod, "__version__", "OK")
            diagnostic_report["checks"][f"lib_{lib_name}"] = {"status": "OK", "version": version}
            logger.info(f"DIAGNOSTIC: Biblioteca '{lib_name}' pronta (Versão {version}).")
        except Exception as ex:
            diagnostic_report["checks"][f"lib_{lib_name}"] = {"status": "ERROR", "message": str(ex)}
            logger.error(f"DIAGNOSTIC ERROR: Falha na biblioteca '{lib_name}': {ex}")

    # 4. Checar Ambiente Virtual Python
    try:
        from src.utils.environment import get_virtual_environment_info
        env_info = get_virtual_environment_info()
        if env_info["is_virtual_env"]:
            diagnostic_report["checks"]["virtual_environment"] = {
                "status": "OK",
                "is_virtual_env": True,
                "env_type": env_info["env_type"],
                "executable": env_info["executable"]
            }
            logger.info(f"DIAGNOSTIC: Ambiente Virtual ativo ({env_info['env_type']}): {env_info['executable']}")
        else:
            diagnostic_report["checks"]["virtual_environment"] = {
                "status": "WARNING",
                "is_virtual_env": False,
                "env_type": "system_global",
                "executable": env_info["executable"]
            }
            logger.warning(f"DIAGNOSTIC WARNING: Ambiente Virtual não identificado! Rodando no interpretador global: {env_info['executable']}")
    except Exception as ex:
        diagnostic_report["checks"]["virtual_environment"] = {"status": "ERROR", "message": str(ex)}
        logger.error(f"DIAGNOSTIC ERROR: Erro na checagem de ambiente virtual: {ex}")

    logger.info("=== DIAGNÓSTICO DO SISTEMA CONCLUÍDO ===")
    return diagnostic_report

