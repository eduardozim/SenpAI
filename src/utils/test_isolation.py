"""
Módulo de Isolamento e Proteção de Dados para Ambientes de Teste no SenpAI.
Garante que execuções de testes automatizados (unitários, integração e via UI)
NUNCA alterem, sobrescrevam ou apaguem dados reais de produção/desenvolvimento
(como histórico de treino, datasets de feedback, perfis de calibração e baselines).
"""

import os
import tempfile
from typing import Optional

ENV_TEST_MODE = "SENPAI_TEST_MODE"

# Arquivos críticos protegidos de produção/desenvolvimento
PROTECTED_FILES = {
    "data/training_history.json": "test_training_history.json",
    "data/feedback_dataset.json": "test_feedback_dataset.json",
    "config/calibration_profiles.json": "test_calibration_profiles.json",
    "data/shinpan_reviewed_videos.json": "test_shinpan_reviewed_videos.json",
    "config/ai_knowledge_base.json": "test_ai_knowledge_base.json",
    "data/auto_training_checkpoint.json": "test_auto_training_checkpoint.json",
    "data/kenshi_baselines.json": "test_kenshi_baselines.json",
    "data/active_learning_queue.json": "test_active_learning_queue.json",
    "data/empirical_reference_distributions.json": "test_empirical_reference_distributions.json",
}

def is_test_mode() -> bool:
    """Retorna True se o sistema estiver rodando em modo de teste."""
    return os.environ.get(ENV_TEST_MODE, "").strip().lower() in ("1", "true", "yes", "on")

def enable_test_mode() -> None:
    """Ativa o modo de teste no ambiente do processo atual."""
    os.environ[ENV_TEST_MODE] = "1"

def disable_test_mode() -> None:
    """Desativa o modo de teste no ambiente do processo atual."""
    if ENV_TEST_MODE in os.environ:
        del os.environ[ENV_TEST_MODE]

def get_test_sandbox_dir() -> str:
    """Retorna o diretório isolado temporário de testes para armazenamento seguro."""
    sandbox_dir = os.path.join(tempfile.gettempdir(), "senpai_test_sandbox")
    os.makedirs(sandbox_dir, exist_ok=True)
    return sandbox_dir

def get_safe_data_path(requested_path: str, default_fallback: Optional[str] = None) -> str:
    """
    Resolve o caminho de arquivo garantindo que, em modo de teste, nenhum arquivo
    real de produção/desenvolvimento seja aberto para gravação ou modificação.
    Se estiver em modo de teste e o caminho solicitado coincidir com um arquivo protegido,
    redireciona automaticamente para a pasta temporária de sandbox.
    """
    if not requested_path:
        requested_path = default_fallback or ""

    if not is_test_mode():
        return requested_path

    # Normalizar para comparação agnóstica de SO
    norm_req = os.path.normpath(requested_path).replace("\\", "/").lower()

    for prot_key, test_name in PROTECTED_FILES.items():
        norm_prot = os.path.normpath(prot_key).replace("\\", "/").lower()
        if norm_req.endswith(norm_prot) or norm_req == norm_prot:
            sandbox = get_test_sandbox_dir()
            safe_target = os.path.join(sandbox, test_name)
            
            # Se o arquivo de produção existir e o de teste ainda não, inicializar cópia segura
            if os.path.isfile(requested_path) and not os.path.isfile(safe_target):
                try:
                    import shutil
                    shutil.copy2(requested_path, safe_target)
                except Exception:
                    pass
            return safe_target

    return requested_path

def assert_not_production_write(file_path: str, operation_name: str = "gravação") -> None:
    """
    Lança RuntimeError se houver tentativa de escrita direta em arquivo protegido em modo de teste.
    """
    if not is_test_mode():
        return

    norm_path = os.path.normpath(file_path).replace("\\", "/").lower()
    for prot_key in PROTECTED_FILES.keys():
        norm_prot = os.path.normpath(prot_key).replace("\\", "/").lower()
        if norm_path.endswith(norm_prot) or norm_path == norm_prot:
            raise RuntimeError(
                f"[SEGURANÇA SENPAI] Tentativa de {operation_name} bloqueada em modo de teste no arquivo real: '{file_path}'. "
                f"Utilize um caminho em diretório temporário isolado ou get_safe_data_path()."
            )

