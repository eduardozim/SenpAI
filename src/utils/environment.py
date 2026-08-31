"""
Módulo de Verificação e Diagnóstico de Ambiente Virtual Python para o SenpAI.
Identifica se a aplicação está em execução dentro de um ambiente isolado (venv, .venv, virtualenv, conda)
e provê alertas descritivos caso esteja sendo executado no interpretador global do sistema.
"""

import sys
import os
from typing import Dict, Any, Tuple
from src.utils.logger_manager import log_event


def is_in_virtual_environment() -> bool:
    """
    Verifica se o interpretador Python atual está executando dentro de um ambiente virtual isolado.
    
    Critérios de detecção:
    1. sys.prefix != sys.base_prefix (Padrão venv / virtualenv do Python 3.3+)
    2. hasattr(sys, 'real_prefix') (virtualenv legado)
    3. Variável de ambiente VIRTUAL_ENV definida
    4. Variável de ambiente CONDA_PREFIX definida
    """
    # 1. Checagem padrão do Python 3.3+ (venv)
    base_prefix = getattr(sys, "base_prefix", None)
    if base_prefix is not None and sys.prefix != base_prefix:
        return True

    # 2. Virtualenv legado
    if hasattr(sys, "real_prefix"):
        return True

    # 3. Variável VIRTUAL_ENV do shell ativo
    if os.environ.get("VIRTUAL_ENV"):
        return True

    # 4. Conda Environment
    if os.environ.get("CONDA_PREFIX"):
        return True

    # 5. base_exec_prefix check
    base_exec_prefix = getattr(sys, "base_exec_prefix", None)
    if base_exec_prefix is not None and sys.exec_prefix != base_exec_prefix:
        return True

    return False


def get_virtual_environment_info() -> Dict[str, Any]:
    """
    Retorna metadados detalhados sobre o ambiente de execução Python atual.
    """
    in_venv = is_in_virtual_environment()
    base_p = getattr(sys, "base_prefix", sys.prefix)
    curr_p = sys.prefix
    
    env_type = "system_global"
    if in_venv:
        if os.environ.get("CONDA_PREFIX"):
            env_type = "conda"
        elif hasattr(sys, "real_prefix"):
            env_type = "virtualenv"
        else:
            env_type = "venv"

    # Verifica se aponta para a pasta local .venv do projeto
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    expected_local_venv = os.path.join(workspace_dir, ".venv")
    is_project_dot_venv = os.path.exists(expected_local_venv) and (
        os.path.normpath(curr_p).lower() == os.path.normpath(expected_local_venv).lower()
        or expected_local_venv.lower() in curr_p.lower()
    )

    return {
        "is_virtual_env": in_venv,
        "env_type": env_type,
        "prefix": curr_p,
        "base_prefix": base_p,
        "executable": sys.executable,
        "python_version": sys.version.split()[0],
        "is_project_dot_venv": is_project_dot_venv,
        "expected_dot_venv_path": expected_local_venv
    }


def validate_virtual_environment(log_warning: bool = True) -> Tuple[bool, str]:
    """
    Valida a presença de ambiente virtual e gera uma mensagem explicativa com passos de correção.
    
    Retorna:
    - (is_valid, message): Tupla contendo status booleano e mensagem descritiva.
    """
    info = get_virtual_environment_info()
    
    if info["is_virtual_env"]:
        msg = f"Ambiente Virtual ativo ({info['env_type']}): {info['prefix']}"
        return True, msg

    error_msg = (
        "⚠️ AMBIENTE VIRTUAL PYTHON NÃO IDENTIFICADO!\n"
        f"O SenpAI está sendo executado diretamente no interpretador global do sistema: '{sys.executable}'.\n\n"
        "Recomenda-se fortemente a utilização de um ambiente virtual isolado (.venv):\n"
        "  1. Criar o ambiente virtual: py -3.11 -m venv .venv\n"
        "  2. Ativar o ambiente no Windows: .\\.venv\\Scripts\\activate\n"
        "  3. Instalar dependências: pip install -r requirements.txt\n"
        "  4. Iniciar o SenpAI: python -m streamlit run app.py"
    )
    
    if log_warning:
        log_event(
            "WARNING",
            f"Ambiente Virtual Python não identificado! Executando no interpretador global: {sys.executable}",
            "environment"
        )
        
    return False, error_msg
