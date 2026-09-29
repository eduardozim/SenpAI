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


def get_execution_environment_info() -> Dict[str, Any]:
    """
    Detecta automaticamente se o SenpAI está executando em um ambiente Local
    (Desktop / Dojo / Localhost no Windows, macOS ou Linux com hardware direto)
    ou em um Servidor Web / Nuvem (Streamlit Community Cloud, Hugging Face Spaces,
    Docker, AWS, GCP, etc.).

    Retorna metadados para que a aplicação adapte suas fontes de captura de câmera
    (priorizando WebRTC no navegador quando na nuvem e OpenCV USB direto quando local).
    """
    # 1. Override explícito via variável de ambiente (útil para testes de simulação)
    override = os.environ.get("SENPAI_DEPLOY_MODE", "").lower().strip()
    if override in ["cloud", "web"]:
        return {
            "is_cloud": True,
            "deployment_mode": "cloud",
            "provider_label": "Servidor Web / Nuvem (Configurado)",
            "description": "Executando em Servidor Web/Nuvem. Câmeras locais do cliente devem ser capturadas via WebRTC pelo navegador.",
            "recommended_source": "webrtc"
        }
    elif override in ["local", "desktop"]:
        return {
            "is_cloud": False,
            "deployment_mode": "local",
            "provider_label": "Local (Desktop / Dojo)",
            "description": "Executando em máquina local. Acesso direto a Webcams USB via OpenCV e aceleração nativa disponíveis.",
            "recommended_source": "webcam"
        }

    # 2. Detecção específica do Streamlit Community Cloud
    # No Streamlit Cloud, repositórios são montados sob /mount/src/<repo>
    is_streamlit_cloud = os.path.exists("/mount/src") or bool(
        os.environ.get("IS_STREAMLIT_CLOUD")
        or os.environ.get("STREAMLIT_SHARING")
        or ("STREAMLIT_SERVER_GATHER_USAGE_STATS" in os.environ and os.name != "nt" and not os.environ.get("DISPLAY"))
    )
    if is_streamlit_cloud:
        return {
            "is_cloud": True,
            "deployment_mode": "cloud",
            "provider_label": "Streamlit Community Cloud",
            "description": "Servidor Streamlit Cloud detectado. Câmeras do computador do usuário devem ser capturadas pelo navegador (WebRTC).",
            "recommended_source": "webrtc"
        }

    # 3. Detecção de outros provedores de nuvem e contêineres
    provider = None
    if os.environ.get("SPACE_ID"):
        provider = "Hugging Face Spaces"
    elif os.environ.get("DYNO"):
        provider = "Heroku Cloud"
    elif os.environ.get("KUBERNETES_SERVICE_HOST"):
        provider = "Kubernetes Cluster"
    elif os.path.exists("/.dockerenv"):
        provider = "Docker Container"
    elif os.environ.get("WEBSITE_SITE_NAME"):
        provider = "Azure App Service"
    elif os.environ.get("AWS_EXECUTION_ENV"):
        provider = "AWS Cloud"
    elif os.environ.get("GAE_APPLICATION"):
        provider = "Google App Engine"
    elif sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY") and not os.path.exists("/dev/video0"):
        provider = "Servidor Linux / Nuvem Headless"

    if provider is not None:
        return {
            "is_cloud": True,
            "deployment_mode": "cloud",
            "provider_label": provider,
            "description": f"Executando em {provider}. Câmeras locais do cliente devem ser capturadas via WebRTC pelo navegador.",
            "recommended_source": "webrtc"
        }

    # 4. Caso padrão: Ambiente Local (Windows, macOS ou Linux Desktop com display/dispositivos)
    return {
        "is_cloud": False,
        "deployment_mode": "local",
        "provider_label": "Local (Desktop / Dojo)",
        "description": "Executando em máquina local. Acesso direto a Webcams USB via OpenCV e aceleração nativa disponíveis.",
        "recommended_source": "webcam"
    }

