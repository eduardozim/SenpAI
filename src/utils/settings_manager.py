"""
Gerenciador de Configurações Globais do ShinpanAI.
Persiste e lê opções de sistema (dispositivo de processamento CPU/GPU, etc.).
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS_PATH = "config/settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "processing_device": "cpu"
}

def load_settings(config_path: str = DEFAULT_SETTINGS_PATH) -> Dict[str, Any]:
    """
    Carrega as configurações globais do arquivo JSON.
    Retorna as configurações padrão se o arquivo não existir ou for inválido.
    """
    if not os.path.exists(config_path):
        return DEFAULT_SETTINGS.copy()
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Garantir chaves padrão
            merged = DEFAULT_SETTINGS.copy()
            merged.update(data)
            return merged
    except Exception as e:
        logger.error(f"Erro ao carregar configurações de '{config_path}': {e}. Usando padrões.")
        return DEFAULT_SETTINGS.copy()

def save_settings(settings: Dict[str, Any], config_path: str = DEFAULT_SETTINGS_PATH) -> None:
    """
    Salva o dicionário de configurações no arquivo JSON especificado.
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Erro ao salvar configurações em '{config_path}': {e}")
        raise e

def get_processing_device(config_path: str = DEFAULT_SETTINGS_PATH) -> str:
    """
    Retorna o dispositivo de processamento configurado ("cpu" ou "gpu").
    """
    settings = load_settings(config_path)
    device = settings.get("processing_device", "cpu").lower().strip()
    return device if device in ["cpu", "gpu"] else "cpu"

def set_processing_device(device: str, config_path: str = DEFAULT_SETTINGS_PATH) -> Dict[str, Any]:
    """
    Atualiza e salva a preferência do dispositivo de processamento ("cpu" ou "gpu").
    """
    clean_device = device.lower().strip() if device else "cpu"
    if clean_device not in ["cpu", "gpu"]:
        clean_device = "cpu"
    
    settings = load_settings(config_path)
    settings["processing_device"] = clean_device
    save_settings(settings, config_path)
    return settings
