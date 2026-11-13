"""
Chaves de API da coleta na internet, configuradas pela tela (ficam só neste computador).

Arquivo: config/api_keys.local.json (fora do git). Uma variável de ambiente com o mesmo nome tem
prioridade sobre o arquivo. As chaves nunca entram no pacote de treinamento exportado.
"""

import os
import json
from typing import Dict, Optional

from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir

KEYS_FILE = os.path.join("config", "api_keys.local.json")

# nome → (rótulo, onde obter)
KNOWN_KEYS: Dict[str, tuple] = {
    "GEMINI_API_KEY": ("Chave do Gemini (IA da coleta)",
                       "Google AI Studio → https://aistudio.google.com/apikey → “Create API key”. Tem cota gratuita."),
    "GEMINI_MODEL": ("Modelo do Gemini (opcional)",
                     "Deixe em branco para usar gemini-2.5-flash. Lista de modelos: https://ai.google.dev/gemini-api/docs/models"),
    "OPENAI_API_KEY": ("Chave da OpenAI (alternativa ao Gemini)",
                       "https://platform.openai.com/api-keys → “Create new secret key”. Só é usada se não houver chave do Gemini."),
    "SERPAPI_KEY": ("Chave da SerpApi (resultados do Google)",
                    "https://serpapi.com/users/sign_up → depois https://serpapi.com/manage-api-key. Tem plano gratuito com cota mensal."),
}


def _path() -> str:
    return os.path.join(get_test_sandbox_dir(), "api_keys.local.json") if is_test_mode() else KEYS_FILE


def load_file_keys() -> Dict[str, str]:
    p = _path()
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: str(v) for k, v in data.items() if k in KNOWN_KEYS and v}
    except Exception:
        return {}


def save_keys(values: Dict[str, Optional[str]]) -> None:
    """Grava as chaves informadas; valor vazio mantém a atual, None apaga."""
    current = load_file_keys()
    for k, v in values.items():
        if k not in KNOWN_KEYS:
            continue
        if v is None:
            current.pop(k, None)
        elif str(v).strip():
            current[k] = str(v).strip()
    p = _path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    apply_to_env(force=True)


def get_key(name: str) -> Optional[str]:
    return os.environ.get(name) or load_file_keys().get(name)


def apply_to_env(force: bool = False) -> None:
    """Disponibiliza as chaves do arquivo para os módulos que leem variáveis de ambiente."""
    for k, v in load_file_keys().items():
        if force or not os.environ.get(k):
            os.environ[k] = v


def key_source(name: str) -> Optional[str]:
    """'arquivo', 'ambiente' ou None (sem mostrar o valor)."""
    if name in load_file_keys():
        return "arquivo"
    if os.environ.get(name):
        return "ambiente"
    return None
