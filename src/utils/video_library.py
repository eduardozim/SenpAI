"""
Pasta única de vídeos do SenpAI: Dev/data/videos.

Gabarito de lutas, ciclo de treino, golden, detector de bandeiras e coleta na internet usam esta mesma
pasta. Os vídeos baixados da internet entram nela com o prefixo "internet_" (o catálogo da coleta guarda
a origem de cada um); só esses são removidos pelo botão Apagar Treinamento — os seus vídeos nunca.

O cache de poses fica à parte (data/feature_cache): é identificado pelo conteúdo do vídeo, então vale
mesmo se o vídeo for renomeado ou movido.
"""

import os
from typing import List

from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir

DEFAULT_VIDEOS_DIR = os.path.join("data", "videos")
INTERNET_PREFIX = "internet_"


def get_videos_dir(create: bool = True) -> str:
    """Pasta única de vídeos (em modo de teste, uma pasta da sandbox)."""
    d = os.path.join(get_test_sandbox_dir(), "videos") if is_test_mode() else DEFAULT_VIDEOS_DIR
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def is_internet_video(name: str) -> bool:
    return os.path.basename(name).startswith(INTERNET_PREFIX)


def internet_videos(videos_dir: str) -> List[str]:
    """Caminhos dos vídeos baixados pela coleta na internet."""
    if not os.path.isdir(videos_dir):
        return []
    return sorted(os.path.join(videos_dir, n) for n in os.listdir(videos_dir)
                  if is_internet_video(n) and os.path.isfile(os.path.join(videos_dir, n)))
