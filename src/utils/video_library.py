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


# ------------------------------------------------------------------------------
# Cortes das análises de lutas: só os golpes em dúvida (~10 s cada), para treino
# ------------------------------------------------------------------------------

ANALYSIS_PREFIX = "analise_"
ANALYSIS_CLIP_HALF_SEC = 5.0       # 5 s antes e 5 s depois do golpe (as bandeiras sobem ~1 s depois)
ANALYSIS_MIN_GAP_SEC = 10.0        # não repete golpes do mesmo vídeo a menos de 10 s de um corte já guardado
DATASET_VIDEO_PREFIXES = (INTERNET_PREFIX, ANALYSIS_PREFIX)


def is_analysis_clip(name: str) -> bool:
    return os.path.basename(name).startswith(ANALYSIS_PREFIX)


def is_dataset_video(name: str) -> bool:
    """Vídeo que o próprio sistema trouxe para o treino (coleta na internet ou corte de análise)."""
    return os.path.basename(name).startswith(DATASET_VIDEO_PREFIXES)


def _clip_stem(video_path: str) -> str:
    import re
    from src.engine.label_bank import canonical_video_name
    stem = os.path.splitext(canonical_video_name(video_path))[0]
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_")
    return stem[:60] or "luta"


def _parse_clip_time(name: str, stem: str):
    import re
    m = re.match(re.escape(ANALYSIS_PREFIX + stem) + r"_(\d+(?:\.\d+)?)s\.mp4$", name)
    return float(m.group(1)) if m else None


def save_analysis_clip(video_path: str, time_sec: float, videos_dir=None, extractor=None):
    """
    Guarda em data/videos um corte de ~10 s em volta de um golpe em dúvida de uma luta analisada.
    Não repete: se já houver corte do mesmo vídeo a menos de 10 s, não grava. Devolve o caminho ou None.
    """
    if not video_path or not os.path.isfile(video_path) or time_sec is None:
        return None
    vdir = videos_dir or get_videos_dir()
    if os.path.abspath(os.path.dirname(video_path)) == os.path.abspath(vdir):
        return None   # vídeo que já está na pasta de treino (internet ou outro corte)
    stem = _clip_stem(video_path)
    for n in os.listdir(vdir):
        t0 = _parse_clip_time(n, stem)
        if t0 is not None and abs(t0 - float(time_sec)) < ANALYSIS_MIN_GAP_SEC:
            return None
    out = os.path.join(vdir, f"{ANALYSIS_PREFIX}{stem}_{float(time_sec):.1f}s.mp4")
    if extractor is None:
        from src.utils.strike_clip_manager import extract_strike_subclip as extractor
    try:
        res = extractor(video_path, out, target_time_sec=float(time_sec), window_before_sec=ANALYSIS_CLIP_HALF_SEC,
                        window_after_sec=ANALYSIS_CLIP_HALF_SEC, strike_type="ANALISE")
    except Exception:
        return None
    return res if res and os.path.isfile(res) else None
