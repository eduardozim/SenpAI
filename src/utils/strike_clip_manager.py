"""
Módulo de Gerenciamento de Clipes de Demonstração de Golpes (Strike Clip Manager).
Responsável por:
1. Localizar o arquivo físico de vídeo real da luta gravada.
2. Extrair o corte real do golpe controverso em torno do impacto da luta original.
3. Excluir o arquivo de vídeo do clipe com segurança após a homologação e retreinamento.
"""

import os
import re
import cv2
import tempfile
import numpy as np
from typing import Dict, Any, Optional

from src.utils.hardware import ensure_browser_compatible_video
from src.utils.logger_manager import log_event


DEFAULT_CLIPS_DIR = "data/active_learning_clips"
DEFAULT_UPLOADS_DIR = "data/tmp/senpai_uploads"


def get_senpai_uploads_dir(ensure_exists: bool = True) -> str:
    """
    Retorna o caminho do diretório padrão para armazenamento de uploads e downloads de vídeos do SenpAI.
    Prioriza uma pasta dentro do projeto ('data/tmp/senpai_uploads'), garantindo:
      1. Permissões de escrita estáveis dentro do workspace da aplicação.
      2. Persistência dos vídeos de combate entre sessões para curadoria ativa (evitando apagamento pelo SO).
      3. Prevenção de estouro de memória RAM por tmpfs no Linux.
    Possui fallback automático para a pasta temporária do sistema operacional caso a pasta do projeto
    seja somente leitura (ex: contêiner imutável).
    Permite customização via variável de ambiente 'SENPAI_UPLOADS_DIR'.
    """
    env_dir = os.environ.get("SENPAI_UPLOADS_DIR")
    candidates = [
        env_dir,
        DEFAULT_UPLOADS_DIR,
        "data/uploads",
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            if ensure_exists:
                os.makedirs(cand, exist_ok=True)
                test_file = os.path.join(cand, f".write_test_{os.getpid()}")
                with open(test_file, "w") as f:
                    f.write("test")
                if os.path.exists(test_file):
                    os.remove(test_file)
            return cand
        except Exception:
            continue

    fallback_dir = os.path.join(tempfile.gettempdir(), "senpai_uploads")
    if ensure_exists:
        try:
            os.makedirs(fallback_dir, exist_ok=True)
        except Exception:
            pass
    return fallback_dir


def get_cross_platform_basename(path_str: str) -> str:
    """
    Extrai o nome do arquivo preservando compatibilidade entre Windows e Linux
    (suporta caminhos contendo tanto barras normais '/' quanto invertidas '\\').
    """
    if not path_str or not isinstance(path_str, str):
        return ""
    clean = path_str.strip().replace("\\", "/")
    return os.path.basename(clean).strip()


def resolve_source_video_path(video_source: str) -> str:
    """
    Normaliza o caminho do vídeo de origem para o ambiente/servidor atual.
    Se o vídeo físico já existir em disco, retorna o caminho correspondente no host atual.
    Se não existir, ajusta caminhos de senpai_uploads ou pastas do projeto para a
    localização esperada no servidor/máquina atual (evitando referências fixas a pastas de outro computador).
    """
    if not video_source or not str(video_source).strip():
        return ""

    found = find_source_video(video_source)
    if found:
        return found

    # Se não foi encontrado no disco, adaptar a indicação para a estrutura do host atual
    base_name = get_cross_platform_basename(video_source)
    video_source_clean = str(video_source).replace("\\", "/")

    if "senpai_uploads" in video_source_clean or base_name.startswith("yt_") or base_name.startswith("upload_"):
        target_dir = get_senpai_uploads_dir(ensure_exists=False)
        return os.path.join(target_dir, base_name)
    elif "data/uploads" in video_source_clean:
        return os.path.join("data", "uploads", base_name)
    elif "data/videos" in video_source_clean:
        return os.path.join("data", "videos", base_name)
    elif "data/" in video_source_clean:
        return os.path.join("data", base_name)
    else:
        return base_name


def find_source_video(video_source: str) -> Optional[str]:
    """
    Localiza o arquivo físico de vídeo real do combate no disco.
    Varre o caminho direto, o diretório de uploads/downloads do SenpAI,
    o diretório raiz e pastas de dados.
    Retorna o caminho do vídeo correspondente ou None caso o arquivo de origem não exista.
    """
    if not video_source or not str(video_source).strip():
        return None

    video_source_clean = str(video_source).strip()

    # 1. Caminho direto se for válido e existir no SO atual
    if os.path.isfile(video_source_clean) and os.path.getsize(video_source_clean) > 0:
        return os.path.abspath(video_source_clean)

    base_name = get_cross_platform_basename(video_source_clean)
    if not base_name:
        return None
    clean_stem = re.sub(r"^(upload_\d+_|yt_[^_]+_)", "", os.path.splitext(base_name)[0]).strip().lower()

    # 2. Diretórios potenciais onde o vídeo original da luta pode residir no host atual
    active_uploads = get_senpai_uploads_dir(ensure_exists=False)
    search_dirs = [
        active_uploads,
        DEFAULT_UPLOADS_DIR,
        "data/uploads",
        os.path.join(tempfile.gettempdir(), "senpai_uploads"),
        "data/videos",
        "data",
        "."
    ]

    for s_dir in search_dirs:
        if not os.path.isdir(s_dir):
            continue

        # 2a. Correspondência direta por nome base
        cand = os.path.join(s_dir, base_name)
        if os.path.isfile(cand) and os.path.getsize(cand) > 0:
            return os.path.abspath(cand)

        # 2b. Correspondência por nome limpo descartando prefixos de upload
        try:
            for f in os.listdir(s_dir):
                if not f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
                    continue
                if f.endswith("_demo.mp4") or f.endswith("_clip.mp4") or f.endswith("_corte.mp4"):
                    continue

                f_path = os.path.join(s_dir, f)
                if not (os.path.isfile(f_path) and os.path.getsize(f_path) > 0):
                    continue

                f_clean_stem = re.sub(r"^(upload_\d+_|yt_[^_]+_)", "", os.path.splitext(f)[0]).strip().lower()

                if clean_stem and (clean_stem == f_clean_stem or clean_stem in f_clean_stem or f_clean_stem in clean_stem):
                    return os.path.abspath(f_path)
        except Exception:
            continue

    # Não faz fallback arbitrário para outro vídeo: integridade estrita entre descrição e vídeo
    return None


def parse_strike_timestamp(item: Dict[str, Any]) -> float:
    """
    Extrai com precisão o momento do golpe na luta (em segundos).
    Descarta strings de data ISO de sistema e prioriza timecodes reais de combate.
    """
    if not item:
        return 0.0

    # 1. Campo explícito em segundos na raiz do item
    if "timestamp_seconds" in item and item["timestamp_seconds"] is not None:
        try:
            val = float(item["timestamp_seconds"])
            if val >= 0:
                return val
        except Exception:
            pass

    # 2. Dados do golpe (strike_data)
    strike_data = item.get("strike_data") or {}
    if "timestamp_seconds" in strike_data and strike_data["timestamp_seconds"] is not None:
        try:
            val = float(strike_data["timestamp_seconds"])
            if val >= 0:
                return val
        except Exception:
            pass

    # 3. Timecode em impact_frame / fps
    if "impact_frame" in strike_data:
        try:
            imp_f = float(strike_data["impact_frame"])
            fps = float(strike_data.get("fps") or 30.0)
            if imp_f > 0 and fps > 0:
                return imp_f / fps
        except Exception:
            pass

    # 4. Campo time numérico
    if "time" in strike_data:
        try:
            return float(strike_data["time"])
        except Exception:
            pass

    # 5. Parsing de string timestamp em strike_data ou item
    for ts_val in [strike_data.get("timestamp"), item.get("timestamp")]:
        if not ts_val:
            continue
        ts_str = str(ts_val).strip()

        # Ignorar datas ISO de registro do sistema (ex: 2026-10-01T22:05:45...)
        if "T" in ts_str or (ts_str.count("-") >= 2 and len(ts_str) >= 10):
            continue

        ts_str_clean = ts_str.replace("s", "").replace("S", "").strip()
        if ":" in ts_str_clean:
            try:
                parts = ts_str_clean.split(":")
                if len(parts) == 2:
                    return float(parts[0]) * 60.0 + float(parts[1])
                elif len(parts) == 3:
                    return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
            except Exception:
                pass
        else:
            try:
                val = float(ts_str_clean)
                if val >= 0:
                    return val
            except Exception:
                pass

    return 0.0


def format_strike_timestamp(seconds: float) -> str:
    """Formata o tempo do golpe em MM:SS.mmm (XX.Xs)."""
    if seconds is None or seconds < 0:
        return "00:00.000 (0.0s)"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f} ({seconds:.1f}s)"


def extract_strike_subclip(
    video_path: str,
    output_path: str,
    target_time_sec: float = 0.0,
    window_before_sec: float = 2.0,
    window_after_sec: float = 2.5,
    strike_type: str = "GOLPE"
) -> Optional[str]:
    """
    Extrai o corte real do vídeo da luta em torno do timestamp do golpe.
    """
    if not os.path.isfile(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Se target_time for 0.0 mas tiver duração, posicionar no primeiro terço ou metade
    if target_time_sec <= 0.0 and total_frames > 0:
        duration_s = total_frames / fps
        target_time_sec = min(3.0, duration_s / 2.0)

    start_frame = max(0, int((target_time_sec - window_before_sec) * fps))
    end_frame = min(total_frames, int((target_time_sec + window_after_sec) * fps))
    if end_frame <= start_frame:
        end_frame = min(total_frames, start_frame + int(4.5 * fps))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_f = start_frame

    while current_f < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Inserir banner discreto com identificação do golpe e timecode da luta original
        overlay_txt = f"SenpAI - {strike_type.upper()} ({current_f/fps:.2f}s)"
        cv2.putText(
            frame,
            overlay_txt,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        out.write(frame)
        current_f += 1

    cap.release()
    out.release()

    try:
        ensure_browser_compatible_video(output_path)
    except Exception:
        pass

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    return None


def get_or_create_strike_clip(
    item: Dict[str, Any],
    output_dir: str = DEFAULT_CLIPS_DIR,
    force_recreate: bool = False
) -> Optional[str]:
    """
    Localiza ou extrai o corte real do vídeo da luta original em torno do golpe controverso.
    Se o clipe recortado já existir em disco, reaproveita-o imediatamente sem exigir o vídeo original.
    Retorna None se o clipe ainda não existir e o vídeo original da luta não estiver acessível no disco.
    """
    os.makedirs(output_dir, exist_ok=True)
    item_id = str(item.get("id", "strike_uncertain"))
    expected_file = os.path.join(output_dir, f"{item_id}_corte.mp4")

    # 1. Se não for recriação forçada, verificar se o clipe já existe em disco
    if not force_recreate:
        existing_clip = item.get("clip_path")
        if existing_clip:
            # 1a. Caminho direto se for válido e existir
            if os.path.isfile(existing_clip) and os.path.getsize(existing_clip) > 0:
                if not existing_clip.endswith("_demo.mp4"):
                    return existing_clip

            # 1b. Por nome base no diretório de clipes (suporte a consolidação servidor -> local / caminhos relativos)
            clip_base = get_cross_platform_basename(existing_clip)
            if clip_base:
                cand_output = os.path.join(output_dir, clip_base)
                if os.path.isfile(cand_output) and os.path.getsize(cand_output) > 0:
                    if not cand_output.endswith("_demo.mp4"):
                        norm_cand = cand_output.replace("\\", "/")
                        item["clip_path"] = norm_cand
                        return norm_cand

        # 1c. Pelo padrão esperado {item_id}_corte.mp4
        if os.path.isfile(expected_file) and os.path.getsize(expected_file) > 0:
            norm_exp = expected_file.replace("\\", "/")
            item["clip_path"] = norm_exp
            return norm_exp

    # 2. Se o corte ainda não existe (ou force_recreate=True), localizar o vídeo real da luta original
    video_source = str(item.get("video_source", "")).strip()
    source_file = find_source_video(video_source)

    if not source_file or not os.path.isfile(source_file):
        log_event(
            "INFO",
            f"Vídeo real de origem não encontrado no disco para o item {item_id} (video_source='{video_source}')",
            "strike_clip_manager"
        )
        return None

    # 3. Calcular tempo exato do golpe para recorte a partir da descrição
    target_time = parse_strike_timestamp(item)
    strike_t = str(item.get("strike_type", "GOLPE")).upper()

    # 4. Extrair o corte real da luta
    extracted = extract_strike_subclip(
        source_file,
        expected_file,
        target_time_sec=target_time,
        strike_type=strike_t
    )
    if extracted:
        norm_ext = extracted.replace("\\", "/")
        item["clip_path"] = norm_ext
        log_event("INFO", f"Corte real do golpe extraído da luta com sucesso: {norm_ext} (Fonte: {source_file}, t={target_time:.2f}s)", "strike_clip_manager")
        return norm_ext

    return None


def delete_strike_clip(clip_path: Optional[str]) -> bool:
    """
    Apaga com segurança o arquivo de vídeo do corte do golpe após o retreinamento.
    Suporta caminhos diretos e resolução no diretório padrão de clipes.
    """
    if not clip_path:
        return False

    deleted = False
    candidates = [clip_path]
    base_name = get_cross_platform_basename(clip_path)
    if base_name:
        candidates.append(os.path.join(DEFAULT_CLIPS_DIR, base_name))

    for path in candidates:
        try:
            if path and os.path.isfile(path):
                os.remove(path)
                log_event("INFO", f"Corte do golpe excluído com segurança após retreinamento: {path}", "strike_clip_manager")
                deleted = True
        except Exception as e:
            log_event("WARNING", f"Falha ao apagar arquivo de corte {path}: {e}", "strike_clip_manager")

    return deleted
