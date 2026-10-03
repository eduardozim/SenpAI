"""
Módulo de Download e Extração de Vídeos de Streaming e YouTube para o SenpAI.
Permite carregar vídeos de lutas de Kendo a partir de links da web (YouTube, Vimeo, etc.),
extrair metadados e preparar o arquivo local para processamento no OpenCV e visualização no Streamlit.
"""

import os
import re
import sys
import shutil
import time
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import cv2

if TYPE_CHECKING:
    import yt_dlp
    import yt_dlp.utils
    from yt_dlp.utils import DownloadError as YtDlpDownloadError
else:
    try:
        import yt_dlp
        import yt_dlp.utils
        from yt_dlp.utils import DownloadError as YtDlpDownloadError
        _YT_DLP_AVAILABLE = True
    except (ImportError, AttributeError):
        yt_dlp = None
        _YT_DLP_AVAILABLE = False

        class YtDlpDownloadError(Exception):
            """Fallback para DownloadError quando yt-dlp não está disponível."""
            pass


from src.utils.logger_manager import log_event


class VideoDownloadError(Exception):
    """Exceção personalizada para falhas no download ou extração de stream de vídeo."""
    pass



def _enrich_with_actual_file_info(file_path: str, info: Dict[str, Any], quality_tag: str) -> Dict[str, Any]:
    """
    Inspeciona o arquivo de vídeo baixado ou recuperado do cache local para obter
    as propriedades físicas reais de reprodução (resolução real, FPS real e tamanho em MB).
    """
    info["quality_selected"] = quality_tag
    info["quality_label"] = QUALITY_LABELS.get(quality_tag, "Média (Intermediária / 30 FPS)")
    
    if file_path and os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        info["downloaded_file_size_mb"] = round(size_bytes / (1024 * 1024), 2)
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if w > 0 and h > 0:
                    info["downloaded_resolution"] = f"{w}x{h}"
                if fps > 0:
                    info["downloaded_fps"] = round(fps, 1)
                cap.release()
        except Exception:
            pass
            
    if "downloaded_resolution" not in info:
        info["downloaded_resolution"] = info.get("resolution", "HD")
    if "downloaded_fps" not in info:
        info["downloaded_fps"] = info.get("fps", 30.0)
        
    return info


def validate_video_url(url: Any) -> bool:
    """
    Valida se a string informada é uma URL suportada de vídeo (YouTube ou streaming).
    
    Formatos aceitos:
    - YouTube padrão: https://www.youtube.com/watch?v=VIDEO_ID
    - YouTube encurtado: https://youtu.be/VIDEO_ID
    - YouTube Shorts: https://www.youtube.com/shorts/VIDEO_ID
    - YouTube Live: https://www.youtube.com/live/VIDEO_ID
    - YouTube Embed: https://www.youtube.com/embed/VIDEO_ID
    - Links diretos com protocolo http/https para arquivos de vídeo ou streaming
    """
    if not url or not isinstance(url, str):
        return False
    
    url_clean = url.strip()
    if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        return False
    
    # Padrões comuns do YouTube
    yt_patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/live\/([a-zA-Z0-9_-]+)",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)",
    ]
    for pat in yt_patterns:
        if re.search(pat, url_clean, re.IGNORECASE):
            return True
            
    # Links diretos para arquivos de vídeo
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
    url_lower = url_clean.lower().split("?")[0]
    if any(url_lower.endswith(ext) for ext in video_extensions):
        return True
        
    # Aceita qualquer URL http/https genérica com domínio válido (yt-dlp suporta 1000+ sites)
    generic_url_pattern = r"^https?:\/\/[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:\/.*)?$"
    return bool(re.match(generic_url_pattern, url_clean))


def format_video_duration(seconds: Optional[float]) -> str:
    """Formata a duração em segundos para o formato 'MM:SS' ou 'HH:MM:SS'."""
    if seconds is None or seconds <= 0:
        return "00:00"
    
    total_sec = round(seconds)
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def sanitize_filename(name: str, max_length: int = 40) -> str:
    """Remove caracteres inválidos para sistemas de arquivos e limita o tamanho."""
    if not name:
        return "video"
    # Remove caracteres proibidos no Windows / Linux
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r'[\s_]+', "_", clean).strip("_")
    return clean[:max_length] if clean else "video"


def format_netscape_cookie_content(raw_text: str) -> str:
    """
    Formata e normaliza o conteúdo de cookies para o formato Netscape HTTP Cookie File estrito.
    Garante o cabeçalho '# Netscape HTTP Cookie File' e que os 7 campos estejam separados por tabulações (\t).
    Converte automaticamente espaços acidentais para tabulações e remove comentários/linhas vazias corrompidas.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""
    
    lines = raw_text.strip().splitlines()
    formatted_lines = [
        "# Netscape HTTP Cookie File",
        "# http://curl.haxx.se/rfc/cookie_spec.html",
        "# This file was generated and normalized by SenpAI",
        ""
    ]

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Se contiver tabulações nativas, normaliza os 7 campos
        if "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 7:
                # Junta garantindo 7 campos padrão
                formatted_lines.append("\t".join(parts[:6] + ["\t".join(parts[6:])]))
                continue

        # Se não tiver tabulações (ex: espaços colados por editores web), divide pelos 6 primeiros espaços
        parts = line.split(None, 6)
        if len(parts) == 7:
            formatted_lines.append("\t".join(parts))
        elif len(parts) == 6:
            parts.append("")
            formatted_lines.append("\t".join(parts))

    return "\n".join(formatted_lines) + "\n"


def get_cookie_file_path(custom_file: Optional[str] = None) -> Optional[str]:
    """
    Retorna o caminho de um arquivo de cookies do YouTube se disponível silenciosamente no ambiente.
    Nunca bloqueia nem obriga o usuário a fornecer cookies.
    """
    if custom_file and os.path.exists(custom_file) and os.path.getsize(custom_file) > 10:
        return custom_file

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for cand in [
        os.path.join(base_dir, "cookies.txt"),
        os.path.join(base_dir, "config", "cookies.txt"),
        os.path.join(base_dir, "data", "cookies.txt"),
    ]:
        if os.path.exists(cand) and os.path.getsize(cand) > 10:
            return cand

    try:
        import streamlit as st
        if hasattr(st, "runtime") and st.runtime.exists():
            cookie_session = st.session_state.get("youtube_cookies_text", "")
            if cookie_session and len(str(cookie_session).strip()) > 10:
                tmp_cookie = os.path.join(tempfile.gettempdir(), "senpai_yt_session_cookies.txt")
                norm_cookies = format_netscape_cookie_content(str(cookie_session))
                with open(tmp_cookie, "w", encoding="utf-8") as f:
                    f.write(norm_cookies)
                return tmp_cookie
    except Exception:
        pass

    cookie_content = os.environ.get("YOUTUBE_COOKIES", "")
    if not cookie_content:
        try:
            import streamlit as st
            cookie_content = st.secrets.get("YOUTUBE_COOKIES", "") or st.secrets.get("cookies", "")
        except Exception:
            pass

    if cookie_content and len(str(cookie_content).strip()) > 10:
        tmp_cookie = os.path.join(tempfile.gettempdir(), "senpai_yt_cookies.txt")
        try:
            norm_cookies = format_netscape_cookie_content(str(cookie_content))
            with open(tmp_cookie, "w", encoding="utf-8") as f:
                f.write(norm_cookies)
            return tmp_cookie
        except Exception:
            pass

    return None


def get_ffmpeg_executable_path() -> Optional[str]:
    """
    Localiza o executável do FFmpeg no sistema ou no ambiente virtual:
    1. PATH do sistema (shutil.which)
    2. Pacote imageio_ffmpeg (binário estático integrado)
    3. Diretórios de Python / Conda (sys.prefix, Library/bin, Scripts)
    4. Diretórios padrões do sistema Windows (C:/ffmpeg, Program Files, Chocolatey, etc.)
    5. Diretório bin/ local do projeto SenpAI

    Se localizado, adiciona o diretório ao os.environ["PATH"] e retorna o caminho absoluto.
    """
    # 1. PATH do sistema
    exe = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if exe and os.path.isfile(exe):
        return os.path.abspath(exe)

    # 2. imageio_ffmpeg (fornece binário estático no Windows/Linux/macOS sem exigir instalação do sistema)
    try:
        import imageio_ffmpeg
        img_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if img_exe and os.path.isfile(img_exe):
            ffmpeg_dir = os.path.dirname(img_exe)
            if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return os.path.abspath(img_exe)
    except Exception:
        pass

    # 3. Diretórios de Python / Conda
    python_candidates = [
        os.path.join(sys.prefix, "Library", "bin", "ffmpeg.exe"),
        os.path.join(sys.prefix, "Scripts", "ffmpeg.exe"),
        os.path.join(sys.prefix, "bin", "ffmpeg"),
        os.path.join(sys.prefix, "bin", "ffmpeg.exe"),
        os.path.join(os.path.dirname(sys.executable), "Library", "bin", "ffmpeg.exe"),
        os.path.join(os.path.dirname(sys.executable), "Scripts", "ffmpeg.exe"),
        os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
    ]

    home = os.path.expanduser("~")
    user_candidates = [
        os.path.join(home, "anaconda3", "Library", "bin", "ffmpeg.exe"),
        os.path.join(home, "miniconda3", "Library", "bin", "ffmpeg.exe"),
        os.path.join(home, "AppData", "Local", "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
        "C:\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe",
        "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "ffmpeg.exe"))
    ]

    for candidate in python_candidates + user_candidates:
        if os.path.isfile(candidate):
            ffmpeg_dir = os.path.dirname(candidate)
            if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return os.path.abspath(candidate)

    return None


def get_base_ydl_opts(
    timeout: int = 20,
    client_list: Optional[list] = None,
    cookie_file: Optional[str] = None
) -> Dict[str, Any]:
    """Retorna opções base para yt-dlp otimizadas para extração automática sem bloqueios."""
    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": timeout,
        "nocheckcertificate": True,
        "geo_bypass": True,
        "hls_prefer_native": True,
        "js_runtimes": {"node": {}, "deno": {}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        }
    }
    
    if client_list:
        clients = client_list
    else:
        # Clientes visionos + android são os mais resilientes, contornam SABR streaming e bloqueio de IP/PO token
        clients = ["visionos", "android"]

    opts["extractor_args"] = {
        "youtube": {
            "player_client": clients,
            "skip": ["translated_subs"],
        }
    }

    # Clientes móveis/XR como visionos, android e ios NÃO suportam cookies no yt-dlp (SUPPORTS_COOKIES = False).
    # Passar cookiefile para eles faz o yt-dlp emitir 'Skipping client since it does not support cookies',
    # descartá-los e cair no cliente 'web', que falha com streaming SABR ("Only images are available") e exige PO token.
    # Portanto, só anexamos cookiefile se algum cliente da lista suportar cookies (ex: web, mweb, tv)!
    supports_cookies = any(c in ("web", "mweb", "web_safari", "web_embedded", "web_creator", "tv") for c in clients)
    if supports_cookies:
        resolved_cookie = get_cookie_file_path(cookie_file)
        if resolved_cookie:
            opts["cookiefile"] = resolved_cookie

    ffmpeg_exe = get_ffmpeg_executable_path()
    if ffmpeg_exe:
        opts["ffmpeg_location"] = ffmpeg_exe

    return opts


def extract_video_info(url: str, timeout: int = 15, cookie_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Extrai metadados do vídeo de forma automática e resiliente contra restrições de IP de datacenter.
    
    Retorna um dicionário contendo:
    - id: ID do vídeo
    - title: Título do vídeo
    - duration_seconds: Duração em segundos (float)
    - duration_formatted: Duração formatada (ex: '03:45')
    - uploader: Canal / Criador do vídeo
    - thumbnail: URL da miniatura
    - resolution: Resolução estimada (ex: '1280x720')
    - fps: FPS estimado (float)
    - webpage_url: URL limpa da página
    - is_live: Indica se é live não finalizada
    """
    if not validate_video_url(url):
        raise VideoDownloadError("URL de vídeo inválida ou em formato não reconhecido.")
    
    if yt_dlp is None:
        raise VideoDownloadError("Módulo yt-dlp não está instalado ou disponível no ambiente.")
    
    # Clientes resilientes contra SABR streaming e exigências de PO Token
    client_strategies = [
        ["visionos", "android"],
        ["android", "visionos"],
        ["visionos"],
        ["android"],
        ["default"],
    ]

    last_error_msg = ""

    for clients in client_strategies:
        ydl_opts = get_base_ydl_opts(timeout=timeout, client_list=clients, cookie_file=cookie_file)
        ydl_opts.update({
            "skip_download": True,
            "extract_flat": False,
            "format": "all",
        })
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url.strip(), download=False)
                if not info:
                    continue
                
                if "entries" in info and info["entries"]:
                    info = info["entries"][0]
                    
                duration = float(info.get("duration") or 0.0)
                width = info.get("width") or 0
                height = info.get("height") or 0
                fps = float(info.get("fps") or 30.0)
                
                resolution = f"{width}x{height}" if width and height else "HD"
                
                return {
                    "id": info.get("id", "video"),
                    "title": info.get("title", "Vídeo de Kendo"),
                    "duration_seconds": duration,
                    "duration_formatted": format_video_duration(duration),
                    "uploader": info.get("uploader", info.get("channel", "Canal do YouTube")),
                    "thumbnail": info.get("thumbnail", ""),
                    "resolution": resolution,
                    "fps": fps,
                    "webpage_url": info.get("webpage_url", url.strip()),
                    "is_live": bool(info.get("is_live", False)),
                }
        except YtDlpDownloadError as e:
            last_error_msg = str(e)
            if "Private video" in last_error_msg:
                raise VideoDownloadError("Este vídeo é privado e não pode ser acessado.")
            elif "Video unavailable" in last_error_msg:
                raise VideoDownloadError("Vídeo indisponível ou excluído no YouTube.")
            continue
        except Exception as e:
            last_error_msg = str(e)
            continue

    if any(token in last_error_msg.lower() for token in [
        "requested format is not available", "only images are available",
        "sign in to confirm", "bot", "403", "forbidden"
    ]):
        raise VideoDownloadError(
            "Não foi possível obter informações deste vídeo através do servidor em nuvem. "
            "Você pode carregar o arquivo diretamente pela aba '📁 Upload de Arquivo Local'."
        )
    raise VideoDownloadError(f"Erro ao obter informações do vídeo: {last_error_msg}")


QUALITY_LABELS = {
    "media": "Média (Resolução intermediária, 30 FPS)",
    "alta": "Alta (Maior resolução disponível)",
    "baixa": "Baixa (Menor resolução disponível / Rápido)"
}


def get_format_selector(quality: str = "media", has_ffmpeg: bool = True) -> str:
    """
    Retorna o seletor de formato do yt-dlp de acordo com a qualidade desejada:
    - 'alta': Máxima qualidade de resolução e FPS disponível.
    - 'media' (padrão): Resolução intermediária (até 720p).
    - 'baixa': Menor qualidade disponível (menor tamanho e download rápido).
    Prioriza protocolos HLS (m3u8) que contornam bloqueios de CDN (HTTP 403) em ambientes em nuvem/datacenters.
    Se has_ffmpeg for False, seleciona formatos progressivos sem necessitar de mesclagem (merging) externa via ffmpeg.
    """
    q = quality.lower().strip() if quality else "media"
    if not has_ffmpeg:
        # Formatos únicos/progressivos que já contêm vídeo+áudio juntos (evita erro de ausência de ffmpeg)
        if q in ["alta", "high"]:
            return (
                "best[protocol^=m3u8]/best[ext=mp4]/best/22/18"
            )
        elif q in ["baixa", "low"]:
            return (
                "worst[protocol^=m3u8]/worst[ext=mp4]/worst/18"
            )
        else:
            return (
                "best[protocol^=m3u8][height<=720]/"
                "best[height<=720][ext=mp4]/"
                "best[height<=720]/22/18/best"
            )

    if q in ["alta", "high"]:
        return (
            "bestvideo[protocol^=m3u8]+bestaudio[protocol^=m3u8]/"
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
            "bestvideo+bestaudio/"
            "best[protocol^=m3u8]/best[ext=mp4]/best"
        )
    elif q in ["baixa", "low"]:
        return (
            "worstvideo[protocol^=m3u8]+worstaudio[protocol^=m3u8]/"
            "worstvideo[ext=mp4]+worstaudio[ext=m4a]/"
            "worst[protocol^=m3u8]/worst[ext=mp4]/worst"
        )
    else:  # "media" padrão
        return (
            "bestvideo[protocol^=m3u8][height<=720]+bestaudio[protocol^=m3u8]/"
            "bestvideo[height<=720]+bestaudio/"
            "best[protocol^=m3u8][height<=720]/"
            "best[height<=720]/best"
        )


def download_video_stream(
    url: str,
    output_dir: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    quality: str = "media",
    max_duration_seconds: int = 1800, # 30 minutos máx para proteção
    cookie_file: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Faz o download do vídeo de streaming / YouTube no formato MP4 otimizado para OpenCV,
    executando fallback automático de clientes InnerTube (visionos + android) e formatos sem necessidade de cookies manuais.
    
    Parâmetros:
    - url: Link do YouTube ou streaming de vídeo.
    - output_dir: Diretório para armazenamento do vídeo baixado.
    - progress_callback: Função para atualização de progresso na interface.
    - quality: Nível de qualidade desejado ('baixa', 'media', 'alta'). Padrão: 'media'.
    - max_duration_seconds: Duração máxima permitida para o vídeo.
    - cookie_file: Caminho opcional para arquivo de cookies (caso configurado).
    
    Retorna:
    - target_file_path (str): Caminho local do arquivo .mp4 salvo
    - info_dict (dict): Metadados completos do vídeo com indicação de qualidade
    """
    if not validate_video_url(url):
        raise VideoDownloadError("URL fornecida é inválida.")

    if yt_dlp is None:
        raise VideoDownloadError("Módulo yt-dlp não está instalado ou disponível no ambiente.")

    if output_dir is None:
        output_dir = os.path.join(tempfile.gettempdir(), "senpai_uploads")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extração prévia de metadados
    info = extract_video_info(url, cookie_file=cookie_file)
    
    if info.get("is_live", False):
        raise VideoDownloadError("Transmissões ao vivo não finalizadas não são suportadas para análise pré-gravada.")
        
    duration = info.get("duration_seconds", 0.0)
    if duration > max_duration_seconds:
        raise VideoDownloadError(
            f"O vídeo possui {format_video_duration(duration)}, excedendo o limite máximo recomendado de {format_video_duration(max_duration_seconds)} para análise gravada."
        )

    # Normaliza a chave de qualidade
    quality_tag = quality.lower().strip() if quality else "media"
    if quality_tag not in QUALITY_LABELS:
        quality_tag = "media"
    info["quality_selected"] = quality_tag
    info["quality_label"] = QUALITY_LABELS.get(quality_tag, "Média (Intermediária / 30 FPS)")

    # 2. Verificação de Cache
    video_id = info.get("id", "yt_video")
    safe_title = sanitize_filename(info.get("title", "kendo_match"))
    cached_filename = f"yt_{video_id}_{quality_tag}_{safe_title}.mp4"
    cached_file_path = os.path.join(output_dir, cached_filename)
    
    # Se já existir e tiver tamanho válido (> 4 KB), reutiliza imediatamente
    if os.path.exists(cached_file_path) and os.path.getsize(cached_file_path) > 4096:
        log_event("INFO", f"Vídeo do YouTube ({quality_tag}) reutilizado do cache local: {cached_file_path}", "video_downloader")
        if progress_callback:
            progress_callback(1.0, f"Vídeo carregado do cache local ({info['quality_label']}).")
        info = _enrich_with_actual_file_info(cached_file_path, info, quality_tag)
        return cached_file_path, info

    # Hook interno de progresso para repassar ao Streamlit ou UI
    def _yt_progress_hook(d: Dict[str, Any]):
        if not progress_callback:
            return
        status = d.get("status")
        if status == "downloading":
            total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes") or 0
            pct = (downloaded / total_bytes) if total_bytes > 0 else 0.5
            speed_str = d.get("_speed_str", "")
            eta_str = d.get("_eta_str", "")
            msg = f"Baixando ({quality_tag}): {pct*100:.1f}% ({speed_str} - ETA: {eta_str})"
            progress_callback(min(0.95, pct), msg)
        elif status == "finished":
            progress_callback(0.98, "Finalizando processamento do arquivo de vídeo...")

    # 3. Configurações de Download com seletor de formato por qualidade e estratégias automáticas
    outtmpl_pattern = os.path.join(output_dir, f"yt_{video_id}_{quality_tag}_{safe_title}.%(ext)s")
    ffmpeg_exe = get_ffmpeg_executable_path()
    has_ffmpeg = bool(ffmpeg_exe)
    format_choice = get_format_selector(quality_tag, has_ffmpeg=has_ffmpeg)

    # Estratégias automáticas de clientes para contornar qualquer bloqueio de IP, SABR streaming e PO token
    client_strategies = [
        ["visionos", "android"],
        ["android", "visionos"],
        ["visionos"],
        ["android"],
        ["web", "android"],
        ["default"],
    ]

    download_success = False
    last_error_msg = ""

    for attempt_idx, clients in enumerate(client_strategies):
        if has_ffmpeg:
            format_candidates = [
                format_choice,
                "bestvideo[protocol^=m3u8]+bestaudio[protocol^=m3u8]/best[protocol^=m3u8]/best",
                "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "best[protocol^=m3u8]/best[ext=mp4]/best",
                "18/22/best",
                "worst/worstvideo+worstaudio/worst"
            ]
        else:
            format_candidates = [
                format_choice,
                "best[protocol^=m3u8][height<=720]/best[height<=720]/best[ext=mp4]/best",
                "best[ext=mp4]/best",
                "22/18/best",
                "worst[protocol^=m3u8]/worst[ext=mp4]/worst"
            ]

        # Remove duplicados preservando a ordem
        seen = set()
        unique_fmts = []
        for f in format_candidates:
            if f not in seen:
                seen.add(f)
                unique_fmts.append(f)

        for fmt_try in unique_fmts:
            ydl_opts = get_base_ydl_opts(timeout=30, client_list=clients, cookie_file=cookie_file)
            ydl_opts.update({
                "format": fmt_try,
                "outtmpl": outtmpl_pattern,
                "progress_hooks": [_yt_progress_hook],
            })
            if has_ffmpeg and ffmpeg_exe:
                ydl_opts["ffmpeg_location"] = ffmpeg_exe
                ydl_opts["merge_output_format"] = "mp4"

            try:
                log_event(
                    "INFO",
                    f"Tentando download automático do YouTube ({quality_tag}, clientes: {clients}, formato: {fmt_try}, ffmpeg: {bool(ffmpeg_exe)}): {url}",
                    "video_downloader"
                )
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url.strip()])
                download_success = True
                break
            except YtDlpDownloadError as e:
                err_str = str(e)
                last_error_msg = err_str
                log_event(
                    "WARNING",
                    f"Tentativa com clientes {clients} e formato {fmt_try} falhou: {err_str}",
                    "video_downloader"
                )
                if "ffmpeg is not installed" in err_str.lower():
                    # Se falhar especificamente por falta de ffmpeg, desativa merge e continua com formatos progressivos
                    has_ffmpeg = False
                    ffmpeg_exe = None
                continue
            except Exception as e:
                err_str = str(e)
                last_error_msg = err_str
                if "ffmpeg is not installed" in err_str.lower():
                    has_ffmpeg = False
                    ffmpeg_exe = None
                    continue
                break

        if download_success:
            break

    if not download_success:
        log_event("ERROR", f"Falha definitiva no download de vídeo do YouTube ({url}): {last_error_msg}", "video_downloader")
        if "ffmpeg is not installed" in last_error_msg.lower():
            raise VideoDownloadError(
                "O download exigiu mesclagem de vídeo e áudio separados, mas o FFmpeg não foi encontrado. "
                "Para resolver: instale o pacote 'imageio-ffmpeg' (`pip install imageio-ffmpeg`) ou instale o FFmpeg no sistema."
            )
        if any(token in last_error_msg.lower() for token in [
            "requested format is not available", "only images are available",
            "sign in to confirm", "bot", "403", "forbidden", "empty"
        ]):
            raise VideoDownloadError(
                "O YouTube bloqueia a transferência direta de vídeos por servidores em nuvem (HTTP 403: Forbidden - Bloqueio de IP de Datacenter AWS/GCP). "
                "Para analisar este combate no Streamlit Cloud, faça o download do vídeo em seu computador e envie pela aba ao lado '📁 Upload de Arquivo Local'. "
                "Caso queira baixar vídeos diretamente via links do YouTube sem restrições, execute o SenpAI localmente em seu computador ('streamlit run app.py')."
            )
        raise VideoDownloadError(f"Falha ao baixar vídeo do YouTube: {last_error_msg}")

    # Localiza o arquivo baixado
    actual_file_path = None
    for ext in ["mp4", "mkv", "webm", "avi"]:
        cand = os.path.join(output_dir, f"yt_{video_id}_{quality_tag}_{safe_title}.{ext}")
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            actual_file_path = cand
            break
            
    if not actual_file_path:
        for f in os.listdir(output_dir):
            if f.startswith(f"yt_{video_id}_{quality_tag}") and os.path.getsize(os.path.join(output_dir, f)) > 0:
                actual_file_path = os.path.join(output_dir, f)
                break

    if not actual_file_path or not os.path.exists(actual_file_path):
        raise VideoDownloadError("Download concluído, mas o arquivo de vídeo não foi encontrado no disco.")

    log_event("INFO", f"Download concluído com sucesso ({quality_tag}): {actual_file_path}", "video_downloader")
    if progress_callback:
        progress_callback(1.0, "Download concluído com sucesso!")
        
    info = _enrich_with_actual_file_info(actual_file_path, info, quality_tag)
    return actual_file_path, info
