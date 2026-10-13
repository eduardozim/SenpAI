"""
Módulo de detecção e gerenciamento de aceleração por hardware (CPU e GPU NVIDIA).
"""

import os
import sys
import glob
import subprocess
import logging
import re
import time
from typing import Dict, Any, Tuple, List, Optional, Callable

logger = logging.getLogger(__name__)

def detect_nvidia_gpu() -> Dict[str, Any]:
    """
    Verifica se o sistema possui GPU aceleradora NVIDIA funcional.
    Realiza checagens multi-nível:
    1. Utilitário de linha de comando `nvidia-smi`
    2. Importação e inspeção de PyTorch (se disponível)
    3. Importação e inspeção de ONNX Runtime (se disponível)
    4. Módulo OpenCV CUDA (se disponível)
    """
    result = {
        "has_nvidia_gpu": False,
        "gpu_name": "Nenhum",
        "gpu_count": 0,
        "driver_version": "N/A",
        "memory_total": "N/A",
        "detection_methods": [],
        "details": "Nenhuma GPU NVIDIA detectada."
    }

    # 1. Checagem via nvidia-smi (muito confiável em sistemas Windows/Linux com drivers NVIDIA)
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=3, text=True).strip()
        if output:
            lines = output.splitlines()
            result["gpu_count"] = len(lines)
            first_gpu = lines[0].split(",")
            if len(first_gpu) >= 1:
                result["gpu_name"] = first_gpu[0].strip()
            if len(first_gpu) >= 2:
                result["driver_version"] = first_gpu[1].strip()
            if len(first_gpu) >= 3:
                result["memory_total"] = first_gpu[2].strip()
            
            result["has_nvidia_gpu"] = True
            result["detection_methods"].append("nvidia-smi")
            result["details"] = f"NVIDIA GPU detectada via nvidia-smi: {result['gpu_name']} ({result['memory_total']}, Driver {result['driver_version']})"
            return result
    except Exception as e:
        logger.debug(f"nvidia-smi não respondeu ou não está presente: {e}")

    # 2. Checagem via PyTorch (se instalado)
    try:
        import torch
        if torch.cuda.is_available():
            result["has_nvidia_gpu"] = True
            result["gpu_count"] = torch.cuda.device_count()
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["detection_methods"].append("torch.cuda")
            result["details"] = f"NVIDIA GPU detectada via PyTorch: {result['gpu_name']} (Dispositivos: {result['gpu_count']})"
            return result
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Erro ao checar PyTorch CUDA: {e}")

    # 3. Checagem via ONNX Runtime (se instalado)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers:
            result["has_nvidia_gpu"] = True
            result["detection_methods"].append("onnxruntime")
            result["gpu_name"] = "NVIDIA CUDA Compatible GPU"
            result["details"] = "GPU NVIDIA compatível com CUDA detectada via ONNX Runtime."
            return result
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Erro ao checar ONNX Runtime CUDA: {e}")

    # 4. Checagem via OpenCV CUDA (se compilado com suporte CUDA)
    try:
        import cv2
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            result["has_nvidia_gpu"] = True
            result["gpu_count"] = cv2.cuda.getCudaEnabledDeviceCount()
            result["detection_methods"].append("cv2.cuda")
            result["gpu_name"] = "NVIDIA CUDA GPU (OpenCV)"
            result["details"] = f"NVIDIA GPU detectada via OpenCV CUDA ({result['gpu_count']} dispositivo(s))."
            return result
    except Exception as e:
        logger.debug(f"Erro ao checar OpenCV CUDA: {e}")

    return result

def check_cuda_framework_support() -> Dict[str, Any]:
    """
    Verifica se existem frameworks de Deep Learning com suporte CUDA ativos no ambiente Python.
    """
    torch_cuda = False
    torch_device_name = ""
    onnx_cuda = False

    ultralytics_ready = False
    try:
        import torch
        from ultralytics import YOLO
        torch_cuda = torch.cuda.is_available()
        if torch_cuda:
            torch_device_name = torch.cuda.get_device_name(0)
            ultralytics_ready = True
    except Exception:
        pass

    # Se ainda não detectou torch_cuda (pode ser cache do módulo importado no processo atual), testar out-of-process
    if not torch_cuda:
        try:
            chk_cmd = [sys.executable, "-c", "import torch; print(f'{torch.cuda.is_available()}|{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\"}')"]
            out = subprocess.check_output(chk_cmd, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
            parts = out.split("|")
            if parts[0] == "True":
                torch_cuda = True
                torch_device_name = parts[1] if len(parts) > 1 else "NVIDIA GPU"
                ultralytics_ready = True
        except Exception:
            pass

    try:
        import onnxruntime as ort
        onnx_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass

    return {
        "torch_cuda": torch_cuda and ultralytics_ready,
        "torch_device_name": torch_device_name,
        "onnx_cuda": onnx_cuda,
        "ultralytics_ready": ultralytics_ready,
        "mediapipe_cpu_only_win": True
    }

def _run_pip_with_progress(
    cmd: List[str],
    progress_callback: Optional[Callable[[float, str, Optional[str]], None]],
    base_progress: float,
    progress_weight: float,
    phase_title: str
) -> Tuple[int, str]:
    """
    Executa comando pip via subprocess.Popen capturando saída em tempo real (incluindo \\r de download)
    para atualizar a barra de progresso e repassar logs detalhados.
    """
    logger.info(f"[Hardware] Executando comando pip: {' '.join(cmd)}")
    full_output: List[str] = []
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace"
        )
    except Exception as e:
        err_msg = f"Falha ao iniciar processo pip: {e}"
        logger.error(err_msg)
        return -1, err_msg

    if proc.stdout is None:
        return -1, "Não foi possível capturar a saída padrão do processo pip."

    pct_pattern = re.compile(r'(\d{1,3}(?:\.\d+)?)%')
    buffer = ""
    last_update_time = 0.0

    while True:
        char = proc.stdout.read(1)
        if not char and proc.poll() is not None:
            break
        if char:
            if char in ('\r', '\n'):
                line = buffer.strip()
                buffer = ""
                if line:
                    full_output.append(line)
                    now = time.time()
                    is_key_event = any(k in line.lower() for k in [
                        "downloading", "collecting", "installing", "successfully", "requirement", "error", "metadata"
                    ])
                    if (now - last_update_time > 0.15) or is_key_event:
                        last_update_time = now
                        calc_pct = base_progress
                        match_pct = pct_pattern.search(line)
                        if match_pct:
                            try:
                                sub_val = float(match_pct.group(1)) / 100.0
                                calc_pct = base_progress + (progress_weight * sub_val * 0.90)
                            except ValueError:
                                pass
                        elif "installing collected packages" in line.lower():
                            calc_pct = base_progress + (progress_weight * 0.92)
                        elif "successfully installed" in line.lower():
                            calc_pct = base_progress + progress_weight

                        calc_pct = max(0.0, min(0.99, calc_pct))
                        if progress_callback:
                            try:
                                progress_callback(calc_pct, phase_title, line)
                            except Exception:
                                pass
            else:
                buffer += char

    if buffer.strip():
        full_output.append(buffer.strip())

    try:
        ret_code = proc.wait(timeout=30)
    except Exception:
        proc.kill()
        ret_code = -1

    return ret_code, "\n".join(full_output)

def install_cuda_packages(
    progress_callback: Optional[Callable[[float, str, Optional[str]], None]] = None
) -> Tuple[bool, str]:
    """
    Executa a instalação dos pacotes PyTorch com suporte CUDA e Ultralytics no ambiente Python atual.
    Testa dinamicamente múltiplos índices de release do PyTorch (cu126, cu124, cu121) para garantir
    compatibilidade com a versão instalada do Python (Python 3.10 a 3.14+).
    Fornece feedback de progresso em tempo real via callback.
    """
    logger.info("[Hardware] Iniciando instalação de dependências CUDA (PyTorch CUDA e Ultralytics)...")
    if progress_callback:
        progress_callback(0.05, "Verificando repositórios CUDA disponíveis...", "Preparando pip...")

    cuda_indices = [
        ("CUDA 12.6", "https://download.pytorch.org/whl/cu126"),
        ("CUDA 12.4", "https://download.pytorch.org/whl/cu124"),
        ("CUDA 12.1", "https://download.pytorch.org/whl/cu121"),
        ("PyPI Padrão", None)
    ]

    torch_installed = False
    last_torch_err = ""

    # Peso alocado para PyTorch (0.10 a 0.75 = 0.65 de peso)
    for idx_name, index_url in cuda_indices:
        phase_msg = f"Instalando PyTorch com aceleração {idx_name}..."
        if progress_callback:
            progress_callback(0.10, phase_msg, f"Conectando ao repositório {index_url or 'PyPI padrão'}...")

        # Utiliza --upgrade --force-reinstall --no-deps para assegurar que a versão CPU existente seja substituída pela compilação CUDA
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--upgrade", "--force-reinstall", "--no-deps",
            "torch", "torchvision"
        ]
        if index_url:
            cmd.extend(["--index-url", index_url])

        ret, out_text = _run_pip_with_progress(
            cmd=cmd,
            progress_callback=progress_callback,
            base_progress=0.10,
            progress_weight=0.65,
            phase_title=phase_msg
        )

        if ret == 0:
            torch_installed = True
            logger.info(f"[Hardware] PyTorch instalado com sucesso ({idx_name}).")
            if progress_callback:
                progress_callback(0.75, f"PyTorch instalado com sucesso ({idx_name})!", "Instalação do PyTorch concluída.")
            break
        else:
            last_torch_err = out_text.strip()
            logger.debug(f"[Hardware] Tentativa com {idx_name} falhou. Tentando próximo repositório...")

    if not torch_installed:
        err_msg = f"Falha ao instalar PyTorch com suporte CUDA: {last_torch_err[-500:]}"
        logger.error(err_msg)
        if progress_callback:
            progress_callback(0.75, "Falha na instalação do PyTorch CUDA.", err_msg)
        return False, err_msg

    # Se estiver no Windows, desbloquear DLLs que possam ter sido bloqueadas por políticas de segurança
    if sys.platform.startswith("win"):
        try:
            torch_lib = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib")
            if os.path.exists(torch_lib):
                if progress_callback:
                    progress_callback(0.76, "Configurando permissões das DLLs no Windows...", "Executando Unblock-File...")
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", f"Get-ChildItem -Path '{torch_lib}' -Filter *.dll -Recurse -ErrorAction SilentlyContinue | Unblock-File"],
                    capture_output=True,
                    timeout=30
                )
        except Exception as e:
            logger.debug(f"Aviso ao desbloquear DLLs: {e}")

    # Instala/Verifica Ultralytics para suporte a modelos YOLOv8-Pose (0.78 a 0.90)
    try:
        phase_yolo = "Instalando e configurando Ultralytics (YOLOv8-Pose)..."
        if progress_callback:
            progress_callback(0.78, phase_yolo, "Verificando pacote ultralytics...")
        cmd_yolo = [sys.executable, "-m", "pip", "install", "ultralytics"]
        ret_yolo, out_yolo = _run_pip_with_progress(
            cmd=cmd_yolo,
            progress_callback=progress_callback,
            base_progress=0.78,
            progress_weight=0.14,
            phase_title=phase_yolo
        )
        if ret_yolo != 0:
            err_yolo = out_yolo.strip()[-500:]
            return False, f"PyTorch instalado, mas falhou ao instalar Ultralytics: {err_yolo}"

        # Validação final de CUDA no PyTorch (0.92 a 1.00)
        if progress_callback:
            progress_callback(0.93, "Validando suporte CUDA na GPU NVIDIA...", "Testando torch.cuda.is_available()...")

        test_cmd = [sys.executable, "-c", "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NO')"]
        test_res = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        is_cuda_ok = "CUDA_OK" in (test_res.stdout or "")

        if progress_callback:
            if is_cuda_ok:
                progress_callback(1.0, "Aceleração NVIDIA CUDA validada e pronta!", "Concluído com sucesso.")
            else:
                progress_callback(1.0, "Instalação concluída. (Verifique compatibilidade de driver caso CUDA não apareça imediatamente)", "Concluído.")

        logger.info("[Hardware] Instalação CUDA e Ultralytics concluída com sucesso.")
        return True, "Instalação das dependências PyTorch CUDA e Ultralytics concluída com sucesso!"
    except Exception as ex:
        err_msg = f"Erro ao configurar componentes finais: {str(ex)}"
        logger.error(err_msg)
        return False, err_msg

def validate_and_setup_gpu_requirements(
    auto_install: bool = True,
    force_install: bool = False,
    progress_callback: Optional[Callable[[float, str, Optional[str]], None]] = None
) -> Dict[str, Any]:
    """
    Valida os requisitos de GPU ao iniciar o sistema.
    Caso o computador possua GPU NVIDIA e os pacotes de aceleração CUDA não estejam instalados,
    executa os comandos de instalação na primeira execução com suporte a barra de progresso.
    """
    gpu_info = detect_nvidia_gpu()
    fw_info = check_cuda_framework_support()

    status = {
        "has_gpu": gpu_info["has_nvidia_gpu"],
        "gpu_name": gpu_info["gpu_name"],
        "cuda_ready": fw_info["torch_cuda"],
        "auto_installed": False,
        "message": ""
    }

    if not gpu_info["has_nvidia_gpu"]:
        status["message"] = "Nenhuma GPU NVIDIA encontrada no sistema. O sistema utilizará CPU."
        return status

    if status["cuda_ready"] and not force_install:
        status["message"] = f"Ambiente GPU verificado: {gpu_info['gpu_name']} pronto com suporte a aceleração por hardware (PyTorch CUDA)."
        return status

    # Se possui GPU NVIDIA mas o suporte CUDA não está instalado (ou a instalação foi forçada pelo usuário)
    if auto_install or force_install:
        logger.info(f"[Hardware] GPU NVIDIA '{gpu_info['gpu_name']}' detectada. Executando instalação de dependências CUDA com progresso...")
        success, msg = install_cuda_packages(progress_callback=progress_callback)
        status["auto_installed"] = success
        if success:
            re_fw = check_cuda_framework_support()
            status["cuda_ready"] = re_fw["torch_cuda"]
            status["message"] = f"Dependências CUDA instaladas com sucesso para a GPU {gpu_info['gpu_name']}!"
        else:
            status["message"] = f"GPU detectada ({gpu_info['gpu_name']}), mas a instalação das dependências falhou: {msg}"
    else:
        status["message"] = f"GPU NVIDIA detectada ({gpu_info['gpu_name']}), mas as dependências PyTorch CUDA ainda não estão instaladas."

    return status

def get_effective_device(preference: str = "cpu") -> Tuple[str, str, Dict[str, Any]]:
    """
    Resolve o dispositivo de processamento efetivo com base na preferência solicitada
    e na presença física de GPU NVIDIA.

    Args:
        preference: "cpu" ou "gpu"

    Returns:
        Tuple (effective_device, status_message, gpu_info)
        - effective_device: "cpu" ou "gpu"
        - status_message: Mensagem amigável para exibição em log/UI
        - gpu_info: Dicionário retornado por detect_nvidia_gpu()
    """
    gpu_info = detect_nvidia_gpu()
    clean_pref = (preference or "cpu").lower().strip()

    if clean_pref == "gpu":
        if gpu_info["has_nvidia_gpu"]:
            effective = "gpu"
            msg = f"⚡ Modo GPU selecionado: Placa {gpu_info['gpu_name']} detectada."
        else:
            effective = "cpu"
            msg = "⚠️ GPU NVIDIA solicitada, mas nenhuma placa aceleradora NVIDIA foi encontrada no sistema. Fallback automático para CPU ativado."
    else:
        effective = "cpu"
        if gpu_info["has_nvidia_gpu"]:
            msg = f"💻 Modo CPU selecionado manualmente. (GPU NVIDIA '{gpu_info['gpu_name']}' está disponível no sistema, mas não será utilizada)."
        else:
            msg = "💻 Modo CPU ativado (Processamento padrão via CPU)."

    return effective, msg, gpu_info

def get_optimal_batch_size(device: str = "cpu", custom_batch_size: Optional[int] = None) -> int:
    """
    Determina o tamanho de lote ideal para inferência de visão computacional.
    
    Args:
        device: "cpu" ou "gpu"
        custom_batch_size: Valor customizado opcional definido pelo usuário
        
    Returns:
        int: Tamanho de lote recomendado (ex: 32 para GPU, 1 para CPU)
    """
    if custom_batch_size is not None and isinstance(custom_batch_size, int) and custom_batch_size > 0:
        return max(1, min(128, custom_batch_size))

    clean_device = (device or "cpu").lower().strip()
    if clean_device == "gpu":
        return 64
    return 1

def detect_connected_cameras() -> List[Dict[str, Any]]:
    """
    Detecta e lista as webcams e dispositivos de captura de vídeo conectados ao sistema,
    retornando o índice do dispositivo e o nome de hardware (quando disponível).
    Retorna lista de dicts: [{'index': 0, 'name': 'BisonCam,NB Pro', 'label': '🎥 [0] BisonCam,NB Pro'}, ...]
    """
    cameras: List[Dict[str, Any]] = []
    
    # 0. Se estiver em Linux/contêiner headless sem dispositivos /dev/video*, não sondar via OpenCV
    # (evita timeout de 30s por índice do backend ffmpeg no Streamlit Cloud / Docker)
    if sys.platform.startswith("linux"):
        v_devices = glob.glob("/dev/video*")
        if not v_devices:
            return [{"index": 0, "name": "Câmera 0", "label": "🎥 Câmera 0 (Padrão)"}]

    # 1. Tentar DirectShow via pygrabber (se instalado)
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        if devices:
            for idx, dev in enumerate(devices):
                cameras.append({
                    "index": idx,
                    "name": str(dev),
                    "label": f"🎥 Câmera {idx} - {dev}"
                })
            return cameras
    except Exception:
        pass

    # 2. Tentar via Windows PowerShell (PnPEntity - Câmeras / Dispositivos de Imagem)
    try:
        cmd = "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | Select-Object -ExpandProperty Name"
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=3)
        names = [n.strip() for n in res.stdout.strip().splitlines() if n.strip()]
        if names:
            for idx, name in enumerate(names):
                cameras.append({
                    "index": idx,
                    "name": name,
                    "label": f"🎥 Câmera {idx} - {name}"
                })
            return cameras
    except Exception:
        pass

    # 3. Fallback: Sondagem rápida OpenCV para índices 0 a 3
    try:
        import cv2
        for idx in range(4):
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cameras.append({
                    "index": idx,
                    "name": f"Dispositivo de Vídeo {idx}",
                    "label": f"🎥 Câmera {idx} (Dispositivo Padrão)"
                })
                cap.release()
    except Exception:
        pass

    # 4. Fallback padrão caso nenhuma câmera física tenha respondido de imediato
    if not cameras:
        for idx in range(4):
            cameras.append({
                "index": idx,
                "name": f"Câmera {idx}",
                "label": f"🎥 Câmera {idx} (Padrão do Sistema)"
            })

    return cameras


def ensure_browser_compatible_video(video_path: str) -> bool:
    """
    Garante que o arquivo de vídeo MP4 gerado pelo OpenCV (originalmente em codec MPEG-4 / mp4v / FMP4)
    seja transcodificado de forma síncrona e ultrarrápida para H.264 (AVC1 com pixel format YUV420p
    e flag +faststart), permitindo reprodução direta, instantânea e compatível em todos os navegadores web
    modernos (Chrome, Edge, Firefox, Safari) dentro do player HTML5 do Streamlit.
    """
    if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        return False

    # 1. Verifica se já está codificado em H.264 para evitar reprocessamento desnecessário
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).lower()
        cap.release()
        if fourcc_str in ("h264", "x264", "avc1"):
            return True
    except Exception:
        pass

    # 2. Localiza o executável FFmpeg no sistema ou pacote imageio-ffmpeg
    try:
        from src.utils.video_downloader import get_ffmpeg_executable_path
        ffmpeg_bin = get_ffmpeg_executable_path() or "ffmpeg"
    except Exception:
        ffmpeg_bin = "ffmpeg"

    tmp_converted = video_path + ".browser_h264.mp4"
    # Ordem de preferência de encoders: aceleradores de hardware primeiro (nvenc, mf), seguidos por libx264 ultrarrápido
    encoders = ["h264_nvenc", "h264_mf", "libx264", "libopenh264", "h264"]

    for enc in encoders:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", video_path,
            "-vcodec", enc,
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Garante largura e altura pares obrigatórias no yuv420p
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if enc in ("libx264", "libopenh264", "h264"):
            cmd.extend(["-preset", "ultrafast"])
        cmd.append(tmp_converted)

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and os.path.exists(tmp_converted) and os.path.getsize(tmp_converted) > 0:
                os.replace(tmp_converted, video_path)
                return True
        except Exception:
            continue

    if os.path.exists(tmp_converted):
        try:
            os.remove(tmp_converted)
        except Exception:
            pass

    return False

