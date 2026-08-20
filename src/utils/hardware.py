"""
Módulo de detecção e gerenciamento de aceleração por hardware (CPU e GPU NVIDIA).
"""

import os
import subprocess
import logging
from typing import Dict, Any, Tuple, List

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

    try:
        import torch
        torch_cuda = torch.cuda.is_available()
        if torch_cuda:
            torch_device_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        import onnxruntime as ort
        onnx_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        pass

    return {
        "torch_cuda": torch_cuda,
        "torch_device_name": torch_device_name,
        "onnx_cuda": onnx_cuda,
        "mediapipe_cpu_only_win": True
    }

def install_cuda_packages() -> Tuple[bool, str]:
    """
    Executa a instalação dos pacotes PyTorch com suporte CUDA no ambiente Python atual.
    """
    import sys
    logger.info("[Hardware] Iniciando instalação automática de dependências CUDA (PyTorch CUDA)...")
    try:
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=600, text=True)
        logger.info(f"[Hardware] Instalação CUDA concluída com sucesso.")
        return True, "Instalação das dependências PyTorch CUDA concluída com sucesso!"
    except Exception as e:
        err_msg = f"Falha ao executar instalação de pacotes CUDA: {e}"
        logger.error(err_msg)
        return False, err_msg

def validate_and_setup_gpu_requirements(auto_install: bool = True) -> Dict[str, Any]:
    """
    Valida os requisitos de GPU ao iniciar o sistema.
    Caso o computador possua GPU NVIDIA e os pacotes de aceleração CUDA não estejam instalados,
    executa os comandos de instalação na primeira execução.
    """
    gpu_info = detect_nvidia_gpu()
    fw_info = check_cuda_framework_support()

    status = {
        "has_gpu": gpu_info["has_nvidia_gpu"],
        "gpu_name": gpu_info["gpu_name"],
        "cuda_ready": fw_info["torch_cuda"] or fw_info["onnx_cuda"],
        "auto_installed": False,
        "message": ""
    }

    if not gpu_info["has_nvidia_gpu"]:
        status["message"] = "Nenhuma GPU NVIDIA encontrada no sistema. O sistema utilizará CPU."
        return status

    if status["cuda_ready"]:
        status["message"] = f"Ambiente GPU verificado: {gpu_info['gpu_name']} pronto com suporte a aceleração por hardware."
        return status

    # Se possui GPU NVIDIA mas o suporte CUDA não está instalado
    if auto_install:
        logger.info(f"[Hardware] GPU NVIDIA '{gpu_info['gpu_name']}' detectada sem suporte CUDA ativo. Executando instalação inicial de dependências...")
        success, msg = install_cuda_packages()
        status["auto_installed"] = success
        if success:
            re_fw = check_cuda_framework_support()
            status["cuda_ready"] = re_fw["torch_cuda"] or re_fw["onnx_cuda"]
            status["message"] = f"Dependências CUDA instaladas com sucesso para a GPU {gpu_info['gpu_name']}!"
        else:
            status["message"] = f"GPU detectada ({gpu_info['gpu_name']}), mas a instalação das dependências falhou: {msg}"
    else:
        status["message"] = f"GPU NVIDIA detectada ({gpu_info['gpu_name']}), mas as dependências CUDA ainda não estão instaladas."

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
 
def detect_connected_cameras() -> List[Dict[str, Any]]:
    """
    Detecta e lista as webcams e dispositivos de captura de vídeo conectados ao sistema,
    retornando o índice do dispositivo e o nome de hardware (quando disponível).
    Retorna lista de dicts: [{'index': 0, 'name': 'BisonCam,NB Pro', 'label': '🎥 [0] BisonCam,NB Pro'}, ...]
    """
    cameras: List[Dict[str, Any]] = []

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

