"""
Gerenciador de Configurações Globais do SenpAI.
Persiste e lê opções de sistema (dispositivo de processamento CPU/GPU, modelo de visão computacional, etc.).
"""

import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS_PATH = "config/settings.json"

VISION_MODELS_CATALOG: Dict[str, Dict[str, Any]] = {
    "yolov8": {
        "id": "yolov8",
        "name": "YOLOv8 Pose",
        "full_name": "Ultralytics YOLOv8 Pose (Baseline Estável)",
        "badge": "Padrão Estável",
        "weights_file": "yolov8n-pose.pt",
        "icon": "🎯",
        "color": "#6366F1",
        "tagline": "Arquitetura consagrada com máxima estabilidade e compatibilidade universal.",
        "summary": "Modelo baseline do SenpAI com pesos neurais já integrados localmente. Oferece alta velocidade de inferência, baixo consumo de VRAM e excelente equilíbrio de rastreamento em movimentos rápidos de artes marciais.",
        "advantages": [
            "Estabilidade Comprovada: Modelo padrão homologado em toda a suíte de testes do SenpAI.",
            "Pesos Locais Embutidos: Não depende de downloads externos ou conexão de rede para inicialização.",
            "Excelente Latência: Operação em alta cadência de FPS tanto em GPU CUDA (FP16) quanto em CPU.",
            "Compatibilidade Ampla: Suporte universal em qualquer versão do PyTorch e drivers NVIDIA."
        ],
        "kendo_focus": "Rastreamento equilibrado de postura corporal, Sonkyō e deslocamentos com Shinai.",
        "specs": {
            "parameters": "3.3M",
            "vram_recommended": "2 GB+",
            "target_latency": "< 12ms (GPU)"
        }
    },
    "yolov11": {
        "id": "yolov11",
        "name": "YOLOv11 Pose",
        "full_name": "Ultralytics YOLO11 Pose (Otimizado & Ágil)",
        "badge": "Otimizado & Ágil",
        "weights_file": "yolo11n-pose.pt",
        "icon": "⚡",
        "color": "#10B981",
        "tagline": "Arquitetura C3k2 com atenção espacial e inferência ultrarrápida.",
        "summary": "Evolução arquitetural focada em eficiência computacional extrema. Reduz parâmetros e operações FLOPS, acelerando o processamento em lote (batch inference) e melhorando a detecção fina de extremidades.",
        "advantages": [
            "Arquitetura C3k2 Refinada: Menor número de parâmetros com ganho de até 15-20% em FPS.",
            "Precisão Anatômica Superior: Rastreamento aprimorado de punhos e antebraços, ideal para o alvo Kote.",
            "Menor Consumo de VRAM: Otimizado para transmissões ao vivo multi-feed e placas de entrada.",
            "Tempo de Resposta Ágil: Redução da latência por frame na detecção em tempo real."
        ],
        "kendo_focus": "Detecção veloz de ataques de punho (Kote) e respostas imediatas em tempo real.",
        "specs": {
            "parameters": "2.9M",
            "vram_recommended": "2 GB+",
            "target_latency": "< 9ms (GPU)"
        }
    },
    "yolov12": {
        "id": "yolov12",
        "name": "YOLOv12 Pose",
        "full_name": "Ultralytics YOLOv12 Pose (Atenção Centrada)",
        "badge": "Anti-Oclusão & Tsubazeriai",
        "weights_file": "yolo12n-pose.pt",
        "icon": "🛡️",
        "color": "#F59E0B",
        "tagline": "Mecanismo Attention-Centric (A-C2f) para cenários com oclusão severa.",
        "summary": "Desenvolvido com camadas de atenção profunda projetadas para desentrelaçar corpos em sobreposição. Mantém a integridade do esqueleto dos dois kendokas mesmo em momentos caóticos de combate fechado.",
        "advantages": [
            "Resiliência no Tsubazeriai: Não confunde os membros dos dois competidores em disputas coladas.",
            "Diferenciação de Armaduras (Bogu): Isola com fidelidade o contorno do Men, Do e Tare em fundos escuros.",
            "Imunidade a Reflexos: Alta robustez contra iluminação irregular e reflexos no piso polido do dojo.",
            "Avaliação Biomecânica Fina: Menor variância nos ângulos calculados de coluna e inclinação de tronco."
        ],
        "kendo_focus": "Máxima precisão em combate corpo a corpo (Tsubazeriai) e Shinai cruzados.",
        "specs": {
            "parameters": "3.8M",
            "vram_recommended": "4 GB+",
            "target_latency": "< 14ms (GPU)"
        }
    },
    "yolov26": {
        "id": "yolov26",
        "name": "YOLOv26 Pose",
        "full_name": "SenpAI Next-Gen YOLOv26 Pose (Arbitragem de Elite 2026)",
        "badge": "Next-Gen AI 2026",
        "weights_file": "yolov26n-pose.pt",
        "icon": "👑",
        "color": "#EC4899",
        "tagline": "Motor neural de vanguarda 2026 com antecipação cinemática de Ki-Ken-Tai-Ichi.",
        "summary": "O ápice em visão computacional especializada para Kendo. Incorpora predição temporal antecipada de trajetória de espada e sincronização milimétrica entre impacto do golpe e pisada (fumikomi), atendendo aos mais rigorosos critérios FIK.",
        "advantages": [
            "Antecipação Cinemática de Golpe: Prediz vetores de ataque antes do impacto físico no Men/Kote/Do.",
            "Sincronização Ki-Ken-Tai-Ichi: Mede intervalos de milissegundos entre som, shinai e pé de apoio.",
            "Aceleração Quântica / TensorRT: Pipeline otimizado para máxima performance em GPUs RTX série 40/50.",
            "Arbitragem de Alto Rendimento: Concebido para grandes campeonatos mundiais e decisões milimétricas."
        ],
        "kendo_focus": "Validação oficial estrita de Yuko-Datotsu e Ki-Ken-Tai-Ichi em torneios internacionais.",
        "specs": {
            "parameters": "4.5M",
            "vram_recommended": "4 GB+",
            "target_latency": "< 11ms (GPU TensorRT)"
        }
    }
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "processing_device": "cpu",
    "vision_model": "yolov8"
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

def get_vision_model(config_path: str = DEFAULT_SETTINGS_PATH) -> str:
    """
    Retorna o modelo de visão computacional configurado ("yolov8", "yolov11", "yolov12", "yolov26").
    Retorna "yolov8" como padrão caso não esteja configurado ou seja inválido.
    """
    settings = load_settings(config_path)
    model = settings.get("vision_model", "yolov8").lower().strip()
    return model if model in VISION_MODELS_CATALOG else "yolov8"

def set_vision_model(model_name: str, config_path: str = DEFAULT_SETTINGS_PATH) -> Dict[str, Any]:
    """
    Atualiza e salva a preferência de modelo de visão computacional.
    """
    clean_model = model_name.lower().strip() if model_name else "yolov8"
    if clean_model not in VISION_MODELS_CATALOG:
        clean_model = "yolov8"
    
    settings = load_settings(config_path)
    settings["vision_model"] = clean_model
    save_settings(settings, config_path)
    return settings

def get_vision_model_info(model_id: str) -> Dict[str, Any]:
    """
    Retorna as informações completas de metadados do modelo de visão informado.
    Faz fallback para YOLOv8 caso o identificador não exista.
    """
    clean_id = (model_id or "yolov8").lower().strip()
    return VISION_MODELS_CATALOG.get(clean_id, VISION_MODELS_CATALOG["yolov8"])
