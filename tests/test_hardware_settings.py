"""
Testes Automatizados para Detecção de Hardware, Menu de Configurações e Dispositivo Efetivo.
"""

import os
import json
import unittest
from unittest.mock import patch

from src.utils.hardware import detect_nvidia_gpu, get_effective_device
from src.utils.settings_manager import (
    load_settings, save_settings, get_processing_device, set_processing_device
)
from src.pipeline import SenpAIPipeline


class TestHardwareAndSettings(unittest.TestCase):
    def setUp(self):
        self.test_settings_path = "config/test_settings.json"
        if os.path.exists(self.test_settings_path):
            os.remove(self.test_settings_path)

    def tearDown(self):
        if os.path.exists(self.test_settings_path):
            os.remove(self.test_settings_path)

    def test_settings_manager_load_and_save(self):
        """Valida o carregamento padrão, salvamento e persistência das configurações de hardware (CPU/GPU)."""
        # 1. Carregar padrão quando arquivo não existe
        settings = load_settings(self.test_settings_path)
        self.assertEqual(settings["processing_device"], "cpu")

        # 2. Salvar e recuperar preferência GPU
        set_processing_device("gpu", self.test_settings_path)
        dev = get_processing_device(self.test_settings_path)
        self.assertEqual(dev, "gpu")

        # 3. Salvar e recuperar preferência CPU
        set_processing_device("cpu", self.test_settings_path)
        dev = get_processing_device(self.test_settings_path)
        self.assertEqual(dev, "cpu")

    def test_detect_nvidia_gpu_structure(self):
        """Verifica a estrutura do dicionário retornado pela detecção dinâmica de GPU NVIDIA."""
        gpu_info = detect_nvidia_gpu()
        self.assertIn("has_nvidia_gpu", gpu_info)
        self.assertIn("gpu_name", gpu_info)
        self.assertIn("details", gpu_info)
        self.assertIsInstance(gpu_info["has_nvidia_gpu"], bool)

    def test_get_effective_device_cpu_preference(self):
        """Valida a resolução de dispositivo quando a preferência do usuário é explicitamente CPU."""
        effective, msg, info = get_effective_device("cpu")
        self.assertEqual(effective, "cpu")
        self.assertIn("Modo CPU", msg)

    @patch("src.utils.hardware.detect_nvidia_gpu")
    def test_get_effective_device_gpu_fallback_when_no_gpu(self, mock_detect):
        """Valida o fallback automático e transparente para CPU quando o usuário escolhe GPU mas não há hardware compatível."""
        mock_detect.return_value = {
            "has_nvidia_gpu": False,
            "gpu_name": "Nenhum",
            "gpu_count": 0,
            "driver_version": "N/A",
            "memory_total": "N/A",
            "detection_methods": [],
            "details": "Sem GPU NVIDIA"
        }
        effective, msg, info = get_effective_device("gpu")
        self.assertEqual(effective, "cpu")
        self.assertIn("Fallback automático para CPU", msg)

    @patch("src.utils.hardware.detect_nvidia_gpu")
    def test_get_effective_device_gpu_success_when_gpu_present(self, mock_detect):
        """Verifica a ativação do acelerador GPU NVIDIA CUDA quando a placa aceleradora está presente no sistema."""
        mock_detect.return_value = {
            "has_nvidia_gpu": True,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "driver_version": "550.00",
            "memory_total": "24576 MiB",
            "detection_methods": ["nvidia-smi"],
            "details": "GPU NVIDIA Detectada"
        }
        effective, msg, info = get_effective_device("gpu")
        self.assertEqual(effective, "gpu")
        self.assertIn("Modo GPU selecionado", msg)
        self.assertIn("RTX 4090", msg)

    @patch("src.utils.hardware.detect_nvidia_gpu")
    def test_validate_and_setup_gpu_requirements_no_gpu(self, mock_detect):
        """Testa o validador de dependências CUDA quando nenhuma GPU NVIDIA está presente no computador."""
        from src.utils.hardware import validate_and_setup_gpu_requirements
        mock_detect.return_value = {
            "has_nvidia_gpu": False,
            "gpu_name": "Nenhum",
            "gpu_count": 0,
            "driver_version": "N/A",
            "memory_total": "N/A",
            "detection_methods": [],
            "details": "Sem GPU NVIDIA"
        }
        res = validate_and_setup_gpu_requirements(auto_install=False)
        self.assertFalse(res["has_gpu"])
        self.assertIn("Nenhuma GPU NVIDIA encontrada", res["message"])

    def test_pipeline_device_integration(self):
        """Verifica a inicialização e integração correta do dispositivo de processamento no SenpAIPipeline."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu")
        self.assertEqual(pipeline.effective_device, "cpu")
        self.assertIn("device_status_message", dir(pipeline))


if __name__ == "__main__":
    unittest.main()
