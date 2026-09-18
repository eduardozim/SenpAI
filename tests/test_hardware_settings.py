"""
Testes Automatizados para Detecção de Hardware, Menu de Configurações, Dispositivo Efetivo e Modelos de Visão.
"""

import os
import json
import unittest
from unittest.mock import patch

from src.utils.hardware import detect_nvidia_gpu, get_effective_device
from src.utils.settings_manager import (
    load_settings, save_settings, get_processing_device, set_processing_device,
    get_vision_model, set_vision_model, get_vision_model_info, VISION_MODELS_CATALOG
)
from src.vision.pose_detector import PoseDetector
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

    def test_settings_manager_vision_model_load_and_save(self):
        """Valida a seleção, persistência e fallback para os modelos de visão (YOLOv8, YOLOv11, YOLOv12, YOLOv26)."""
        # 1. Padrão inicial deve ser yolov8
        m_default = get_vision_model(self.test_settings_path)
        self.assertEqual(m_default, "yolov8")

        # 2. Salvar e recuperar cada um dos 4 modelos
        for model_id in ["yolov8", "yolov11", "yolov12", "yolov26"]:
            set_vision_model(model_id, self.test_settings_path)
            retrieved = get_vision_model(self.test_settings_path)
            self.assertEqual(retrieved, model_id, f"O modelo {model_id} deve ser persistido corretamente.")

        # 3. Fallback para modelo inválido
        set_vision_model("modelo_inexistente_xyz", self.test_settings_path)
        fallback_m = get_vision_model(self.test_settings_path)
        self.assertEqual(fallback_m, "yolov8", "Modelos desconhecidos devem sofrer fallback seguro para yolov8.")

    def test_vision_models_catalog_completeness(self):
        """Garante a existência e integridade dos 4 modelos de visão com seus respectivos descritivos de vantagens."""
        expected_models = ["yolov8", "yolov11", "yolov12", "yolov26"]
        for mid in expected_models:
            self.assertIn(mid, VISION_MODELS_CATALOG, f"Modelo {mid} deve estar registrado no catálogo oficial.")
            info = VISION_MODELS_CATALOG[mid]
            self.assertIn("name", info)
            self.assertIn("full_name", info)
            self.assertIn("badge", info)
            self.assertIn("weights_file", info)
            self.assertIn("summary", info)
            self.assertIn("advantages", info)
            self.assertIsInstance(info["advantages"], list)
            self.assertGreaterEqual(len(info["advantages"]), 3, f"Modelo {mid} deve listar pelo menos 3 vantagens descritivas.")
            self.assertIn("kendo_focus", info)

        # Testar get_vision_model_info com fallback
        info_v11 = get_vision_model_info("yolov11")
        self.assertEqual(info_v11["id"], "yolov11")
        self.assertIn("C3k2", str(info_v11["advantages"]))

        info_fallback = get_vision_model_info("invalido")
        self.assertEqual(info_fallback["id"], "yolov8")

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

    def test_pose_detector_vision_model_resolution(self):
        """Valida que o PoseDetector aceita e armazena os metadados do modelo de visão selecionado."""
        detector = PoseDetector(device="cpu", model_name="yolov11")
        self.assertEqual(detector.model_name, "yolov11")
        self.assertEqual(detector.model_info["name"], "YOLOv11 Pose")
        path = detector._resolve_model_path_or_name()
        self.assertTrue(len(path) > 0)

    def test_pipeline_device_and_vision_model_integration(self):
        """Verifica a inicialização e integração conjunta do dispositivo de hardware e modelo de visão no SenpAIPipeline."""
        pipeline = SenpAIPipeline(calibration_profile="normal", device_preference="cpu", vision_model="yolov26")
        self.assertEqual(pipeline.effective_device, "cpu")
        self.assertEqual(pipeline.vision_model, "yolov26")
        self.assertEqual(pipeline.vision_model_info["name"], "YOLOv26 Pose")
        self.assertEqual(pipeline.pose_detector.model_name, "yolov26")
        self.assertIn("device_status_message", dir(pipeline))


if __name__ == "__main__":
    unittest.main()
