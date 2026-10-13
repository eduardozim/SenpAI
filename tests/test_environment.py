"""
Testes Unitários para o Módulo de Verificação de Ambiente Virtual Python (environment.py).
"""

import sys
import os
import unittest
from unittest.mock import patch

from src.utils.environment import (
    is_in_virtual_environment,
    get_virtual_environment_info,
    validate_virtual_environment,
    get_execution_environment_info,
    is_web_server_environment
)
from src.utils.logger_manager import run_system_diagnostic_check


class TestEnvironmentVerification(unittest.TestCase):
    def test_is_in_virtual_environment_returns_bool(self):
        """Verifica se a função is_in_virtual_environment retorna um tipo booleano válido."""
        res = is_in_virtual_environment()
        self.assertIsInstance(res, bool)

    @patch("sys.prefix", "D:\\Projetos\\SenpAI\\Dev\\.venv")
    @patch("sys.base_prefix", "C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311")
    def test_is_in_virtual_environment_detected_when_prefix_differs(self):
        """Valida que o ambiente virtual é detectado quando sys.prefix difere de sys.base_prefix."""
        self.assertTrue(is_in_virtual_environment())

    @patch("sys.prefix", "C:\\Python311")
    @patch("sys.base_prefix", "C:\\Python311")
    @patch("sys.exec_prefix", "C:\\Python311")
    @patch("sys.base_exec_prefix", "C:\\Python311")
    @patch.dict(os.environ, {}, clear=True)
    def test_is_in_virtual_environment_false_when_global(self):
        """Valida que retorna False quando executando no interpretador global sem variáveis de ambiente."""
        self.assertFalse(is_in_virtual_environment())

    @patch("sys.prefix", "C:\\Python311")
    @patch("sys.base_prefix", "C:\\Python311")
    @patch.dict(os.environ, {"VIRTUAL_ENV": "D:\\Projetos\\SenpAI\\Dev\\.venv"})
    def test_is_in_virtual_environment_detected_via_virtual_env_var(self):
        """Valida que o ambiente virtual é detectado via variável VIRTUAL_ENV."""
        self.assertTrue(is_in_virtual_environment())

    @patch("sys.prefix", "C:\\Python311")
    @patch("sys.base_prefix", "C:\\Python311")
    @patch.dict(os.environ, {"CONDA_PREFIX": "C:\\Users\\User\\anaconda3\\envs\\senpai"})
    def test_is_in_virtual_environment_detected_via_conda_prefix(self):
        """Valida que o ambiente virtual é detectado via variável CONDA_PREFIX."""
        self.assertTrue(is_in_virtual_environment())

    def test_get_virtual_environment_info_structure(self):
        """Verifica a presença de todas as chaves de metadados obrigatórias."""
        info = get_virtual_environment_info()
        self.assertIn("is_virtual_env", info)
        self.assertIn("env_type", info)
        self.assertIn("prefix", info)
        self.assertIn("base_prefix", info)
        self.assertIn("executable", info)
        self.assertIn("python_version", info)
        self.assertIn("is_project_dot_venv", info)
        self.assertIn("expected_dot_venv_path", info)
        self.assertIsInstance(info["is_virtual_env"], bool)

    @patch("src.utils.environment.is_in_virtual_environment", return_value=True)
    def test_validate_virtual_environment_when_active(self, mock_is_venv):
        """Valida o retorno positivo de validação quando o ambiente virtual está ativo."""
        is_valid, msg = validate_virtual_environment(log_warning=False)
        self.assertTrue(is_valid)
        self.assertIn("Ambiente Virtual ativo", msg)

    @patch("src.utils.environment.is_in_virtual_environment", return_value=False)
    def test_validate_virtual_environment_when_inactive(self, mock_is_venv):
        """Valida o alerta de erro detalhado gerado quando nenhum ambiente virtual é detectado."""
        is_valid, msg = validate_virtual_environment(log_warning=False)
        self.assertFalse(is_valid)
        self.assertIn("AMBIENTE VIRTUAL PYTHON NÃO IDENTIFICADO", msg)
        self.assertIn(".venv", msg)

    def test_diagnostic_report_includes_virtual_environment_check(self):
        """Verifica se o relatório de diagnóstico do sistema inclui a validação de ambiente virtual."""
        report = run_system_diagnostic_check()
        self.assertIn("virtual_environment", report["checks"])
        venv_check = report["checks"]["virtual_environment"]
        self.assertIn("status", venv_check)
        self.assertIn("is_virtual_env", venv_check)
        self.assertIn("executable", venv_check)

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "local"}, clear=True)
    def test_execution_environment_local_mode(self):
        """Valida a detecção de ambiente local quando configurado explicitamente."""
        info = get_execution_environment_info()
        self.assertFalse(info["is_cloud"])
        self.assertFalse(info["is_web_server"])
        self.assertEqual(info["deployment_mode"], "local")
        self.assertFalse(is_web_server_environment())

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "web"}, clear=True)
    def test_execution_environment_web_mode(self):
        """Valida a detecção de servidor web via SENPAI_DEPLOY_MODE=web."""
        info = get_execution_environment_info()
        self.assertTrue(info["is_cloud"])
        self.assertTrue(info["is_web_server"])
        self.assertEqual(info["deployment_mode"], "cloud")
        self.assertTrue(is_web_server_environment())

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "server"}, clear=True)
    def test_execution_environment_server_mode(self):
        """Valida a detecção de servidor web via SENPAI_DEPLOY_MODE=server."""
        info = get_execution_environment_info()
        self.assertTrue(info["is_web_server"])
        self.assertTrue(is_web_server_environment())

    @patch.dict(os.environ, {"SENPAI_IS_WEB_SERVER": "1"}, clear=True)
    def test_execution_environment_flag_var(self):
        """Valida a detecção de servidor web via flag SENPAI_IS_WEB_SERVER=1."""
        info = get_execution_environment_info()
        self.assertTrue(info["is_web_server"])
        self.assertTrue(is_web_server_environment())

    @patch.dict(os.environ, {"SPACE_ID": "huggingface/senpai"}, clear=True)
    def test_execution_environment_huggingface(self):
        """Valida a detecção automática de servidor web em nuvem (Hugging Face Spaces)."""
        info = get_execution_environment_info()
        self.assertTrue(info["is_cloud"])
        self.assertTrue(info["is_web_server"])
        self.assertEqual(info["provider_label"], "Hugging Face Spaces")
        self.assertTrue(is_web_server_environment())

    @patch.dict(os.environ, {"RENDER": "true"}, clear=True)
    def test_execution_environment_render(self):
        """Valida a detecção automática de servidor web em nuvem (Render)."""
        info = get_execution_environment_info()
        self.assertTrue(info["is_web_server"])
        self.assertEqual(info["provider_label"], "Render Cloud")
        self.assertTrue(is_web_server_environment())


if __name__ == "__main__":
    unittest.main()

