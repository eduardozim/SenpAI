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
    validate_virtual_environment
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


if __name__ == "__main__":
    unittest.main()
