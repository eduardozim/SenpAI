"""
A tela de Configurações conta os testes (get_total_test_count) dentro do processo do app. Descobrir os
testes importa tests/__init__.py, que liga o modo de teste; isso não pode vazar para o app, senão
Apagar/Baixar/Carregar Treinamento passam a agir na sandbox em vez dos dados reais.
"""

import unittest

from src.utils import test_isolation
from src.utils.test_runner import get_total_test_count


class TestTestRunnerIsolation(unittest.TestCase):
    def test_counting_tests_does_not_leave_test_mode_on(self):
        test_isolation.disable_test_mode()
        try:
            self.assertGreater(get_total_test_count(), 0)
            self.assertFalse(test_isolation.is_test_mode())
        finally:
            test_isolation.enable_test_mode()


if __name__ == "__main__":
    unittest.main()
