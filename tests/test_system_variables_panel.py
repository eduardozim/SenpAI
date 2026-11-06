"""
Testes do painel "Variáveis do Sistema" (Configurações > Variáveis do Sistema).
"""

import os
import unittest
from unittest import mock

from src.utils import api_keys as ak
from src.utils.system_variables_panel import render_system_variables_panel


def _sys_vars_app():
    from src.utils.system_variables_panel import render_system_variables_panel
    render_system_variables_panel()


class TestSystemVariablesPanel(unittest.TestCase):
    def setUp(self):
        ak.save_keys({k: None for k in ak.KNOWN_KEYS})
        self._env = {k: os.environ.pop(k, None) for k in ak.KNOWN_KEYS}

    def tearDown(self):
        ak.save_keys({k: None for k in ak.KNOWN_KEYS})
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_panel_renders_cleanly(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_function(_sys_vars_app, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any("Variáveis do Sistema" in m.value for m in at.markdown))
        # Verifica a presença dos inputs para as chaves conhecidas
        self.assertGreaterEqual(len(at.text_input), len(ak.KNOWN_KEYS))
        # Verifica a presença dos botões do formulário
        self.assertTrue(any("Salvar" in b.label for b in at.button))
        self.assertTrue(any("Apagar" in b.label for b in at.button))

    def test_panel_reflects_configured_keys(self):
        from streamlit.testing.v1 import AppTest

        ak.save_keys({"GEMINI_API_KEY": "AIzaSyTestKey123"})
        self.assertEqual(ak.key_source("GEMINI_API_KEY"), "arquivo")

        at = AppTest.from_function(_sys_vars_app, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertTrue(any("Configurada no arquivo local" in inp.label for inp in at.text_input))


if __name__ == "__main__":
    unittest.main()
