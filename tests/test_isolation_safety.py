"""
Testes Automatizados para Isolamento de Dados e Proteção de Arquivos Reais de Produção/Dev.
Valida que execuções de testes nunca alteram ou apagam data/training_history.json,
config/calibration_profiles.json ou data/kenshi_baselines.json.
"""

import os
import unittest
from src.utils.test_isolation import (
    is_test_mode,
    enable_test_mode,
    get_safe_data_path,
    assert_not_production_write
)
from src.engine.feedback_manager import FeedbackManager

class TestIsolationSafety(unittest.TestCase):

    def setUp(self):
        enable_test_mode()

    def test_test_mode_is_active(self):
        self.assertTrue(is_test_mode(), "O modo de teste deve estar ativo.")

    def test_safe_path_redirects_protected_files(self):
        safe_hist = get_safe_data_path("data/training_history.json")
        self.assertNotEqual(os.path.abspath(safe_hist), os.path.abspath("data/training_history.json"))
        self.assertIn("senpai_test_sandbox", safe_hist.replace("\\", "/"))

        safe_prof = get_safe_data_path("config/calibration_profiles.json")
        self.assertNotEqual(os.path.abspath(safe_prof), os.path.abspath("config/calibration_profiles.json"))

        safe_base = get_safe_data_path("data/kenshi_baselines.json")
        self.assertNotEqual(os.path.abspath(safe_base), os.path.abspath("data/kenshi_baselines.json"))

    def test_direct_production_write_raises_error_in_test_mode(self):
        with self.assertRaises(RuntimeError):
            assert_not_production_write("data/training_history.json", "teste de escrita")

        with self.assertRaises(RuntimeError):
            assert_not_production_write("config/calibration_profiles.json", "teste de calibração")

    def test_feedback_manager_defaults_are_sandboxed_in_test_mode(self):
        mgr = FeedbackManager()
        self.assertNotEqual(os.path.abspath(mgr.history_path), os.path.abspath("data/training_history.json"))
        self.assertNotEqual(os.path.abspath(mgr.profiles_path), os.path.abspath("config/calibration_profiles.json"))
        self.assertNotEqual(os.path.abspath(mgr.dataset_path), os.path.abspath("data/feedback_dataset.json"))
        self.assertIn("senpai_test_sandbox", mgr.history_path.replace("\\", "/"))

if __name__ == "__main__":
    unittest.main()

