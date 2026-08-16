import os
import unittest
from src.utils.logger_manager import (
    setup_system_logger, log_event, get_memory_logs,
    get_debug_log_file_content, clear_debug_logs,
    get_log_summary, run_system_diagnostic_check
)

class TestLoggerManager(unittest.TestCase):
    def setUp(self):
        clear_debug_logs()

    def test_logger_initialization_and_logging(self):
        logger = setup_system_logger()
        self.assertIsNotNone(logger)

        log_event("INFO", "Teste de mensagem de informação", "test_module")
        log_event("WARNING", "Teste de mensagem de aviso", "test_module")
        log_event("ERROR", "Teste de mensagem de erro", "test_module")

        logs = get_memory_logs(max_entries=10)
        self.assertGreaterEqual(len(logs), 3)

        levels = [l["level"] for l in logs]
        self.assertIn("INFO", levels)
        self.assertIn("WARNING", levels)
        self.assertIn("ERROR", levels)

    def test_log_summary_counts(self):
        log_event("ERROR", "Erro 1", "test")
        log_event("ERROR", "Erro 2", "test")
        log_event("WARNING", "Aviso 1", "test")

        summary = get_log_summary()
        self.assertGreaterEqual(summary["errors_count"], 2)
        self.assertGreaterEqual(summary["warnings_count"], 1)
        self.assertGreaterEqual(summary["total_logs"], 3)

    def test_get_debug_log_file_content(self):
        log_event("INFO", "Conteúdo para arquivo de log", "test")
        content = get_debug_log_file_content()
        self.assertIn("Conteúdo para arquivo de log", content)

    def test_run_system_diagnostic_check(self):
        report = run_system_diagnostic_check()
        self.assertIn("python_version", report)
        self.assertIn("checks", report)
        self.assertEqual(report["checks"]["filesystem"]["status"], "OK")

    def test_clear_debug_logs(self):
        log_event("INFO", "Mensagem antes de limpar", "test")
        clear_debug_logs()
        summary = get_log_summary()
        content = get_debug_log_file_content()
        self.assertIn("Log de debug reiniciado", content)

if __name__ == "__main__":
    unittest.main()
