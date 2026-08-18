#!/usr/bin/env python3
"""
Script de Execução de Testes Automatizados do ShinpanAI.
Executa a suíte de 44 testes automatizados, exibe o progresso em tempo real
e salva o log detalhado descritivo em 'logs/shinpanai_test_report.log'.
Mantém estritamente apenas o último log de testes na pasta logs/.
"""

import sys
import os

# Garantir que o diretório raiz está no PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configurar stdout para UTF-8 se possível
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.utils.test_runner import run_automated_tests, TEST_LOG_PATH

def main():
    verbosity = 2
    if "--quiet" in sys.argv or "-q" in sys.argv:
        verbosity = 1

    res = run_automated_tests(test_dir="tests", log_file=TEST_LOG_PATH, verbosity=verbosity)
    
    if not res["success"]:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
