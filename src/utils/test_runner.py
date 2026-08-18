"""
Módulo de Execução e Relatório Detalhado de Testes Automatizados do ShinpanAI.
Executa a suíte de testes unitários/integração, registra a descrição detalhada de cada
teste executado e mantém estritamente o último log de testes na pasta de logs.
"""

import os
import sys
import time
import glob
import unittest
import datetime
import platform
import traceback
from typing import Dict, Any, List, Optional

from src.utils.logger_manager import log_event

TEST_LOG_PATH = "logs/shinpanai_test_report.log"

def cleanup_old_test_logs(logs_dir: str = "logs", current_log_name: str = "shinpanai_test_report.log") -> None:
    """
    Remove logs de testes antigos da pasta de logs, garantindo que apenas
    o último log de testes seja mantido na pasta.
    """
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
        return

    # Buscar padrões de logs de testes existentes
    patterns = [
        os.path.join(logs_dir, "test_*.log"),
        os.path.join(logs_dir, "shinpanai_test_*.log"),
        os.path.join(logs_dir, "*test_report*.log"),
    ]

    target_full_path = os.path.abspath(os.path.join(logs_dir, current_log_name))

    for pat in patterns:
        for fpath in glob.glob(pat):
            abs_p = os.path.abspath(fpath)
            # Remove arquivos antigos que não sejam o arquivo de destino atual
            if abs_p != target_full_path and os.path.isfile(abs_p):
                try:
                    os.remove(abs_p)
                except Exception:
                    pass

def get_test_description(test: unittest.TestCase) -> str:
    """
    Obtém uma descrição clara e humanamente legível do teste a partir de sua docstring
    ou infere a partir do nome do método de teste.
    """
    doc = getattr(test, "_testMethodDoc", None)
    if doc:
        # Usar a primeira linha não vazia da docstring
        lines = [line.strip() for line in doc.strip().split("\n") if line.strip()]
        if lines:
            return lines[0]

    method_name = getattr(test, "_testMethodName", str(test))
    # Inferir descrição amigável a partir do nome do método
    clean_name = method_name.replace("test_", "").replace("_", " ").strip()
    return f"Valida {clean_name}"

class ShinpanAITestResult(unittest.TestResult):
    """
    Coletor de resultados customizado que grava métricas e descrições
    individuais de cada teste executado.
    """
    def __init__(self, stream=None, descriptions=True, verbosity=1):
        super().__init__(stream, descriptions, verbosity)
        self.stream = stream or sys.stdout
        self.test_records: List[Dict[str, Any]] = []
        self._current_test_start_time = 0.0
        self.verbosity = verbosity

    def startTest(self, test: unittest.TestCase):
        super().startTest(test)
        self._current_test_start_time = time.perf_counter()

    def addSuccess(self, test: unittest.TestCase):
        super().addSuccess(test)
        duration = time.perf_counter() - self._current_test_start_time
        self._record_test(test, "PASS", duration)

    def addFailure(self, test: unittest.TestCase, err):
        super().addFailure(test, err)
        duration = time.perf_counter() - self._current_test_start_time
        err_msg = self._format_error(err)
        self._record_test(test, "FAIL", duration, err_msg)

    def addError(self, test: unittest.TestCase, err):
        super().addError(test, err)
        duration = time.perf_counter() - self._current_test_start_time
        err_msg = self._format_error(err)
        self._record_test(test, "ERROR", duration, err_msg)

    def addSkip(self, test: unittest.TestCase, reason: str):
        super().addSkip(test, reason)
        duration = time.perf_counter() - self._current_test_start_time
        self._record_test(test, "SKIP", duration, details=reason)

    def _record_test(self, test: unittest.TestCase, status: str, duration: float, details: Optional[str] = None):
        cls = test.__class__
        method_name = getattr(test, "_testMethodName", str(test))
        desc = get_test_description(test)

        rec = {
            "index": len(self.test_records) + 1,
            "module": cls.__module__,
            "class_name": cls.__name__,
            "method_name": method_name,
            "full_name": f"{cls.__name__}.{method_name}",
            "description": desc,
            "status": status,
            "duration_seconds": duration,
            "details": details
        }
        self.test_records.append(rec)

        if self.verbosity >= 2:
            status_icon = "[OK]" if status == "PASS" else ("[FAIL]" if status == "FAIL" else "[ERR]")
            try:
                print(f"  {rec['index']:02d}. {status_icon} {rec['full_name']} ({duration:.3f}s) - {desc}")
            except UnicodeEncodeError:
                safe_desc = desc.encode("ascii", "replace").decode("ascii")
                print(f"  {rec['index']:02d}. {status_icon} {rec['full_name']} ({duration:.3f}s) - {safe_desc}")

    def _format_error(self, err) -> str:
        exctype, value, tb = err
        return "".join(traceback.format_exception(exctype, value, tb))

class ShinpanAITestRunner:
    """
    TestRunner customizado que orquestra a execução da suíte,
    gera o log detalhado e mantém o histórico na pasta logs/.
    """
    def __init__(self, verbosity: int = 2, log_file: str = TEST_LOG_PATH, stream=sys.stdout):
        self.verbosity = verbosity
        self.log_file = log_file
        self.stream = stream

    def run(self, test_suite: unittest.TestSuite) -> Dict[str, Any]:
        logs_dir = os.path.dirname(self.log_file) or "logs"
        os.makedirs(logs_dir, exist_ok=True)

        # 1. Limpar logs de testes antigos garantindo apenas o mais recente
        cleanup_old_test_logs(logs_dir=logs_dir, current_log_name=os.path.basename(self.log_file))

        start_time_real = datetime.datetime.now()
        start_clock = time.perf_counter()

        if self.verbosity >= 1:
            print("=" * 80)
            print(f"  SHINPANAI - SUITE DE TESTES AUTOMATIZADOS")
            print(f"  Inicio: {start_time_real.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

        result = ShinpanAITestResult(stream=self.stream, verbosity=self.verbosity)
        test_suite.run(result)

        total_duration = time.perf_counter() - start_clock
        end_time_real = datetime.datetime.now()

        total_tests = len(result.test_records)
        passed_tests = sum(1 for r in result.test_records if r["status"] == "PASS")
        failed_tests = len(result.failures)
        error_tests = len(result.errors)
        skipped_tests = len(result.skipped)
        is_success = (failed_tests == 0 and error_tests == 0)

        # 2. Gerar relatório textual detalhado
        report_text = self._build_report_text(
            start_time=start_time_real,
            end_time=end_time_real,
            total_duration=total_duration,
            records=result.test_records,
            total=total_tests,
            passed=passed_tests,
            failed=failed_tests,
            errors=error_tests,
            skipped=skipped_tests,
            is_success=is_success
        )

        # 3. Gravar no arquivo de log definitivo (sobrescrevendo o anterior)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        # 4. Registrar evento no log central do sistema
        log_event(
            level="INFO" if is_success else "ERROR",
            message=f"Suíte de Testes Automatizados finalizada: {passed_tests}/{total_tests} testes aprovados em {total_duration:.2f}s. Log salvo em '{self.log_file}'.",
            module_name="test_runner"
        )

        if self.verbosity >= 1:
            print("-" * 80)
            print(f"  Resultados: {passed_tests}/{total_tests} aprovados | Duracao: {total_duration:.2f}s")
            print(f"  Log detalhado salvo em: {self.log_file}")
            print("=" * 80)

        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "errors": error_tests,
            "skipped": skipped_tests,
            "success": is_success,
            "duration_seconds": total_duration,
            "log_file": self.log_file,
            "timestamp": end_time_real.strftime("%Y-%m-%d %H:%M:%S"),
            "records": result.test_records,
            "report_text": report_text
        }

    def _build_report_text(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        total_duration: float,
        records: List[Dict[str, Any]],
        total: int,
        passed: int,
        failed: int,
        errors: int,
        skipped: int,
        is_success: bool
    ) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("             SHINPANAI (審判 AI) - RELATÓRIO DE TESTES AUTOMATIZADOS")
        lines.append("=" * 80)
        lines.append(f"Data e Hora de Início : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Data e Hora de Término: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Versão do Sistema     : v1.6.0")
        lines.append(f"Ambiente Python       : {sys.version.split()[0]} ({platform.platform()})")
        lines.append(f"Processador / HW      : {platform.processor() or 'N/A'}")
        lines.append(f"Arquivo de Log        : {self.log_file}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("DETALHAMENTO DE CADA TESTE EXECUTADO:")
        lines.append("-" * 80)

        current_module = ""
        for rec in records:
            if rec["module"] != current_module:
                current_module = rec["module"]
                lines.append("")
                lines.append(f"📦 MÓDULO: {current_module}")
                lines.append("-" * 80)

            status_str = f"[{rec['status']}]"
            lines.append(f"{rec['index']:02d}. {status_str:<7} {rec['full_name']}")
            lines.append(f"    Descrição : {rec['description']}")
            lines.append(f"    Duração   : {rec['duration_seconds']:.4f} segundos")

            if rec.get("details"):
                lines.append("    Detalhes  :")
                for d_line in rec["details"].strip().split("\n"):
                    lines.append(f"      {d_line}")
            lines.append("-" * 40)

        lines.append("")
        lines.append("=" * 80)
        lines.append("                             RESUMO FINAL DA EXECUÇÃO")
        lines.append("=" * 80)
        lines.append(f"Total de Testes Executados : {total}")
        lines.append(f"Aprovados (PASS)           : {passed}")
        lines.append(f"Falhas (FAIL)              : {failed}")
        lines.append(f"Erros (ERROR)              : {errors}")
        lines.append(f"Pulados (SKIP)             : {skipped}")
        success_pct = (passed / total * 100.0) if total > 0 else 0.0
        lines.append(f"Taxa de Sucesso            : {success_pct:.1f}%")
        lines.append(f"Tempo Total de Execução    : {total_duration:.3f} segundos")
        lines.append(f"Resultado Geral            : {'✅ APROVADO COM SUCESSO' if is_success else '❌ FALHA DETECTADA'}")
        lines.append("=" * 80)

        return "\n".join(lines)

def run_automated_tests(test_dir: str = "tests", log_file: str = TEST_LOG_PATH, verbosity: int = 2) -> Dict[str, Any]:
    """
    Função utilitária de alto nível para descobrir e executar todos os testes da pasta tests/,
    gerar o log detalhado e retornar o sumário estruturado.
    """
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")
    runner = ShinpanAITestRunner(verbosity=verbosity, log_file=log_file)
    return runner.run(suite)

def get_latest_test_report_content(log_file: str = TEST_LOG_PATH) -> str:
    """
    Retorna o conteúdo textual do último log de testes salvo para download e exibição na UI.
    """
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as ex:
            return f"Erro ao ler log de testes: {ex}"
    return "Nenhum relatório de testes automatizados encontrado. Execute a suíte de testes para gerar o log."

# Aliases de compatibilidade retroativa
ShinpanaiTestResult = ShinpanAITestResult
ShinpanaiTestRunner = ShinpanAITestRunner
