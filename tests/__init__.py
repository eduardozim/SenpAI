"""
Pacote de testes do SenpAI.

Liga o modo de teste assim que qualquer teste é importado: caches, filas, banco de rótulos, golden e
modelos vão para a sandbox isolada, nunca para data/ (um `unittest discover` sem SENPAI_TEST_MODE=1
já apagou o cache real de poses).
"""

from src.utils.test_isolation import enable_test_mode

enable_test_mode()
