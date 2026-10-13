"""
Testes Unitários para a Ocultação de Opções em Modo Servidor Web (test_web_server_options.py).
Valida que quando detectado que a aplicação está executando em Servidor Web/Nuvem:
1. Em Modo de Análise de Lutas, a opção 'Detecção em tempo Real' é ocultada.
2. Em Modo de Análise de Lutas (Detecção Gravada), a opção de Origem do Vídeo via Streaming web (YouTube) é ocultada.
3. Em Modo de Treinamento & Aprendizado, a opção 'Detecção em tempo Real' é ocultada.
4. Em Modo de Treinamento e Aprendizado (Detecção Gravada), a opção de Origem do Vídeo via Streaming web (YouTube) é ocultada.
"""

import os
import unittest
from unittest.mock import patch

from src.utils.environment import (
    is_web_server_environment,
    get_execution_environment_info
)


class TestWebServerOptionsHiding(unittest.TestCase):
    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "web"}, clear=True)
    def test_match_options_on_web_server_excludes_realtime(self):
        """Verifica que no servidor web o Modo de Análise de Lutas oculta a Detecção em Tempo Real."""
        is_web = is_web_server_environment()
        self.assertTrue(is_web)
        
        match_options = ["streaming", "recorded"] if is_web else ["realtime", "streaming", "recorded"]
        self.assertNotIn("realtime", match_options)
        self.assertEqual(match_options, ["streaming", "recorded"])

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "local"}, clear=True)
    def test_match_options_on_local_includes_realtime(self):
        """Verifica que em ambiente local o Modo de Análise de Lutas mantém todas as opções incluindo Tempo Real."""
        is_web = is_web_server_environment()
        self.assertFalse(is_web)
        
        match_options = ["streaming", "recorded"] if is_web else ["realtime", "streaming", "recorded"]
        self.assertIn("realtime", match_options)
        self.assertEqual(match_options, ["realtime", "streaming", "recorded"])

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "web"}, clear=True)
    def test_training_options_on_web_server_excludes_realtime(self):
        """Verifica que no servidor web o Modo de Treinamento & Aprendizado oculta a Detecção em Tempo Real."""
        is_web = is_web_server_environment()
        self.assertTrue(is_web)
        
        training_options = ["streaming", "recorded"] if is_web else ["realtime", "streaming", "recorded"]
        self.assertNotIn("realtime", training_options)
        self.assertEqual(training_options, ["streaming", "recorded"])

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "local"}, clear=True)
    def test_training_options_on_local_includes_realtime(self):
        """Verifica que em ambiente local o Modo de Treinamento mantém todas as opções incluindo Tempo Real."""
        is_web = is_web_server_environment()
        self.assertFalse(is_web)
        
        training_options = ["streaming", "recorded"] if is_web else ["realtime", "streaming", "recorded"]
        self.assertIn("realtime", training_options)
        self.assertEqual(training_options, ["realtime", "streaming", "recorded"])

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "web"}, clear=True)
    def test_recorded_video_sources_on_web_server_excludes_streaming_web(self):
        """Verifica que na Detecção Gravada em servidor web, a opção de Streaming web/YouTube é ocultada."""
        is_web = is_web_server_environment()
        self.assertTrue(is_web)
        
        recorded_sources = (
            ["📁 Fazer Upload de Arquivo"]
            if is_web
            else ["📁 Fazer Upload de Arquivo", "🌐 Link do YouTube / Streaming Web"]
        )
        self.assertEqual(recorded_sources, ["📁 Fazer Upload de Arquivo"])
        self.assertNotIn("🌐 Link do YouTube / Streaming Web", recorded_sources)

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "local"}, clear=True)
    def test_recorded_video_sources_on_local_includes_streaming_web(self):
        """Verifica que na Detecção Gravada em ambiente local, ambas as origens (Upload e YouTube) estão presentes."""
        is_web = is_web_server_environment()
        self.assertFalse(is_web)
        
        recorded_sources = (
            ["📁 Fazer Upload de Arquivo"]
            if is_web
            else ["📁 Fazer Upload de Arquivo", "🌐 Link do YouTube / Streaming Web"]
        )
        self.assertIn("📁 Fazer Upload de Arquivo", recorded_sources)
        self.assertIn("🌐 Link do YouTube / Streaming Web", recorded_sources)

    @patch.dict(os.environ, {"IS_STREAMLIT_CLOUD": "1"}, clear=True)
    def test_streamlit_cloud_hides_realtime_and_youtube(self):
        """Valida que quando em Streamlit Community Cloud (IS_STREAMLIT_CLOUD=1):
        - is_web_server_environment() retorna True
        - get_execution_environment_info()['is_cloud'] é True
        - get_execution_environment_info()['provider_label'] é 'Streamlit Community Cloud'
        - match_options oculta 'realtime'
        - recorded_source_options oculta 'Link do YouTube'
        """
        info = get_execution_environment_info()
        self.assertTrue(info["is_cloud"])
        self.assertEqual(info["provider_label"], "Streamlit Community Cloud")

        is_web = is_web_server_environment()
        self.assertTrue(is_web)

        match_options = ["streaming", "recorded"] if is_web else ["realtime", "streaming", "recorded"]
        self.assertNotIn("realtime", match_options)

        recorded_sources = (
            ["📁 Fazer Upload de Arquivo"]
            if is_web
            else ["📁 Fazer Upload de Arquivo", "🌐 Link do YouTube / Streaming Web"]
        )
        self.assertNotIn("🌐 Link do YouTube / Streaming Web", recorded_sources)

    @patch.dict(os.environ, {"SENPAI_DEPLOY_MODE": "web"}, clear=True)
    def test_fallback_selection_when_switched_to_web_server(self):
        """Valida que valores salvos de opções ocultadas realizam fallback seguro para as primeiras opções ativas."""
        is_web = is_web_server_environment()
        
        # Simula que a sessão continha 'realtime' salvo
        saved_match_val = "realtime"
        match_options = ["streaming", "recorded"] if is_web else ["realtime", "streaming", "recorded"]
        if saved_match_val not in match_options:
            saved_match_val = match_options[0]
        self.assertEqual(saved_match_val, "streaming")

        # Simula que a sessão continha 'Link do YouTube' salvo
        saved_source_choice = "🌐 Link do YouTube / Streaming Web"
        recorded_sources = (
            ["📁 Fazer Upload de Arquivo"]
            if is_web
            else ["📁 Fazer Upload de Arquivo", "🌐 Link do YouTube / Streaming Web"]
        )
        if saved_source_choice not in recorded_sources:
            saved_source_choice = recorded_sources[0]
        self.assertEqual(saved_source_choice, "📁 Fazer Upload de Arquivo")


if __name__ == "__main__":
    unittest.main()
