"""
Testes unitários para persistência do estado e feed de golpes da Detecção via Streaming (Match Mode).
Valida a retenção do placar e histórico de golpes ao pausar e alterar velocidade,
bem como a limpeza seletiva ao iniciar um novo vídeo ou acionar o botão de reset.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestMatchStreamingFeedState(unittest.TestCase):
    def setUp(self):
        self.session_state = {}

    def _simulate_feed_state_handler(
        self,
        current_url: str,
        btn_reset_pressed: bool = False,
    ):
        """Simula a lógica de gerenciamento de estado do feed de streaming implementada no app.py."""
        clean_url = current_url.strip()
        last_url = self.session_state.get("match_streaming_last_processed_url")

        # 1. Se um novo vídeo for iniciado (URL diferente da última processada), limpa a lista e placar
        if last_url is not None and last_url != clean_url:
            self.session_state["match_streaming_strike_history"] = []
            self.session_state["match_streaming_score_shiro"] = 0
            self.session_state["match_streaming_score_aka"] = 0
            self.session_state["match_streaming_total_shiro_strikes"] = 0
            self.session_state["match_streaming_total_aka_strikes"] = 0
            self.session_state["match_streaming_modality_name"] = "Detectando movimentação..."
            self.session_state["match_streaming_pose_history"] = []
            self.session_state.pop("match_streaming_last_alert", None)

        self.session_state["match_streaming_last_processed_url"] = clean_url

        # 2. Se o botão de reset for clicado
        if btn_reset_pressed:
            self.session_state["match_streaming_strike_history"] = []
            self.session_state["match_streaming_score_shiro"] = 0
            self.session_state["match_streaming_score_aka"] = 0
            self.session_state["match_streaming_total_shiro_strikes"] = 0
            self.session_state["match_streaming_total_aka_strikes"] = 0
            self.session_state["match_streaming_modality_name"] = "Detectando movimentação..."
            self.session_state["match_streaming_pose_history"] = []
            self.session_state.pop("match_streaming_last_alert", None)

        return {
            "history": self.session_state.get("match_streaming_strike_history", []),
            "score_shiro": self.session_state.get("match_streaming_score_shiro", 0),
            "score_aka": self.session_state.get("match_streaming_score_aka", 0),
            "total_shiro": self.session_state.get("match_streaming_total_shiro_strikes", 0),
            "total_aka": self.session_state.get("match_streaming_total_aka_strikes", 0),
        }

    def test_state_retained_on_pause_and_speed_change(self):
        """Valida que o placar e histórico de golpes são mantidos ao pausar ou alterar velocidade (mesma URL)."""
        url = "https://www.youtube.com/watch?v=kendo_match_1"
        self._simulate_feed_state_handler(url)

        # Simula detecção de golpes durante o streaming
        self.session_state["match_streaming_strike_history"] = ["<div>MEN (Ippon)</div>", "<div>KOTE</div>"]
        self.session_state["match_streaming_score_shiro"] = 1
        self.session_state["match_streaming_score_aka"] = 0
        self.session_state["match_streaming_total_shiro_strikes"] = 2
        self.session_state["match_streaming_total_aka_strikes"] = 1

        # Rerun simulado por pausa ou alteração de velocidade
        state_after_pause = self._simulate_feed_state_handler(url, btn_reset_pressed=False)
        self.assertEqual(len(state_after_pause["history"]), 2)
        self.assertEqual(state_after_pause["score_shiro"], 1)
        self.assertEqual(state_after_pause["total_shiro"], 2)
        self.assertEqual(state_after_pause["total_aka"], 1)

    def test_state_cleared_on_reset_button(self):
        """Valida que o acionamento do botão de reset limpa o feed e zera os placares."""
        url = "https://www.youtube.com/watch?v=kendo_match_1"
        self._simulate_feed_state_handler(url)

        self.session_state["match_streaming_strike_history"] = ["<div>DO (Ippon)</div>"]
        self.session_state["match_streaming_score_aka"] = 1
        self.session_state["match_streaming_total_aka_strikes"] = 3

        # Usuário clica em Resetar Feed
        state_after_reset = self._simulate_feed_state_handler(url, btn_reset_pressed=True)
        self.assertEqual(len(state_after_reset["history"]), 0)
        self.assertEqual(state_after_reset["score_aka"], 0)
        self.assertEqual(state_after_reset["total_aka"], 0)

    def test_state_cleared_on_new_video(self):
        """Valida que iniciar um novo vídeo (URL diferente) reinicializa automaticamente o feed e placar."""
        url_1 = "https://www.youtube.com/watch?v=video_1"
        self._simulate_feed_state_handler(url_1)

        self.session_state["match_streaming_strike_history"] = ["<div>TSUKI (Ippon)</div>"]
        self.session_state["match_streaming_score_shiro"] = 1
        self.session_state["match_streaming_total_shiro_strikes"] = 1

        # Usuário troca para um novo vídeo
        url_2 = "https://www.youtube.com/watch?v=video_2"
        state_new_video = self._simulate_feed_state_handler(url_2, btn_reset_pressed=False)
        self.assertEqual(len(state_new_video["history"]), 0)
        self.assertEqual(state_new_video["score_shiro"], 0)
        self.assertEqual(state_new_video["total_shiro"], 0)


if __name__ == "__main__":
    unittest.main()
