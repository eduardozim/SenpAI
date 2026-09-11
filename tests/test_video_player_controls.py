"""
Testes unitários para o módulo de controles avançados de vídeo (video_player_controls.py).
Compatível com unittest e pytest.
"""

import unittest
from unittest.mock import MagicMock, patch

from src.utils.video_player_controls import (
    _parse_ts_str,
    generate_video_playback_controls_html,
    render_video_playback_controls
)
import streamlit.components.v1 as components


class TestVideoPlayerControls(unittest.TestCase):
    """Conjunto de testes para o módulo video_player_controls."""

    def test_parse_ts_str(self):
        """Valida a conversão de timestamps para segundos em float."""
        self.assertEqual(_parse_ts_str("00:00.000"), 0.0)
        self.assertEqual(_parse_ts_str("00:02.500"), 2.5)
        self.assertEqual(_parse_ts_str("01:30.000"), 90.0)
        self.assertEqual(_parse_ts_str("01:02:03.000"), 3723.0)
        self.assertEqual(_parse_ts_str("2.5s"), 2.5)
        self.assertEqual(_parse_ts_str("10"), 10.0)
        self.assertEqual(_parse_ts_str(""), 0.0)
        self.assertEqual(_parse_ts_str(None), 0.0)
        self.assertEqual(_parse_ts_str("invalid_ts"), 0.0)

    def test_generate_video_playback_controls_html_basic(self):
        """Valida a geração básica de HTML e presença dos controles principais."""
        html = generate_video_playback_controls_html()
        
        self.assertIn("senpai-video-ctrl-panel", html)
        # Botões de avanço e retrocesso de 0.5s solicitados pelo usuário
        self.assertIn("-0.5s", html)
        self.assertIn("+0.5s", html)
        self.assertIn("⏮️ Frame", html)
        self.assertIn("Frame ⏭️", html)
        self.assertIn("-1s", html)
        self.assertIn("+1s", html)
        self.assertIn("-5s", html)
        self.assertIn("+5s", html)
        # Câmera lenta (VAR)
        self.assertIn("0.1x", html)
        self.assertIn("0.25x", html)
        self.assertIn("0.5x", html)
        self.assertIn("1.0x", html)
        # Atalhos de teclado
        self.assertIn("ArrowLeft", html)
        self.assertIn("ArrowRight", html)
        self.assertIn("playbackRate", html)

    def test_generate_video_playback_controls_html_with_events_and_sonkyo(self):
        """Valida a inclusão correta de marcadores de golpes e Sonkyo nos controles."""
        events = [
            {
                "event_info": {"type": "MEN", "timestamp": "00:05.400", "impact_frame": 162},
                "evaluation": {"is_valid": True}
            },
            {
                "event_info": {"type": "KOTE", "timestamp": "00:12.100", "impact_frame": 363},
                "evaluation": {"is_valid": False}
            }
        ]
        sonkyo_info = {
            "has_initial_sonkyo": True,
            "initial_sonkyo": {"start_timestamp": "00:01.200"},
            "has_final_sonkyo": True,
            "final_sonkyo": {"start_timestamp": "00:25.000"}
        }
        
        html = generate_video_playback_controls_html(
            events=events,
            sonkyo_info=sonkyo_info,
            default_fps=30.0
        )
        
        self.assertIn("Sonkyō Inicial", html)
        self.assertIn("Sonkyō Final", html)
        self.assertIn("MEN", html)
        self.assertIn("KOTE", html)
        self.assertIn("00:05.400", html)

    @patch("streamlit.iframe")
    def test_render_video_playback_controls_iframe(self, mock_iframe):
        """Valida a chamada de renderização através de st.iframe."""
        render_video_playback_controls(
            events=[{"event_info": {"type": "MEN", "timestamp": "00:02.000"}}],
            default_fps=30.0,
            target_start_time=12.5
        )

        mock_iframe.assert_called_once()
        args, kwargs = mock_iframe.call_args
        self.assertIn("senpai-video-ctrl-panel", args[0])
        self.assertIn("const targetStartTime = 12.5;", args[0])
        self.assertGreater(kwargs.get("height", 0), 0)

    def test_render_video_playback_controls_fallback(self):
        """Valida o fallback para components.html caso st.iframe não esteja disponível."""
        import streamlit as st
        original_iframe = getattr(st, "iframe", None)
        try:
            if hasattr(st, "iframe"):
                delattr(st, "iframe")
            with patch("streamlit.components.v1.html") as mock_html:
                render_video_playback_controls(
                    events=[{"event_info": {"type": "MEN", "timestamp": "00:02.000"}}],
                    default_fps=30.0,
                    target_start_time=12.5
                )
                mock_html.assert_called_once()
                args, kwargs = mock_html.call_args
                self.assertIn("senpai-video-ctrl-panel", args[0])
                self.assertFalse(kwargs.get("scrolling"))
        finally:
            if original_iframe is not None:
                st.iframe = original_iframe


if __name__ == "__main__":
    unittest.main()
