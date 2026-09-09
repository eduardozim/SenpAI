"""
Testes unitários para o módulo de controles avançados de vídeo (video_player_controls.py).
"""

import pytest
from src.utils.video_player_controls import (
    _parse_ts_str,
    generate_video_playback_controls_html
)


def test_parse_ts_str():
    """Valida a conversão de timestamps para segundos em float."""
    assert _parse_ts_str("00:00.000") == 0.0
    assert _parse_ts_str("00:02.500") == 2.5
    assert _parse_ts_str("01:30.000") == 90.0
    assert _parse_ts_str("01:02:03.000") == 3723.0
    assert _parse_ts_str("2.5s") == 2.5
    assert _parse_ts_str("10") == 10.0
    assert _parse_ts_str("") == 0.0
    assert _parse_ts_str(None) == 0.0
    assert _parse_ts_str("invalid_ts") == 0.0


def test_generate_video_playback_controls_html_basic():
    """Valida a geração básica de HTML e presença dos controles principais."""
    html = generate_video_playback_controls_html()
    
    assert "senpai-video-ctrl-panel" in html
    # Botões de avanço e retrocesso de 0.5s solicitados pelo usuário
    assert "-0.5s" in html
    assert "+0.5s" in html
    assert "⏮️ Frame" in html
    assert "Frame ⏭️" in html
    assert "-1s" in html
    assert "+1s" in html
    assert "-5s" in html
    assert "+5s" in html
    # Câmera lenta (VAR)
    assert "0.1x" in html
    assert "0.25x" in html
    assert "0.5x" in html
    assert "1.0x" in html
    # Atalhos de teclado
    assert "ArrowLeft" in html
    assert "ArrowRight" in html
    assert "playbackRate" in html


def test_generate_video_playback_controls_html_with_events_and_sonkyo():
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
    
    assert "Sonkyō Inicial" in html
    assert "Sonkyō Final" in html
    assert "MEN" in html
    assert "KOTE" in html
    assert "00:05.400" in html


def test_render_video_playback_controls(monkeypatch):
    """Valida a chamada de renderização através de components.html."""
    from unittest.mock import MagicMock
    from src.utils.video_player_controls import render_video_playback_controls
    import streamlit.components.v1 as components

    mock_html = MagicMock()
    monkeypatch.setattr(components, "html", mock_html)

    render_video_playback_controls(
        events=[{"event_info": {"type": "MEN", "timestamp": "00:02.000"}}],
        default_fps=30.0,
        target_start_time=12.5
    )

    mock_html.assert_called_once()
    args, kwargs = mock_html.call_args
    assert "senpai-video-ctrl-panel" in args[0]
    assert "const targetStartTime = 12.5;" in args[0]
    assert kwargs.get("scrolling") is False
    assert kwargs.get("height", 0) > 0
