"""
Módulo de Controles Interativos de Reprodução de Vídeo para o SenpAI.
Permite controle avançado de reprodução (câmera lenta, passo a passo por quadro,
avanço/retrocesso de 0.5s, atalhos de teclado e marcadores rápidos de golpes)
diretamente no cliente (DOM) sem latência de recarregamento do Streamlit.
"""

import json
from typing import Any, Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components


def generate_video_playback_controls_html(
    events: Optional[List[Dict[str, Any]]] = None,
    sonkyo_info: Optional[Dict[str, Any]] = None,
    sonkyo_edits: Optional[Dict[str, Any]] = None,
    default_fps: float = 30.0,
    container_id: str = "senpai-video-controller",
    target_start_time: float = 0.0
) -> str:
    """
    Gera o código HTML/CSS/JavaScript do painel de controle interativo de vídeo.
    
    Args:
        events: Lista de eventos/golpes detectados pela análise.
        sonkyo_info: Dicionário com informações do Sonkyo detectado.
        sonkyo_edits: Dicionário com eventuais edições manuais de Sonkyo.
        default_fps: Taxa estimada de quadros por segundo do vídeo.
        container_id: Identificador único do container DOM.
        
    Returns:
        String contendo bloco HTML com estilos e scripts prontos para st.iframe().
    """
    # 1. Preparação dos marcadores rápidos de eventos para os Chips de Navegação
    jump_markers = []
    
    # Sonkyo Inicial
    has_init = (sonkyo_info and sonkyo_info.get("has_initial_sonkyo") and sonkyo_info.get("initial_sonkyo")) or (sonkyo_edits and sonkyo_edits.get("initial"))
    if has_init:
        init_data = {}
        if sonkyo_edits and isinstance(sonkyo_edits.get("initial"), dict):
            init_data = sonkyo_edits["initial"]
        elif sonkyo_info and isinstance(sonkyo_info.get("initial_sonkyo"), dict):
            init_data = sonkyo_info["initial_sonkyo"]
        ts_i = init_data.get("start_timestamp", "00:00.000")
        jump_markers.append({
            "type": "sonkyo_init",
            "label": f"🥋 Sonkyō Inicial ({ts_i})",
            "time_str": ts_i,
            "seconds": max(0.0, _parse_ts_str(ts_i) - 1.0),
            "color": "#6366F1",
            "border": "#818CF8",
            "icon": "🥋"
        })

    # Golpes detectados / revisados
    if events:
        for idx, ev_item in enumerate(events):
            # Suporte a formatos diferentes de evento (bruto do pipeline ou formatado da revisão)
            ev_info = ev_item.get("event_info", ev_item)
            eval_info = ev_item.get("evaluation", {})
            review_info = ev_item.get("review", {})
            
            strike_type = review_info.get("strike_type", ev_info.get("type", "GOLPE"))
            ts_strike = review_info.get("timestamp", ev_info.get("timestamp", "00:00.000"))
            is_valid = review_info.get("is_valid_ippon", eval_info.get("is_valid", False))
            
            # Formatação de cores do badge
            if is_valid:
                badge_color = "#10B981"  # Verde esmeralda (Ippon)
                border_color = "#34D399"
                icon = "⭐"
            else:
                badge_color = "#EF4444"  # Vermelho suave (Inválido)
                border_color = "#F87171"
                icon = "🥊"
                
            sec_val = max(0.0, _parse_ts_str(ts_strike) - 0.5)
            jump_markers.append({
                "type": "strike",
                "label": f"{icon} #{idx+1} {strike_type} ({ts_strike})",
                "time_str": ts_strike,
                "seconds": sec_val,
                "color": badge_color,
                "border": border_color,
                "icon": icon
            })
            
    # Sonkyo Final
    has_final = (sonkyo_info and sonkyo_info.get("has_final_sonkyo") and sonkyo_info.get("final_sonkyo")) or (sonkyo_edits and sonkyo_edits.get("final"))
    if has_final:
        fin_data = {}
        if sonkyo_edits and isinstance(sonkyo_edits.get("final"), dict):
            fin_data = sonkyo_edits["final"]
        elif sonkyo_info and isinstance(sonkyo_info.get("final_sonkyo"), dict):
            fin_data = sonkyo_info["final_sonkyo"]
        ts_f = fin_data.get("start_timestamp", "00:00.000")
        jump_markers.append({
            "type": "sonkyo_final",
            "label": f"🥋 Sonkyō Final ({ts_f})",
            "time_str": ts_f,
            "seconds": max(0.0, _parse_ts_str(ts_f) - 1.0),
            "color": "#8B5CF6",
            "border": "#A78BFA",
            "icon": "🥋"
        })

    # Serialização segura para o JavaScript
    markers_json = json.dumps(jump_markers, ensure_ascii=False)
    
    html_content = f"""
    <div id="{container_id}" class="senpai-video-ctrl-panel">
        <style>
            html, body {{
                margin: 0 !important;
                padding: 0 !important;
                background: transparent !important;
                overflow-x: hidden !important;
                overflow-y: hidden !important;
                box-sizing: border-box !important;
            }}
            *, *:before, *:after {{
                box-sizing: inherit !important;
            }}
            #{container_id} {{
                background: linear-gradient(145deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.90));
                border: 1px solid rgba(99, 102, 241, 0.35);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-radius: 12px;
                padding: 10px 14px;
                margin: 2px 0 4px 0;
                color: #F1F5F9;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                user-select: none;
            }}
            #{container_id} .ctrl-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                padding-bottom: 6px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            }}
            #{container_id} .ctrl-title {{
                font-size: 0.85rem;
                font-weight: 700;
                color: #A5B4FC;
                display: flex;
                align-items: center;
                gap: 6px;
            }}
            #{container_id} .ctrl-hud {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-family: 'Courier New', Courier, monospace;
                font-size: 0.82rem;
                background: rgba(0, 0, 0, 0.45);
                padding: 3px 8px;
                border-radius: 6px;
                border: 1px solid rgba(99, 102, 241, 0.25);
            }}
            #{container_id} .hud-time {{
                color: #38BDF8;
                font-weight: 700;
            }}
            #{container_id} .hud-frame {{
                color: #94A3B8;
                font-size: 0.75rem;
            }}
            #{container_id} .btn-copy-ts {{
                background: rgba(56, 189, 248, 0.15);
                border: 1px solid rgba(56, 189, 248, 0.4);
                color: #38BDF8;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 0.72rem;
                cursor: pointer;
                transition: all 0.15s ease;
            }}
            #{container_id} .btn-copy-ts:hover {{
                background: rgba(56, 189, 248, 0.35);
                color: #FFFFFF;
            }}
            #{container_id} .ctrl-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 6px;
                margin-bottom: 8px;
            }}
            #{container_id} .step-group {{
                display: flex;
                align-items: center;
                gap: 4px;
                flex-wrap: wrap;
            }}
            #{container_id} .ctrl-btn {{
                background: rgba(30, 41, 59, 0.85);
                border: 1px solid rgba(148, 163, 184, 0.25);
                color: #E2E8F0;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 0.76rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 28px;
            }}
            #{container_id} .ctrl-btn:hover {{
                background: rgba(99, 102, 241, 0.25);
                border-color: #818CF8;
                color: #FFFFFF;
                transform: translateY(-1px);
            }}
            #{container_id} .ctrl-btn:active {{
                transform: translateY(0);
                background: rgba(99, 102, 241, 0.45);
            }}
            #{container_id} .ctrl-btn.btn-play-pause {{
                background: linear-gradient(135deg, #4F46E5, #3730A3);
                border-color: #6366F1;
                color: #FFFFFF;
                padding: 4px 12px;
                font-size: 0.82rem;
            }}
            #{container_id} .ctrl-btn.btn-play-pause:hover {{
                background: linear-gradient(135deg, #6366F1, #4338CA);
                box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
            }}
            #{container_id} .ctrl-btn.btn-highlight {{
                border-color: #38BDF8;
                color: #BAE6FD;
                background: rgba(14, 165, 233, 0.15);
            }}
            #{container_id} .ctrl-btn.btn-highlight:hover {{
                background: rgba(14, 165, 233, 0.35);
                border-color: #7DD3FC;
            }}
            #{container_id} .speed-group {{
                display: flex;
                align-items: center;
                gap: 3px;
                background: rgba(15, 23, 42, 0.6);
                padding: 2px 4px;
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.08);
            }}
            #{container_id} .speed-label {{
                font-size: 0.70rem;
                color: #94A3B8;
                margin-right: 4px;
                font-weight: 600;
            }}
            #{container_id} .speed-btn {{
                background: transparent;
                border: 1px solid transparent;
                color: #94A3B8;
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 0.72rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.15s ease;
            }}
            #{container_id} .speed-btn:hover {{
                color: #FFFFFF;
                background: rgba(255, 255, 255, 0.1);
            }}
            #{container_id} .speed-btn.active {{
                background: #4F46E5;
                color: #FFFFFF;
                border-color: #818CF8;
                font-weight: 700;
            }}
            #{container_id} .markers-bar {{
                display: flex;
                align-items: center;
                gap: 5px;
                overflow-x: auto;
                padding-top: 6px;
                margin-top: 4px;
                border-top: 1px solid rgba(255, 255, 255, 0.06);
                scrollbar-width: thin;
                scrollbar-color: #475569 transparent;
            }}
            #{container_id} .marker-chip {{
                background: rgba(30, 41, 59, 0.90);
                border: 1px solid #475569;
                color: #CBD5E1;
                border-radius: 12px;
                padding: 2px 8px;
                font-size: 0.70rem;
                font-weight: 600;
                white-space: nowrap;
                cursor: pointer;
                transition: all 0.15s ease;
                display: inline-flex;
                align-items: center;
                gap: 4px;
            }}
            #{container_id} .marker-chip:hover {{
                transform: scale(1.04);
                color: #FFFFFF;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
            }}
            #{container_id} .shortcuts-hint {{
                font-size: 0.68rem;
                color: #64748B;
                margin-top: 6px;
                text-align: right;
            }}
            #{container_id} .shortcuts-hint kbd {{
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 3px;
                padding: 1px 4px;
                font-size: 0.65rem;
                color: #94A3B8;
            }}
        </style>

        <!-- Cabeçalho do HUD e tempo -->
        <div class="ctrl-header">
            <div class="ctrl-title">
                <span>⏱️ Controles de Análise de Golpes</span>
            </div>
            <div class="ctrl-hud">
                <span class="hud-time" id="{container_id}-time">00:00.000</span>
                <span style="color:#64748B;">/</span>
                <span style="color:#94A3B8;" id="{container_id}-duration">00:00.000</span>
                <span class="hud-frame" id="{container_id}-frame">Frame #0</span>
                <button class="btn-copy-ts" id="{container_id}-btn-copy" title="Copiar timestamp atual para colar nas anotações">📋 Copiar</button>
            </div>
        </div>

        <!-- Linha de Controles de Passo / Frame -->
        <div class="ctrl-row">
            <div class="step-group">
                <button class="ctrl-btn" id="{container_id}-step-b5" title="Retroceder 5 segundos (Shift + Seta Esquerda)">⏮️ -5s</button>
                <button class="ctrl-btn" id="{container_id}-step-b1" title="Retroceder 1 segundo">⏪ -1s</button>
                <button class="ctrl-btn btn-highlight" id="{container_id}-step-b05" title="Retroceder meio segundo (Seta Esquerda)">◀️ -0.5s</button>
                <button class="ctrl-btn" id="{container_id}-step-bframe" title="Retroceder 1 quadro (Tecla Vírgula <)">⏮️ Frame</button>
                
                <button class="ctrl-btn btn-play-pause" id="{container_id}-play-pause" title="Reproduzir / Pausar (Espaço)">⏯️ Play</button>
                
                <button class="ctrl-btn" id="{container_id}-step-fframe" title="Avançar 1 quadro (Tecla Ponto >)">Frame ⏭️</button>
                <button class="ctrl-btn btn-highlight" id="{container_id}-step-f05" title="Avançar meio segundo (Seta Direita)">+0.5s ▶️</button>
                <button class="ctrl-btn" id="{container_id}-step-f1" title="Avançar 1 segundo">+1s ⏩</button>
                <button class="ctrl-btn" id="{container_id}-step-f5" title="Avançar 5 segundos (Shift + Seta Direita)">+5s ⏭️</button>
            </div>

            <!-- Seletor de Câmera Lenta / Velocidade -->
            <div class="speed-group">
                <span class="speed-label">🐢 CÂMERA LENTA:</span>
                <button class="speed-btn" data-speed="0.1" title="Super Câmera Lenta (0.1x)">0.1x</button>
                <button class="speed-btn" data-speed="0.25" title="Câmera Lenta Clássica VAR (0.25x)">0.25x</button>
                <button class="speed-btn" data-speed="0.5" title="Meia velocidade (0.5x)">0.5x</button>
                <button class="speed-btn" data-speed="0.75" title="Moderada (0.75x)">0.75x</button>
                <button class="speed-btn active" data-speed="1.0" title="Velocidade Normal (1.0x)">1.0x</button>
                <button class="speed-btn" data-speed="1.5" title="Acelerado (1.5x)">1.5x</button>
                <button class="speed-btn" data-speed="2.0" title="Avanço Rápido (2.0x)">2.0x</button>
            </div>
        </div>

        <!-- Barra de Salto Rápido para Golpes e Sonkyo (se houver marcadores) -->
        <div id="{container_id}-markers-container" class="markers-bar" style="{'display: flex;' if jump_markers else 'display: none;'}">
            <span style="font-size: 0.70rem; color: #94A3B8; font-weight: 700; margin-right: 4px; flex-shrink: 0;">🎯 PULAR PARA:</span>
            <!-- Os chips de marcadores são inseridos dinamicamente ou populados pelo script -->
        </div>

        <!-- Legenda de Atalhos de Teclado -->
        <div class="shortcuts-hint">
            <span>⌨️ <b>Atalhos:</b> <kbd>Espaço</kbd> Play/Pause &nbsp;|&nbsp; <kbd>←</kbd> <kbd>→</kbd> ±0.5s &nbsp;|&nbsp; <kbd>&lt;</kbd> <kbd>&gt;</kbd> Quadro a Quadro &nbsp;|&nbsp; <kbd>[</kbd> <kbd>]</kbd> Velocidade &nbsp;|&nbsp; <kbd>0</kbd> Reiniciar</span>
        </div>
    </div>

    <script>
    (function() {{
        const containerId = "{container_id}";
        const markersData = {markers_json};
        const defaultFps = {default_fps};
        const targetStartTime = {target_start_time};
        let initialSeekApplied = false;
        let activeSpeed = 1.0;
        let lastKnownVideo = null;
        let pollTimer = null;

        function formatTimeMMSSmmm(seconds) {{
            if (isNaN(seconds) || seconds < 0) seconds = 0;
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            const millis = Math.floor((seconds % 1) * 1000);
            return String(mins).padStart(2, '0') + ':' +
                   String(secs).padStart(2, '0') + '.' +
                   String(millis).padStart(3, '0');
        }}

        function findVideoElement() {{
            // 1. Procura primeiro no documento principal do Streamlit (window.parent)
            try {{
                if (window.parent && window.parent.document) {{
                    const pVideos = window.parent.document.querySelectorAll('video');
                    if (pVideos && pVideos.length > 0) {{
                        for (let i = 0; i < pVideos.length; i++) {{
                            const v = pVideos[i];
                            if (v && (v.src || v.querySelector('source') || v.duration > 0 || v.readyState > 0)) {{
                                return v;
                            }}
                        }}
                        return pVideos[0];
                    }}
                }}
            }} catch (e) {{
                console.warn('[SenpAI Controls] Cross-frame parent access warning:', e);
            }}

            // 2. Fallback no documento local
            const allVideos = document.querySelectorAll('video');
            if (allVideos && allVideos.length > 0) {{
                return allVideos[0];
            }}
            return null;
        }}

        function getTargetVideo() {{
            const v = findVideoElement();
            if (v) {{
                lastKnownVideo = v;
                return v;
            }}
            return lastKnownVideo;
        }}

        function updateHUD(video) {{
            if (!video) video = getTargetVideo();
            if (!video) return;

            const timeSpan = document.getElementById(containerId + '-time');
            const durSpan = document.getElementById(containerId + '-duration');
            const frameSpan = document.getElementById(containerId + '-frame');
            const playPauseBtn = document.getElementById(containerId + '-play-pause');

            const curr = video.currentTime || 0;
            const dur = video.duration || 0;
            const frame = Math.floor(curr * defaultFps);

            if (timeSpan) timeSpan.textContent = formatTimeMMSSmmm(curr);
            if (durSpan && dur > 0) durSpan.textContent = formatTimeMMSSmmm(dur);
            if (frameSpan) frameSpan.textContent = 'Frame #' + frame;
            if (playPauseBtn) {{
                playPauseBtn.textContent = video.paused ? '▶️ Play' : '⏸️ Pausa';
            }}
        }}

        function populateMarkers() {{
            const markersBox = document.getElementById(containerId + '-markers-container');
            if (!markersBox || !markersData || markersData.length === 0) return;

            markersBox.querySelectorAll('.marker-chip').forEach(c => c.remove());

            markersData.forEach((m) => {{
                const chip = document.createElement('button');
                chip.className = 'marker-chip';
                chip.style.borderColor = m.border || '#4F46E5';
                chip.style.color = '#F8FAFC';
                chip.style.background = 'rgba(15, 23, 42, 0.85)';
                chip.innerHTML = '<span>' + m.label + '</span>';

                chip.addEventListener('click', function(e) {{
                    e.preventDefault();
                    const vid = getTargetVideo();
                    if (vid) {{
                        vid.currentTime = Math.max(0, m.seconds);
                        if (vid.paused) {{
                            vid.play().catch(() => {{}});
                        }}
                        updateHUD(vid);
                        chip.style.transform = 'scale(1.08)';
                        chip.style.borderColor = '#FCD34D';
                        setTimeout(() => {{
                            chip.style.transform = 'scale(1.0)';
                            chip.style.borderColor = m.border || '#4F46E5';
                        }}, 300);
                    }}
                }});
                markersBox.appendChild(chip);
            }});
        }}

        function updateActiveSpeedButton(rate) {{
            const speedBtns = document.querySelectorAll('#' + containerId + ' .speed-btn');
            speedBtns.forEach(b => {{
                const bRate = parseFloat(b.getAttribute('data-speed'));
                if (Math.abs(bRate - rate) < 0.05) {{
                    b.classList.add('active');
                }} else {{
                    b.classList.remove('active');
                }}
            }});
        }}

        function adjustSpeed(vid, delta) {{
            if (!vid) vid = getTargetVideo();
            if (!vid) return;

            const speeds = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
            const current = vid.playbackRate || 1.0;
            let closestIdx = 0;
            let minDiff = 999;
            for (let i = 0; i < speeds.length; i++) {{
                const diff = Math.abs(speeds[i] - current);
                if (diff < minDiff) {{
                    minDiff = diff;
                    closestIdx = i;
                }}
            }}
            let nextIdx = closestIdx + (delta > 0 ? 1 : -1);
            if (nextIdx < 0) nextIdx = 0;
            if (nextIdx >= speeds.length) nextIdx = speeds.length - 1;
            vid.playbackRate = speeds[nextIdx];
            activeSpeed = speeds[nextIdx];
            updateActiveSpeedButton(speeds[nextIdx]);
        }}

        function autoResize() {{
            try {{
                const panel = document.getElementById(containerId);
                if (panel && window.frameElement) {{
                    const neededH = panel.scrollHeight + 12;
                    window.frameElement.style.height = neededH + 'px';
                }}
            }} catch(e) {{}}
        }}

        function wireButtons() {{
            const stepB5 = document.getElementById(containerId + '-step-b5');
            const stepB1 = document.getElementById(containerId + '-step-b1');
            const stepB05 = document.getElementById(containerId + '-step-b05');
            const stepBFrame = document.getElementById(containerId + '-step-bframe');
            const playPause = document.getElementById(containerId + '-play-pause');
            const stepFFrame = document.getElementById(containerId + '-step-fframe');
            const stepF05 = document.getElementById(containerId + '-step-f05');
            const stepF1 = document.getElementById(containerId + '-step-f1');
            const stepF5 = document.getElementById(containerId + '-step-f5');
            const btnCopy = document.getElementById(containerId + '-btn-copy');

            const frameTime = 1.0 / defaultFps;

            if (stepB5) stepB5.onclick = (e) => {{ e.preventDefault(); const v = getTargetVideo(); if (v) {{ v.currentTime = Math.max(0, v.currentTime - 5.0); updateHUD(v); }} }};
            if (stepB1) stepB1.onclick = (e) => {{ e.preventDefault(); const v = getTargetVideo(); if (v) {{ v.currentTime = Math.max(0, v.currentTime - 1.0); updateHUD(v); }} }};
            if (stepB05) stepB05.onclick = (e) => {{ e.preventDefault(); const v = getTargetVideo(); if (v) {{ v.currentTime = Math.max(0, v.currentTime - 0.5); updateHUD(v); }} }};
            if (stepBFrame) stepBFrame.onclick = (e) => {{ e.preventDefault(); const v = getTargetVideo(); if (v) {{ v.currentTime = Math.max(0, v.currentTime - frameTime); updateHUD(v); }} }};

            if (playPause) {{
                playPause.onclick = (e) => {{
                    e.preventDefault();
                    const v = getTargetVideo();
                    if (v) {{
                        if (v.paused) {{
                            v.play().catch(() => {{}});
                        }} else {{
                            v.pause();
                        }}
                        updateHUD(v);
                    }}
                }};
            }}

            if (stepFFrame) stepFFrame.onclick = (e) => {{ e.preventDefault(); const v = getTargetVideo(); if (v) {{ v.currentTime = Math.min(v.duration || 99999, v.currentTime + frameTime); updateHUD(v); }} }};
            if (stepF05) stepF05.onclick = (e) => {{ e.preventDefault(); const v = getTargetVideo(); if (v) {{ v.currentTime = Math.min(v.duration || 99999, v.currentTime + 0.5); updateHUD(v); }} }};
            if (stepF1) stepF1.onclick = (e) => {{ e.preventDefault(); const v = getTargetVideo(); if (v) {{ v.currentTime = Math.min(v.duration || 99999, v.currentTime + 1.0); updateHUD(v); }} }};
            if (stepF5) stepF5.onclick = (e) => {{ e.preventDefault(); const v = getTargetVideo(); if (v) {{ v.currentTime = Math.min(v.duration || 99999, v.currentTime + 5.0); updateHUD(v); }} }};

            if (btnCopy) {{
                btnCopy.onclick = (e) => {{
                    e.preventDefault();
                    const v = getTargetVideo();
                    const curr = v ? v.currentTime : 0;
                    const tsStr = formatTimeMMSSmmm(curr);
                    navigator.clipboard.writeText(tsStr).then(() => {{
                        const origText = btnCopy.textContent;
                        btnCopy.textContent = '✅ Copiado!';
                        btnCopy.style.background = '#10B981';
                        btnCopy.style.color = '#FFFFFF';
                        setTimeout(() => {{
                            btnCopy.textContent = origText;
                            btnCopy.style.background = '';
                            btnCopy.style.color = '';
                        }}, 1500);
                    }}).catch(() => {{}});
                }};
            }}

            const speedBtns = document.querySelectorAll('#' + containerId + ' .speed-btn');
            speedBtns.forEach(btn => {{
                btn.onclick = (e) => {{
                    e.preventDefault();
                    const rate = parseFloat(btn.getAttribute('data-speed'));
                    const v = getTargetVideo();
                    if (v) {{
                        v.playbackRate = rate;
                        activeSpeed = rate;
                        updateActiveSpeedButton(rate);
                    }}
                }};
            }});
        }}

        function attachKeyboardShortcuts() {{
            function handleKeyDown(e) {{
                const tag = (e.target && e.target.tagName) ? e.target.tagName.toUpperCase() : '';
                if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || e.target.isContentEditable) {{
                    return;
                }}

                const activeVid = getTargetVideo();
                if (!activeVid) return;

                const frameStep = 1.0 / defaultFps;

                switch (e.key) {{
                    case ' ':
                        e.preventDefault();
                        if (activeVid.paused) activeVid.play().catch(() => {{}});
                        else activeVid.pause();
                        updateHUD(activeVid);
                        break;
                    case 'ArrowLeft':
                        e.preventDefault();
                        const stepLeft = e.shiftKey ? 1.0 : 0.5;
                        activeVid.currentTime = Math.max(0, activeVid.currentTime - stepLeft);
                        updateHUD(activeVid);
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        const stepRight = e.shiftKey ? 1.0 : 0.5;
                        activeVid.currentTime = Math.min(activeVid.duration || 99999, activeVid.currentTime + stepRight);
                        updateHUD(activeVid);
                        break;
                    case ',':
                    case '<':
                        e.preventDefault();
                        activeVid.currentTime = Math.max(0, activeVid.currentTime - frameStep);
                        updateHUD(activeVid);
                        break;
                    case '.':
                    case '>':
                        e.preventDefault();
                        activeVid.currentTime = Math.min(activeVid.duration || 99999, activeVid.currentTime + frameStep);
                        updateHUD(activeVid);
                        break;
                    case '[':
                        e.preventDefault();
                        adjustSpeed(activeVid, -0.25);
                        break;
                    case ']':
                        e.preventDefault();
                        adjustSpeed(activeVid, +0.25);
                        break;
                    case '0':
                        if (!e.ctrlKey && !e.metaKey) {{
                            e.preventDefault();
                            activeVid.currentTime = 0;
                            updateHUD(activeVid);
                        }}
                        break;
                }}
            }}

            try {{
                if (window.parent && window.parent.document && !window.parent.__senpai_video_shortcuts_attached) {{
                    window.parent.__senpai_video_shortcuts_attached = true;
                    window.parent.document.addEventListener('keydown', handleKeyDown);
                }}
            }} catch (e) {{}}

            if (!window.__senpai_local_shortcuts_attached) {{
                window.__senpai_local_shortcuts_attached = true;
                document.addEventListener('keydown', handleKeyDown);
            }}
        }}

        function applyInitialSeek(video) {{
            if (initialSeekApplied || targetStartTime <= 0 || !video) return;
            initialSeekApplied = true;
            try {{
                video.currentTime = targetStartTime;
                video.play().catch(() => {{}});
                updateHUD(video);
            }} catch (e) {{}}
        }}

        let attachedVideo = null;
        function bindVideoListeners(video) {{
            if (!video || video === attachedVideo) return;
            attachedVideo = video;
            lastKnownVideo = video;

            video.addEventListener('timeupdate', () => updateHUD(video));
            video.addEventListener('play', () => updateHUD(video));
            video.addEventListener('pause', () => updateHUD(video));
            video.addEventListener('seeking', () => updateHUD(video));
            video.addEventListener('seeked', () => updateHUD(video));
            video.addEventListener('ratechange', () => {{
                activeSpeed = video.playbackRate;
                updateActiveSpeedButton(activeSpeed);
            }});
            video.addEventListener('loadedmetadata', () => {{
                applyInitialSeek(video);
                updateHUD(video);
                autoResize();
            }});
            applyInitialSeek(video);
            updateHUD(video);
        }}

        function initControls() {{
            wireButtons();
            populateMarkers();
            attachKeyboardShortcuts();
            autoResize();

            const v = findVideoElement();
            if (v) {{
                bindVideoListeners(v);
            }}

            if (pollTimer) clearInterval(pollTimer);
            pollTimer = setInterval(() => {{
                const currentV = findVideoElement();
                if (currentV) {{
                    if (currentV !== attachedVideo) {{
                        bindVideoListeners(currentV);
                    }} else {{
                        updateHUD(currentV);
                    }}
                }}
            }}, 400);
        }}

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initControls);
        }} else {{
            initControls();
        }}
        setTimeout(autoResize, 300);
        setTimeout(autoResize, 1000);
    }})();
    </script>
    """
    return html_content


def _parse_ts_str(ts_str: Optional[str] = None) -> float:
    """Converte string de timestamp (ex: '00:02.500' ou '2.5s') em segundos float."""
    if not ts_str:
        return 0.0
    try:
        ts = ts_str.strip().lower().replace("s", "")
        if ":" in ts:
            parts = ts.split(":")
            if len(parts) == 2:
                mins, secs = float(parts[0]), float(parts[1])
                return mins * 60.0 + secs
            elif len(parts) == 3:
                hrs, mins, secs = float(parts[0]), float(parts[1]), float(parts[2])
                return hrs * 3600.0 + mins * 60.0 + secs
        return float(ts)
    except Exception:
        return 0.0


def render_video_playback_controls(
    events: Optional[List[Dict[str, Any]]] = None,
    sonkyo_info: Optional[Dict[str, Any]] = None,
    sonkyo_edits: Optional[Dict[str, Any]] = None,
    default_fps: float = 30.0,
    container_id: str = "senpai-video-controller",
    target_start_time: float = 0.0,
    height: Optional[int] = None
) -> None:
    """
    Renderiza os controles interativos de vídeo diretamente na interface do Streamlit usando st.iframe().
    
    Args:
        events: Lista de eventos/golpes detectados pela análise.
        sonkyo_info: Dicionário com informações do Sonkyo detectado.
        sonkyo_edits: Dicionário com eventuais edições manuais de Sonkyo.
        default_fps: Taxa de FPS estimada.
        container_id: ID do container HTML.
        target_start_time: Segundo opcional para o qual o vídeo deve saltar imediatamente.
        height: Altura opcional do container (calculada automaticamente se None).
    """
    html_code = generate_video_playback_controls_html(
        events=events,
        sonkyo_info=sonkyo_info,
        sonkyo_edits=sonkyo_edits,
        default_fps=default_fps,
        container_id=container_id,
        target_start_time=target_start_time
    )
    has_markers = bool(
        (events and len(events) > 0) or
        (sonkyo_info and (sonkyo_info.get("has_initial_sonkyo") or sonkyo_info.get("has_final_sonkyo"))) or
        (sonkyo_edits and (sonkyo_edits.get("initial") or sonkyo_edits.get("final")))
    )
    calc_height = height or (200 if has_markers else 150)
    if hasattr(st, "iframe"):
        st.iframe(html_code, height=calc_height)
    else:
        components.html(html_code, height=calc_height, scrolling=False)
