"""
SenpAI - Web Dashboard Interativo de Análise de Kendo (Streamlit App)
Suporta 3 Modos Principais de Operação:
1. 📹 Modo de Detecção Gravada
2. 🎓 Modo de Treinamento & Aprendizado
3. 🔴 Modo de Detecção em Tempo Real (Webcam / Stream RTSP/RTCP)
"""

import streamlit as st
import tempfile
import os
import warnings

# Suprime aviso benigno interno de depreciação do protobuf com mediapipe
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import cv2
import json
import time
from typing import Any, Dict, List, Optional

from src.pipeline import SenpAIPipeline, AnalysisWorker
from src.utils.demo_generator import generate_demo_kendo_video
from src.engine.feedback_manager import FeedbackManager
from src.engine.reporter import DiagnosticReporter
from src.analytics.sonkyo_detector import SonkyoDetector
from src.analytics.training_analyzer import (
    TrainingAnalyzer,
    TRAINING_MODALITIES_METADATA,
    TrainingPillarMetrics,
    KendokaTrainingProfile,
    TrainingSessionResult
)
from src.utils.hardware import (
    detect_nvidia_gpu, get_effective_device, check_cuda_framework_support,
    validate_and_setup_gpu_requirements, detect_connected_cameras
)
from src.utils.settings_manager import load_settings, save_settings, get_processing_device, set_processing_device
from src.utils.logger_manager import (
    setup_system_logger, get_log_summary, get_memory_logs,
    get_debug_log_file_content, clear_debug_logs, run_system_diagnostic_check, log_event
)
from src.utils.test_runner import (
    run_automated_tests, get_latest_test_report_content, TEST_LOG_PATH
)
from src.utils.video_downloader import (
    validate_video_url, extract_video_info, download_video_stream,
    format_video_duration, VideoDownloadError, QUALITY_LABELS
)

# Inicializa o logger central do sistema
setup_system_logger()

st.set_page_config(
    page_title="SenpAI - AI Kendo Referee & Analysis System",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

feedback_mgr = FeedbackManager()

# Mapeamento oficial de marcações Katakana para o Painel de Pontuação Final (Sanbon-Shobu)
SCOREBOARD_KATAKANA_MARKS = {
    "MEN": "メ",
    "KOTE": "コ",
    "DO": "ド",
    "TSUKI": "ツ",
    "メ MEN": "メ",
    "コ KOTE": "コ",
    "ド DO": "ド",
    "ツ TSUKI": "ツ",
    "メ": "メ",
    "コ": "コ",
    "ド": "ド",
    "ツ": "ツ",
}

def format_scoreboard_strike_name(strike_type: str) -> str:
    """Formata a nomenclatura do golpe exclusivamente com o caractere Katakana oficial (メ, コ, ド, ツ) para o painel de pontuação final."""
    if not strike_type:
        return ""
    st_clean = strike_type.strip()
    return SCOREBOARD_KATAKANA_MARKS.get(st_clean.upper(), SCOREBOARD_KATAKANA_MARKS.get(st_clean, st_clean))

# Mapeamento oficial para destaque na Linha do Tempo de golpes com marcação válida (Ippon)
TIMELINE_KATAKANA_STRIKES = {
    "MEN": "メ MEN",
    "KOTE": "コ KOTE",
    "DO": "ド DO",
    "TSUKI": "ツ TSUKI",
    "メ MEN": "メ MEN",
    "コ KOTE": "コ KOTE",
    "ド DO": "ド DO",
    "ツ TSUKI": "ツ TSUKI",
    "メ": "メ MEN",
    "コ": "コ KOTE",
    "ド": "ド DO",
    "ツ": "ツ TSUKI",
}

def format_katakana_strike(strike_type: str) -> str:
    """Formata a nomenclatura do golpe com prefixo Katakana oficial (ex: 'メ MEN', 'コ KOTE', 'ド DO', 'ツ TSUKI')."""
    if not strike_type:
        return ""
    st_clean = strike_type.strip()
    return TIMELINE_KATAKANA_STRIKES.get(st_clean.upper(), TIMELINE_KATAKANA_STRIKES.get(st_clean, st_clean))

def parse_ts_to_seconds(ts_str: str) -> float:
    """Converte timestamps (ex: '00:02.500', '02.500', '2.5s') em segundos (float)."""
    if not ts_str:
        return 0.0
    try:
        ts = ts_str.strip().lower().replace("s", "")
        if ":" in ts:
            parts = ts.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(ts)
    except Exception:
        return 0.0

def format_seconds_to_ts(seconds: float) -> str:
    """Converte segundos em string no formato MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"

def render_training_analysis_view(res: Dict[str, Any], is_inverted: bool):
    """
    Renderiza o Painel Especializado de Treinamento & Aprendizado de Kendo:
    - Identificação e metadados das 14 modalidades de treinamento com Kanjis.
    - Avaliação minuciosa dos 3 Pilares (Movimentação, Precisão e Constância).
    - Rastreamento e nomeação customizada de cada Kendoca no Shiaijo.
    - Diagnósticos pedagógicos (Pontos Fortes, Pontos de Melhoria e Prescrições Práticas).
    - Exportação de relatório individualizado (.MD) por Kendoca e relatório consolidado (.JSON).
    """
    train_data = res.get("training_analysis", {})
    if not train_data:
        st.info("ℹ️ Nenhuma métrica de treinamento disponível para este vídeo.")
        return

    mod_key = train_data.get("modality_key", "suburi")
    mod_conf = int(train_data.get("detection_confidence", 0.8) * 100)
    det_method = train_data.get("detection_method", "AUTO_DETECTED")
    kendokas = train_data.get("kendokas", [])
    dur_sec = train_data.get("duration_seconds", res.get("duration_seconds", 0.0))

    meta = TRAINING_MODALITIES_METADATA.get(mod_key, TRAINING_MODALITIES_METADATA.get("suburi", {}))
    mod_display_name = meta.get("name", "Treinamento de Kendo")

    # 1. CABEÇALHO DA MODALIDADE DE TREINO
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #090D16 0%, #1E1B4B 100%); border: 2px solid #6366F1; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(99, 102, 241, 0.35); padding-bottom: 8px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 26px;">🎓</span>
                    <div>
                        <div style="color: #A5B4FC; font-size: 11px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase;">AVALIAÇÃO DE TREINAMENTO & APRENDIZADO DE KENDO</div>
                        <div style="color: #FFFFFF; font-size: 20px; font-weight: 900; font-family: monospace; line-height: 1.2;">
                            {mod_display_name}
                        </div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(99, 102, 241, 0.25); color: #C7D2FE; border: 1px solid #6366F1; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 700;">
                        {'🔍 IA Detectou' if det_method == 'AUTO_DETECTED' else '⚙️ Seleção Manual'} ({mod_conf}% Confiança)
                    </span>
                    <div style="color: #94A3B8; font-size: 11px; margin-top: 3px;">⏱️ Duração: {dur_sec:.1f}s • 🥋 Praticantes: {len(kendokas)}</div>
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 8px 12px; border: 1px solid rgba(255,255,255,0.06);">
                <div style="color: #E2E8F0; font-size: 13px; margin-bottom: 3px;"><b>{meta.get('category', 'Modalidade')}:</b> {meta.get('description', '')}</div>
                <div style="color: #94A3B8; font-size: 12px;">🎯 <b>Focos Principais de Avaliação:</b> {' • '.join(meta.get('focus_areas', []))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. CONTROLES: DOWNLOAD DO RELATÓRIO CONSOLIDADO
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.caption("💡 *Os 3 Pilares avaliam a biomecânica e trabalho de pés (Movimentação), sincronismo Ki-Ken-Tai-Ichi e alvo (Precisão) e cadência com ritmo respiratório (Constância).*")
    with top_c2:
        report_json_str = json.dumps(train_data, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Exportar Sessão (.JSON)",
            data=report_json_str,
            file_name=f"sessao_treino_{mod_key}_{int(time.time())}.json",
            mime="application/json",
            width="stretch",
            key="btn_dl_train_report_json"
        )

    # 3. RASTREAMENTO INDIVIDUAL DOS KENDOCAS NO DOJO
    for idx_k, k_prof in enumerate(kendokas):
        k_id = k_prof.get("kendoka_id", f"KENSHI_{idx_k+1}")
        k_def = k_prof.get("default_name", f"Kendoca {idx_k+1}")
        k_cust = k_prof.get("custom_name", k_def)
        k_role = k_prof.get("role", "Praticante")
        pillars = k_prof.get("pillars", {})
        strengths = k_prof.get("strengths", [])
        improvements = k_prof.get("improvements", [])
        exercises = k_prof.get("recommended_exercises", [])
        timeline = k_prof.get("repetition_timeline", [])

        overall = float(pillars.get("overall_score", 0.0))
        mov = float(pillars.get("movimentacao", pillars.get("forma", 0.0)))
        prec = float(pillars.get("precisao", 0.0))
        const = float(pillars.get("constancia", 0.0))
        cadence_cpm = float(pillars.get("cadence_cpm", 0.0))
        reps = int(pillars.get("total_repetitions", 0))

        if overall >= 85:
            badge_perf = ("🏆 EXCELENTE (NÍVEL AVANÇADO)", "#22C55E", "rgba(34, 197, 94, 0.15)", "#15803D")
        elif overall >= 70:
            badge_perf = ("🥇 MUITO BOM (NÍVEL INTERMEDIÁRIO)", "#38BDF8", "rgba(56, 189, 248, 0.15)", "#0284C7")
        elif overall >= 55:
            badge_perf = ("🥈 SATISFATÓRIO (EM DESENVOLVIMENTO)", "#EAB308", "rgba(234, 179, 8, 0.15)", "#A16207")
        else:
            badge_perf = ("⚠️ NECESSITA AJUSTES DE FUNDAMENTO", "#EF4444", "rgba(239, 68, 68, 0.15)", "#B91C1C")

        st.markdown(
            f"""
            <div style="background: #0B1120; border: 1.5px solid #334155; border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E293B; padding-bottom: 8px; margin-bottom: 12px;">
                    <div>
                        <span style="color: #F8FAFC; font-size: 17px; font-weight: 800;">🥋 {k_cust}</span>
                        <span style="color: #94A3B8; font-size: 12px; margin-left: 8px;">({k_def} • {k_role})</span>
                    </div>
                    <div style="background: {badge_perf[2]}; border: 1px solid {badge_perf[3]}; border-radius: 6px; padding: 4px 10px;">
                        <span style="color: {badge_perf[1]}; font-weight: 800; font-size: 13px;">{badge_perf[0]} — {overall:.1f}/100</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_in_name, col_meta_k, col_dl_indiv = st.columns([1.5, 2.0, 1.2])
        with col_in_name:
            curr_name_input = st.text_input(
                f"Nomear Kendoca ({k_def}):",
                value=k_cust,
                key=f"train_name_in_{k_id}_{idx_k}"
            )
            if curr_name_input and curr_name_input != k_cust:
                if "training_kendoka_names" not in st.session_state:
                    st.session_state["training_kendoka_names"] = {}
                st.session_state["training_kendoka_names"][k_id] = curr_name_input
                k_prof["custom_name"] = curr_name_input
                k_cust = curr_name_input
        with col_meta_k:
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.4); border-radius: 6px; padding: 8px 12px; margin-top: 14px; display: flex; justify-content: space-around; font-size: 12px;">
                    <span>⏱️ Cadência: <b style="color:#F8FAFC;">{cadence_cpm:.1f} CPM</b></span>
                    <span>🔁 Repetições: <b style="color:#F8FAFC;">{reps}</b></span>
                    <span>📊 Desvio: <b style="color:#F8FAFC;">{pillars.get('cadence_std_dev_seconds', 0.0):.2f}s DP</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_dl_indiv:
            # Gerar relatório individual em Markdown
            k_obj = KendokaTrainingProfile(
                kendoka_id=k_id,
                default_name=k_def,
                custom_name=k_cust,
                role=k_role,
                pillars=TrainingPillarMetrics(
                    movimentacao_score=mov,
                    precisao_score=prec,
                    constancia_score=const,
                    movimentacao_submetrics=pillars.get("movimentacao_submetrics", pillars.get("forma_submetrics", {})),
                    precisao_submetrics=pillars.get("precisao_submetrics", {}),
                    constancia_submetrics=pillars.get("constancia_submetrics", {}),
                    cadence_cpm=cadence_cpm,
                    cadence_std_dev_seconds=pillars.get("cadence_std_dev_seconds", 0.0),
                    total_repetitions=reps
                ),
                strengths=strengths,
                improvements=improvements,
                recommended_exercises=exercises,
                repetition_timeline=timeline
            )
            indiv_md = k_obj.generate_individual_report_markdown(meta)
            safe_k_name = k_cust.replace(" ", "_").lower()
            st.download_button(
                label=f"📥 Relatório de {k_cust[:10]} (.MD)",
                data=indiv_md,
                file_name=f"relatorio_kenshi_{safe_k_name}_{int(time.time())}.md",
                mime="text/markdown",
                width="stretch",
                key=f"btn_dl_indiv_md_{k_id}_{idx_k}"
            )

        # OS 3 PILARES (MOVIMENTAÇÃO, PRECISÃO, CONSTÂNCIA)
        p_col1, p_col2, p_col3 = st.columns(3)
        mov_sub = pillars.get('movimentacao_submetrics', pillars.get('forma_submetrics', {}))
        prec_sub = pillars.get('precisao_submetrics', {})
        const_sub = pillars.get('constancia_submetrics', {})

        with p_col1:
            st.markdown(
                f"""
                <div style="background: #0F172A; border: 1.5px solid #38BDF8; border-radius: 8px; padding: 12px; height: 100%;">
                    <div style="color: #38BDF8; font-size: 13px; font-weight: 800; display: flex; justify-content: space-between;">
                        <span>🥋 PILAR 1: MOVIMENTAÇÃO</span>
                        <span>{mov:.1f}%</span>
                    </div>
                    <div style="background: #1E293B; border-radius: 4px; height: 8px; margin: 6px 0 10px 0; overflow: hidden;">
                        <div style="background: #38BDF8; width: {mov}%; height: 100%;"></div>
                    </div>
                    <div style="font-size: 11px; color: #94A3B8; line-height: 1.6;">
                        • Verticalidade da Coluna (Shisei): <b style="color:#F1F5F9;">{mov_sub.get('verticalidade_coluna', 0):.1f}%</b><br>
                        • Nivelamento de Ombros: <b style="color:#F1F5F9;">{mov_sub.get('nivelamento_ombros', 0):.1f}%</b><br>
                        • Base e Calcanhar Esquerdo (Ashi): <b style="color:#F1F5F9;">{mov_sub.get('alinhamento_base_pes', 0):.1f}%</b><br>
                        • Amplitude de Furikaburi: <b style="color:#F1F5F9;">{mov_sub.get('amplitude_furikaburi', 0):.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with p_col2:
            st.markdown(
                f"""
                <div style="background: #0F172A; border: 1.5px solid #A855F7; border-radius: 8px; padding: 12px; height: 100%;">
                    <div style="color: #C084FC; font-size: 13px; font-weight: 800; display: flex; justify-content: space-between;">
                        <span>🎯 PILAR 2: PRECISÃO</span>
                        <span>{prec:.1f}%</span>
                    </div>
                    <div style="background: #1E293B; border-radius: 4px; height: 8px; margin: 6px 0 10px 0; overflow: hidden;">
                        <div style="background: #A855F7; width: {prec}%; height: 100%;"></div>
                    </div>
                    <div style="font-size: 11px; color: #94A3B8; line-height: 1.6;">
                        • Trajetória no Ponto Alvo: <b style="color:#F1F5F9;">{prec_sub.get('trajetoria_alvo', 0):.1f}%</b><br>
                        • Sincronismo Ki-Ken-Tai-Ichi: <b style="color:#F1F5F9;">{prec_sub.get('kikentai_sincronismo', 0):.1f}%</b><br>
                        • Controle da Linha Central: <b style="color:#F1F5F9;">{prec_sub.get('controle_linha_centro', 0):.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with p_col3:
            st.markdown(
                f"""
                <div style="background: #0F172A; border: 1.5px solid #F59E0B; border-radius: 8px; padding: 12px; height: 100%;">
                    <div style="color: #FBBF24; font-size: 13px; font-weight: 800; display: flex; justify-content: space-between;">
                        <span>⏱️ PILAR 3: CONSTÂNCIA</span>
                        <span>{const:.1f}%</span>
                    </div>
                    <div style="background: #1E293B; border-radius: 4px; height: 8px; margin: 6px 0 10px 0; overflow: hidden;">
                        <div style="background: #F59E0B; width: {const}%; height: 100%;"></div>
                    </div>
                    <div style="font-size: 11px; color: #94A3B8; line-height: 1.6;">
                        • Regularidade do Ritmo: <b style="color:#F1F5F9;">{const_sub.get('regularidade_ritmo', 0):.1f}%</b><br>
                        • Resistência à Fadiga (Stamina): <b style="color:#F1F5F9;">{const_sub.get('resistencia_fadiga', 0):.1f}%</b><br>
                        • Adequação à Cadência da Modalidade: <b style="color:#F1F5F9;">{const_sub.get('adequacao_cadencia', 0):.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # DIAGNÓSTICOS PEDAGÓGICOS E PRESCRIÇÕES DE EXERCÍCIOS
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("##### 🌟 Pontos Fortes Observados")
            if strengths:
                for s in strengths:
                    st.markdown(f"✅ <span style='color:#86EFAC; font-size:13px;'>{s}</span>", unsafe_allow_html=True)
            else:
                st.caption("Padrão de execução em fase de consolidação.")

        with d_col2:
            st.markdown("##### ⚠️ Pontos de Atenção & Correção Técnica")
            if improvements:
                for imp in improvements:
                    st.markdown(f"⚠️ <span style='color:#FCA5A5; font-size:13px;'>{imp}</span>", unsafe_allow_html=True)
            else:
                st.success("Nenhum vício biomecânico crítico detectado.")

        # Prescrição de Exercícios do Kendo
        st.markdown("##### 🏋️ Exercícios Recomendados para Evolução Técnica no Dojo")
        if exercises:
            for ex in exercises:
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #6366F1; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                        <div style="color: #A5B4FC; font-weight: 800; font-size: 13px;">🥋 {ex.get('name', 'Exercício')} <span style="color:#94A3B8; font-size:11px; font-weight:400;">(Foco: {ex.get('target', 'Fundamentos')})</span></div>
                        <div style="color: #E2E8F0; font-size: 12px; margin-top: 3px;">📋 <b>Prescrição:</b> {ex.get('prescription', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Estilização CSS Moderna, Equilibrada (Escala 90%) e sem Cortes na Interface
st.markdown("""
<style>
    /* 1. Escala Global Ajustada para 90% via Container Principal para não interferir em Popovers/Comboboxes */
    .stApp {
        zoom: 0.90;
        -moz-transform: scale(0.90);
        -moz-transform-origin: 0 0;
        height: auto !important;
        min-height: 100vh !important;
    }

    [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main {
        height: auto !important;
        min-height: 100vh !important;
        overflow-y: visible !important;
    }

    /* 2. Expansão do Container Principal e Amplo Espaço Inferior (Sem Cortes) */
    .main .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 12.0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 98% !important;
        width: 100% !important;
    }

    /* 3. Expansão e Fundo Contínuo da Barra Lateral (Cobrindo 100% dos Sliders e Conteúdo) */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] .block-container {
        background-color: #0F172A !important;
        min-height: 100% !important;
        height: auto !important;
    }

    [data-testid="stSidebar"] {
        min-width: 360px !important;
        width: 360px !important;
        height: 100vh !important;
        min-height: 100vh !important;
        scrollbar-width: thin !important;
        scrollbar-color: #475569 #0F172A !important;
    }

    [data-testid="stSidebarUserContent"], [data-testid="stSidebar"] .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 7.0rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }

    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.15rem !important;
    }

    [data-testid="stSidebar"] .stRadio {
        margin-bottom: 0.2rem !important;
    }

    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.3rem !important;
    }

    [data-testid="stSidebar"] .stSlider {
        margin-top: 0.1rem !important;
        margin-bottom: 0.2rem !important;
    }

    [data-testid="stSidebar"] .stAlert {
        padding: 0.5rem 0.75rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.5rem !important;
        font-size: 0.90rem !important;
    }

    /* 4. Estabilidade e Visibilidade dos Menus Suspensos / Comboboxes */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"] {
        z-index: 9999999 !important;
        pointer-events: auto !important;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #E2E8F0;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
    }
    .valid-badge {
        background-color: #166534;
        color: #4ADE80;
        padding: 0.25rem 0.7rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .invalid-badge {
        background-color: #991B1B;
        color: #FCA5A5;
        padding: 0.25rem 0.7rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .mode-banner-recorded {
        background-color: #0F172A;
        border-left: 4px solid #3B82F6;
        padding: 0.7rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .mode-banner-training {
        background-color: #1E1B4B;
        border-left: 4px solid #8B5CF6;
        padding: 0.7rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .mode-banner-realtime {
        background-color: #311313;
        border-left: 4px solid #EF4444;
        padding: 0.7rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
    }
    /* Estilização da Coluna Fixa (Sticky) do Vídeo */
    div[data-testid="stColumn"]:has(div.sticky-video-marker) {
        position: -webkit-sticky;
        position: sticky;
        top: 1rem;
        align-self: flex-start;
        z-index: 99;
    }
    /* Card de Métricas do Combate */
    .summary-card {
        background-color: #1E293B;
        border-radius: 0.75rem;
        border: 1px solid #334155;
        padding: 0.85rem;
        margin-top: 0.8rem;
    }

    /* Estilização moderna da barra de rolagem da Lista de Golpes & Eventos */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0.75rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        scrollbar-width: thin !important;
        scrollbar-color: #475569 #1E293B !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar {
        width: 6px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-track {
        background: #1E293B !important;
        border-radius: 4px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb {
        background: #475569 !important;
        border-radius: 4px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div::-webkit-scrollbar-thumb:hover {
        background: #64748B !important;
    }

    /* 5. Estilização Premium para Guias (Tabs) */
    div[data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #0B1120 !important;
        padding: 6px !important;
        border-radius: 10px !important;
        border: 1px solid #1E293B !important;
        margin-bottom: 1.5rem !important;
    }
    button[data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #94A3B8 !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease-in-out !important;
    }
    button[data-baseweb="tab"]:hover {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border-color: #334155 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1E293B !important;
        color: #60A5FA !important;
        border: 1px solid #3B82F6 !important;
        box-shadow: 0 0 14px rgba(59, 130, 246, 0.25) !important;
    }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚔️ SenpAI (先輩 AI)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistema de Visão Computacional para Análise de Lutas de Kendo, Detecção de Golpes e Avaliação de Yuko-Datotsu</div>', unsafe_allow_html=True)

# --- SIDEBAR: NAVEGAÇÃO PRINCIPAL ---
st.sidebar.markdown("## 📌 Navegação")
nav_page = st.sidebar.radio(
    "Selecione a Página",
    options=["analysis", "settings"],
    format_func=lambda x: {
        "analysis": "⚔️ Análise de Lutas",
        "settings": "⚙️ Menu de Configurações"
    }[x]
)

st.sidebar.markdown("---")

# ==============================================================================
# PÁGINA 1: MENU DE CONFIGURAÇÕES (LAYOUT EM GUIAS / TABS)
# ==============================================================================
if nav_page == "settings":
    st.header("⚙️ Configurações Gerais do Sistema")
    st.markdown("Gerencie os parâmetros de aceleração de hardware, governança de modelos, perfis de calibração e ferramentas de diagnóstico.")

    saved_sys_settings = load_settings()
    default_device_pref = st.session_state.get("device_preference", saved_sys_settings.get("processing_device", "cpu"))

    tab_hw, tab_train, tab_calib, tab_diag = st.tabs([
        "🖥️ Processamento & Hardware",
        "🎓 Governança de Treinamento",
        "🎛️ Perfis de Calibração",
        "🐛 Diagnóstico, Alertas & Logs"
    ])

    # --------------------------------------------------------------------------
    # GUIA 1: PROCESSAMENTO & HARDWARE
    # --------------------------------------------------------------------------
    with tab_hw:
        st.markdown("### 🖥️ Aceleração de Hardware & Seleção de Dispositivo")
        st.caption("Configure o dispositivo de inferência para os modelos neurais (CPU ou GPU NVIDIA CUDA com aceleração FP16).")

        col_hw1, col_hw2 = st.columns([1, 1])

        with col_hw1:
            st.markdown("**Seletor do Modo de Processamento:**")
            selected_hw_option = st.radio(
                "Escolha o acelerador:",
                options=["cpu", "gpu"],
                index=0 if default_device_pref == "cpu" else 1,
                format_func=lambda x: {
                    "cpu": "💻 Processamento por CPU somente",
                    "gpu": "⚡ Processamento por GPU (quando houver)"
                }[x],
                help="• CPU Somente: Utiliza o processador da máquina.\n• GPU (quando houver): Processa via GPU NVIDIA se disponível no computador (RTX/GTX), ou faz fallback automático para CPU."
            )

            if st.button("💾 Salvar Configurações de Hardware", type="primary", width="stretch", key="btn_save_hw_tab"):
                set_processing_device(selected_hw_option)
                st.session_state["device_preference"] = selected_hw_option
                st.success("✅ Configurações de hardware salvas com sucesso!")

        with col_hw2:
            st.markdown("**Status e Diagnóstico de Hardware em Tempo Real:**")
            gpu_check_info = detect_nvidia_gpu()
            cuda_fw = check_cuda_framework_support()

            if gpu_check_info["has_nvidia_gpu"]:
                st.success(f"🟢 **Placa NVIDIA Aceleradora Detectada:** {gpu_check_info['gpu_name']}")
                st.caption(f"Driver: {gpu_check_info['driver_version']} | VRAM: {gpu_check_info['memory_total']}")

                if cuda_fw["torch_cuda"]:
                    st.info(f"✅ **Ambiente PyTorch CUDA Ativo:** Dispositivo `{cuda_fw['torch_device_name']}` pronto para inferência rápida.")
                else:
                    st.warning("⚠️ **Dependências CUDA incompletas:** Suporte PyTorch CUDA não detectado.")
                    if st.button("🚀 Instalar Requisitos CUDA para GPU NVIDIA", width="stretch", key="btn_install_cuda_tab"):
                        with st.spinner(f"Instalando pacotes PyTorch CUDA para {gpu_check_info['gpu_name']}..."):
                            install_res = validate_and_setup_gpu_requirements(auto_install=True)
                            if install_res["cuda_ready"]:
                                st.success("✅ Pacotes CUDA instalados com sucesso!")
                                st.rerun()
                            else:
                                st.error(install_res["message"])
            else:
                st.info("💻 **Computador rodando em Modo CPU.** Nenhuma GPU NVIDIA dedicada detectada.")

    # --------------------------------------------------------------------------
    # GUIA 2: GOVERNANÇA DE TREINAMENTO & PAINEL DE REVISÃO POR DAN
    # --------------------------------------------------------------------------
    with tab_train:
        st.markdown("### 🎓 Governança de Treinamento & Painel de Revisão por Dan")
        st.caption("Acompanhe as métricas globais de retreinamento do modelo, distribuição por graduação Dan e gerencie os dados de revisão.")

        training_metrics = feedback_mgr.get_training_metrics()

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total de Treinamentos Realizados", training_metrics["total_trainings_count"])
        m_col2.metric("Nível Médio (Dan) dos Treinamentos", training_metrics["average_dan_label"])
        m_col3.metric("Total de Marcações de Revisão", training_metrics["total_review_items"])

        st.markdown("#### 📊 Distribuição de Treinamentos por Graduação Dan:")
        dan_table_md = "| Dan | Nome da Graduação | Quantidade de Treinamentos | Percentual (%) |\n| :--- | :--- | :---: | :---: |\n"
        for d_row in training_metrics["dan_distribution"]:
            dan_table_md += f"| **{d_row['Dan']}** | {d_row['Nome Graduação']} | {d_row['Quantidade Treinamentos']} | {d_row['Percentual (%)']} |\n"
        st.markdown(dan_table_md)

        st.markdown("#### 🛠️ Gerenciamento do Dataset de Treinamento:")
        act_col1, act_col2, act_col3 = st.columns(3)

        with act_col1:
            st.markdown("**🗑️ Apagar Treinamento do Sistema**")
            st.caption("Reseta todo o histórico de revisões e restaura o modelo ao estágio inicial.")
            confirm_reset = st.checkbox("Confirmo que desejo apagar todo o treinamento", key="chk_confirm_reset_tab")
            if st.button("🗑️ Apagar Treinamento", type="secondary", width="stretch", key="btn_reset_train_tab"):
                if confirm_reset:
                    feedback_mgr.reset_all_training_data()
                    st.success("✅ Treinamento do sistema apagado com sucesso! Sistema restaurado ao estágio inicial.")
                    st.rerun()
                else:
                    st.warning("⚠️ Marque a caixa de confirmação acima antes de apagar.")

        with act_col2:
            st.markdown("**📥 Baixar Treinamento Atual**")
            st.caption("Baixa pacote contendo todas as revisões, Dan dos revisores e datas dos treinamentos.")
            pkg_data = feedback_mgr.export_training_package()
            pkg_json_str = json.dumps(pkg_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Baixar Treinamento (.json)",
                data=pkg_json_str,
                file_name=f"senpai_training_package_{int(time.time())}.json",
                mime="application/json",
                width="stretch",
                key="btn_dl_train_tab"
            )

        with act_col3:
            st.markdown("**📤 Carregar Treinamento Baixado**")
            st.caption("Importa arquivos de revisão previamente baixados para recalibrar o modelo.")
            imported_file = st.file_uploader("Selecione pacote (.json)", type=["json"], key="import_pkg_file_tab")
            if imported_file is not None:
                if st.button("📤 Importar e Retreinar Modelo", type="primary", width="stretch", key="btn_import_train_tab"):
                    try:
                        imported_file.seek(0)
                        pkg_content = json.loads(imported_file.read().decode("utf-8"))
                        import_res = feedback_mgr.import_training_package(pkg_content)
                        st.success(f"🎉 Pacote importado com sucesso! {import_res['new_items_added']} novos itens integrados. Novo Dan médio: {import_res['average_dan_now']}.")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"❌ Erro ao importar pacote de treinamento: {ex}")

    # --------------------------------------------------------------------------
    # GUIA 3: PERFIS DE CALIBRAÇÃO & CRITÉRIOS DE ARBITRAGEM
    # --------------------------------------------------------------------------
    with tab_calib:
        st.markdown("### 🎛️ Perfis de Calibração & Critérios Técnicos")
        st.caption("Consulte os perfis de rigor de arbitragem e os pesos de validação de Ki-Ken-Tai-Ichi aplicados na detecção de Ippon.")

        calib_file_path = "config/calibration_profiles.json"
        calib_data = {}
        if os.path.exists(calib_file_path):
            with open(calib_file_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)

        c_card1, c_card2, c_card3 = st.columns(3)

        p_perm = calib_data.get("permissivo", {})
        p_norm = calib_data.get("normal", {})
        p_rig = calib_data.get("rigido", {})

        with c_card1:
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border-left: 4px solid #10B981; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                    <div style="font-weight: 700; color: #34D399; font-size: 1.05rem;">🟢 {p_perm.get('name', 'Permissivo')}</div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px; margin-bottom: 10px;">{p_perm.get('description', '')}</div>
                    <div style="font-size: 0.88rem; color: #E2E8F0;">
                        <b>Pontuação Mínima:</b> {int(p_perm.get('min_total_score', 0.50) * 100)}%<br>
                        <b>Alvo:</b> {int(p_perm.get('weights', {}).get('target_impact', 0.35) * 100)}% |
                        <b>Fumikomi:</b> {int(p_perm.get('weights', {}).get('fumikomi_sync', 0.25) * 100)}%<br>
                        <b>Postura:</b> {int(p_perm.get('weights', {}).get('posture', 0.20) * 100)}% |
                        <b>Zanshin:</b> {int(p_perm.get('weights', {}).get('zanshin', 0.20) * 100)}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c_card2:
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border-left: 4px solid #3B82F6; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                    <div style="font-weight: 700; color: #60A5FA; font-size: 1.05rem;">🔵 {p_norm.get('name', 'Normal')}</div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px; margin-bottom: 10px;">{p_norm.get('description', '')}</div>
                    <div style="font-size: 0.88rem; color: #E2E8F0;">
                        <b>Pontuação Mínima:</b> {int(p_norm.get('min_total_score', 0.65) * 100)}%<br>
                        <b>Alvo:</b> {int(p_norm.get('weights', {}).get('target_impact', 0.40) * 100)}% |
                        <b>Fumikomi:</b> {int(p_norm.get('weights', {}).get('fumikomi_sync', 0.25) * 100)}%<br>
                        <b>Postura:</b> {int(p_norm.get('weights', {}).get('posture', 0.20) * 100)}% |
                        <b>Zanshin:</b> {int(p_norm.get('weights', {}).get('zanshin', 0.15) * 100)}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c_card3:
            st.markdown(
                f"""
                <div style="background-color: #1E293B; border-left: 4px solid #8B5CF6; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                    <div style="font-weight: 700; color: #A78BFA; font-size: 1.05rem;">🟣 {p_rig.get('name', 'Rígido')}</div>
                    <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px; margin-bottom: 10px;">{p_rig.get('description', '')}</div>
                    <div style="font-size: 0.88rem; color: #E2E8F0;">
                        <b>Pontuação Mínima:</b> {int(p_rig.get('min_total_score', 0.78) * 100)}%<br>
                        <b>Alvo:</b> {int(p_rig.get('weights', {}).get('target_impact', 0.45) * 100)}% |
                        <b>Fumikomi:</b> {int(p_rig.get('weights', {}).get('fumikomi_sync', 0.25) * 100)}%<br>
                        <b>Postura:</b> {int(p_rig.get('weights', {}).get('posture', 0.15) * 100)}% |
                        <b>Zanshin:</b> {int(p_rig.get('weights', {}).get('zanshin', 0.15) * 100)}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("#### 📋 Tabela Comparativa Detalhada dos Perfis:")
        calib_table_md = """| Critério Técnico | 🟢 Permissivo (Iniciante) | 🔵 Normal (Keiko) | 🟣 Rígido (Campeonato / Dan) |
| :--- | :---: | :---: | :---: |
| **Pontuação Mínima Global** | **50%** | **65%** | **78%** |
| **Peso: Impacto no Alvo (Datotsu-bu)** | 35% | 40% | 45% |
| **Peso: Fumikomi (Sincronia Mão-Pé)** | 25% | 25% | 25% |
| **Peso: Postura Corporal** | 20% | 20% | 15% |
| **Peso: Zanshin (Vontade & Guarda)** | 20% | 15% | 15% |
| **Sub-limiar de Impacto** | 0.45 | 0.60 | 0.70 |
| **Sub-limiar de Fumikomi** | 0.35 | 0.50 | 0.60 |
| **Sub-limiar de Postura** | 0.35 | 0.50 | 0.60 |
| **Sub-limiar de Zanshin** | 0.30 | 0.45 | 0.55 |
"""
        st.markdown(calib_table_md)
        st.info("💡 **Dica de Calibração:** Durante a análise de lutas, você pode selecionar o perfil desejado ou escolher a opção **'⚙️ Personalizado'** na barra lateral para ajustar os sliders de limiares em tempo real.")

    # --------------------------------------------------------------------------
    # GUIA 4: DIAGNÓSTICO, ALERTAS & LOG DE DEBUG DO SISTEMA
    # --------------------------------------------------------------------------
    with tab_diag:
        st.markdown("### 🐛 Diagnóstico do Sistema, Alertas & Registros de Debug")
        st.caption("Rastreie alertas e erros do sistema em tempo real, execute testes de integridade e baixe o arquivo de log completo.")

        log_summary = get_log_summary()

        l_col1, l_col2, l_col3, l_col4 = st.columns(4)
        l_col1.metric("Total de Eventos Registrados", log_summary["total_logs"])
        l_col2.metric("Erros do Sistema", log_summary["errors_count"], delta_color="inverse")
        l_col3.metric("Alertas & Avisos", log_summary["warnings_count"], delta_color="inverse")
        l_col4.metric("Informações de Execução", log_summary["info_count"])

        st.markdown("#### 🛠️ Ferramentas de Diagnóstico e Rastreamento:")
        dbg_col1, dbg_col2, dbg_col3, dbg_col4 = st.columns(4)

        with dbg_col1:
            st.markdown("**📥 Log de Debug**")
            st.caption("Baixa o arquivo completo de eventos do sistema (`senpai_debug.log`).")
            debug_log_text = get_debug_log_file_content()
            st.download_button(
                label="📥 Baixar Log (.log)",
                data=debug_log_text,
                file_name=f"senpai_debug_{int(time.time())}.log",
                mime="text/plain",
                width="stretch",
                key="btn_dl_debug_log_tab"
            )

        with dbg_col2:
            st.markdown("**🧪 Diagnóstico Rápido**")
            st.caption("Verifica hardware, CUDA, integridade de arquivos e dependências.")
            if st.button("🧪 Executar Diagnóstico", type="secondary", width="stretch", key="btn_run_diag_tab"):
                with st.spinner("Executando checagem de diagnóstico do sistema..."):
                    diag_res = run_system_diagnostic_check()
                    st.success("✅ Teste de diagnóstico concluído! Alertas gravados no log.")
                    st.rerun()

        with dbg_col3:
            st.markdown("**🔬 Testes Automatizados**")
            st.caption("Executa toda a suíte de testes automatizados e salva o log detalhado.")
            if st.button("🔬 Rodar Testes (52)", type="primary", width="stretch", key="btn_run_tests_tab"):
                with st.spinner("Executando suíte de testes automatizados..."):
                    t_res = run_automated_tests(test_dir="tests", log_file=TEST_LOG_PATH, verbosity=1)
                    if t_res["success"]:
                        st.success(f"✅ Todos os {t_res['total_tests']} testes aprovados com sucesso ({t_res['duration_seconds']:.2f}s)!")
                    else:
                        st.error(f"❌ {t_res['failed']} falha(s) e {t_res['errors']} erro(s) detectados nos testes.")
                    st.rerun()

        with dbg_col4:
            st.markdown("**📥 Log dos Testes**")
            st.caption("Baixa o relatório descritivo do último teste (`senpai_test_report.log`).")
            test_log_content = get_latest_test_report_content(TEST_LOG_PATH)
            st.download_button(
                label="📥 Baixar Log Testes (.log)",
                data=test_log_content,
                file_name="senpai_test_report.log",
                mime="text/plain",
                width="stretch",
                key="btn_dl_test_rep_tab"
            )

        col_clean, _ = st.columns([1, 3])
        with col_clean:
            if st.button("🧹 Limpar Logs de Debug", type="secondary", width="stretch", key="btn_clear_logs_tab"):
                clear_debug_logs()
                st.success("✅ Histórico de logs de debug zerado com sucesso!")
                st.rerun()

        st.markdown("#### 📜 Alertas e Registros de Debug em Tempo Real:")
        filter_col1, filter_col2 = st.columns([1, 2])
        with filter_col1:
            lvl_filter = st.selectbox("Filtrar por Nível:", ["TODOS", "ERROR", "WARNING", "INFO", "DEBUG"], index=0, key="log_lvl_filter_select_tab")

        logs_display = get_memory_logs(max_entries=150, level_filter=lvl_filter)

        with st.container(height=350):
            if not logs_display:
                st.info("Nenhum registro de log encontrado para o filtro selecionado.")
            else:
                for entry in logs_display:
                    timestamp_str = entry.get("timestamp", "")
                    level_str = entry.get("level", "INFO")
                    mod_str = entry.get("module", "sys")
                    msg_str = entry.get("message", "")

                    if level_str == "ERROR":
                        st.error(f"🔴 `[{timestamp_str}]` **[{mod_str}]** {msg_str}")
                    elif level_str in ["WARNING", "WARN"]:
                        st.warning(f"🟡 `[{timestamp_str}]` **[{mod_str}]** {msg_str}")
                    elif level_str == "DEBUG":
                        st.caption(f"⚙️ `[{timestamp_str}]` **[{mod_str}]** {msg_str}")
                    else:
                        st.markdown(f"🔵 `[{timestamp_str}]` **[{mod_str}]** {msg_str}")

    st.markdown("---")
    st.info("💡 **Dica:** Para alterar os parâmetros de hardware ou perfis, selecione as abas acima. Para iniciar a análise de lutas, navegue no menu lateral até **'⚔️ Análise de Lutas'**.")


# ==============================================================================
# PÁGINA 2: ANÁLISE DE LUTAS (PÁGINA PRINCIPAL)
# ==============================================================================
else:
    # --- SIDEBAR DA ANÁLISE: SELEÇÃO DOS 3 MODOS DE OPERAÇÃO ---
    st.sidebar.markdown("### 🕹️ Modo de Operação")
    app_mode = st.sidebar.radio(
        "Selecione o Modo de Operação",
        options=["realtime", "recorded", "training"],
        format_func=lambda x: {
            "realtime": "🔴 Modo de Detecção em Tempo Real",
            "recorded": "📹 Modo de Detecção Gravada",
            "training": "🎓 Modo de Treinamento & Aprendizado"
        }[x]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Aceleração de Hardware")
    saved_hw_device = get_processing_device()
    dev_pref_current = st.session_state.get("device_preference", saved_hw_device)
    effective_dev, dev_msg, dev_gpu = get_effective_device(dev_pref_current)

    if effective_dev == "gpu":
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 6px; padding: 6px 10px; margin-bottom: 4px;">
                <div style="font-weight: 700; color: #4ade80; font-size: 0.86rem;">🚀 Aceleração Ativada</div>
                <div style="font-size: 0.78rem; color: #e2e8f0; font-weight: 600;">{dev_gpu.get('gpu_name', 'NVIDIA GPU')}</div>
                <div style="font-size: 0.72rem; color: #94a3b8;">⚡ YOLOv8-Pose (CUDA)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.12); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 6px; padding: 6px 10px; margin-bottom: 4px;">
                <div style="font-weight: 700; color: #cbd5e1; font-size: 0.86rem;">💻 Aceleração Desativada</div>
                <div style="font-size: 0.78rem; color: #94a3b8;">Processamento por CPU (MediaPipe)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.sidebar.caption("⚙️ *Para alterar acelerador, acesse Menu de Configurações.*")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Calibração de Sensibilidade")
    profile_choice = st.sidebar.selectbox(
        "Perfil de Calibração Predefinido",
        options=["permissivo", "normal", "rigido", "custom"],
        index=1,
        key="sidebar_profile_selector",
        format_func=lambda x: {
            "permissivo": "Iniciantes / Educacional (Permissivo)",
            "normal": "Treino Geral / Keiko (Normal)",
            "rigido": "Campeonato / Audit de Dan (Rígido)",
            "custom": "⚙️ Personalizado (Sliders Manual)"
        }[x]
    )

    with open("config/calibration_profiles.json", "r", encoding="utf-8") as f:
        profiles_data = json.load(f)

    current_p = profiles_data.get(profile_choice, profiles_data["normal"])

    if profile_choice == "custom":
        st.sidebar.markdown("#### ⚙️ Ajuste Fino de Limiares")
        min_score_pct = st.sidebar.slider("Pontuação Mínima Global para Ponto Válido (%)", 30, 95, 65, key="custom_min_score")
        st.sidebar.markdown("**Pesos dos Critérios de Ki-Ken-Tai-Ichi:**")
        w_target = st.sidebar.slider("Peso do Impacto no Alvo", 0.0, 1.0, 0.40, key="custom_w_target")
        w_fumikomi = st.sidebar.slider("Peso do Fumikomi (Sincronia Mão-Pé)", 0.0, 1.0, 0.25, key="custom_w_fumikomi")
        w_posture = st.sidebar.slider("Peso da Postura Corporal", 0.0, 1.0, 0.20, key="custom_w_posture")
        w_zanshin = st.sidebar.slider("Peso do Zanshin", 0.0, 1.0, 0.15, key="custom_w_zanshin")
    else:
        st.sidebar.info(f"**Descrição do Perfil:**\n{current_p.get('description', '')}")
        min_score_pct = int(round(current_p.get("min_total_score", 0.65) * 100))
        weights = current_p.get("weights", {})
        w_target = float(weights.get("target_impact", 0.40))
        w_fumikomi = float(weights.get("fumikomi_sync", 0.25))
        w_posture = float(weights.get("posture", 0.20))
        w_zanshin = float(weights.get("zanshin", 0.15))

        st.sidebar.markdown("#### 🔒 Limiares do Perfil (Fixos)")
        st.sidebar.slider(
            "Pontuação Mínima Global para Ponto Válido (%)",
            min_value=30,
            max_value=95,
            value=min_score_pct,
            disabled=True,
            key=f"disabled_min_score_{profile_choice}"
        )
        st.sidebar.markdown("**Pesos dos Critérios de Ki-Ken-Tai-Ichi:**")
        st.sidebar.slider(
            "Peso do Impacto no Alvo",
            min_value=0.0,
            max_value=1.0,
            value=w_target,
            disabled=True,
            key=f"disabled_w_target_{profile_choice}"
        )
        st.sidebar.slider(
            "Peso do Fumikomi (Sincronia Mão-Pé)",
            min_value=0.0,
            max_value=1.0,
            value=w_fumikomi,
            disabled=True,
            key=f"disabled_w_fumikomi_{profile_choice}"
        )
        st.sidebar.slider(
            "Peso da Postura Corporal",
            min_value=0.0,
            max_value=1.0,
            value=w_posture,
            disabled=True,
            key=f"disabled_w_posture_{profile_choice}"
        )
        st.sidebar.slider(
            "Peso do Zanshin",
            min_value=0.0,
            max_value=1.0,
            value=w_zanshin,
            disabled=True,
            key=f"disabled_w_zanshin_{profile_choice}"
        )

    # BANNER DO MODO ATIVO
    if app_mode == "recorded":
        st.markdown('<div class="mode-banner-recorded">📹 <b>Modo de Detecção Gravada Ativo:</b> Análise de vídeos pré-gravados de combates de Kendo, detecção de Yuko-Datotsu e relatórios diagnósticos.</div>', unsafe_allow_html=True)
    elif app_mode == "training":
        st.markdown('<div class="mode-banner-training">🎓 <b>Modo de Treinamento & Aprendizado Ativo:</b> Anotação por reforço (TP, FP, FN), calibração por Dan/nível de graduação e otimização adaptativa dos perfis técnicos.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-banner-realtime">🔴 <b>Modo de Detecção em Tempo Real Ativo:</b> Processamento instantâneo de vídeo ao vivo via Webcam ou Câmeras IP (RTSP/RTCP) com sinalização em tempo real.</div>', unsafe_allow_html=True)

    # ==========================================================================
    # MODO 3: DETECÇÃO EM TEMPO REAL MULTI-CÂMERAS (1 A 4 CÂMERAS)
    # ==========================================================================
    if app_mode == "realtime":
        st.subheader("🔴 Detecção em Tempo Real Multi-Câmeras (1 a 4 Câmeras)")

        # Obter lista de câmeras detectadas no sistema
        detected_cams = detect_connected_cameras()

        col_rt_config, col_rt_diagram = st.columns([6, 5])

        with col_rt_config:
            st.markdown("##### 📹 1. Seleção e Configuração das Câmeras")
            num_cameras = st.radio(
                "Quantidade de Câmeras Simultâneas:",
                options=[1, 2, 3, 4],
                index=0,
                horizontal=True,
                key="rt_num_cameras_radio",
                format_func=lambda x: f"{x} Câmera{'s' if x > 1 else ''}"
            )

            st.markdown("**Configuração Individual por Câmera:**")
            cam_configs = []

            for k in range(num_cameras):
                st.markdown(f"**📷 Câmera {k + 1}:**")
                row_c1, row_c2 = st.columns([1.2, 2.2])

                with row_c1:
                    src_type = st.selectbox(
                        f"Tipo de Fonte (Câmera {k + 1}):",
                        options=["webcam", "rtsp"],
                        index=0,
                        key=f"rt_src_type_row_{k}",
                        format_func=lambda x: "🎥 Webcam Local" if x == "webcam" else "📡 Stream RTSP / IP",
                        label_visibility="collapsed"
                    )
                with row_c2:
                    if src_type == "webcam":
                        webcam_opts = [c["label"] for c in detected_cams] + ["➕ Outro Índice Manual..."]
                        default_idx = min(k, len(detected_cams) - 1) if detected_cams else 0
                        selected_cam_label = st.selectbox(
                            f"Dispositivo de Vídeo (Câmera {k + 1}):",
                            options=webcam_opts,
                            index=default_idx,
                            key=f"rt_webcam_select_row_{k}",
                            label_visibility="collapsed"
                        )
                        if selected_cam_label == "➕ Outro Índice Manual...":
                            cam_idx_val = st.number_input(
                                f"Índice Numérico (Câmera {k + 1}):",
                                min_value=0,
                                max_value=10,
                                value=k,
                                key=f"rt_manual_idx_row_{k}",
                                label_visibility="collapsed"
                            )
                            cam_val = int(cam_idx_val)
                            cam_name_display = f"Webcam (Índice {cam_val})"
                        else:
                            found_cam = next((c for c in detected_cams if c["label"] == selected_cam_label), None)
                            cam_val = found_cam["index"] if found_cam else k
                            cam_name_display = found_cam["name"] if found_cam else f"Webcam {k}"
                    else:
                        rtsp_val = st.text_input(
                            f"Endereço Stream RTSP/RTCP (Câmera {k + 1}):",
                            value=f"rtsp://192.168.1.{100 + k}:554/live.sdp",
                            key=f"rt_rtsp_url_row_{k}",
                            label_visibility="collapsed"
                        )
                        cam_val = rtsp_val.strip()
                        cam_name_display = f"RTSP (Cam {k + 1})"

                    cam_configs.append({
                        "id": k + 1,
                        "type": src_type,
                        "source": cam_val,
                        "label": cam_name_display
                    })

        with col_rt_diagram:
            diagram_map = {
                1: ("assets/camera_layouts/1camdisp.png", "📐 Posicionamento: 1 Câmera (Visão Lateral Principal)"),
                2: ("assets/camera_layouts/2camdisp.png", "📐 Posicionamento: 2 Câmeras (Visões Laterais Opostas em Linha)"),
                3: ("assets/camera_layouts/3camdisp.png", "📐 Posicionamento: 3 Câmeras em Pirâmide (Topo/Frontal + 2 Laterais)"),
                4: ("assets/camera_layouts/4camdisp.png", "📐 Posicionamento: 4 Câmeras em Quadrado 2x2 (4 Cantos do Shiai-jo)")
            }
            img_rel_path, img_title = diagram_map[num_cameras]
            img_filename = img_rel_path if os.path.exists(img_rel_path) else os.path.basename(img_rel_path)
            st.markdown(f"##### {img_title}")
            if os.path.exists(img_filename):
                st.image(img_filename, caption=f"Disposição recomendada no Shiai-jo para {num_cameras} câmera{'s' if num_cameras > 1 else ''}", width="stretch")
            else:
                st.info(f"Instruções de posicionamento no Shiai-jo para {num_cameras} câmera(s).")

        st.markdown("---")
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            run_live_detection = st.checkbox("▶️ Iniciar Transmissão Ao Vivo Multi-Câmeras", value=False, key="run_multi_live_detection")
        with col_ctrl2:
            st.caption("💡 *Marque para ativar o processamento em tempo real de todas as câmeras. Desmarque a qualquer momento para pausar.*")

        if run_live_detection:
            dev_pref = st.session_state.get("device_preference", get_processing_device())
            pipeline = SenpAIPipeline(
                calibration_profile=profile_choice if profile_choice != "custom" else "normal",
                device_preference=dev_pref
            )

            active_profile_str = profile_choice if profile_choice != "custom" else "normal"
            pipeline.multicam_fusion.profile_name = active_profile_str

            col_live_cams, col_live_feed = st.columns([7, 5])

            with col_live_feed:
                st.markdown("##### 📊 Feed de Golpes & Painel de Métricas")
                fps_metric = st.empty()
                strike_alert_box = st.empty()
                st.markdown("**Histórico de Golpes Detectados na Sessão:**")
                live_events_container = st.container(height=420)

            with col_live_cams:
                st.markdown(f"##### 🎥 Feeds de Vídeo ({num_cameras} Câmera{'s' if num_cameras > 1 else ''})")
                frame_placeholders = []
                # 1 Câmera: Única
                if num_cameras == 1:
                    frame_placeholders.append(st.empty())
                # 2 Câmeras: Em linha
                elif num_cameras == 2:
                    c1, c2 = st.columns(2)
                    frame_placeholders.append(c1.empty())
                    frame_placeholders.append(c2.empty())
                # 3 Câmeras: Pirâmide (1 topo + 2 base)
                elif num_cameras == 3:
                    top_col1, top_col2, top_col3 = st.columns([1, 6, 1])
                    top_ph = top_col2.empty()
                    bot_col1, bot_col2 = st.columns(2)
                    bot1_ph = bot_col1.empty()
                    bot2_ph = bot_col2.empty()
                    frame_placeholders.extend([top_ph, bot1_ph, bot2_ph])
                # 4 Câmeras: Quadrado 2x2
                elif num_cameras == 4:
                    r1_c1, r1_c2 = st.columns(2)
                    r2_c1, r2_c2 = st.columns(2)
                    frame_placeholders.extend([r1_c1.empty(), r1_c2.empty(), r2_c1.empty(), r2_c2.empty()])

            # Abrir conexões de captura de vídeo para cada câmera
            caps = []
            for cfg in cam_configs:
                src = cfg["source"]
                if isinstance(src, int) and hasattr(cv2, "CAP_DSHOW"):
                    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(src)
                caps.append(cap)

            open_indices = [i for i, c in enumerate(caps) if c.isOpened()]
            if not open_indices:
                st.error("❌ Não foi possível conectar a nenhuma das câmeras configuradas. Verifique conexões e permissões.")
            else:
                live_pose_histories = [[] for _ in range(num_cameras)]
                latest_drawn_frames = [None for _ in range(num_cameras)]
                frame_count = 0
                start_time = time.time()
                current_fps = 30.0

                while run_live_detection:
                    any_frame_read = False

                    for k in range(num_cameras):
                        cap = caps[k]
                        if not cap.isOpened():
                            continue
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        any_frame_read = True

                        # Processar Pose Tracking na câmera k
                        if num_cameras == 1:
                            candidates, _ = pipeline.pose_detector.process_frame_candidates(frame)
                            aka_lm, shiro_lm, disc = pipeline.combatant_tracker.associate_and_filter(candidates, frame=frame)
                            drawn_frame = pipeline.pose_detector.draw_combatants_overlay(
                                frame,
                                aka_landmarks=aka_lm,
                                shiro_landmarks=shiro_lm,
                                discarded_items=disc,
                                sonkyo_status="🔴 AO VIVO | ⚪ SHIRO (ESQUERDA) ⇄ AKA (DIREITA) 🔴"
                            )
                            active_lm = aka_lm or shiro_lm
                            live_pose_histories[0].append(active_lm)
                            latest_drawn_frames[0] = drawn_frame
                        else:
                            landmarks, drawn_frame = pipeline.pose_detector.process_frame(frame)
                            live_pose_histories[k].append(landmarks)
                            latest_drawn_frames[k] = drawn_frame

                        # Exibir frame anotado
                        frame_rgb = cv2.cvtColor(drawn_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholders[k].image(
                            frame_rgb,
                            caption=f"📷 Câmera {k + 1}: {cam_configs[k]['label']}",
                            channels="RGB",
                            width="stretch"
                        )

                    if not any_frame_read:
                        st.warning("⚠️ Nenhuma transmissão ativa ou sinal de vídeo interrompido.")
                        break

                    # Avaliação conjunta do golpe pelo conjunto de imagens das câmeras (processado em background)
                    if frame_count % 3 == 0 and any(len(h) >= 15 for h in live_pose_histories):
                        multicam_eval = pipeline.multicam_fusion.evaluate_live_step(
                            live_pose_histories=live_pose_histories,
                            camera_configs=cam_configs,
                            current_fps=current_fps or 30.0,
                            current_frame_idx=frame_count,
                            latest_frames=latest_drawn_frames
                        )

                        if multicam_eval and multicam_eval.is_strike_confirmed:
                            ts_str = multicam_eval.timestamp_ref
                            strike_alert_box.error(f"🚨 GOLPE DETECTADO: **{multicam_eval.technique}** (Timestamp: `{ts_str}`)")
                            with live_events_container:
                                st.markdown(f"🥊 **{multicam_eval.technique}** no tempo `{ts_str}` (Frame #{frame_count})")

                    frame_count += 1
                    elapsed = time.time() - start_time
                    current_fps = (frame_count * len(open_indices)) / elapsed if elapsed > 0 else 0.0
                    fps_metric.metric(
                        "Desempenho Multi-Câmeras Ao Vivo",
                        f"{current_fps:.1f} FPS",
                        f"Câmeras Ativas: {len(open_indices)}/{num_cameras}"
                    )

            # Liberar todas as câmeras ao finalizar
            for cap in caps:
                try:
                    cap.release()
                except Exception:
                    pass


    # ==========================================================================
    # MODOS 1 E 2: DETECÇÃO GRAVADA & TREINAMENTO & APRENDIZADO
    # ==========================================================================
    else:
        expander_title = "🥋 Carregar Vídeo de Treinamento & Aprendizado" if app_mode == "training" else "📹 Carregar Vídeo da Luta"
        with st.expander(expander_title, expanded=("analysis_result" not in st.session_state)):
            col_in1, col_in2 = st.columns([1, 1])
            video_file_path = st.session_state.get("video_file_path", None)
            
            with col_in1:
                st.subheader("🥋 Carregar Vídeo de Treino" if app_mode == "training" else "📹 Carregar Vídeo")
                source_choice = st.radio(
                    "Selecione a Origem do Vídeo:",
                    ["📁 Fazer Upload de Arquivo", "🌐 Link do YouTube / Streaming Web"],
                    horizontal=True,
                    key="recorded_source_choice"
                )

                if source_choice == "📁 Fazer Upload de Arquivo":
                    st.markdown("Selecione o arquivo de vídeo local do treinamento de Kendo a ser analisado:" if app_mode == "training" else "Selecione o arquivo de vídeo local da luta de Kendo a ser analisado:")
                    uploaded_file = st.file_uploader(
                        "Vídeo de Treinamento (.mp4, .avi, .mov)" if app_mode == "training" else "Vídeo da Luta (.mp4, .avi, .mov)",
                        type=["mp4", "avi", "mov"],
                        help="Suporta arquivos de vídeo de qualquer tamanho. Arquivos acima de 200MB podem ter tempos de carregamento extensos.",
                        key="recorded_local_file_uploader"
                    )
                    if uploaded_file is not None:
                        cached_file_name = st.session_state.get("uploaded_file_name")
                        cached_file_size = st.session_state.get("uploaded_file_size")
                        cached_file_path = st.session_state.get("video_file_path")

                        # Reutilizar o arquivo salvo caso seja exatamente o mesmo upload
                        if cached_file_path and os.path.exists(cached_file_path) and cached_file_name == uploaded_file.name and cached_file_size == uploaded_file.size:
                            video_file_path = cached_file_path
                        else:
                            # Limpar arquivo temporário anterior se existir
                            if cached_file_path and os.path.exists(cached_file_path) and ("senpai_uploads" in cached_file_path or "tmp" in cached_file_path):
                                try:
                                    os.remove(cached_file_path)
                                except Exception:
                                    pass

                            uploads_dir = os.path.join(tempfile.gettempdir(), "senpai_uploads")
                            os.makedirs(uploads_dir, exist_ok=True)
                            
                            # Limpar arquivos temporários antigos de sessões anteriores
                            try:
                                now = time.time()
                                for old_f in os.listdir(uploads_dir):
                                    f_p = os.path.join(uploads_dir, old_f)
                                    if os.path.isfile(f_p) and (now - os.path.getmtime(f_p) > 3600):
                                        os.remove(f_p)
                            except Exception:
                                pass

                            safe_filename = f"upload_{int(time.time())}_{uploaded_file.name}"
                            target_file_path = os.path.join(uploads_dir, safe_filename)

                            uploaded_file.seek(0)
                            with open(target_file_path, "wb") as f_out:
                                while True:
                                    chunk = uploaded_file.read(8 * 1024 * 1024)
                                    if not chunk:
                                        break
                                    f_out.write(chunk)

                            video_file_path = target_file_path
                            st.session_state["video_file_path"] = video_file_path
                            st.session_state["uploaded_file_name"] = uploaded_file.name
                            st.session_state["uploaded_file_size"] = uploaded_file.size
                            st.session_state["video_source_type"] = "upload"
                            st.session_state.pop("youtube_video_info", None)
                            st.session_state.pop("youtube_url", None)
                    else:
                        if st.session_state.get("video_source_type") == "upload":
                            if "uploaded_file_name" in st.session_state:
                                cached_file_path = st.session_state.get("video_file_path")
                                if cached_file_path and os.path.exists(cached_file_path) and ("senpai_uploads" in cached_file_path or "tmp" in cached_file_path):
                                    try:
                                        os.remove(cached_file_path)
                                    except Exception:
                                        pass
                                st.session_state.pop("uploaded_file_name", None)
                                st.session_state.pop("uploaded_file_size", None)
                                st.session_state.pop("video_file_path", None)
                                video_file_path = None

                else:
                    # Origem: Link do YouTube / Streaming Web
                    st.markdown("Insira o link de vídeo de treino de Kendo do YouTube ou stream da web:" if app_mode == "training" else "Insira o link de vídeo de Kendo do YouTube ou stream da web:")
                    yt_url_input = st.text_input(
                        "🔗 Link do Vídeo (YouTube, Shorts, Streaming):",
                        placeholder="https://www.youtube.com/watch?v=... ou https://youtu.be/...",
                        value=st.session_state.get("youtube_url", ""),
                        key="recorded_youtube_url_input"
                    )

                    yt_quality_keys = ["media", "alta", "baixa"]
                    selected_quality = st.selectbox(
                        "⚙️ Resolução / Qualidade do Download:",
                        options=yt_quality_keys,
                        format_func=lambda k: QUALITY_LABELS.get(k, k),
                        index=0,  # "media" padrão
                        key="recorded_youtube_quality_select",
                        help="• Média (Padrão): Resolução intermediária (até 720p) a 30 FPS.\n• Alta: Máxima resolução e FPS originais do vídeo.\n• Baixa: Menor resolução disponível com download mais rápido."
                    )

                    yt_loaded_path = st.session_state.get("video_file_path") if st.session_state.get("video_source_type") == "youtube" else None
                    yt_info = st.session_state.get("youtube_video_info", {})

                    col_yt_btn1, col_yt_btn2 = st.columns([1.8, 1.2])
                    with col_yt_btn1:
                        load_yt_btn = st.button(
                            "📥 Carregar Vídeo do Link" if not yt_loaded_path else "🔄 Recarregar Link",
                            type="primary" if not yt_loaded_path else "secondary",
                            width="stretch",
                            key="btn_load_recorded_youtube"
                        )
                    with col_yt_btn2:
                        clear_yt_btn = st.button(
                            "🗑️ Limpar Vídeo",
                            type="secondary",
                            width="stretch",
                            disabled=not yt_loaded_path,
                            key="btn_clear_recorded_youtube"
                        )

                    if clear_yt_btn:
                        st.session_state.pop("video_file_path", None)
                        st.session_state.pop("youtube_video_info", None)
                        st.session_state.pop("youtube_url", None)
                        st.session_state.pop("video_source_type", None)
                        st.session_state.pop("analysis_result", None)
                        st.toast("Vídeo descarregado com sucesso!", icon="🗑️")
                        st.rerun()

                    if load_yt_btn and yt_url_input:
                        if not validate_video_url(yt_url_input):
                            st.error("❌ Link inválido. Forneça uma URL válida do YouTube (ex: youtube.com/watch?v=... ou youtu.be/...) ou streaming de vídeo.")
                        else:
                            prog_bar = st.progress(0.0)
                            status_txt = st.empty()
                            def _ui_progress(pct: float, msg: str):
                                prog_bar.progress(min(1.0, max(0.0, pct)))
                                status_txt.markdown(f"<span style='color: #60a5fa; font-size: 0.88rem;'>⏳ {msg}</span>", unsafe_allow_html=True)
                            
                            try:
                                with st.spinner(f"⏳ Conectando e baixando vídeo ({QUALITY_LABELS.get(selected_quality, selected_quality)})..."):
                                    dl_path, extracted_info = download_video_stream(
                                        url=yt_url_input,
                                        quality=selected_quality,
                                        progress_callback=_ui_progress
                                    )
                                st.session_state["video_file_path"] = dl_path
                                st.session_state["youtube_video_info"] = extracted_info
                                st.session_state["youtube_url"] = yt_url_input
                                st.session_state["video_source_type"] = "youtube"
                                # Limpa upload local anterior
                                st.session_state.pop("uploaded_file_name", None)
                                st.session_state.pop("uploaded_file_size", None)
                                st.toast(f"✅ Vídeo '{extracted_info.get('title', 'Kendo')}' carregado com sucesso!", icon="🎥")
                                st.rerun()
                            except VideoDownloadError as e:
                                st.error(f"❌ {str(e)}")
                            except Exception as e:
                                st.error(f"❌ Erro ao carregar vídeo do YouTube: {str(e)}")

                    # Exibir card informativo se o vídeo do YouTube estiver carregado
                    if yt_loaded_path and yt_info and os.path.exists(yt_loaded_path):
                        video_file_path = yt_loaded_path
                        yt_thumb = yt_info.get("thumbnail", "")
                        yt_title = yt_info.get("title", "Vídeo do YouTube")
                        yt_uploader = yt_info.get("uploader", "Canal")
                        yt_dur = yt_info.get("duration_formatted", "00:00")
                        yt_res = yt_info.get("downloaded_resolution", yt_info.get("resolution", "HD"))
                        yt_fps = yt_info.get("downloaded_fps", yt_info.get("fps", 30.0))
                        yt_qual = yt_info.get("quality_label", QUALITY_LABELS.get(selected_quality, "Média (Intermediária / 30 FPS)"))
                        yt_size = yt_info.get("downloaded_file_size_mb", 0.0)
                        size_str = f" &nbsp;|&nbsp; 💾 {yt_size:.1f} MB" if yt_size > 0 else ""
                        
                        thumb_html = f'<img src="{yt_thumb}" style="width: 100%; border-radius: 6px; aspect-ratio: 16/9; object-fit: cover; border: 1px solid #3b82f6;">' if yt_thumb else ''
                        
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, rgba(30, 27, 75, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%); border: 1px solid #4f46e5; border-radius: 8px; padding: 10px 14px; margin-top: 10px;">
                                <div style="display: flex; gap: 12px; align-items: center;">
                                    <div style="flex: 0 0 110px;">
                                        {thumb_html}
                                    </div>
                                    <div style="flex: 1; min-width: 0;">
                                        <span style="background: #ef4444; color: white; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px;">▶ YOUTUBE / STREAM</span>
                                        <div style="color: #f8fafc; font-weight: 600; font-size: 0.90rem; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{yt_title}">{yt_title}</div>
                                        <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 2px;">
                                            👤 {yt_uploader} &nbsp;|&nbsp; ⏱️ {yt_dur}{size_str}
                                        </div>
                                        <div style="background: rgba(99, 102, 241, 0.18); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 4px; padding: 3px 8px; margin-top: 4px; font-size: 0.76rem; color: #c7d2fe;">
                                            ⚙️ <b>Qualidade Baixada:</b> <span style="color: #38bdf8; font-weight: 700;">{yt_qual}</span> &nbsp;•&nbsp; 📐 <b>Resolução:</b> {yt_res} @ {yt_fps:.0f} FPS
                                        </div>
                                        <div style="color: #34d399; font-size: 0.78rem; font-weight: 600; margin-top: 4px;">
                                            ✅ Vídeo pronto para execução da análise
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            with col_in2:
                st.subheader("🥋 Executar Análise de Treinamento" if app_mode == "training" else "⚡ Executar Análise de Combate")
                st.markdown("Inicie o rastreamento de pose, identificação da modalidade e avaliação dos 3 Pilares (Movimentação, Precisão, Constância):" if app_mode == "training" else "Inicie o rastreamento de pose, detecção de impactos e avaliação de Yuko-Datotsu:")

                dev_pref = st.session_state.get("device_preference", get_processing_device())
                effective_dev, dev_msg, dev_gpu = get_effective_device(dev_pref)
                if effective_dev == "gpu":
                    st.markdown(f'<div style="background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem; color: #4ade80;">🚀 <b>Aceleração GPU Ativa:</b> {dev_gpu.get("gpu_name", "NVIDIA GPU")} (YOLOv8-Pose CUDA)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background: rgba(148, 163, 184, 0.12); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem; color: #cbd5e1;">💻 <b>Processamento por CPU:</b> MediaPipe Pose (TFLite CPU)</div>', unsafe_allow_html=True)

                if app_mode == "training":
                    st.markdown("##### 🥋 Modalidade de Treino & Aprendizado (14 Modalidades Oficiais)")
                    mod_keys = ["auto"] + list(TRAINING_MODALITIES_METADATA.keys())
                    selected_mod_key = st.selectbox(
                        "Tipo de Treinamento:",
                        options=mod_keys,
                        format_func=lambda k: "🔍 Detecção Inteligente Automática pela IA" if k == "auto" else f"{TRAINING_MODALITIES_METADATA[k]['name']} — {TRAINING_MODALITIES_METADATA[k]['category']}",
                        index=0,
                        key="training_modality_select_pre"
                    )
                    st.session_state["training_modality_selected"] = None if selected_mod_key == "auto" else selected_mod_key
                    if selected_mod_key != "auto":
                        m_desc = TRAINING_MODALITIES_METADATA[selected_mod_key]
                        st.caption(f"ℹ️ **{m_desc['category']}:** {m_desc['description']}")

                active_worker = st.session_state.get("analysis_worker", None)
                is_running = (active_worker is not None and not active_worker.is_done)

                # Alerta visual caso o processamento anterior tenha sido interrompido
                if st.session_state.get("processing_cancelled", False) and not is_running:
                    st.warning("⚠️ O processamento foi interrompido pelo usuário.")
                    st.session_state["processing_cancelled"] = False

                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    btn_start_label = "⚡ Executar Análise de Treinamento" if app_mode == "training" else "⚡ Executar Análise com SenpAI"
                    start_btn = st.button(
                        btn_start_label if not is_running else "⏳ Processando Vídeo...",
                        type="primary",
                        width="stretch",
                        disabled=(video_file_path is None or is_running),
                        key="btn_start_recorded_analysis"
                    )
                with col_btn2:
                    stop_btn = st.button(
                        "⏹️ Interromper Processamento",
                        type="secondary",
                        width="stretch",
                        disabled=not is_running,
                        key="btn_stop_recorded_analysis"
                    )

                # 1. Se o usuário clicar no botão de Interromper
                if stop_btn and active_worker:
                    elapsed_cancel = active_worker.elapsed_seconds
                    active_worker.cancel()
                    st.session_state.pop("analysis_worker", None)
                    st.session_state["processing_cancelled"] = True
                    annotated_out = "annotated_match.mp4"
                    if os.path.exists(annotated_out):
                        try:
                            os.remove(annotated_out)
                        except Exception:
                            pass
                    log_event("WARNING", f"Processamento de vídeo interrompido via botão pelo usuário após {elapsed_cancel:.2f}s.", "app")
                    st.rerun()

                # 2. Se o usuário clicar em Iniciar Análise
                if start_btn and video_file_path and not is_running:
                    dev_pref = st.session_state.get("device_preference", get_processing_device())
                    pipeline = SenpAIPipeline(
                        calibration_profile=profile_choice if profile_choice != "custom" else "normal",
                        device_preference=dev_pref
                    )
                    
                    if profile_choice == "custom":
                        pipeline.calibrator.update_custom_settings(
                            min_total_score=min_score_pct / 100.0,
                            weight_target=w_target,
                            weight_fumikomi=w_fumikomi,
                            weight_posture=w_posture,
                            weight_zanshin=w_zanshin
                        )

                    annotated_output = "annotated_match.mp4"
                    worker = AnalysisWorker(
                        pipeline=pipeline,
                        video_path=video_file_path,
                        output_video_path=annotated_output,
                        invert_combatants=st.session_state.get("invert_aka_shiro", False),
                        training_modality_override=st.session_state.get("training_modality_selected"),
                        custom_kendoka_names=st.session_state.get("training_kendoka_names", {})
                    )
                    worker.start()
                    st.session_state["analysis_worker"] = worker
                    st.session_state["processing_cancelled"] = False
                    st.rerun()

                # 3. Se estiver em processamento ativo, renderiza barra de progresso e cronômetro em tempo real
                if is_running and active_worker:
                    st.progress(active_worker.progress)
                    timer_col1, timer_col2 = st.columns([1.2, 0.8])
                    with timer_col1:
                        st.info(f"⏳ **{active_worker.status_message}**")
                    with timer_col2:
                        st.markdown(
                            f"""
                            <div style="background-color: #1E1B4B; border: 1px solid #6366F1; border-radius: 8px; padding: 10px 14px; text-align: center;">
                                <span style="color: #A5B4FC; font-size: 13px; font-weight: 600; letter-spacing: 0.5px;">⏱️ TEMPO DECORRIDO</span><br>
                                <span style="color: #FFFFFF; font-size: 20px; font-weight: 700; font-family: monospace;">{active_worker.elapsed_formatted}</span>
                                <span style="color: #94A3B8; font-size: 13px;"> ({active_worker.elapsed_seconds:.1f}s)</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    time.sleep(0.25)
                    st.rerun()

                # 4. Se o processamento finalizou
                if active_worker and active_worker.is_done:
                    if active_worker.is_cancelled:
                        st.session_state.pop("analysis_worker", None)
                        st.session_state["processing_cancelled"] = True
                        annotated_out = "annotated_match.mp4"
                        if os.path.exists(annotated_out):
                            try:
                                os.remove(annotated_out)
                            except Exception:
                                pass
                        st.rerun()
                    elif active_worker.error:
                        err_msg = active_worker.error
                        st.session_state.pop("analysis_worker", None)
                        st.error(f"❌ Erro durante o processamento de vídeo: {err_msg}")
                    else:
                        res = active_worker.result
                        st.session_state["analysis_result"] = res
                        st.session_state["annotated_output"] = active_worker.output_video_path
                        st.session_state["last_processing_time"] = res.get("processing_time_seconds", round(active_worker.elapsed_seconds, 2))
                        st.session_state["last_processing_fps"] = res.get("processing_fps", round(res.get("total_frames", 0) / max(0.001, active_worker.elapsed_seconds), 1))
                        st.session_state.pop("analysis_worker", None)
                        st.toast(f"✅ Processamento finalizado em {st.session_state['last_processing_time']:.2f}s!", icon="⏱️")
                        st.rerun()

                # 5. Painel fixo de tempo de processamento mantido na tela após finalização
                if "analysis_result" in st.session_state and not is_running:
                    res_p = st.session_state["analysis_result"]
                    proc_time = res_p.get("processing_time_seconds", st.session_state.get("last_processing_time", 0.0))
                    proc_fps = res_p.get("processing_fps", st.session_state.get("last_processing_fps", 0.0))
                    st.markdown(
                        f"""
                        <div style="background-color: #0F2E1B; border: 1px solid #10B981; border-radius: 8px; padding: 8px 14px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #A7F3D0; font-size: 14px; font-weight: 500;">
                                ✅ <b>Processamento Concluído</b> ({res_p.get('total_frames', 0)} quadros analisados)
                            </span>
                            <span style="color: #FFFFFF; font-size: 14px; font-weight: 700; font-family: monospace;">
                                ⏱️ Tempo Total: <span style="color: #34D399;">{proc_time:.2f}s</span> &nbsp;|&nbsp; ⚡ Taxa: <span style="color: #34D399;">{proc_fps:.1f} FPS</span>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        video_file_path = st.session_state.get("video_file_path", None)

        # PAINEL PRINCIPAL DE RESULTADOS
        if video_file_path or "analysis_result" in st.session_state:
            st.markdown("---")

            # 0. SE EXISTIR RESULTADO: PLACAR OFICIAL (SANBON-SHOBU) NO TOPO DO PAINEL
            enable_editing = st.session_state.get("editing_enabled", False)
            selected_dan = 3
            dan_options = {
                1: "1º Dan (Shodan)",
                2: "2º Dan (Nidan)",
                3: "3º Dan (Sandan)",
                4: "4º Dan (Yondan)",
                5: "5º Dan (Godan)",
                6: "6º Dan (Rokudan)",
                7: "7º Dan (Nanadan)",
                8: "8º Dan (Hachidan)"
            }
            if "session_reviews" not in st.session_state:
                st.session_state["session_reviews"] = {}
            if "sonkyo_edits" not in st.session_state:
                st.session_state["sonkyo_edits"] = {}

            session_revs = st.session_state.get("session_reviews", {})
            sonkyo_edits = st.session_state.get("sonkyo_edits", {})
            is_inverted = st.session_state.get("invert_aka_shiro", False)
            if st.session_state.get("video_source_type") == "youtube" and "youtube_video_info" in st.session_state:
                video_name_simple = st.session_state["youtube_video_info"].get("title", os.path.basename(video_file_path) if video_file_path else "youtube_match.mp4")
            else:
                video_name_simple = os.path.basename(video_file_path) if video_file_path else "recorded_match.mp4"

            if "analysis_result" in st.session_state:
                res = st.session_state["analysis_result"]
                raw_scoreboard = res.get("scoreboard", {})

                # Extração dos golpes válidos (Ippon) considerando revisões ativas da sessão
                raw_aka_strikes = []
                raw_shiro_strikes = []

                # 1. Golpes detectados automaticamente pelo modelo
                for ev_i, ev_d in enumerate(res.get("events", [])):
                    ev_info_d = ev_d["event_info"]
                    ev_id_d = f"event_{ev_i+1}_frame_{ev_info_d['impact_frame']}"
                    rev_d = session_revs.get(ev_id_d)

                    if rev_d:
                        if rev_d.get("is_edited"):
                            is_valid_d = (rev_d.get("category") == "VALID_IPPON")
                        elif rev_d.get("is_confirmed"):
                            is_valid_d = ev_d["evaluation"].get("is_valid", False)
                        else:
                            is_valid_d = (rev_d.get("label") == "TP" and rev_d.get("category") not in ["INVALID_HIT", "NO_STRIKE"])
                    else:
                        is_valid_d = ev_d["evaluation"].get("is_valid", False)

                    if is_valid_d:
                        if ev_info_d.get("attacker_id") == "KENSHI_AKA":
                            raw_aka_strikes.append(ev_d)
                        else:
                            raw_shiro_strikes.append(ev_d)

                # 2. Golpes adicionais incluídos manualmente pelo usuário
                for fn_k, fn_v in session_revs.items():
                    if fn_v.get("is_included"):
                        is_fn_ippon = fn_v.get("is_valid_ippon", fn_v.get("category") == "VALID_IPPON")
                        if is_fn_ippon:
                            fake_ev = {
                                "event_info": {
                                    "attacker_id": fn_v.get("attacker_id", "KENSHI_AKA"),
                                    "attacker_name": fn_v.get("attacker_name", "Kenshi Aka (Vermelho)"),
                                    "type": fn_v.get("strike_type", "MEN"),
                                    "timestamp": fn_v.get("timestamp", "00:00.000"),
                                    "impact_frame": 0
                                },
                                "evaluation": {"is_valid": True, "total_score": 100.0}
                            }
                            if fn_v.get("attacker_id") == "KENSHI_AKA":
                                raw_aka_strikes.append(fake_ev)
                            else:
                                raw_shiro_strikes.append(fake_ev)

                if not is_inverted:
                    aka_val_strikes = raw_aka_strikes
                    shiro_val_strikes = raw_shiro_strikes
                else:
                    aka_val_strikes = raw_shiro_strikes
                    shiro_val_strikes = raw_aka_strikes

                aka_score_val = len(aka_val_strikes)
                shiro_score_val = len(shiro_val_strikes)

                if aka_score_val > shiro_score_val:
                    winner_txt = f"🏆 Vitória de Kenshi Aka (Vermelho) [{aka_score_val} - {shiro_score_val}]"
                    winner_bg = "rgba(239, 68, 68, 0.18)"
                    winner_border = "#EF4444"
                    winner_color = "#FCA5A5"
                elif shiro_score_val > aka_score_val:
                    winner_txt = f"🏆 Vitória de Kenshi Shiro (Branco) [{shiro_score_val} - {aka_score_val}]"
                    winner_bg = "rgba(243, 244, 246, 0.15)"
                    winner_border = "#E5E7EB"
                    winner_color = "#F3F4F6"
                else:
                    winner_txt = f"🤝 Empate (Hikiwake) [{aka_score_val} - {shiro_score_val}]"
                    winner_bg = "rgba(148, 163, 184, 0.15)"
                    winner_border = "#64748B"
                    winner_color = "#CBD5E1"

                flag_info = raw_scoreboard.get("flag_detection", {})
                flag_dec = flag_info.get("flag_decision", "POSITION_DEFAULT")
                flag_conf = int(flag_info.get("confidence", 0.5) * 100)

                if "RIGHT" in flag_dec:
                    flag_badge = f"🚩 Flag Vermelha (Tasukuki) detectada nas costas do lutador à direita ({flag_conf}%)"
                elif "LEFT" in flag_dec:
                    flag_badge = f"🚩 Flag Vermelha (Tasukuki) detectada nas costas do lutador à esquerda ({flag_conf}%)"
                else:
                    flag_badge = "🚩 Identificação por posição inicial no Shiaijo"

                if is_inverted:
                    flag_badge += " • 🔄 Lados Invertidos Manualmente"

                # HTML de Ippons do Aka
                if aka_val_strikes:
                    aka_items = "".join([f'<span style="display:inline-block; background: #991B1B; color: #FEE2E2; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; margin: 2px;">🔴 {format_scoreboard_strike_name(s["event_info"]["type"])} ({s["event_info"]["timestamp"]})</span>' for s in aka_val_strikes])
                else:
                    aka_items = '<span style="color: #9CA3AF; font-size: 12px; font-style: italic;">Nenhum Ippon detectado</span>'

                # HTML de Ippons do Shiro
                if shiro_val_strikes:
                    shiro_items = "".join([f'<span style="display:inline-block; background: #475569; color: #F8FAFC; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; margin: 2px;">⚪ {format_scoreboard_strike_name(s["event_info"]["type"])} ({s["event_info"]["timestamp"]})</span>' for s in shiro_val_strikes])
                else:
                    shiro_items = '<span style="color: #9CA3AF; font-size: 12px; font-style: italic;">Nenhum Ippon validado</span>'

                # RENDERIZAÇÃO PRINCIPAL DO TOPO CONFORME O MODO DE OPERAÇÃO
                if app_mode == "training":
                    # MODO DE TREINAMENTO: PAINEL DE 10 MODALIDADES, 3 PILARES E PRESCRIÇÃO PEDAGÓGICA
                    render_training_analysis_view(res, is_inverted)
                    with st.expander("⚔️ Visualizar Simulação de Placar Sanbon-Shobu (Competitivo)", expanded=False):
                        st.markdown(
                            f"""
                            <div style="background: #090D16; border: 2px solid #374151; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
                                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1F2937; padding-bottom: 8px; margin-bottom: 12px;">
                                    <span style="color: #D1D5DB; font-size: 13px; font-weight: 800; letter-spacing: 0.8px;">🥋 PONTUAÇÃO FINAL (SANBON-SHOBU)</span>
                                    <span style="color: #93C5FD; font-size: 12px; font-weight: 500;">{flag_badge}</span>
                                </div>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                                    <div style="background: linear-gradient(180deg, rgba(243, 244, 246, 0.10) 0%, rgba(100, 116, 139, 0.18) 100%); border: 1.5px solid #E5E7EB; border-radius: 8px; padding: 12px; text-align: center;">
                                        <div style="color: #F3F4F6; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;">⚪ KENSHI SHIRO (BRANCO - ESQUERDA)</div>
                                        <div style="color: #FFFFFF; font-size: 38px; font-weight: 900; font-family: monospace; line-height: 1.1; margin: 4px 0;">{shiro_score_val} <span style="font-size: 14px; font-weight: 700; color: #E5E7EB;">IPPON</span></div>
                                        <div style="margin-top: 6px;">{shiro_items}</div>
                                    </div>
                                    <div style="background: linear-gradient(180deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.22) 100%); border: 1.5px solid #EF4444; border-radius: 8px; padding: 12px; text-align: center;">
                                        <div style="color: #FCA5A5; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;">🔴 KENSHI AKA (VERMELHO - DIREITA)</div>
                                        <div style="color: #FFFFFF; font-size: 38px; font-weight: 900; font-family: monospace; line-height: 1.1; margin: 4px 0;">{aka_score_val} <span style="font-size: 14px; font-weight: 700; color: #FCA5A5;">IPPON</span></div>
                                        <div style="margin-top: 6px;">{aka_items}</div>
                                    </div>
                                </div>
                                <div style="background: {winner_bg}; border: 1px solid {winner_border}; border-radius: 6px; padding: 8px; margin-top: 10px; text-align: center;">
                                    <span style="color: {winner_color}; font-size: 15px; font-weight: 800;">{winner_txt}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    # MODO GRAVADO / COMPETITIVO: PLACAR SANBON-SHOBU NO TOPO
                    st.markdown(
                        f"""
                        <div style="background: #090D16; border: 2px solid #374151; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1F2937; padding-bottom: 8px; margin-bottom: 12px;">
                                <span style="color: #D1D5DB; font-size: 13px; font-weight: 800; letter-spacing: 0.8px;">🥋 PONTUAÇÃO FINAL (SANBON-SHOBU)</span>
                                <span style="color: #93C5FD; font-size: 12px; font-weight: 500;">{flag_badge}</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                                <div style="background: linear-gradient(180deg, rgba(243, 244, 246, 0.10) 0%, rgba(100, 116, 139, 0.18) 100%); border: 1.5px solid #E5E7EB; border-radius: 8px; padding: 12px; text-align: center;">
                                    <div style="color: #F3F4F6; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;">⚪ KENSHI SHIRO (BRANCO - ESQUERDA)</div>
                                    <div style="color: #FFFFFF; font-size: 38px; font-weight: 900; font-family: monospace; line-height: 1.1; margin: 4px 0;">{shiro_score_val} <span style="font-size: 14px; font-weight: 700; color: #E5E7EB;">IPPON</span></div>
                                    <div style="margin-top: 6px;">{shiro_items}</div>
                                </div>
                                <div style="background: linear-gradient(180deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.22) 100%); border: 1.5px solid #EF4444; border-radius: 8px; padding: 12px; text-align: center;">
                                    <div style="color: #FCA5A5; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;">🔴 KENSHI AKA (VERMELHO - DIREITA)</div>
                                    <div style="color: #FFFFFF; font-size: 38px; font-weight: 900; font-family: monospace; line-height: 1.1; margin: 4px 0;">{aka_score_val} <span style="font-size: 14px; font-weight: 700; color: #FCA5A5;">IPPON</span></div>
                                    <div style="margin-top: 6px;">{aka_items}</div>
                                </div>
                            </div>
                            <div style="background: {winner_bg}; border: 1px solid {winner_border}; border-radius: 6px; padding: 8px; margin-top: 10px; text-align: center;">
                                <span style="color: {winner_color}; font-size: 15px; font-weight: 800;">{winner_txt}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    with st.expander("🎓 Diagnóstico Pedagógico de Treinamento & 3 Pilares", expanded=False):
                        render_training_analysis_view(res, is_inverted)

                # BARRA DE CONTROLES: INVERSÃO DE LUTADORES, HABILITAR EDIÇÃO & PAINEL DAN
                col_ctrl1, col_ctrl2 = st.columns([1.6, 2.4])
                with col_ctrl1:
                    if st.button("🔄 Inverter Lutadores (Aka ⇄ Shiro)", width="stretch", key="btn_toggle_invert_aka_shiro", help="Inverte os lados de Aka e Shiro na pontuação, nos relatórios e nos eventos caso a câmera esteja invertida"):
                        st.session_state["invert_aka_shiro"] = not is_inverted
                        st.toast(f"🔄 Identidades invertidas: Aka ⇄ Shiro {'(Ativado)' if not is_inverted else '(Restaurado)'}!", icon="🔄")
                        st.rerun()
                with col_ctrl2:
                    enable_editing = st.toggle("✏️ Habilitar Edição e Revisão dos Golpes Detectados", value=st.session_state.get("editing_enabled", False), key="toggle_enable_editing")
                    st.session_state["editing_enabled"] = enable_editing

                if enable_editing:
                    rev_header_col1, rev_header_col2 = st.columns([3, 1])
                    with rev_header_col1:
                        selected_dan = st.selectbox(
                            "🥋 Graduação DAN do Revisor:",
                            options=list(dan_options.keys()),
                            format_func=lambda x: dan_options[x],
                            index=2,
                            key="reviewer_dan_select"
                        )
                    with rev_header_col2:
                        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                        if st.button("🔄 Resetar Revisão", width="stretch", help="Reseta todas as alterações de marcação e edições feitas nesta sessão"):
                            st.session_state["session_reviews"] = {}
                            st.toast("🔄 Marcações da sessão resetadas ao estado original!", icon="🔄")
                            st.rerun()

                # Banner de Reprocessamento de Sonkyō
                if sonkyo_edits:
                    st.markdown(
                        """
                        <div style="background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%); border: 2px solid #818CF8; border-radius: 10px; padding: 12px 16px; margin: 10px 0;">
                            <h4 style="color: #E0E7FF; margin: 0 0 4px 0;">⚡ Momentos de Sonkyō Alterados pelo Revisor</h4>
                            <p style="color: #C7D2FE; font-size: 0.88rem; margin: 0 0 8px 0;">
                                Os limites regulamentares de Sonkyō foram modificados. O SenpAI irá <b>aprender a movimentação corporal</b> deste combate para reprocessar a analise.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    col_rep1, col_rep2 = st.columns([3, 1])
                    with col_rep1:
                        if st.button("🔄 Reprocessar Analise com Aprendizado de Sonkyō", type="primary", width="stretch", key="btn_reprocess_sonkyo_learning"):
                            dev_pref = st.session_state.get("device_preference", get_processing_device())
                            pipeline = SenpAIPipeline(
                                calibration_profile=profile_choice if profile_choice != "custom" else "normal",
                                device_preference=dev_pref
                            )
                            if profile_choice == "custom":
                                pipeline.calibrator.update_custom_settings(
                                    min_total_score=min_score_pct / 100.0,
                                    weight_target=w_target,
                                    weight_fumikomi=w_fumikomi,
                                    weight_posture=w_posture,
                                    weight_zanshin=w_zanshin
                                )
                            annotated_output = "annotated_match.mp4"
                            worker = AnalysisWorker(
                                pipeline=pipeline,
                                video_path=video_file_path,
                                output_video_path=annotated_output,
                                initial_sonkyo_override=sonkyo_edits.get("initial"),
                                final_sonkyo_override=sonkyo_edits.get("final"),
                                invert_combatants=st.session_state.get("invert_aka_shiro", False)
                            )
                            worker.start()
                            st.session_state["analysis_worker"] = worker
                            st.session_state["sonkyo_edits"] = {}
                            st.session_state["processing_cancelled"] = False
                            st.toast("⚡ Reprocessamento iniciado com aprendizado contínuo de Sonkyō!", icon="🔄")
                            st.rerun()
                    with col_rep2:
                        if st.button("❌ Descartar Edições", width="stretch", key="btn_clear_sonkyo_edits"):
                            st.session_state["sonkyo_edits"] = {}
                            st.toast("Edições de Sonkyō descartadas!", icon="🔄")
                            st.rerun()

            # DUAS COLUNAS PERFEITAMENTE ALINHADAS LADO A LADO: VÍDEO (ESQUERDA) & LINHA DO TEMPO / GOLPES (DIREITA)
            col_video, col_results = st.columns([5, 7])
            
            with col_video:
                st.subheader("🎥 Vídeo da Luta")
                if st.session_state.get("video_source_type") == "youtube" and "youtube_url" in st.session_state:
                    yt_u = st.session_state["youtube_url"]
                    yt_inf = st.session_state.get("youtube_video_info", {})
                    yt_t = yt_inf.get("title", "Vídeo do YouTube")
                    yt_q = yt_inf.get("quality_label", "Média (Intermediária / 30 FPS)")
                    yt_r = yt_inf.get("downloaded_resolution", yt_inf.get("resolution", ""))
                    yt_f = yt_inf.get("downloaded_fps", yt_inf.get("fps", 30.0))
                    res_tag = f" &nbsp;[📐 {yt_r} @ {yt_f:.0f} FPS]" if yt_r else ""
                    st.markdown(
                        f'<div style="background: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; padding: 6px 10px; margin-bottom: 8px; font-size: 0.82rem; display: flex; justify-content: space-between; align-items: center;">'
                        f'<span style="color: #FCA5A5;">🌐 <b>Origem:</b> {yt_t[:32]}{"..." if len(yt_t) > 32 else ""} &nbsp;|&nbsp; ⚙️ <b>Qualidade:</b> <span style="color: #38bdf8; font-weight: 700;">{yt_q}</span>{res_tag}</span>'
                        f'<a href="{yt_u}" target="_blank" style="color: #93C5FD; text-decoration: underline; font-size: 0.78rem; flex-shrink: 0; margin-left: 8px;">Ver no YouTube ↗️</a>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                has_annotated = "annotated_output" in st.session_state and os.path.exists(st.session_state.get("annotated_output", ""))
                if has_annotated:
                    video_type = st.radio("Exibição do Vídeo:", ["📹 Vídeo Original", "🎥 Vídeo Anotado (Pose & Tracking)"], horizontal=True)
                    selected_video = video_file_path if "Original" in video_type else st.session_state["annotated_output"]
                else:
                    selected_video = video_file_path
                    
                if selected_video and os.path.exists(selected_video):
                    # Banner indicativo se o vídeo foi posicionado em um evento específico
                    if "video_seek_label" in st.session_state:
                        seek_lbl = st.session_state["video_seek_label"]
                        seek_time_val = st.session_state.get("video_start_time", 0.0)
                        col_sk1, col_sk2 = st.columns([3.5, 1.5])
                        with col_sk1:
                            st.markdown(
                                f'<div style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 6px; padding: 6px 10px; font-size: 0.82rem; color: #93C5FD; margin-bottom: 6px;">'
                                f'🎯 <b>Posicionado no Vídeo ({seek_time_val:.1f}s)</b><br>'
                                f'<span style="color: #E2E8F0; font-size: 0.78rem;">{seek_lbl}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        with col_sk2:
                            if st.button("✖️ Início", key="btn_reset_seek_video", width="stretch", help="Voltar a reprodução para o início"):
                                st.session_state.pop("video_start_time", None)
                                st.session_state.pop("video_seek_label", None)
                                st.rerun()

                    active_start_time = int(round(st.session_state.get("video_start_time", 0.0)))
                    st.video(
                        selected_video,
                        start_time=active_start_time,
                        autoplay=("video_start_time" in st.session_state and st.session_state["video_start_time"] > 0)
                    )
                else:
                    st.info("Nenhum vídeo disponível para reprodução.")
                    
                if "analysis_result" in st.session_state:
                    res = st.session_state["analysis_result"]
                    sonkyo_info = res.get("sonkyo_analysis", {})
                    plane_info = res.get("plane_filtering", {})
                    proc_time = res.get("processing_time_seconds", st.session_state.get("last_processing_time", 0.0))
                    proc_fps = res.get("processing_fps", st.session_state.get("last_processing_fps", 0.0))

                    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
                    st.markdown("##### 📌 Resumo do Combate & Delimitação por Sonkyō")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Duração Total", f"{res['duration_seconds']}s")
                    eff_sec = res.get('effective_combat_duration_seconds', res['duration_seconds'])
                    m2.metric("Tempo Líquido (Luta)", f"{eff_sec}s", delta=f"{eff_sec - res['duration_seconds']:.1f}s" if eff_sec < res['duration_seconds'] else None)

                    m_s1, m_s2 = st.columns(2)
                    start_ts = sonkyo_info.get("match_start_timestamp", "00:00.000")
                    end_ts = sonkyo_info.get("match_end_timestamp", f"{res['duration_seconds']}s")
                    m_s1.metric("🥋 Início (Sonkyō Inicial)", start_ts)
                    m_s2.metric("🥋 Fim (Sonkyō Final)", end_ts)

                    m3, m4 = st.columns(2)
                    m3.metric("Golpes Válidos na Janela", res['events_detected_count'])
                    m4.metric("Perfil Aplicado", res['profile_applied'])
                    
                    m5, m6 = st.columns(2)
                    dev_used = res.get('device_used', 'cpu').lower()
                    m5.metric("Processamento", "⚡ GPU NVIDIA" if dev_used == "gpu" else "💻 CPU Somente")
                    bg_disc = plane_info.get("discarded_background_count", 0)
                    fg_disc = plane_info.get("discarded_foreground_count", 0)
                    m6.metric("Planos Descartados", f"{bg_disc + fg_disc}", help=f"Segundo Plano: {bg_disc} | Frente da Câmera: {fg_disc}")

                    m_t1, m_t2 = st.columns(2)
                    m_t1.metric("⏱️ Tempo de Processamento", f"{proc_time:.2f}s")
                    m_t2.metric("⚡ Taxa de Processamento", f"{proc_fps:.1f} FPS" if proc_fps > 0 else "N/A")

                    if sonkyo_info.get("status_message"):
                        st.caption(f"🥋 **Status do Sonkyō:** {sonkyo_info['status_message']}")
                    st.caption(f"ℹ️ {res.get('device_status', '')}")
                    if st.session_state.get("video_source_type") == "youtube":
                        yt_inf = st.session_state.get("youtube_video_info", {})
                        yt_q = yt_inf.get("quality_label", "Média")
                        yt_r = yt_inf.get("downloaded_resolution", "")
                        yt_f = yt_inf.get("downloaded_fps", "")
                        fps_tag = f" @ {yt_f:.0f} FPS" if yt_f else ""
                        st.caption(f"🌐 **Fonte:** Streaming Web / YouTube &nbsp;|&nbsp; ⚙️ **Qualidade do Vídeo Baixado:** {yt_q} ({yt_r}{fps_tag})")
                    st.markdown('</div>', unsafe_allow_html=True)

            with col_results:
                st.subheader("🥋 Linha do Tempo & Revisão de Golpes")
                if "analysis_result" not in st.session_state:
                    st.info("👈 Clique em **⚡ Executar Analise** para visualizar a linha do tempo de eventos e análise detalhada.")
                else:
                    res = st.session_state["analysis_result"]
                    sonkyo_info = res.get("sonkyo_analysis", {})

                    # Montagem da lista unificada e cronológica de todos os golpes (detectados + incluídos)
                    combined_strikes = []
                    session_revs = st.session_state.get("session_reviews", {})

                    # 1. Golpes detectados pelo modelo
                    for idx_raw, ev_data in enumerate(res.get("events", [])):
                        ev = ev_data["event_info"]
                        eval_info = ev_data["evaluation"]
                        event_id_str = f"event_{idx_raw+1}_frame_{ev['impact_frame']}"

                        orig_att_name = ev.get("attacker_name", "Kenshi Aka (Vermelho)")
                        orig_att_id = ev.get("attacker_id", "KENSHI_AKA")
                        if is_inverted:
                            attacker_label = "Kenshi Shiro (Branco)" if "AKA" in orig_att_id else "Kenshi Aka (Vermelho)"
                            attacker_id = "KENSHI_SHIRO" if "AKA" in orig_att_id else "KENSHI_AKA"
                        else:
                            attacker_label = orig_att_name
                            attacker_id = orig_att_id

                        orig_is_valid = eval_info.get('is_valid', False)

                        # Estado da revisão desta marcação
                        current_rev = session_revs.get(event_id_str, {
                            "event_id": event_id_str,
                            "label": "TP" if orig_is_valid else "FP",
                            "category": "VALID_IPPON" if orig_is_valid else "INVALID_HIT",
                            "is_valid_ippon": orig_is_valid,
                            "strike_type": ev['type'],
                            "timestamp": ev['timestamp'],
                            "attacker_id": attacker_id,
                            "attacker_name": attacker_label,
                            "total_score": eval_info.get('total_score', 0.0),
                            "sub_scores": eval_info.get('sub_scores', {}),
                            "is_edited": False,
                            "is_confirmed": False,
                            "is_included": False,
                            "notes": ""
                        })
                        current_rev["attacker_name"] = attacker_label
                        current_rev["attacker_id"] = attacker_id

                        strike_ts = current_rev.get("timestamp", ev.get("timestamp", "00:00.000"))
                        combined_strikes.append({
                            "event_id": event_id_str,
                            "source": "AI_DETECTED",
                            "raw_event": ev_data,
                            "review": current_rev,
                            "timestamp": strike_ts,
                            "time_sec": parse_ts_to_seconds(strike_ts),
                            "orig_is_valid": orig_is_valid,
                            "attacker_label": attacker_label,
                            "attacker_id": attacker_id,
                            "impact_frame": ev.get("impact_frame", 0),
                            "diagnostic_report": ev_data.get("diagnostic_report", "")
                        })

                    # 2. Golpes incluídos manualmente pelo revisor
                    for fn_k, fn_v in session_revs.items():
                        if fn_v.get("is_included"):
                            ts_val = fn_v.get("timestamp", "00:00.000")
                            inc_ippon = fn_v.get("is_valid_ippon", fn_v.get("category") == "VALID_IPPON")
                            combined_strikes.append({
                                "event_id": fn_k,
                                "source": "INCLUDED",
                                "raw_event": None,
                                "review": fn_v,
                                "timestamp": ts_val,
                                "time_sec": parse_ts_to_seconds(ts_val),
                                "orig_is_valid": inc_ippon,
                                "attacker_label": fn_v.get("attacker_name", "Kenshi Aka (Vermelho)"),
                                "attacker_id": fn_v.get("attacker_id", "KENSHI_AKA"),
                                "impact_frame": 0,
                                "diagnostic_report": None
                            })

                    # Ordenação estrita cronológica pelo tempo do golpe
                    combined_strikes.sort(key=lambda s: s["time_sec"])

                    # Seletor Rápido de Navegação por Eventos
                    jump_options = ["-- 🎯 Selecionar evento para assistir no vídeo --"]
                    jump_map = {}

                    has_init_jump = sonkyo_info.get("has_initial_sonkyo", False) and sonkyo_info.get("initial_sonkyo")
                    init_edit_jump = sonkyo_edits.get("initial")
                    if has_init_jump or init_edit_jump:
                        ts_i = init_edit_jump.get("start_timestamp") if init_edit_jump else sonkyo_info.get("initial_sonkyo", {}).get("start_timestamp", "00:00.000")
                        label_i = f"🥋 Sonkyō Inicial (Abertura) @ {ts_i}"
                        jump_options.append(label_i)
                        jump_map[label_i] = (max(0.0, parse_ts_to_seconds(ts_i) - 1.0), label_i)

                    for idx_j, strike_item in enumerate(combined_strikes):
                        rev_j = strike_item["review"]
                        is_ippon_j = rev_j.get("is_valid_ippon", rev_j.get("category") == "VALID_IPPON" or rev_j.get("label") == "TP")
                        status_sym_j = "✅ Ippon" if is_ippon_j else "❌ Inválido"
                        tag_inc = " [➕ Incluído]" if strike_item["source"] == "INCLUDED" else ""
                        st_display_j = format_katakana_strike(rev_j.get('strike_type')) if is_ippon_j else rev_j.get('strike_type')
                        label_sj = f"🥊 Golpe #{idx_j+1}: {st_display_j} @ {strike_item['timestamp']} ({status_sym_j} - {strike_item['attacker_label']}){tag_inc}"
                        jump_options.append(label_sj)
                        jump_map[label_sj] = (max(0.0, strike_item["time_sec"] - 1.0), label_sj)

                    has_final_jump = sonkyo_info.get("has_final_sonkyo", False) and sonkyo_info.get("final_sonkyo")
                    final_edit_jump = sonkyo_edits.get("final")
                    if has_final_jump or final_edit_jump:
                        ts_fj = final_edit_jump.get("start_timestamp") if final_edit_jump else sonkyo_info.get("final_sonkyo", {}).get("start_timestamp", "00:00.000")
                        label_fj = f"🥋 Sonkyō Final (Encerramento) @ {ts_fj}"
                        jump_options.append(label_fj)
                        jump_map[label_fj] = (max(0.0, parse_ts_to_seconds(ts_fj) - 1.0), label_fj)

                    if len(jump_options) > 1:
                        selected_jump = st.selectbox(
                            "🎯 Navegação Rápida de Eventos no Vídeo:",
                            options=jump_options,
                            index=0,
                            key="event_quick_jump_select",
                            help="Selecione um evento para posicionar a reprodução do vídeo e facilitar a revisão."
                        )
                        if selected_jump in jump_map:
                            target_sec, target_lbl = jump_map[selected_jump]
                            if st.session_state.get("video_start_time") != target_sec or st.session_state.get("video_seek_label") != target_lbl:
                                st.session_state["video_start_time"] = target_sec
                                st.session_state["video_seek_label"] = target_lbl
                                st.toast(f"🎥 Vídeo posicionado em {target_sec:.1f}s (1s antes)!", icon="🎬")
                                st.rerun()

                    # Função interna para renderizar o inseridor inline de golpes entre eventos (+)
                    def render_inline_strike_inserter(slot_id: str, prev_time_s: float, next_time_s: float, prev_desc: str, next_desc: str):
                        if not enable_editing and app_mode != "training":
                            return
                        mid_s = max(0.0, (prev_time_s + next_time_s) / 2.0)
                        suggested_ts = format_seconds_to_ts(mid_s)

                        with st.expander(f"➕ Inserir Golpe entre {prev_desc} e {next_desc} (~{suggested_ts})", expanded=False):
                            c_in1, c_in2 = st.columns(2)
                            with c_in1:
                                ins_ts = st.text_input("Timestamp", value=suggested_ts, key=f"ins_ts_{slot_id}", help="Momento exato do golpe a ser inserido")
                                ins_type = st.selectbox("Técnica", ["MEN", "KOTE", "DO", "TSUKI"], key=f"ins_type_{slot_id}")
                            with c_in2:
                                if is_inverted:
                                    att_opts = [
                                        ("KENSHI_AKA", "🔴 Kenshi Aka (Vermelho - Esquerda)"),
                                        ("KENSHI_SHIRO", "⚪ Kenshi Shiro (Branco - Direita)")
                                    ]
                                else:
                                    att_opts = [
                                        ("KENSHI_SHIRO", "⚪ Kenshi Shiro (Branco - Esquerda)"),
                                        ("KENSHI_AKA", "🔴 Kenshi Aka (Vermelho - Direita)")
                                    ]
                                att_labels = [o[1] for o in att_opts]
                                ins_att_sel = st.selectbox("Lutador Atacante", att_labels, key=f"ins_att_{slot_id}")
                                ins_att_id = att_opts[att_labels.index(ins_att_sel)][0]
                                ins_att_name = "Kenshi Aka (Vermelho)" if ins_att_id == "KENSHI_AKA" else "Kenshi Shiro (Branco)"

                                ins_val_opts = [
                                    ("VALID_IPPON", "✅ Golpe Válido (Ippon)"),
                                    ("INVALID_HIT", "❌ Golpe Inválido (Não foi Ippon)")
                                ]
                                ins_val_labels = [v[1] for v in ins_val_opts]
                                ins_val_sel = st.radio("Validação do Golpe", ins_val_labels, horizontal=True, key=f"ins_val_{slot_id}")
                                ins_val_code = ins_val_opts[ins_val_labels.index(ins_val_sel)][0]
                                ins_is_ippon = (ins_val_code == "VALID_IPPON")

                            ins_notes = st.text_input("Observação", value=f"Golpe inserido entre {prev_desc} e {next_desc}", key=f"ins_notes_{slot_id}")

                            if st.button("💾 Adicionar Golpe Nesta Posição", key=f"btn_apply_ins_{slot_id}", type="secondary", width="stretch"):
                                new_id = f"fn_{ins_ts.replace(':', '_').replace('.', '_')}_{ins_att_id.lower()}_{slot_id}"
                                new_item = {
                                    "event_id": new_id,
                                    "label": "TP" if ins_is_ippon else "FP",
                                    "category": ins_val_code,
                                    "decision_category": ins_val_code,
                                    "is_valid_ippon": ins_is_ippon,
                                    "strike_type": ins_type,
                                    "timestamp": ins_ts,
                                    "attacker_id": ins_att_id,
                                    "attacker_name": ins_att_name,
                                    "total_score": 100.0 if ins_is_ippon else 0.0,
                                    "sub_scores": {},
                                    "is_included": True,
                                    "is_confirmed": False,
                                    "is_edited": True,
                                    "notes": ins_notes
                                }
                                st.session_state["session_reviews"][new_id] = new_item
                                feedback_mgr.save_feedback(
                                    video_name=video_name_simple,
                                    profile_key=profile_choice,
                                    event_id=new_id,
                                    label="TP" if ins_is_ippon else "FP",
                                    strike_type=ins_type,
                                    timestamp=ins_ts,
                                    notes=ins_notes,
                                    reviewer_dan=selected_dan,
                                    is_included=True,
                                    decision_category=ins_val_code
                                )
                                st.toast(f"✅ Golpe ({ins_type} de {ins_att_name} às {ins_ts}) inserido com sucesso na sequência!", icon="➕")
                                st.rerun()

                    with st.container(height=650):
                        has_initial = sonkyo_info.get("has_initial_sonkyo", False) and sonkyo_info.get("initial_sonkyo")
                        has_final = sonkyo_info.get("has_final_sonkyo", False) and sonkyo_info.get("final_sonkyo")
                        has_strikes = bool(combined_strikes)

                        if not has_initial and not has_final and not has_strikes and not sonkyo_edits:
                            st.warning("Nenhum evento (Sonkyō ou Golpes) foi identificado no vídeo.")
                        else:
                            # 1. EVENTO DE SONKYŌ INICIAL (Abertura do Combate)
                            initial_edit = sonkyo_edits.get("initial")
                            curr_end_s = 0.0
                            if has_initial or initial_edit:
                                init_s = sonkyo_info.get("initial_sonkyo") or {}
                                is_init_detected = init_s.get("is_detected", True)
                                curr_start_ts = initial_edit.get("start_timestamp") if initial_edit else init_s.get("start_timestamp", "00:00.000")
                                curr_end_ts = initial_edit.get("end_timestamp") if initial_edit else init_s.get("end_timestamp", "00:01.500")
                                curr_end_s = parse_ts_to_seconds(curr_end_ts)

                                if initial_edit:
                                    title_status = "✏️ EDITADO"
                                elif is_init_detected:
                                    title_status = "🥋 INÍCIO OFICIAL"
                                else:
                                    title_status = "📌 INÍCIO DO VÍDEO"

                                with st.expander(f"🥋 Sonkyō Inicial (Abertura) @ {curr_start_ts} – {curr_end_ts} • {title_status}", expanded=bool(initial_edit)):
                                    c_info1, c_info2 = st.columns([3, 1])
                                    with c_info1:
                                        st.markdown(f"**Intervalo Ritual:** `{curr_start_ts}` a `{curr_end_ts}` &nbsp;|&nbsp; **Início da Luta:** `{curr_end_ts}` (`Frame #{sonkyo_info.get('match_start_frame', 0)}`)")
                                        seek_init_s = max(0.0, parse_ts_to_seconds(curr_start_ts) - 1.0)
                                        if st.button("🎬 Assistir no Vídeo", key="btn_seek_sonkyo_init", help="Reproduzir o vídeo no momento do Sonkyō Inicial"):
                                            st.session_state["video_start_time"] = seek_init_s
                                            st.session_state["video_seek_label"] = f"Sonkyō Inicial ({curr_start_ts})"
                                            st.toast(f"🎥 Vídeo posicionado em {seek_init_s:.1f}s", icon="🎬")
                                            st.rerun()
                                    with c_info2:
                                        if initial_edit:
                                            st.markdown('<div class="valid-badge" style="background-color:#1E3A8A; color:#93C5FD; border: 1px solid #3B82F6; margin:0;">✏️ EDITADO</div>', unsafe_allow_html=True)
                                        elif is_init_detected:
                                            st.markdown('<div class="valid-badge" style="background-color:#1E1B4B; color:#C4B5FD; border: 1px solid #6366F1; margin:0;">🥋 DETECTADO</div>', unsafe_allow_html=True)
                                        else:
                                            st.markdown('<div class="valid-badge" style="background-color:#374151; color:#F3F4F6; border: 1px solid #9CA3AF; margin:0;">📌 PADRÃO</div>', unsafe_allow_html=True)

                                    if enable_editing:
                                        st.markdown("---")
                                        st.markdown(f"**✏️ Editar Intervalo ({dan_options.get(selected_dan, 'Dan')}):**")
                                        ed_col1, ed_col2, ed_btn1, ed_btn2 = st.columns([1.2, 1.2, 1.2, 0.8])
                                        new_init_start = ed_col1.text_input("Início", value=curr_start_ts, key="edit_init_start_input", label_visibility="collapsed")
                                        new_init_end = ed_col2.text_input("Fim", value=curr_end_ts, key="edit_init_end_input", label_visibility="collapsed")
                                        if ed_btn1.button("💾 Salvar", key="btn_apply_sonkyo_init_edit", width="stretch"):
                                            if "sonkyo_edits" not in st.session_state:
                                                st.session_state["sonkyo_edits"] = {}
                                            st.session_state["sonkyo_edits"]["initial"] = {
                                                "start_timestamp": new_init_start,
                                                "end_timestamp": new_init_end
                                            }
                                            st.toast("✏️ Tempo do Sonkyō Inicial salvo!", icon="✏️")
                                            st.rerun()
                                        if initial_edit and ed_btn2.button("🔄", key="btn_restore_sonkyo_init", help="Restaurar", width="stretch"):
                                            st.session_state["sonkyo_edits"].pop("initial", None)
                                            st.toast("Sonkyō Inicial restaurado.", icon="🔄")
                                            st.rerun()

                            # Determinar tempo de início do Sonkyo Final para cálculo de intervalos
                            final_edit_calc = sonkyo_edits.get("final")
                            fin_s_calc = sonkyo_info.get("final_sonkyo") or {}
                            fin_start_ts_calc = final_edit_calc.get("start_timestamp") if final_edit_calc else fin_s_calc.get("start_timestamp", f"{res['duration_seconds']}s")
                            fin_start_s_calc = parse_ts_to_seconds(fin_start_ts_calc)

                            # 2. GOLPES NA JANELA REGULAMENTAR DE COMBATE (ORDENADOS CRONOLOGICAMENTE)
                            if not has_strikes:
                                st.info("ℹ️ Nenhum golpe regulamentar registrado entre os momentos de Sonkyō.")
                                render_inline_strike_inserter("slot_init_to_fin", curr_end_s, fin_start_s_calc, "Sonkyō Inicial", "Sonkyō Final")
                            else:
                                # Botão de + entre Sonkyō Inicial e Golpe #1
                                first_strike_t = combined_strikes[0]["time_sec"]
                                first_strike_ts_lbl = combined_strikes[0]["timestamp"]
                                render_inline_strike_inserter("slot_0", curr_end_s, first_strike_t, "Sonkyō Inicial", f"Golpe #1 ({first_strike_ts_lbl})")

                                for idx, strike_item in enumerate(combined_strikes):
                                    strike_source = strike_item["source"]
                                    event_id_str = strike_item["event_id"]
                                    current_rev = strike_item["review"]
                                    attacker_label = strike_item["attacker_label"]
                                    attacker_id = strike_item["attacker_id"]
                                    orig_is_valid = strike_item["orig_is_valid"]
                                    eval_info = strike_item["raw_event"]["evaluation"] if strike_item["raw_event"] else {"total_score": 100.0 if orig_is_valid else 0.0, "min_required": 65.0, "sub_scores": {}}

                                    is_this_ippon = False
                                    if current_rev.get("is_included"):
                                        inc_is_ippon = current_rev.get("is_valid_ippon", current_rev.get("category") == "VALID_IPPON")
                                        is_this_ippon = inc_is_ippon
                                        if inc_is_ippon:
                                            status_badge = "➕ INCLUÍDO: IPPON"
                                            badge_html = '<div class="valid-badge" style="background-color:#1E3A8A; color:#93C5FD; border: 1px solid #3B82F6;">➕ INCLUÍDO: GOLPE VÁLIDO (IPPON)</div>'
                                        else:
                                            status_badge = "➕ INCLUÍDO: INVÁLIDO"
                                            badge_html = '<div class="valid-badge" style="background-color:#7F1D1D; color:#FECACA; border: 1px solid #EF4444;">➕ INCLUÍDO: GOLPE INVÁLIDO (NÃO FOI IPPON)</div>'
                                    elif current_rev.get("is_edited"):
                                        cat = current_rev.get("category", "")
                                        if cat == "VALID_IPPON":
                                            is_this_ippon = True
                                            status_badge = "✏️ EDITADO: IPPON"
                                            badge_html = '<div class="valid-badge" style="background-color:#1E3A8A; color:#93C5FD; border: 1px solid #3B82F6;">✏️ EDITADO: GOLPE VÁLIDO (IPPON)</div>'
                                        elif cat == "INVALID_HIT":
                                            status_badge = "✏️ EDITADO: INVÁLIDO"
                                            badge_html = '<div class="valid-badge" style="background-color:#7F1D1D; color:#FECACA; border: 1px solid #EF4444;">✏️ EDITADO: GOLPE INVÁLIDO (NÃO FOI IPPON)</div>'
                                        elif cat == "NO_STRIKE":
                                            status_badge = "🚫 NÃO HOUVE GOLPE"
                                            badge_html = '<div class="valid-badge" style="background-color:#374151; color:#F3F4F6; border: 1px solid #9CA3AF;">🚫 EDITADO: NÃO HOUVE GOLPE (DETECÇÃO ERRADA)</div>'
                                        else:
                                            status_badge = f"✏️ EDITADO ({current_rev.get('label', 'EDIT')})"
                                            badge_html = f'<div class="valid-badge" style="background-color:#1E3A8A; color:#93C5FD;">✏️ EDITADO ({current_rev.get("label", "EDIT")})</div>'
                                    elif current_rev.get("is_confirmed"):
                                        is_this_ippon = orig_is_valid
                                        status_badge = "✅ CONFIRMADO"
                                        badge_html = f'<div class="valid-badge" style="background-color:#14532D; color:#86EFAC; border: 1px solid #22C55E;">✅ CONFIRMADO ({dan_options.get(selected_dan, "Dan")})</div>'
                                    elif orig_is_valid:
                                        is_this_ippon = True
                                        status_badge = "✅ IPPON"
                                        badge_html = '<div class="valid-badge">✅ PONTO VÁLIDO (IPPON)</div>'
                                    else:
                                        is_this_ippon = False
                                        status_badge = "❌ INVÁLIDO"
                                        badge_html = '<div class="invalid-badge">❌ GOLPE INVÁLIDO</div>'

                                    # Destaque do golpe responsável pela marcação (Katakana: メ MEN, コ KOTE, ド DO, ツ TSUKI)
                                    display_strike_title = format_katakana_strike(current_rev['strike_type']) if is_this_ippon else current_rev['strike_type']

                                    with st.expander(f"🥊 Golpe #{idx+1}: {display_strike_title} @ {current_rev['timestamp']} ({attacker_label}) - {status_badge}", expanded=True):
                                        c_a, c_b = st.columns([1, 1.5])
                                        with c_a:
                                            seek_strike_s = max(0.0, parse_ts_to_seconds(current_rev['timestamp']) - 1.0)
                                            if st.button("🎬 Assistir no Vídeo", key=f"btn_seek_strike_{idx}_{event_id_str}", width="stretch", help=f"Reproduzir o vídeo no momento deste golpe ({seek_strike_s:.1f}s)"):
                                                st.session_state["video_start_time"] = seek_strike_s
                                                st.session_state["video_seek_label"] = f"Golpe #{idx+1} {display_strike_title} @ {current_rev['timestamp']}"
                                                st.toast(f"🎥 Vídeo posicionado em {seek_strike_s:.1f}s!", icon="🎬")
                                                st.rerun()

                                            if is_this_ippon:
                                                katakana_name = format_katakana_strike(current_rev['strike_type'])
                                                is_aka_attacker = (attacker_id == "KENSHI_AKA")
                                                if not is_inverted:
                                                    is_aka_actual = is_aka_attacker
                                                else:
                                                    is_aka_actual = not is_aka_attacker

                                                if is_aka_actual:
                                                    st.markdown(
                                                        f'<div style="background: rgba(239, 68, 68, 0.18); border: 1.5px solid #EF4444; border-radius: 6px; padding: 6px 12px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">'
                                                        f'<span style="color: #FCA5A5; font-weight: 800; font-size: 12px; letter-spacing: 0.5px;">🔴 MARCAÇÃO OFICIAL (AKA):</span>'
                                                        f'<span style="background: #991B1B; color: #FEE2E2; font-weight: 900; font-size: 13px; padding: 2px 10px; border-radius: 4px;">{katakana_name}</span>'
                                                        f'</div>',
                                                        unsafe_allow_html=True
                                                    )
                                                else:
                                                    st.markdown(
                                                        f'<div style="background: rgba(148, 163, 184, 0.16); border: 1.5px solid #CBD5E1; border-radius: 6px; padding: 6px 12px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">'
                                                        f'<span style="color: #F1F5F9; font-weight: 800; font-size: 12px; letter-spacing: 0.5px;">⚪ MARCAÇÃO OFICIAL (SHIRO):</span>'
                                                        f'<span style="background: #475569; color: #F8FAFC; font-weight: 900; font-size: 13px; padding: 2px 10px; border-radius: 4px;">{katakana_name}</span>'
                                                        f'</div>',
                                                        unsafe_allow_html=True
                                                    )
                                                st.markdown(f"**Técnica:** **`{katakana_name}`** 🥋 *(Golpe Pontuado no Placar)*")
                                            else:
                                                st.markdown(f"**Técnica:** `{current_rev['strike_type']}`")
                                            st.markdown(f"**Atacante:** `{attacker_label}`")
                                            st.markdown(f"**Timestamp:** `{current_rev['timestamp']}`" + (f" (Frame {strike_item['impact_frame']})" if strike_item['impact_frame'] > 0 else ""))
                                            if strike_source == "AI_DETECTED":
                                                st.markdown(f"**Pontuação original:** `{eval_info['total_score']}%` (Exigido: `{eval_info['min_required']}%`)")

                                            st.markdown(badge_html, unsafe_allow_html=True)
                                            if current_rev.get("notes"):
                                                st.markdown(f"**Observações:** _{current_rev['notes']}_")

                                            # Painel de Edição/Confirmação por Dan quando ativado
                                            if enable_editing:
                                                st.markdown("---")
                                                if strike_source == "INCLUDED":
                                                    st.markdown(f"**Ações para Golpe Incluído ({dan_options.get(selected_dan, 'Dan')}):**")
                                                    if st.button("🗑️ Remover esta inclusão", key=f"btn_del_inc_slot_{idx}_{event_id_str}", width="stretch"):
                                                        if event_id_str in st.session_state["session_reviews"]:
                                                             del st.session_state["session_reviews"][event_id_str]
                                                        st.toast(f"Golpe #{idx+1} incluído removido com sucesso!", icon="🗑️")
                                                        st.rerun()
                                                else:
                                                    st.markdown(f"**Ações de Revisão ({dan_options[selected_dan]}):**")
                                                    btn_col1, btn_col2 = st.columns(2)
                                                    
                                                    if btn_col1.button("✅ Confirmar", key=f"btn_cfm_{idx}_{event_id_str}"):
                                                        current_rev["is_confirmed"] = True
                                                        current_rev["is_edited"] = False
                                                        if orig_is_valid:
                                                            current_rev["label"] = "TP"
                                                            current_rev["category"] = "VALID_IPPON"
                                                            current_rev["is_valid_ippon"] = True
                                                        else:
                                                            current_rev["label"] = "FP"
                                                            current_rev["category"] = "INVALID_HIT"
                                                            current_rev["is_valid_ippon"] = False
                                                        st.session_state["session_reviews"][event_id_str] = current_rev
                                                        st.toast(f"Marcação #{idx+1} confirmada por {dan_options[selected_dan]}!", icon="✅")
                                                        st.rerun()

                                                    with btn_col2:
                                                        show_edit = st.checkbox("✏️ Editar", key=f"chk_edit_{idx}_{event_id_str}")

                                                    if show_edit:
                                                        curr_st = current_rev['strike_type']
                                                        st_opts = ["MEN", "KOTE", "DO", "TSUKI"]
                                                        st_idx = st_opts.index(curr_st) if curr_st in st_opts else 0
                                                        new_type = st.selectbox("Editar Técnica", st_opts, index=st_idx, format_func=DiagnosticReporter.format_strike_name, key=f"sel_type_{idx}_{event_id_str}")
                                                        new_ts = st.text_input("Editar Timestamp", value=current_rev['timestamp'], key=f"inp_ts_{idx}_{event_id_str}")
                                                        
                                                        # Estratégias de Revisão conforme diretrizes oficiais
                                                        if orig_is_valid:
                                                            strat_options = [
                                                                ("VALID_IPPON", "✅ Golpe Válido (Manter Ippon)"),
                                                                ("INVALID_HIT", "❌ Golpe Inválido (Houve golpe/acerto, mas não foi Ippon)"),
                                                                ("NO_STRIKE", "🚫 Não Houve Golpe (Detecção errada / Não houve golpe)")
                                                            ]
                                                        else:
                                                            strat_options = [
                                                                ("INVALID_HIT", "❌ Golpe Inválido (Manter Não Ippon)"),
                                                                ("VALID_IPPON", "✅ Golpe Válido (Foi Ippon)"),
                                                                ("NO_STRIKE", "🚫 Não Houve Golpe (Detecção errada / Não houve golpe)")
                                                            ]

                                                        strat_codes = [c[0] for c in strat_options]
                                                        strat_labels = [c[1] for c in strat_options]
                                                        curr_cat = current_rev.get("category", "VALID_IPPON" if (current_rev.get("label") == "TP" and orig_is_valid) else "INVALID_HIT")
                                                        default_strat_idx = strat_codes.index(curr_cat) if curr_cat in strat_codes else 0

                                                        selected_strat_lbl = st.radio(
                                                            "Classificação pelo Revisor:",
                                                            options=strat_labels,
                                                            index=default_strat_idx,
                                                            key=f"rad_strat_{idx}_{event_id_str}"
                                                        )
                                                        selected_strat_code = strat_codes[strat_labels.index(selected_strat_lbl)]
                                                        new_notes = st.text_input("Observações do Revisor", value=current_rev.get("notes", ""), key=f"inp_notes_{idx}_{event_id_str}")

                                                        if st.button("💾 Aplicar Edição neste Golpe", key=f"btn_apply_edit_{idx}_{event_id_str}"):
                                                            if selected_strat_code == "VALID_IPPON":
                                                                current_rev["label"] = "TP"
                                                                current_rev["category"] = "VALID_IPPON"
                                                                current_rev["is_valid_ippon"] = True
                                                            elif selected_strat_code == "INVALID_HIT":
                                                                current_rev["label"] = "FP"
                                                                current_rev["category"] = "INVALID_HIT"
                                                                current_rev["is_valid_ippon"] = False
                                                            elif selected_strat_code == "NO_STRIKE":
                                                                current_rev["label"] = "FP"
                                                                current_rev["category"] = "NO_STRIKE"
                                                                current_rev["is_valid_ippon"] = False

                                                            current_rev["strike_type"] = new_type
                                                            current_rev["timestamp"] = new_ts
                                                            current_rev["notes"] = new_notes
                                                            current_rev["is_edited"] = True
                                                            current_rev["is_confirmed"] = False
                                                            st.session_state["session_reviews"][event_id_str] = current_rev
                                                            st.toast(f"Marcação #{idx+1} atualizada com sucesso por {dan_options[selected_dan]}!", icon="✏️")
                                                            st.rerun()

                                                    if current_rev.get("is_confirmed") or current_rev.get("is_edited"):
                                                        if st.button("🔄 Resetar este golpe", key=f"btn_reset_single_{idx}_{event_id_str}"):
                                                            if event_id_str in st.session_state["session_reviews"]:
                                                                del st.session_state["session_reviews"][event_id_str]
                                                            st.toast(f"Golpe #{idx+1} restaurado ao estado original!", icon="🔄")
                                                            st.rerun()

                                            elif app_mode == "training" and strike_source == "AI_DETECTED":
                                                st.markdown("---")
                                                st.markdown("**🎓 Anotação (Reforço):**")
                                                btn_col1, btn_col2 = st.columns(2)
                                                sub_scores_raw = eval_info.get("sub_scores")
                                                sub_scores_val: Dict[str, Any] = sub_scores_raw if isinstance(sub_scores_raw, dict) else {}
                                                total_score_raw = eval_info.get("total_score", 0.0)
                                                total_score_val: float = float(total_score_raw) if isinstance(total_score_raw, (int, float, str)) else 0.0

                                                if btn_col1.button("👍 Correto", key=f"btn_tp_{idx}_{event_id_str}"):
                                                    feedback_mgr.save_feedback(
                                                        video_name=video_name_simple, profile_key=profile_choice, event_id=event_id_str, label="TP",
                                                        sub_scores=sub_scores_val, total_score=total_score_val,
                                                        strike_type=current_rev['strike_type'], timestamp=current_rev['timestamp'], reviewer_dan=selected_dan,
                                                        decision_category="VALID_IPPON" if orig_is_valid else "INVALID_HIT"
                                                    )
                                                    st.toast("✅ Anotado como Correto (TP)!", icon="👍")
                                                if btn_col2.button("👎 Falso Positivo", key=f"btn_fp_{idx}_{event_id_str}"):
                                                    feedback_mgr.save_feedback(
                                                        video_name=video_name_simple, profile_key=profile_choice, event_id=event_id_str, label="FP",
                                                        sub_scores=sub_scores_val, total_score=total_score_val,
                                                        strike_type=current_rev['strike_type'], timestamp=current_rev['timestamp'], reviewer_dan=selected_dan,
                                                        decision_category="INVALID_HIT" if orig_is_valid else "NO_STRIKE"
                                                    )
                                                    st.toast("❌ Anotado como Falso Positivo (FP)!", icon="👎")

                                        with c_b:
                                            if strike_item["diagnostic_report"]:
                                                st.markdown(strike_item["diagnostic_report"])
                                            else:
                                                st.info("ℹ️ Este golpe foi inserido manualmente pelo revisor na sequência temporal do combate.")

                                    # Inseridor inline de golpe (+) entre este golpe e o próximo (ou Sonkyō Final)
                                    if idx < len(combined_strikes) - 1:
                                        next_strike = combined_strikes[idx+1]
                                        render_inline_strike_inserter(
                                            f"slot_{idx}_{idx+1}",
                                            strike_item["time_sec"],
                                            next_strike["time_sec"],
                                            f"Golpe #{idx+1} ({strike_item['timestamp']})",
                                            f"Golpe #{idx+2} ({next_strike['timestamp']})"
                                        )
                                    else:
                                        render_inline_strike_inserter(
                                            "slot_last",
                                            strike_item["time_sec"],
                                            fin_start_s_calc,
                                            f"Golpe #{idx+1} ({strike_item['timestamp']})",
                                            f"Sonkyō Final ({fin_start_ts_calc})"
                                        )

                            # 3. EVENTO DE SONKYŌ FINAL (Encerramento do Combate)
                            final_edit = sonkyo_edits.get("final")
                            if has_final or final_edit:
                                fin_s = sonkyo_info.get("final_sonkyo") or {}
                                is_fin_detected = fin_s.get("is_detected", True)
                                curr_start_ts_fin = final_edit.get("start_timestamp") if final_edit else fin_s.get("start_timestamp", "00:04.000")
                                curr_end_ts_fin = final_edit.get("end_timestamp") if final_edit else fin_s.get("end_timestamp", f"{res['duration_seconds']}s")
                                
                                if final_edit:
                                    title_status_fin = "✏️ EDITADO"
                                elif is_fin_detected:
                                    title_status_fin = "🥋 ENCERRAMENTO OFICIAL"
                                else:
                                    title_status_fin = "📌 FIM DO VÍDEO"

                                with st.expander(f"🥋 Sonkyō Final (Encerramento) @ {curr_start_ts_fin} – {curr_end_ts_fin} • {title_status_fin}", expanded=bool(final_edit)):
                                    c_finfo1, c_finfo2 = st.columns([3, 1])
                                    with c_finfo1:
                                        st.markdown(f"**Intervalo Ritual:** `{curr_start_ts_fin}` a `{curr_end_ts_fin}` &nbsp;|&nbsp; **Término da Luta:** `{curr_start_ts_fin}` (`Frame #{sonkyo_info.get('match_end_frame', 0)}`)")
                                        seek_fin_s = max(0.0, parse_ts_to_seconds(curr_start_ts_fin) - 1.0)
                                        if st.button("🎬 Assistir no Vídeo", key="btn_seek_sonkyo_fin", help="Reproduzir o vídeo no momento do Sonkyō Final"):
                                            st.session_state["video_start_time"] = seek_fin_s
                                            st.session_state["video_seek_label"] = f"Sonkyō Final ({curr_start_ts_fin})"
                                            st.toast(f"🎥 Vídeo posicionado em {seek_fin_s:.1f}s", icon="🎬")
                                            st.rerun()
                                    with c_finfo2:
                                        if final_edit:
                                            st.markdown('<div class="valid-badge" style="background-color:#1E3A8A; color:#93C5FD; border: 1px solid #3B82F6; margin:0;">✏️ EDITADO</div>', unsafe_allow_html=True)
                                        elif is_fin_detected:
                                            st.markdown('<div class="valid-badge" style="background-color:#1E1B4B; color:#C4B5FD; border: 1px solid #6366F1; margin:0;">🥋 DETECTADO</div>', unsafe_allow_html=True)
                                        else:
                                            st.markdown('<div class="valid-badge" style="background-color:#374151; color:#F3F4F6; border: 1px solid #9CA3AF; margin:0;">📌 PADRÃO</div>', unsafe_allow_html=True)

                                    if enable_editing:
                                        st.markdown("---")
                                        st.markdown(f"**✏️ Editar Intervalo ({dan_options.get(selected_dan, 'Dan')}):**")
                                        ed_fcol1, ed_fcol2, ed_fbtn1, ed_fbtn2 = st.columns([1.2, 1.2, 1.2, 0.8])
                                        new_fin_start = ed_fcol1.text_input("Início", value=curr_start_ts_fin, key="edit_fin_start_input", label_visibility="collapsed")
                                        new_fin_end = ed_fcol2.text_input("Fim", value=curr_end_ts_fin, key="edit_fin_end_input", label_visibility="collapsed")
                                        if ed_fbtn1.button("💾 Salvar", key="btn_apply_sonkyo_fin_edit", width="stretch"):
                                            if "sonkyo_edits" not in st.session_state:
                                                st.session_state["sonkyo_edits"] = {}
                                            st.session_state["sonkyo_edits"]["final"] = {
                                                "start_timestamp": new_fin_start,
                                                "end_timestamp": new_fin_end
                                            }
                                            st.toast("✏️ Tempo do Sonkyō Final salvo!", icon="✏️")
                                            st.rerun()
                                        if final_edit and ed_fbtn2.button("🔄", key="btn_restore_sonkyo_fin", help="Restaurar", width="stretch"):
                                            st.session_state["sonkyo_edits"].pop("final", None)
                                            st.toast("Sonkyō Final restaurado.", icon="🔄")
                                            st.rerun()

                        # Seção de Inclusão de Novo Golpe Perdido (FN / Adicional)
                        if enable_editing or app_mode == "training":
                            st.markdown("---")
                            st.subheader("➕ Incluir Nova Marcação de Golpe (Golpe Perdido)")
                            
                            fn_col1, fn_col2 = st.columns(2)
                            with fn_col1:
                                fn_timestamp = st.text_input("Timestamp (ex: 00:02.500)", value="00:00.000", key="fn_ts_input", help="Momento exato do golpe no vídeo")
                                fn_strike_type = st.selectbox("Técnica Executada", ["MEN", "KOTE", "DO", "TSUKI"], key="fn_type_input")
                            
                            with fn_col2:
                                # 1. Lutador Aka ou Shiro
                                if is_inverted:
                                    att_options = [
                                        ("KENSHI_AKA", "🔴 Kenshi Aka (Vermelho - Esquerda)"),
                                        ("KENSHI_SHIRO", "⚪ Kenshi Shiro (Branco - Direita)")
                                    ]
                                else:
                                    att_options = [
                                        ("KENSHI_SHIRO", "⚪ Kenshi Shiro (Branco - Esquerda)"),
                                        ("KENSHI_AKA", "🔴 Kenshi Aka (Vermelho - Direita)")
                                    ]
                                att_labels = [opt[1] for opt in att_options]
                                fn_att_sel = st.selectbox("Lutador Atacante", att_labels, key="fn_attacker_input")
                                fn_att_id = att_options[att_labels.index(fn_att_sel)][0]
                                fn_att_name = "Kenshi Aka (Vermelho)" if fn_att_id == "KENSHI_AKA" else "Kenshi Shiro (Branco)"

                                # 2. Se foi Golpe Válido (Ippon) ou Golpe Inválido
                                fn_validity_options = [
                                    ("VALID_IPPON", "✅ Golpe Válido (Ippon)"),
                                    ("INVALID_HIT", "❌ Golpe Inválido (Não foi Ippon)")
                                ]
                                fn_val_labels = [v[1] for v in fn_validity_options]
                                fn_val_sel = st.radio("Validação do Golpe", fn_val_labels, horizontal=True, key="fn_validity_input")
                                fn_val_code = fn_validity_options[fn_val_labels.index(fn_val_sel)][0]
                                fn_is_ippon = (fn_val_code == "VALID_IPPON")

                            fn_notes = st.text_input("Observação do Revisor", value="Golpe não detectado pelo modelo", key="fn_notes_input")

                            if st.button("➕ Incluir Marcação no Dataset", width="stretch"):
                                new_fn_id = f"fn_{fn_timestamp.replace(':', '_').replace('.', '_')}_{fn_att_id.lower()}_{len(st.session_state.get('session_reviews', {}))+1}"
                                new_fn_item = {
                                    "event_id": new_fn_id,
                                    "label": "TP" if fn_is_ippon else "FP",
                                    "category": fn_val_code,
                                    "decision_category": fn_val_code,
                                    "is_valid_ippon": fn_is_ippon,
                                    "strike_type": fn_strike_type,
                                    "timestamp": fn_timestamp,
                                    "attacker_id": fn_att_id,
                                    "attacker_name": fn_att_name,
                                    "total_score": 100.0 if fn_is_ippon else 0.0,
                                    "sub_scores": {},
                                    "is_included": True,
                                    "is_confirmed": False,
                                    "is_edited": True,
                                    "notes": fn_notes
                                }
                                st.session_state["session_reviews"][new_fn_id] = new_fn_item
                                feedback_mgr.save_feedback(
                                    video_name=video_name_simple,
                                    profile_key=profile_choice,
                                    event_id=new_fn_id,
                                    label="TP" if fn_is_ippon else "FP",
                                    strike_type=fn_strike_type,
                                    timestamp=fn_timestamp,
                                    notes=fn_notes,
                                    reviewer_dan=selected_dan,
                                    is_included=True,
                                    decision_category=fn_val_code
                                )
                                st.toast(f"✅ Golpe Adicional ({fn_strike_type} de {fn_att_name} às {fn_timestamp}) incluído!", icon="➕")
                                st.rerun()

                        # Botão de Salvar Alterações e Retreinar Modelo ao Final
                        if enable_editing:
                            st.markdown("---")
                            st.subheader("💾 Finalizar Revisão & Retreinar Modelo")
                            st.caption(f"Salva todas as confirmações, edições e inclusões feitas sob a responsabilidade do revisor **{dan_options.get(selected_dan, 'Dan')}** e executa o retreinamento adaptativo.")

                            if st.button("💾 Salvar Alterações e Retreinar Modelo", type="primary", width="stretch"):
                                items_to_save = list(st.session_state["session_reviews"].values())
                                if not items_to_save:
                                    # Se nenhuma alteração explícita foi feita, incluir todos os detectados padrão como confirmados
                                    for idx, ev_data in enumerate(res["events"]):
                                        ev = ev_data["event_info"]
                                        eval_info = ev_data["evaluation"]
                                        is_val = eval_info.get("is_valid", False)
                                        items_to_save.append({
                                            "event_id": f"event_{idx+1}_frame_{ev['impact_frame']}",
                                            "label": "TP" if is_val else "FP",
                                            "decision_category": "VALID_IPPON" if is_val else "INVALID_HIT",
                                            "strike_type": ev['type'],
                                            "timestamp": ev['timestamp'],
                                            "total_score": eval_info.get('total_score', 0.0),
                                            "sub_scores": eval_info.get('sub_scores', {}),
                                            "is_confirmed": True
                                        })

                                new_cfg, session_rec = feedback_mgr.save_review_session(
                                    video_name=video_name_simple,
                                    profile_key=profile_choice,
                                    reviewer_dan=selected_dan,
                                    review_items=items_to_save,
                                    current_profile_config=current_p
                                )
                                # Atualizar o perfil ativo no calibrador
                                pipeline_temp = SenpAIPipeline(calibration_profile=profile_choice)
                                pipeline_temp.calibrator.update_and_save_profile(profile_choice, new_cfg)

                                st.success(f"🎉 Revisão salva e modelo retreinado com sucesso! ({len(items_to_save)} marcações processadas por {dan_options.get(selected_dan)}).")
                                if session_rec.get("optimization_summary", {}).get("changes"):
                                    st.markdown("**Alterações da Calibração:**")
                                    for chg in session_rec["optimization_summary"]["changes"]:
                                        st.markdown(f"- {chg}")

                        elif app_mode == "training":
                            st.markdown("---")
                            st.subheader(f"🧠 Painel de Otimização - Perfil '{profile_choice.upper()}'")
                            stats = feedback_mgr.get_stats(profile_key=profile_choice)
                            s1, s2, s3, s4 = st.columns(4)
                            s1.metric("Anotações", stats["total_feedback"])
                            s2.metric("Acertos (TP)", stats["true_positives"])
                            s3.metric("Falsos Pos. (FP)", stats["false_positives"])
                            s4.metric("Precisão", f"{stats['precision_pct']}%")

                            if st.button("🚀 Treinar e Atualizar Perfil", type="primary", width="stretch"):
                                updated_config, opt_summary = feedback_mgr.optimize_profile_config(profile_choice, current_p)
                                if opt_summary["status"] == "no_data":
                                    st.warning(opt_summary["message"])
                                else:
                                    pipeline_temp = SenpAIPipeline(calibration_profile=profile_choice)
                                    pipeline_temp.calibrator.update_and_save_profile(profile_choice, updated_config)
                                    st.success(f"🎉 O perfil '{profile_choice}' foi recalibrado com sucesso!")
                                    for chg in opt_summary["changes"]:
                                        st.markdown(f"- {chg}")

