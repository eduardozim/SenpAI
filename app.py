"""
Shinpanai - Web Dashboard Interativo de Arbitragem e Análise de Kendo (Streamlit App)
Suporta Modo Usuário (análise padrão) e Modo Aprendizagem (anotação por reforço e otimização de perfil).
"""

import streamlit as st
import tempfile
import os
import cv2
import json

from src.pipeline import ShinpanaiPipeline
from src.utils.demo_generator import generate_demo_kendo_video
from src.engine.feedback_manager import FeedbackManager

st.set_page_config(
    page_title="Shinpanai - AI Kendo Referee & Analysis System",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

feedback_mgr = FeedbackManager()

# Estilização CSS Moderna para a Interface
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #E2E8F0;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #334155;
    }
    .valid-badge {
        background-color: #166534;
        color: #4ADE80;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .invalid-badge {
        background-color: #991B1B;
        color: #FCA5A5;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: bold;
        display: inline-block;
    }
    .mode-banner-user {
        background-color: #0F172A;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .mode-banner-learning {
        background-color: #1E1B4B;
        border-left: 4px solid #8B5CF6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
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
        padding: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚔️ Shinpanai (審判 AI)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistema de Visão Computacional para Análise de Lutas de Kendo, Detecção de Golpes e Avaliação de Yuko-Datotsu</div>', unsafe_allow_html=True)

# --- SIDEBAR: SELEÇÃO DE MODO E CALIBRAÇÃO ---
st.sidebar.header("🕹️ Modo de Operação")
app_mode = st.sidebar.radio(
    "Selecione o Modo",
    options=["user", "learning"],
    format_func=lambda x: {
        "user": "👤 Modo Usuário (Análise Padrão)",
        "learning": "🎓 Modo Aprendizagem (Treino por Reforço)"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Calibração de Sensibilidade")
st.sidebar.markdown("Ajuste a rigidez da arbitragem para o nível dos praticantes:")

profile_choice = st.sidebar.selectbox(
    "Perfil de Arbitragem Predefinido",
    options=["normal", "rigido", "permissivo", "custom"],
    format_func=lambda x: {
        "normal": "Treino Geral / Keiko (Normal)",
        "rigido": "Campeonato / Dan Audit (Rígido)",
        "permissivo": "Iniciantes / Educacional (Permissivo)",
        "custom": "⚙️ Personalizado (Sliders Manual)"
    }[x]
)

# Carregar config para sliders se customizado
with open("config/calibration_profiles.json", "r", encoding="utf-8") as f:
    profiles_data = json.load(f)

current_p = profiles_data.get(profile_choice, profiles_data["normal"])

if profile_choice == "custom":
    st.sidebar.subheader("Ajuste Fino de Limiares")
    min_score_pct = st.sidebar.slider("Pontuação Mínima Global para Ponto Válido (%)", 30, 95, 65)
    
    st.sidebar.markdown("**Pesos dos Critérios de Ki-Ken-Tai-Ichi:**")
    w_target = st.sidebar.slider("Peso do Impacto no Alvo", 0.0, 1.0, 0.40)
    w_fumikomi = st.sidebar.slider("Peso do Fumikomi (Sincronia Mão-Pé)", 0.0, 1.0, 0.25)
    w_posture = st.sidebar.slider("Peso da Postura Corporal", 0.0, 1.0, 0.20)
    w_zanshin = st.sidebar.slider("Peso do Zanshin", 0.0, 1.0, 0.15)
else:
    st.sidebar.info(f"**Descrição do Perfil:**\n{current_p.get('description', '')}")
    min_score_pct = int(current_p.get("min_total_score", 0.65) * 100)
    weights = current_p.get("weights", {})
    w_target = weights.get("target_impact", 0.40)
    w_fumikomi = weights.get("fumikomi_sync", 0.25)
    w_posture = weights.get("posture", 0.20)
    w_zanshin = weights.get("zanshin", 0.15)

# --- BANNER DO MODO ATIVO ---
if app_mode == "user":
    st.markdown('<div class="mode-banner-user">👤 <b>Modo Usuário Ativo:</b> Exibição direta dos resultados da arbitragem sem poluição visual. Os parâmetros do perfil ativo serão aplicados à luta.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="mode-banner-learning">🎓 <b>Modo Aprendizagem por Reforço Ativo:</b> Você pode rotular detecções corretas (TP) ou falsos positivos (FP) e registrar golpes perdidos (FN) para otimizar dinamicamente o perfil selecionado.</div>', unsafe_allow_html=True)

# --- ÁREA PRINCIPAL: CARREGAMENTO DE VÍDEO E CONFIGURAÇÃO ---
with st.expander("📹 Carregar / Selecionar Vídeo da Luta", expanded=("analysis_result" not in st.session_state)):
    col_in1, col_in2 = st.columns([1, 1])
    
    video_file_path = st.session_state.get("video_file_path", None)
    
    with col_in1:
        st.subheader("Fonte do Vídeo")
        input_source = st.radio("Selecione a Origem", ["Vídeo Sintético de Teste (Demo)", "Fazer Upload de Vídeo (.mp4, .avi)"])
        
        if input_source == "Vídeo Sintético de Teste (Demo)":
            if st.button("🎬 Gerar Novo Vídeo Demo"):
                video_file_path = generate_demo_kendo_video("demo_kendo_match.mp4")
                st.session_state["video_file_path"] = video_file_path
                st.success("Vídeo sintético gerado com sucesso!")
            else:
                if not os.path.exists("demo_kendo_match.mp4"):
                    video_file_path = generate_demo_kendo_video("demo_kendo_match.mp4")
                else:
                    video_file_path = "demo_kendo_match.mp4"
                st.session_state["video_file_path"] = video_file_path
        else:
            uploaded_file = st.file_uploader("Selecione o vídeo da luta de Kendo", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_file_path = tfile.name
                st.session_state["video_file_path"] = video_file_path

    with col_in2:
        st.subheader("Executar Arbitragem")
        st.markdown("Clique abaixo para iniciar o rastreamento de pose, detecção de impactos e avaliação de Yuko-Datotsu.")
        if video_file_path and st.button("⚡ Executar Arbitragem com Shinpanai", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Inicializando pipeline de visão e pose tracking...")

            def update_p(p):
                progress_bar.progress(min(1.0, max(0.0, p)))
                status_text.text(f"Processando frames... {int(p*100)}%")

            pipeline = ShinpanaiPipeline(calibration_profile=profile_choice if profile_choice != "custom" else "normal")
            
            if profile_choice == "custom":
                pipeline.calibrator.update_custom_settings(
                    min_total_score=min_score_pct / 100.0,
                    weight_target=w_target,
                    weight_fumikomi=w_fumikomi,
                    weight_posture=w_posture,
                    weight_zanshin=w_zanshin
                )

            annotated_output = "annotated_match.mp4"
            analysis_result = pipeline.process_video(
                video_path=video_file_path,
                output_video_path=annotated_output,
                progress_callback=update_p
            )

            st.session_state["analysis_result"] = analysis_result
            st.session_state["annotated_output"] = annotated_output
            status_text.text("Análise concluída!")
            progress_bar.progress(1.0)
            st.rerun()

video_file_path = st.session_state.get("video_file_path", None)

# --- PAINEL PRINCIPAL: DIVISÃO EM COLUNAS (VÍDEO FIXO + LISTA COM SCROLL) ---
if video_file_path or "analysis_result" in st.session_state:
    st.markdown("---")
    
    col_video, col_results = st.columns([5, 7])
    
    # 🎥 COLUNA ESQUERDA (FIXA / STICKY): VÍDEO + RESUMO DE MÉTRICAS
    with col_video:
        st.markdown('<div class="sticky-video-marker"></div>', unsafe_allow_html=True)
        st.subheader("🎥 Vídeo da Luta")
        
        # Alternador de exibição de vídeo (Anotado AI vs Original)
        has_annotated = "annotated_output" in st.session_state and os.path.exists(st.session_state.get("annotated_output", ""))
        
        if has_annotated:
            video_type = st.radio(
                "Exibição do Vídeo:",
                ["🎥 Vídeo Anotado (Pose & Tracking)", "📹 Vídeo Original"],
                horizontal=True
            )
            selected_video = st.session_state["annotated_output"] if "Anotado" in video_type else video_file_path
        else:
            selected_video = video_file_path
            
        if selected_video and os.path.exists(selected_video):
            st.video(selected_video)
        else:
            st.info("Nenhum vídeo disponível para reprodução.")
            
        # Métricas resumidas do combate na coluna do vídeo
        if "analysis_result" in st.session_state:
            res = st.session_state["analysis_result"]
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.markdown("##### 📌 Resumo do Combate")
            m1, m2 = st.columns(2)
            m1.metric("Duração", f"{res['duration_seconds']}s")
            m2.metric("Total Frames", res['total_frames'])
            
            m3, m4 = st.columns(2)
            m3.metric("Golpes Detectados", res['events_detected_count'])
            m4.metric("Perfil Aplicado", res['profile_applied'])
            st.markdown('</div>', unsafe_allow_html=True)

    # 📊 COLUNA DIREITA (SCROLLABLE): LISTA DOS GOLPES IDENTIFICADOS & DIAGNÓSTICO
    with col_results:
        st.subheader("🥊 Golpes Identificados & Diagnóstico")
        
        if "analysis_result" not in st.session_state:
            st.info("👈 Clique em **⚡ Executar Arbitragem** para visualizar a análise detalhada dos golpes.")
        else:
            res = st.session_state["analysis_result"]
            video_name_simple = os.path.basename(res.get("video_path", "video.mp4"))
            
            # Container com barra de rolagem dedicada para os golpes
            with st.container(height=680):
                if not res["events"]:
                    st.warning("Nenhum evento claro de golpe foi identificado neste trecho de vídeo.")
                else:
                    for idx, ev_data in enumerate(res["events"]):
                        ev = ev_data["event_info"]
                        eval_info = ev_data["evaluation"]
                        event_id_str = f"event_{idx+1}_frame_{ev['impact_frame']}"
                        
                        with st.expander(f"Golpe #{idx+1}: {ev['type']} @ {ev['timestamp']} - {'✅ IPPON' if eval_info['is_valid'] else '❌ INVÁLIDO'}", expanded=True):
                            c_a, c_b = st.columns([1, 1.5])
                            
                            with c_a:
                                st.markdown(f"**Técnica:** `{ev['type']}`")
                                st.markdown(f"**Timestamp:** `{ev['timestamp']}` (Frame {ev['impact_frame']})")
                                st.markdown(f"**Pontuação:** `{eval_info['total_score']}%` (Exigido: `{eval_info['min_required']}%`)")
                                
                                if eval_info['is_valid']:
                                    st.markdown('<div class="valid-badge">✅ PONTO VÁLIDO</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<div class="invalid-badge">❌ GOLPE INVÁLIDO</div>', unsafe_allow_html=True)

                                # INTERFACE DE FEEDBACK (MODO APRENDIZAGEM)
                                if app_mode == "learning":
                                    st.markdown("---")
                                    st.markdown("**🎓 Anotação (Reforço):**")
                                    
                                    btn_col1, btn_col2 = st.columns(2)
                                    
                                    if btn_col1.button(f"👍 Correto", key=f"btn_tp_{idx}"):
                                        feedback_mgr.save_feedback(
                                            video_name=video_name_simple,
                                            profile_key=profile_choice,
                                            event_id=event_id_str,
                                            label="TP",
                                            sub_scores=eval_info.get("sub_scores", {}),
                                            total_score=eval_info.get("total_score", 0.0),
                                            strike_type=ev['type'],
                                            timestamp=ev['timestamp']
                                        )
                                        st.toast(f"✅ Anotado como Correto (TP)!", icon="👍")

                                    if btn_col2.button(f"👎 Falso Positivo", key=f"btn_fp_{idx}"):
                                        feedback_mgr.save_feedback(
                                            video_name=video_name_simple,
                                            profile_key=profile_choice,
                                            event_id=event_id_str,
                                            label="FP",
                                            sub_scores=eval_info.get("sub_scores", {}),
                                            total_score=eval_info.get("total_score", 0.0),
                                            strike_type=ev['type'],
                                            timestamp=ev['timestamp']
                                        )
                                        st.toast(f"❌ Anotado como Falso Positivo (FP)!", icon="👎")

                            with c_b:
                                st.markdown(ev_data["diagnostic_report"])

                # PAINEL DE APRENDIZAGEM E REGISTRO DE FALSOS NEGATIVOS (MODO APRENDIZAGEM)
                if app_mode == "learning":
                    st.markdown("---")
                    st.subheader("➕ Registrar Golpe Perdido (Falso Negativo - FN)")
                    st.markdown("Se a IA deixou de identificar um golpe ocorrido, registre abaixo:")
                    
                    fn_col1, fn_col2 = st.columns(2)
                    fn_timestamp = fn_col1.text_input("Timestamp (ex: 00:02.500)", value="00:00.000", key="fn_ts_input")
                    fn_strike_type = fn_col2.selectbox("Técnica Executada", ["MEN", "KOTE", "DO", "TSUKI"], key="fn_type_input")
                    fn_notes = st.text_input("Observação", value="Golpe rápido não detectado", key="fn_notes_input")
                    
                    if st.button("➕ Adicionar Golpe Perdido ao Dataset", use_container_width=True):
                        feedback_mgr.save_feedback(
                            video_name=video_name_simple,
                            profile_key=profile_choice,
                            event_id=f"fn_{fn_timestamp}",
                            label="FN",
                            strike_type=fn_strike_type,
                            timestamp=fn_timestamp,
                            notes=fn_notes
                        )
                        st.success(f"Golpe Perdido ({fn_strike_type} às {fn_timestamp}) registrado!")

                    st.markdown("---")
                    st.subheader(f"🧠 Painel de Otimização - Perfil '{profile_choice.upper()}'")
                    
                    stats = feedback_mgr.get_stats(profile_key=profile_choice)
                    
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Anotações", stats["total_feedback"])
                    s2.metric("Acertos (TP)", stats["true_positives"])
                    s3.metric("Falsos Pos. (FP)", stats["false_positives"])
                    s4.metric("Precisão", f"{stats['precision_pct']}%")

                    if st.button("🚀 Treinar e Atualizar Perfil", type="primary", use_container_width=True):
                        updated_config, opt_summary = feedback_mgr.optimize_profile_config(profile_choice, current_p)
                        
                        if opt_summary["status"] == "no_data":
                            st.warning(opt_summary["message"])
                        else:
                            pipeline_temp = ShinpanaiPipeline(calibration_profile=profile_choice)
                            pipeline_temp.calibrator.update_and_save_profile(profile_choice, updated_config)
                            
                            st.success(f"🎉 O perfil '{profile_choice}' foi recalibrado com sucesso!")
                            st.markdown("**Resumo das Otimizações Aplicadas:**")
                            for chg in opt_summary["changes"]:
                                st.markdown(f"- {chg}")
