"""
Shinpanai - Web Dashboard Interativo de Arbitragem e Análise de Kendo (Streamlit App)
Suporta 3 Modos Principais de Operação:
1. 📹 Modo de Arbitragem Gravada
2. 🎓 Modo de Treinamento & Aprendizado
3. 🔴 Modo de Detecção em Tempo Real (Webcam / Stream RTSP/RTCP)
"""

import streamlit as st
import tempfile
import os
import cv2
import json
import time

from src.pipeline import ShinpanaiPipeline
from src.utils.demo_generator import generate_demo_kendo_video
from src.engine.feedback_manager import FeedbackManager
from src.utils.hardware import detect_nvidia_gpu, get_effective_device, check_cuda_framework_support, validate_and_setup_gpu_requirements
from src.utils.settings_manager import load_settings, save_settings, get_processing_device, set_processing_device

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
    .mode-banner-recorded {
        background-color: #0F172A;
        border-left: 4px solid #3B82F6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .mode-banner-training {
        background-color: #1E1B4B;
        border-left: 4px solid #8B5CF6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .mode-banner-realtime {
        background-color: #311313;
        border-left: 4px solid #EF4444;
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

# --- SIDEBAR: NAVEGAÇÃO PRINCIPAL ---
st.sidebar.markdown("## 📌 Navegação")
nav_page = st.sidebar.radio(
    "Selecione a Página",
    options=["analysis", "settings"],
    format_func=lambda x: {
        "analysis": "⚔️ Arbitragem & Análise de Lutas",
        "settings": "⚙️ Menu de Configurações"
    }[x]
)

st.sidebar.markdown("---")

# ==============================================================================
# PÁGINA 1: MENU DE CONFIGURAÇÕES (PÁGINA DEDICADA SEPARADA)
# ==============================================================================
if nav_page == "settings":
    st.header("⚙️ Configurações Gerais do Sistema")
    st.markdown("Gerencie os parâmetros de aceleração de hardware, calibração de arbitragem e preferências globais.")

    saved_sys_settings = load_settings()
    default_device_pref = st.session_state.get("device_preference", saved_sys_settings.get("processing_device", "cpu"))

    # --- SEÇÃO 1: ACELERAÇÃO DE HARDWARE & DISPOSITIVO DE PROCESSAMENTO ---
    st.subheader("🖥️ 1. Processamento & Aceleração de Hardware")
    
    col_hw1, col_hw2 = st.columns([1, 1])
    
    with col_hw1:
        st.markdown("**Seletor do Modo de Processamento dos Modelos:**")
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

        if st.button("💾 Salvar Configurações de Hardware", type="primary", use_container_width=True):
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
                if st.button("🚀 Instalar Requisitos CUDA para GPU NVIDIA", use_container_width=True):
                    with st.spinner(f"Instalando pacotes PyTorch CUDA para {gpu_check_info['gpu_name']}..."):
                        install_res = validate_and_setup_gpu_requirements(auto_install=True)
                        if install_res["cuda_ready"]:
                            st.success("✅ Pacotes CUDA instalados com sucesso!")
                            st.rerun()
                        else:
                            st.error(install_res["message"])
        else:
            st.info("💻 **Computador rodando em Modo CPU.** Nenhuma GPU NVIDIA dedicada detectada.")

    st.markdown("---")

    # --- SEÇÃO 2: PERFIS DE ARBITRAGEM & RIGIDEZ ---
    st.subheader("🎛️ 2. Perfis de Calibração de Arbitragem")
    with open("config/calibration_profiles.json", "r", encoding="utf-8") as f:
        profiles_data = json.load(f)

    p_names = list(profiles_data.keys())
    selected_p_key = st.selectbox("Selecione o perfil para visualizar os parâmetros:", p_names)
    p_info = profiles_data[selected_p_key]

    c_p1, c_p2 = st.columns(2)
    c_p1.metric("Limiar Mínimo para Ponto Válido", f"{int(p_info.get('min_total_score', 0.65)*100)}%")
    c_p2.markdown(f"**Descrição:** {p_info.get('description', '')}")

    st.markdown("**Pesos dos Critérios de Ki-Ken-Tai-Ichi:**")
    w = p_info.get("weights", {})
    w_cols = st.columns(4)
    w_cols[0].metric("Alvo (Impacto)", f"{int(w.get('target_impact', 0.4)*100)}%")
    w_cols[1].metric("Fumikomi (Mão-Pé)", f"{int(w.get('fumikomi_sync', 0.25)*100)}%")
    w_cols[2].metric("Postura Corporal", f"{int(w.get('posture', 0.2)*100)}%")
    w_cols[3].metric("Zanshin", f"{int(w.get('zanshin', 0.15)*100)}%")

    st.markdown("---")
    st.info("💡 **Dica:** Para alterar os parâmetros de hardware selecionados, utilize o botão de salvamento acima. Para iniciar a análise de lutas, navegue no menu lateral até **'⚔️ Arbitragem & Análise de Lutas'**.")

# ==============================================================================
# PÁGINA 2: ARBITRAGEM & ANÁLISE DE LUTAS (PÁGINA PRINCIPAL)
# ==============================================================================
else:
    # --- SIDEBAR DA ANÁLISE: SELEÇÃO DOS 3 MODOS DE OPERAÇÃO ---
    st.sidebar.header("🕹️ Modo de Operação")
    app_mode = st.sidebar.radio(
        "Selecione o Modo de Operação",
        options=["realtime", "recorded", "training"],
        format_func=lambda x: {
            "realtime": "🔴 Modo de Detecção em Tempo Real",
            "recorded": "📹 Modo de Arbitragem Gravada",
            "training": "🎓 Modo de Treinamento & Aprendizado"
        }[x]
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Calibração de Sensibilidade")
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

    # BANNER DO MODO ATIVO
    if app_mode == "recorded":
        st.markdown('<div class="mode-banner-recorded">📹 <b>Modo de Arbitragem Gravada Ativo:</b> Análise de vídeos pré-gravados de combates de Kendo, detecção de Yuko-Datotsu e relatórios diagnósticos.</div>', unsafe_allow_html=True)
    elif app_mode == "training":
        st.markdown('<div class="mode-banner-training">🎓 <b>Modo de Treinamento & Aprendizado Ativo:</b> Anotação por reforço (TP, FP, FN), calibração por Dan/nível de graduação e otimização adaptativa dos perfis técnicos.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-banner-realtime">🔴 <b>Modo de Detecção em Tempo Real Ativo:</b> Processamento instantâneo de vídeo ao vivo via Webcam ou Câmeras IP (RTSP/RTCP) com sinalização em tempo real.</div>', unsafe_allow_html=True)

    # ==========================================================================
    # MODO 3: DETECÇÃO EM TEMPO REAL (WEBCAM / STREAM RTSP)
    # ==========================================================================
    if app_mode == "realtime":
        st.subheader("🔴 Transmissão Ao Vivo (Webcam / Stream RTSP/RTCP)")
        col_rt1, col_rt2 = st.columns([1, 1])

        with col_rt1:
            rt_source_choice = st.radio(
                "Selecione a Origem da Transmissão Ao Vivo:",
                ["🎥 Webcam Local (Câmera 0)", "📡 Stream RTSP / RTCP / Câmera IP"]
            )
            if "Webcam" in rt_source_choice:
                camera_idx = st.number_input("Dispositivo de Vídeo (Índice)", min_value=0, max_value=5, value=0)
                live_source_val = camera_idx
            else:
                rtsp_url_input = st.text_input("Endereço do Stream RTSP/RTCP", value="rtsp://192.168.1.100:554/live.sdp")
                live_source_val = rtsp_url_input

        with col_rt2:
            st.markdown("**Controle do Processamento em Tempo Real:**")
            run_live_detection = st.checkbox("▶️ Iniciar Transmissão Ao Vivo", value=False)
            st.markdown("*Dica: Desmarque a caixa acima a qualquer momento para interromper a transmissão ao vivo.*")

        if run_live_detection:
            dev_pref = st.session_state.get("device_preference", get_processing_device())
            pipeline = ShinpanaiPipeline(
                calibration_profile=profile_choice if profile_choice != "custom" else "normal",
                device_preference=dev_pref
            )

            col_live_v, col_live_m = st.columns([7, 5])
            with col_live_v:
                st.markdown("##### 🎥 Feed de Vídeo com Pose Tracking Ao Vivo")
                frame_placeholder = st.empty()

            with col_live_m:
                st.markdown("##### 📊 Métricas & Alertas em Tempo Real")
                fps_metric = st.empty()
                strike_alert_box = st.empty()
                st.markdown("**Histórico de Golpes Detectados nesta Sessão Ao Vivo:**")
                live_events_list = st.container(height=350)

            cap = cv2.VideoCapture(live_source_val)
            if not cap.isOpened():
                st.error(f"❌ Não foi possível conectar à fonte de vídeo ao vivo: `{live_source_val}`")
            else:
                frame_count = 0
                start_time = time.time()
                live_pose_history = []

                while run_live_detection and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Transmissão finalizada ou sinal de vídeo interrompido.")
                        break

                    landmarks, drawn_frame = pipeline.pose_detector.process_frame(frame)
                    live_pose_history.append(landmarks)

                    # Exibir frame anotado ao vivo no Streamlit
                    frame_rgb = cv2.cvtColor(drawn_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                    frame_count += 1
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0.0
                    fps_metric.metric("Desempenho Ao Vivo", f"{current_fps:.1f} FPS", f"Total Quadros: {frame_count}")

                    # Detecção instantânea de golpes a cada 10 frames
                    if len(live_pose_history) >= 15 and frame_count % 10 == 0:
                        detected_strikes = pipeline.event_spotter.detect_strikes(live_pose_history[-30:], fps=current_fps or 30.0)
                        if detected_strikes:
                            last_ev = detected_strikes[-1]
                            strike_alert_box.error(f"🚨 GOLPE DETECTADO AO VIVO: **{last_ev.type}** (Timestamp: {last_ev.timestamp})")
                            with live_events_list:
                                st.markdown(f"🥊 **{last_ev.type}** detectado no tempo `{last_ev.timestamp}` (Frame #{frame_count})")

                cap.release()

    # ==========================================================================
    # MODOS 1 E 2: ARBITRAGEM GRAVADA & TREINAMENTO & APRENDIZADO
    # ==========================================================================
    else:
        with st.expander("📹 Carregar Vídeo da Luta", expanded=("analysis_result" not in st.session_state)):
            col_in1, col_in2 = st.columns([1, 1])
            video_file_path = st.session_state.get("video_file_path", None)
            
            with col_in1:
                st.subheader("📹 Fazer Upload de Vídeo")
                st.markdown("Selecione o arquivo de vídeo da luta de Kendo a ser analisado:")
                uploaded_file = st.file_uploader("Vídeo da Luta (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
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

                    dev_pref = st.session_state.get("device_preference", get_processing_device())
                    pipeline = ShinpanaiPipeline(
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

        # PAINEL PRINCIPAL DE RESULTADOS (2 COLUNAS)
        if video_file_path or "analysis_result" in st.session_state:
            st.markdown("---")
            col_video, col_results = st.columns([5, 7])
            
            with col_video:
                st.markdown('<div class="sticky-video-marker"></div>', unsafe_allow_html=True)
                st.subheader("🎥 Vídeo da Luta")
                
                has_annotated = "annotated_output" in st.session_state and os.path.exists(st.session_state.get("annotated_output", ""))
                if has_annotated:
                    video_type = st.radio("Exibição do Vídeo:", ["🎥 Vídeo Anotado (Pose & Tracking)", "📹 Vídeo Original"], horizontal=True)
                    selected_video = st.session_state["annotated_output"] if "Anotado" in video_type else video_file_path
                else:
                    selected_video = video_file_path
                    
                if selected_video and os.path.exists(selected_video):
                    st.video(selected_video)
                else:
                    st.info("Nenhum vídeo disponível para reprodução.")
                    
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
                    
                    m5, m6 = st.columns(2)
                    dev_used = res.get('device_used', 'cpu').lower()
                    m5.metric("Processamento", "⚡ GPU NVIDIA" if dev_used == "gpu" else "💻 CPU Somente")
                    
                    st.caption(f"ℹ️ {res.get('device_status', '')}")
                    st.markdown('</div>', unsafe_allow_html=True)

            with col_results:
                st.subheader("🥊 Golpes Identificados & Diagnóstico")
                if "analysis_result" not in st.session_state:
                    st.info("👈 Clique em **⚡ Executar Arbitragem** para visualizar a análise detalhada dos golpes.")
                else:
                    res = st.session_state["analysis_result"]
                    video_name_simple = os.path.basename(res.get("video_path", "video.mp4"))
                    
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

                                        if app_mode == "training":
                                            st.markdown("---")
                                            st.markdown("**🎓 Anotação (Reforço):**")
                                            btn_col1, btn_col2 = st.columns(2)
                                            if btn_col1.button(f"👍 Correto", key=f"btn_tp_{idx}"):
                                                feedback_mgr.save_feedback(
                                                    video_name=video_name_simple, profile_key=profile_choice, event_id=event_id_str, label="TP",
                                                    sub_scores=eval_info.get("sub_scores", {}), total_score=eval_info.get("total_score", 0.0),
                                                    strike_type=ev['type'], timestamp=ev['timestamp']
                                                )
                                                st.toast(f"✅ Anotado como Correto (TP)!", icon="👍")
                                            if btn_col2.button(f"👎 Falso Positivo", key=f"btn_fp_{idx}"):
                                                feedback_mgr.save_feedback(
                                                    video_name=video_name_simple, profile_key=profile_choice, event_id=event_id_str, label="FP",
                                                    sub_scores=eval_info.get("sub_scores", {}), total_score=eval_info.get("total_score", 0.0),
                                                    strike_type=ev['type'], timestamp=ev['timestamp']
                                                )
                                                st.toast(f"❌ Anotado como Falso Positivo (FP)!", icon="👎")

                                    with c_b:
                                        st.markdown(ev_data["diagnostic_report"])

                        if app_mode == "training":
                            st.markdown("---")
                            st.subheader("➕ Registrar Golpe Perdido (Falso Negativo - FN)")
                            fn_col1, fn_col2 = st.columns(2)
                            fn_timestamp = fn_col1.text_input("Timestamp (ex: 00:02.500)", value="00:00.000", key="fn_ts_input")
                            fn_strike_type = fn_col2.selectbox("Técnica Executada", ["MEN", "KOTE", "DO", "TSUKI"], key="fn_type_input")
                            fn_notes = st.text_input("Observação", value="Golpe rápido não detectado", key="fn_notes_input")
                            
                            if st.button("➕ Adicionar Golpe Perdido ao Dataset", use_container_width=True):
                                feedback_mgr.save_feedback(
                                    video_name=video_name_simple, profile_key=profile_choice, event_id=f"fn_{fn_timestamp}",
                                    label="FN", strike_type=fn_strike_type, timestamp=fn_timestamp, notes=fn_notes
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
                                    for chg in opt_summary["changes"]:
                                        st.markdown(f"- {chg}")
