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

from src.pipeline import ShinpanaiPipeline, AnalysisWorker
from src.utils.demo_generator import generate_demo_kendo_video
from src.engine.feedback_manager import FeedbackManager
from src.utils.hardware import detect_nvidia_gpu, get_effective_device, check_cuda_framework_support, validate_and_setup_gpu_requirements
from src.utils.settings_manager import load_settings, save_settings, get_processing_device, set_processing_device
from src.utils.logger_manager import (
    setup_system_logger, get_log_summary, get_memory_logs,
    get_debug_log_file_content, clear_debug_logs, run_system_diagnostic_check, log_event
)

# Inicializa o logger central do sistema
setup_system_logger()

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

        if st.button("💾 Salvar Configurações de Hardware", type="primary", width="stretch"):
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
                if st.button("🚀 Instalar Requisitos CUDA para GPU NVIDIA", width="stretch"):
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

    # --- SEÇÃO 3: GOVERNANÇA DE TREINAMENTO & APRENDIZADO POR DAN ---
    st.subheader("🎓 3. Governança de Treinamento & Painel de Revisão por Dan")
    st.markdown("Acompanhe as métricas globais de retreinamento do modelo, distribuição por graduação Dan e gerenciamento de arquivos de revisão.")

    training_metrics = feedback_mgr.get_training_metrics()

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total de Treinamentos Realizados", training_metrics["total_trainings_count"])
    m_col2.metric("Nível Médio (Dan) dos Treinamentos", training_metrics["average_dan_label"])
    m_col3.metric("Total de Marcações de Revisão", training_metrics["total_review_items"])

    st.markdown("**Tabela de Quantidade de Treinamentos por Dan:**")
    dan_table_md = "| Dan | Nome da Graduação | Quantidade de Treinamentos | Percentual (%) |\n| :--- | :--- | :---: | :---: |\n"
    for d_row in training_metrics["dan_distribution"]:
        dan_table_md += f"| **{d_row['Dan']}** | {d_row['Nome Graduação']} | {d_row['Quantidade Treinamentos']} | {d_row['Percentual (%)']} |\n"
    st.markdown(dan_table_md)

    st.markdown("##### 🛠️ Gerenciamento do Dataset de Treinamento:")
    act_col1, act_col2, act_col3 = st.columns(3)

    with act_col1:
        st.markdown("**🗑️ Apagar Treinamento do Sistema**")
        st.caption("Reseta todo o histórico de revisões e restaura o modelo ao estágio inicial.")
        confirm_reset = st.checkbox("Confirmo que desejo apagar todo o treinamento", key="chk_confirm_reset")
        if st.button("🗑️ Apagar Treinamento", type="secondary", width="stretch"):
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
            file_name=f"shinpanai_training_package_{int(time.time())}.json",
            mime="application/json",
            width="stretch"
        )

    with act_col3:
        st.markdown("**📤 Carregar Treinamento Baixado**")
        st.caption("Importa arquivos de revisão previamente baixados para recalibrar o modelo.")
        imported_file = st.file_uploader("Selecione pacote (.json)", type=["json"], key="import_pkg_file")
        if imported_file is not None:
            if st.button("📤 Importar e Retreinar Modelo", type="primary", width="stretch"):
                try:
                    imported_file.seek(0)
                    pkg_content = json.loads(imported_file.read().decode("utf-8"))
                    import_res = feedback_mgr.import_training_package(pkg_content)
                    st.success(f"🎉 Pacote importado com sucesso! {import_res['new_items_added']} novos itens integrados. Novo Dan médio: {import_res['average_dan_now']}.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"❌ Erro ao importar pacote de treinamento: {ex}")

    st.markdown("---")

    # --- SEÇÃO 4: DIAGNÓSTICO, ALERTAS & LOG DE DEBUG DO SISTEMA ---
    st.subheader("🐛 4. Diagnóstico, Alertas & Log de Debug do Sistema")
    st.markdown("Rastreie alertas e erros do sistema em tempo real, execute testes de integridade e baixe o arquivo de log completo.")

    log_summary = get_log_summary()

    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Total de Eventos Registrados", log_summary["total_logs"])
    l_col2.metric("Erros do Sistema", log_summary["errors_count"], delta_color="inverse")
    l_col3.metric("Alertas & Avisos", log_summary["warnings_count"], delta_color="inverse")
    l_col4.metric("Informações de Execução", log_summary["info_count"])

    st.markdown("##### 🛠️ Ferramentas de Diagnóstico e Rastreamento:")
    dbg_col1, dbg_col2, dbg_col3 = st.columns(3)

    with dbg_col1:
        st.markdown("**📥 Baixar Log de Debug**")
        st.caption("Baixa o arquivo completo de logs do sistema (`shinpanai_debug.log`).")
        debug_log_text = get_debug_log_file_content()
        st.download_button(
            label="📥 Baixar Log de Debug (.log)",
            data=debug_log_text,
            file_name=f"shinpanai_debug_{int(time.time())}.log",
            mime="text/plain",
            width="stretch"
        )

    with dbg_col2:
        st.markdown("**🧪 Teste de Diagnóstico**")
        st.caption("Executa verificação completa de hardware, CUDA, bibliotecas e arquivos.")
        if st.button("🧪 Executar Diagnóstico", type="primary", width="stretch"):
            with st.spinner("Executando checagem de diagnóstico do sistema..."):
                diag_res = run_system_diagnostic_check()
                st.success("✅ Teste de diagnóstico concluído! Alertas gravados no log.")
                st.rerun()

    with dbg_col3:
        st.markdown("**🧹 Limpar Log de Debug**")
        st.caption("Reseta o arquivo de log no disco e limpa o buffer de memória.")
        if st.button("🧹 Limpar Logs", type="secondary", width="stretch"):
            clear_debug_logs()
            st.success("✅ Histórico de logs de debug zerado com sucesso!")
            st.rerun()

    st.markdown("##### 📜 Alertas e Registros de Debug em Tempo Real:")
    filter_col1, filter_col2 = st.columns([1, 2])
    with filter_col1:
        lvl_filter = st.selectbox("Filtrar por Nível:", ["TODOS", "ERROR", "WARNING", "INFO", "DEBUG"], index=0, key="log_lvl_filter_select")

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
    st.info("💡 **Dica:** Para alterar os parâmetros de hardware selecionados, utilize a Seção 1. Para iniciar a análise de lutas, navegue no menu lateral até **'⚔️ Arbitragem & Análise de Lutas'**.")


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
        options=["permissivo", "normal", "rigido", "custom"],
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
                    frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")

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
                st.markdown("Inicie o rastreamento de pose, detecção de impactos e avaliação de Yuko-Datotsu:")

                active_worker = st.session_state.get("analysis_worker", None)
                is_running = (active_worker is not None and not active_worker.is_done)

                # Alerta visual caso o processamento anterior tenha sido interrompido
                if st.session_state.get("processing_cancelled", False) and not is_running:
                    st.warning("⚠️ O processamento de arbitragem foi interrompido pelo usuário.")
                    st.session_state["processing_cancelled"] = False

                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    start_btn = st.button(
                        "⚡ Executar Arbitragem com Shinpanai" if not is_running else "⏳ Processando Arbitragem...",
                        type="primary",
                        width="stretch",
                        disabled=(video_file_path is None or is_running),
                        key="btn_start_recorded_arbitration"
                    )
                with col_btn2:
                    stop_btn = st.button(
                        "⏹️ Interromper Processamento",
                        type="secondary",
                        width="stretch",
                        disabled=not is_running,
                        key="btn_stop_recorded_arbitration"
                    )

                # 1. Se o usuário clicar no botão de Interromper
                if stop_btn and active_worker:
                    active_worker.cancel()
                    st.session_state.pop("analysis_worker", None)
                    st.session_state["processing_cancelled"] = True
                    annotated_out = "annotated_match.mp4"
                    if os.path.exists(annotated_out):
                        try:
                            os.remove(annotated_out)
                        except Exception:
                            pass
                    log_event("WARNING", "app", "Processamento de vídeo interrompido via botão pelo usuário.")
                    st.rerun()

                # 2. Se o usuário clicar em Iniciar Arbitragem
                if start_btn and video_file_path and not is_running:
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
                    worker = AnalysisWorker(
                        pipeline=pipeline,
                        video_path=video_file_path,
                        output_video_path=annotated_output
                    )
                    worker.start()
                    st.session_state["analysis_worker"] = worker
                    st.session_state["processing_cancelled"] = False
                    st.rerun()

                # 3. Se estiver em processamento ativo, renderiza barra de progresso e atualiza
                if is_running and active_worker:
                    st.progress(active_worker.progress)
                    st.info(f"⏳ **{active_worker.status_message}**")
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
                        st.session_state["analysis_result"] = active_worker.result
                        st.session_state["annotated_output"] = active_worker.output_video_path
                        st.session_state.pop("analysis_worker", None)
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
                    video_type = st.radio("Exibição do Vídeo:", ["📹 Vídeo Original", "🎥 Vídeo Anotado (Pose & Tracking)"], horizontal=True)
                    selected_video = video_file_path if "Original" in video_type else st.session_state["annotated_output"]
                else:
                    selected_video = video_file_path
                    
                if selected_video and os.path.exists(selected_video):
                    st.video(selected_video)
                else:
                    st.info("Nenhum vídeo disponível para reprodução.")
                    
                if "analysis_result" in st.session_state:
                    res = st.session_state["analysis_result"]
                    sonkyo_info = res.get("sonkyo_analysis", {})
                    plane_info = res.get("plane_filtering", {})

                    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
                    st.markdown("##### 📌 Resumo do Combate & Delimitação por Sonkyō")
                    
                    m1, m2 = st.columns(2)
                    m1.metric("Duração Total", f"{res['duration_seconds']}s")
                    eff_sec = res.get('effective_combat_duration_seconds', res['duration_seconds'])
                    m2.metric("Tempo Líquido (Luta)", f"{eff_sec}s", delta=f"{eff_sec - res['duration_seconds']:.1f}s" if eff_sec < res['duration_seconds'] else None)

                    m_s1, m_s2 = st.columns(2)
                    start_ts = sonkyo_info.get("match_start_timestamp", "00:00.000")
                    end_ts = sonkyo_info.get("match_end_timestamp", f"{res['duration_seconds']}s")
                    m_s1.metric("🥋 Início (Pós-Sonkyō)", start_ts)
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

                    if sonkyo_info.get("status_message"):
                        st.caption(f"🥋 **Status do Sonkyō:** {sonkyo_info['status_message']}")
                    st.caption(f"ℹ️ {res.get('device_status', '')}")
                    st.markdown('</div>', unsafe_allow_html=True)

            with col_results:
                st.subheader("🥊 Golpes Identificados & Diagnóstico")
                if "analysis_result" not in st.session_state:
                    st.info("👈 Clique em **⚡ Executar Arbitragem** para visualizar a análise detalhada dos golpes.")
                else:
                    res = st.session_state["analysis_result"]
                    video_name_simple = os.path.basename(res.get("video_path", "video.mp4"))

                    # Botão para Habilitar Edição dos Golpes Detectados (Modo Gravado / Treinamento)
                    enable_editing = st.toggle("✏️ Habilitar Edição dos Golpes Detectados", value=st.session_state.get("editing_enabled", False), key="toggle_enable_editing")
                    st.session_state["editing_enabled"] = enable_editing

                    selected_dan = 3
                    if enable_editing:
                        st.markdown("#### 🥋 Painel de Revisão Técnica por Árbitro Dan")
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

                        rev_header_col1, rev_header_col2 = st.columns([3, 1])
                        with rev_header_col1:
                            selected_dan = st.selectbox(
                                "Selecione a Graduação DAN do Revisor:",
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

                        st.info("🔒 **Regra de Auditoria e Governança:** A exclusão de marcações detectadas é **desabilitada**. É permitido confirmar, editar os parâmetros da detecção ou incluir novos golpes perdidos.")

                    # Lista de itens revisados para salvamento ao final
                    if "session_reviews" not in st.session_state:
                        st.session_state["session_reviews"] = {}

                    with st.container(height=650):
                        if not res["events"]:
                            st.warning("Nenhum evento claro de golpe foi identificado na janela regulamentar de combate.")
                        else:
                            for idx, ev_data in enumerate(res["events"]):
                                ev = ev_data["event_info"]
                                eval_info = ev_data["evaluation"]
                                event_id_str = f"event_{idx+1}_frame_{ev['impact_frame']}"
                                attacker_label = ev.get("attacker_name", "Kenshi Aka")

                                # Estado da revisão desta marcação
                                current_rev = st.session_state["session_reviews"].get(event_id_str, {
                                    "event_id": event_id_str,
                                    "label": "TP" if eval_info['is_valid'] else "FP",
                                    "strike_type": ev['type'],
                                    "timestamp": ev['timestamp'],
                                    "attacker_name": attacker_label,
                                    "total_score": eval_info.get('total_score', 0.0),
                                    "sub_scores": eval_info.get('sub_scores', {}),
                                    "is_edited": False,
                                    "is_confirmed": False,
                                    "notes": ""
                                })

                                if current_rev.get("is_edited"):
                                    status_badge = "✏️ EDITADO"
                                elif current_rev.get("is_confirmed"):
                                    status_badge = "✅ CONFIRMADO"
                                elif eval_info['is_valid']:
                                    status_badge = "✅ IPPON"
                                else:
                                    status_badge = "❌ INVÁLIDO"

                                with st.expander(f"Golpe #{idx+1}: {current_rev['strike_type']} @ {current_rev['timestamp']} ({attacker_label}) - {status_badge}", expanded=True):
                                    c_a, c_b = st.columns([1, 1.5])
                                    with c_a:
                                        st.markdown(f"**Técnica:** `{current_rev['strike_type']}`")
                                        st.markdown(f"**Atacante:** `{attacker_label}`")
                                        st.markdown(f"**Timestamp:** `{current_rev['timestamp']}` (Frame {ev['impact_frame']})")
                                        st.markdown(f"**Pontuação original:** `{eval_info['total_score']}%` (Exigido: `{eval_info['min_required']}%`)")

                                        if current_rev.get("is_edited"):
                                            st.markdown(f'<div class="valid-badge" style="background-color:#1E3A8A; color:#93C5FD;">✏️ EDITADO ({current_rev["label"]})</div>', unsafe_allow_html=True)
                                        elif current_rev.get("is_confirmed"):
                                            st.markdown('<div class="valid-badge" style="background-color:#14532D; color:#86EFAC;">✅ CONFIRMADO</div>', unsafe_allow_html=True)
                                        elif eval_info['is_valid']:
                                            st.markdown('<div class="valid-badge">✅ PONTO VÁLIDO</div>', unsafe_allow_html=True)
                                        else:
                                            st.markdown('<div class="invalid-badge">❌ GOLPE INVÁLIDO</div>', unsafe_allow_html=True)

                                        st.caption("🥋 Golpe registrado estritamente dentro da janela regulamentar de Sonkyō.")

                                        # Painel de Edição/Confirmação por Dan quando ativado
                                        if enable_editing:
                                            st.markdown("---")
                                            st.markdown(f"**Ações de Revisão ({dan_options[selected_dan]}):**")
                                            btn_col1, btn_col2 = st.columns(2)
                                            
                                            if btn_col1.button("✅ Confirmar", key=f"btn_cfm_{idx}"):
                                                current_rev["is_confirmed"] = True
                                                current_rev["is_edited"] = False
                                                current_rev["label"] = "TP" if eval_info['is_valid'] else "FP"
                                                st.session_state["session_reviews"][event_id_str] = current_rev
                                                st.toast(f"Marcação #{idx+1} confirmada por {dan_options[selected_dan]}!", icon="✅")
                                                st.rerun()

                                            with btn_col2:
                                                show_edit = st.checkbox("✏️ Editar", key=f"chk_edit_{idx}")

                                            if show_edit:
                                                new_type = st.selectbox("Editar Técnica", ["MEN", "KOTE", "DO", "TSUKI"], index=["MEN", "KOTE", "DO", "TSUKI"].index(current_rev['strike_type']), key=f"sel_type_{idx}")
                                                new_ts = st.text_input("Editar Timestamp", value=current_rev['timestamp'], key=f"inp_ts_{idx}")
                                                new_label_sel = st.radio("Validação pelo Revisor", ["Válido (TP)", "Falso Positivo (FP)"], index=0 if current_rev["label"] == "TP" else 1, key=f"rad_lbl_{idx}")
                                                new_notes = st.text_input("Observações", value=current_rev.get("notes", ""), key=f"inp_notes_{idx}")

                                                if st.button("💾 Aplicar Edição neste Golpe", key=f"btn_apply_edit_{idx}"):
                                                    current_rev["strike_type"] = new_type
                                                    current_rev["timestamp"] = new_ts
                                                    current_rev["label"] = "TP" if "TP" in new_label_sel else "FP"
                                                    current_rev["notes"] = new_notes
                                                    current_rev["is_edited"] = True
                                                    current_rev["is_confirmed"] = False
                                                    st.session_state["session_reviews"][event_id_str] = current_rev
                                                    st.toast(f"Marcação #{idx+1} atualizada como EDITADA por {dan_options[selected_dan]}!", icon="✏️")
                                                    st.rerun()

                                            if current_rev.get("is_confirmed") or current_rev.get("is_edited"):
                                                if st.button("🔄 Resetar este golpe", key=f"btn_reset_single_{idx}"):
                                                    if event_id_str in st.session_state["session_reviews"]:
                                                        del st.session_state["session_reviews"][event_id_str]
                                                    st.toast(f"Golpe #{idx+1} restaurado ao estado original!", icon="🔄")
                                                    st.rerun()

                                        elif app_mode == "training":
                                            st.markdown("---")
                                            st.markdown("**🎓 Anotação (Reforço):**")
                                            btn_col1, btn_col2 = st.columns(2)
                                            if btn_col1.button("👍 Correto", key=f"btn_tp_{idx}"):
                                                feedback_mgr.save_feedback(
                                                    video_name=video_name_simple, profile_key=profile_choice, event_id=event_id_str, label="TP",
                                                    sub_scores=eval_info.get("sub_scores", {}), total_score=eval_info.get("total_score", 0.0),
                                                    strike_type=ev['type'], timestamp=ev['timestamp'], reviewer_dan=selected_dan
                                                )
                                                st.toast("✅ Anotado como Correto (TP)!", icon="👍")
                                            if btn_col2.button("👎 Falso Positivo", key=f"btn_fp_{idx}"):
                                                feedback_mgr.save_feedback(
                                                    video_name=video_name_simple, profile_key=profile_choice, event_id=event_id_str, label="FP",
                                                    sub_scores=eval_info.get("sub_scores", {}), total_score=eval_info.get("total_score", 0.0),
                                                    strike_type=ev['type'], timestamp=ev['timestamp'], reviewer_dan=selected_dan
                                                )
                                                st.toast("❌ Anotado como Falso Positivo (FP)!", icon="👎")

                                    with c_b:
                                        st.markdown(ev_data["diagnostic_report"])

                        # Seção de Inclusão de Novo Golpe Perdido (FN / Adicional)
                        if enable_editing or app_mode == "training":
                            st.markdown("---")
                            st.subheader("➕ Incluir Nova Marcação de Golpe (Golpe Perdido)")
                            fn_col1, fn_col2 = st.columns(2)
                            fn_timestamp = fn_col1.text_input("Timestamp (ex: 00:02.500)", value="00:00.000", key="fn_ts_input")
                            fn_strike_type = fn_col2.selectbox("Técnica Executada", ["MEN", "KOTE", "DO", "TSUKI"], key="fn_type_input")
                            fn_notes = st.text_input("Observação do Revisor", value="Golpe não detectado pelo modelo", key="fn_notes_input")

                            if st.button("➕ Incluir Marcação no Dataset", width="stretch"):
                                new_fn_id = f"fn_{fn_timestamp.replace(':', '_').replace('.', '_')}"
                                new_fn_item = {
                                    "event_id": new_fn_id,
                                    "label": "INCLUDED",
                                    "strike_type": fn_strike_type,
                                    "timestamp": fn_timestamp,
                                    "total_score": 0.0,
                                    "sub_scores": {},
                                    "is_included": True,
                                    "notes": fn_notes
                                }
                                st.session_state["session_reviews"][new_fn_id] = new_fn_item
                                feedback_mgr.save_feedback(
                                    video_name=video_name_simple, profile_key=profile_choice, event_id=new_fn_id,
                                    label="INCLUDED", strike_type=fn_strike_type, timestamp=fn_timestamp, notes=fn_notes,
                                    reviewer_dan=selected_dan, is_included=True
                                )
                                st.success(f"✅ Golpe Adicional ({fn_strike_type} às {fn_timestamp}) incluído!")

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
                                        items_to_save.append({
                                            "event_id": f"event_{idx+1}_frame_{ev['impact_frame']}",
                                            "label": "TP" if eval_info['is_valid'] else "FP",
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
                                pipeline_temp = ShinpanaiPipeline(calibration_profile=profile_choice)
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
                                    pipeline_temp = ShinpanaiPipeline(calibration_profile=profile_choice)
                                    pipeline_temp.calibrator.update_and_save_profile(profile_choice, updated_config)
                                    st.success(f"🎉 O perfil '{profile_choice}' foi recalibrado com sucesso!")
                                    for chg in opt_summary["changes"]:
                                        st.markdown(f"- {chg}")

