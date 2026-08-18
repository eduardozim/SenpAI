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
from src.analytics.sonkyo_detector import SonkyoDetector
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

def parse_ts_to_seconds(ts_str: str) -> float:
    """Converte timestamps (ex: '00:02.500', '02.500', '2.5s') em segundos (float)."""
    if not ts_str:
        return 0.0
    try:
        ts = str(ts_str).strip().lower().replace("s", "")
        if ":" in ts:
            parts = ts.split(":")
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(ts)
    except Exception:
        return 0.0

# Estilização CSS Moderna para a Interface com redução global de 20%
st.markdown("""
<style>
    /* 1. Redução Global da Interface em 20% (Zoom 80%) */
    .stApp {
        zoom: 0.8;
        -moz-transform: scale(0.8);
        -moz-transform-origin: 0 0;
    }

    /* 2. Otimização do espaçamento superior/inferior para melhor aproveitamento de tela */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 96% !important;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #E2E8F0;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
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

    # --- SEÇÃO 4: APRENDIZADO CONTÍNUO DO SONKYŌ & CALIBRAÇÃO DE LIMITES ---
    st.subheader("🥋 4. Aprendizado Contínuo de Sonkyō & Calibração de Limites")
    st.markdown("Acompanhe os parâmetros biométricos adaptados e calibrados a partir das edições de Sonkyō realizadas pelos árbitros.")

    sonkyo_detector_stats = SonkyoDetector()
    s_stats = sonkyo_detector_stats.get_learned_stats()

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("Rituais de Sonkyō Aprendidos", f"{s_stats['samples_count']} amostras")
    s_col2.metric("Compressão de Altura Adaptada", f"{s_stats['learned_rel_height_threshold']:.2f}")
    s_col3.metric("Rebaixamento de Quadril (ΔY)", f"{s_stats['learned_hip_drop_threshold']:.2f}")
    s_col4.metric("Ângulo Máx. Joelho Calibrado", f"{s_stats['learned_knee_angle_threshold']:.1f}°")

    st.caption(f"ℹ️ **Última Atualização do Modelo de Sonkyō:** `{s_stats['last_updated_at']}` | Exemplares em memória: `{s_stats['exemplars_count']}`")

    if s_stats['samples_count'] > 0:
        if st.button("🔄 Resetar Perfil Aprendido de Sonkyō para os Padrões de Fábrica", key="btn_reset_learned_sonkyo"):
            sonkyo_detector_stats.reset_learned_profile()
            st.success("✅ Perfil de aprendizado de Sonkyō resetado para os padrões de fábrica!")
            st.rerun()

    st.markdown("---")

    # --- SEÇÃO 5: DIAGNÓSTICO, ALERTAS & LOG DE DEBUG DO SISTEMA ---
    st.subheader("🐛 5. Diagnóstico, Alertas & Log de Debug do Sistema")
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
    st.sidebar.header("⚡ Aceleração de Hardware")
    saved_hw_device = get_processing_device()
    dev_pref_current = st.session_state.get("device_preference", saved_hw_device)
    effective_dev, dev_msg, dev_gpu = get_effective_device(dev_pref_current)

    if effective_dev == "gpu":
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 8px; padding: 10px 12px; margin-bottom: 6px;">
                <div style="font-weight: 700; color: #4ade80; font-size: 0.92rem; margin-bottom: 4px;">🚀 Aceleração Ativada</div>
                <div style="font-size: 0.82rem; color: #e2e8f0; font-weight: 600;">{dev_gpu.get('gpu_name', 'NVIDIA GPU')}</div>
                <div style="font-size: 0.74rem; color: #94a3b8; margin-top: 2px;">⚡ YOLOv8-Pose (PyTorch CUDA)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.12); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 8px; padding: 10px 12px; margin-bottom: 6px;">
                <div style="font-weight: 700; color: #cbd5e1; font-size: 0.92rem; margin-bottom: 4px;">💻 Aceleração Desativada</div>
                <div style="font-size: 0.82rem; color: #94a3b8;">Processamento por CPU</div>
                <div style="font-size: 0.74rem; color: #64748b; margin-top: 2px;">MediaPipe Pose (TFLite CPU)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.sidebar.caption("⚙️ *Para alterar o acelerador, acesse Configurações Globais no Modo de Treinamento.*")

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
                    cached_file_name = st.session_state.get("uploaded_file_name")
                    cached_file_size = st.session_state.get("uploaded_file_size")
                    cached_file_path = st.session_state.get("video_file_path")

                    # Reutilizar o arquivo salvo caso seja exatamente o mesmo upload
                    if cached_file_path and os.path.exists(cached_file_path) and cached_file_name == uploaded_file.name and cached_file_size == uploaded_file.size:
                        video_file_path = cached_file_path
                    else:
                        # Limpar arquivo temporário anterior se existir
                        if cached_file_path and os.path.exists(cached_file_path) and ("shinpanai_uploads" in cached_file_path or "tmp" in cached_file_path):
                            try:
                                os.remove(cached_file_path)
                            except Exception:
                                pass

                        uploads_dir = os.path.join(tempfile.gettempdir(), "shinpanai_uploads")
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
                            f_out.write(uploaded_file.read())

                        video_file_path = target_file_path
                        st.session_state["video_file_path"] = video_file_path
                        st.session_state["uploaded_file_name"] = uploaded_file.name
                        st.session_state["uploaded_file_size"] = uploaded_file.size
                else:
                    if "uploaded_file_name" in st.session_state:
                        cached_file_path = st.session_state.get("video_file_path")
                        if cached_file_path and os.path.exists(cached_file_path) and ("shinpanai_uploads" in cached_file_path or "tmp" in cached_file_path):
                            try:
                                os.remove(cached_file_path)
                            except Exception:
                                pass
                        st.session_state.pop("uploaded_file_name", None)
                        st.session_state.pop("uploaded_file_size", None)
                        st.session_state.pop("video_file_path", None)
                        video_file_path = None

            with col_in2:
                st.subheader("Executar Arbitragem")
                st.markdown("Inicie o rastreamento de pose, detecção de impactos e avaliação de Yuko-Datotsu:")

                dev_pref = st.session_state.get("device_preference", get_processing_device())
                effective_dev, dev_msg, dev_gpu = get_effective_device(dev_pref)
                if effective_dev == "gpu":
                    st.markdown(f'<div style="background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem; color: #4ade80;">🚀 <b>Aceleração GPU Ativa:</b> {dev_gpu.get("gpu_name", "NVIDIA GPU")} (YOLOv8-Pose CUDA)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background: rgba(148, 163, 184, 0.12); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.88rem; color: #cbd5e1;">💻 <b>Processamento por CPU:</b> MediaPipe Pose (TFLite CPU)</div>', unsafe_allow_html=True)

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
                        output_video_path=annotated_output,
                        invert_combatants=st.session_state.get("invert_aka_shiro", False)
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
                    # Banner indicativo se o vídeo foi posicionado em um evento específico
                    if "video_seek_label" in st.session_state:
                        seek_lbl = st.session_state["video_seek_label"]
                        seek_time_val = st.session_state.get("video_start_time", 0.0)
                        col_sk1, col_sk2 = st.columns([3.5, 1.5])
                        with col_sk1:
                            st.markdown(
                                f'<div style="background: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 6px; padding: 6px 10px; font-size: 0.82rem; color: #93C5FD; margin-bottom: 6px;">'
                                f'🎯 <b>Posicionado em {seek_time_val:.1f}s</b> (1s antes do evento)<br>'
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

                    m_t1, m_t2 = st.columns(2)
                    m_t1.metric("⏱️ Tempo de Processamento", f"{proc_time:.2f}s")
                    m_t2.metric("⚡ Taxa de Processamento", f"{proc_fps:.1f} FPS" if proc_fps > 0 else "N/A")

                    if sonkyo_info.get("status_message"):
                        st.caption(f"🥋 **Status do Sonkyō:** {sonkyo_info['status_message']}")
                    st.caption(f"ℹ️ {res.get('device_status', '')}")
                    st.markdown('</div>', unsafe_allow_html=True)

            with col_results:
                st.subheader("🥋 Eventos Identificados & Diagnóstico de Combate")
                if "analysis_result" not in st.session_state:
                    st.info("👈 Clique em **⚡ Executar Arbitragem** para visualizar a linha do tempo de eventos e análise detalhada.")
                else:
                    res = st.session_state["analysis_result"]
                    video_name_simple = os.path.basename(video_file_path) if video_file_path else "recorded_match.mp4"
                    # 0. PLACAR OFICIAL DE ARBITRAGEM (SANBON-SHOBU) & CONTROLE DE PONTUAÇÃO
                    is_inverted = st.session_state.get("invert_aka_shiro", False)
                    raw_scoreboard = res.get("scoreboard", {})
                    
                    # Extração dos golpes válidos (Ippon)
                    raw_aka_strikes = [ev for ev in res.get("events", []) if ev["event_info"].get("attacker_id") == "KENSHI_AKA" and ev["evaluation"].get("is_valid", False)]
                    raw_shiro_strikes = [ev for ev in res.get("events", []) if ev["event_info"].get("attacker_id") == "KENSHI_SHIRO" and ev["evaluation"].get("is_valid", False)]

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
                        aka_items = "".join([f'<span style="display:inline-block; background: #991B1B; color: #FEE2E2; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; margin: 2px;">🔴 {s["event_info"]["type"]} ({s["event_info"]["timestamp"]})</span>' for s in aka_val_strikes])
                    else:
                        aka_items = '<span style="color: #9CA3AF; font-size: 12px; font-style: italic;">Nenhum Ippon validado</span>'

                    # HTML de Ippons do Shiro
                    if shiro_val_strikes:
                        shiro_items = "".join([f'<span style="display:inline-block; background: #475569; color: #F8FAFC; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 11px; margin: 2px;">⚪ {s["event_info"]["type"]} ({s["event_info"]["timestamp"]})</span>' for s in shiro_val_strikes])
                    else:
                        shiro_items = '<span style="color: #9CA3AF; font-size: 12px; font-style: italic;">Nenhum Ippon validado</span>'

                    st.markdown(
                        f"""
                        <div style="background: #090D16; border: 2px solid #374151; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);">
                            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1F2937; padding-bottom: 8px; margin-bottom: 12px;">
                                <span style="color: #D1D5DB; font-size: 13px; font-weight: 800; letter-spacing: 0.8px;">🥋 PLACAR OFICIAL DE ARBITRAGEM</span>
                                <span style="color: #93C5FD; font-size: 12px; font-weight: 500;">{flag_badge}</span>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                                <div style="background: linear-gradient(180deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.22) 100%); border: 1.5px solid #EF4444; border-radius: 8px; padding: 12px; text-align: center;">
                                    <div style="color: #FCA5A5; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;">🔴 KENSHI AKA (VERMELHO)</div>
                                    <div style="color: #FFFFFF; font-size: 38px; font-weight: 900; font-family: monospace; line-height: 1.1; margin: 4px 0;">{aka_score_val} <span style="font-size: 14px; font-weight: 700; color: #FCA5A5;">IPPON</span></div>
                                    <div style="margin-top: 6px;">{aka_items}</div>
                                </div>
                                <div style="background: linear-gradient(180deg, rgba(243, 244, 246, 0.10) 0%, rgba(100, 116, 139, 0.18) 100%); border: 1.5px solid #E5E7EB; border-radius: 8px; padding: 12px; text-align: center;">
                                    <div style="color: #F3F4F6; font-size: 13px; font-weight: 800; letter-spacing: 0.5px;">⚪ KENSHI SHIRO (BRANCO)</div>
                                    <div style="color: #FFFFFF; font-size: 38px; font-weight: 900; font-family: monospace; line-height: 1.1; margin: 4px 0;">{shiro_score_val} <span style="font-size: 14px; font-weight: 700; color: #E5E7EB;">IPPON</span></div>
                                    <div style="margin-top: 6px;">{shiro_items}</div>
                                </div>
                            </div>
                            <div style="background: {winner_bg}; border: 1px solid {winner_border}; border-radius: 6px; padding: 8px; margin-top: 10px; text-align: center;">
                                <span style="color: {winner_color}; font-size: 15px; font-weight: 800;">{winner_txt}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Botão de Ação Rápida de Inversão de Cores/Lutadores
                    col_inv1, col_inv2 = st.columns([1.6, 2.4])
                    with col_inv1:
                        if st.button("🔄 Inverter Lutadores (Aka ⇄ Shiro)", width="stretch", key="btn_toggle_invert_aka_shiro", help="Inverte os lados de Aka e Shiro na pontuação, nos relatórios e nos eventos caso a câmera esteja invertida"):
                            st.session_state["invert_aka_shiro"] = not is_inverted
                            st.toast(f"🔄 Identidades invertidas: Aka ⇄ Shiro {'(Ativado)' if not is_inverted else '(Restaurado)'}!", icon="🔄")
                            st.rerun()
                    with col_inv2:
                        st.caption("ℹ️ Caso a câmera esteja gravando pelo lado oposto do Shiaijo ou a cor da flag tenha sido afetada, utilize o botão ao lado para inverter a pontuação.")

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
                    if "sonkyo_edits" not in st.session_state:
                        st.session_state["sonkyo_edits"] = {}

                    sonkyo_edits = st.session_state.get("sonkyo_edits", {})

                    # Banner de Ação de Reprocessamento com Aprendizado quando o Sonkyō for editado
                    if sonkyo_edits:
                        st.markdown(
                            """
                            <div style="background: linear-gradient(135deg, #1E1B4B 0%, #312E81 100%); border: 2px solid #818CF8; border-radius: 10px; padding: 14px 18px; margin-bottom: 15px;">
                                <h4 style="color: #E0E7FF; margin: 0 0 6px 0;">⚡ Momentos de Sonkyō Alterados pelo Árbitro</h4>
                                <p style="color: #C7D2FE; font-size: 0.90rem; margin: 0 0 10px 0;">
                                    Os limites regulamentares de Sonkyō foram modificados. O ShinpanAI irá <b>aprender a movimentação corporal</b> deste combate para reprocessar a arbitragem e aplicar o aprendizado em todas as próximas análises.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        col_rep1, col_rep2 = st.columns([3, 1])
                        with col_rep1:
                            if st.button("🔄 Reprocessar Arbitragem com Aprendizado de Sonkyō", type="primary", width="stretch", key="btn_reprocess_sonkyo_learning"):
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

                    # Seletor Rápido de Navegação por Eventos
                    jump_options = ["-- 🎯 Selecionar evento para saltar no vídeo (-1s antes) --"]
                    jump_map = {}

                    has_init_jump = sonkyo_info.get("has_initial_sonkyo", False) and sonkyo_info.get("initial_sonkyo")
                    init_edit_jump = sonkyo_edits.get("initial")
                    if has_init_jump or init_edit_jump:
                        ts_i = init_edit_jump.get("start_timestamp") if init_edit_jump else sonkyo_info.get("initial_sonkyo", {}).get("start_timestamp", "00:00.000")
                        label_i = f"🥋 Sonkyō Inicial (Abertura) @ {ts_i}"
                        jump_options.append(label_i)
                        jump_map[label_i] = (max(0.0, parse_ts_to_seconds(ts_i) - 1.0), label_i)

                    for idx_j, ev_data_j in enumerate(res.get("events", [])):
                        ev_ij = ev_data_j["event_info"]
                        ev_eval_j = ev_data_j["evaluation"]
                        status_sym_j = "✅ Ippon" if ev_eval_j.get("is_valid", False) else "❌ Inválido"
                        att_j = ev_ij.get("attacker_name", "Kenshi Aka (Vermelho)")
                        if is_inverted:
                            att_j = "Kenshi Shiro (Branco)" if "AKA" in ev_ij.get("attacker_id", "KENSHI_AKA") else "Kenshi Aka (Vermelho)"
                        label_sj = f"🥊 Golpe #{idx_j+1}: {ev_ij.get('type')} @ {ev_ij.get('timestamp')} ({status_sym_j} - {att_j})"
                        jump_options.append(label_sj)
                        jump_map[label_sj] = (max(0.0, parse_ts_to_seconds(ev_ij.get("timestamp")) - 1.0), label_sj)

                    has_final_jump = sonkyo_info.get("has_final_sonkyo", False) and sonkyo_info.get("final_sonkyo")
                    final_edit_jump = sonkyo_edits.get("final")
                    if has_final_jump or final_edit_jump:
                        ts_fj = final_edit_jump.get("start_timestamp") if final_edit_jump else sonkyo_info.get("final_sonkyo", {}).get("start_timestamp", "00:00.000")
                        label_fj = f"🥋 Sonkyō Final (Encerramento) @ {ts_fj}"
                        jump_options.append(label_fj)
                        jump_map[label_fj] = (max(0.0, parse_ts_to_seconds(ts_fj) - 1.0), label_fj)

                    if len(jump_options) > 1:
                        selected_jump = st.selectbox(
                            "🎯 Navegação Rápida de Eventos no Vídeo (1s antes):",
                            options=jump_options,
                            index=0,
                            key="event_quick_jump_select",
                            help="Selecione um evento para saltar o vídeo automaticamente para 1 segundo antes do ocorrido para facilitar a revisão."
                        )
                        if selected_jump in jump_map:
                            target_sec, target_lbl = jump_map[selected_jump]
                            if st.session_state.get("video_start_time") != target_sec or st.session_state.get("video_seek_label") != target_lbl:
                                st.session_state["video_start_time"] = target_sec
                                st.session_state["video_seek_label"] = target_lbl
                                st.toast(f"🎥 Vídeo posicionado em {target_sec:.1f}s (1s antes)!", icon="🎬")
                                st.rerun()

                    with st.container(height=650):
                        has_initial = sonkyo_info.get("has_initial_sonkyo", False) and sonkyo_info.get("initial_sonkyo")
                        has_final = sonkyo_info.get("has_final_sonkyo", False) and sonkyo_info.get("final_sonkyo")
                        has_strikes = bool(res.get("events"))

                        if not has_initial and not has_final and not has_strikes and not sonkyo_edits:
                            st.warning("Nenhum evento (Sonkyō ou Golpes) foi identificado no vídeo.")
                        else:
                            # 1. EVENTO DE SONKYŌ INICIAL (Abertura do Combate)
                            initial_edit = sonkyo_edits.get("initial")
                            if has_initial or initial_edit:
                                init_s = sonkyo_info.get("initial_sonkyo") or {}
                                is_init_detected = init_s.get("is_detected", True)
                                curr_start_ts = initial_edit.get("start_timestamp") if initial_edit else init_s.get("start_timestamp", "00:00.000")
                                curr_end_ts = initial_edit.get("end_timestamp") if initial_edit else init_s.get("end_timestamp", "00:01.500")
                                
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
                                        if st.button("🎬 Assistir no Vídeo (1s antes)", key="btn_seek_sonkyo_init", help="Reproduzir o vídeo 1 segundo antes do início do Sonkyō Inicial"):
                                            st.session_state["video_start_time"] = seek_init_s
                                            st.session_state["video_seek_label"] = f"Sonkyō Inicial ({curr_start_ts})"
                                            st.toast(f"🎥 Vídeo posicionado em {seek_init_s:.1f}s (1s antes)", icon="🎬")
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

                            # 2. GOLPES DETECTADOS NA JANELA REGULAMENTAR DE COMBATE
                            if not has_strikes:
                                st.info("ℹ️ Nenhum golpe regulamentar registrado entre os momentos de Sonkyō.")
                            else:
                                for idx, ev_data in enumerate(res["events"]):
                                    ev = ev_data["event_info"]
                                    eval_info = ev_data["evaluation"]
                                    event_id_str = f"event_{idx+1}_frame_{ev['impact_frame']}"
                                    
                                    orig_att_name = ev.get("attacker_name", "Kenshi Aka (Vermelho)")
                                    orig_att_id = ev.get("attacker_id", "KENSHI_AKA")
                                    if is_inverted:
                                        attacker_label = "Kenshi Shiro (Branco)" if "AKA" in orig_att_id else "Kenshi Aka (Vermelho)"
                                    else:
                                        attacker_label = orig_att_name

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
                                    current_rev["attacker_name"] = attacker_label

                                    if current_rev.get("is_edited"):
                                        status_badge = "✏️ EDITADO"
                                    elif current_rev.get("is_confirmed"):
                                        status_badge = "✅ CONFIRMADO"
                                    elif eval_info['is_valid']:
                                        status_badge = "✅ IPPON"
                                    else:
                                        status_badge = "❌ INVÁLIDO"

                                    with st.expander(f"🥊 Golpe #{idx+1}: {current_rev['strike_type']} @ {current_rev['timestamp']} ({attacker_label}) - {status_badge}", expanded=True):
                                        c_a, c_b = st.columns([1, 1.5])
                                        with c_a:
                                            seek_strike_s = max(0.0, parse_ts_to_seconds(current_rev['timestamp']) - 1.0)
                                            if st.button(f"🎬 Assistir Golpe no Vídeo (1s antes: {seek_strike_s:.1f}s)", key=f"btn_seek_strike_{idx}", width="stretch", help=f"Salta a reprodução para 1 segundo antes do impacto deste golpe ({seek_strike_s:.1f}s)"):
                                                st.session_state["video_start_time"] = seek_strike_s
                                                st.session_state["video_seek_label"] = f"Golpe #{idx+1} {current_rev['strike_type']} @ {current_rev['timestamp']}"
                                                st.toast(f"🎥 Vídeo posicionado em {seek_strike_s:.1f}s (1s antes do impacto)!", icon="🎬")
                                                st.rerun()

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
                                        if st.button("🎬 Assistir no Vídeo (1s antes)", key="btn_seek_sonkyo_fin", help="Reproduzir o vídeo 1 segundo antes do início do Sonkyō Final"):
                                            st.session_state["video_start_time"] = seek_fin_s
                                            st.session_state["video_seek_label"] = f"Sonkyō Final ({curr_start_ts_fin})"
                                            st.toast(f"🎥 Vídeo posicionado em {seek_fin_s:.1f}s (1s antes)", icon="🎬")
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

