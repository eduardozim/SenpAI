"""
Painel de Variáveis do Sistema e Chaves de API (Configurações > Variáveis do Sistema).
Centraliza o gerenciamento de credenciais locais (Gemini, OpenAI, SerpApi) e variáveis de ambiente.
"""

import os
import sys
import streamlit as st
from typing import Any, Dict

from src.utils import api_keys as ak
from src.engine import web_research as wr


def render_system_variables_panel() -> None:
    st.markdown("### ⚙️ Variáveis do Sistema")
    st.caption(
        "Gerencie as chaves de API, credenciais de Inteligência Artificial, motores de busca e variáveis "
        "de ambiente do SenpAI. As chaves salvas ficam restritas exclusivamente a este computador em "
        "`config/api_keys.local.json` (fora do Git e fora do pacote de treinamento exportado)."
    )

    # --------------------------------------------------------------------------
    # 1. CARDS DE STATUS DOS SERVIÇOS CONECTADOS
    # --------------------------------------------------------------------------
    ai_stat = wr.ai_status()
    google_stat = wr.google_status()

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        if ai_stat["ativa"]:
            st.success(f"🧠 **IA Conectada:** {ai_stat['descricao']}")
        else:
            st.info(f"🧠 **IA em Modo Local / Offline:** {ai_stat['descricao']}")

    with col_stat2:
        if google_stat["ativo"]:
            st.success(f"🌐 **Busca Google Conectada:** {google_stat['descricao']}")
        else:
            st.info(f"🌐 **Busca Web Alternativa:** {google_stat['descricao']}")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 2. FORMULÁRIO DE CHAVES DE API E MODELOS
    # --------------------------------------------------------------------------
    st.markdown("#### 🔑 Chaves de API (IA & Busca)")
    st.caption("Deixe o campo em branco para manter a chave atual. Digite um novo valor para atualizar ou use o botão de apagar para remover.")

    with st.form("system_variables_form"):
        values: Dict[str, Any] = {}
        for name, (label, where) in ak.KNOWN_KEYS.items():
            src = ak.key_source(name)
            if src == "arquivo":
                status_badge = "✅ Configurada no arquivo local"
            elif src == "ambiente":
                status_badge = "✅ Definida nas variáveis de ambiente do SO"
            else:
                status_badge = "⚪ Não configurada"

            is_secret = "KEY" in name
            values[name] = st.text_input(
                f"{label} — {status_badge}",
                value="",
                type="password" if is_secret else "default",
                key=f"sysvar_{name}",
                help=where,
                placeholder="••••••••••••••••" if src else "Insira a chave ou valor aqui..."
            )
            st.caption(f"📌 **Onde obter:** {where}")

        c_btn1, c_btn2 = st.columns(2)
        save_btn = c_btn1.form_submit_button("💾 Salvar Variáveis", type="primary", use_container_width=True)
        clear_btn = c_btn2.form_submit_button("🗑️ Apagar Chaves Salvas", use_container_width=True)

    if save_btn:
        ak.save_keys(values)
        st.success("✅ Variáveis e chaves de API salvas com sucesso!")
        st.rerun()

    if clear_btn:
        ak.save_keys({k: None for k in ak.KNOWN_KEYS})
        st.success("🗑️ Chaves salvas apagadas (as variáveis de ambiente do sistema, se existirem, continuam ativas).")
        st.rerun()

    # --------------------------------------------------------------------------
    # 3. VARIÁVEIS DE AMBIENTE DO RUNTIME SENPAI (INSPEÇÃO SEGURA)
    # --------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🔍 Inspecionar Variáveis de Ambiente do Runtime", expanded=False):
        st.caption("Variáveis de ambiente do processo atual do SenpAI (valores sensíveis são omitidos).")
        runtime_vars = [
            ("PYTHON_EXECUTABLE", sys.executable),
            ("PYTHON_VERSION", sys.version.split()[0]),
            ("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", os.environ.get("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "51200 (50 GB)")),
            ("STREAMLIT_SERVER_MAX_MESSAGE_SIZE", os.environ.get("STREAMLIT_SERVER_MAX_MESSAGE_SIZE", "51200")),
            ("OPENCV_FFMPEG_LOGLEVEL", os.environ.get("OPENCV_FFMPEG_LOGLEVEL", "-8")),
            ("OPENCV_LOG_LEVEL", os.environ.get("OPENCV_LOG_LEVEL", "OFF")),
            ("OPENCV_VIDEOIO_DEBUG", os.environ.get("OPENCV_VIDEOIO_DEBUG", "0")),
            ("IS_STREAMLIT_CLOUD", os.environ.get("IS_STREAMLIT_CLOUD", "0")),
            ("IS_WEB_SERVER", os.environ.get("IS_WEB_SERVER", "0")),
            ("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "(todas as GPUs disponíveis)")),
        ]
        var_rows = [{"Variável de Ambiente": k, "Valor Configurado": v} for k, v in runtime_vars]
        st.dataframe(var_rows, width="stretch")
