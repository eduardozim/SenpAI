"""
Painel do Ciclo de Treino (Pendência I8), em Configurações > Governança de Treinamento.
Substitui o antigo "Treinamento Automático por IA", que simulava aprendizado por tempo.
"""

import os
import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.engine import training_cycle as tc

STATUS_ICON = {"ok": "✅", "pulado": "⏭️", "falhou": "❌"}
from src.utils.video_library import DEFAULT_VIDEOS_DIR  # pasta única: Dev/data/videos


def cycles_table(history: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in reversed(history):
        steps = r.get("etapas", {})
        medicao = steps.get("medicao", {}).get("total") or {}
        f1 = medicao.get("f1")
        rows.append({
            "quando": r.get("iniciado_em", ""),
            **{tc.STEP_NAMES[k]: STATUS_ICON.get(steps.get(k, {}).get("status"), "—") for k in tc.STEP_NAMES},
            "modelo promovido": "sim" if r.get("promoveu_modelo") else "não",
            "F1 no golden": "—" if f1 is None else f"{f1 * 100:.0f}%",
        })
    return pd.DataFrame(rows)


def render_training_cycle_panel(
    device_preference: str = "gpu",
    is_web_server: bool = False,
    title: str = "#### 🔄 3. Ciclo de Treino"
) -> None:
    st.markdown("---")
    st.markdown(title)
    st.caption(
        "O SenpAI melhora com dados, não com tempo de execução. Cada ciclo: (1) extrai as poses dos vídeos novos; "
        "(2) grava pontos lidos das bandeiras, se o detector já provou ser confiável; (3) treina um filtro de golpes "
        "desafiante e só o coloca em uso se ele melhorar o Golden Benchmark; (4) mede o sistema no golden. "
        "Etapas sem dados suficientes são puladas, com o motivo."
    )
    if is_web_server:
        st.info("Disponível no ambiente Local: o ciclo usa os vídeos, o cache de poses e a GPU da máquina.")
        return

    st.session_state.setdefault("labels_videos_dir", DEFAULT_VIDEOS_DIR)
    videos_dir = st.session_state.get("labels_videos_dir", DEFAULT_VIDEOS_DIR)
    state = tc.get_cycle_state()

    with st.container(border=True):
        st.caption(f"Pasta dos vídeos: `{videos_dir}` (a mesma da aba Gabarito de Lutas).")
        if state["rodando"]:
            st.info(f"⏳ Ciclo em andamento: **{state['etapa'] or 'iniciando'}**...")
            time.sleep(2.0)
            st.rerun()
        if st.button("▶️ Rodar ciclo agora", type="primary", key="btn_run_training_cycle", disabled=state["rodando"]):
            if tc.start_cycle_thread(videos_dir, device_preference=device_preference):
                st.toast("Ciclo de treino iniciado.", icon="🔁")
            time.sleep(0.3)
            st.rerun()
        if state.get("erro"):
            st.error(f"O último ciclo parou com erro: {state['erro']}")

        history = tc.load_cycle_history()
        if not history:
            st.info("Nenhum ciclo executado ainda.")
        else:
            last = history[-1]
            st.markdown(f"**Último ciclo** ({last.get('iniciado_em', '')}):")
            for key, name in tc.STEP_NAMES.items():
                step = last.get("etapas", {}).get(key)
                if step:
                    st.markdown(f"- {STATUS_ICON.get(step['status'], '—')} **{name}:** {step.get('detalhe', '')}")
            with st.expander(f"Histórico de ciclos ({len(history)})", expanded=False):
                st.dataframe(cycles_table(history), width="stretch", hide_index=True)

    with st.expander("⏰ Rodar automaticamente (agendador do Windows)", expanded=False):
        st.caption("Para o ciclo rodar sozinho, por exemplo toda noite às 2h, crie uma tarefa no Agendador de Tarefas "
                   "com o comando abaixo (a partir da pasta Dev):")
        st.code(f'.venv_win\\Scripts\\python.exe -m src.engine.training_cycle run "{videos_dir}"', language="bash")
