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


def _elapsed(iso: Any) -> str:
    import datetime
    try:
        s = int((datetime.datetime.now() - datetime.datetime.fromisoformat(str(iso))).total_seconds())
    except Exception:
        return ""
    return f"{s // 3600} h {s % 3600 // 60:02d} min" if s >= 3600 else f"{s // 60} min {s % 60:02d} s"


def running_text(state: Dict[str, Any]) -> str:
    """Etapa atual (N de M), o que ela está fazendo agora e há quanto tempo."""
    text = f"⏳ Ciclo em andamento: **{state.get('etapa') or 'iniciando'}**"
    if state.get("etapa_inicio"):
        text += f" — nesta etapa há {_elapsed(state['etapa_inicio'])}"
    if state.get("detalhe"):
        text += f"\n\n↳ {state['detalhe']}"
    if state.get("inicio"):
        text += f"\n\nCiclo iniciado há {_elapsed(state['inicio'])}."
    return text


def internet_status_text(st_: Dict[str, Any]) -> str:
    """Uma linha com a situação real das lutas baixadas da internet (nada estimado)."""
    return (f"**{st_['lutas']}** luta(s) baixada(s) · **{st_['preprocessadas']}** analisada(s) "
            f"({st_['golpes_analisados']} golpes) · **{st_['pendentes']}** por analisar · "
            f"**{st_['duvidas_abertas']}** dúvida(s) delas esperando resposta na Fila de Curadoria")


def render_internet_preprocess_section(videos_dir: str, device_preference: str = "gpu") -> None:
    """Lutas da internet: pré-processar (análise + dúvidas na curadoria) e acompanhar."""
    from src.engine import internet_preprocess as ip
    state = ip.get_state()
    with st.container(border=True):
        st.markdown("**🌐 Lutas baixadas da internet**")
        st.caption("Baixada, uma luta só tem a modalidade pelo título. O pré-processamento analisa cada uma (com o cache "
                   "de poses) e manda à Fila de Curadoria os golpes em que o SenpAI está em dúvida. Suas respostas, com "
                   "o seu dan, viram rótulos e o próximo ciclo de treino aprende com elas. O ciclo também faz isso sozinho.")
        other = None
        try:
            st_now = ip.status(videos_dir)
            other = st_now.get("rodando_em_outro_processo")
            st.markdown(internet_status_text(st_now))
        except Exception as e:
            st.warning(f"Não foi possível ler a situação das lutas da internet: {e}")
        if other and not state["rodando"]:
            st.info(f"⏳ Já há um pré-processamento rodando em outro processo (PID {other}, ex.: linha de comando). "
                    "Os números acima se atualizam ao recarregar a página.")
        if state["rodando"]:
            st.info(f"⏳ Pré-processando… {state.get('detalhe') or 'iniciando'}"
                    + (f" (há {_elapsed(state['inicio'])})" if state.get("inicio") else ""))
            if st.button("⏹️ Parar depois desta luta", key="btn_stop_internet_preprocess"):
                ip.request_stop()
            time.sleep(2.0)
            st.rerun()
        cycle_running = tc.get_cycle_state().get("rodando")
        if st.button("⚙️ Pré-processar agora", key="btn_run_internet_preprocess",
                     disabled=state["rodando"] or bool(cycle_running) or bool(other),
                     help="Analisa as lutas da internet ainda não vistas. Para sozinho se a fila de curadoria encher."):
            if ip.start_thread(videos_dir, device_preference=device_preference):
                st.toast("Pré-processamento iniciado.", icon="⚙️")
            time.sleep(0.3)
            st.rerun()
        if state.get("resultado"):
            st.success(ip.summary_text(state["resultado"]))
        if state.get("erro"):
            st.error(f"O pré-processamento parou com erro: {state['erro']}")


def render_training_cycle_panel(
    device_preference: str = "gpu",
    is_web_server: bool = False,
    title: str = "#### 🔄 3. Ciclo de Treino"
) -> None:
    st.markdown("---")
    st.markdown(title)
    st.caption(
        "O SenpAI melhora com dados, não com tempo de execução. O ciclo usa só o que você já classificou e o que a "
        "procura ativa já coletou (ele não coleta nada). Cada ciclo: (1) extrai as poses dos vídeos novos e analisa as lutas novas da internet, mandando "
        "os golpes em dúvida à curadoria; (2) junta "
        "as revisões dos golpes (só elas dizem se um golpe é válido) e manda à curadoria os pontos dos shinpan ainda não "
        "revisados; (3) treina filtro de golpes, técnica e modalidade, e só coloca em uso o que melhorar; (4) mede o "
        "sistema no golden. Etapas sem dados suficientes "
        "são puladas, com o motivo."
    )
    if is_web_server:
        st.info("Disponível no ambiente Local: o ciclo usa os vídeos, o cache de poses e a GPU da máquina.")
        return

    st.session_state.setdefault("labels_videos_dir", DEFAULT_VIDEOS_DIR)
    videos_dir = st.session_state.get("labels_videos_dir", DEFAULT_VIDEOS_DIR)
    state = tc.get_cycle_state()
    render_internet_preprocess_section(videos_dir, device_preference)

    with st.container(border=True):
        st.caption(f"Pasta dos vídeos: `{videos_dir}` (a mesma da aba Gabarito de Lutas).")
        if state["rodando"]:
            st.info(running_text(state))
            time.sleep(2.0)
            st.rerun()
        ready = tc.readiness(videos_dir)
        confirmed = True
        if ready["pendencias"]:
            with st.container(border=True):
                st.markdown("**⚠️ Ainda há classificação pendente.** O ciclo treina só com o que já está rotulado:")
                for it in ready["pendencias"]:
                    st.markdown(f"- **{it['quantidade']}** {it['texto']} — _{it['onde']}_")
                confirmed = st.checkbox("Rodar mesmo assim, com o que já está classificado", key="cycle_confirm_pending")
        else:
            st.success(f"✅ Tudo classificado: {ready['videos']} vídeo(s) prontos para o ciclo.")
        if not ready["golden"]:
            st.caption("Sem lutas no golden (Perfis de Calibração > Eixo 4): os modelos treinam, mas a medição final é pulada.")
        from src.engine.internet_preprocess import get_state as _pre_state
        if st.button("▶️ Rodar ciclo agora", type="primary", key="btn_run_training_cycle",
                     disabled=state["rodando"] or not confirmed or bool(_pre_state().get("rodando"))):
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
