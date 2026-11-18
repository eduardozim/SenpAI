"""
Painel do diagnóstico detalhado do modo de treinamento (src/analytics/training_feedback.py).

Mostra, para cada kendoca: pontos fortes, pontos de atenção do mais urgente ao mais leve (o que foi visto,
por que importa, causas prováveis e botões que levam o vídeo às piores repetições), o que não pôde ser
medido, os exercícios com dose e progressão, o plano da semana e as metas para a próxima sessão.
"""

import html
from typing import Any, Dict

import streamlit as st

from src.analytics.training_feedback import SEVERITY_LABEL, fmt_time

SEVERITY_ICON = {"prioritaria": "🔴", "moderada": "🟠", "leve": "🟡"}
KIND_LABEL = {"corretivo": "Corretivo", "reforco": "Reforço do ponto forte", "modalidade": "Da modalidade",
              "manutencao": "Manutenção"}


def _seek(sec: float, label: str) -> None:
    st.session_state["video_start_time"] = float(sec)
    st.session_state["video_seek_label"] = label
    st.session_state["video_seek_key"] = st.session_state.get("video_seek_key", 0) + 1
    st.rerun()


def _ai_available() -> bool:
    try:
        from src.engine.web_research import get_llm
        return get_llm() is not None
    except Exception:
        return False


def _render_narrative(diag: Dict[str, Any], key_prefix: str) -> None:
    """Resumo corrido do diagnóstico, com botão para a IA reescrever (só a partir das medidas)."""
    from src.analytics.training_narrative import ai_narrative, narrative_text
    nar = narrative_text(diag)
    fonte = ("✍️ Escrito pela IA a partir das medidas desta sessão" if nar.get("fonte") == "ia"
             else "📐 Gerado a partir das medidas desta sessão")
    st.markdown("##### 📝 Resumo do Sensei")
    st.markdown(nar["texto"])
    alert_key = f"narrative_alert_{key_prefix}"
    if alert_key in st.session_state:
        st.warning(st.session_state.pop(alert_key))
    c1, c2 = st.columns([3, 1.4])
    with c1:
        st.caption(fonte + ". O nome do kenshi não é enviado à IA; textos com números que não estão no diagnóstico "
                           "são descartados.")
    with c2:
        online = _ai_available()
        label = "✍️ Reescrever com a IA" if nar.get("fonte") == "ia" else "✍️ Escrever com a IA"
        if st.button(label, key=f"btn_narrative_{key_prefix}", width="stretch", disabled=not online,
                     help=("O assistente LLM escreve o resumo como um sensei, usando só as medidas do diagnóstico."
                           if online else "IA desligada: informe a chave do Gemini ou da OpenAI em "
                                          "Configurações > Variáveis do Sistema.")):
            with st.spinner("A IA está escrevendo o resumo..."):
                new_nar = ai_narrative(diag)
            if new_nar.get("fonte") == "ia":
                diag["narrativa"] = new_nar      # fica no resultado da análise: tela, Markdown e PDF
            elif new_nar.get("aviso"):
                st.session_state[alert_key] = new_nar["aviso"]
            st.rerun()


def _render_evolution(diag: Dict[str, Any], kendoka_name: str) -> None:
    """Comparação com a sessão anterior do mesmo nome (src/analytics/training_progress.py)."""
    evo = diag.get("evolucao")
    if evo is None:
        return
    st.markdown("##### 📈 Evolução")
    if evo.get("sem_nome"):
        st.caption("Dê um nome a este kendoca para guardar a sessão no histórico dele e comparar com as próximas. "
                   "O mesmo kenshi é reconhecido entre vídeos pelo nome, então use sempre o mesmo.")
        return
    if evo.get("primeira_sessao"):
        st.caption(f"Primeira sessão de **{html.escape(str(evo.get('nome') or kendoka_name))}** no histórico. "
                   "As próximas sessões com o mesmo nome serão comparadas com esta.")
        return
    prev = evo.get("anterior", {})
    try:
        from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA
        mod_name = TRAINING_MODALITIES_METADATA.get(prev.get("modalidade", ""), {}).get("name", prev.get("modalidade", ""))
    except Exception:
        mod_name = prev.get("modalidade", "")
    geral = evo.get("geral", {})
    delta = float(geral.get("delta") or 0.0)
    st.caption(f"Comparado com a sessão de **{prev.get('data', '?')}** ({html.escape(str(mod_name))}"
               + ("" if prev.get("mesma_modalidade") else ", outra modalidade: compare com cuidado")
               + f"). {evo.get('sessoes_no_historico', 0)} sessões no histórico de **{html.escape(str(evo.get('nome', '')))}**.")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Nota geral", f"{float(geral.get('agora') or 0):.1f}", f"{delta:+.1f} desde a anterior")
        metas = evo.get("metas_anteriores", [])
        if metas:
            st.markdown("**Metas da sessão anterior:**")
            for g in metas:
                if g["atingida"] is None:
                    st.markdown(f"❔ {html.escape(g['label'])}: não medido nesta sessão")
                else:
                    icon = "✅" if g["atingida"] else "⏳"
                    st.markdown(f"{icon} {html.escape(g['label'])}: {g['agora']:.0f}% (meta {g['meta']:.0f}%)")
    with c2:
        rows = evo.get("metricas", [])
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame([{"Métrica": r["label"], "Antes": f"{r['antes']:.0f}%", "Agora": f"{r['agora']:.0f}%",
                                        "Variação": f"{'▲' if r['delta'] > 0 else ('▼' if r['delta'] < 0 else '•')} {r['delta']:+.0f}"}
                                       for r in rows]), hide_index=True, width="stretch")
    hist = evo.get("historico", [])
    if len(hist) >= 3:
        import pandas as pd
        st.line_chart(pd.DataFrame([{"Sessão": f"{h['data']} #{i + 1}", "Nota geral": h["geral"]} for i, h in enumerate(hist)])
                      .set_index("Sessão"), height=160)


def render_training_diagnosis(diag: Dict[str, Any], key_prefix: str, kendoka_name: str = "",
                              seekable: bool = True) -> None:
    """
    Renderiza o diagnóstico estruturado de um kendoca. `seekable=False` (sessão ao vivo, sem vídeo gravado)
    mostra os instantes das piores repetições como texto, sem os botões que levam o vídeo ao instante.
    """
    _render_narrative(diag, key_prefix)
    _render_evolution(diag, kendoka_name)
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("##### 🌟 Pontos Fortes")
        fortes = diag.get("pontos_fortes", [])
        if not fortes:
            st.caption("Nenhuma métrica acima da referência nesta sessão.")
        for s in fortes:
            st.markdown(f"✅ <span style='font-size:13px;'><b>{html.escape(s['label'])} ({s['valor']:.0f}%)</b> — "
                        f"{html.escape(s['texto'])}</span>", unsafe_allow_html=True)
    with d_col2:
        st.markdown("##### 🎯 Metas para a Próxima Sessão")
        metas = diag.get("metas", [])
        if not metas:
            st.caption("Manter as notas atuais.")
        for m in metas:
            st.markdown(f"🎯 <span style='font-size:13px;'>{html.escape(m['texto'])}</span>", unsafe_allow_html=True)

    st.markdown("##### ⚠️ Pontos de Atenção (do mais urgente ao mais leve)")
    achados = diag.get("achados", [])
    if not achados:
        st.success("Todas as métricas medidas estão dentro da referência.")
    for i, f in enumerate(achados):
        title = (f"{SEVERITY_ICON[f['gravidade']]} [{SEVERITY_LABEL[f['gravidade']]}] {f['label']}: "
                 f"{f['valor']:.0f}% (referência {f['ref']:.0f}%)")
        with st.expander(title, expanded=(f["gravidade"] == "prioritaria")):
            st.markdown(f"**O que foi visto:** {f['o_que']}")
            st.markdown(f"**Por que importa:** {f['por_que']}")
            st.markdown("**Causas prováveis:**\n" + "\n".join(f"- {c}" for c in f["causas"]))
            momentos = [m for m in f.get("momentos", []) if m.get("tempo_s") is not None]
            if momentos and not seekable:
                st.markdown(f"**Onde (tempo da sessão):** {f['onde']}")
            elif momentos:
                st.markdown("**Onde ver no vídeo:**")
                cols = st.columns(len(momentos))
                for j, (col, m) in enumerate(zip(cols, momentos)):
                    with col:
                        lbl = f"▶ {fmt_time(m['tempo_s'])}" + (f" · {m['valor']:.0f}%" if m.get("valor") is not None else "")
                        if st.button(lbl, key=f"diag_seek_{key_prefix}_{i}_{j}", width="stretch",
                                     help=m.get("detalhe") or f"Ir para {fmt_time(m['tempo_s'])} no vídeo"):
                            _seek(m["tempo_s"], f"{kendoka_name or 'Kendoca'}: {f['label']} ({fmt_time(m['tempo_s'])})")
                        if m.get("detalhe"):
                            st.caption(m["detalhe"])
            elif f.get("onde"):
                st.markdown(f"**Onde:** {f['onde']}")

    for n in diag.get("nao_medidos", []):
        st.caption(f"❔ **{n['label']}** não medido nesta sessão: {n['motivo']}.")

    st.markdown("##### 🏋️ Exercícios Recomendados")
    for k, ex in enumerate(diag.get("exercicios", [])):
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #6366F1; border-radius: 6px; padding: 10px 14px; margin-bottom: 4px;">
                <div style="color: #A5B4FC; font-weight: 800; font-size: 13px;">🥋 {html.escape(ex['name'])}
                    <span style="color:#94A3B8; font-size:11px; font-weight:400;">({html.escape(KIND_LABEL.get(ex['tipo'], ex['tipo']))} · {html.escape(ex['formato'])} · nível {html.escape(ex['nivel'])})</span></div>
                <div style="color: #CBD5E1; font-size: 12px; margin-top: 3px;">💡 {html.escape(ex['motivo'])}</div>
                <div style="color: #E2E8F0; font-size: 12px; margin-top: 3px;">📋 <b>Dose:</b> {html.escape(ex['dose'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Como fazer", expanded=False):
            if ex.get("passos"):
                st.markdown("**Progressão:**\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(ex["passos"], 1)))
            if ex.get("pontos_de_atencao"):
                st.markdown("**Pontos de atenção:**\n" + "\n".join(f"- {p}" for p in ex["pontos_de_atencao"]))
            if ex.get("erros_comuns"):
                st.markdown("**Erros comuns:**\n" + "\n".join(f"- {p}" for p in ex["erros_comuns"]))
            if ex.get("como_medir"):
                st.markdown(f"**Como medir a evolução:** {ex['como_medir']}")

    if diag.get("plano"):
        st.markdown("##### 📅 Plano da Semana")
        st.markdown("\n".join(f"- {p}" for p in diag["plano"]))
