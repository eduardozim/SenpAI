"""
Métricas de treinamento e espaço em disco ("quanto o sistema já foi treinado").
Usa a mesma lista do apagar/exportar (src/engine/training_dataset.py): só números reais do disco.
"""

import pandas as pd
import streamlit as st

from src.engine import training_dataset as td

_CARD = ('<div style="background: rgba(15, 23, 42, 0.65); border: 1px solid {border}; border-radius: 10px; '
         'padding: 12px 14px; min-height: 105px;">'
         '<div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px;">'
         '<span style="font-size: 0.78rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">{title}</span>'
         '<span style="font-size: 1.15rem; font-weight: 700; color: {color};">{size}</span></div>'
         '<div style="font-size: 0.74rem; color: #94A3B8; margin-top: 4px;">{desc}</div></div>')
_COLORS = {"dataset": ("#38BDF8", "rgba(56, 189, 248, 0.35)"), "modelos": ("#818CF8", "rgba(129, 140, 248, 0.35)"),
           "conhecimento": ("#34D399", "rgba(52, 211, 153, 0.35)")}


def _fmt_metric(v):
    return "não medido" if v is None else f"{v:.2f}"


def render_training_metrics(training_metrics) -> None:
    summary = td.summarize()
    progress = td.training_progress()
    storage = td.storage_report()

    st.markdown("#### 📈 Quanto o sistema já foi treinado")
    c = st.columns(6)
    c[0].metric("Rótulos no banco", summary["rotulos_no_banco"], help="Banco Único de Rótulos: tudo de onde o ciclo de treino aprende.")
    c[1].metric("Lutas com gabarito", summary["lutas_com_gabarito"])
    c[2].metric("Lutas no golden", summary["lutas_no_golden"], help="Lutas congeladas para medir o sistema (nunca treinam).")
    c[3].metric("Modelos treinados", f"{progress['modelos_ativos']}/{len(td.MODEL_FAMILIES)}",
                help="Modelos aprendidos em uso: " + ", ".join(m["nome"] for m in progress["modelos"].values()))
    c[4].metric("Ciclos de treino", summary["ciclos_de_treino"])
    c[5].metric("Disco do treinamento", td.format_bytes(storage["total_bytes"]),
                help="Dataset + modelos treinados + conhecimento e calibração (o mesmo que o botão Baixar exporta).")

    last = progress["ultima_medicao_golden"]
    if last:
        t = last.get("total", {})
        st.caption(f"🎯 Última medição no golden ({str(last.get('executado_em', ''))[:16].replace('T', ' ')}, "
                   f"{t.get('lutas', '?')} luta(s)): acurácia {_fmt_metric(t.get('acuracia'))} · precisão "
                   f"{_fmt_metric(t.get('precisao'))} · recall {_fmt_metric(t.get('recall'))} · F1 {_fmt_metric(t.get('f1'))}.")
    else:
        st.caption("🎯 Ainda não medido no golden: congele lutas em Perfis de Calibração > Eixo 4 e rode o ciclo de treino.")

    st.markdown("#### 🧑‍⚖️ Revisões por Dan")
    r = st.columns(4)
    r[0].metric("Sessões de revisão", training_metrics["total_trainings_count"])
    r[1].metric("Nível médio (Dan)", training_metrics["average_dan_label"])
    r[2].metric("Decisões dos Shinpans", f"{training_metrics.get('shinpan_trainings_count', 0)} sessões")
    r[3].metric("Golpes revisados", training_metrics["total_review_items"])

    st.markdown("#### 💾 Espaço em disco do treinamento")
    with st.container(border=True):
        cols = st.columns(3)
        for col, (key, (title, desc)) in zip(cols, td.STORE_GROUPS.items()):
            color, border = _COLORS[key]
            col.markdown(_CARD.format(title=title, size=td.format_bytes(storage["grupos"][key]), desc=desc,
                                      color=color, border=border), unsafe_allow_html=True)
        st.caption(f"Total do treinamento: **{td.format_bytes(storage['total_bytes'])}**. Apoio fora do treinamento "
                   f"(não é apagado nem exportado): **{td.format_bytes(storage['apoio_bytes'])}** — "
                   + "; ".join(f"{s['descricao']}: {td.format_bytes(s['bytes'])}" for s in storage["apoio"]) + ".")
        with st.expander("🔍 Ver cada item, arquivos e caminhos", expanded=False):
            rows = [{"Item": i["descricao"], "Caminho": i["caminho"], "Grupo": td.STORE_GROUPS[i["grupo"]][0],
                     "Arquivos": i["arquivos"], "Tamanho": td.format_bytes(i["bytes"])} for i in storage["itens"]]
            rows += [{"Item": s["descricao"], "Caminho": s["caminho"], "Grupo": "Apoio (não é treinamento)",
                      "Arquivos": s["arquivos"], "Tamanho": td.format_bytes(s["bytes"])} for s in storage["apoio"]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            ativos = [f"{m['nome']}: `{m['ativo']['arquivo']}` (desde {str(m['ativo'].get('ativado_em') or '?')[:10]})"
                      for m in progress["modelos"].values() if m["ativo"]]
            st.caption("Modelos treinados em uso: " + ("; ".join(ativos) if ativos else "nenhum (regras fixas)."))
