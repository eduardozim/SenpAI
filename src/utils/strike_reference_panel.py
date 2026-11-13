"""
Painel da Referência dos Golpes Validados (Perfis de Calibração > Eixo 2).

Só leitura: mostra o que o último ciclo de treino mediu. Abaixo do mínimo de golpes, a técnica aparece como
"não medida" (nenhum número inventado). Não há botão de mineração: a referência é recalculada no ciclo.
"""

import pandas as pd
import streamlit as st

from src.analytics import strike_reference as sr


def reference_table(ref) -> pd.DataFrame:
    rows = []
    for tech in sr.TECHNIQUES:
        t = ref.get("tecnicas", {}).get(tech, {})
        row = {"Técnica": tech, "Golpes validados": t.get("golpes", 0), "Vídeos": t.get("videos", 0),
               "Situação": "✅ medida" if t.get("medido") else f"não medida (mín. {sr.MIN_SAMPLES} golpes de {sr.MIN_VIDEOS} vídeos)"}
        for p in sr.PILLARS:
            d = t.get("pilares", {}).get(p)
            row[sr.PILLAR_NAMES[p].capitalize() + " (p25–p50–p75)"] = (
                f"{d['p25']:.0%} – {d['p50']:.0%} – {d['p75']:.0%}" if d and t.get("medido") else "—")
        rows.append(row)
    return pd.DataFrame(rows)


def render_strike_reference_panel() -> None:
    st.markdown("#### 📐 Referência dos golpes validados")
    st.caption(
        "Distribuição real das notas de cada pilar nos golpes que o banco de rótulos considera válidos, por técnica. "
        "Válido = revisado no vídeo por kenshi ou sensei; a decisão dos shinpan (pontos do gabarito) não valida golpe. "
        "Cada golpe pesa o dan de quem revisou. Serve para (1) mostrar na análise em que percentil cada pilar do golpe ficou e (2) mandar à "
        "curadoria os golpes fora do padrão. Não altera pesos nem limiares (isso é da calibração, medida no golden)."
    )
    ref = sr.load_reference()
    if not ref:
        st.info("Ainda não calculada: o próximo Ciclo de Treino calcula a referência a partir dos golpes rotulados.")
        return
    st.dataframe(reference_table(ref), hide_index=True, width="stretch")
    st.caption(f"Calculada no ciclo de treino de {str(ref.get('gerado_em', '—')).replace('T', ' ')}. "
               "Golden fica de fora. Cresce com as revisões das análises e as decisões da curadoria.")
