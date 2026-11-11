"""
Painel único do Golden Benchmark na interface (Configurações > Perfis de Calibração > Eixo 4).

Tudo o que é do golden fica aqui: métricas da trava anti-regressão, lutas congeladas
(congelar, recongelar, tirar) e benchmark por luta com histórico. A verdade de cada golpe vem das
revisões (kenshi e senseis, pesadas pelo dan); a decisão dos shinpan aparece só como informação. A lógica e as métricas
vêm do GoldenBenchmark (src/engine/active_learning.py) e de src/engine/weak_labels.py.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.engine import weak_labels as wl

from src.utils.video_library import DEFAULT_VIDEOS_DIR  # pasta única: Dev/data/videos


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def golden_rows_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "luta": r["video_name"],
        "golpes revisados": r["golpes_revisados"],
        "válidos pela revisão": r["validos_revisados"],
        "acertos": r["acertos"],
        "falsos pontos": r["falsos_pontos"],
        "perdidos": r["perdidos"],
        "shinpan (A x S)": f"{r['placar_shinpan']['aka']} x {r['placar_shinpan']['shiro']}",
        "shinpan × revisão": (f"{r['shinpan_concordam']} de {r['shinpan_comparados']}" if r["shinpan_comparados"] else "—"),
    } for r in rows])


def reviewed_counts(bank=None) -> Dict[str, int]:
    """Golpes revisados (ponto/golpe) por luta no banco de rótulos: o que o golden consegue medir."""
    from src.engine.label_bank import LabelBank, SHINPAN_ORIGINS
    bank = bank or LabelBank()
    out: Dict[str, int] = {}
    for x in bank.load().values():
        if x.get("claim") in ("ponto", "golpe") and x.get("origin") not in SHINPAN_ORIGINS:
            out[x["video_name"]] = out.get(x["video_name"], 0) + 1
    return out


def golden_matches_table(gabarito: Dict[str, Dict[str, Any]], golden: Dict[str, Dict[str, Any]], gb,
                         reviewed: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Lutas com golpes revisados (ou já congeladas) e o estado de cada uma no golden."""
    reviewed = reviewed if reviewed is not None else reviewed_counts()
    rows = []
    lutas = {n for n, e in gabarito.items() if wl.entry_video_type(e) == wl.VIDEO_TYPE_LUTA and not wl.is_entry_ignored(e)}
    for name in sorted(set(golden) | {n for n in lutas if reviewed.get(n)}):
        frozen = golden.get(name)
        if frozen and name in gabarito and gb.differs_from_gabarito(name, gabarito[name]):
            status = "🥇 congelada (gabarito mudou)"
        elif frozen:
            status = "🥇 congelada"
        else:
            status = "disponível"
        source = frozen or gabarito.get(name, {})
        rows.append({"luta": name, "golden": status, "golpes revisados": reviewed.get(name, 0),
                     "pontos dos shinpan": len(source.get("pontos") or []),
                     "congelada em": (frozen or {}).get("congelado_em", "")})
    return pd.DataFrame(rows, columns=["luta", "golden", "golpes revisados", "pontos dos shinpan", "congelada em"])


def _get_pipeline(device_preference: str):
    key = f"golden_panel_pipeline_{device_preference}"
    if key not in st.session_state:
        from src.pipeline import SenpAIPipeline
        st.session_state[key] = SenpAIPipeline(calibration_profile="normal", device_preference=device_preference)
    return st.session_state[key]


def latest_measurement(gb=None) -> Optional[Dict[str, Any]]:
    """Última rodada real do benchmark por luta (ou None): a única fonte de 'quanto o SenpAI acerta'."""
    if gb is None:
        from src.engine.active_learning import GoldenBenchmark
        gb = GoldenBenchmark()
    history = gb.load_history()
    return history[-1] if history else None


def render_golden_benchmark_panel() -> None:
    """Desempenho real do SenpAI: última rodada do benchmark por luta. Sem rodada, nenhum número."""
    from src.engine.active_learning import GoldenBenchmark
    gb = GoldenBenchmark()
    last = latest_measurement(gb)

    if last is None:
        st.info(
            "**Ainda não há medição real.** Os números aparecem depois de congelar lutas com golpes revisados "
            "(por você ou outro sensei, na análise ou na curadoria) e rodar o benchmark por luta (logo abaixo)."
        )
    else:
        t = last["total"]
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Acurácia", _pct(t.get("acuracia")),
                    help="Das decisões válido / não válido sobre os golpes revisados, quantas batem com a revisão.")
        m_c2.metric("Precisão", _pct(t.get("precisao")), help="Dos pontos que o SenpAI deu, quantos a revisão considera válidos.")
        m_c3.metric("Recall", _pct(t.get("recall")), help="Dos golpes que a revisão considera válidos, quantos o SenpAI deu.")
        m_c4.metric("F1", _pct(t.get("f1")))
        if "golpes_revisados" in t:
            shinpan = (f" · shinpan × revisão: {t['shinpan_concordam']} de {t['shinpan_comparados']} batem"
                       if t.get("shinpan_comparados") else "")
            st.caption(f"Medido em {t['lutas']} luta(s) golden, contra {t['golpes_revisados']} golpes revisados "
                       f"(a decisão dos shinpan não é a verdade){shinpan} · rodada de {last['executado_em']}.")
        else:
            st.caption(f"⚠️ Rodada de {last['executado_em']} medida pelo placar dos shinpan (formato antigo). "
                       "Rode o benchmark de novo para medir pelas revisões.")

    src = gb.samples_source()
    st.caption(
        "🛡️ **Trava anti-regressão ativa:** nenhuma recalibração é gravada se piorar o golden. "
        + ("Ela usa os golpes reais das lutas golden." if src["fonte"] == "real"
           else f"Enquanto as lutas golden não somarem {src['minimo_reais']} golpes (hoje: {src['golpes_reais']}), "
                "ela usa amostras sintéticas de referência, que não medem o desempenho do sistema.")
    )


def render_golden_matches_manager(
    device_preference: str = "gpu",
    labels_path: str = wl.DEFAULT_LABELS_PATH,
    is_web_server: bool = False
) -> None:
    """Lutas congeladas e benchmark por luta (largura total, abaixo das colunas do Eixo 4)."""
    from src.engine.active_learning import GoldenBenchmark
    gb = GoldenBenchmark()
    golden = gb.load_matches()

    with st.expander(f"🥇 Lutas do golden & benchmark por luta ({len(golden)} congelada(s))", expanded=False):
        st.caption(
            "Lutas reais com golpes revisados por kenshi ou senseis (na análise ou na curadoria), congeladas como estão. "
            "Nunca entram no treino. A verdade é a revisão (pesada pelo dan); os pontos dos shinpan só apontam o que revisar."
        )
        if is_web_server:
            st.info("Disponível no ambiente Local: o benchmark usa os vídeos e o cache de poses em disco.")
            return

        gabarito = wl.load_match_labels(labels_path)["videos"]
        table = golden_matches_table(gabarito, golden, gb)
        if table.empty:
            st.info("Nenhuma luta com golpes revisados ainda. Revise os golpes de uma luta na análise ou responda as "
                    "dúvidas \"Decisão dos shinpan a revisar\" na curadoria; depois congele a luta aqui.")
            return
        st.dataframe(table, width="stretch", hide_index=True)

        available = [n for n in table["luta"] if n not in golden]
        frozen_names = sorted(golden)
        changed = [n for n in frozen_names if n in gabarito and gb.differs_from_gabarito(n, gabarito[n])]

        c1, c2 = st.columns(2)
        with c1:
            to_freeze = st.multiselect("Congelar no golden", options=available, key="golden_to_freeze")
            if st.button("🥇 Congelar selecionadas", disabled=not to_freeze, key="golden_freeze_btn", width="stretch"):
                for n in to_freeze:
                    gb.freeze_match(n, gabarito[n])
                st.session_state.pop("golden_to_freeze", None)
                st.rerun()
        with c2:
            selected = st.multiselect("Lutas congeladas", options=frozen_names, key="golden_selected_frozen")
            b1, b2 = st.columns(2)
            if b1.button("🔄 Recongelar", disabled=not [n for n in selected if n in changed], key="golden_refreeze_btn", width="stretch",
                         help="Atualiza a cópia congelada com o gabarito atual (só lutas cujo gabarito mudou)."):
                for n in selected:
                    if n in changed:
                        gb.freeze_match(n, gabarito[n])
                st.session_state.pop("golden_selected_frozen", None)
                st.rerun()
            if b2.button("Tirar do golden", disabled=not selected, key="golden_unfreeze_btn", width="stretch"):
                for n in selected:
                    gb.unfreeze_match(n)
                st.session_state.pop("golden_selected_frozen", None)
                st.rerun()
        if changed:
            st.warning("Gabarito alterado depois do congelamento: " + ", ".join(changed) + ". O golden usa a versão congelada até recongelar.")

        if not golden:
            return
        st.markdown("---")
        st.session_state.setdefault("labels_videos_dir", DEFAULT_VIDEOS_DIR)
        videos_dir = st.session_state.get("labels_videos_dir", DEFAULT_VIDEOS_DIR)
        st.caption(f"Pasta dos vídeos: `{videos_dir}` (a mesma da aba Gabarito de Lutas).")
        if st.button("▶️ Rodar benchmark por luta", key="golden_run", type="primary"):
            with st.spinner("Reanalisando as lutas golden (usa o cache de poses)..."):
                st.session_state["golden_result"] = gb.run_matches(_get_pipeline(device_preference), videos_dir)
        res = st.session_state.get("golden_result")
        if res:
            t = res["total"]
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Golpes revisados", t["golpes_revisados"])
            r2.metric("Precisão", _pct(t["precisao"]), help="Dos pontos que o SenpAI deu, quantos a revisão considera válidos.")
            r3.metric("Recall", _pct(t["recall"]), help="Dos golpes válidos pela revisão, quantos o SenpAI deu.")
            r4.metric("F1", _pct(t["f1"]))
            st.caption(f"Detector bruto encontra {_pct(t['recall_candidatos'])} dos golpes válidos pela revisão · "
                       f"{res['golpes_reais']} golpes revisados gravados no golden"
                       + (f" · shinpan × revisão: {t['shinpan_concordam']} de {t['shinpan_comparados']} batem"
                          if t.get("shinpan_comparados") else "") + ".")
            if res.get("sem_revisao"):
                st.info("Sem golpe revisado (não medidas): " + ", ".join(res["sem_revisao"]))
            if res["lutas"]:
                st.dataframe(golden_rows_table(res["lutas"]), width="stretch", hide_index=True)
            for missing in res["videos_ausentes"]:
                st.warning(f"Vídeo não encontrado na pasta: {missing}")
            for fail in res["falhas"]:
                st.error(f"Falha em {fail['video_name']}: {fail['erro']}")

        history = gb.load_history()
        if history:
            st.markdown("**Histórico do benchmark por luta**")
            st.dataframe(pd.DataFrame([{
                "quando": h["executado_em"], "lutas": h["total"]["lutas"],
                "medido por": "revisão" if "golpes_revisados" in h["total"] else "shinpan (antigo)",
                "precisão": _pct(h["total"]["precisao"]), "recall": _pct(h["total"]["recall"]), "F1": _pct(h["total"]["f1"]),
            } for h in reversed(history)]), width="stretch", hide_index=True)
