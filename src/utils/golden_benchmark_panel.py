"""
Painel único do Golden Benchmark na interface (Configurações > Perfis de Calibração > Eixo 4).

Tudo o que é do golden fica aqui: métricas da trava anti-regressão, lutas congeladas
(congelar, recongelar, tirar) e benchmark por luta com histórico. A lógica e as métricas
vêm do GoldenBenchmark (src/engine/active_learning.py) e de src/engine/weak_labels.py.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.engine import weak_labels as wl

DEFAULT_VIDEOS_DIR = os.path.join("..", "QA", "Videos Teste")


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"


def golden_rows_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([{
        "luta": r["video_name"],
        "oficial (A x S)": f"{r['placar_oficial']['aka']} x {r['placar_oficial']['shiro']}",
        "SenpAI (A x S)": f"{r['placar_senpai']['aka']} x {r['placar_senpai']['shiro']}",
        "acertos": r["acertos"],
        "falsos pontos": r["falsos_pontos"],
        "perdidos": r["perdidos"],
    } for r in rows])


def golden_matches_table(gabarito: Dict[str, Dict[str, Any]], golden: Dict[str, Dict[str, Any]], gb) -> pd.DataFrame:
    """Lutas com gabarito completo e o estado de cada uma no golden."""
    rows = []
    for name in sorted(set(golden) | {n for n, e in gabarito.items() if e.get("pontos_completos")}):
        frozen = golden.get(name)
        if frozen and name in gabarito and gb.differs_from_gabarito(name, gabarito[name]):
            status = "🥇 congelada (gabarito mudou)"
        elif frozen:
            status = "🥇 congelada"
        else:
            status = "disponível"
        source = frozen or gabarito.get(name, {})
        rows.append({"luta": name, "golden": status, "pontos": len(source.get("pontos") or []),
                     "congelada em": (frozen or {}).get("congelado_em", "")})
    return pd.DataFrame(rows, columns=["luta", "golden", "pontos", "congelada em"])


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
            "**Ainda não há medição real.** Os números aparecem depois de congelar lutas com gabarito completo "
            "e rodar o benchmark por luta (logo abaixo)."
        )
    else:
        t = last["total"]
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Acurácia", _pct(t.get("acuracia")),
                    help="Das decisões ponto / não-ponto sobre os golpes avaliados, quantas estavam certas.")
        m_c2.metric("Precisão", _pct(t.get("precisao")), help="Dos pontos que o SenpAI deu, quantos eram oficiais.")
        m_c3.metric("Recall", _pct(t.get("recall")), help="Dos pontos oficiais, quantos o SenpAI deu.")
        m_c4.metric("F1", _pct(t.get("f1")))
        st.caption(
            f"Medido em {t['lutas']} luta(s) golden ({t['pontos_oficiais']} pontos oficiais, "
            f"{t.get('decisoes_avaliadas', 0)} golpes avaliados) · placares exatos: {t['placares_exatos']} de {t['lutas']} · "
            f"rodada de {last['executado_em']}."
        )

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
            "Lutas reais com gabarito completo (aba Gabarito de Lutas), congeladas como estão no momento. "
            "Nunca entram no treino nem na curadoria. Congele as lutas do golden **antes** de gravar o feedback das demais."
        )
        if is_web_server:
            st.info("Disponível no ambiente Local: o benchmark usa os vídeos e o cache de poses em disco.")
            return

        gabarito = wl.load_match_labels(labels_path)["videos"]
        table = golden_matches_table(gabarito, golden, gb)
        if table.empty:
            st.info("Nenhuma luta com gabarito completo ainda. Preencha na aba Gabarito de Lutas e marque \"todos os pontos listados\".")
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
            r1.metric("Placares exatos", f"{t['placares_exatos']} de {t['lutas']}")
            r2.metric("Precisão dos pontos", _pct(t["precisao"]), help="Dos pontos que o SenpAI deu, quantos eram oficiais.")
            r3.metric("Recall dos pontos", _pct(t["recall"]), help="Dos pontos oficiais, quantos o SenpAI deu.")
            r4.metric("F1 dos pontos", _pct(t["f1"]))
            st.caption(f"Detector bruto encontra {_pct(t['recall_candidatos'])} dos pontos oficiais · {res['golpes_reais']} golpes reais gravados no golden.")
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
                "quando": h["executado_em"], "lutas": h["total"]["lutas"], "placares exatos": h["total"]["placares_exatos"],
                "precisão": _pct(h["total"]["precisao"]), "recall": _pct(h["total"]["recall"]), "F1": _pct(h["total"]["f1"]),
            } for h in reversed(history)]), width="stretch", hide_index=True)
