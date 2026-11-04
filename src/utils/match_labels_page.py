"""
Tela "Gabarito de Lutas" (Pendência I2 — Rotulagem barata e em escala).

Permite preencher, direto no app, os pontos oficiais de cada luta (kenshi, técnica e tempo)
que alimentam a rotulagem fraca de `src.engine.weak_labels`, e rotular a luta em seguida
reaproveitando o cache de features (I1). Disponível apenas no ambiente Local: precisa dos
vídeos em disco, do cache e de um arquivo de gabarito que persista entre execuções.

O vídeo original nunca é transcodificado aqui: alterar o arquivo mudaria a impressão digital
do conteúdo e invalidaria o cache de features.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from src.engine import weak_labels as wl
from src.utils.feature_cache import list_videos

DEFAULT_VIDEOS_DIR = os.path.join("..", "QA", "Videos Teste")
KENSHI_OPTIONS = ["AKA", "SHIRO"]
TECHNIQUE_OPTIONS = ["MEN", "KOTE", "DO", "TSUKI", "HANSOKU", "FUSEN", "HANTEI"]
POINT_COLUMNS = ["kenshi", "tecnica", "tempo"]

LABEL_DISPLAY = {
    wl.POSITIVE: "✅ Positivo",
    wl.NEG_OUTSIDE: "⛔ Fora da luta",
    wl.NEG_TSUBAZERIAI: "⛔ Tsubazeriai",
    wl.NEG_NO_POINT: "⛔ Sem ponto oficial",
    wl.AMBIGUOUS: "❓ Ambíguo",
    wl.UNKNOWN: "· Desconhecido",
}


# ------------------------------------------------------------------------------
# Funções puras (testáveis sem Streamlit)
# ------------------------------------------------------------------------------

def entry_status(entry: Optional[Dict[str, Any]]) -> str:
    """Rótulo curto do estado de preenchimento de uma luta."""
    if not entry or not wl.is_entry_filled(entry):
        return "⬜ vazio"
    n = len(entry.get("pontos") or [])
    return f"✅ {n} ponto(s), completo" if entry.get("pontos_completos") else f"🟡 {n} ponto(s), incompleto"


def points_to_rows(entry: Dict[str, Any]) -> pd.DataFrame:
    rows = [{c: str(p.get(c, "") or "") for c in POINT_COLUMNS} for p in (entry.get("pontos") or [])]
    return pd.DataFrame(rows, columns=POINT_COLUMNS)


def rows_to_points(rows: Any) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Converte as linhas da tabela em pontos do gabarito, validando cada uma.
    Linhas totalmente vazias são ignoradas. Retorna (pontos, erros).
    """
    records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows or [])
    points, errors = [], []
    for n, raw in enumerate(records, start=1):
        vals = {c: ("" if raw.get(c) is None or (isinstance(raw.get(c), float) and pd.isna(raw.get(c))) else str(raw.get(c)).strip())
                for c in POINT_COLUMNS}
        if not any(vals.values()):
            continue
        kenshi, technique, tempo = vals["kenshi"].upper(), vals["tecnica"].upper(), vals["tempo"]
        if wl.normalize_kenshi(kenshi) is None:
            errors.append(f"Linha {n}: escolha o kenshi (AKA ou SHIRO).")
        if technique not in TECHNIQUE_OPTIONS:
            errors.append(f"Linha {n}: escolha a técnica.")
        point = {"kenshi": kenshi, "tecnica": technique, "tempo": tempo}
        if technique not in wl.NON_STRIKE_TECHNIQUES:
            try:
                wl.parse_time_to_seconds(tempo)
            except ValueError:
                errors.append(f"Linha {n}: tempo '{tempo}' inválido (use mm:ss, ex.: 01:23).")
        points.append(point)
    points.sort(key=lambda p: _sort_time(p["tempo"]))
    return points, errors


def _sort_time(tempo: str) -> float:
    try:
        return wl.parse_time_to_seconds(tempo)
    except ValueError:
        return float("inf")


def events_table(report: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for ev in report["events"]["items"]:
        rows.append({
            "tempo": wl_format_time(ev["time_sec"]),
            "kenshi": "AKA" if ev.get("attacker_id") == "KENSHI_AKA" else "SHIRO",
            "técnica": ev.get("strike_type"),
            "SenpAI deu ponto": "sim" if ev.get("predicted_valid") else "não",
            "nota": ev.get("total_score"),
            "rótulo": LABEL_DISPLAY.get(ev["weak_label"], ev["weak_label"]),
            "motivo": ev.get("reason"),
        })
    return pd.DataFrame(rows)


def wl_format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# ------------------------------------------------------------------------------
# Tela
# ------------------------------------------------------------------------------

def _get_pipeline(device_preference: str):
    key = f"labels_page_pipeline_{device_preference}"
    if key not in st.session_state:
        from src.pipeline import SenpAIPipeline
        st.session_state[key] = SenpAIPipeline(calibration_profile="normal", device_preference=device_preference)
    return st.session_state[key]


def render_match_labels_page(device_preference: str = "gpu", labels_path: str = wl.DEFAULT_LABELS_PATH) -> None:
    st.markdown("## 🏷️ Gabarito de Lutas")
    st.caption(
        "Informe só os pontos oficiais de cada luta. O SenpAI rotula sozinho todas as detecções "
        "(positivo, negativo ou ambíguo) e mede a precisão real do placar."
    )

    st.session_state.setdefault("labels_videos_dir", DEFAULT_VIDEOS_DIR)
    videos_dir = st.text_input("Pasta dos vídeos", key="labels_videos_dir")
    if not os.path.isdir(videos_dir):
        st.error("Pasta não encontrada.")
        return
    videos = list_videos(videos_dir)
    if not videos:
        st.warning("Nenhum vídeo nesta pasta.")
        return

    data = wl.load_match_labels(labels_path)
    missing = [v for v in videos if os.path.basename(v) not in data["videos"]]
    if missing:
        wl.make_labels_template(missing, labels_path)
        data = wl.load_match_labels(labels_path)

    filled = sum(1 for v in videos if wl.is_entry_filled(data["videos"].get(os.path.basename(v), {})))
    st.progress(filled / len(videos), text=f"{filled} de {len(videos)} lutas com gabarito")

    by_name = {os.path.basename(v): v for v in videos}
    names = list(by_name.keys())
    name = st.selectbox(
        "Luta",
        options=names,
        format_func=lambda n: f"{entry_status(data['videos'].get(n))} — {n}",
        key="labels_selected_video"
    )
    video_path = by_name[name]
    entry = data["videos"].get(name) or {}

    col_video, col_form = st.columns([3, 2])
    with col_video:
        st.video(video_path)
        st.caption("Pause no instante do golpe (não da bandeira) e copie o tempo do player para a tabela.")

    with col_form:
        with st.form(key=f"labels_form_{name}"):
            c1, c2 = st.columns(2)
            aka_name = c1.text_input("Aka (vermelho)", value=entry.get("aka", ""))
            shiro_name = c2.text_input("Shiro (branco)", value=entry.get("shiro", ""))
            edited = st.data_editor(
                points_to_rows(entry),
                num_rows="dynamic",
                width="stretch",
                key=f"labels_points_{name}",
                column_config={
                    "kenshi": st.column_config.SelectboxColumn("Kenshi", options=KENSHI_OPTIONS, required=True),
                    "tecnica": st.column_config.SelectboxColumn("Técnica", options=TECHNIQUE_OPTIONS, required=True),
                    "tempo": st.column_config.TextColumn("Tempo (mm:ss)", help="Instante do golpe no vídeo, ex.: 01:23"),
                },
            )
            complete = st.checkbox(
                "Todos os pontos da luta estão listados",
                value=bool(entry.get("pontos_completos")),
                help="Marque também para luta que terminou 0 x 0 (tabela vazia). Só com isso as demais detecções viram negativos."
            )
            invert = st.checkbox(
                "O SenpAI troca Aka e Shiro nesta luta",
                value=bool(entry.get("inverter_aka_shiro")),
                help="Use se o rastreador inverter as cores; a rotulagem roda com Aka/Shiro trocados."
            )
            notes = st.text_area("Observações", value="" if str(entry.get("observacoes", "")).startswith("Preencha") else entry.get("observacoes", ""), height=68)
            submitted = st.form_submit_button("💾 Salvar gabarito", type="primary", width="stretch")

        if submitted:
            points, errors = rows_to_points(edited)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                data = wl.load_match_labels(labels_path)
                data["videos"][name] = {
                    "aka": aka_name.strip(),
                    "shiro": shiro_name.strip(),
                    "pontos_completos": bool(complete),
                    "inverter_aka_shiro": bool(invert),
                    "pontos": points,
                    "observacoes": notes.strip(),
                }
                wl.save_match_labels(data, labels_path)
                st.session_state["labels_flash"] = f"Gabarito salvo: {len(points)} ponto(s)."
                st.rerun()

        flash = st.session_state.pop("labels_flash", None)
        if flash:
            st.success(flash)

    # Controles do player (câmera lenta, quadro a quadro, ±0,5 s) em largura total: na coluna estreita ficam ilegíveis
    try:
        from src.utils.video_player_controls import render_video_playback_controls
        render_video_playback_controls(container_id="senpai-labels-video-controller", height=170)
    except Exception:
        pass

    st.markdown("---")
    _render_labeling_section(name, video_path, entry, device_preference)


def _render_labeling_section(name: str, video_path: str, entry: Dict[str, Any], device_preference: str) -> None:
    st.markdown("### 🤖 Rotular esta luta")
    if not wl.is_entry_filled(entry):
        st.info("Salve os pontos oficiais (ou marque a luta como completa sem pontos) para rotular.")
        return

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    save_fb = c1.checkbox("Gravar TP/FP no feedback", value=False, key="labels_opt_feedback",
                          help="Positivos viram TP e negativos viram FP no feedback_dataset, com as sub-notas reais.")
    enqueue = c2.checkbox("Enviar ambíguos à curadoria", value=False, key="labels_opt_enqueue")
    run = c3.button("▶️ Rotular agora", type="primary", width="stretch", key="labels_run")

    report_key = f"labels_report_{name}"
    if run:
        with st.spinner("Reanalisando a luta (usa o cache de poses; a 1ª vez processa o vídeo inteiro)..."):
            try:
                pipeline = _get_pipeline(device_preference)
                report = wl.weak_label_video(pipeline, video_path, entry,
                                             invert_combatants=bool(entry.get("inverter_aka_shiro", False)))
                report["report_path"] = wl.save_report(report)
                msgs = []
                if save_fb:
                    saved = wl.export_to_feedback(report)
                    msgs.append(f"feedback: {saved['TP']} TP e {saved['FP']} FP gravados")
                if enqueue:
                    msgs.append(f"{wl.enqueue_ambiguous(report)} ambíguo(s) enviados à curadoria")
                report["actions"] = msgs
                st.session_state[report_key] = report
            except Exception as e:
                st.error(f"Falha ao rotular: {e}")
                return

    report = st.session_state.get(report_key)
    if not report:
        return

    s = report["system"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Placar oficial (Aka x Shiro)", f"{s['placar_oficial']['aka']} x {s['placar_oficial']['shiro']}")
    m2.metric("Placar SenpAI (Aka x Shiro)", f"{s['placar_senpai']['aka']} x {s['placar_senpai']['shiro']}")
    m3.metric("Precisão dos pontos", "—" if s["precisao"] is None else f"{s['precisao'] * 100:.0f}%",
              help="Dos pontos que o SenpAI deu, quantos eram oficiais. Só com gabarito completo.")
    m4.metric("Recall", "—" if s["recall"] is None else f"{s['recall'] * 100:.0f}%",
              help="Dos pontos oficiais, quantos o SenpAI deu.")

    cand = report["candidates"]
    st.caption(
        f"Detector bruto: {len(cand['items'])} candidatos, encontrou "
        f"{cand['counts'].get(wl.POSITIVE, 0)} de {cand['official_strike_points']} pontos oficiais. "
        f"Relatório: {report.get('report_path', '')}"
    )
    for msg in report.get("actions", []):
        st.success(msg)

    df = events_table(report)
    if df.empty:
        st.info("A análise não avaliou nenhum golpe nesta luta.")
    else:
        st.dataframe(df, width="stretch", hide_index=True)

    missed = report["events"]["missed_points"]
    if missed:
        st.warning("Pontos oficiais que o SenpAI não detectou: " + ", ".join(
            f"{'AKA' if p['kenshi_id'] == 'KENSHI_AKA' else 'SHIRO'} {p['technique']} em {wl_format_time(p['time_sec'])}" for p in missed
        ))
