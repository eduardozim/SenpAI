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

from src.utils.video_library import DEFAULT_VIDEOS_DIR  # pasta única: Dev/data/videos
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
    if wl.is_entry_ignored(entry):
        return "🚫 não é luta"
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


def render_match_labels_page(
    device_preference: str = "gpu",
    labels_path: str = wl.DEFAULT_LABELS_PATH,
    show_header: bool = True
) -> None:
    if show_header:
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

    if str(entry.get("origem", "")).startswith("automatico"):
        st.info("🚩 Os pontos desta luta foram lidos automaticamente das bandeiras dos shinpan. "
                "Ao salvar o formulário, eles passam a valer como gabarito manual.")

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
            ignore = st.checkbox(
                "Não é luta (ignorar)",
                value=wl.is_entry_ignored(entry),
                help="Para gravações de treino, exame etc.: o vídeo fica fora do gabarito, do cache do ciclo e do treino."
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
                    "origem": "manual",
                    "aka": aka_name.strip(),
                    "shiro": shiro_name.strip(),
                    "pontos_completos": bool(complete),
                    "inverter_aka_shiro": bool(invert),
                    "ignorar": bool(ignore),
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
    st.markdown("---")
    render_flag_detector_section(videos_dir, device_preference)


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
                    msgs.append("luta do golden benchmark: rótulos não vão para o feedback" if saved.get("golden")
                                else f"feedback: {saved['TP']} TP e {saved['FP']} FP gravados")
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

    s = report["metricas"]
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


# ------------------------------------------------------------------------------
# Detector de bandeiras dos shinpan (Pendência I7)
# ------------------------------------------------------------------------------

FLAG_OPTIONS = ["vermelha", "branca", "nenhuma", "pular"]
FLAG_LABELS = {"vermelha": "🔴 Vermelha erguida", "branca": "⚪ Branca erguida", "nenhuma": "Nenhuma", "pular": "Pular"}


def render_flag_detector_section(videos_dir: str, device_preference: str = "gpu") -> None:
    from src.vision import flag_detector as fd
    from src.utils import model_registry

    st.markdown("### 🚩 Detector de bandeiras dos shinpan")
    st.caption(
        "Ensine o SenpAI a reconhecer a bandeira erguida: com ele, os pontos de lutas novas são lidos "
        "automaticamente das bandeiras (2 de 3 juízes). Marque cada recorte abaixo."
    )
    stats = fd.dataset_stats()
    active = model_registry.get_active_info(fd.MODEL_FAMILY)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Vermelha", f"{stats['vermelha']}/{fd.MIN_LABELS['vermelha']}")
    c2.metric("⚪ Branca", f"{stats['branca']}/{fd.MIN_LABELS['branca']}")
    c3.metric("Nenhuma", f"{stats['nenhuma']}/{fd.MIN_LABELS['nenhuma']}")
    c4.metric("Sem rótulo", stats["sem_rotulo"])
    st.caption("Modelo ativo: " + (os.path.basename(active["model_path"]) if active else "nenhum"))

    with st.expander("1. Minerar recortes dos vídeos", expanded=stats["total"] == 0):
        st.caption(f"Amostra 1 quadro/s de cada vídeo da pasta `{videos_dir}` e recorta a região acima dos ombros de cada pessoa. "
                   "Leva alguns minutos na GPU; também dá para rodar: `python -m src.vision.flag_detector mine \"<pasta>\"`.")
        if st.button("⛏️ Minerar recortes", key="flag_mine"):
            with st.spinner("Minerando recortes..."):
                added = fd.mine_crops(list_videos(videos_dir), device="cpu" if device_preference == "cpu" else "0")
            st.success(f"{sum(added.values())} recortes novos em {len(added)} vídeos.")
            st.rerun()

    queue = fd.next_to_label(limit=12)
    with st.expander(f"2. Rotular recortes ({stats['sem_rotulo']} na fila)", expanded=bool(queue)):
        if not queue:
            st.info("Nenhum recorte esperando rótulo.")
        else:
            st.caption("Marque a bandeira **erguida** (acima do ombro). Bandeira abaixada, camiseta, papel ou parede = Nenhuma.")
            with st.form("flag_label_form"):
                choices = {}
                for row_start in range(0, len(queue), 4):
                    cols = st.columns(4)
                    for col, item in zip(cols, queue[row_start:row_start + 4]):
                        with col:
                            if os.path.isfile(item["arquivo"]):
                                st.image(item["arquivo"], width="stretch")
                            st.caption(f"{item['video_name'][:28]} · {item['tempo_s']:.0f}s")
                            choices[item["id"]] = st.radio("Rótulo", FLAG_OPTIONS, index=None, key=f"flag_{item['id']}",
                                                           format_func=lambda o: FLAG_LABELS[o], label_visibility="collapsed")
                if st.form_submit_button("💾 Salvar rótulos", type="primary"):
                    n = fd.set_labels({k: v for k, v in choices.items() if v})
                    st.session_state["flag_flash"] = f"{n} recorte(s) rotulado(s)."
                    st.rerun()
        flash = st.session_state.pop("flag_flash", None)
        if flash:
            st.success(flash)

    with st.expander("3. Treinar e medir", expanded=False):
        problem = fd.check_enough_labels(stats)
        if problem:
            st.info(problem)
        if st.button("🧠 Treinar detector", disabled=bool(problem), key="flag_train"):
            with st.spinner("Treinando na GPU (alguns minutos)..."):
                try:
                    st.session_state["flag_train_meta"] = fd.train_flag_classifier(device="cpu" if device_preference == "cpu" else "0")
                except ValueError as e:
                    st.error(str(e))
        meta = st.session_state.get("flag_train_meta")
        if meta:
            v = meta["validacao"]
            st.write(f"Validação em vídeos não vistos ({', '.join(meta['videos_validacao'])}): "
                     f"vermelha F1 {_pct(v['vermelha']['f1'])} · branca F1 {_pct(v['branca']['f1'])} · acurácia {_pct(v['acuracia'])}")
            if st.button("✅ Ativar este modelo", key="flag_activate"):
                model_registry.set_active_model(fd.MODEL_FAMILY, meta["modelo"], "Ativado pela tela do Gabarito.")
                st.rerun()
        if active:
            st.caption("Medir o detector ativo contra os seus gabaritos manuais completos. Os pontos automáticos só podem ser gravados "
                       f"com F1 ≥ {fd.MIN_F1_TO_APPLY:.0%} em pelo menos {fd.MIN_VIDEOS_TO_APPLY} lutas.")
            if st.button("📏 Medir contra o gabarito", key="flag_eval"):
                with st.spinner("Lendo as bandeiras das lutas com gabarito..."):
                    st.session_state["flag_eval"] = fd.evaluate_against_gabarito(videos_dir)
            ev = st.session_state.get("flag_eval")
            if ev:
                t = ev["total"]
                st.write(f"{t['videos']} luta(s): precisão {_pct(t['precisao'])} · recall {_pct(t['recall'])} · F1 {_pct(t['f1'])}")
                ok = t["videos"] >= fd.MIN_VIDEOS_TO_APPLY and (t.get("f1") or 0) >= fd.MIN_F1_TO_APPLY
                if st.button("🚩 Gravar pontos automáticos nas lutas sem gabarito", disabled=not ok, key="flag_apply"):
                    with st.spinner("Lendo as bandeiras das lutas sem gabarito..."):
                        res = fd.apply_to_gabarito(videos_dir, ev)
                    st.success(res.get("motivo") or f"Gabarito automático gravado em {len(res['lutas'])} luta(s).")


def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.0f}%"
