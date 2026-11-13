"""
Tela "Gabarito de Lutas" (Pendência I2 — Rotulagem barata e em escala).

Permite classificar cada vídeo (luta, treino, exame ou descartado) e preencher os pontos oficiais
das lutas (kenshi, técnica e tempo). A rotulagem das detecções a partir desses pontos é feita
sozinha pelo ciclo de treino e pelo golden (não há mais botão manual aqui). Disponível apenas no
ambiente Local: precisa dos vídeos em disco, do cache e de um gabarito que persista entre execuções.

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


# ------------------------------------------------------------------------------
# Funções puras (testáveis sem Streamlit)
# ------------------------------------------------------------------------------

def video_type_options() -> List[str]:
    """Luta (inclui shiai-geiko), as outras modalidades de treino/exame e "descartar"."""
    from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA
    mods = [k for k in TRAINING_MODALITIES_METADATA if k not in wl.LUTA_ALIASES]
    return [wl.VIDEO_TYPE_LUTA, *mods, wl.VIDEO_TYPE_DISCARD]


def video_type_label(tipo: str) -> str:
    if tipo == wl.VIDEO_TYPE_LUTA:
        return "🥋 Luta / Shiai-geiko"
    if tipo == wl.VIDEO_TYPE_DISCARD:
        return "🚫 Não aproveitável (descartar)"
    from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA
    meta = TRAINING_MODALITIES_METADATA.get(tipo)
    return f"🏋️ {meta['name']}" if meta else tipo


def classification_progress(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    """Vídeos já classificados: lutas com gabarito + treinos/exames com modalidade + descartados."""
    out = {"lutas": 0, "lutas_sem_pontos": 0, "treinos": 0, "descartados": 0}
    for e in entries:
        tipo = wl.entry_video_type(e)
        if tipo == wl.VIDEO_TYPE_DISCARD:
            out["descartados"] += 1
        elif tipo != wl.VIDEO_TYPE_LUTA:
            out["treinos"] += 1
        elif wl.is_entry_filled(e):
            out["lutas"] += 1
        elif (e or {}).get("tipo"):
            out["lutas_sem_pontos"] += 1   # marcada como luta, pontos ainda por preencher
    out["classificados"] = sum(out.values())
    return out


def progress_text(prog: Dict[str, int], total: int) -> str:
    parts = [f"{prog['lutas']} luta(s) com gabarito"]
    if prog["lutas_sem_pontos"]:
        parts.append(f"{prog['lutas_sem_pontos']} luta(s) sem pontos")
    parts.append(f"{prog['treinos']} treino(s)/exame(s)")
    if prog["descartados"]:
        parts.append(f"{prog['descartados']} descartado(s)")
    return f"{prog['classificados']} de {total} vídeos classificados — " + " · ".join(parts)


def entry_status(entry: Optional[Dict[str, Any]]) -> str:
    """Rótulo curto do estado de preenchimento de uma luta."""
    tipo = wl.entry_video_type(entry)
    if tipo == wl.VIDEO_TYPE_DISCARD:
        return "🚫 descartado"
    if tipo != wl.VIDEO_TYPE_LUTA:
        return video_type_label(tipo).split(" (")[0]
    if not entry or not wl.is_entry_filled(entry):
        return "🥋 luta, sem pontos" if (entry or {}).get("tipo") else "⬜ não classificado"
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


def wl_format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


# ------------------------------------------------------------------------------
# Tela
# ------------------------------------------------------------------------------

def render_match_labels_page(
    device_preference: str = "gpu",
    labels_path: str = wl.DEFAULT_LABELS_PATH,
    show_header: bool = True
) -> None:
    if show_header:
        st.markdown("## 🏷️ Gabarito de Lutas")
        st.caption(
            "Informe os pontos oficiais de cada luta. Eles registram a decisão dos shinpan, que também erram: "
            "servem para apontar os golpes a revisar (curadoria), não como rótulo. A validade vem das revisões, pesadas pelo dan."
        )

    st.session_state.setdefault("labels_videos_dir", DEFAULT_VIDEOS_DIR)
    videos_dir = st.text_input("Pasta dos vídeos", key="labels_videos_dir")
    if not os.path.isdir(videos_dir):
        st.error("Pasta não encontrada.")
        return
    from src.utils.video_library import is_analysis_clip
    all_videos = list_videos(videos_dir)
    # Cortes de 10 s dos golpes em dúvida das análises servem à curadoria, não ao gabarito
    videos = [v for v in all_videos if not is_analysis_clip(v)]
    if not videos:
        st.warning("Nenhum vídeo nesta pasta.")
        return

    data = wl.load_match_labels(labels_path)
    missing = [v for v in videos if os.path.basename(v) not in data["videos"]]
    if missing:
        wl.make_labels_template(missing, labels_path)
        data = wl.load_match_labels(labels_path)

    prog = classification_progress([data["videos"].get(os.path.basename(v), {}) for v in videos])
    st.progress(prog["classificados"] / len(videos), text=progress_text(prog, len(videos)))

    from src.analytics.modality_classifier import load_suggestions, is_classified
    suggestions = load_suggestions()

    def _status(n: str) -> str:
        e = data["videos"].get(n)
        sug = suggestions.get(n)
        if sug and not is_classified(e):
            return f"🤖 sugerido: {video_type_label(sug['tipo']).split(' ', 1)[-1].split(' (')[0]} ({sug['probabilidade']:.0%})"
        return entry_status(e)

    by_name = {os.path.basename(v): v for v in videos}
    names = list(by_name.keys())
    name = st.selectbox(
        "Vídeo",
        options=names,
        format_func=lambda n: f"{_status(n)} — {n}",
        key="labels_selected_video"
    )
    video_path = by_name[name]
    entry = data["videos"].get(name) or {}
    suggestion = suggestions.get(name) if not is_classified(entry) else None
    if suggestion:
        conf = suggestion["confianca"]
        st.info(f"🤖 O classificador de modalidade sugere **{video_type_label(suggestion['tipo'])}** "
                f"({suggestion['probabilidade']:.0%}, confiança {conf}"
                + (f"; limiar medido {suggestion['limiar']:.0%}" if suggestion.get("limiar") else "; limiar ainda não medido")
                + "). É só um palpite, que não treina o modelo: confira e salve para confirmar."
                + (" Esta sugestão também está na Fila de Curadoria." if suggestion.get("duvida") else ""))

    col_video, col_form = st.columns([3, 2])
    with col_video:
        st.video(video_path)
        st.caption("Pause no instante do golpe (não o da bandeira) e copie o tempo do player para a tabela.")

    with col_form:
        with st.form(key=f"labels_form_{name}"):
            types = video_type_options()
            current = suggestion["tipo"] if suggestion else wl.entry_video_type(entry)
            tipo = st.selectbox(
                "Tipo do vídeo", types, index=types.index(current) if current in types else 0,
                format_func=video_type_label, key=f"labels_tipo_{name}",
                help="Luta / Shiai-geiko: preencha os pontos abaixo (treina golpes, técnica, calibração e a modalidade luta). "
                     "Treino ou exame (kakari-geiko, ji-geiko, shinsa…): o vídeo inteiro vira rótulo da modalidade e os "
                     "pontos são ignorados. Não aproveitável: o vídeo não ensina nada."
            )
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
            if tipo != wl.VIDEO_TYPE_LUTA:
                errors = []   # treino/exame/descartado: os pontos não são usados
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
                    "tipo": tipo,
                    "ignorar": tipo != wl.VIDEO_TYPE_LUTA,
                    "pontos": points,
                    "observacoes": notes.strip(),
                }
                wl.save_match_labels(data, labels_path)
                st.session_state["labels_flash"] = (
                    f"Gabarito salvo: {len(points)} ponto(s)." if tipo == wl.VIDEO_TYPE_LUTA else
                    f"Salvo como {video_type_label(tipo)}." + (" O vídeo vira rótulo dessa modalidade." if tipo != wl.VIDEO_TYPE_DISCARD else ""))
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
