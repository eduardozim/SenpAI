"""
Tela "Gabarito de Lutas" (Pendência I2 — Rotulagem barata e em escala).

Classifica cada vídeo pelo tipo: luta, uma das modalidades de treino/exame ou descartado. É só isso que a
tela pede (decisão do usuário, 2026-11-23): a validade e a técnica dos golpes vêm das revisões na curadoria,
e as lutas são analisadas sozinhas pelo pré-processamento. Pontos oficiais salvos antes continuam no arquivo
e ainda valem onde existem (técnica tentada e pedidos de revisão), mas não se preenchem mais aqui.
Disponível apenas no ambiente Local: precisa dos vídeos em disco e de um gabarito que persista.

O vídeo original nunca é transcodificado aqui: alterar o arquivo mudaria a impressão digital
do conteúdo e invalidaria o cache de features.
"""

import os
from typing import Any, Dict, List, Optional

import streamlit as st

from src.engine import weak_labels as wl
from src.utils.feature_cache import list_videos

from src.utils.video_library import DEFAULT_VIDEOS_DIR  # pasta única: Dev/data/videos


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
    """Vídeos já classificados pelo tipo: lutas, treinos/exames e descartados."""
    from src.analytics.modality_classifier import is_classified
    out = {"lutas": 0, "treinos": 0, "descartados": 0}
    for e in entries:
        if not is_classified(e):
            continue
        tipo = wl.entry_video_type(e)
        if tipo == wl.VIDEO_TYPE_DISCARD:
            out["descartados"] += 1
        elif tipo == wl.VIDEO_TYPE_LUTA:
            out["lutas"] += 1
        else:
            out["treinos"] += 1
    out["classificados"] = sum(out.values())
    return out


def progress_text(prog: Dict[str, int], total: int) -> str:
    parts = [f"{prog['lutas']} luta(s)", f"{prog['treinos']} treino(s)/exame(s)"]
    if prog["descartados"]:
        parts.append(f"{prog['descartados']} descartado(s)")
    return f"{prog['classificados']} de {total} vídeos classificados — " + " · ".join(parts)


def entry_status(entry: Optional[Dict[str, Any]]) -> str:
    """Rótulo curto do tipo do vídeo (ou "não classificado")."""
    from src.analytics.modality_classifier import is_classified
    if not is_classified(entry):
        return "⬜ não classificado"
    return video_type_label(wl.entry_video_type(entry)).split(" (")[0]


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
            "Classifique cada vídeo pelo tipo. As lutas são analisadas sozinhas e os golpes em dúvida vão para a "
            "Fila de Curadoria, onde a sua revisão (com o dan) decide a validade e a técnica."
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

    with col_form:
        with st.form(key=f"labels_form_{name}"):
            types = video_type_options()
            current = suggestion["tipo"] if suggestion else wl.entry_video_type(entry)
            tipo = st.selectbox(
                "Tipo do vídeo", types, index=types.index(current) if current in types else 0,
                format_func=video_type_label, key=f"labels_tipo_{name}",
                help="Luta / Shiai-geiko: os golpes são analisados e os em dúvida vão para a curadoria. "
                     "Treino ou exame (kakari-geiko, ji-geiko, shinsa…): o vídeo inteiro vira rótulo da modalidade. "
                     "Não aproveitável: o vídeo não ensina nada."
            )
            submitted = st.form_submit_button("💾 Salvar tipo", type="primary", width="stretch")

        if submitted:
            data = wl.load_match_labels(labels_path)
            saved = dict(data["videos"].get(name) or {})   # pontos antigos e demais campos ficam como estão
            saved.update({"origem": "manual", "tipo": tipo, "ignorar": tipo != wl.VIDEO_TYPE_LUTA})
            if str(saved.get("observacoes", "")).startswith("Preencha"):
                saved.pop("observacoes")
            data["videos"][name] = saved
            wl.save_match_labels(data, labels_path)
            st.session_state["labels_flash"] = (
                f"Salvo como {video_type_label(tipo)}."
                + (" O vídeo vira rótulo dessa modalidade." if tipo not in (wl.VIDEO_TYPE_LUTA, wl.VIDEO_TYPE_DISCARD) else "")
                + (" Os golpes dela serão analisados no pré-processamento." if tipo == wl.VIDEO_TYPE_LUTA else ""))
            st.rerun()

        flash = st.session_state.pop("labels_flash", None)
        if flash:
            st.success(flash)
