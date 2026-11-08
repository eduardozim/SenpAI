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
            "Informe só os pontos oficiais de cada luta. O SenpAI rotula sozinho todas as detecções "
            "(positivo, negativo ou ambíguo) e mede a precisão real do placar."
        )

    st.session_state.setdefault("labels_videos_dir", DEFAULT_VIDEOS_DIR)
    videos_dir = st.text_input("Pasta dos vídeos", key="labels_videos_dir")
    if not os.path.isdir(videos_dir):
        st.error("Pasta não encontrada.")
        return
    from src.utils.video_library import is_analysis_clip
    all_videos = list_videos(videos_dir)
    # Cortes de 10 s dos golpes em dúvida das análises servem às bandeiras e à curadoria, não ao gabarito
    videos = [v for v in all_videos if not is_analysis_clip(v)]
    if not videos:
        st.warning("Nenhum vídeo nesta pasta.")
        render_flag_detector_section(videos_dir, device_preference)
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

    if str(entry.get("origem", "")).startswith("automatico"):
        st.info("🚩 Os pontos desta luta foram lidos automaticamente das bandeiras dos shinpan. "
                "Ao salvar o formulário, eles passam a valer como gabarito manual.")

    col_video, col_form = st.columns([3, 2])
    with col_video:
        st.video(video_path)
        st.caption("Pause no instante do golpe (não da bandeira) e copie o tempo do player para a tabela.")

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

    st.markdown("---")
    render_flag_detector_section(videos_dir, device_preference)


# ------------------------------------------------------------------------------
# Detector de bandeiras dos shinpan (Pendência I7)
# ------------------------------------------------------------------------------

FLAG_OPTIONS = ["vermelha", "branca", "ambas", "nenhuma", "pular"]
FLAG_LABELS = {"vermelha": "🔴 Vermelha erguida", "branca": "⚪ Branca erguida", "ambas": "✋ Ambas erguidas (parada)",
               "nenhuma": "Nenhuma", "pular": "Pular"}


def _fmt_secs(s: Optional[float]) -> str:
    if s is None:
        return "—"
    s = int(round(s))
    return f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"


def flag_progress_text(info: Dict[str, Any]) -> str:
    """Uma linha com o andamento real do treino (sem nada estimado além do tempo restante)."""
    if info["etapa"] == "preparando":
        tr, va = info["treino"], info["validacao"]
        reps = {c: r for c, r in (info.get("repeticoes") or {}).items() if r > 1}
        return (f"Recortes de treino: {' · '.join(f'{c} {n}' for c, n in tr.items())}"
                + (f" (equilibrados no treino repetindo {', '.join(f'{c} {r}×' for c, r in reps.items())})" if reps else "") + ". "
                f"Validação ({len(info['videos_validacao'])} vídeo(s) separados): {' · '.join(f'{c} {n}' for c, n in va.items())}. "
                "O acerto por época é geral e dominado pela classe mais comum; o F1 de cada cor é medido no fim.")
    if info["etapa"] == "medindo":
        return f"Medindo o modelo final nos {info['recortes']} recortes de validação (vídeos que ele não viu)…"
    acc = info.get("acerto_validacao")
    return (f"Época {info['epoca']}/{info['epocas']} · {_fmt_secs(info['decorrido_s'])} decorridos, ~{_fmt_secs(info['restante_s'])} "
            f"restantes (o treino para antes se a validação não melhorar em {info.get('paciencia') or '—'} épocas) · "
            f"perda treino {info.get('perda_treino') if info.get('perda_treino') is not None else '—'} · "
            f"perda validação {round(info['perda_validacao'], 4) if info.get('perda_validacao') is not None else '—'} · "
            f"acerto validação {'—' if acc is None else f'{acc:.1%}'}")


def _show_flag_progress(info, box, bar, line, table, history) -> None:
    line.caption(flag_progress_text(info))
    if info["etapa"] == "preparando":
        bar.progress(0.02, text="Recortes separados por vídeo (treino × validação)")
    elif info["etapa"] == "treino":
        bar.progress(min(1.0, info["epoca"] / max(1, info["epocas"])), text=f"Época {info['epoca']} de {info['epocas']}")
        box.update(label=f"Treinando o detector de bandeiras — época {info['epoca']}/{info['epocas']}")
        num = lambda v, pct=False: "—" if v is None else (f"{v:.1%}" if pct else f"{v:.4f}")  # noqa: E731
        history.append({"época": info["epoca"], "perda treino": num(info.get("perda_treino")),
                        "perda validação": num(info.get("perda_validacao")),
                        "acerto validação": num(info.get("acerto_validacao"), pct=True)})
        table.dataframe(pd.DataFrame(history), hide_index=True, width="stretch", height=min(300, 38 + 35 * len(history)))
    elif info["etapa"] == "medindo":
        bar.progress(1.0, text="Medindo na validação")
        box.update(label="Medindo o modelo final…")


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
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🔴 Vermelha", f"{stats['vermelha']}/{fd.MIN_LABELS['vermelha']}")
    c2.metric("⚪ Branca", f"{stats['branca']}/{fd.MIN_LABELS['branca']}")
    c3.metric("✋ Parada", f"{stats['ambas']}/{fd.MIN_STOP_LABELS}",
              help="Opcional. As duas bandeiras erguidas = parada (yame): fim da luta, consulta dos shinpan ou "
                   "irregularidade de um kenshi. Nunca vira ponto. Entra no treino ao chegar no mínimo.")
    c4.metric("Nenhuma", f"{stats['nenhuma']}/{fd.MIN_LABELS['nenhuma']}")
    c5.metric("Sem rótulo", stats["sem_rotulo"])
    st.caption("Modelo ativo: " + (os.path.basename(active["model_path"]) if active else "nenhum"))

    todo, skipped = fd.mining_candidates(list_videos(videos_dir))
    with st.expander(f"1. Minerar recortes dos vídeos ({len(todo)} vídeo(s) novo(s))", expanded=bool(todo)):
        st.caption(f"Recorta a região acima dos ombros de cada pessoa nos vídeos da pasta `{videos_dir}` (coleta na internet e "
                   "cortes dos golpes em dúvida das análises): 1 quadro/s nos vídeos longos e 4 quadros/s nos cortes curtos. "
                   f"Só vídeos novos: {skipped['ja_minerados']} já minerado(s) e {skipped['nao_luta']} marcado(s) como treino, "
                   "exame ou descartado (sem bandeiras de shinpan) ficam de fora.")
        if st.button("⛏️ Minerar recortes", key="flag_mine", disabled=not todo):
            with st.status(f"Minerando {len(todo)} vídeo(s)…", expanded=True) as box:
                bar = st.progress(0.0)

                def _mine_progress(i, n, name):
                    bar.progress(i / max(1, n), text=f"Vídeo {i + 1} de {n}: {name}")
                added = fd.mine_crops(todo, device="cpu" if device_preference == "cpu" else "0", progress=_mine_progress)
                bar.progress(1.0, text="Concluído")
                box.update(label=f"✅ {sum(added.values())} recortes novos em {len(added)} vídeo(s)", state="complete")
            st.rerun()

    queue = fd.next_to_label(limit=12)
    with st.expander(f"2. Rotular recortes ({stats['sem_rotulo']} na fila)", expanded=bool(queue)):
        if not queue:
            st.info("Nenhum recorte esperando rótulo.")
        else:
            st.caption("Marque a bandeira **erguida** (acima do ombro). As **duas** erguidas = Ambas (parada: fim da luta, "
                       "consulta dos shinpan ou irregularidade). Bandeira abaixada, camiseta, papel ou parede = Nenhuma.")
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
            device = "cpu" if device_preference == "cpu" else "0"
            with st.status(f"Treinando o detector de bandeiras ({'CPU' if device == 'cpu' else 'GPU'})…", expanded=True) as box:
                bar = st.progress(0.0, text="Preparando os recortes…")
                line = st.empty()
                history: List[Dict[str, Any]] = []
                table = st.empty()
                try:
                    st.session_state["flag_train_meta"] = fd.train_flag_classifier(
                        device=device, progress=lambda info: _show_flag_progress(info, box, bar, line, table, history))
                    box.update(label="✅ Treino concluído", state="complete", expanded=False)
                except (ValueError, FileNotFoundError) as e:
                    box.update(label="❌ O treino falhou", state="error")
                    st.error(str(e))
        meta = st.session_state.get("flag_train_meta")
        if meta:
            v = meta["validacao"]
            st.write(f"Validação em vídeos não vistos ({', '.join(meta['videos_validacao'])}): "
                     f"vermelha F1 {_pct(v['vermelha']['f1'])} · branca F1 {_pct(v['branca']['f1'])}"
                     + (f" · parada F1 {_pct(v['ambas']['f1'])}" if "ambas" in v else "")
                     + f" · acurácia {_pct(v['acuracia'])}")
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
