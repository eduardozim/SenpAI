"""
Painel da Fila de Curadoria Ativa (dúvidas do SenpAI), em Configurações > Governança de Treinamento.
Cada resposta vira treino: recalibra o perfil (protegido pelo Golden Benchmark) e apaga o clipe do golpe.
"""

import os
import re

import streamlit as st

from src.utils.logger_manager import log_event
from src.engine.weak_labels import normalize_technique


TECHNIQUES = ["MEN", "KOTE", "DO", "TSUKI"]


def _delete_item_clip(unc_item, it_id: str) -> None:
    """Apaga o corte de vídeo do golpe em disco."""
    try:
        from src.utils.strike_clip_manager import delete_strike_clip
        c_path = unc_item.get("clip_path")
        if c_path:
            delete_strike_clip(c_path)
        target_c = os.path.join("data", "active_learning_clips", f"{it_id}_corte.mp4")
        if os.path.isfile(target_c):
            delete_strike_clip(target_c)
    except Exception as err:
        log_event("WARNING", f"Erro ao excluir corte de vídeo pós-curadoria: {err}", "app")


def _resolve_strike_decision(auto_trainer, feedback_mgr, unc_item, it_id: str, cur_dan, decision: str,
                             approved: bool, notes: str, technique) -> None:
    """Registra a decisão (IPPON / INVALID_STRIKE / NO_STRIKE) com o golpe escolhido, retreina e apaga o clipe."""
    p_key = unc_item.get("profile_name", "normal")
    kwargs = dict(trigger_retrain=True, cleanup_clip=True, decision=decision)
    if technique:
        kwargs["technique"] = technique
    # 1. Resolver na fila / feedback (com o golpe escolhido)
    try:
        if hasattr(auto_trainer, "resolve_active_learning_item"):
            auto_trainer.resolve_active_learning_item(it_id, approved, cur_dan, notes, **kwargs)
        elif hasattr(feedback_mgr, "resolve_active_learning_item"):
            feedback_mgr.resolve_active_learning_item(it_id, approved, cur_dan, notes, **kwargs)
    except Exception as err:
        log_event("WARNING", f"Erro ao resolver item na curadoria: {err}", "app")
    # 2. Retreinamento e recalibração formal do modelo
    try:
        if hasattr(auto_trainer, "run_mathematical_optimization"):
            auto_trainer.run_mathematical_optimization(p_key)
        elif hasattr(feedback_mgr, "optimize_profile_config"):
            profs = feedback_mgr.load_profiles()
            n_cfg, _ = feedback_mgr.optimize_profile_config(p_key, profs.get(p_key, {}))
            profs[p_key] = n_cfg
            feedback_mgr.save_profiles(profs)
    except Exception as err:
        log_event("WARNING", f"Erro no retreinamento explícito pós-curadoria: {err}", "app")
    # 3. Apagar o corte do golpe em disco
    _delete_item_clip(unc_item, it_id)


def render_curation_queue_panel(auto_trainer, feedback_mgr, title: str = "#### 📥 Fila de Curadoria") -> None:
    """Mostra as dúvidas pendentes e registra a decisão do árbitro para cada uma."""
    if title:
        st.markdown("---")
        st.markdown(title)
        st.caption("Dúvidas que o SenpAI mandou para revisão humana (pontuação, golpe, técnica, modalidade, regra ou conflito entre fontes). "
                   "Responda aqui: cada decisão entra no próximo treino.")
    unc_queue = []
    try:
        fm = getattr(auto_trainer, "feedback_mgr", None)
        if hasattr(auto_trainer, "get_active_learning_queue"):
            unc_queue = auto_trainer.get_active_learning_queue(status="pending_curation")
        elif fm is not None and hasattr(fm, "get_active_learning_queue"):
            unc_queue = fm.get_active_learning_queue(status="pending_curation")
        elif hasattr(feedback_mgr, "get_active_learning_queue"):
            unc_queue = feedback_mgr.get_active_learning_queue(status="pending_curation")
    except Exception:
        unc_queue = []

    if "curation_toast" in st.session_state:
        st.toast(st.session_state.pop("curation_toast"), icon="🥋")

    with st.expander(f"📥 Fila de Curadoria Ativa (dúvidas do SenpAI) — {len(unc_queue)} pendentes", expanded=bool(unc_queue)):
        if "curation_alert" in st.session_state:
            st.success(st.session_state.pop("curation_alert"))

        if not unc_queue:
            st.info("Nenhuma dúvida pendente: pontuação, golpe, técnica, regra ou conflito entre fontes.")
        else:
            st.caption(
                "🥋 **Como revisar:** Assista ao trecho do golpe controverso abaixo, avalie a biomecânica de Ki-Ken-Tai-Ichi "
                "e registre a decisão arbitral. Sua decisão **vira treino**: o SenpAI recalibra o perfil (a gravação só ocorre se não piorar o Golden Benchmark) "
                "e **apaga o clipe** do disco. Esta fila não mede o desempenho do sistema; quem mede é o Golden Benchmark (Perfis de Calibração > Eixo 4)."
            )
            # Graduação de quem decide: o peso da decisão no treino é o dan (shinpan credenciado 4,5; kyu 0,8)
            _cur_dan_opts = {1: "1º Dan", 2: "2º Dan", 3: "3º Dan", 4: "4º Dan", 5: "5º Dan", 6: "6º Dan", 7: "7º Dan",
                             8: "8º Dan", "shinpan": "Shinpan credenciado"}
            _cur_dan_keys = list(_cur_dan_opts)
            cur_dan = st.selectbox(
                "🥋 Sua graduação (pesa a decisão no treino: quanto maior o dan, mais confiável; shinpan credenciado vale 4,5)",
                _cur_dan_keys, index=_cur_dan_keys.index(st.session_state.get("curation_reviewer_dan", 3))
                if st.session_state.get("curation_reviewer_dan", 3) in _cur_dan_keys else 2,
                format_func=lambda k: _cur_dan_opts[k], key="curation_reviewer_dan",
            )
            from src.engine.active_learning import DOUBT_LABELS
            tipos_pendentes = {}
            for _it in unc_queue:
                _t = _it.get("tipo_duvida", "pontuacao")
                tipos_pendentes[_t] = tipos_pendentes.get(_t, 0) + 1
            st.caption("Pendentes por tipo: " + " · ".join(f"{DOUBT_LABELS.get(t, t)} {n}" for t, n in sorted(tipos_pendentes.items(), key=lambda kv: -kv[1]))
                       + ". As que mais ensinam aparecem primeiro.")
            for idx, unc_item in enumerate(unc_queue[:5]):
                # 1. Obter metadados de origem e timecode do golpe
                try:
                    import src.utils.strike_clip_manager as scm_mod
                    if not hasattr(scm_mod, "parse_strike_timestamp") or not hasattr(scm_mod, "get_cross_platform_basename"):
                        import importlib
                        scm_mod = importlib.reload(scm_mod)
                    parse_strike_timestamp = getattr(scm_mod, "parse_strike_timestamp")
                    format_strike_timestamp = getattr(scm_mod, "format_strike_timestamp")
                    get_cross_platform_basename = getattr(scm_mod, "get_cross_platform_basename")
                    resolve_source_video_path = getattr(scm_mod, "resolve_source_video_path")
                except Exception:
                    def get_cross_platform_basename(p):
                        return os.path.basename(str(p).replace("\\", "/")) if p else ""
                    def parse_strike_timestamp(item):
                        sd = item.get("strike_data") or {}
                        return float(sd.get("timestamp_seconds") or item.get("timestamp_seconds") or 0.0)
                    def format_strike_timestamp(sec):
                        m = int(sec // 60)
                        s = sec % 60
                        return f"{m:02d}:{s:06.3f} ({sec:.1f}s)"
                    def resolve_source_video_path(p):
                        return get_cross_platform_basename(p)

                strike_t = str(unc_item.get("strike_type", "Men")).upper()
                raw_src = str(unc_item.get("video_source", "")).strip() or "Vídeo de Combate"
                raw_base = get_cross_platform_basename(raw_src)
                clean_src_name = re.sub(r"^(upload_\d+_|yt_[^_]+_)", "", raw_base) if raw_base else raw_src
                resolved_src_path = resolve_source_video_path(raw_src) if callable(resolve_source_video_path) else raw_src
                t_sec = parse_strike_timestamp(unc_item)
                formatted_time = format_strike_timestamp(t_sec)

                # 2. Assegurar clipe de vídeo para visualização do Benchmark
                clip_path = None
                try:
                    if hasattr(auto_trainer, "ensure_active_learning_clip"):
                        clip_path = auto_trainer.ensure_active_learning_clip(unc_item)
                    elif hasattr(feedback_mgr, "ensure_active_learning_clip"):
                        clip_path = feedback_mgr.ensure_active_learning_clip(unc_item)
                    else:
                        from src.utils.strike_clip_manager import get_or_create_strike_clip
                        clip_path = get_or_create_strike_clip(unc_item)
                except Exception as e:
                    log_event("WARNING", f"Erro ao preparar clipe de demonstração do item {unc_item.get('id')}: {e}", "app")

                # Fallback de localização direta caso o caminho no item venha de outro ambiente / servidor
                if not (clip_path and os.path.isfile(clip_path)):
                    direct_clip = unc_item.get("clip_path")
                    if direct_clip:
                        if os.path.isfile(direct_clip):
                            clip_path = direct_clip
                        else:
                            c_base = get_cross_platform_basename(direct_clip)
                            cand_direct = os.path.join("data", "active_learning_clips", c_base)
                            if os.path.isfile(cand_direct):
                                clip_path = cand_direct

                c_vid, c_info = st.columns([1, 1])

                with c_vid:
                    st.markdown('<div class="video-preview-marker"></div>', unsafe_allow_html=True)
                    if clip_path and os.path.isfile(clip_path):
                        st.video(clip_path, format="video/mp4")
                        st.caption(f"🎥 **Corte Real do Golpe ({strike_t}) aos {formatted_time}** de `{clean_src_name}`")
                    else:
                        st.warning(
                            f"⚠️ **Corte de vídeo indisponível:** O arquivo da luta original **`{clean_src_name}`** "
                            f"(esperado em `{resolved_src_path}`) não foi localizado no disco do servidor para extração do trecho aos `{formatted_time}`."
                        )

                with c_info:
                    st.markdown('<div class="video-side-panel-marker"></div>', unsafe_allow_html=True)
                    conf_pct = float(unc_item.get("confidence") or 0.0) * 100
                    unc_pct = float(unc_item.get("uncertainty_score", 0.0)) * 100
                    _tipo = unc_item.get("tipo_duvida", "pontuacao")
                    if _tipo == "modalidade":
                        st.markdown(f"#### 🏋️ Sugestão do modelo: `{unc_item.get('strike_type', '—')}`")
                    else:
                        st.markdown(f"#### 🥋 Golpe: `{strike_t}`")
                    st.markdown(f"- ❓ **Dúvida:** `{DOUBT_LABELS.get(_tipo, _tipo)}` &nbsp;|&nbsp; 📈 **Ordem na fila:** `{idx + 1}º`")
                    if unc_item.get("motivo"):
                        st.caption(unc_item["motivo"])
                    if resolved_src_path and resolved_src_path != clean_src_name:
                        st.markdown(f"- 📁 **Vídeo de Origem:** `{clean_src_name}` &nbsp;(`{resolved_src_path}`)")
                    else:
                        st.markdown(f"- 📁 **Vídeo de Origem:** `{clean_src_name}`")
                    st.markdown(f"- ⏱️ **Momento na Luta:** `{formatted_time}`")
                    if _tipo == "pontuacao" and unc_item.get("confidence") is not None:
                        st.markdown(f"- 🤖 **Nota do golpe:** `{conf_pct:.1f}%` &nbsp;|&nbsp; ⚖️ **Incerteza Arbitral:** `{unc_pct:.1f}%`")
                    elif unc_item.get("confidence") is not None:
                        st.markdown(f"- 🤖 **Confiança do modelo:** `{conf_pct:.1f}%`")

                    # Métricas dos 4 Pilares
                    s_data = unc_item.get("strike_data") or {}
                    sub_scores = s_data.get("sub_scores") or s_data.get("scores") or {}
                    if sub_scores:
                        def _pct_score(key: str) -> str:
                            if key not in sub_scores:
                                return "—"
                            v = float(sub_scores[key])
                            return f"{(v if v > 1.0 else v * 100):.0f}%"
                        st.caption(
                            f"🎯 Alvo: **{_pct_score('target_impact')}** | 🦶 Fumikomi: **{_pct_score('fumikomi_sync')}** | "
                            f"🥋 Postura: **{_pct_score('posture')}** | ⚡ Zanshin: **{_pct_score('zanshin')}**"
                            + (f" | 🗡️ Hasuji: **{_pct_score('hasuji')}**" if "hasuji" in sub_scores else "")
                        )
                    elif _tipo == "pontuacao":
                        st.caption("⚠️ Lance sem as notas do golpe (gerado antes da correção P6): a decisão grava o rótulo, mas sem as notas.")

                    triage = unc_item.get("llm_triage") or {}
                    if triage:
                        st.markdown(f"- 🏛️ **Parecer FIK/LLM:** `{triage.get('verdict', 'DOUBTFUL')}` ({triage.get('fik_rule_rationale', '')})")
                        st.markdown(f"- 💡 **Orientação Técnica:** {triage.get('coaching_advice', '')}")

                    it_id = str(unc_item.get("id", f"item_{idx}"))
                    if _tipo == "modalidade":
                        # Dúvida de modalidade: a resposta vira o "Tipo do vídeo" no Gabarito de Lutas
                        from src.utils.match_labels_page import video_type_options, video_type_label
                        _opts = unc_item.get("opcoes") or {}
                        _types = video_type_options()
                        _sug = _opts.get("sugestao")
                        st.markdown("**Qual é o tipo deste vídeo?**")
                        _choice = st.selectbox("Tipo do vídeo", _types, index=_types.index(_sug) if _sug in _types else 0,
                                               format_func=video_type_label, key=f"sel_cur_mod_{it_id}",
                                               label_visibility="collapsed")
                        _mc1, _mc2 = st.columns(2)
                        with _mc1:
                            if st.button("✅ Confirmar tipo", key=f"btn_cur_mod_{it_id}", type="primary", width="stretch"):
                                try:
                                    auto_trainer.feedback_mgr.resolve_modality_doubt(it_id, _choice, reviewer_dan=cur_dan)
                                    st.session_state["curation_alert"] = (f"✅ Vídeo classificado como {video_type_label(_choice)} no Gabarito: "
                                                                          "o classificador de modalidade aprende no próximo ciclo.")
                                except Exception as err:
                                    st.session_state["curation_alert"] = f"⚠️ Não foi possível registrar o tipo: {err}"
                                st.rerun()
                        with _mc2:
                            if st.button("🗑️ Descartar dúvida", key=f"btn_cur_mod_discard_{it_id}", width="stretch"):
                                auto_trainer.discard_active_learning_item(it_id, notes="Descartado: modalidade indeterminada")
                                st.session_state["curation_alert"] = "🗑️ Dúvida de modalidade descartada (sem afetar o treinamento)."
                                st.rerun()
                        st.markdown("---")
                        continue
                    if _tipo == "tecnica":
                        # Dúvida de técnica: a decisão vai para o Banco Único de Rótulos (treina o classificador de técnica)
                        _probs = unc_item.get("opcoes") or {}
                        if _probs:
                            st.caption("Probabilidades do modelo: " + " · ".join(f"{k} {float(v) * 100:.0f}%" for k, v in sorted(_probs.items(), key=lambda kv: -float(kv[1]))))
                        st.markdown("**Qual é a técnica?**")
                        _tcols = st.columns(5)
                        for _col, _tech in zip(_tcols[:4], ("MEN", "KOTE", "DO", "TSUKI")):
                            with _col:
                                if st.button(_tech, key=f"btn_cur_tech_{_tech}_{it_id}", width="stretch"):
                                    try:
                                        auto_trainer.feedback_mgr.resolve_technique_doubt(it_id, _tech, reviewer_dan=cur_dan)
                                        st.session_state["curation_alert"] = f"✅ Técnica registrada como {_tech}: o classificador de técnica aprende no próximo ciclo de treino."
                                    except Exception as err:
                                        st.session_state["curation_alert"] = f"⚠️ Não foi possível registrar a técnica: {err}"
                                    st.rerun()
                        with _tcols[4]:
                            if st.button("🗑️ Descartar", key=f"btn_cur_tech_discard_{it_id}", width="stretch"):
                                auto_trainer.discard_active_learning_item(it_id, notes="Descartado: técnica indeterminada")
                                st.session_state["curation_alert"] = "🗑️ Dúvida de técnica descartada (sem afetar o treinamento)."
                                st.rerun()
                        st.markdown("---")
                        continue

                    st.markdown("**Decisão do Árbitro / Retorno:**")
                    st.caption(
                        "• **✅ Ippon (Válido):** Há movimento e é válido para pontuação.\n"
                        "• **❌ Golpe Inválido:** Há movimento/tentativa, mas o golpe não foi válido.\n"
                        "• **🚫 Não tem Golpe:** Nenhuma tentativa de movimento para pontuação (detecção espúria).\n"
                        "• **🗑️ Descartar:** Não foi possível determinar o resultado do golpe (descarta da fila sem afetar o treinamento)."
                    )
                    it_id = str(unc_item.get("id", f"item_{idx}"))

                    # Golpe que houve, pré-marcado com o identificado pelo sistema (vale para Ippon e Golpe Inválido)
                    detected_tech = normalize_technique(unc_item.get("strike_type"))
                    chosen_tech = st.radio(
                        "Qual golpe foi?", TECHNIQUES,
                        index=TECHNIQUES.index(detected_tech) if detected_tech in TECHNIQUES else 0,
                        horizontal=True, key=f"sel_cur_strike_{it_id}",
                        help="Pré-marcado com o golpe identificado pelo SenpAI. Se a tentativa foi de outro golpe, "
                             "troque aqui antes de decidir Ippon ou Golpe Inválido.",
                    )
                    if chosen_tech != detected_tech:
                        st.caption(f"✏️ Golpe corrigido de **{detected_tech or strike_t}** para **{chosen_tech}**: "
                                   "a correção também ensina o classificador de técnica, e as notas dos pilares "
                                   "(medidas para o golpe errado) ficam fora da calibração.")
                    fix_txt = f" (corrigido de {detected_tech or strike_t})" if chosen_tech != detected_tech else ""

                    btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
                    with btn_c1:
                        if st.button("✅ Ippon (Válido)", key=f"btn_cur_ok_{it_id}", width="stretch", type="primary", help="Há movimento e é válido para pontuação (Ippon legítimo)."):
                            _resolve_strike_decision(auto_trainer, feedback_mgr, unc_item, it_id, cur_dan, "IPPON", True,
                                                     "Homologado como Ippon via Benchmark Arbitral", chosen_tech)
                            st.session_state["curation_alert"] = f"✅ Golpe {chosen_tech}{fix_txt} homologado como Ippon! Modelo retreinado com sucesso e item removido da Fila de Curadoria."
                            st.session_state["curation_toast"] = f"✅ Golpe {chosen_tech} homologado: Modelo retreinado!"
                            st.rerun()

                    with btn_c2:
                        if st.button("❌ Golpe Inválido", key=f"btn_cur_inv_{it_id}", width="stretch", help="Há movimento/tentativa de ataque, mas o golpe NÃO atendeu aos critérios para pontuação."):
                            _resolve_strike_decision(auto_trainer, feedback_mgr, unc_item, it_id, cur_dan, "INVALID_STRIKE", False,
                                                     "Golpe Inválido (Tentativa de movimento sem pontuação) via Benchmark Arbitral", chosen_tech)
                            st.session_state["curation_alert"] = f"❌ Golpe {chosen_tech}{fix_txt} classificado como Golpe Inválido. Modelo retreinado com sucesso e item removido da Fila de Curadoria."
                            st.session_state["curation_toast"] = f"❌ Golpe {chosen_tech} invalidado: Modelo retreinado!"
                            st.rerun()

                    with btn_c3:
                        if st.button("🚫 Não tem Golpe", key=f"btn_cur_none_{it_id}", width="stretch", help="Não existe nenhuma tentativa de movimento para pontuação (detecção espúria)."):
                            _resolve_strike_decision(auto_trainer, feedback_mgr, unc_item, it_id, cur_dan, "NO_STRIKE", False,
                                                     "Não tem golpe (Nenhuma tentativa de movimento para pontuação) via Benchmark Arbitral", None)
                            st.session_state["curation_alert"] = "🚫 Lance classificado como 'Não tem Golpe' (Detecção Espúria). Modelo retreinado para supressão de falsos disparos e item removido da Fila de Curadoria."
                            st.session_state["curation_toast"] = "🚫 Detecção espúria suprimida: Modelo retreinado!"
                            st.rerun()

                    with btn_c4:
                        if st.button("🗑️ Descartar", key=f"btn_cur_discard_{it_id}", width="stretch", help="Não foi possível determinar o resultado do golpe. O item será descartado sem afetar o treinamento."):
                            c_path = unc_item.get("clip_path")

                            # 1. Descartar da fila sem registrar feedback e sem disparar retreinamento
                            try:
                                if hasattr(auto_trainer, "discard_active_learning_item"):
                                    auto_trainer.discard_active_learning_item(it_id, notes="Descartado: Resultado indeterminado")
                                elif hasattr(auto_trainer, "resolve_active_learning_item"):
                                    auto_trainer.resolve_active_learning_item(
                                        it_id, False, cur_dan, "Descartado: Resultado indeterminado",
                                        trigger_retrain=False, cleanup_clip=True, decision="DISCARDED"
                                    )
                                elif hasattr(feedback_mgr, "discard_active_learning_item"):
                                    feedback_mgr.discard_active_learning_item(it_id, notes="Descartado: Resultado indeterminado")
                                elif hasattr(feedback_mgr, "resolve_active_learning_item"):
                                    feedback_mgr.resolve_active_learning_item(
                                        it_id, False, cur_dan, "Descartado: Resultado indeterminado",
                                        trigger_retrain=False, cleanup_clip=True, decision="DISCARDED"
                                    )
                            except Exception as err:
                                log_event("WARNING", f"Erro ao descartar item da curadoria: {err}", "app")

                            # 2. Apagar arquivo de vídeo do golpe em disco
                            try:
                                from src.utils.strike_clip_manager import delete_strike_clip
                                if c_path:
                                    delete_strike_clip(c_path)
                                target_c = os.path.join("data", "active_learning_clips", f"{it_id}_corte.mp4")
                                if os.path.isfile(target_c):
                                    delete_strike_clip(target_c)
                            except Exception as err:
                                log_event("WARNING", f"Erro ao excluir corte de vídeo pós-descarte: {err}", "app")

                            st.session_state["curation_alert"] = f"🗑️ Item descartado da Fila de Curadoria com sucesso (sem afetar o treinamento)."
                            st.session_state["curation_toast"] = f"🗑️ Item removido da Fila de Curadoria!"
                            st.rerun()

                st.markdown("---")
