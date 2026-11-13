"""
Painel do Catálogo de Exercícios (Configurações > Catálogo de Exercícios).

O catálogo (config/exercise_catalog.json) é de onde o diagnóstico do modo de treinamento escolhe os exercícios
para cada ponto de atenção e para o ponto mais forte (src/analytics/training_feedback.py). Aqui um sensei
consulta, filtra, edita, cria e exclui exercícios, vê quais métricas estão sem exercício e baixa ou carrega
o catálogo inteiro. Toda gravação é validada antes.
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.analytics import training_feedback as tf

LEVEL_LABEL = {"todos": "Todos os níveis", "iniciante": "Iniciante", "intermediario": "Intermediário",
               "avancado": "Avançado"}
FORMAT_LABEL = {"solo": "Solo", "dupla": "Em dupla", "bogu": "Com bogu (motodachi)"}
SEVERITY_TITLE = {"leve": "Leve", "moderada": "Moderada", "prioritaria": "Prioritária"}


def _metric_label(key: str) -> str:
    return tf.METRICS.get(key, {}).get("label", key)


def _modality_options() -> Dict[str, str]:
    try:
        from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA
        return {k: v.get("name", k) for k, v in TRAINING_MODALITIES_METADATA.items()}
    except Exception:
        return {}


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]


def _save(exercises: List[Dict[str, Any]], message: str) -> bool:
    try:
        tf.save_exercise_catalog(exercises)
    except Exception as err:
        st.error(f"Não foi possível gravar: {err}")
        return False
    st.session_state["exercise_catalog_alert"] = message
    return True


def _exercise_form(ex: Optional[Dict[str, Any]], form_key: str, existing_ids: List[str]) -> Optional[Dict[str, Any]]:
    """Formulário de criação/edição. Devolve o exercício validado quando o usuário grava."""
    ex = ex or {}
    is_new = not ex
    mods = _modality_options()
    with st.form(form_key, clear_on_submit=False):
        nome = st.text_input("Nome do exercício", value=ex.get("nome", ""))
        c1, c2 = st.columns(2)
        with c1:
            formato = st.selectbox("Formato", list(tf.FORMATS), index=list(tf.FORMATS).index(ex.get("formato", "solo")),
                                   format_func=lambda f: FORMAT_LABEL.get(f, f),
                                   help="Modalidades feitas sozinho só recebem exercícios solo.")
        with c2:
            nivel = st.selectbox("Nível", list(tf.LEVELS), index=list(tf.LEVELS).index(ex.get("nivel", "todos")),
                                 format_func=lambda n: LEVEL_LABEL.get(n, n))
        corrige = st.multiselect("Corrige (métricas fracas que este exercício trabalha)", list(tf.METRICS),
                                 default=[m for m in ex.get("corrige", []) if m in tf.METRICS], format_func=_metric_label)
        reforca = st.multiselect("Reforça (métricas fortes que este exercício leva adiante)", list(tf.METRICS),
                                 default=[m for m in ex.get("reforca", []) if m in tf.METRICS], format_func=_metric_label)
        modalidades = st.multiselect("Só para estas modalidades (vazio = qualquer modalidade)", list(mods),
                                     default=[m for m in ex.get("modalidades", []) if m in mods],
                                     format_func=lambda m: mods.get(m, m))
        manutencao = st.checkbox("Exercício de manutenção (recomendado quando não há nenhum ponto de atenção)",
                                 value=bool(ex.get("manutencao")))
        st.markdown("**Dose conforme a gravidade do ponto de atenção**")
        dose = {}
        dcols = st.columns(3)
        for col, sev in zip(dcols, tf.SEVERITIES):
            with col:
                dose[sev] = st.text_input(SEVERITY_TITLE[sev], value=(ex.get("dose") or {}).get(sev, ""),
                                          key=f"{form_key}_dose_{sev}")
        passos = st.text_area("Progressão (um passo por linha)", value="\n".join(ex.get("passos", [])))
        atencao = st.text_area("Pontos de atenção (um por linha)", value="\n".join(ex.get("pontos_de_atencao", [])))
        erros = st.text_area("Erros comuns (um por linha)", value="\n".join(ex.get("erros_comuns", [])))
        como_medir = st.text_input("Como medir a evolução", value=ex.get("como_medir", ""))
        submitted = st.form_submit_button("💾 Criar exercício" if is_new else "💾 Gravar alterações", type="primary")
    if not submitted:
        return None
    new_id = ex.get("id") or tf.slugify(nome)
    if is_new and new_id in existing_ids:
        base, n = new_id, 2
        while f"{base}_{n}" in existing_ids:
            n += 1
        new_id = f"{base}_{n}"
    out = {"id": new_id, "nome": nome.strip(), "corrige": corrige, "reforca": reforca, "formato": formato,
           "nivel": nivel, "dose": {k: v.strip() for k, v in dose.items()}, "passos": _lines(passos),
           "pontos_de_atencao": _lines(atencao), "erros_comuns": _lines(erros), "como_medir": como_medir.strip()}
    if modalidades:
        out["modalidades"] = modalidades
    if manutencao:
        out["manutencao"] = True
    errs = tf.validate_exercise(out)
    if errs:
        for e in errs:
            st.error(e)
        return None
    return out


def _render_exercise(ex: Dict[str, Any], exercises: List[Dict[str, Any]], idx: int) -> None:
    ex_id = ex.get("id", f"ex_{idx}")
    title = f"🥋 {ex.get('nome', ex_id)} · {FORMAT_LABEL.get(ex.get('formato'), ex.get('formato'))} · {LEVEL_LABEL.get(ex.get('nivel', 'todos'))}"
    editing = st.session_state.get("exercise_catalog_editing") == ex_id
    with st.expander(title, expanded=editing):
        if editing:
            new_ex = _exercise_form(ex, f"form_edit_{ex_id}", [e.get("id") for e in exercises])
            if new_ex is not None:
                updated = [new_ex if e.get("id") == ex_id else e for e in exercises]
                if _save(updated, f"✅ Exercício '{new_ex['nome']}' atualizado."):
                    st.session_state.pop("exercise_catalog_editing", None)
                    st.rerun()
            if st.button("Cancelar edição", key=f"btn_cancel_{ex_id}"):
                st.session_state.pop("exercise_catalog_editing", None)
                st.rerun()
            return

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Corrige:** " + (", ".join(_metric_label(m) for m in ex.get("corrige", [])) or "—"))
            st.markdown("**Reforça:** " + (", ".join(_metric_label(m) for m in ex.get("reforca", [])) or "—"))
            if ex.get("modalidades"):
                mods = _modality_options()
                st.markdown("**Só para:** " + ", ".join(mods.get(m, m) for m in ex["modalidades"]))
            if ex.get("manutencao"):
                st.markdown("**Manutenção:** recomendado quando não há pontos de atenção")
        with c2:
            st.markdown("**Dose por gravidade:**\n" + "\n".join(
                f"- *{SEVERITY_TITLE[s]}:* {(ex.get('dose') or {}).get(s, '—')}" for s in tf.SEVERITIES))
        if ex.get("passos"):
            st.markdown("**Progressão:**\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(ex["passos"], 1)))
        if ex.get("pontos_de_atencao"):
            st.markdown("**Pontos de atenção:**\n" + "\n".join(f"- {p}" for p in ex["pontos_de_atencao"]))
        if ex.get("erros_comuns"):
            st.markdown("**Erros comuns:**\n" + "\n".join(f"- {p}" for p in ex["erros_comuns"]))
        if ex.get("como_medir"):
            st.markdown(f"**Como medir a evolução:** {ex['como_medir']}")
        st.caption(f"Identificador: `{ex_id}`")

        b1, b2, _ = st.columns([1, 1, 2])
        with b1:
            if st.button("✏️ Editar", key=f"btn_edit_{ex_id}", width="stretch"):
                st.session_state["exercise_catalog_editing"] = ex_id
                st.rerun()
        with b2:
            confirm_key = f"confirm_del_{ex_id}"
            if st.session_state.get(confirm_key):
                if st.button("⚠️ Confirmar exclusão", key=f"btn_del_ok_{ex_id}", type="primary", width="stretch"):
                    st.session_state.pop(confirm_key, None)
                    if _save([e for e in exercises if e.get("id") != ex_id], f"🗑️ Exercício '{ex.get('nome')}' excluído."):
                        st.rerun()
            elif st.button("🗑️ Excluir", key=f"btn_del_{ex_id}", width="stretch"):
                st.session_state[confirm_key] = True
                st.rerun()


def render_exercise_catalog_panel() -> None:
    st.markdown("### 🏋️ Catálogo de Exercícios")
    st.caption(
        "Exercícios que o diagnóstico do modo de treinamento recomenda: para cada ponto de atenção ele escolhe um "
        "exercício que **corrige** aquela métrica (com a dose da gravidade) e, para o ponto mais forte, um que o "
        "**reforça**. Modalidades feitas sozinho só recebem exercícios solo. As alterações valem para as próximas "
        "análises. Arquivo: `config/exercise_catalog.json`."
    )
    if "exercise_catalog_alert" in st.session_state:
        st.success(st.session_state.pop("exercise_catalog_alert"))

    exercises = list(tf.load_exercise_catalog())

    # Resumo e cobertura
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Exercícios", len(exercises))
    m2.metric("Solo", sum(1 for e in exercises if e.get("formato") == "solo"))
    m3.metric("Em dupla", sum(1 for e in exercises if e.get("formato") == "dupla"))
    m4.metric("Com bogu", sum(1 for e in exercises if e.get("formato") == "bogu"))

    coverage = tf.catalog_coverage(exercises)
    uncovered = [c["label"] for c in coverage if c["corrige"] == 0]
    if uncovered:
        st.warning("Métricas sem nenhum exercício corretivo (o diagnóstico não terá o que recomendar para elas): "
                   + ", ".join(uncovered))
    solo_uncovered = [c["label"] for c in coverage
                      if not any(c["metrica"] in (e.get("corrige") or []) and e.get("formato") == "solo" for e in exercises)]
    if solo_uncovered:
        st.info("Sem exercício corretivo **solo** (modalidades individuais, como suburi, ficam sem recomendação): "
                + ", ".join(solo_uncovered))
    with st.expander("📊 Cobertura das métricas", expanded=False):
        st.dataframe(pd.DataFrame([{"Métrica": c["label"], "Exercícios que corrigem": c["corrige"],
                                    "Exercícios que reforçam": c["reforca"]} for c in coverage]),
                     hide_index=True, width="stretch")

    # Novo exercício
    with st.expander("➕ Novo exercício", expanded=False):
        # A versão no nome do formulário o recria vazio depois de cada criação
        new_ver = st.session_state.get("exercise_catalog_new_ver", 0)
        new_ex = _exercise_form(None, f"form_new_exercise_{new_ver}", [e.get("id") for e in exercises])
        if new_ex is not None and _save(exercises + [new_ex], f"✅ Exercício '{new_ex['nome']}' criado."):
            st.session_state["exercise_catalog_new_ver"] = new_ver + 1
            st.rerun()

    # Filtros e lista
    st.markdown("#### 📋 Exercícios")
    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        query = st.text_input("Buscar", placeholder="nome, passo, ponto de atenção...", key="exercise_catalog_query")
    with f2:
        metric = st.selectbox("Métrica", ["(todas)"] + list(tf.METRICS), key="exercise_catalog_metric",
                              format_func=lambda m: m if m == "(todas)" else _metric_label(m))
    with f3:
        fmt = st.selectbox("Formato", ["(todos)"] + list(tf.FORMATS), key="exercise_catalog_format",
                           format_func=lambda f: f if f == "(todos)" else FORMAT_LABEL.get(f, f))
    shown = exercises
    if query.strip():
        q = query.strip().lower()
        shown = [e for e in shown if q in json.dumps(e, ensure_ascii=False).lower()]
    if metric != "(todas)":
        shown = [e for e in shown if metric in (e.get("corrige") or []) + (e.get("reforca") or [])]
    if fmt != "(todos)":
        shown = [e for e in shown if e.get("formato") == fmt]
    st.caption(f"{len(shown)} de {len(exercises)} exercício(s)")
    for i, ex in enumerate(shown):
        _render_exercise(ex, exercises, i)

    # Baixar / carregar
    st.markdown("#### 💾 Baixar ou carregar o catálogo")
    doc = tf.load_catalog_document()
    st.download_button("📥 Baixar catálogo (.json)", data=json.dumps(doc, ensure_ascii=False, indent=2),
                       file_name="exercise_catalog.json", mime="application/json", key="btn_dl_exercise_catalog")
    up = st.file_uploader("Carregar um catálogo (.json) — substitui o atual depois de validado", type=["json"],
                          key="exercise_catalog_upload")
    if up is not None:
        try:
            loaded = json.loads(up.getvalue().decode("utf-8"))
            new_list = loaded.get("exercicios") if isinstance(loaded, dict) else loaded
            if not isinstance(new_list, list):
                raise ValueError("o arquivo precisa ter a lista 'exercicios'")
            st.caption(f"Arquivo com {len(new_list)} exercício(s).")
            if st.button("✅ Substituir o catálogo atual", key="btn_replace_exercise_catalog", type="primary"):
                if _save(new_list, f"✅ Catálogo substituído: {len(new_list)} exercício(s)."):
                    st.rerun()
        except Exception as err:
            st.error(f"Arquivo inválido: {err}")
