"""
Painel da Coleta Automática na Internet (Pendência I15), dentro da Base de Conhecimento de Referência.
Mostra só o que foi coletado de verdade: fatos com a frase do regulamento, catálogo de vídeos e a trava
do detector de bandeiras.
"""

import streamlit as st

from src.engine import internet_collector as ic


def _fmt_time(iso) -> str:
    return str(iso or "")[11:16] or "—"


def _render_session_controls(cfg) -> None:
    """Procura de vídeos por tempo determinado: ligar por X horas e desligar quando quiser."""
    sess = ic.get_session()
    with st.container(border=True):
        if sess.get("ativa"):
            st.info(f"🟢 **Procura ligada** até {_fmt_time(sess.get('fim_previsto'))} "
                    f"(download {'ligado' if sess.get('baixar') else 'desligado'}). "
                    f"Agora: {sess.get('etapa') or '—'}")
            a, b, c = st.columns(3)
            a.metric("Rodadas", sess.get("rodadas", 0))
            b.metric("Novos no catálogo", sess.get("novos", 0))
            c.metric("Baixados", len(sess.get("baixados", [])))
            d1, d2 = st.columns(2)
            if d1.button("⏹️ Desligar procura", type="primary", key="i15_stop"):
                ic.stop_collection_session()
                st.toast("Procura desligada. Um download em andamento termina antes de parar.", icon="⏹️")
                st.rerun()
            if d2.button("🔄 Atualizar", key="i15_refresh"):
                st.rerun()
            return

        st.markdown("🔴 **Procura desligada**")
        if sess.get("encerrada_por"):
            st.caption(f"Última sessão: {sess.get('rodadas', 0)} rodada(s), {sess.get('novos', 0)} novo(s), "
                       f"{len(sess.get('baixados', []))} baixado(s) — encerrada às {_fmt_time(sess.get('encerrada_em'))} "
                       f"({sess['encerrada_por']}).")
        if sess.get("erro"):
            st.error(f"A última sessão parou com erro: {sess['erro']}")
        k1, k2 = st.columns(2)
        horas = k1.number_input("Ligar por (horas)", min_value=0.25, max_value=72.0, value=2.0, step=0.25, key="i15_hours")
        baixar = k2.checkbox("Baixar vídeos durante a procura", value=bool(cfg["videos"].get("baixar")), key="i15_dl",
                             help="Segue os filtros abaixo: licença, duração, limite de disco e downloads por rodada.")
        st.caption(f"A busca se repete a cada {cfg['videos'].get('intervalo_sessao_min', 15)} min, lendo mais resultados por "
                   "consulta a cada rodada, e para sozinha no fim do prazo.")
        if st.button(f"▶️ Ligar procura por {horas:g} h", key="i15_start"):
            if ic.start_collection_session(float(horas), bool(baixar)):
                st.toast("Procura de vídeos ligada.", icon="🟢")
            st.rerun()


def render_internet_collection_panel() -> None:
    st.markdown("##### 🌐 Coleta Automática na Internet")
    st.caption(
        "Regulamentos oficiais viram referência (e restrições biomecânicas, quando o texto traz números). "
        "Vídeos encontrados entram como rótulos fracos de origem \"internet\" (peso 0,5). "
        "Os pontos lidos pelas bandeiras só entram depois que o detector provar F1 ≥ 0,9."
    )
    cfg = ic.load_config()
    st_now = ic.status()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📜 Coletar regulamentos agora", key="i15_rules"):
            with st.spinner("Baixando e lendo os regulamentos..."):
                res = ic.collect_rules(cfg)
            for r in res["fontes"]:
                (st.success if r["status"] == "ok" else st.warning)(
                    f"{r['fonte']}: " + (r.get("detalhe") or f"{r['fatos']} fato(s), {r['regras']} tipo(s) de regra"))
    with c2:
        if st.button("🔎 Buscar vídeos (sem baixar)", key="i15_search"):
            with st.spinner("Buscando vídeos..."):
                res = ic.collect_videos(cfg, allow_download=False)
            st.success(f"{res['novos_no_catalogo']} vídeo(s) novo(s) no catálogo.")

    rules = ic.load_rules().get("fontes", {})
    if rules:
        st.markdown("**Fatos extraídos dos regulamentos** (com a frase original como evidência):")
        for src in rules.values():
            with st.expander(f"📜 {src['titulo']} — {len(src.get('fatos', {}))} fato(s), {src.get('autoridade', '')}", expanded=False):
                for f in src.get("fatos", {}).values():
                    v = f["valor"]
                    val = f"{v['min']:g} a {v['max']:g}" if isinstance(v, dict) else f"{v:g}"
                    st.markdown(f"- **{f['rotulo']}:** {val} {f['unidade']}  \n  > _{f['evidencia']}_")
                for r in src.get("regras", {}).values():
                    st.markdown(f"- **{r['rotulo']}:** {len(r['trechos'])} trecho(s)")
                    for t in r["trechos"][:2]:
                        st.caption(f"“{t}”")
    else:
        st.info("Nenhum regulamento coletado ainda.")

    st.markdown("**Vídeos**")
    _render_session_controls(cfg)
    gate = st_now["bandeiras"]
    cat = st_now["catalogo"]
    m1, m2, m3 = st.columns(3)
    m1.metric("No catálogo", sum(cat.values()))
    m2.metric("Baixados", cat.get("baixado", 0))
    m3.metric("Disco (GB)", st_now["disco_gb"])
    if gate["liberado"]:
        st.success(f"Detector de bandeiras liberado (F1 {gate['f1']} em {gate['videos']} lutas): pontos dos vídeos baixados entram no banco.")
    else:
        st.warning("Detector de bandeiras ainda não validado (F1 ≥ 0,9 em ≥ 3 lutas): os vídeos entram só com o rótulo de modalidade do título.")

    v = cfg["videos"]
    with st.form("i15_cfg"):
        baixar = st.checkbox("Permitir download dos vídeos encontrados", value=bool(v.get("baixar")),
                             help="Desligado por padrão: termos de uso das plataformas e direito de imagem dos atletas.")
        only_cc = st.checkbox("Baixar só vídeos com licença Creative Commons", value=bool(v.get("somente_creative_commons", True)))
        k1, k2, k3 = st.columns(3)
        limite = k1.number_input("Limite de disco (GB)", 1.0, 500.0, float(v.get("limite_disco_gb", 10.0)))
        dur = k2.number_input("Duração máxima (min)", 1, 180, int(v.get("max_duracao_min", 20)))
        por_ciclo = k3.number_input("Downloads por rodada", 0, 50, int(v.get("max_downloads_por_ciclo", 3)))
        intervalo = st.number_input("Intervalo entre rodadas da procura (min)", 1, 240, int(v.get("intervalo_sessao_min", 15)))
        if st.form_submit_button("Salvar"):
            v.update(baixar=baixar, somente_creative_commons=only_cc, limite_disco_gb=float(limite),
                     max_duracao_min=int(dur), max_downloads_por_ciclo=int(por_ciclo), intervalo_sessao_min=int(intervalo))
            ic.save_config(cfg)
            st.success("Configuração salva.")
