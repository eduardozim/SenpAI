"""
Painel da Coleta Automática na Internet (Pendência I15), dentro da Base de Conhecimento de Referência.
Mostra só o que foi coletado de verdade: fatos com a frase do regulamento, e catálogo de vídeos.
"""

import streamlit as st

from src.engine import internet_collector as ic


def _fmt_time(iso) -> str:
    return str(iso or "")[11:16] or "—"


@st.fragment(run_every=3)
def _render_active_session_card(cfg) -> None:
    """Card de sessão ativa atualizado automaticamente a cada 3 segundos via fragment isolado."""
    sess = ic.get_session()
    if not sess.get("ativa"):
        st.rerun()
        return

    hora_fim = sess.get("hora_termino_fmt") or _fmt_time(sess.get("fim_previsto"))
    elapsed_fmt = sess.get("tempo_decorrido_fmt", "00m 00s")
    prog_pct = float(sess.get("percentual_avanco", 0.0))
    remaining_fmt = sess.get("tempo_restante_fmt", "—")
    etapa = sess.get("etapa") or "—"

    st.info(
        f"🟢 **Procura ligada** (download {'ligado' if sess.get('baixar') else 'desligado'}).  \n"
        f"⏱️ **Tempo decorrido:** `{elapsed_fmt}` &nbsp;|&nbsp; "
        f"📊 **Avanço:** `{prog_pct:.1f}%` (restam ~`{remaining_fmt}`) &nbsp;|&nbsp; "
        f"🏁 **Término previsto:** `{hora_fim}`  \n"
        f"🔍 **Agora:** {etapa}"
    )

    # Barra visual de progresso da sessão
    st.progress(
        min(1.0, max(0.0, prog_pct / 100.0)),
        text=f"Avanço do Prazo: {prog_pct:.1f}% ({elapsed_fmt} decorridos • Término previsto às {hora_fim})"
    )

    # Métricas em destaque: tempo decorrido, percentual e hora de término
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱️ Tempo Decorrido", elapsed_fmt)
    c2.metric("📊 Avanço do Prazo", f"{prog_pct:.1f}%", help=f"Restam aproximadamente {remaining_fmt}")
    c3.metric("🏁 Hora de Término", hora_fim)

    # Métricas operacionais da busca e downloads
    a, b, c, d = st.columns(4)
    a.metric("Rodadas", sess.get("rodadas", 0))
    b.metric("Textos no Corpus", sess.get("textos_carregados", 0), help=f"{sess.get('textos_lidos', 0)} página(s) lida(s)")
    c.metric("Vídeos novos", sess.get("novos", 0))
    d.metric("Baixados", len(sess.get("baixados", [])))
    _render_activity_feed()

    st.caption("⚡ *Atualização automática ativa (a cada 3s). Sem necessidade de recarregar a tela.*")

    d1, d2 = st.columns(2)
    if d1.button("⏹️ Desligar procura", type="primary", key="i15_stop"):
        ic.stop_collection_session()
        st.toast("Procura desligada. Um download em andamento termina antes de parar.", icon="⏹️")
        st.rerun()
    if d2.button("🔄 Atualizar agora", key="i15_refresh"):
        st.rerun()


def _render_session_controls(cfg) -> None:
    """Procura de vídeos por tempo determinado: ligar por X horas e desligar quando quiser."""
    sess = ic.get_session()
    with st.container(border=True):
        if sess.get("ativa"):
            _render_active_session_card(cfg)
            return

        st.markdown("🔴 **Coleta desligada**")
        if sess.get("encerrada_por"):
            dur_enc = sess.get("tempo_decorrido_fmt")
            dur_str = f", tempo decorrido: {dur_enc}" if dur_enc else ""
            st.caption(f"Última sessão: {sess.get('rodadas', 0)} rodada(s), {sess.get('novos', 0)} novo(s), "
                       f"{len(sess.get('baixados', []))} baixado(s){dur_str} — encerrada às {_fmt_time(sess.get('encerrada_em'))} "
                       f"({sess['encerrada_por']}).")
        if sess.get("erro"):
            st.error(f"A última sessão parou com erro: {sess['erro']}")
        k1, k2 = st.columns(2)
        horas = k1.number_input("Ligar por (horas)", min_value=0.25, max_value=72.0, value=2.0, step=0.25, key="i15_hours")
        baixar = k2.checkbox("Baixar vídeos durante a coleta", value=bool(cfg["videos"].get("baixar")), key="i15_dl",
                             help="Segue os filtros abaixo: licença, duração, limite de disco e downloads por rodada.")
        st.caption(f"A cada {cfg['videos'].get('intervalo_sessao_min', 15)} min, uma rodada pesquisa textos na web e vídeos no "
                   "YouTube, indo mais fundo a cada rodada, e para sozinha no fim do prazo.")
        if st.button(f"▶️ Ligar coleta por {horas:g} h", key="i15_start", type="primary"):
            if ic.start_collection_session(float(horas), bool(baixar)):
                st.toast("Procura de vídeos ligada.", icon="🟢")
            st.rerun()


_ICON = {"busca": "🔎", "pagina": "📄", "carregado": "✅", "descartado": "🚫", "video_busca": "🎬", "video_novo": "🎞️",
         "download": "⬇️", "ia": "🧠", "erro": "⚠️", "rodada": "▶️"}


def _render_activity_feed(limit: int = 15) -> None:
    """O que está sendo pesquisado e carregado, do mais recente para o mais antigo."""
    acts = ic.load_activity(limit)
    st.markdown("**📡 Atividade da coleta**")
    if not acts:
        st.caption("Nenhuma atividade registrada ainda.")
        return
    lines = []
    for a in acts:
        msg = a["msg"]
        if a.get("url"):
            msg += f" — [link]({a['url']})"
        lines.append(f"`{a['em'][11:19]}` {msg}")
    st.markdown("  \n".join(lines))


def _render_api_keys_form() -> None:
    """Chaves da IA e do Google, com onde obter cada uma. Ficam só neste computador (fora do git)."""
    from src.utils import api_keys as ak
    with st.expander("🔑 Chaves de API (IA e Google)", expanded=False):
        st.caption("Ficam só neste computador, em `config/api_keys.local.json` (fora do git e fora do pacote exportado). "
                   "Deixe um campo em branco para manter a chave atual.")
        with st.form("api_keys_form"):
            values = {}
            for name, (label, where) in ak.KNOWN_KEYS.items():
                src = ak.key_source(name)
                status = {"arquivo": "✅ configurada", "ambiente": "✅ definida no ambiente"}.get(src, "⚪ não configurada")
                values[name] = st.text_input(f"{label} — {status}", value="", type="password" if "KEY" in name else "default",
                                             key=f"apikey_{name}", help=where)
                st.caption(f"Onde obter: {where}")
            c1, c2 = st.columns(2)
            save = c1.form_submit_button("💾 Salvar chaves", type="primary")
            clear = c2.form_submit_button("🗑️ Apagar chaves salvas")
        if save:
            ak.save_keys(values)
            st.success("Chaves salvas.")
            st.rerun()
        if clear:
            ak.save_keys({k: None for k in ak.KNOWN_KEYS})
            st.success("Chaves salvas apagadas (as variáveis de ambiente, se houver, continuam valendo).")
            st.rerun()


def render_live_collection() -> None:
    """Bloco ao vivo da coleta (texto + vídeo): modo da IA, fontes de busca, controles e atividade."""
    from src.engine import web_research as wr
    cfg = ic.load_config()
    ai, google = wr.ai_status(), wr.google_status()
    st.markdown("##### 🌐 Coleta automática de textos e vídeos")
    st.caption(("🧠 " if ai["ativa"] else "⚪ ") + ai["descricao"])
    st.caption(f"Buscas: {'🟢' if google['ativo'] else '⚪'} Google · 🟢 DuckDuckGo · 🟢 Wikipédia · 🟢 YouTube (vídeos). "
               + ("" if google["ativo"] else google["descricao"]))
    st.caption("ℹ️ *Para configurar ou alterar as chaves de API do Gemini, OpenAI ou SerpApi, acesse **Menu de Configurações > ⚙️ Variáveis do Sistema**.*")
    _render_session_controls(cfg)
    if not ic.get_session().get("ativa"):
        with st.expander("📡 Atividade da última coleta", expanded=False):
            _render_activity_feed(30)


def render_internet_collection_panel() -> None:
    st.markdown("##### 🌐 Coleta Automática na Internet")
    st.caption(
        "Regulamentos oficiais viram referência (e restrições biomecânicas, quando o texto traz números). "
        "Vídeos encontrados entram só com o rótulo fraco de modalidade pelo título (origem \"internet\", peso 0,5); "
        "a validade dos golpes vem das revisões."
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

    rules = {u: s_ for u, s_ in ic.load_rules().get("fontes", {}).items() if s_.get("fatos")}
    if rules:
        st.markdown("**Fatos numéricos extraídos** (com a frase original como evidência; as fontes completas estão no Corpus):")
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
        st.info("Nenhum fato numérico coletado ainda.")

    st.markdown("**Vídeos (YouTube)**")
    cat = st_now["catalogo"]
    m1, m2, m3 = st.columns(3)
    m1.metric("No catálogo", sum(cat.values()))
    m2.metric("Baixados", cat.get("baixado", 0))
    m3.metric("Disco (GB)", st_now["disco_gb"])

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
