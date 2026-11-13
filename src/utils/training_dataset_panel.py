"""
Painel "Gerenciamento do Dataset de Treinamento": apagar, baixar e carregar o treinamento.
Os três usam a mesma lista (src/engine/training_dataset.TRAINING_STORES) e mostram só contagens reais.
"""

import time

import pandas as pd
import streamlit as st

from src.engine import training_dataset as td


def _summary_table(summary) -> pd.DataFrame:
    return pd.DataFrame([{"Item": td.SUMMARY_LABELS.get(k, k), "Quantidade": v} for k, v in summary.items()])


def _after_change() -> None:
    """Descarta o que a tela guardou em memória do treinamento anterior."""
    for key in ("last_auto_train_res",):
        st.session_state.pop(key, None)
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass


def render_training_dataset_panel(feedback_mgr=None, auto_trainer=None) -> None:
    st.markdown("#### 🛠️ Gerenciamento do Dataset de Treinamento")
    st.caption(
        "Os três botões usam a mesma lista do que é treinamento: revisões, gabarito de lutas, Banco Único de Rótulos, "
        "golden, fila e histórico da curadoria (com clipes), modelos treinados, ciclos, coleta na "
        "internet, perfis de calibração e base de conhecimento. Não entram: seus vídeos, o cache de poses e as configurações do app."
    )

    msg = st.session_state.get("dataset_action_msg")
    if msg:
        kind, text = msg
        (st.success if kind == "ok" else st.error)(text)
        if st.button("✖️ Fechar aviso", key="btn_dismiss_dataset_msg", type="tertiary"):
            st.session_state.pop("dataset_action_msg", None)
            st.rerun()

    summary = td.summarize()
    with st.expander(f"📊 Treinamento atual ({sum(summary.values())} itens)", expanded=False):
        st.dataframe(_summary_table(summary), hide_index=True, width="stretch")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🗑️ Apagar Treinamento**")
        st.caption("Zera todo o treinamento e volta ao padrão de fábrica. Antes, guarda uma cópia de segurança em `data/backups/`.")
        ver = st.session_state.get("reset_chk_ver", 0)
        confirm = st.checkbox("Confirmo que desejo apagar todo o treinamento", key=f"chk_confirm_reset_{ver}")
        if st.button("🗑️ Apagar Treinamento", type="secondary", width="stretch", key="btn_reset_train_tab"):
            if not confirm:
                st.warning("⚠️ Marque a caixa de confirmação acima antes de apagar.")
            else:
                try:
                    res = td.reset_training()
                    _after_change()
                    removed = ", ".join(f"{td.SUMMARY_LABELS[k]}: {v}" for k, v in res["antes"].items() if v) or "nada (já estava vazio)"
                    st.session_state["dataset_action_msg"] = ("ok",
                        f"✅ **Treinamento apagado.** Removido: {removed}. "
                        f"{res['arquivos_removidos']} arquivo(s) removido(s); perfis de calibração e base de conhecimento no padrão."
                        + (f"\n\nCópia de segurança: `{res['copia_de_seguranca']}`" if res["copia_de_seguranca"] else ""))
                except td.TrainingBusyError as e:
                    st.session_state["dataset_action_msg"] = ("erro", f"❌ {e}")
                except Exception as e:
                    st.session_state["dataset_action_msg"] = ("erro", f"❌ O apagar não foi concluído e nada foi alterado: {type(e).__name__}: {e}")
                st.session_state["reset_chk_ver"] = ver + 1
                st.rerun()

    with col2:
        st.markdown("**📥 Baixar Treinamento**")
        st.caption("Pacote .zip com tudo o que o botão Apagar zera, com manifesto e verificação de integridade (SHA-256). "
                   "Vídeos baixados da internet ficam de fora (o catálogo guarda os links).")
        st.download_button(
            label="📦 Baixar Pacote (.zip)",
            data=td.export_training_zip,
            file_name=f"senpai_treinamento_{time.strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            width="stretch",
            key="btn_dl_train_zip_tab",
        )

    with col3:
        st.markdown("**📤 Carregar Treinamento**")
        st.caption("Restaura o estado exato do pacote: o treinamento atual é substituído (com cópia de segurança antes). "
                   "Pacotes antigos, sem manifesto, são mesclados como antes.")
        from src.utils.environment import get_max_upload_size_mb
        up = st.file_uploader("Selecione o pacote (.zip ou .json)", type=["zip", "json"], key="import_pkg_file_tab",
                              max_upload_size=get_max_upload_size_mb())
        if up is not None:
            data = up.getvalue()
            new_fmt = up.name.lower().endswith(".zip") and td.is_new_format(data)
            st.caption(f"`{up.name}` ({len(data) / 1024:.1f} KB) — " + ("pacote completo: vai **substituir** o treinamento atual."
                                                                       if new_fmt else "formato antigo: será **mesclado**."))
            if st.button("📤 Restaurar treinamento", type="primary", width="stretch", key="btn_import_train_tab"):
                with st.spinner("Restaurando..."):
                    try:
                        if new_fmt:
                            res = td.import_training_zip(data)
                            itens = ", ".join(f"{td.SUMMARY_LABELS[k]}: {v}" for k, v in res["estado_atual"].items() if v) or "pacote vazio"
                            text = (f"✅ **Treinamento restaurado** do pacote de {res['pacote_criado_em']} "
                                    f"({res['arquivos']} arquivo(s)). Agora: {itens}."
                                    + (f"\n\nCópia do estado anterior: `{res['copia_de_seguranca']}`" if res["copia_de_seguranca"] else ""))
                        else:
                            up.seek(0)
                            res = feedback_mgr.import_training_package_archive(up, auto_trainer_instance=auto_trainer)
                            text = (f"✅ **Pacote antigo mesclado:** {res.get('new_items_added', 0)} revisão(ões), "
                                    f"{res.get('active_learning_queue_imported', 0)} dúvida(s) na fila, "
                                    f"{res.get('clips_extracted_count', 0)} clipe(s).")
                        _after_change()
                        st.session_state["dataset_action_msg"] = ("ok", text)
                    except (td.InvalidPackageError, td.TrainingBusyError) as e:
                        st.session_state["dataset_action_msg"] = ("erro", f"❌ Nada foi alterado: {e}")
                    except Exception as e:
                        st.session_state["dataset_action_msg"] = ("erro", f"❌ Erro ao restaurar: {e}")
                st.rerun()
