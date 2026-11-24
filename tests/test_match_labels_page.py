"""
Testes da tela "Gabarito de Lutas" (Pendência I2).
"""

import os
import tempfile
import unittest

from src.engine import weak_labels as wl
from src.utils import match_labels_page as page


class TestMatchLabelsPageHelpers(unittest.TestCase):
    def test_entry_status(self):
        self.assertTrue(page.entry_status({}).startswith("⬜"))
        self.assertTrue(page.entry_status({"pontos": [], "pontos_completos": True}).startswith("🥋"))   # gabarito antigo = luta
        self.assertTrue(page.entry_status({"tipo": "luta", "pontos": []}).startswith("🥋"))
        self.assertIn("Kakari-geiko", page.entry_status({"tipo": "kakari_geiko", "ignorar": True}))

    def test_progress_counts_every_classified_video(self):
        entries = [{"pontos": [], "pontos_completos": True},              # luta com gabarito antigo
                   {"tipo": "luta", "pontos": []},                         # luta (só o tipo)
                   {"tipo": "kakari_geiko", "ignorar": True},              # treino
                   {"tipo": "shinsa", "ignorar": True},                    # exame
                   {"tipo": "descartar", "ignorar": True},
                   {"ignorar": True},                                      # antigo "não é luta" = descartado
                   {}, {"pontos": []}]                                     # não classificados
        prog = page.classification_progress(entries)
        self.assertEqual((prog["classificados"], prog["lutas"], prog["treinos"], prog["descartados"]), (6, 2, 2, 2))
        self.assertTrue(page.progress_text(prog, 8).startswith("6 de 8 vídeos classificados"))


def _labels_app(videos_dir: str, labels_path: str):
    import streamlit as st
    from src.utils.match_labels_page import render_match_labels_page
    st.session_state["labels_videos_dir"] = videos_dir
    render_match_labels_page(device_preference="cpu", labels_path=labels_path)


class TestMatchLabelsPageRender(unittest.TestCase):
    def test_page_renders_and_creates_template(self):
        from streamlit.testing.v1 import AppTest
        from src.utils.demo_generator import generate_demo_kendo_video

        with tempfile.TemporaryDirectory() as tmp:
            videos_dir = os.path.join(tmp, "videos")
            os.makedirs(videos_dir)
            generate_demo_kendo_video(os.path.join(videos_dir, "luta_demo.mp4"), duration_sec=1, fps=30)
            labels_path = os.path.join(tmp, "gabarito.json")

            at = AppTest.from_function(_labels_app, args=(videos_dir, labels_path), default_timeout=60)
            at.run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
            self.assertEqual(len(at.selectbox), 2)   # luta + tipo do vídeo
            self.assertEqual(at.selectbox[1].value, wl.VIDEO_TYPE_LUTA)
            self.assertIn("luta_demo.mp4", wl.load_match_labels(labels_path)["videos"])
            # A rotulagem manual saiu da tela: o ciclo de treino e o golden rotulam sozinhos
            self.assertFalse(any("Rotular agora" in b.label for b in at.button))

            # A tela só pede o tipo: nada de pontos, nomes ou "todos os pontos listados"
            self.assertEqual((len(at.checkbox), len(at.text_input)), (0, 1))   # só a pasta dos vídeos
            self.assertFalse(any("golden" in b.label.lower() for b in at.button))

            # Salvar como luta grava o tipo e preserva pontos antigos do arquivo
            data = wl.load_match_labels(labels_path)
            data["videos"]["luta_demo.mp4"]["pontos"] = [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:01"}]
            wl.save_match_labels(data, labels_path)
            [b for b in at.button if "Salvar tipo" in b.label][0].click().run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
            saved = wl.load_match_labels(labels_path)["videos"]["luta_demo.mp4"]
            self.assertEqual((saved["tipo"], saved["ignorar"], len(saved["pontos"])), ("luta", False, 1))
            self.assertTrue(any("pré-processamento" in s.value for s in at.success))

            # Trocar o tipo para um treino (kakari-geiko): vira rótulo de modalidade
            at.selectbox(key="labels_tipo_luta_demo.mp4").set_value("kakari_geiko")
            [b for b in at.button if "Salvar tipo" in b.label][0].click().run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
            saved = wl.load_match_labels(labels_path)["videos"]["luta_demo.mp4"]
            self.assertEqual((saved["tipo"], saved["ignorar"]), ("kakari_geiko", True))
            self.assertTrue(any("Kakari-geiko" in s.value for s in at.success))
            self.assertIn("Kakari-geiko", at.selectbox[0].format_func(at.selectbox[0].value))


def _golden_panel_app(labels_path: str):
    from src.utils.golden_benchmark_panel import render_golden_benchmark_panel, render_golden_matches_manager
    render_golden_benchmark_panel()
    render_golden_matches_manager(device_preference="cpu", labels_path=labels_path)


class TestGoldenBenchmarkPanel(unittest.TestCase):
    def test_freeze_and_unfreeze_from_the_single_panel(self):
        from streamlit.testing.v1 import AppTest
        from src.engine.active_learning import GoldenBenchmark

        gb = GoldenBenchmark()
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = os.path.join(tmp, "gabarito.json")
            wl.save_match_labels({"versao": 1, "videos": {
                "luta_a.mp4": {"pontos_completos": True, "pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:10"}]},
                "luta_b.mp4": {"pontos_completos": False, "pontos": []},
            }}, labels_path)
            from unittest import mock
            patcher = mock.patch("src.utils.golden_benchmark_panel.reviewed_counts", return_value={"luta_a.mp4": 2})
            patcher.start()
            try:
                at = AppTest.from_function(_golden_panel_app, args=(labels_path,), default_timeout=60)
                at.run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                # Sem rodada real do benchmark, nenhum número de desempenho aparece
                if not gb.load_history():
                    self.assertEqual(len(at.metric), 0)
                    self.assertTrue(any("Ainda não há medição real" in i.value for i in at.info))
                self.assertEqual(at.multiselect[0].options, ["luta_a.mp4"])  # só lutas com golpes revisados

                at.multiselect[0].select("luta_a.mp4")
                next(b for b in at.button if "Congelar selecionadas" in b.label).click().run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertTrue(gb.is_golden_video("luta_a.mp4"))
                self.assertTrue(any("Rodar benchmark por luta" in b.label for b in at.button))

                at.multiselect[1].select("luta_a.mp4")
                next(b for b in at.button if "Tirar do golden" in b.label).click().run()
                self.assertFalse(gb.is_golden_video("luta_a.mp4"))
            finally:
                patcher.stop()
                gb.unfreeze_match("luta_a.mp4")


if __name__ == "__main__":
    unittest.main()
