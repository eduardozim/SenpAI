"""
Testes da tela "Gabarito de Lutas" (Pendência I2).
"""

import os
import tempfile
import unittest

import pandas as pd

from src.engine import weak_labels as wl
from src.utils import match_labels_page as page


class TestMatchLabelsPageHelpers(unittest.TestCase):
    def test_rows_to_points_validates_and_sorts(self):
        rows = pd.DataFrame([
            {"kenshi": "SHIRO", "tecnica": "KOTE", "tempo": "01:15"},
            {"kenshi": None, "tecnica": None, "tempo": None},          # linha vazia ignorada
            {"kenshi": "aka", "tecnica": "men", "tempo": "00:42"},
            {"kenshi": "AKA", "tecnica": "HANSOKU", "tempo": ""},       # penalidade sem tempo é válida
        ])
        points, errors = page.rows_to_points(rows)
        self.assertEqual(errors, [])
        self.assertEqual([p["tempo"] for p in points], ["00:42", "01:15", ""])
        self.assertEqual(points[0]["kenshi"], "AKA")
        self.assertEqual(points[0]["tecnica"], "MEN")

    def test_rows_to_points_reports_errors(self):
        rows = pd.DataFrame([
            {"kenshi": "", "tecnica": "MEN", "tempo": "00:10"},
            {"kenshi": "AKA", "tecnica": "MEN", "tempo": "1m20"},
        ])
        _, errors = page.rows_to_points(rows)
        self.assertEqual(len(errors), 2)
        self.assertIn("Linha 1", errors[0])
        self.assertIn("Linha 2", errors[1])

    def test_round_trip_with_gabarito(self):
        entry = {"pontos": [{"kenshi": "AKA", "tecnica": "MEN", "tempo": "00:42"}]}
        points, errors = page.rows_to_points(page.points_to_rows(entry))
        self.assertEqual(errors, [])
        self.assertEqual(points, entry["pontos"])
        self.assertEqual(len(wl.parse_official_points({"pontos": points})), 1)

    def test_entry_status(self):
        self.assertTrue(page.entry_status({}).startswith("⬜"))
        self.assertTrue(page.entry_status({"pontos": [], "pontos_completos": True}).startswith("✅"))
        self.assertTrue(page.entry_status({"pontos": [{"kenshi": "AKA"}]}).startswith("🟡"))
        self.assertTrue(page.entry_status({"tipo": "luta", "pontos": []}).startswith("🥋"))

    def test_progress_counts_every_classified_video(self):
        entries = [{"pontos": [], "pontos_completos": True},              # luta com gabarito
                   {"tipo": "luta", "pontos": []},                         # luta sem pontos ainda
                   {"tipo": "kakari_geiko", "ignorar": True},              # treino
                   {"tipo": "shinsa", "ignorar": True},                    # exame
                   {"tipo": "descartar", "ignorar": True},
                   {"ignorar": True},                                      # antigo "não é luta" = descartado
                   {}, {"pontos": []}]                                     # não classificados
        prog = page.classification_progress(entries)
        self.assertEqual((prog["classificados"], prog["lutas"], prog["lutas_sem_pontos"], prog["treinos"], prog["descartados"]),
                         (6, 1, 1, 2, 2))
        self.assertTrue(page.progress_text(prog, 8).startswith("6 de 8 vídeos classificados"))

    def test_format_time(self):
        self.assertEqual(page.wl_format_time(83.46), "01:23.5")
        self.assertEqual(page.wl_format_time(5), "00:05.0")


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

            # Salvar a luta como completa (0 x 0) grava o gabarito
            at.checkbox[0].check()
            at.text_input[1].input("Fulano")
            at.button[0].click().run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
            saved = wl.load_match_labels(labels_path)["videos"]["luta_demo.mp4"]
            self.assertTrue(saved["pontos_completos"])
            self.assertEqual(saved["aka"], "Fulano")
            self.assertTrue(any("Gabarito salvo" in s.value for s in at.success))
            # O golden tem um único local na interface (painel do Eixo 4), não nesta tela
            self.assertFalse(any("golden" in b.label.lower() for b in at.button))

            # Trocar o tipo para um treino (kakari-geiko): vira rótulo de modalidade e sai dos pontos
            at.selectbox(key="labels_tipo_luta_demo.mp4").set_value("kakari_geiko")
            [b for b in at.button if "Salvar gabarito" in b.label][0].click().run()
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
            try:
                at = AppTest.from_function(_golden_panel_app, args=(labels_path,), default_timeout=60)
                at.run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                # Sem rodada real do benchmark, nenhum número de desempenho aparece
                if not gb.load_history():
                    self.assertEqual(len(at.metric), 0)
                    self.assertTrue(any("Ainda não há medição real" in i.value for i in at.info))
                self.assertEqual(at.multiselect[0].options, ["luta_a.mp4"])  # só lutas completas

                at.multiselect[0].select("luta_a.mp4")
                next(b for b in at.button if "Congelar selecionadas" in b.label).click().run()
                self.assertFalse(at.exception, [e.value for e in at.exception])
                self.assertTrue(gb.is_golden_video("luta_a.mp4"))
                self.assertTrue(any("Rodar benchmark por luta" in b.label for b in at.button))

                at.multiselect[1].select("luta_a.mp4")
                next(b for b in at.button if "Tirar do golden" in b.label).click().run()
                self.assertFalse(gb.is_golden_video("luta_a.mp4"))
            finally:
                gb.unfreeze_match("luta_a.mp4")


if __name__ == "__main__":
    unittest.main()
