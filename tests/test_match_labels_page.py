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
            self.assertEqual(len(at.selectbox), 1)
            self.assertIn("luta_demo.mp4", wl.load_match_labels(labels_path)["videos"])
            # Luta ainda sem gabarito: a seção de rotulagem pede o preenchimento
            self.assertTrue(any("Salve os pontos oficiais" in i.value for i in at.info))

            # Salvar a luta como completa (0 x 0) grava o gabarito e libera a rotulagem
            at.checkbox[0].check()
            at.text_input[1].input("Fulano")
            at.button[0].click().run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
            saved = wl.load_match_labels(labels_path)["videos"]["luta_demo.mp4"]
            self.assertTrue(saved["pontos_completos"])
            self.assertEqual(saved["aka"], "Fulano")
            self.assertTrue(any("Gabarito salvo" in s.value for s in at.success))
            self.assertTrue(any("Rotular agora" in b.label for b in at.button))


if __name__ == "__main__":
    unittest.main()
