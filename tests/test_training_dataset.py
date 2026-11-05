"""
Testes do controle do Dataset de Treinamento: apagar zera tudo, exportar leva tudo, importar restaura o
estado exato. Tudo numa pasta temporária (nunca nos dados reais).
"""

import io
import json
import os
import tempfile
import unittest
import zipfile
from unittest import mock

from src.engine import training_dataset as td


def _w(root, rel, content):
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(p, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(content if isinstance(content, (bytes, str)) else json.dumps(content))
    return p


def _populate(root):
    _w(root, "data/feedback_dataset.json", [{"id": "r1"}, {"id": "r2"}])
    _w(root, "data/training_history.json", [{"x": 1}])
    _w(root, "data/label_bank.json", {"versao": 1, "rotulos": {"a": {}, "b": {}, "c": {}}})
    _w(root, "data/match_labels/gabarito_lutas.json", {"videos": {"l1.mp4": {"pontos_completos": True}, "l2.mp4": {}}})
    _w(root, "data/benchmark_golden/golden_matches.json", {"versao": 1, "lutas": {"l1.mp4": {}}})
    _w(root, "data/active_learning_queue.json", [{"id": "q"}])
    _w(root, "data/active_learning_curated_history.json", [{"id": "h"}])
    _w(root, "data/active_learning_clips/c1_corte.mp4", b"\x00clip")
    _w(root, "data/training_cycles.jsonl", '{"id": "c1"}\n{"id": "c2"}\n')
    _w(root, "data/flag_dataset/crops/a.jpg", b"jpgA")
    _w(root, "data/flag_dataset/crops/b.jpg", b"jpgB")
    _w(root, "data/flag_dataset/labels.json", {"versao": 1, "recortes": {
        "a": {"arquivo": "data\\flag_dataset\\crops\\a.jpg", "rotulo": "vermelha", "rotulado_em": "x"},
        "b": {"arquivo": "data\\flag_dataset\\crops\\b.jpg"}}})
    _w(root, "data/models/strike_filter/active.json", {"path": "m.json"})
    _w(root, "data/models/strike_filter/m.json", {"w": [1, 2]})
    _w(root, "data/models/flag_detector/runs/r1/weights.bin", b"big")
    _w(root, "data/internet/catalogo.json", {"v1": {}})
    _w(root, "data/videos/internet_v1.mp4", b"video")
    _w(root, "config/calibration_profiles.json", {"normal": {"custom": True}})
    _w(root, "config/ai_knowledge_base.json", {"custom": True})
    # Não é treinamento: não pode ser tocado
    _w(root, "data/feature_cache/k.pkl.gz", b"cache")
    _w(root, "config/settings.json", {"tema": "escuro"})


def _snapshot(root):
    out = {}
    for rel in td.iter_training_files(root, for_export=True):
        with open(os.path.join(root, *rel.split("/")), "rb") as f:
            out[rel] = f.read()
    return out


class TestTrainingDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        _populate(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _exists(self, rel):
        return os.path.exists(os.path.join(self.root, *rel.split("/")))

    def test_summary_counts_real_items(self):
        s = td.summarize(self.root)
        self.assertEqual(s, {"revisoes": 2, "rotulos_no_banco": 3, "lutas_com_gabarito": 1, "lutas_no_golden": 1,
                             "duvidas_na_fila": 1, "decisoes_de_curadoria": 1, "recortes_de_bandeira_rotulados": 1,
                             "modelos_ativos": 1, "ciclos_de_treino": 2})

    def test_export_contains_everything_but_cache(self):
        with zipfile.ZipFile(io.BytesIO(td.export_training_zip(self.root))) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("manifest.json"))
        for rel in ("data/label_bank.json", "data/match_labels/gabarito_lutas.json", "data/benchmark_golden/golden_matches.json",
                    "data/active_learning_clips/c1_corte.mp4", "data/models/strike_filter/m.json", "data/flag_dataset/labels.json",
                    "data/flag_dataset/crops/a.jpg", "data/internet/catalogo.json", "config/calibration_profiles.json"):
            self.assertIn(rel, names)
        for rel in ("data/flag_dataset/crops/b.jpg", "data/videos/internet_v1.mp4",
                    "data/models/flag_detector/runs/r1/weights.bin", "data/feature_cache/k.pkl.gz", "config/settings.json"):
            self.assertNotIn(rel, names)
        self.assertEqual(manifest["resumo"]["rotulos_no_banco"], 3)
        self.assertEqual(len(manifest["arquivos"]), len(names) - 1)

    def test_reset_zeroes_everything_and_keeps_non_training(self):
        res = td.reset_training(self.root)
        self.assertTrue(all(v == 0 for v in res["depois"].values()), res["depois"])
        self.assertEqual(res["antes"]["revisoes"], 2)
        for rel in ("data/label_bank.json", "data/match_labels", "data/benchmark_golden", "data/active_learning_clips",
                    "data/models", "data/internet", "data/training_cycles.jsonl", "config/ai_knowledge_base.json"):
            self.assertFalse(self._exists(rel), rel)
        with open(os.path.join(self.root, "config", "calibration_profiles.json"), encoding="utf-8") as f:
            self.assertNotIn("custom", json.load(f).get("normal", {}))
        # Índice dos recortes fica, só sem rótulos; cache e configurações do app intactos
        with open(os.path.join(self.root, "data", "flag_dataset", "labels.json"), encoding="utf-8") as f:
            crops = json.load(f)["recortes"]
        self.assertEqual(set(crops), {"a", "b"})
        self.assertNotIn("rotulo", crops["a"])
        self.assertTrue(self._exists("data/flag_dataset/crops/a.jpg"))
        self.assertTrue(self._exists("data/feature_cache/k.pkl.gz"))
        self.assertTrue(self._exists("config/settings.json"))
        self.assertTrue(res["copia_de_seguranca"] and os.path.isfile(res["copia_de_seguranca"]))

    def test_export_reset_import_restores_exact_state(self):
        before = _snapshot(self.root)
        pkg = td.export_training_zip(self.root)
        td.reset_training(self.root)
        res = td.import_training_zip(pkg, self.root)
        self.assertEqual(_snapshot(self.root), before)
        self.assertEqual(res["estado_atual"], td.summarize(self.root))
        self.assertEqual(res["estado_atual"]["rotulos_no_banco"], 3)

    def test_import_replaces_instead_of_merging(self):
        pkg = td.export_training_zip(self.root)
        _w(self.root, "data/label_bank.json", {"versao": 1, "rotulos": {"novo": {}}})
        _w(self.root, "data/models/technique_classifier/active.json", {"path": "t.json"})
        td.import_training_zip(pkg, self.root)
        self.assertEqual(td.summarize(self.root)["rotulos_no_banco"], 3)
        self.assertFalse(self._exists("data/models/technique_classifier"))
        backups = os.listdir(os.path.join(self.root, "data", "backups"))
        self.assertTrue(any("importar" in b for b in backups))

    def test_tampered_package_changes_nothing(self):
        pkg = td.export_training_zip(self.root)
        buf = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(pkg)) as src, zipfile.ZipFile(buf, "w") as dst:
            for n in src.namelist():
                data = src.read(n)
                dst.writestr(n, b'{"versao": 1, "rotulos": {}}' if n == "data/label_bank.json" else data)
        before = _snapshot(self.root)
        with self.assertRaises(td.InvalidPackageError):
            td.import_training_zip(buf.getvalue(), self.root)
        self.assertEqual(_snapshot(self.root), before)

    def test_path_outside_training_is_rejected(self):
        manifest = {"formato": td.PACKAGE_FORMAT, "versao": 1, "arquivos": [
            {"caminho": "../../evil.py", "bytes": 1, "sha256": "x"}]}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("../../evil.py", "x")
        with self.assertRaises(td.InvalidPackageError):
            td.import_training_zip(buf.getvalue(), self.root)
        manifest["arquivos"] = [{"caminho": "app.py", "bytes": 1, "sha256": "x"}]
        buf2 = io.BytesIO()
        with zipfile.ZipFile(buf2, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("app.py", "x")
        with self.assertRaises(td.InvalidPackageError):
            td.import_training_zip(buf2.getvalue(), self.root)

    def test_legacy_and_garbage(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("training_package.json", "{}")
        self.assertFalse(td.is_new_format(buf.getvalue()))
        self.assertTrue(td.is_new_format(td.export_training_zip(self.root)))
        with self.assertRaises(td.InvalidPackageError):
            td.import_training_zip(b"nao e zip", self.root)

    def test_busy_cycle_blocks_reset(self):
        with mock.patch("src.engine.training_cycle.get_cycle_state", return_value={"rodando": True}):
            with self.assertRaises(td.TrainingBusyError):
                td.reset_training(self.root)
        self.assertEqual(td.summarize(self.root)["revisoes"], 2)

    def test_backups_are_rotated(self):
        for _ in range(td.MAX_BACKUPS + 2):
            _populate(self.root)
            with mock.patch("src.engine.training_dataset.datetime") as dt:
                dt.datetime.now.return_value.strftime.return_value = f"{len(os.listdir(os.path.join(self.root, 'data', 'backups'))) if self._exists('data/backups') else 0:06d}"
                dt.datetime.now.return_value.isoformat.return_value = "2026-11-05T00:00:00"
                td.reset_training(self.root)
        self.assertLessEqual(len(os.listdir(os.path.join(self.root, "data", "backups"))), td.MAX_BACKUPS)

    def test_failure_midway_rolls_back(self):
        before = _snapshot(self.root)
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if calls["n"] == 6:
                raise PermissionError("arquivo em uso")
            return real_replace(src, dst)
        with mock.patch("src.engine.training_dataset.os.replace", side_effect=flaky):
            with self.assertRaises(PermissionError):
                td.reset_training(self.root, backup=False)
        self.assertEqual(_snapshot(self.root), before)
        self.assertFalse([n for n in os.listdir(os.path.join(self.root, "data")) if n.startswith(".lixeira")])

    def test_import_failure_keeps_current_state(self):
        pkg = td.export_training_zip(self.root)
        _w(self.root, "data/label_bank.json", {"versao": 1, "rotulos": {"atual": {}}})
        before = _snapshot(self.root)
        real_replace = os.replace
        calls = {"n": 0}

        def flaky(src, dst):
            calls["n"] += 1
            if ".importando_" in str(src) and calls["n"] > 12:
                raise PermissionError("disco cheio")
            return real_replace(src, dst)
        with mock.patch("src.engine.training_dataset.os.replace", side_effect=flaky):
            with self.assertRaises(PermissionError):
                td.import_training_zip(pkg, self.root, backup=False)
        self.assertEqual(_snapshot(self.root), before)

    def test_runs_in_own_thread(self):
        import threading
        names = []
        real = td._delete_all

        def spy(root):
            names.append(threading.current_thread().name)
            return real(root)
        with mock.patch("src.engine.training_dataset._delete_all", side_effect=spy):
            td.reset_training(self.root, backup=False)
        self.assertNotEqual(names[0], threading.current_thread().name)

    def test_storage_report_and_progress(self):
        _w(self.root, "models/yolov8n-pose.pt", b"0" * 1000)
        rep = td.storage_report(self.root)
        self.assertEqual(rep["total_bytes"], sum(rep["grupos"].values()))
        self.assertGreater(rep["grupos"]["dataset"], 0)
        self.assertGreater(rep["grupos"]["modelos"], 0)
        apoio = {a["caminho"]: a for a in rep["apoio"]}
        self.assertEqual(apoio["models"]["bytes"], 1000)                      # base baixado: apoio, não treino
        self.assertEqual(apoio["data/flag_dataset/crops"]["arquivos"], 1)      # b.jpg (a.jpg rotulado conta no dataset)
        prog = td.training_progress(self.root)
        self.assertEqual(prog["modelos_ativos"], 1)
        self.assertIsNone(prog["ultima_medicao_golden"])
        _w(self.root, "data/benchmark_golden/benchmark_history.jsonl", '{"executado_em": "x", "total": {"f1": 0.5}}\n')
        self.assertEqual(td.training_progress(self.root)["ultima_medicao_golden"]["total"]["f1"], 0.5)
        td.reset_training(self.root, backup=False)
        self.assertEqual(td.storage_report(self.root)["grupos"]["modelos"], 0)

    def test_internet_videos_live_in_videos_folder(self):
        _w(self.root, "data/videos/internet_abc.mp4", b"baixado")
        _w(self.root, "data/videos/Minha luta.mp4", b"meu video")
        with zipfile.ZipFile(io.BytesIO(td.export_training_zip(self.root))) as zf:
            self.assertFalse([n for n in zf.namelist() if n.startswith("data/videos/")])
        rep = td.storage_report(self.root)
        item = [i for i in rep["itens"] if i["caminho"].startswith("data/videos")][0]
        self.assertEqual((item["arquivos"], item["bytes"]), (2, len(b"baixado") + len(b"video")))  # + internet_v1 do _populate
        user = [a for a in rep["apoio"] if a["caminho"] == "data/videos"][0]
        self.assertEqual((user["arquivos"], user["bytes"]), (1, len(b"meu video")))
        td.reset_training(self.root, backup=False)
        self.assertFalse(self._exists("data/videos/internet_abc.mp4"))
        self.assertTrue(self._exists("data/videos/Minha luta.mp4"))

    def test_default_root_is_sandbox_in_tests(self):
        self.assertNotEqual(os.path.abspath(td.get_root()), os.path.abspath(os.getcwd()))


if __name__ == "__main__":
    unittest.main()
