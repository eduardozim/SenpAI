"""
Testes da Coleta Automática na Internet (Pendência I15). Nada de rede: busca, download, bandeiras e
páginas são simulados.
"""

import json
import os
import shutil
import tempfile
import unittest
import datetime

from src.engine import internet_collector as ic
from src.engine import web_research as wr
from src.engine.label_bank import LabelBank

REG_TEXT = (
    "Article 6: The standard duration for shiai shall be five minutes, and encho shall be three minutes. "
    "Article 7: In sanbon-shobu, the shiai-sha who first scores two points within the shiai duration shall be the winner. "
    "1. The shiai-jo should be a square or a rectangle with sides of 9 to 11 meters in length inclusive of the line. "
    "Article 12: A yuko- datotsu is defined as an accurate strike or thrust made onto designated targets (datotsu-bui). "
    "Article 20: In the case whereby a shiai-sha commits two hansoku stipulated under Article 17, the opponent shall be given one point. "
)


def _item(vid, title, lic="Creative Commons Attribution license (reuse allowed)", dur=120):
    return {"id": vid, "url": f"https://example.org/{vid}", "titulo": title, "canal": "canal", "duracao_s": dur, "licenca": lic}


class _Base(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(ic.get_root_dir(), ignore_errors=True)
        cfg_path = ic._config_path()
        if os.path.isfile(cfg_path):
            os.remove(cfg_path)
        bank = LabelBank()
        bank._save({})
        self.tmp = tempfile.TemporaryDirectory()
        self.hist = os.path.join(self.tmp.name, "ciclos.jsonl")

    def tearDown(self):
        self.tmp.cleanup()
        shutil.rmtree(ic.get_root_dir(), ignore_errors=True)


class TestRuleExtraction(_Base):
    def test_facts_come_with_evidence(self):
        facts = ic.extract_rule_facts(REG_TEXT)["fatos"]
        self.assertEqual(facts["tempo_luta_min"]["valor"], 5.0)
        self.assertEqual(facts["encho_min"]["valor"], 3.0)
        self.assertEqual(facts["pontos_para_vencer"]["valor"], 2.0)
        self.assertEqual(facts["shiaijo_lado_m"]["valor"], {"min": 9.0, "max": 11.0})
        self.assertEqual(facts["hansoku_para_ponto"]["valor"], 2.0)
        for f in facts.values():
            self.assertIn(f["evidencia"].split()[1].lower(), REG_TEXT.lower())

    def test_nothing_is_filled_by_default(self):
        out = ic.extract_rule_facts("Kendo is the way of the sword. Practice with sincerity and respect every day.")
        self.assertEqual(out["fatos"], {})
        self.assertNotIn("spine_tilt_max_deg", out["fatos"])

    def test_pdf_hyphen_breaks_are_joined(self):
        rules = ic.extract_rule_facts(REG_TEXT)["regras"]
        self.assertIn("yuko_datotsu", rules)
        self.assertIn("yuko-datotsu is defined", rules["yuko_datotsu"]["trechos"][0])

    def test_llm_facts_need_literal_quote(self):
        proposed = {
            "tempo_luta_min": {"valor": 5, "evidencia": "The standard duration for shiai shall be five minutes"},
            "encho_min": {"valor": 4, "evidencia": "encho shall be four minutes"},          # não está no texto
            "shiaijo_lado_m": {"valor": 12, "evidencia": "sides of 9 to 11 meters in length"},  # número não bate
        }
        ok = ic.verify_llm_facts(proposed, REG_TEXT)
        self.assertEqual(list(ok), [])  # "5" não aparece literalmente ("five")
        proposed["tempo_luta_min"] = {"valor": 9, "evidencia": "sides of 9 to 11 meters in length"}
        self.assertEqual(list(ic.verify_llm_facts(proposed, REG_TEXT)), ["tempo_luta_min"])

    def test_collect_rules_uses_cache_and_stores(self):
        calls = []

        def fetcher(url):
            calls.append(url)
            return f"<html><body><p>{REG_TEXT}</p><script>x=1</script></body></html>".encode(), "text/html"

        cfg = {"regras": [{"url": "https://www.kendo-fik.org/regs.html", "titulo": "FIK Regulations"}], "videos": {}}
        r1 = ic.collect_rules(cfg, fetcher=fetcher)
        r2 = ic.collect_rules(cfg, fetcher=fetcher)
        self.assertEqual(len(calls), 1)
        self.assertTrue(r2["fontes"][0]["cache"])
        self.assertEqual(r1["fontes"][0]["fatos"], 5)
        self.assertEqual(r1["restricoes_injetadas"], [])   # nenhum número biomecânico no texto
        src = ic.load_rules()["fontes"]["https://www.kendo-fik.org/regs.html"]
        self.assertIn("FIK", src["autoridade"])

    def test_failed_source_is_reported(self):
        def fetcher(url):
            raise OSError("sem rede")
        res = ic.collect_rules({"regras": [{"url": "https://x.org/a.pdf", "titulo": "A"}], "videos": {}}, fetcher=fetcher)
        self.assertEqual(res["fontes"][0]["status"], "falhou")


class TestTitleModality(unittest.TestCase):
    def test_titles(self):
        cases = {
            "Kendo Basics: Kirikaeshi": "kirikaeshi",
            "All Japan Kendo Championship Final 2025": "luta",
            "Shiai geiko at the dojo": "shiai_geiko",
            "Nihon Kendo Kata 1-7": "nihon_kendo_kata",
            "Kendo footwork drills (ashi-sabaki)": "ashi_sabaki",
            "Uchikomi geiko training": "uchikomi_geiko",
            "My trip to Japan": None,
        }
        for title, expected in cases.items():
            self.assertEqual(ic.title_modality(title), expected, title)


class TestVideoCollection(_Base):
    def _cfg(self, **video):
        cfg = json.loads(json.dumps(ic.DEFAULT_CONFIG))
        cfg["videos"].update({"consultas": ["kendo"], "max_por_consulta": 5}, **video)
        return cfg

    def _searcher(self, items):
        return lambda q, n: items[:n]

    def _downloader(self, item, out_dir):
        p = os.path.join(out_dir, f"internet_{item['id']}.mp4")
        with open(p, "wb") as f:
            f.write(b"0" * 100)
        return p

    def test_search_only_catalogs(self):
        items = [_item("a", "Kendo Kirikaeshi"), _item("b", "Random vlog")]
        res = ic.collect_videos(self._cfg(), searcher=self._searcher(items), history_path=self.hist)
        self.assertEqual(res["novos_no_catalogo"], 2)
        self.assertEqual(res["baixados"], [])
        self.assertFalse(res["download_ligado"])
        self.assertEqual(ic.load_catalog()["a"]["modalidade_titulo"], "kirikaeshi")
        self.assertEqual(LabelBank().labels(), [])
        again = ic.collect_videos(self._cfg(), searcher=self._searcher(items), history_path=self.hist)
        self.assertEqual(again["novos_no_catalogo"], 0)

    def test_download_filters_and_title_label(self):
        items = [_item("cc", "Kendo Kirikaeshi"), _item("std", "Kendo Suburi", lic="não informada"),
                 _item("long", "Kendo Kakari Geiko", dur=3600), _item("none", "Random vlog")]
        res = ic.collect_videos(self._cfg(baixar=True), searcher=self._searcher(items), downloader=self._downloader,
                                history_path=self.hist)
        self.assertEqual(res["baixados"], ["internet_cc.mp4"])
        cat = ic.load_catalog()
        self.assertEqual(cat["std"]["status"], "rejeitado")
        self.assertIn("Creative Commons", cat["std"]["motivo"])
        self.assertIn("min", cat["long"]["motivo"])
        self.assertIn("título", cat["none"]["motivo"])
        labels = LabelBank().labels()
        self.assertEqual(len(labels), 1)
        lb = labels[0]
        self.assertEqual((lb["origin"], lb["claim"], lb["value"]), ("internet", "modalidade", "kirikaeshi"))
        self.assertAlmostEqual(LabelBank.authority(lb), 0.25)

    def test_license_is_checked_when_missing(self):
        items = [_item("x", "Kendo Suburi", lic=None)]
        res = ic.collect_videos(self._cfg(baixar=True), searcher=self._searcher(items), downloader=self._downloader,
                                license_reader=lambda url: None, history_path=self.hist)
        self.assertEqual(res["baixados"], [])
        self.assertEqual(ic.load_catalog()["x"]["licenca"], "não informada")

    def test_flag_points_only_when_gate_open(self):
        items = [_item("m", "Kendo World Championship Final")]
        reader = lambda path: {"eventos": [{"kenshi": "AKA", "golpe_estimado_s": 12.3, "juizes": 2}]}  # noqa: E731
        ic.collect_videos(self._cfg(baixar=True), searcher=self._searcher(items), downloader=self._downloader,
                          flag_reader=reader, history_path=self.hist)
        self.assertEqual([lb["claim"] for lb in LabelBank().labels()], ["modalidade"])   # trava fechada

        shutil.rmtree(ic.get_root_dir(), ignore_errors=True)
        LabelBank()._save({})
        with open(self.hist, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "c1", "etapas": {"bandeiras": {"status": "ok", "f1": 0.93, "videos": 4}}}) + "\n")
        self.assertTrue(ic.flag_gate(self.hist)["liberado"])
        res = ic.collect_videos(self._cfg(baixar=True), searcher=self._searcher(items), downloader=self._downloader,
                                flag_reader=reader, history_path=self.hist)
        claims = sorted(lb["claim"] for lb in LabelBank().labels())
        self.assertEqual(claims, ["modalidade", "ponto"])
        ponto = [lb for lb in LabelBank().labels() if lb["claim"] == "ponto"][0]
        self.assertEqual((ponto["origin"], ponto["kenshi"], ponto["time_sec"]), ("internet", "AKA", 12.3))
        self.assertEqual(res["duvidas"], 1)

    def test_gate_needs_enough_videos(self):
        with open(self.hist, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "c1", "etapas": {"bandeiras": {"f1": 0.95, "videos": 2}}}) + "\n")
        self.assertFalse(ic.flag_gate(self.hist)["liberado"])

    def test_disk_budget_stops_downloads(self):
        items = [_item("a", "Kendo Kirikaeshi"), _item("b", "Kendo Suburi")]
        res = ic.collect_videos(self._cfg(baixar=True, limite_disco_gb=0.0), searcher=self._searcher(items),
                                downloader=self._downloader, history_path=self.hist)
        self.assertEqual(res["baixados"], [])

    def test_resolve_video_path_falls_back_to_collected(self):
        p = self._downloader({"id": "z"}, ic.get_videos_dir())
        self.assertEqual(ic.resolve_video_path(self.tmp.name, "internet_z.mp4"), p)
        self.assertEqual(ic.resolve_video_path(self.tmp.name, "nao_existe.mp4"), os.path.join(self.tmp.name, "nao_existe.mp4"))


class TestCollectionSession(_Base):
    def _searcher_log(self):
        calls = []

        def searcher(q, n):
            calls.append(n)
            return [_item(f"{q}-{i}", "Kendo Kirikaeshi") for i in range(n)]
        return searcher, calls

    def _cfg(self):
        cfg = json.loads(json.dumps(ic.DEFAULT_CONFIG))
        cfg["videos"].update({"consultas": ["kendo"], "max_por_consulta": 2})
        cfg["textos"]["ativar"] = False   # sem rede nos testes de sessão
        return cfg

    def test_session_ends_at_deadline_and_goes_deeper(self):
        searcher, calls = self._searcher_log()
        sess = ic.run_collection_session(hours=0.4 / 3600, allow_download=False, config=self._cfg(),
                                         interval_min=0.1 / 60, searcher=searcher, history_path=self.hist)
        self.assertFalse(sess["ativa"])
        self.assertEqual(sess["encerrada_por"], "prazo encerrado")
        self.assertGreaterEqual(sess["rodadas"], 2)
        self.assertEqual(calls[:2], [2, 4])                 # cada rodada lê mais resultados
        self.assertEqual(sess["novos"], len(ic.load_catalog()))

    def test_stop_interrupts_wait(self):
        import threading
        import time
        searcher, _ = self._searcher_log()
        self.assertTrue(ic.start_collection_session(1.0, False, config=self._cfg(), interval_min=60,
                                                    searcher=searcher, history_path=self.hist))
        self.assertFalse(ic.start_collection_session(1.0, False))   # só uma por vez
        for _ in range(50):
            if ic.get_session().get("rodadas", 0) >= 1:
                break
            time.sleep(0.05)
        self.assertTrue(ic.get_session()["ativa"])
        t0 = time.time()
        self.assertTrue(ic.stop_collection_session(wait_sec=5))
        self.assertLess(time.time() - t0, 5)
        sess = ic.get_session()
        self.assertFalse(sess["ativa"])
        self.assertEqual(sess["encerrada_por"], "desligada pelo usuário")
        self.assertFalse(ic.stop_collection_session())
        self.assertIsInstance(threading.active_count(), int)

    def test_stale_active_session_is_marked_interrupted(self):
        ic._save_json("sessao.json", {"ativa": True, "rodadas": 3})
        sess = ic.get_session()
        self.assertFalse(sess["ativa"])
        self.assertIn("reiniciado", sess["encerrada_por"])

    def test_stop_check_skips_remaining_queries(self):
        searcher, calls = self._searcher_log()
        cfg = self._cfg()
        cfg["videos"]["consultas"] = ["a", "b", "c"]
        ic.collect_videos(cfg, searcher=searcher, history_path=self.hist, should_stop=lambda: len(calls) >= 1)
        self.assertEqual(len(calls), 1)

    def test_invalid_duration(self):
        with self.assertRaises(ValueError):
            ic.start_collection_session(0, False)

    def test_session_progress_telemetry(self):
        now = datetime.datetime.now()
        start = (now - datetime.timedelta(minutes=30)).isoformat(timespec="seconds")
        end = (now + datetime.timedelta(minutes=30)).isoformat(timespec="seconds")
        sess_mock = {
            "ativa": True,
            "inicio": start,
            "fim_previsto": end,
            "horas_planejadas": 1.0,
        }
        telemetry = ic.compute_session_progress(sess_mock)
        self.assertAlmostEqual(telemetry["percentual_avanco"], 50.0, delta=2.0)
        self.assertIn("30m", telemetry["tempo_decorrido_fmt"])
        self.assertEqual(telemetry["hora_termino_fmt"], str(end)[11:16])
        self.assertGreater(telemetry["tempo_decorrido_segundos"], 1700)
        self.assertGreater(telemetry["tempo_restante_segundos"], 1700)

        # Sessão encerrada mantém o tempo decorrido até o encerramento
        encerrada = (now - datetime.timedelta(minutes=10)).isoformat(timespec="seconds")
        sess_mock_closed = {
            "ativa": False,
            "inicio": start,
            "fim_previsto": end,
            "encerrada_em": encerrada,
        }
        telem_closed = ic.compute_session_progress(sess_mock_closed)
        self.assertIn("20m", telem_closed["tempo_decorrido_fmt"])
        self.assertEqual(telem_closed["tempo_restante_segundos"], 0.0)


KENDO_PAGE = ("<html><body><h1>Kendo rules</h1><p>In kendo, a yuko-datotsu is defined as an accurate strike with the shinai "
              "onto the datotsu-bui. The kendo shinpan judge men, kote, do and tsuki. Zanshin must be shown after the strike. "
              "The standard duration for shiai shall be five minutes, and encho shall be three minutes.</p></body></html>")
OTHER_PAGE = "<html><body><p>Recipe for bread: flour, water and salt. Bake for forty minutes.</p></body></html>"


class _FakeLLM:
    is_online = True
    provider = "teste"

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def _call_remote_llm(self, prompt):
        self.prompts.append(prompt)
        return self.response(prompt) if callable(self.response) else self.response


class TestTextCollection(_Base):
    def _cfg(self, **textos):
        cfg = json.loads(json.dumps(ic.DEFAULT_CONFIG))
        cfg["textos"].update({"consultas": ["kendo rules"], "consultas_por_rodada": 1, "paginas_por_consulta": 3}, **textos)
        return cfg

    def _searcher(self, q, n):
        return [{"titulo": "Kendo rules", "url": "https://example.org/kendo", "trecho": "", "fonte": "web"},
                {"titulo": "Bread", "url": "https://example.org/bread", "trecho": "", "fonte": "web"}]

    def _fetcher(self, url):
        return (KENDO_PAGE if "kendo" in url else OTHER_PAGE).encode(), "text/html"

    def test_relevant_pages_go_to_corpus_and_others_are_discarded(self):
        res = ic.collect_texts(self._cfg(), searcher=self._searcher, fetcher=self._fetcher, llm=None)
        self.assertEqual((res["lidas"], res["carregadas"], res["descartadas"]), (2, 1, 1))
        src = ic.load_rules()["fontes"]["https://example.org/kendo"]
        self.assertEqual(src["fatos"]["tempo_luta_min"]["valor"], 5.0)
        self.assertEqual(src["modo"], "regras")
        from src.utils.test_isolation import get_safe_data_path
        with open(get_safe_data_path("config/ai_knowledge_base.json"), encoding="utf-8") as f:
            kb_sources = json.load(f)["sources"]
        web = [v for v in kb_sources.values() if v.get("origem") == "coleta_internet"]
        self.assertIn("https://example.org/kendo", [v["url"] for v in web])
        self.assertNotIn("https://example.org/bread", [v["url"] for v in web])
        kinds = [a["tipo"] for a in ic.load_activity(20)]
        for k in ("busca", "pagina", "carregado", "descartado"):
            self.assertIn(k, kinds)
        again = ic.collect_texts(self._cfg(), searcher=self._searcher, fetcher=self._fetcher, llm=None)
        self.assertEqual(again["lidas"], 0)   # já vistas

    def test_ai_queries_and_only_literal_quotes_are_kept(self):
        def respond(prompt):
            if "Proponha" in prompt:
                return {"consultas": ["kendo zanshin criteria"]}
            return {"relevante": True, "resumo": "Regras de kendo.",
                    "fatos": {"encho_min": {"rotulo": "Encho", "valor": 3, "unidade": "min",
                                            "evidencia": "encho shall be three minutes"},
                              "inventado": {"rotulo": "X", "valor": 9, "unidade": "s",
                                            "evidencia": "this sentence is not in the page at all"}},
                    "trechos": [{"tema": "zanshin", "trecho": "Zanshin must be shown after the strike"},
                                {"tema": "falso", "trecho": "a frase que a IA inventou sobre kendo"}]}
        llm = _FakeLLM(respond)
        queries = []

        def searcher(q, n):
            queries.append(q)
            return self._searcher(q, n)
        res = ic.collect_texts(self._cfg(), searcher=searcher, fetcher=self._fetcher, llm=llm)
        self.assertEqual(queries, ["kendo zanshin criteria"])
        self.assertEqual(res["carregadas"], 1)
        src = ic.load_rules()["fontes"]["https://example.org/kendo"]
        self.assertEqual(src["modo"], "ia")
        self.assertNotIn("inventado", src["fatos"])
        self.assertIn("Zanshin must be shown after the strike", src["regras"]["zanshin"]["trechos"])
        self.assertNotIn("falso", src["regras"])
        self.assertIn("ia", [a["tipo"] for a in ic.load_activity(30)])

    def test_other_martial_arts_are_discarded_even_with_kendo_terms(self):
        karate = ("Karate is a martial art. In karate the karateka trains kata in the dojo with a sensei. Zanshin, kiai "
                  "and maai matter in karate kumite; karate ippon is scored with zanshin. Kendo and judo are other "
                  "budo. Karate seme and ashi work. ") * 3
        mixed = ("Budo overview. Judo, karate and aikido are taught. Judo uses kata, the dojo, the sensei, kiai, zanshin, "
                 "ippon and seme. Judo shiai. Kendo uses a shinai and men, kote. ") * 3

        def searcher(q, n):
            return [{"titulo": "Karate", "url": "https://example.org/karate", "trecho": "", "fonte": "wikipedia"},
                    {"titulo": "Budo", "url": "https://example.org/budo", "trecho": "", "fonte": "web"},
                    {"titulo": "Kendo rules", "url": "https://example.org/kendo", "trecho": "", "fonte": "web"}]

        def fetcher(url):
            page = {"karate": karate, "budo": mixed}.get(url.rsplit("/", 1)[-1], KENDO_PAGE)
            return page.encode(), "text/html"
        llm = _FakeLLM(lambda p: {"relevante": True, "resumo": "", "fatos": {}, "trechos": []})
        res = ic.collect_texts(self._cfg(), searcher=searcher, fetcher=fetcher, llm=llm)
        self.assertEqual((res["lidas"], res["carregadas"], res["descartadas"]), (2, 1, 2))
        fontes = ic.load_rules()["fontes"]
        self.assertEqual(list(fontes), ["https://example.org/kendo"])
        self.assertEqual(sum(1 for p in llm.prompts if "Página:" in p), 1)   # IA só vê a página de kendo
        self.assertFalse(wr.kendo_focus(karate, "Karate")["ok"])
        self.assertFalse(wr.kendo_focus(mixed, "Budo")["ok"])
        self.assertTrue(wr.kendo_focus(KENDO_PAGE, "Kendo rules")["ok"])

    def test_texts_can_be_disabled(self):
        res = ic.collect_texts(self._cfg(ativar=False), searcher=self._searcher, fetcher=self._fetcher, llm=None)
        self.assertEqual(res["consultas"], 0)

    def test_session_runs_texts_and_videos(self):
        cfg = self._cfg()
        cfg["videos"].update({"consultas": ["kendo"], "max_por_consulta": 1})
        sess = ic.run_collection_session(hours=0.2 / 3600, allow_download=False, config=cfg, interval_min=0.05 / 60,
                                         text_searcher=self._searcher, fetcher=self._fetcher, llm=None,
                                         searcher=lambda q, n: [_item("v1", "Kendo Suburi")], history_path=self.hist)
        self.assertEqual(sess["textos_carregados"], 1)
        self.assertEqual(sess["novos"], 1)
        kinds = [a["tipo"] for a in ic.load_activity(50)]
        self.assertIn("video_busca", kinds)
        self.assertIn("video_novo", kinds)

    def test_activity_log_is_trimmed(self):
        for i in range(ic.ACTIVITY_KEEP * 3):
            ic.log_activity("busca", "x" * 400 + str(i))
        self.assertLessEqual(len(ic.load_activity(10_000)), ic.ACTIVITY_KEEP * 2)
        self.assertTrue(ic.load_activity(1)[0]["msg"].endswith(str(ic.ACTIVITY_KEEP * 3 - 1)))


class TestApiKeys(unittest.TestCase):
    def setUp(self):
        from src.utils import api_keys as ak
        self.ak = ak
        ak.save_keys({k: None for k in ak.KNOWN_KEYS})
        self._env = {k: os.environ.pop(k, None) for k in ak.KNOWN_KEYS}

    def tearDown(self):
        self.ak.save_keys({k: None for k in self.ak.KNOWN_KEYS})
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_save_load_and_google_status(self):
        from src.engine import web_research as wr
        self.assertFalse(wr.google_status()["ativo"])
        self.assertEqual(wr.search_google("kendo"), [])
        self.ak.save_keys({"SERPAPI_KEY": "abc123", "GEMINI_API_KEY": ""})
        self.assertEqual(self.ak.key_source("SERPAPI_KEY"), "arquivo")
        self.assertIsNone(self.ak.key_source("GEMINI_API_KEY"))
        self.assertTrue(wr.google_status()["ativo"])
        self.ak.save_keys({"SERPAPI_KEY": ""})                 # em branco mantém
        self.assertEqual(self.ak.get_key("SERPAPI_KEY"), "abc123")
        self.ak.save_keys({"SERPAPI_KEY": None})               # None apaga
        self.assertIsNone(self.ak.load_file_keys().get("SERPAPI_KEY"))

    def test_keys_file_is_in_sandbox(self):
        self.assertNotEqual(os.path.abspath(self.ak._path()), os.path.abspath(self.ak.KEYS_FILE))


if __name__ == "__main__":
    unittest.main()
