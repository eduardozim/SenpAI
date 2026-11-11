"""
Coleta Automática na Internet (Pendência I15 — 3º tipo de treinamento).

Duas frentes, com travas para que nada inventado ou não confiável entre no treino:

1. REGRAS (texto oficial → referência e restrições, nunca treino de pose)
   - Baixa os regulamentos configurados (FIK/AJKF, PDF ou HTML), guarda o texto em cache e extrai
     FATOS com a frase de origem como evidência (ex.: tempo de luta, encho, tamanho do shiaijo,
     definição de yuko-datotsu, faltas/hansoku).
   - Só vale o que está escrito: número sem a frase que o contém é descartado. Com assistente LLM
     online, cada fato proposto só é aceito se a citação aparecer literalmente no texto.
   - Restrições biomecânicas numéricas (graus, ms) entram no prior (BayesianPriorInjector) com a
     autoridade da fonte; o resto fica como referência consultável.

2. VÍDEOS (internet → rótulos fracos de origem "internet", peso 0,5)
   - Busca (yt-dlp `ytsearch`) por termos e modalidades e cataloga título, canal, duração e licença,
     sem baixar nada.
   - O título vira rótulo fraco de modalidade ("kirikaeshi", "final"…) para a I14.
   - Download só se o usuário ligar (`baixar: true`), dentro do limite de disco e de duração, de
     preferência só Creative Commons (termos de uso e direito de imagem dos atletas).
   - O vídeo entra só com o rótulo de modalidade. A validade dos golpes vem das revisões (curadoria),
     nunca da decisão dos shinpan.
   - Os rótulos entram no Banco Único (I11); o ciclo (I8) só promove modelos que melhorem o golden.

Uso via linha de comando (a partir da pasta Dev/):
    python -m src.engine.internet_collector regras            # baixa e extrai as fontes de regras
    python -m src.engine.internet_collector buscar [--baixar]  # busca vídeos (e baixa, se permitido)
    python -m src.engine.internet_collector status
"""

import os
import re
import json
import time
import shutil
import hashlib
import datetime
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir
from src.utils.file_utils import safe_replace, safe_write_json, safe_load_json, _GLOBAL_FILE_IO_LOCK

DEFAULT_CONFIG_PATH = os.path.join("config", "internet_sources.json")
INTERNET_ORIGIN = "internet"
TITLE_LABEL_CONFIDENCE = 0.5     # rótulo de modalidade pelo título (fraco)
MAX_SEARCH_DEPTH = 100           # resultados por consulta, no máximo

DEFAULT_CONFIG: Dict[str, Any] = {
    "textos": {"ativar": True, "consultas": [], "consultas_por_rodada": 4, "paginas_por_consulta": 3},
    "regras": [
        {"url": "https://kendo.rs/files/202107-Regulations-of-Kendo-Shiai-and-Shinpan.pdf",
         "titulo": "FIK — Regulations of Kendo Shiai and Shinpan (jul/2021, texto completo; cópia da federação sérvia)", "autoridade": "FIK"},
        {"url": "https://www.kendo.or.jp/wp/wp-content/uploads/2025/01/RevisionE-Regulations-of-Kendo-Shiai-and-Shinpan-20230726.pdf",
         "titulo": "FIK — Regulations of Kendo Shiai and Shinpan (revisão 26/07/2023)", "autoridade": "FIK"},
        {"url": "https://www.kendo-fik.org/wp-content/uploads/2026/01/FIK-Handbook-for-Kendo-Shiai-and-Shinpan-Management-20250707.pdf",
         "titulo": "FIK — Handbook for Kendo Shiai and Shinpan Management (07/07/2025)", "autoridade": "FIK"},
    ],
    "videos": {
        "consultas": [
            "kendo world championship final", "kendo shiai all japan championship", "kendo kirikaeshi",
            "kendo suburi", "kendo kakari geiko", "kendo uchikomi geiko", "kendo ji geiko", "kendo nihon kendo kata",
            "kendo ashi sabaki", "kendo oji waza", "kendo shinsa exam",
        ],
        "max_por_consulta": 5,
        "max_duracao_min": 20,
        "limite_disco_gb": 10.0,
        "max_downloads_por_ciclo": 3,
        "prioridade_luta": True,   # lutas (shiai) primeiro: têm golpes para revisar e pontos para o gabarito
        "baixar": False,
        "somente_creative_commons": True,
        "intervalo_sessao_min": 15,
    },
}


# ------------------------------------------------------------------------------
# Caminhos e configuração
# ------------------------------------------------------------------------------

def get_root_dir() -> str:
    root = os.path.join(get_test_sandbox_dir(), "internet") if is_test_mode() else os.path.join("data", "internet")
    os.makedirs(root, exist_ok=True)
    return root


def get_videos_dir() -> str:
    """Os vídeos baixados vão para a pasta única de vídeos (Dev/data/videos), com o prefixo internet_."""
    from src.utils.video_library import get_videos_dir as _library_dir
    return _library_dir()


def _internet_bytes() -> int:
    """Disco ocupado só pelos vídeos baixados pela coleta (os seus vídeos não contam no limite)."""
    from src.utils.video_library import internet_videos
    return sum(os.path.getsize(p) for p in internet_videos(get_videos_dir()))


def _config_path(path: Optional[str] = None) -> str:
    if path:
        return path
    return os.path.join(get_test_sandbox_dir(), "internet_sources.json") if is_test_mode() else DEFAULT_CONFIG_PATH


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Configuração das fontes; cria o arquivo com as fontes oficiais na primeira vez."""
    p = _config_path(path)
    if not os.path.isfile(p):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged["regras"] = cfg.get("regras", merged["regras"])
    merged["videos"].update(cfg.get("videos", {}))
    merged["textos"].update(cfg.get("textos", {}))
    return merged


def save_config(cfg: Dict[str, Any], path: Optional[str] = None) -> None:
    p = _config_path(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def _load_json(name: str, default):
    return safe_load_json(os.path.join(get_root_dir(), name), default)


def _save_json(name: str, data) -> None:
    safe_write_json(os.path.join(get_root_dir(), name), data, indent=2)


def _use_system_certs() -> None:
    """Usa os certificados do sistema (Windows), como o navegador; sem isso o Python recusa alguns sites."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def resolve_video_path(videos_dir: str, name: str) -> str:
    """Vídeo da pasta escolhida ou, se não estiver lá, da pasta única de vídeos (Dev/data/videos)."""
    p = os.path.join(videos_dir, name)
    if os.path.isfile(p):
        return p
    alt = os.path.join(get_videos_dir(), name)
    return alt if os.path.isfile(alt) else p


# ------------------------------------------------------------------------------
# 1. Regras: texto oficial → fatos com evidência
# ------------------------------------------------------------------------------

def decode_html(data: bytes, ctype: str = "") -> str:
    """Bytes da página em texto: charset do cabeçalho ou da <meta>; páginas japonesas antigas usam Shift_JIS/EUC-JP."""
    m = re.search(r"charset=[\"']?([\w-]+)", ctype or "", re.I) or re.search(rb"charset=[\"']?([\w-]+)", data[:4000], re.I)
    declared = m.group(1) if m else None
    if isinstance(declared, bytes):
        declared = declared.decode("ascii", "ignore")
    for enc in [e for e in (declared, "utf-8", "cp932", "euc_jp") if e]:
        try:
            return data.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return data.decode("utf-8", errors="replace")


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>|</tr>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    import html as _html
    text = _html.unescape(text)
    return re.sub(r"[ \t\r\f\v]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()


def pdf_to_text(data: bytes) -> str:
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def fetch_source_text(url_or_path: str, fetcher: Optional[Callable[[str], Tuple[bytes, str]]] = None) -> Dict[str, Any]:
    """Texto da fonte (PDF ou HTML), com cache em disco por endereço."""
    key = hashlib.sha1(url_or_path.encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(get_root_dir(), "regras_texto")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{key}.txt")
    if os.path.isfile(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return {"texto": f.read(), "cache": True}
    if os.path.isfile(url_or_path):
        with open(url_or_path, "rb") as f:
            data = f.read()
        ctype = "application/pdf" if url_or_path.lower().endswith(".pdf") else "text/html"
    elif fetcher is not None:
        data, ctype = fetcher(url_or_path)
    else:
        import requests
        _use_system_certs()
        resp = requests.get(url_or_path, timeout=60, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 SenpAI-kendo-research/1.0"})
        resp.raise_for_status()
        data, ctype = resp.content, resp.headers.get("Content-Type", "")
    if "pdf" in ctype.lower() or data[:5] == b"%PDF-":
        text = pdf_to_text(data)
    else:
        text = html_to_text(decode_html(data, ctype))
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(text)
    return {"texto": text, "cache": False}


def split_sentences(text: str) -> List[str]:
    flat = re.sub(r"\s+", " ", text)
    flat = re.sub(r"(\w)-\s+(\w)", r"\1-\2", flat)   # "yuko- datotsu" (quebra de linha do PDF) → "yuko-datotsu"
    parts = re.split(r"(?<=[.;:])\s+(?=[A-Z0-9(])|(?<=[。！？])", flat)   # inglês/português e japonês (。)
    return [p.strip() for p in parts if (10 if re.search(r"[\u3040-\u9fff]", p) else 20) <= len(p.strip()) <= 600]


_NUM = r"(\d+(?:[.,]\d+)?|one|two|three|four|five|six|ten)"
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10}

# Fatos numéricos: (chave, rótulo, regex na frase, unidade). Só valem com o número dentro da frase.
NUMERIC_FACTS = [
    ("tempo_luta_min", "Tempo regulamentar da luta", r"duration (?:for|of)\s+shiai[^.]*?" + _NUM + r"\s*minutes?", "min"),
    ("encho_min", "Tempo do encho (prorrogação)", r"encho\s+shall\s+be\s+" + _NUM + r"\s*minutes?", "min"),
    ("shiaijo_lado_m", "Lado do shiaijo", r"shiai-?jo[^.]*?sides?\s+of\s+(\d+(?:[.,]\d+)?)(?:\s*to\s*(\d+(?:[.,]\d+)?))?\s*(?:m|meters|metres)\b", "m"),
    ("pontos_para_vencer", "Pontos para vencer (sanbon-shobu)", r"sanbon-?shobu,?\s+the\s+shiai-?sha\s+who\s+first\s+scores\s+" + _NUM + r"\s+points?", "pontos"),
    ("hansoku_para_ponto", "Hansoku que dão um ponto ao oponente", r"commits\s+" + _NUM + r"\s+hansoku[^.]*?opponent\s+shall\s+be\s+given\s+(?:one|1)\s+point", "hansoku"),
    ("spine_tilt_max_deg", "Inclinação máxima da coluna", r"(?:spine|posture|shisei)[^.]*?" + _NUM + r"\s*(?:°|degrees?)", "deg"),
    ("fumikomi_hand_foot_window_ms", "Janela mão-pé do fumikomi", r"(?:fumikomi|ki-ken-tai)[^.]*?" + _NUM + r"\s*(?:ms|milliseconds?)", "ms"),
]

# Fatos textuais: frases que definem regras (referência e dúvidas de regra na curadoria)
TEXT_RULES = [
    ("yuko_datotsu", "Definição de yuko-datotsu", r"yuko[- ]datotsu\s+(?:is|shall be|means|refers)"),
    ("datotsu_bui", "Alvos válidos (datotsu-bui)", r"datotsu[- ]bui"),
    ("zanshin", "Zanshin", r"\bzanshin\b"),
    ("hansoku", "Faltas (hansoku)", r"\bhansoku\b[^.]*?(?:shall|is|are|will)"),
    ("jogai", "Saída do shiaijo (jogai)", r"(?:outside the (?:shiaijo|court|boundary)|jogai)"),
    ("tsubazeriai", "Tsubazeriai", r"tsuba[- ]?zeri[- ]?ai"),
    ("aiuchi", "Golpes simultâneos (aiuchi)", r"(?:aiuchi|simultaneous(?:ly)? strik)"),
    ("invalidacao", "Invalidação de ponto", r"(?:shall|will) (?:not|be) (?:be )?(?:considered|regarded) (?:as )?(?:yuko|valid|invalid)"),
]

BIOMECHANICAL_KEYS = {"spine_tilt_max_deg", "fumikomi_hand_foot_window_ms"}


def _to_number(s: str) -> Optional[float]:
    s = s.lower().strip()
    if s in _WORD_NUM:
        return float(_WORD_NUM[s])
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def extract_rule_facts(text: str, max_quotes_per_rule: int = 5) -> Dict[str, Any]:
    """Fatos com a frase de origem como evidência. Nada é preenchido por padrão."""
    facts: Dict[str, Dict[str, Any]] = {}
    rules: Dict[str, Dict[str, Any]] = {}
    for sent in split_sentences(text):
        low = sent.lower()
        for key, label, pattern, unit in NUMERIC_FACTS:
            if key in facts:
                continue
            m = re.search(pattern, low)
            if not m:
                continue
            nums = [_to_number(g) for g in m.groups() if g]
            if not nums or any(n is None for n in nums):
                continue
            value = {"min": nums[0], "max": nums[1]} if len(nums) == 2 else nums[0]
            facts[key] = {"rotulo": label, "valor": value, "unidade": unit, "evidencia": sent}
        for key, label, pattern in TEXT_RULES:
            if re.search(pattern, low):
                r = rules.setdefault(key, {"rotulo": label, "trechos": []})
                if len(r["trechos"]) < max_quotes_per_rule and sent not in r["trechos"]:
                    r["trechos"].append(sent)
    return {"fatos": facts, "regras": rules}


def verify_llm_facts(proposed: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Aceita um fato do LLM só se a citação existir no texto e contiver o número informado."""
    norm = re.sub(r"\s+", " ", text).lower()
    accepted = {}
    for key, f in (proposed or {}).items():
        quote = re.sub(r"\s+", " ", str(f.get("evidencia", ""))).strip().lower()
        val = f.get("valor")
        if len(quote) < 15 or quote not in norm:
            continue
        if isinstance(val, (int, float)) and not re.search(rf"\b{re.escape(str(val).rstrip('0').rstrip('.'))}\b", quote):
            continue
        accepted[key] = dict(f, verificado=True)
    return accepted


def collect_rules(config: Optional[Dict[str, Any]] = None, fetcher=None, llm=None) -> Dict[str, Any]:
    """Baixa e extrai todas as fontes de regras; grava em regras.json e injeta restrições biomecânicas."""
    from src.engine.actionable_research import SourceHierarchyResolver
    cfg = config or load_config()
    resolver = SourceHierarchyResolver()
    store = _load_json("regras.json", {"fontes": {}})
    report = []
    for src in cfg.get("regras", []):
        url, title = src.get("url"), src.get("titulo") or src.get("url")
        try:
            fetched = fetch_source_text(url, fetcher=fetcher)
        except Exception as e:
            report.append({"fonte": title, "status": "falhou", "detalhe": f"{type(e).__name__}: {e}"})
            continue
        text = fetched["texto"]
        extracted = extract_rule_facts(text)
        if llm is not None and getattr(llm, "is_online", False):
            try:
                proposed = llm.extract_rule_facts(text[:12000], title) if hasattr(llm, "extract_rule_facts") else None
                for k, f in verify_llm_facts((proposed or {}).get("fatos", {}), text).items():
                    extracted["fatos"].setdefault(k, f)
            except Exception as e:
                log_event("WARNING", f"LLM indisponível para {title}: {e}", "internet_collector")
        tier = resolver.classify_source({"title": title, "url": url, "authority": src.get("autoridade")})
        store["fontes"][url] = {"titulo": title, "autoridade": tier.display_name, "nivel": tier.priority,
                                "coletado_em": datetime.datetime.now().isoformat(timespec="seconds"),
                                "caracteres": len(text), **extracted}
        add_to_corpus(url, title, extracted, tier.display_name, tier.priority)
        log_activity("carregado", f"📜 Regulamento lido: {title} — {len(extracted['fatos'])} fato(s)", url)
        report.append({"fonte": title, "status": "ok" if text.strip() else "vazio",
                       "fatos": len(extracted["fatos"]), "regras": len(extracted["regras"]), "cache": fetched["cache"]})
    _save_json("regras.json", store)
    injected = inject_biomechanical_facts(store)
    return {"fontes": report, "restricoes_injetadas": injected}


def inject_biomechanical_facts(store: Dict[str, Any]) -> List[str]:
    """Restrições biomecânicas com evidência viram prior do calibrador (hierarquia de fontes decide conflitos)."""
    constraints_map: Dict[str, Any] = {}
    for url, src in store.get("fontes", {}).items():
        cons = {}
        for key, f in src.get("fatos", {}).items():
            if key not in BIOMECHANICAL_KEYS:
                continue
            if key == "fumikomi_hand_foot_window_ms":
                v = f["valor"]
                cons[key] = v if isinstance(v, dict) else {"min": -v, "max": v, "ideal": 0.0, "unit": "ms"}
            else:
                cons[key] = f["valor"]
        if cons:
            constraints_map[f"internet::{src['titulo']}"] = {
                "concept": "general_strike_biomechanics", "source": src["titulo"], "url": url,
                "authority_tier": src.get("nivel", 3), "authority_name": src.get("autoridade"),
                "constraints": cons, "evidencias": {k: src["fatos"][k]["evidencia"] for k in cons},
            }
    if not constraints_map:
        return []
    from src.engine.actionable_research import BayesianPriorInjector
    from src.utils.test_isolation import get_safe_data_path
    BayesianPriorInjector(get_safe_data_path("config/ai_knowledge_base.json")).inject_constraints_into_knowledge_base(constraints_map)
    return sorted(constraints_map)


def load_rules() -> Dict[str, Any]:
    return _load_json("regras.json", {"fontes": {}})


# ------------------------------------------------------------------------------
# 2. Vídeos: busca, catálogo, rótulo fraco pelo título, download controlado
# ------------------------------------------------------------------------------

# Ordem importa: frases específicas antes das genéricas ("shiai geiko" é treino, "shiai" sozinho é luta)
TITLE_MODALITY_KEYWORDS = [
    ("shiai_geiko", [r"shiai[- ]?geiko"]),
    ("nihon_kendo_kata", [r"kendo[- ]?kata", r"nihon[- ]?kendo[- ]?kata", r"\bkata\b"]),
    ("bokuto_kihon_waza", [r"bokuto(?:[- ]ni[- ]yoru)?[- ]kihon"]),
    ("kirikaeshi", [r"kiri[- ]?kaeshi"]),
    ("kakari_geiko", [r"kakari[- ]?geiko"]),
    ("uchikomi_geiko", [r"uchi[- ]?komi"]),
    ("yakusoku_geiko", [r"yakusoku[- ]?geiko"]),
    ("waza_geiko", [r"waza[- ]?geiko"]),
    ("oji_waza", [r"oji[- ]?waza"]),
    ("ji_geiko", [r"\bji[- ]?geiko"]),
    ("ashi_sabaki", [r"ashi[- ]?sabaki", r"footwork"]),
    ("suburi", [r"suburi"]),
    ("shinsa", [r"shinsa", r"\bexam(?:e|ination)?\b", r"\bdan test"]),
    ("kihon", [r"\bkihon\b"]),
    ("luta", [r"\bshiai\b", r"champion(?:ship)?s?", r"taikai", r"\bfinals?\b", r"semi[- ]?final", r"\bwkc\b",
              r"campeonato", r"\bmatch(?:es)?\b", r"tournament"]),
]


def title_modality(title: str) -> Optional[str]:
    t = (title or "").lower()
    for key, patterns in TITLE_MODALITY_KEYWORDS:
        if any(re.search(p, t) for p in patterns):
            return key
    return None


def search_videos(query: str, limit: int, searcher: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None) -> List[Dict[str, Any]]:
    """Metadados dos resultados (sem baixar)."""
    if searcher is not None:
        return searcher(query, limit)
    import yt_dlp
    _use_system_certs()
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": "in_playlist", "noplaylist": True}) as ydl:
        info = ydl.extract_info(f"ytsearch{int(limit)}:{query}", download=False)
    out = []
    for e in (info or {}).get("entries", []) or []:
        if not e or not e.get("id"):
            continue
        out.append({"id": e["id"], "url": e.get("url") or f"https://www.youtube.com/watch?v={e['id']}",
                    "titulo": e.get("title", ""), "canal": e.get("channel") or e.get("uploader"),
                    "duracao_s": e.get("duration"), "licenca": e.get("license")})
    return out


def load_catalog() -> Dict[str, Dict[str, Any]]:
    return _load_json("catalogo.json", {})


def _dir_size_gb(d: str) -> float:
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def fetch_license(url: str) -> Optional[str]:
    """Licença declarada do vídeo (a busca rápida não traz); None se não informada."""
    import yt_dlp
    _use_system_certs()
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return (info or {}).get("license")


def _license_ok(item: Dict[str, Any], only_cc: bool) -> bool:
    return (not only_cc) or "creative commons" in str(item.get("licenca") or "").lower()


def download_order(catalog: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Ordem de download dos candidatos. Com "prioridade_luta" (padrão), lutas (shiai) vêm primeiro: são elas que
    têm golpes para revisar e pontos para o gabarito. Depois, a modalidade com menos vídeos baixados, para equilibrar.
    """
    baixados: Dict[str, int] = {}
    for it in catalog.values():
        if it.get("status") == "baixado":
            m = it.get("modalidade_titulo") or "?"
            baixados[m] = baixados.get(m, 0) + 1
    luta_first = bool(cfg.get("prioridade_luta", True))
    order = list(catalog.items())
    order.sort(key=lambda kv: (0 if luta_first and kv[1].get("modalidade_titulo") == "luta" else 1,
                               baixados.get(kv[1].get("modalidade_titulo") or "?", 0)))
    return order


def collect_videos(config: Optional[Dict[str, Any]] = None, searcher=None, downloader=None, license_reader=None,
                   allow_download: Optional[bool] = None, history_path: Optional[str] = None,
                   progress: Optional[Callable[[str], None]] = None, depth: int = 1,
                   should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """
    Busca e cataloga; baixa (se permitido) e grava rótulos de origem "internet" no Banco Único.
    depth multiplica quantos resultados por consulta são lidos (a sessão com prazo vai mais fundo a cada rodada).
    should_stop é consultado entre consultas e entre downloads (desligar a procura).
    """
    stop = should_stop or (lambda: False)
    cfg = (config or load_config())["videos"]
    catalog = load_catalog()
    try:
        from src.engine.web_research import get_llm
        video_llm = get_llm()
    except Exception:
        video_llm = None
    new = 0
    per_query = min(MAX_SEARCH_DEPTH, int(cfg.get("max_por_consulta", 5)) * max(1, int(depth)))
    for q in cfg.get("consultas", []):
        if stop():
            break
        if progress:
            progress(f"Buscando no YouTube: {q}")
        log_activity("video_busca", f"🎬 Pesquisando no YouTube: “{q}”")
        try:
            results = search_videos(q, per_query, searcher=searcher)
        except Exception as e:
            log_event("WARNING", f"Busca falhou ({q}): {e}", "internet_collector")
            continue
        for r in results:
            if r["id"] in catalog:
                continue
            mod = title_modality(r.get("titulo", ""))
            if mod is None and video_llm is not None:
                try:
                    from src.engine.web_research import ai_classify_video
                    mod = ai_classify_video(video_llm, r.get("titulo", ""))
                except Exception:
                    mod = None
            log_activity("video_novo", f"🎞️ Novo vídeo: {r.get('titulo', '')} → {mod or 'sem modalidade reconhecida'}", r.get("url"))
            catalog[r["id"]] = dict(r, consulta=q, modalidade_titulo=mod, status="candidato",
                                    encontrado_em=datetime.datetime.now().isoformat(timespec="seconds"))
            new += 1
    _save_json("catalogo.json", catalog)

    do_download = cfg.get("baixar", False) if allow_download is None else allow_download
    downloaded, labels, doubts = [], 0, 0
    if do_download:
        max_s = float(cfg.get("max_duracao_min", 20)) * 60
        budget = float(cfg.get("limite_disco_gb", 10.0))
        for vid, item in download_order(catalog, cfg):
            if stop() or len(downloaded) >= int(cfg.get("max_downloads_por_ciclo", 3)):
                break
            if item["status"] != "candidato":
                continue
            if not item.get("modalidade_titulo"):
                item.update(status="rejeitado", motivo="título sem modalidade reconhecível")
                continue
            if item.get("duracao_s") and item["duracao_s"] > max_s:
                item.update(status="rejeitado", motivo=f"mais longo que {cfg.get('max_duracao_min')} min")
                continue
            only_cc = bool(cfg.get("somente_creative_commons", True))
            if only_cc and item.get("licenca") is None:
                try:
                    item["licenca"] = (license_reader or fetch_license)(item["url"]) or "não informada"
                except Exception as e:
                    item.update(status="falhou", motivo=f"licença ilegível: {e}")
                    continue
            if not _license_ok(item, only_cc):
                item.update(status="rejeitado", motivo="licença não é Creative Commons")
                continue
            if _internet_bytes() / 1e9 >= budget:
                log_event("INFO", f"Limite de disco da coleta atingido ({budget} GB).", "internet_collector")
                break
            if progress:
                progress(f"Baixando: {item['titulo']}")
            try:
                path = _download(item, downloader)
            except Exception as e:
                item.update(status="falhou", motivo=f"{type(e).__name__}: {e}")
                continue
            item.update(status="baixado", arquivo=os.path.basename(path), baixado_em=datetime.datetime.now().isoformat(timespec="seconds"))
            log_activity("download", f"⬇️ Baixado: {item['titulo']} → data/videos/{item['arquivo']}", item.get("url"))
            downloaded.append(item["arquivo"])
            labels += label_video(item, path)
        _save_json("catalogo.json", catalog)
    status_count: Dict[str, int] = {}
    for it in catalog.values():
        status_count[it["status"]] = status_count.get(it["status"], 0) + 1
    return {"novos_no_catalogo": new, "catalogo": status_count, "baixados": downloaded, "rotulos": labels,
            "duvidas": doubts, "download_ligado": bool(do_download)}


def _download(item: Dict[str, Any], downloader=None) -> str:
    if downloader is not None:
        return downloader(item, get_videos_dir())
    from src.utils.video_downloader import download_video_stream
    from src.utils.file_utils import safe_replace
    _use_system_certs()
    max_s = None
    # Baixa numa pasta temporária e só então move para a pasta de vídeos, já com o nome final
    staging = os.path.join(get_root_dir(), "baixando")
    os.makedirs(staging, exist_ok=True)
    path, _info = download_video_stream(item["url"], output_dir=staging, quality="media", max_duration_seconds=max_s)
    target = os.path.join(get_videos_dir(), f"internet_{item['id']}.mp4")
    safe_replace(path, target)
    return target


def label_video(item: Dict[str, Any], path: str) -> int:
    """Rótulo de modalidade pelo título (fraco). Os golpes do vídeo só ganham validade quando alguém revisa."""
    from src.engine.label_bank import LabelBank
    labels = [{"origin": INTERNET_ORIGIN, "video_name": os.path.basename(path), "claim": "modalidade",
               "value": item["modalidade_titulo"], "time_sec": 0.0, "fim_s": None, "kenshi": None, "ref": "titulo",
               "confidence": TITLE_LABEL_CONFIDENCE, "fonte": item.get("url"), "titulo": item.get("titulo")}]
    LabelBank().add_labels(labels)
    return len(labels)


# ------------------------------------------------------------------------------
# 2b. Registro de atividade (o que está sendo pesquisado e carregado, para a tela)
# ------------------------------------------------------------------------------

ACTIVITY_FILE = "atividade.jsonl"
ACTIVITY_KEEP = 400


def log_activity(tipo: str, msg: str, url: Optional[str] = None) -> None:
    """Uma linha do que a coleta está fazendo agora (busca, leitura, carregado, descartado, vídeo, erro)."""
    p = os.path.join(get_root_dir(), ACTIVITY_FILE)
    entry = {"em": datetime.datetime.now().isoformat(timespec="seconds"), "tipo": tipo, "msg": msg}
    if url:
        entry["url"] = url
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if os.path.getsize(p) > 400_000:
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()[-ACTIVITY_KEEP:]
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(lines)
    except OSError:
        pass


def load_activity(limit: int = 30) -> List[Dict[str, Any]]:
    """Últimas atividades, da mais recente para a mais antiga."""
    p = os.path.join(get_root_dir(), ACTIVITY_FILE)
    if not os.path.isfile(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()[-limit:]
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


# ------------------------------------------------------------------------------
# 2c. Textos da internet → Corpus Técnico
# ------------------------------------------------------------------------------

def add_to_corpus(url: str, title: str, analysis: Dict[str, Any], tier_name: str, tier_level: int) -> Optional[str]:
    """Fonte coletada vira item do Corpus Técnico (base de conhecimento), com os fatos e trechos citados."""
    from src.utils.test_isolation import get_safe_data_path
    kb_path = get_safe_data_path("config/ai_knowledge_base.json")
    if not os.path.isfile(kb_path):
        try:
            from src.engine.auto_trainer import AutoTrainingEngine
            AutoTrainingEngine()  # cria a base com o conteúdo de referência
        except Exception:
            return None
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
    except Exception:
        return None
    rules = []
    for fct in analysis.get("fatos", {}).values():
        v = fct["valor"]
        val = f"{v['min']:g} a {v['max']:g}" if isinstance(v, dict) else f"{v:g}"
        rules.append(f"{fct['rotulo']}: {val} {fct['unidade']} — “{fct['evidencia'][:220]}”")
    for r in analysis.get("regras", {}).values():
        for t in r.get("trechos", [])[:2]:
            rules.append(f"{r['rotulo']}: “{t[:220]}”")
    kind = "official_manual" if tier_level <= 2 else ("wikipedia" if "wikipedia.org" in url else "web_article")
    key = "web_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    kb.setdefault("sources", {})[key] = {
        "title": title, "type": kind, "url": url, "authority": tier_name,
        "focus": ", ".join(list(analysis.get("regras", {}))[:3]) or "Geral",
        "key_rules": rules[:10], "summary": analysis.get("resumo", ""),
        "coletado_em": datetime.datetime.now().isoformat(timespec="seconds"), "origem": "coleta_internet",
        "modo_extracao": analysis.get("modo", "regras"),
    }
    kb["total_web_sources_indexed"] = len(kb["sources"])
    safe_write_json(kb_path, kb, indent=2)
    return key


def revalidate_sources(fetcher=None) -> List[str]:
    """
    Reaplica o filtro atual às páginas que a pesquisa já levou ao Corpus e tira as que não passam mais (ex.: a
    lista de capítulos de um mangá que citava kendo de passagem). Só mexe no que a coleta pesquisou; as fontes
    oficiais configuradas (regulamentos da FIK) não passam por aqui. Devolve os títulos removidos.
    """
    from src.engine import web_research as wr
    store = _load_json("regras.json", {"fontes": {}})
    seen = _load_json("textos.json", {})
    removed: Dict[str, str] = {}
    for url, src in list(store.get("fontes", {}).items()):
        if not src.get("origem"):
            continue
        reason = wr.title_rejection(src.get("titulo", ""))
        if not reason:
            try:
                reason = wr.kendo_focus(fetch_source_text(url, fetcher=fetcher)["texto"], src.get("titulo", ""))["motivo"]
            except Exception:
                continue   # sem o texto não dá para julgar de novo: fica como está
        if reason:
            store["fontes"].pop(url)
            seen[url] = {"status": "descartado", "titulo": src.get("titulo", ""), "motivo": reason}
            removed[url] = reason
    if not removed:
        return []
    _save_json("regras.json", store)
    _save_json("textos.json", seen)
    try:
        from src.utils.test_isolation import get_safe_data_path
        kb_path = get_safe_data_path("config/ai_knowledge_base.json")
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        keys = {"web_" + hashlib.sha1(u.encode("utf-8")).hexdigest()[:12] for u in removed}
        kb["sources"] = {k: v for k, v in kb.get("sources", {}).items() if k not in keys}
        safe_write_json(kb_path, kb, indent=2)
    except Exception:
        pass
    titles = []
    for url, reason in removed.items():
        title = seen[url]["titulo"]
        titles.append(title)
        log_activity("descartado", f"🧹 Retirado do Corpus pelo filtro novo ({reason}): {title}", url)
    return titles


def collect_texts(config: Optional[Dict[str, Any]] = None, depth: int = 1, searcher=None, fetcher=None, llm="auto",
                  should_stop: Optional[Callable[[], bool]] = None, progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Pesquisa textos de kendo na web, lê as páginas e leva ao Corpus só o que é sobre kendo, com fatos e
    trechos citados. Com IA, ela cria as consultas e filtra; sem IA, consultas da configuração e regras.
    """
    from src.engine import web_research as wr
    from src.engine.actionable_research import SourceHierarchyResolver
    stop = should_stop or (lambda: False)
    cfg = (config or load_config()).get("textos", {})
    if not cfg.get("ativar", True):
        return {"consultas": 0, "lidas": 0, "carregadas": 0, "descartadas": 0}
    if llm == "auto":
        llm = wr.get_llm()
    revalidate_sources(fetcher=fetcher)
    seen = _load_json("textos.json", {})
    store = _load_json("regras.json", {"fontes": {}})
    resolver = SourceHierarchyResolver()
    per_round = int(cfg.get("consultas_por_rodada", 4))
    queries: List[str] = []
    if llm is not None:
        covered = [s.get("titulo", "") for s in store.get("fontes", {}).values()]
        try:
            queries = wr.ai_queries(llm, covered, per_round)
            if queries:
                log_activity("ia", "IA propôs as buscas: " + " · ".join(queries))
        except Exception as e:
            log_activity("erro", f"IA não respondeu ({e}); usando as consultas da configuração.")
    if not queries:
        base = cfg.get("consultas") or wr.DEFAULT_TEXT_QUERIES
        start = ((max(1, depth) - 1) * per_round) % len(base)
        queries = [base[(start + i) % len(base)] for i in range(min(per_round, len(base)))]
    counts = {"consultas": 0, "lidas": 0, "carregadas": 0, "descartadas": 0}
    for q in queries:
        if stop():
            break
        counts["consultas"] += 1
        if progress:
            progress(f"Pesquisando textos: {q}")
        log_activity("busca", f"🔎 Pesquisando na web: “{q}”")
        try:
            results = (searcher or wr.search_web)(q, int(cfg.get("paginas_por_consulta", 3)))
        except Exception as e:
            log_activity("erro", f"Busca falhou: {e}")
            continue
        for r in results:
            if stop():
                break
            url = r["url"]
            if url in seen:
                continue
            reason = wr.title_rejection(r["titulo"])
            if reason:
                seen[url] = {"status": "descartado", "titulo": r["titulo"], "motivo": reason}
                counts["descartadas"] += 1
                log_activity("descartado", f"🚫 Não é kendo ({reason}): {r['titulo']}", url)
                continue
            if progress:
                progress(f"Lendo: {r['titulo']}")
            log_activity("pagina", f"📄 Lendo ({r.get('fonte', 'web')}): {r['titulo']}", url)
            try:
                text = fetch_source_text(url, fetcher=fetcher)["texto"]
            except Exception as e:
                seen[url] = {"status": "falhou", "titulo": r["titulo"], "motivo": str(e)[:200]}
                log_activity("erro", f"Não foi possível ler: {r['titulo']} ({type(e).__name__})", url)
                continue
            counts["lidas"] += 1
            analysis = wr.analyze_page(text, r["titulo"], llm=llm)
            if not analysis["relevante"]:
                seen[url] = {"status": "descartado", "titulo": r["titulo"], "termos_kendo": analysis["termos_kendo"],
                             "motivo": analysis.get("motivo", "")}
                counts["descartadas"] += 1
                log_activity("descartado", f"🚫 Não é kendo: {r['titulo']} ({analysis.get('motivo') or 'fora do tema'})", url)
                continue
            tier = resolver.classify_source({"title": r["titulo"], "url": url})
            store.setdefault("fontes", {})[url] = {
                "titulo": r["titulo"], "autoridade": tier.display_name, "nivel": tier.priority, "origem": r.get("fonte", "web"),
                "coletado_em": datetime.datetime.now().isoformat(timespec="seconds"), "caracteres": len(text),
                "consulta": q, "resumo": analysis["resumo"], "modo": analysis["modo"],
                "fatos": analysis["fatos"], "regras": analysis["regras"]}
            add_to_corpus(url, r["titulo"], analysis, tier.display_name, tier.priority)
            seen[url] = {"status": "carregado", "titulo": r["titulo"]}
            counts["carregadas"] += 1
            n_quotes = sum(len(x.get("trechos", [])) for x in analysis["regras"].values())
            log_activity("carregado", f"✅ Adicionado ao Corpus: {r['titulo']} — {len(analysis['fatos'])} fato(s), "
                                      f"{n_quotes} trecho(s) de regra ({'IA' if analysis['modo'] == 'ia' else 'regras'})", url)
        _save_json("textos.json", seen)
        _save_json("regras.json", store)
    inject_biomechanical_facts(store)
    return counts


# ------------------------------------------------------------------------------
# 3. Sessão de procura com prazo (ligar por X horas / desligar)
# ------------------------------------------------------------------------------

import threading  # noqa: E402

_session_lock = threading.RLock()
_session_stop = threading.Event()
_session_thread: Optional[threading.Thread] = None
SESSION_THREAD_NAME = "senpai-internet-session"
# A espera entre rodadas é feita em fatias curtas conferindo o relógio: no Windows, a espera de um Event não
# conta o tempo em que o PC ficou suspenso, e uma espera de 15 min podia durar a noite toda.
WAIT_SLICE_SEC = 5.0


def _live_session_thread() -> Optional[threading.Thread]:
    """A thread da procura, mesmo que este módulo tenha sido recarregado (o Streamlit recarrega ao editar o código)."""
    if _session_thread is not None and _session_thread.is_alive():
        return _session_thread
    return next((t for t in threading.enumerate() if t.name == SESSION_THREAD_NAME and t.is_alive()), None)


def _keep_awake(on: bool) -> None:
    """Pede ao Windows para não suspender o PC enquanto a procura roda (a tela pode apagar). Sem efeito fora do Windows."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def _wait_until(stop_event: threading.Event, deadline: datetime.datetime) -> None:
    """Espera até o horário (relógio de parede) ou até desligarem."""
    while not stop_event.is_set():
        left = (deadline - _now()).total_seconds()
        if left <= 0:
            return
        stop_event.wait(min(WAIT_SLICE_SEC, left))


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def format_seconds_hms(seconds: float) -> str:
    s = int(max(0, seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {sec:02d}s"
    return f"{m:02d}m {sec:02d}s"


def compute_session_progress(sess: Dict[str, Any]) -> Dict[str, Any]:
    inicio_str = sess.get("inicio")
    fim_str = sess.get("fim_previsto")
    now = _now()

    if not inicio_str:
        return {
            "tempo_decorrido_segundos": 0.0,
            "tempo_restante_segundos": 0.0,
            "duracao_total_segundos": 0.0,
            "percentual_avanco": 0.0,
            "tempo_decorrido_fmt": "00m 00s",
            "tempo_restante_fmt": "00m 00s",
            "hora_termino_fmt": str(fim_str or "")[11:16] or "—",
        }

    try:
        inicio_dt = datetime.datetime.fromisoformat(inicio_str)
    except Exception:
        inicio_dt = now

    if fim_str:
        try:
            fim_dt = datetime.datetime.fromisoformat(fim_str)
        except Exception:
            fim_dt = inicio_dt + datetime.timedelta(hours=float(sess.get("horas_planejadas", 2.0)))
    else:
        fim_dt = inicio_dt + datetime.timedelta(hours=float(sess.get("horas_planejadas", 2.0)))

    if not sess.get("ativa") and sess.get("encerrada_em"):
        try:
            fim_efetivo = datetime.datetime.fromisoformat(sess["encerrada_em"])
        except Exception:
            fim_efetivo = now
        elapsed_sec = max(0.0, (fim_efetivo - inicio_dt).total_seconds())
    else:
        elapsed_sec = max(0.0, (now - inicio_dt).total_seconds())

    total_sec = max(1.0, (fim_dt - inicio_dt).total_seconds())
    remaining_sec = max(0.0, (fim_dt - now).total_seconds()) if sess.get("ativa") else 0.0
    pct = min(100.0, max(0.0, (elapsed_sec / total_sec) * 100.0))

    return {
        "tempo_decorrido_segundos": elapsed_sec,
        "tempo_restante_segundos": remaining_sec,
        "duracao_total_segundos": total_sec,
        "percentual_avanco": round(pct, 1),
        "tempo_decorrido_fmt": format_seconds_hms(elapsed_sec),
        "tempo_restante_fmt": format_seconds_hms(remaining_sec),
        "hora_termino_fmt": str(fim_str or "")[11:16] or "—",
    }


def get_session() -> Dict[str, Any]:
    """Estado da sessão (persistido em sessao.json). Sessão marcada ativa sem thread viva = interrompida."""
    with _session_lock:
        sess = _load_json("sessao.json", {"ativa": False})
        thread = _live_session_thread()
        alive = thread is not None
        if sess.get("ativa") and not alive:
            sess.update(ativa=False, etapa=None, encerrada_em=sess.get("encerrada_em") or _now().isoformat(timespec="seconds"),
                        encerrada_por="interrompida (o app foi reiniciado)")
            _save_json("sessao.json", sess)
        elif sess.get("ativa") and _past_deadline(sess):
            # Prazo vencido e a thread ainda viva (ex.: terminando um download): pede para parar e mostra isso
            getattr(thread, "stop_event", _session_stop).set()
            sess["etapa"] = "prazo encerrado — finalizando"
        telemetry = compute_session_progress(sess)
        sess.update(telemetry)
        return sess


def _past_deadline(sess: Dict[str, Any]) -> bool:
    try:
        return _now() >= datetime.datetime.fromisoformat(sess["fim_previsto"])
    except Exception:
        return False


def _update_session(**kw) -> Dict[str, Any]:
    with _session_lock:
        sess = _load_json("sessao.json", {"ativa": False})
        sess.update(kw)
        _save_json("sessao.json", sess)
        return sess


def run_collection_session(hours: float, allow_download: bool, config: Optional[Dict[str, Any]] = None,
                           stop_event: Optional[threading.Event] = None, interval_min: Optional[float] = None,
                           **collect_kwargs) -> Dict[str, Any]:
    """
    Procura vídeos até o prazo (ou até ser desligada): uma rodada por intervalo, cada rodada lendo mais
    resultados por consulta. O download, se permitido, segue os mesmos filtros e limites de disco.
    """
    stop_event = stop_event or threading.Event()
    cfg = config or load_config()
    interval = float(interval_min if interval_min is not None else cfg["videos"].get("intervalo_sessao_min", 15))
    end = _now() + datetime.timedelta(hours=float(hours))
    _update_session(ativa=True, inicio=_now().isoformat(timespec="seconds"), fim_previsto=end.isoformat(timespec="seconds"),
                    baixar=bool(allow_download), rodadas=0, novos=0, baixados=[], rotulos=0, duvidas=0,
                    textos_lidos=0, textos_carregados=0,
                    etapa="iniciando", encerrada_em=None, encerrada_por=None, erro=None)
    log_event("INFO", f"Procura de vídeos ligada por {hours} h (download {'ligado' if allow_download else 'desligado'}).",
              "internet_collector")
    rounds = 0
    reason = "prazo encerrado"
    _keep_awake(True)
    try:
        while not stop_event.is_set() and _now() < end:
            rounds += 1
            log_activity("rodada", f"▶️ Rodada {rounds} iniciada")
            texts = collect_texts(cfg, depth=rounds, should_stop=lambda: stop_event.is_set() or _now() >= end,
                                  progress=lambda msg: _update_session(etapa=msg),
                                  **{k: v for k, v in collect_kwargs.items() if k in ("fetcher", "llm")},
                                  **({"searcher": collect_kwargs["text_searcher"]} if "text_searcher" in collect_kwargs else {}))
            with _session_lock:
                sess = _load_json("sessao.json", {})
                _update_session(textos_lidos=sess.get("textos_lidos", 0) + texts["lidas"],
                                textos_carregados=sess.get("textos_carregados", 0) + texts["carregadas"])
            if stop_event.is_set():
                break
            video_kwargs = {k: v for k, v in collect_kwargs.items() if k not in ("fetcher", "llm", "text_searcher")}
            res = collect_videos(cfg, allow_download=allow_download, depth=rounds,
                                 should_stop=lambda: stop_event.is_set() or _now() >= end,
                                 progress=lambda msg: _update_session(etapa=msg), **video_kwargs)
            with _session_lock:
                sess = _load_json("sessao.json", {})
                _update_session(rodadas=rounds, novos=sess.get("novos", 0) + res["novos_no_catalogo"],
                                baixados=sess.get("baixados", []) + res["baixados"],
                                rotulos=sess.get("rotulos", 0) + res["rotulos"], duvidas=sess.get("duvidas", 0) + res["duvidas"],
                                etapa=f"aguardando a próxima rodada ({interval:g} min)")
            remaining = (end - _now()).total_seconds()
            if stop_event.is_set() or remaining <= 0:
                break
            _wait_until(stop_event, min(end, _now() + datetime.timedelta(minutes=interval)))
        if stop_event.is_set() and _now() < end:
            reason = "desligada pelo usuário"
    except Exception as e:
        reason = "erro"
        _update_session(erro=f"{type(e).__name__}: {e}")
        log_event("WARNING", f"Sessão de procura parou com erro: {e}", "internet_collector")
    finally:
        _keep_awake(False)
    sess = _update_session(ativa=False, etapa=None, encerrada_em=_now().isoformat(timespec="seconds"), encerrada_por=reason)
    log_event("INFO", f"Procura de vídeos encerrada ({reason}): {sess.get('rodadas')} rodada(s), "
                      f"{sess.get('novos')} novo(s), {len(sess.get('baixados', []))} baixado(s).", "internet_collector")
    return sess


def start_collection_session(hours: float, allow_download: bool, **kwargs) -> bool:
    """Liga a procura em segundo plano por `hours` horas; False se já houver uma ligada."""
    global _session_thread
    if hours <= 0:
        raise ValueError("O prazo precisa ser maior que zero.")
    with _session_lock:
        if _live_session_thread() is not None:
            return False
        _session_stop.clear()
        now_dt = _now()
        end_dt = now_dt + datetime.timedelta(hours=float(hours))
        # Marca como ativa já aqui, para a tela mostrar "ligada" no mesmo instante com prazos calculados
        _update_session(ativa=True, inicio=now_dt.isoformat(timespec="seconds"),
                        fim_previsto=end_dt.isoformat(timespec="seconds"), horas_planejadas=float(hours),
                        baixar=bool(allow_download), etapa="iniciando", encerrada_por=None, erro=None)
        _session_thread = threading.Thread(
            target=run_collection_session, args=(hours, allow_download),
            kwargs=dict(kwargs, stop_event=_session_stop), name=SESSION_THREAD_NAME, daemon=True)
        _session_thread.stop_event = _session_stop   # quem recarregar o módulo ainda consegue desligar esta procura
        _session_thread.start()
    return True


def stop_collection_session(wait_sec: float = 0.0) -> bool:
    """Desliga a procura. Um download em andamento termina antes de parar. False se não havia sessão."""
    thread = _live_session_thread()
    if thread is None:
        return False
    getattr(thread, "stop_event", _session_stop).set()
    if wait_sec:
        thread.join(wait_sec)
    return True


def status() -> Dict[str, Any]:
    cat = load_catalog()
    counts: Dict[str, int] = {}
    for it in cat.values():
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    rules = load_rules()
    return {"catalogo": counts, "disco_gb": round(_internet_bytes() / 1e9, 2), 
            "fontes_regras": {s["titulo"]: {"fatos": len(s.get("fatos", {})), "regras": len(s.get("regras", {}))}
                              for s in rules.get("fontes", {}).values()}}


def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Coleta automática na internet (regras e vídeos).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("regras")
    p_b = sub.add_parser("buscar")
    p_b.add_argument("--baixar", action="store_true", help="Baixa os candidatos permitidos (sobrepõe a configuração).")
    p_s = sub.add_parser("sessao", help="Procura por um tempo determinado (Ctrl+C desliga).")
    p_s.add_argument("--horas", type=float, required=True)
    p_s.add_argument("--baixar", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "sessao":
        ev = threading.Event()
        try:
            print(json.dumps(run_collection_session(args.horas, args.baixar, stop_event=ev), indent=2, ensure_ascii=False))
        except KeyboardInterrupt:
            ev.set()
            print(json.dumps(_update_session(ativa=False, encerrada_por="desligada pelo usuário",
                                             encerrada_em=_now().isoformat(timespec="seconds")), indent=2, ensure_ascii=False))
        return 0
    if args.command == "regras":
        res = collect_rules()
        for r in res["fontes"]:
            print(f"[{r['status']}] {r['fonte']}: " + (r.get("detalhe") or f"{r['fatos']} fato(s), {r['regras']} regra(s)"))
        print(f"Restrições biomecânicas injetadas: {res['restricoes_injetadas'] or 'nenhuma (o texto não traz números biomecânicos)'}")
    elif args.command == "buscar":
        res = collect_videos(allow_download=True if args.baixar else None, progress=print)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(status(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
