"""
Pesquisa de textos de kendo na internet, com IA quando disponível (3º tipo de treinamento).

- Busca: Google (via SerpApi, chave SERPAPI_KEY), DuckDuckGo e Wikipédia. Vídeos: YouTube (pesquisa via
  yt-dlp, em internet_collector). A página de resultados do Google não é raspada (CAPTCHA e termos de
  uso), e a API oficial do Google fechou para novos clientes em 2026 (desligamento em 01/01/2027).
- Chaves: configuradas na tela (src/utils/api_keys.py, arquivo local fora do git) ou por variável de ambiente.
- Leitura: páginas HTML e PDFs (cache em data/internet/regras_texto/).
- IA (KendoLLMAssistant, com GEMINI_API_KEY ou OPENAI_API_KEY):
    * cria as consultas a partir do que o corpus ainda não cobre;
    * decide se a página é sobre kendo e extrai fatos e trechos de regra;
    * classifica títulos de vídeo (luta / modalidade / irrelevante).
  Tudo o que a IA propõe só é aceito se a citação existir literalmente no texto (nada inventado entra).
- Sem chave: consultas da configuração, relevância por termos de kendo e extração por regras. A tela
  mostra qual modo está em uso.
"""

import re
import json
import html as _html
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

KENDO_TERMS = ("kendo", "shinai", "men", "kote", "tsuki", "datotsu", "shinpan", "shiai", "kirikaeshi", "suburi",
               "kamae", "zanshin", "fumikomi", "kiai", "bogu", "dojo", "sensei", "ippon", "hansoku", "kata", "geiko",
               "ashi", "seme", "tenouchi", "hasuji", "sonkyo", "maai", "剣道")
MIN_KENDO_TERMS = 4
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SenpAI-kendo-research/1.0"

DEFAULT_TEXT_QUERIES = [
    "kendo yuko-datotsu definition criteria", "kendo shinpan rules hansoku list", "kendo zanshin after strike referee",
    "kendo ki-ken-tai-ichi explanation", "kendo kirikaeshi purpose method", "kendo kakari geiko uchikomi geiko difference",
    "kendo oji waza techniques list", "kendo tsubazeriai rules", "kendo valid strike men kote do tsuki target",
    "kendo fumikomi footwork technique", "kendo nihon kendo kata", "kendo shinsa grading criteria",
]


# ------------------------------------------------------------------------------
# IA (opcional)
# ------------------------------------------------------------------------------

def get_llm():
    try:
        from src.utils.api_keys import apply_to_env
        apply_to_env()
        from src.engine.llm_assistant import KendoLLMAssistant
        llm = KendoLLMAssistant()
        return llm if llm.is_online else None
    except Exception:
        return None


def ai_status() -> Dict[str, Any]:
    llm = get_llm()
    if llm is None:
        return {"ativa": False, "descricao": "IA desligada: informe a chave do Gemini (ou da OpenAI) em “⚙️ Variáveis do Sistema” para a "
                                             "IA criar as consultas, filtrar e extrair. Sem chave, a coleta usa consultas fixas e regras."}
    return {"ativa": True, "descricao": f"IA ligada ({llm.provider}): cria consultas, filtra páginas e vídeos e extrai fatos citados."}


def _quote_in(quote: str, text: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    q = norm(quote)
    return len(q) >= 15 and q in norm(text)


def ai_queries(llm, covered_topics: List[str], n: int = 6) -> List[str]:
    """Consultas novas sobre o que o corpus ainda não cobre."""
    prompt = (
        "Você ajuda um sistema de análise de kendo a aprender regras, critérios de arbitragem, técnicas e as 14 "
        "modalidades de treino. Temas já cobertos: " + "; ".join(covered_topics[:40]) + ".\n"
        f"Proponha {n} buscas na web (em inglês ou japonês) sobre temas AINDA NÃO cobertos e úteis para julgar golpes "
        'válidos e reconhecer treinos. Responda só JSON: {"consultas": ["...", "..."]}'
    )
    res = llm._call_remote_llm(prompt) or {}
    out = [str(q).strip() for q in res.get("consultas", []) if str(q).strip()]
    return out[:n]


def ai_analyze_page(llm, text: str, title: str) -> Optional[Dict[str, Any]]:
    """Relevância, resumo, fatos e trechos — só fica o que tem citação literal na página."""
    prompt = (
        f"Página: {title}\nTexto (trecho):\n{text[:6000]}\n\n"
        "1) A página é sobre KENDO (regras, arbitragem, técnica ou treino de kendo)? Responda relevante=false se "
        "ela for sobre outra arte marcial (karatê, judô, aikidô, iaidô, kenjutsu, esgrima etc.), mesmo que use "
        "termos como zanshin, kiai, kata ou dojo. 2) Extraia fatos numéricos e trechos de "
        "regra, copiando a frase EXATA do texto como evidência. Responda só JSON: "
        '{"relevante": true, "resumo": "1 frase", "fatos": {"chave": {"rotulo": "...", "valor": 5, "unidade": "min", '
        '"evidencia": "frase exata"}}, "trechos": [{"tema": "zanshin", "trecho": "frase exata"}]}'
    )
    res = llm._call_remote_llm(prompt)
    if not isinstance(res, dict):
        return None
    from src.engine.internet_collector import verify_llm_facts
    facts = verify_llm_facts(res.get("fatos") or {}, text)
    quotes = [t for t in (res.get("trechos") or []) if isinstance(t, dict) and _quote_in(str(t.get("trecho", "")), text)]
    return {"relevante": bool(res.get("relevante")), "resumo": str(res.get("resumo") or "")[:300], "fatos": facts,
            "trechos": [{"tema": str(t.get("tema", "regra"))[:40], "trecho": t["trecho"]} for t in quotes][:12]}


def ai_classify_video(llm, title: str) -> Optional[str]:
    from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA
    keys = ["luta", *TRAINING_MODALITIES_METADATA.keys(), "irrelevante"]
    res = llm._call_remote_llm(f'Título de vídeo: "{title}". Classifique como uma destas chaves: {", ".join(keys)}. '
                               'Responda só JSON: {"classe": "chave"}') or {}
    c = str(res.get("classe", "")).strip()
    return c if c in keys and c != "irrelevante" else None


# ------------------------------------------------------------------------------
# Busca e leitura
# ------------------------------------------------------------------------------

def _clean_ddg_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    q = parse_qs(urlparse(href).query)
    return unquote(q["uddg"][0]) if "uddg" in q else href


def google_status() -> Dict[str, Any]:
    from src.utils.api_keys import get_key
    ok = bool(get_key("SERPAPI_KEY"))
    return {"ativo": ok, "descricao": "Google ligado (via SerpApi)." if ok else
            "Google desligado: informe a chave da SerpApi em “⚙️ Variáveis do Sistema”. A API oficial do Google não aceita "
            "novos clientes desde 2026 (e será desligada em 01/01/2027). Enquanto isso, a busca usa DuckDuckGo e Wikipédia."}


def search_google(query: str, limit: int = 5) -> List[Dict[str, str]]:
    """Resultados do Google pela SerpApi. Sem chave, devolve vazio."""
    import requests
    from src.utils.api_keys import get_key
    key = get_key("SERPAPI_KEY")
    if not key:
        return []
    r = requests.get("https://serpapi.com/search.json",
                     params={"engine": "google", "q": query, "num": max(1, min(10, limit)), "api_key": key}, timeout=30)
    r.raise_for_status()
    return [{"titulo": it.get("title", ""), "url": it.get("link", ""), "trecho": it.get("snippet", ""), "fonte": "google"}
            for it in r.json().get("organic_results", [])[:limit] if it.get("link")]


def search_web(query: str, limit: int = 5, http_get: Optional[Callable] = None) -> List[Dict[str, str]]:
    """Resultados de Google (se configurado) + DuckDuckGo + Wikipédia: [{titulo, url, trecho, fonte}]."""
    import requests
    from src.engine.internet_collector import _use_system_certs
    _use_system_certs()
    out: List[Dict[str, str]] = []
    try:
        out.extend(search_google(query, limit))
    except Exception:
        pass
    try:
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers={"User-Agent": USER_AGENT}, timeout=20)
        for m in re.finditer(r'class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
                             r.text, re.S):
            url = _clean_ddg_url(_html.unescape(m.group(1)))
            if "duckduckgo.com/y.js" in url:   # anúncio
                continue
            out.append({"titulo": _html.unescape(re.sub("<[^>]+>", "", m.group(2))).strip(), "url": url,
                        "trecho": _html.unescape(re.sub("<[^>]+>", "", m.group(3))).strip(), "fonte": "web"})
            if sum(1 for x in out if x["fonte"] == "web") >= limit:
                break
    except Exception:
        pass
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": query,
                                                                      "format": "json", "srlimit": 2},
                         headers={"User-Agent": USER_AGENT}, timeout=20)
        for x in r.json().get("query", {}).get("search", []):
            out.append({"titulo": x["title"], "url": "https://en.wikipedia.org/wiki/" + x["title"].replace(" ", "_"),
                        "trecho": re.sub("<[^>]+>", "", x.get("snippet", "")), "fonte": "wikipedia"})
    except Exception:
        pass
    seen, uniq = set(), []
    for r_ in out:
        if r_["url"] not in seen:
            seen.add(r_["url"])
            uniq.append(r_)
    return uniq


def kendo_relevance(text: str) -> int:
    """Quantos termos distintos de kendo aparecem (relevância sem IA)."""
    low = text.lower()
    return sum(1 for t in KENDO_TERMS if re.search(rf"\b{re.escape(t)}\b", low) or (not t.isascii() and t in low))


KENDO_NAMES = ("kendo", "kendoka", "剣道")
# Outras artes marciais: termos como zanshin, kiai, kata, dojo e sensei também aparecem nelas, então a página
# só conta como kendo se falar de kendo bem mais do que de qualquer uma delas.
OTHER_ARTS = ("karate", "karatê", "karateka", "judo", "judô", "judoka", "aikido", "aikidô", "taekwondo", "jujutsu",
              "jujitsu", "jiu-jitsu", "kung fu", "wushu", "boxing", "boxe", "muay thai", "kickboxing", "sumo", "kyudo",
              "iaido", "iaijutsu", "kenjutsu", "naginata", "jodo", "kobudo", "hapkido", "sambo", "capoeira", "fencing",
              "esgrima", "mma", "wrestling", "ninjutsu", "kenpo", "kempo", "battodo", "空手", "柔道", "合気道", "居合")
MIN_KENDO_MENTIONS = 3


def _count(term: str, low: str) -> int:
    if not term.isascii():
        return low.count(term)
    return len(re.findall(rf"(?<![a-z]){re.escape(term)}(?![a-z])", low))


def kendo_focus(text: str, title: str = "") -> Dict[str, Any]:
    """A página é sobre KENDO (e não sobre outra arte marcial que usa os mesmos termos)?"""
    low, tlow = text.lower(), title.lower()
    kendo = sum(_count(t, low) for t in KENDO_NAMES)
    others = {a: _count(a, low) for a in OTHER_ARTS}
    top_art, top_n = max(others.items(), key=lambda kv: kv[1])
    title_kendo = any(_count(t, tlow) for t in KENDO_NAMES)
    title_other = next((a for a in OTHER_ARTS if _count(a, tlow)), None)
    if title_other and not title_kendo:
        return {"ok": False, "motivo": f"página sobre {title_other}, não kendo", "mencoes_kendo": kendo}
    if kendo < MIN_KENDO_MENTIONS:
        return {"ok": False, "motivo": f"kendo citado só {kendo} vez(es)", "mencoes_kendo": kendo}
    if top_n and kendo < 2 * top_n:
        return {"ok": False, "motivo": f"fala de {top_art} ({top_n}×) quase tanto quanto de kendo ({kendo}×)",
                "mencoes_kendo": kendo}
    return {"ok": True, "motivo": "", "mencoes_kendo": kendo}


def title_is_other_art(title: str) -> Optional[str]:
    """Nome da outra arte marcial se o título for dela e não de kendo (descarta antes de baixar a página)."""
    tlow = title.lower()
    if any(_count(t, tlow) for t in KENDO_NAMES):
        return None
    return next((a for a in OTHER_ARTS if _count(a, tlow)), None)


def key_sentences(text: str, n: int = 3) -> List[str]:
    """As frases do próprio texto com mais termos de kendo (resumo extrativo, sem inventar nada)."""
    from src.engine.internet_collector import split_sentences
    scored = []
    for i, sent in enumerate(split_sentences(text)):
        if not 60 <= len(sent) <= 400:
            continue
        k = kendo_relevance(sent)
        if k >= 2:
            scored.append((k, -i, sent))
    best = sorted(scored, reverse=True)[:n]
    return [s_ for _, _, s_ in sorted(best, key=lambda x: -x[1])]


def analyze_page(text: str, title: str, llm=None) -> Dict[str, Any]:
    """Resultado da análise de uma página: com IA (citações verificadas) ou por regras."""
    from src.engine.internet_collector import extract_rule_facts
    base = extract_rule_facts(text)
    relevance = kendo_relevance(text)
    focus = kendo_focus(text, title)
    motivo = focus["motivo"] or ("" if relevance >= MIN_KENDO_TERMS else f"só {relevance} termos de kendo")
    out = {"relevante": not motivo, "motivo": motivo, "termos_kendo": relevance, "resumo": "", "fatos": base["fatos"],
           "regras": base["regras"], "modo": "regras"}
    if motivo:
        return out   # fora do tema: nem chama a IA
    sentences = key_sentences(text)
    if sentences:
        out["resumo"] = sentences[0][:300]
        out["regras"].setdefault("conteudo", {"rotulo": "Conteúdo", "trechos": sentences})
    if llm is not None:
        ai = ai_analyze_page(llm, text, title)
        if ai is not None:
            out.update(relevante=ai["relevante"], resumo=ai["resumo"], modo="ia")
            if not ai["relevante"]:
                out["motivo"] = "a IA julgou que não é sobre kendo"
            for k, f in ai["fatos"].items():
                out["fatos"].setdefault(k, f)
            for t in ai["trechos"]:
                r = out["regras"].setdefault(t["tema"], {"rotulo": t["tema"].capitalize(), "trechos": []})
                if t["trecho"] not in r["trechos"]:
                    r["trechos"].append(t["trecho"])
    return out
