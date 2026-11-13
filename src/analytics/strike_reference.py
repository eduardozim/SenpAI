"""
Referência dos golpes validados (substitui o antigo "padrão áureo" de números fixos).

O que é: para cada técnica (MEN, KOTE, DO, TSUKI), a distribuição REAL das notas dos pilares nos golpes que o
banco único de rótulos considera válidos. Válido aqui é só o que alguém REVISOU no vídeo (kenshi ou sensei); a
decisão dos shinpan (gabarito e bandeiras) não valida golpe. Cada golpe pesa o dan de quem revisou (um MEN
confirmado por um 7º dan pesa mais que um confirmado por um 1º dan). O ciclo de treino recalcula a referência;
o golden nunca entra.

Para que serve:
1. Retorno por pilar na análise: "zanshin no percentil 18 dos MEN validados" (o que puxa o golpe para baixo).
2. Golpes fora do padrão viram dúvida na curadoria: dado como válido com um pilar abaixo de quase todos os
   validados, ou dado como inválido com todos os pilares na metade de cima dos validados.

O que NÃO faz: não mexe em pesos nem limiares (isso é da calibração, medida no golden) e não inventa nada:
abaixo de MIN_SAMPLES golpes validados (de MIN_VIDEOS vídeos) a técnica aparece como "não medida".
"""

import os
import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir
from src.utils.file_utils import safe_write_json, safe_load_json

DEFAULT_REFERENCE_PATH = os.path.join("data", "strike_reference.json")
TECHNIQUES = ("MEN", "KOTE", "DO", "TSUKI")
PILLARS = ("target_impact", "fumikomi_sync", "posture", "zanshin", "hasuji")
PILLAR_NAMES = {"target_impact": "alvo", "fumikomi_sync": "fumikomi", "posture": "postura", "zanshin": "zanshin",
                "hasuji": "hasuji"}
MIN_SAMPLES = 20          # golpes validados por técnica para a referência valer
MIN_VIDEOS = 3            # de vídeos diferentes (um vídeo só ensinaria o estilo de um kenshi)
LOW_PCT = 5.0             # válido com pilar abaixo do p5 dos validados → dúvida
HIGH_PCT = 50.0           # inválido com TODOS os pilares acima da mediana dos validados → dúvida
SUMMARY_PCTS = (5, 25, 50, 75, 95)


def get_reference_path() -> str:
    return os.path.join(get_test_sandbox_dir(), "strike_reference.json") if is_test_mode() else DEFAULT_REFERENCE_PATH


def _norm(v: Any) -> Optional[float]:
    if not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v / 100.0 if v > 1.0 else v


def _technique(item: Dict[str, Any], report: Dict[str, Any]) -> Optional[str]:
    """Técnica corrigida por revisor > técnica do ponto oficial > técnica que a análise detectou."""
    from src.engine import weak_labels as wl
    if item.get("tecnica_corrigida"):
        return wl.normalize_technique(item["tecnica_corrigida"])
    idx = item.get("matched_point")
    if idx is not None:
        for p in report.get("official_points", []):
            if p.get("index") == idx and p.get("technique"):
                tech = wl.normalize_technique(p["technique"])
                if tech:
                    return tech
    return wl.normalize_technique(item.get("strike_type"))


# ------------------------------------------------------------------------------
# Construção (ciclo de treino)
# ------------------------------------------------------------------------------

def collect_samples(training: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Golpes validados (rótulo final POSITIVO, depois das correções por dan) com as notas dos pilares."""
    from src.engine import weak_labels as wl
    out: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TECHNIQUES}
    for report in training.get("reports", []):
        for item in report.get("events", {}).get("items", []):
            if item.get("weak_label") != wl.POSITIVE:
                continue
            tech = _technique(item, report)
            scores = {p: _norm((item.get("sub_scores") or {}).get(p)) for p in PILLARS}
            scores = {p: v for p, v in scores.items() if v is not None}
            if tech not in out or not scores:
                continue
            out[tech].append({"notas": scores, "peso": float(item.get("autoridade_rotulo") or 1.0),
                              "video": report.get("video_name"), "origem": item.get("origem_rotulo", "revisao")})
    return out


def _weighted_quantile(values: List[Tuple[float, float]], q: float) -> float:
    """values = [(valor, peso)] ordenados; q em 0–100."""
    total = sum(w for _, w in values)
    acc = 0.0
    for v, w in values:
        acc += w
        if acc / total * 100.0 >= q:
            return v
    return values[-1][0]


def build_reference(training: Dict[str, Any]) -> Dict[str, Any]:
    samples = collect_samples(training)
    ref: Dict[str, Any] = {"gerado_em": datetime.datetime.now().isoformat(timespec="seconds"),
                           "minimo": {"golpes": MIN_SAMPLES, "videos": MIN_VIDEOS}, "tecnicas": {}}
    for tech, rows in samples.items():
        videos = {r["video"] for r in rows}
        pillars = {}
        for p in PILLARS:
            vals = sorted((r["notas"][p], r["peso"]) for r in rows if p in r["notas"])
            if not vals:
                continue
            pillars[p] = {"n": len(vals), "valores": [[round(v, 4), round(w, 3)] for v, w in vals],
                          **{f"p{q}": round(_weighted_quantile(vals, q), 4) for q in SUMMARY_PCTS}}
        origins: Dict[str, int] = {}
        for r in rows:
            origins[r["origem"]] = origins.get(r["origem"], 0) + 1
        ref["tecnicas"][tech] = {"golpes": len(rows), "videos": len(videos), "por_origem": origins,
                                 "medido": len(rows) >= MIN_SAMPLES and len(videos) >= MIN_VIDEOS, "pilares": pillars}
    return ref


def save_reference(ref: Dict[str, Any], path: Optional[str] = None) -> str:
    path = path or get_reference_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    safe_write_json(path, ref, indent=1)
    return path


def load_reference(path: Optional[str] = None) -> Dict[str, Any]:
    data = safe_load_json(path or get_reference_path(), default={})
    return data if isinstance(data, dict) else {}


_cache: Dict[str, Any] = {"chave": None, "ref": {}}


def cached_reference() -> Dict[str, Any]:
    """A referência salva, relida só quando o arquivo muda (a tela chama para cada golpe)."""
    path = get_reference_path()
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return {}
    if _cache["chave"] != key:
        _cache.update(chave=key, ref=load_reference(path))
    return _cache["ref"]


def summary_text(ref: Dict[str, Any]) -> str:
    parts = []
    for tech in TECHNIQUES:
        t = ref.get("tecnicas", {}).get(tech, {})
        n, v = t.get("golpes", 0), t.get("videos", 0)
        parts.append(f"{tech} {n} golpe(s)/{v} vídeo(s)" + (" ✅" if t.get("medido") else ""))
    return "; ".join(parts)


# ------------------------------------------------------------------------------
# Uso (análise e curadoria)
# ------------------------------------------------------------------------------

def percentile_rank(ref: Dict[str, Any], tech: str, pillar: str, value: float) -> Optional[float]:
    """Que fração (ponderada) dos golpes validados da técnica tem nota menor que esta (0–100)."""
    t = ref.get("tecnicas", {}).get(tech)
    if not t or not t.get("medido") or pillar not in t.get("pilares", {}):
        return None
    vals = t["pilares"][pillar]["valores"]
    total = sum(w for _, w in vals)
    below = sum(w for v, w in vals if v < value) + 0.5 * sum(w for v, w in vals if v == value)
    return round(100.0 * below / total, 1) if total else None


def compare(strike_type: Any, sub_scores: Dict[str, Any], ref: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from src.engine import weak_labels as wl
    ref = ref if ref is not None else load_reference()
    tech = wl.normalize_technique(strike_type)
    t = ref.get("tecnicas", {}).get(tech or "", {})
    out = {"tecnica": tech, "medido": bool(t.get("medido")), "golpes": t.get("golpes", 0), "videos": t.get("videos", 0),
           "percentis": {}}
    if not out["medido"]:
        return out
    for p in PILLARS:
        v = _norm((sub_scores or {}).get(p))
        if v is not None:
            pct = percentile_rank(ref, tech, p, v)
            if pct is not None:
                out["percentis"][p] = pct
    return out


def describe(strike_type: Any, sub_scores: Dict[str, Any], ref: Optional[Dict[str, Any]] = None) -> str:
    """Uma linha para a tela; sem referência medida, diz quanto falta (nunca mostra número inventado)."""
    c = compare(strike_type, sub_scores, ref)
    tech = c["tecnica"] or str(strike_type or "golpe")
    if not c["medido"]:
        return (f"Referência dos {tech} validados ainda não medida: {c['golpes']} de {MIN_SAMPLES} golpes "
                f"({c['videos']} de {MIN_VIDEOS} vídeos). Cresce com as revisões das análises e a curadoria.")
    if not c["percentis"]:
        return f"Golpe sem as notas dos pilares para comparar com os {c['golpes']} {tech} validados."
    worst = min(c["percentis"], key=c["percentis"].get)
    parts = " · ".join(f"{PILLAR_NAMES[p]} p{c['percentis'][p]:.0f}" for p in PILLARS if p in c["percentis"])
    return (f"Comparado aos {c['golpes']} {tech} validados: {parts}. Ponto mais fraco: {PILLAR_NAMES[worst]} "
            f"(acima de só {c['percentis'][worst]:.0f}% deles).")


def outlier_reason(strike_type: Any, sub_scores: Dict[str, Any], predicted_valid: bool,
                   ref: Optional[Dict[str, Any]] = None) -> Optional[Tuple[str, float]]:
    """(motivo, incerteza) se o golpe destoa dos validados; None se está dentro do padrão ou sem referência."""
    c = compare(strike_type, sub_scores, ref)
    pcts = c["percentis"]
    if not c["medido"] or len(pcts) < 3:
        return None
    tech = c["tecnica"]
    if predicted_valid:
        low = {p: v for p, v in pcts.items() if v < LOW_PCT}
        if low:
            p = min(low, key=low.get)
            return (f"Dado como válido, mas o {PILLAR_NAMES[p]} ficou abaixo de {100 - low[p]:.0f}% dos {c['golpes']} {tech} "
                    "validados. É mesmo ponto?", 0.85)
    elif min(pcts.values()) >= HIGH_PCT:
        return (f"Dado como inválido, mas todos os pilares estão na metade de cima dos {c['golpes']} {tech} validados. "
                "Pode ser um golpe válido que o sistema (ou os shinpan) deixou passar.", 0.8)
    return None
