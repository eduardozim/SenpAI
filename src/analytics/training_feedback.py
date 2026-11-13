"""
Diagnóstico pedagógico do modo de treinamento: análise detalhada e exercícios para forças e fraquezas.

Substitui as regras fixas "nota abaixo do limite → frase fixa → exercício fixo". Para cada métrica medida:
- o que foi visto (valor medido × referência), onde aconteceu (as piores repetições, com o instante no vídeo),
  por que importa (o princípio do kendo), causa provável e gravidade (leve, moderada, prioritária);
- exercícios escolhidos do catálogo editável (config/exercise_catalog.json) pela gravidade e pelo formato
  da modalidade (solo, dupla, bogu), com dose, progressão, pontos de atenção, erros comuns e como medir;
- um exercício que leva adiante o ponto mais forte;
- plano semanal e metas para a próxima sessão.

Métricas que não puderam ser medidas aparecem como "não medido", com o motivo; nada é inventado.
Também mede o Ki-Ken-Tai-Ichi de verdade (o pé do fumikomi × o impacto da espada), que antes era fixo.
"""

import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CATALOG_PATH = os.path.join("config", "exercise_catalog.json")

SEVERITY_LABEL = {"prioritaria": "PRIORITÁRIO", "moderada": "MODERADO", "leve": "LEVE"}
SEVERITY_RANK = {"prioritaria": 0, "moderada": 1, "leve": 2}
PRIORITY_GAP = 15.0     # abaixo de (referência - 15) é prioritário
MODERATE_GAP = 7.0      # entre 7 e 15 pontos abaixo da referência é moderado
MAX_CORRECTIVE = 3
MAX_EXERCISES = 4

# Modalidades com fumikomi: só nelas o Ki-Ken-Tai-Ichi (pé × espada) é medido
FUMIKOMI_MODALITIES = {
    "kihon", "kirikaeshi", "uchikomi_geiko", "kakari_geiko", "yakusoku_geiko", "waza_geiko",
    "oji_waza", "ji_geiko", "shiai_geiko", "shinsa",
}
KIKENTAI_MIN_MEASURED_FRACTION = 0.3

# Cada métrica: rótulo, pilar, referência ("boa a partir de"), o que significa estar abaixo, por que importa,
# causas prováveis, o que dizer quando está forte e qual valor por repetição mostra "onde" aconteceu.
METRICS: Dict[str, Dict[str, Any]] = {
    "verticalidade_coluna": {
        "label": "Verticalidade da coluna (Shisei)", "pilar": "movimentacao", "ref": 80.0, "rep_key": "postura",
        "abaixo": "O tronco se inclina à frente nos golpes.",
        "por_que": "No kendo o golpe sai do quadril com a coluna ereta; inclinar o tronco tira o equilíbrio, "
                   "encurta o alcance e atrasa o zanshin.",
        "causas": ["Tentar alcançar o alvo com o tronco em vez de entrar com os pés",
                   "Olhar para baixo ou para o alvo, curvando a nuca", "Distância (maai) longa demais"],
        "forte": "Coluna ereta e estável, com o centro de gravidade sobre a base.",
    },
    "nivelamento_ombros": {
        "label": "Nivelamento dos ombros", "pilar": "movimentacao", "ref": 80.0, "rep_key": None,
        "abaixo": "Os ombros ficam em alturas diferentes durante o movimento.",
        "por_que": "Ombros desnivelados indicam força desigual entre os braços: o shinai sai da linha central "
                   "e o golpe perde precisão.",
        "causas": ["Puxar o golpe com a mão direita", "Tensão no ombro direito na subida do shinai"],
        "forte": "Ombros nivelados e relaxados, com os dois braços trabalhando por igual.",
    },
    "alinhamento_base_pes": {
        "label": "Alinhamento da base (Ashi-gamae)", "pilar": "movimentacao", "ref": 80.0, "rep_key": None,
        "abaixo": "A base dos pés perde o alinhamento (pé de trás atrasado ou base aberta demais).",
        "por_que": "A base é o que dá impulso ao fumikomi e permite voltar ao kamae; sem ela o ataque perde "
                   "explosão e o corpo fica exposto depois do golpe.",
        "causas": ["Calcanhar esquerdo apoiado no chão", "Hikitsuke lento (o pé de trás não volta)",
                   "Pés abertos ou cruzados durante o deslocamento"],
        "forte": "Base firme e alinhada, pronta para impulsionar o ataque.",
    },
    "amplitude_furikaburi": {
        "label": "Amplitude do furikaburi", "pilar": "movimentacao", "ref": 75.0, "rep_key": "furikaburi",
        "abaixo": "O shinai sobe pouco antes do golpe (furikaburi curto).",
        "por_que": "O furikaburi completo usa as costas e os ombros e dá trajetória e força ao corte; "
                   "um movimento curto vira um golpe só de braço.",
        "causas": ["Pressa em golpear", "Ombros ou cotovelos travados", "Pegada apertada demais"],
        "forte": "Furikaburi amplo e fluido, usando a força das costas.",
    },
    "trajetoria_alvo": {
        "label": "Trajetória no alvo (Datotsu-bui)", "pilar": "precisao", "ref": 78.0, "rep_key": "alvo",
        "abaixo": "A espada chega fora do ponto de golpe.",
        "por_que": "Só conta golpe que atinge o datotsu-bui com a parte certa do shinai; trajetória errada "
                   "desperdiça a oportunidade.",
        "causas": ["Desvio lateral da mão direita", "Parada baixa ou alta demais na altura do alvo"],
        "forte": "Golpes chegando com precisão no alvo.",
    },
    "kikentai_sincronismo": {
        "label": "Sincronismo Ki-Ken-Tai-Ichi", "pilar": "precisao", "ref": 80.0, "rep_key": "kikentai",
        "abaixo": "O pé do fumikomi e o impacto da espada não chegam juntos.",
        "por_que": "Ki-Ken-Tai-Ichi (espírito, espada e corpo como um só) é condição para o yūkō-datotsu: "
                   "pé e espada separados não valem ponto.",
        "causas": ["Bater o pé depois da mão (o mais comum)", "Antecipar o pé e 'esperar' a espada",
                   "Fumikomi alto demais, que atrasa a chegada do pé"],
        "forte": "Pé e espada chegando juntos no impacto.",
    },
    "controle_linha_centro": {
        "label": "Controle da linha central (Chūshin-sen)", "pilar": "precisao", "ref": 78.0, "rep_key": "centro",
        "abaixo": "A espada sai da linha central durante o corte.",
        "por_que": "Quem domina o centro controla o combate; sair do centro abre a guarda e desvia o golpe.",
        "causas": ["Mão esquerda fora da linha do umbigo", "Puxar com a mão direita", "Cotovelos abertos"],
        "forte": "Controle firme da linha central, com a ponta dominando o centro.",
    },
    "regularidade_ritmo": {
        "label": "Regularidade do ritmo", "pilar": "constancia", "ref": 78.0, "rep_key": None,
        "abaixo": "O ritmo entre as repetições oscila (acelera e para).",
        "por_que": "Ritmo constante mostra controle da respiração e do corpo; oscilação costuma indicar "
                   "respiração presa ou cansaço.",
        "causas": ["Respiração desorganizada", "Pausas para recuperar o fôlego", "Pressa no início da série"],
        "forte": "Ritmo constante entre as repetições.",
    },
    "resistencia_fadiga": {
        "label": "Resistência à fadiga", "pilar": "constancia", "ref": 75.0, "rep_key": None,
        "abaixo": "A postura cai no fim da sessão.",
        "por_que": "A técnica precisa se manter até o último golpe; queda no fim mostra que o fôlego ou a "
                   "força ainda limitam a forma.",
        "causas": ["Condicionamento físico", "Ritmo inicial acima do sustentável", "Tensão desnecessária nos ombros"],
        "forte": "Mantém a qualidade técnica até o fim da sessão.",
    },
    "adequacao_cadencia": {
        "label": "Cadência da modalidade", "pilar": "constancia", "ref": 80.0, "rep_key": None,
        "abaixo": "A cadência ficou fora do ritmo esperado para esta modalidade.",
        "por_que": "Cada modalidade tem um ritmo próprio: rápido demais sacrifica a forma, lento demais não "
                   "treina a resposta.",
        "causas": ["Ritmo acima do que a técnica sustenta", "Pausas longas entre as repetições"],
        "forte": "Cadência dentro do ritmo esperado para a modalidade.",
    },
}

_catalog_cache: Optional[Dict[str, Any]] = None


# ----------------------------------------------------------------------------------------------------------
# Medição do Ki-Ken-Tai-Ichi (pé do fumikomi × impacto da espada)
# ----------------------------------------------------------------------------------------------------------
def _pt(lm: Optional[Dict[str, Any]], *names: str) -> Optional[np.ndarray]:
    if not lm:
        return None
    for n in names:
        p = lm.get(n)
        if p and "x" in p and "y" in p:
            return np.array([float(p["x"]), float(p["y"])])
    return None


def _torso_length(lm: Optional[Dict[str, Any]]) -> Optional[float]:
    sh_l, sh_r = _pt(lm, "LEFT_SHOULDER"), _pt(lm, "RIGHT_SHOULDER")
    hp_l, hp_r = _pt(lm, "LEFT_HIP"), _pt(lm, "RIGHT_HIP")
    sh = [p for p in (sh_l, sh_r) if p is not None]
    hp = [p for p in (hp_l, hp_r) if p is not None]
    if not sh or not hp:
        return None
    d = float(np.linalg.norm(np.mean(sh, axis=0) - np.mean(hp, axis=0)))
    return d if d > 1e-4 else None


def measure_kikentai(pose_history: Sequence[Optional[Dict[str, Any]]], impact_idx: int,
                     fps: float = 30.0) -> Optional[Tuple[float, float]]:
    """
    Diferença entre a chegada do pé direito (fumikomi) e o impacto da espada, numa janela de ±0,25 s.
    Devolve (diferença em ms, nota 0-100) ou None quando não dá para medir: pés fora da imagem, sem
    fumikomi na janela, ou pé que não para dentro da janela. Diferença positiva = o pé chegou DEPOIS.
    Nota: até 30 ms de diferença é sincronismo pleno; a partir de 180 ms, zero.
    """
    fps = max(1.0, float(fps or 30.0))
    n = len(pose_history)
    if not (0 <= impact_idx < n):
        return None
    win = max(3, int(round(0.25 * fps)))
    start, end = max(0, impact_idx - win), min(n - 1, impact_idx + win)
    feet, torsos = [], []
    for f in range(start, end + 1):
        lm = pose_history[f]
        feet.append(_pt(lm, "RIGHT_FOOT_INDEX", "RIGHT_ANKLE", "RIGHT_HEEL"))
        t = _torso_length(lm)
        if t:
            torsos.append(t)
    valid = sum(1 for p in feet if p is not None)
    if valid < 0.6 * len(feet) or not torsos:
        return None
    scale = float(np.median(torsos))
    speeds = []  # velocidade do pé em comprimentos de tronco por segundo, entre quadros consecutivos
    for i in range(1, len(feet)):
        a, b = feet[i - 1], feet[i]
        speeds.append(float(np.linalg.norm(b - a)) / scale * fps if a is not None and b is not None else 0.0)
    if not speeds:
        return None
    peak = int(np.argmax(speeds))
    if speeds[peak] < 1.0:      # pé praticamente parado: não houve fumikomi na janela
        return None
    landing = None
    for i in range(peak + 1, len(speeds)):
        if speeds[i] < 0.25 * speeds[peak]:
            landing = i
            break
    if landing is None:
        return None
    landing_frame = start + landing      # speeds[i] é o movimento que termina no quadro start + i
    offset_ms = (landing_frame - impact_idx) * 1000.0 / fps
    score = float(np.clip(1.0 - max(0.0, abs(offset_ms) - 30.0) / 150.0, 0.0, 1.0)) * 100.0
    return round(offset_ms, 1), round(score, 1)


# ----------------------------------------------------------------------------------------------------------
# Catálogo
# ----------------------------------------------------------------------------------------------------------
FORMATS = ("solo", "dupla", "bogu")
LEVELS = ("todos", "iniciante", "intermediario", "avancado")
SEVERITIES = ("leve", "moderada", "prioritaria")


def _catalog_path(path: Optional[str] = None) -> str:
    """Caminho do catálogo; em modo de teste o catálogo real nunca é gravado (vai para a sandbox)."""
    p = path or CATALOG_PATH
    try:
        from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir
        if is_test_mode() and os.path.abspath(p) == os.path.abspath(CATALOG_PATH):
            return os.path.join(get_test_sandbox_dir(), "test_exercise_catalog.json")
    except Exception:
        pass
    return p


def load_catalog_document(path: Optional[str] = None) -> Dict[str, Any]:
    """Documento completo do catálogo (com _sobre e versao). Recarrega quando o arquivo muda no disco."""
    global _catalog_cache
    p = path or CATALOG_PATH
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        mtime = None
    if path is None and _catalog_cache is not None and _catalog_cache.get("_mtime") == mtime:
        return _catalog_cache["data"]
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {"exercicios": []}
    if path is None:
        _catalog_cache = {"_mtime": mtime, "data": data}
    return data


def load_exercise_catalog(path: Optional[str] = None) -> List[Dict[str, Any]]:
    return load_catalog_document(path).get("exercicios", [])


def slugify(text: str) -> str:
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", t).strip("_") or "exercicio"


def validate_exercise(ex: Dict[str, Any]) -> List[str]:
    """Problemas que impedem gravar um exercício (lista vazia = válido)."""
    errs = []
    if not str(ex.get("id", "")).strip():
        errs.append("Falta o identificador (id).")
    if not str(ex.get("nome", "")).strip():
        errs.append("Falta o nome.")
    if ex.get("formato") not in FORMATS:
        errs.append(f"Formato inválido: {ex.get('formato')!r} (use {', '.join(FORMATS)}).")
    if ex.get("nivel", "todos") not in LEVELS:
        errs.append(f"Nível inválido: {ex.get('nivel')!r}.")
    for field in ("corrige", "reforca"):
        bad = [m for m in ex.get(field, []) or [] if m not in METRICS]
        if bad:
            errs.append(f"Métrica desconhecida em '{field}': {', '.join(bad)}.")
    dose = ex.get("dose") or {}
    missing = [s for s in SEVERITIES if not str(dose.get(s, "")).strip()]
    if missing:
        names = {"leve": "Leve", "moderada": "Moderada", "prioritaria": "Prioritária"}
        errs.append(f"Falta a dose para a gravidade: {', '.join(names[m] for m in missing)}.")
    if not (ex.get("corrige") or ex.get("reforca") or ex.get("modalidades") or ex.get("manutencao")):
        errs.append("O exercício precisa corrigir ou reforçar alguma métrica, ser de uma modalidade ou de manutenção; "
                    "senão nunca será recomendado.")
    return errs


def save_exercise_catalog(exercises: List[Dict[str, Any]], path: Optional[str] = None) -> str:
    """Valida e grava o catálogo inteiro (gravação atômica). Devolve o caminho gravado."""
    ids = [str(e.get("id")) for e in exercises]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"Identificadores repetidos: {', '.join(dup)}")
    for e in exercises:
        errs = validate_exercise(e)
        if errs:
            raise ValueError(f"{e.get('nome') or e.get('id')}: {' '.join(errs)}")
    doc = dict(load_catalog_document())
    doc["exercicios"] = exercises
    doc["versao"] = int(doc.get("versao", 1))
    target = _catalog_path(path)
    from src.utils.file_utils import safe_write_json
    safe_write_json(target, doc)
    global _catalog_cache
    _catalog_cache = None
    return target


def catalog_coverage(exercises: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Quantos exercícios corrigem e reforçam cada métrica (métrica sem corretivo nunca recebe exercício)."""
    exercises = load_exercise_catalog() if exercises is None else exercises
    return [{"metrica": k, "label": spec["label"],
             "corrige": sum(1 for e in exercises if k in (e.get("corrige") or [])),
             "reforca": sum(1 for e in exercises if k in (e.get("reforca") or []))}
            for k, spec in METRICS.items()]


def _format_ok(exercise: Dict[str, Any], solo_or_pair: str) -> bool:
    fmt = exercise.get("formato", "solo")
    if solo_or_pair == "solo":
        return fmt == "solo"
    return True


def _modality_ok(exercise: Dict[str, Any], modality_key: str) -> bool:
    mods = exercise.get("modalidades")
    return not mods or modality_key in mods


# ----------------------------------------------------------------------------------------------------------
# Diagnóstico
# ----------------------------------------------------------------------------------------------------------
def fmt_time(sec: Optional[float]) -> str:
    if sec is None:
        return "—"
    sec = max(0.0, float(sec))
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def _as_dict(pillars: Any) -> Dict[str, Any]:
    if isinstance(pillars, dict):
        return pillars
    return pillars.to_dict() if hasattr(pillars, "to_dict") else {}


def _severity(value: float, ref: float) -> Optional[str]:
    gap = ref - value
    if gap <= 0:
        return None
    if gap >= PRIORITY_GAP:
        return "prioritaria"
    if gap >= MODERATE_GAP:
        return "moderada"
    return "leve"


def _where(metric: str, spec: Dict[str, Any], reps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Piores momentos de uma métrica: lista de {tempo_s, valor, detalhe} e um texto curto."""
    if not reps:
        return [], ""
    rep_key = spec.get("rep_key")
    if rep_key:
        scored = [r for r in reps if r.get(rep_key) is not None and r.get("tempo_s") is not None]
        worst = sorted(scored, key=lambda r: r[rep_key])[:3]
        worst = [r for r in worst if r[rep_key] < spec["ref"]]
        out = []
        for r in worst:
            det = ""
            if metric == "kikentai_sincronismo" and r.get("kikentai_ms") is not None:
                ms = r["kikentai_ms"]
                det = f"pé {abs(ms):.0f} ms {'depois' if ms > 0 else 'antes'} da espada"
            out.append({"tempo_s": r["tempo_s"], "valor": r[rep_key], "detalhe": det})
        if not out:
            return [], ""
        txt = "Piores repetições: " + ", ".join(
            f"{fmt_time(o['tempo_s'])} ({o['valor']:.0f}%{'; ' + o['detalhe'] if o['detalhe'] else ''})" for o in out)
        return out, txt
    times = sorted(float(r["tempo_s"]) for r in reps if r.get("tempo_s") is not None)
    if metric == "resistencia_fadiga" and len(times) >= 3:
        t0 = times[int(len(times) * 2 / 3)]
        return [{"tempo_s": t0, "valor": None, "detalhe": "início do último terço"}], f"A queda aparece a partir de {fmt_time(t0)}."
    if metric == "regularidade_ritmo" and len(times) >= 4:
        gaps = [(times[i] - times[i - 1], times[i]) for i in range(1, len(times))]
        med = float(np.median([g for g, _ in gaps]))
        # Só intervalos bem fora do normal (25% acima ou abaixo da mediana)
        off = [g for g in gaps if abs(g[0] - med) > 0.25 * max(med, 0.1)]
        worst = sorted(off, key=lambda g: -abs(g[0] - med))[:3]
        if not worst:
            return [], ""
        out = [{"tempo_s": t, "valor": None,
                "detalhe": f"{'pausa' if g > med else 'aceleração'}: {g:.1f}s entre golpes (o normal é {med:.1f}s)"}
               for g, t in sorted(worst, key=lambda g: g[1])]
        return out, "Maiores quebras de ritmo: " + ", ".join(f"{fmt_time(o['tempo_s'])} ({o['detalhe']})" for o in out)
    return [], ""


def _kikentai_cause(reps: List[Dict[str, Any]]) -> Optional[str]:
    offs = [r["kikentai_ms"] for r in reps if r.get("kikentai_ms") is not None]
    if len(offs) < 2:
        return None
    mean = float(np.mean(offs))
    late = sum(1 for o in offs if o > 30)
    early = sum(1 for o in offs if o < -30)
    if late >= early and mean > 30:
        return f"Medido: em média o pé chega {mean:.0f} ms depois da espada ({late} de {len(offs)} repetições atrasadas)"
    if early > late and mean < -30:
        return f"Medido: em média o pé chega {abs(mean):.0f} ms antes da espada ({early} de {len(offs)} repetições adiantadas)"
    return None


def build_diagnosis(pillars: Any, modality_key: str, modality_meta: Optional[Dict[str, Any]] = None,
                    catalog: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Monta o diagnóstico estruturado a partir das métricas (objeto TrainingPillarMetrics ou dict)."""
    p = _as_dict(pillars)
    meta = modality_meta or {}
    solo_or_pair = meta.get("solo_or_pair", "solo_or_pair")
    catalog = load_exercise_catalog() if catalog is None else catalog
    reps = list(p.get("rep_details") or [])
    values: Dict[str, float] = {}
    for group in ("movimentacao_submetrics", "precisao_submetrics", "constancia_submetrics"):
        for k, v in (p.get(group) or {}).items():
            if k in METRICS and isinstance(v, (int, float)):
                values[k] = float(v)
    not_measured = list(p.get("nao_medidos") or [])

    findings, strengths = [], []
    for key, spec in METRICS.items():
        if key not in values:
            continue
        v = values[key]
        sev = _severity(v, spec["ref"])
        if sev is None:
            strengths.append({"metrica": key, "label": spec["label"], "valor": round(v, 1), "ref": spec["ref"],
                              "texto": spec["forte"]})
            continue
        moments, where_txt = _where(key, spec, reps)
        causes = list(spec["causas"])
        if key == "kikentai_sincronismo":
            measured_cause = _kikentai_cause(reps)
            if measured_cause:
                causes.insert(0, measured_cause)
        if key == "adequacao_cadencia":
            exp = meta.get("expected_cadence_cpm")
            cpm = p.get("cadence_cpm")
            if exp and cpm is not None:
                lado = "acima" if float(cpm) > exp[1] else "abaixo"
                causes.insert(0, f"Medido: {float(cpm):.0f} golpes por minuto, {lado} da faixa de {exp[0]}–{exp[1]} da modalidade")
        findings.append({
            "metrica": key, "label": spec["label"], "pilar": spec["pilar"], "valor": round(v, 1), "ref": spec["ref"],
            "gravidade": sev, "o_que": spec["abaixo"], "onde": where_txt, "momentos": moments,
            "por_que": spec["por_que"], "causas": causes,
        })
    findings.sort(key=lambda f: (SEVERITY_RANK[f["gravidade"]], -(f["ref"] - f["valor"])))
    strengths.sort(key=lambda s: -(s["valor"] - s["ref"]))

    # Exercícios: corretivos para as prioridades, um de reforço para o ponto mais forte, específico da modalidade
    exercises: List[Dict[str, Any]] = []
    used = set()

    def pick(match, kind: str, reason_metric: Dict[str, Any], severity: str) -> bool:
        cands = [e for e in catalog if e.get("id") not in used and match(e)
                 and _format_ok(e, solo_or_pair) and _modality_ok(e, modality_key)]
        if not cands:
            return False
        cands.sort(key=lambda e: (0 if e.get("modalidades") else 1, 0 if e.get("formato") == "solo" else 1))
        ex = cands[0]
        used.add(ex.get("id"))
        exercises.append(_exercise_entry(ex, kind, reason_metric, severity))
        return True

    for f in findings:
        if sum(1 for e in exercises if e["tipo"] == "corretivo") >= MAX_CORRECTIVE:
            break
        pick(lambda e, m=f["metrica"]: m in e.get("corrige", []), "corretivo", f, f["gravidade"])
    for s in strengths:
        if len(exercises) >= MAX_EXERCISES:
            break
        if pick(lambda e, m=s["metrica"]: m in e.get("reforca", []), "reforco", s, "moderada"):
            break
    if len(exercises) < MAX_EXERCISES:
        pick(lambda e: bool(e.get("modalidades")), "modalidade", {"label": meta.get("name", modality_key)}, "moderada")
    if not exercises:
        pick(lambda e: bool(e.get("manutencao")), "manutencao", {"label": "Fundamentos"}, "leve")
    exp_cpm = meta.get("expected_cadence_cpm")
    if exp_cpm:
        for ex in exercises:
            if "ritmo alvo" in ex["dose"]:
                ex["dose"] = ex["prescription"] = ex["dose"].replace(
                    "ritmo alvo", f"ritmo alvo ({exp_cpm[0]}–{exp_cpm[1]} golpes por minuto)")

    goals = []
    for f in findings[:MAX_CORRECTIVE]:
        target = min(f["ref"], round(f["valor"] + 10.0))
        goals.append({"metrica": f["metrica"], "label": f["label"], "atual": f["valor"], "meta": target,
                      "texto": f"{f['label']}: de {f['valor']:.0f}% para {target:.0f}% ou mais"})

    return {
        "versao": 1,
        "modality_key": modality_key,
        "achados": findings,
        "pontos_fortes": strengths,
        "nao_medidos": not_measured,
        "exercicios": exercises,
        "plano": _weekly_plan(exercises, findings),
        "metas": goals,
        "repeticoes_analisadas": len(reps),
    }


def _exercise_entry(ex: Dict[str, Any], kind: str, reason: Dict[str, Any], severity: str) -> Dict[str, Any]:
    dose = (ex.get("dose") or {}).get(severity) or (ex.get("dose") or {}).get("moderada", "")
    if kind == "corretivo":
        motivo = f"Corrige: {reason['label']} ({reason['valor']:.0f}%, referência {reason['ref']:.0f}%)"
    elif kind == "reforco":
        motivo = f"Leva adiante o ponto forte: {reason['label']} ({reason['valor']:.0f}%)"
    elif kind == "modalidade":
        motivo = f"Específico da modalidade: {reason['label']}"
    else:
        motivo = "Manutenção dos fundamentos"
    alvo = ", ".join(METRICS[m]["label"] for m in (ex.get("corrige") or ex.get("reforca") or []) if m in METRICS)
    return {
        "id": ex.get("id"), "tipo": kind, "motivo": motivo,
        "name": ex.get("nome", "Exercício"), "target": alvo or reason.get("label", "Fundamentos"),
        "prescription": dose, "dose": dose, "formato": ex.get("formato", "solo"), "nivel": ex.get("nivel", "todos"),
        "passos": ex.get("passos", []), "pontos_de_atencao": ex.get("pontos_de_atencao", []),
        "erros_comuns": ex.get("erros_comuns", []), "como_medir": ex.get("como_medir", ""),
    }


def _weekly_plan(exercises: List[Dict[str, Any]], findings: List[Dict[str, Any]]) -> List[str]:
    if not exercises:
        return []
    corr = [e for e in exercises if e["tipo"] == "corretivo"]
    other = [e for e in exercises if e["tipo"] != "corretivo"]
    plan = ["Aquecimento (5 min): mobilidade de ombros e 30 Jōge-buri lentos."]
    if corr:
        first = corr[0]
        plan.append(f"Todos os dias de treino, primeiro: {first['name']} — {first['dose']}.")
        if len(corr) > 1:
            plan.append("Alternar entre os dias: " + "; ".join(f"{e['name']} — {e['dose']}" for e in corr[1:]) + ".")
    for e in other:
        plan.append(f"2 vezes na semana, depois dos corretivos: {e['name']} — {e['dose']}.")
    plan.append("Frequência sugerida: 3 sessões por semana. Grave uma nova sessão da mesma modalidade "
                "em 1 a 2 semanas e compare com as metas abaixo.")
    return plan


def legacy_lists(diag: Dict[str, Any]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Pontos fortes, pontos de atenção e exercícios no formato de lista usado pelas telas e relatórios."""
    strengths = [f"{s['label']} ({s['valor']:.0f}%): {s['texto']}" for s in diag.get("pontos_fortes", [])]
    improvements = []
    for f in diag.get("achados", []):
        txt = f"[{SEVERITY_LABEL[f['gravidade']]}] {f['o_que']} {f['label']}: {f['valor']:.0f}% (referência {f['ref']:.0f}%)."
        if f.get("onde"):
            txt += f" {f['onde']}"
        improvements.append(txt)
    return strengths, improvements, list(diag.get("exercicios", []))


def metric_text(submetrics: Dict[str, Any], key: str, nao_medidos: Optional[List[Dict[str, str]]] = None) -> str:
    """Valor de uma submétrica para exibição: '82.0%' ou 'não medido'."""
    v = (submetrics or {}).get(key)
    if isinstance(v, (int, float)):
        return f"{float(v):.1f}%"
    return "não medido"


def diagnosis_markdown(diag: Dict[str, Any]) -> List[str]:
    """Linhas em Markdown do diagnóstico detalhado (relatório individual)."""
    lines: List[str] = []
    try:
        from src.analytics.training_narrative import narrative_text
        nar = narrative_text(diag)
        fonte = "escrito pela IA a partir das medidas" if nar.get("fonte") == "ia" else "gerado a partir das medidas"
        lines += [f"### 📝 Resumo do Sensei *({fonte})*", nar["texto"], ""]
    except Exception:
        pass
    try:
        from src.analytics.training_progress import evolution_lines
        evo_lines = evolution_lines(diag.get("evolucao") or {})
        if evo_lines:
            lines += ["### 📈 Evolução"] + [f"- {t}" for t in evo_lines] + [""]
    except Exception:
        pass
    fortes = diag.get("pontos_fortes", [])
    lines.append("### 🌟 Pontos Fortes")
    if fortes:
        for s in fortes:
            lines.append(f"- ✅ **{s['label']} ({s['valor']:.0f}%)** — {s['texto']}")
    else:
        lines.append("- *Nenhuma métrica acima da referência nesta sessão.*")
    lines.append("")
    lines.append("### ⚠️ Pontos de Atenção (do mais urgente ao mais leve)")
    achados = diag.get("achados", [])
    if not achados:
        lines.append("- *Todas as métricas medidas estão dentro da referência.*")
    for f in achados:
        lines.append(f"#### [{SEVERITY_LABEL[f['gravidade']]}] {f['label']}: {f['valor']:.0f}% (referência {f['ref']:.0f}%)")
        lines.append(f"- **O que foi visto:** {f['o_que']}")
        if f.get("onde"):
            lines.append(f"- **Onde:** {f['onde']}")
        lines.append(f"- **Por que importa:** {f['por_que']}")
        lines.append(f"- **Causas prováveis:** " + "; ".join(f["causas"]))
        lines.append("")
    if diag.get("nao_medidos"):
        lines.append("### ❔ Não medido nesta sessão")
        for n in diag["nao_medidos"]:
            lines.append(f"- **{n['label']}:** {n['motivo']}")
        lines.append("")
    lines.append("### 🏋️ Exercícios Recomendados")
    for ex in diag.get("exercicios", []):
        lines.append(f"#### 🥋 {ex['name']} *({ex['formato']} · nível {ex['nivel']})*")
        lines.append(f"- **Por quê:** {ex['motivo']}")
        lines.append(f"- **Dose:** {ex['dose']}")
        if ex.get("passos"):
            lines.append("- **Progressão:** " + " → ".join(f"({i}) {p}" for i, p in enumerate(ex["passos"], 1)))
        if ex.get("pontos_de_atencao"):
            lines.append("- **Pontos de atenção:** " + "; ".join(ex["pontos_de_atencao"]))
        if ex.get("erros_comuns"):
            lines.append("- **Erros comuns:** " + "; ".join(ex["erros_comuns"]))
        if ex.get("como_medir"):
            lines.append(f"- **Como medir a evolução:** {ex['como_medir']}")
        lines.append("")
    if diag.get("plano"):
        lines.append("### 📅 Plano da Semana")
        for p in diag["plano"]:
            lines.append(f"- {p}")
        lines.append("")
    if diag.get("metas"):
        lines.append("### 🎯 Metas para a Próxima Sessão")
        for m in diag["metas"]:
            lines.append(f"- {m['texto']}")
        lines.append("")
    return lines
