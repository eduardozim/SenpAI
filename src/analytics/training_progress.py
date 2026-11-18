"""
Histórico de evolução de cada kenshi no modo de treinamento (data/training_progress.json).

O mesmo kenshi é reconhecido entre vídeos PELO NOME dado ao kendoca. A identificação automática (zekken,
tenugui, biometria) foi testada para isso em 2026-11-13 e reprovada: entre vídeos diferentes ela confundia
pessoas (com limiar 0,90, 19% dos kenshis seriam ligados ao histórico de outra pessoa, e só 62% dos da
mesma luta seriam reconhecidos). Ela continua servindo só dentro de um mesmo vídeo.

- Ao dar nome a um kendoca, a sessão entra no histórico daquele nome; renomear move o registro (a chave do
  registro é a sessão + o kendoca, não o nome). Nomes padrão ("Kenshi 01", "Kendoca Praticante") não entram.
- A comparação usa a sessão anterior do mesmo nome, de preferência da mesma modalidade, e confere as metas
  que o diagnóstico daquela sessão tinha proposto.
"""

import os
import re
import datetime
import unicodedata
from typing import Any, Dict, List, Optional

from src.analytics.training_feedback import METRICS
from src.utils.file_utils import safe_load_json, safe_write_json

PROGRESS_PATH = os.path.join("data", "training_progress.json")
DEFAULT_NAME_RE = re.compile(r"^\s*(kenshi|kendoca|kendoka|praticante)\b", re.IGNORECASE)
MAX_HISTORY_POINTS = 30


def _path(path: Optional[str] = None) -> str:
    p = path or PROGRESS_PATH
    try:
        from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir
        if is_test_mode() and os.path.abspath(p) == os.path.abspath(PROGRESS_PATH):
            return os.path.join(get_test_sandbox_dir(), "test_training_progress.json")
    except Exception:
        pass
    return p


def name_key(name: str) -> str:
    """Chave do nome: sem acento, sem diferença de maiúsculas e espaços ("Eduardo  Zim" = "eduardo zim")."""
    t = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", t).strip().casefold()


def is_default_name(name: str, default_name: str = "") -> bool:
    n = str(name or "").strip()
    return (not n) or name_key(n) == name_key(default_name) or bool(DEFAULT_NAME_RE.match(n))


def load_progress(path: Optional[str] = None) -> Dict[str, Any]:
    data = safe_load_json(_path(path), default=None)
    if not isinstance(data, dict) or not isinstance(data.get("sessoes"), list):
        data = {"versao": 1, "sessoes": []}
    return data


def _snapshot(diag: Dict[str, Any], pillars: Dict[str, Any]) -> Dict[str, Any]:
    values = {}
    for group in ("movimentacao_submetrics", "precisao_submetrics", "constancia_submetrics"):
        for k, v in (pillars.get(group) or {}).items():
            if k in METRICS and isinstance(v, (int, float)):
                values[k] = round(float(v), 1)
    return {
        "metricas": values,
        "geral": round(float(pillars.get("overall_score") or 0.0), 1),
        "pilares": {k: round(float(pillars.get(k) or 0.0), 1) for k in ("movimentacao", "precisao", "constancia")},
        "repeticoes": int(pillars.get("total_repetitions") or 0),
        "metas": [{"metrica": m["metrica"], "label": m["label"], "meta": m["meta"]} for m in diag.get("metas", [])],
    }


def record_session(name: str, session_key: str, kendoka_id: str, modality_key: str, diag: Dict[str, Any],
                   pillars: Dict[str, Any], video: str = "", source: str = "gravado", default_name: str = "",
                   session_date: Optional[str] = None, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Grava (ou atualiza) a sessão no histórico do nome. Nome padrão remove o registro desta sessão + kendoca
    (o kendoca deixou de ter nome). Só grava no disco quando algo mudou. Devolve o registro ou None.
    """
    data = load_progress(path)
    sessions = data["sessoes"]
    idx = next((i for i, s in enumerate(sessions)
                if s.get("sessao") == session_key and s.get("kendoka_id") == kendoka_id), None)
    if is_default_name(name, default_name):
        if idx is not None:
            sessions.pop(idx)
            safe_write_json(_path(path), data)
        return None
    snap = _snapshot(diag, pillars)
    old = sessions[idx] if idx is not None else None
    rec = {
        "sessao": session_key, "kendoka_id": kendoka_id, "nome": str(name).strip(), "chave_nome": name_key(name),
        "modalidade": modality_key, "video": video, "origem": source,
        # Data da sessão (da análise); sem ela, a do primeiro registro
        "registrado_em": session_date or (old or {}).get("registrado_em") or datetime.datetime.now().isoformat(timespec="seconds"),
        **snap,
    }
    if old == rec:
        return rec
    if idx is None:
        sessions.append(rec)
    else:
        sessions[idx] = rec
    safe_write_json(_path(path), data)
    return rec


def history_for(name: str, path: Optional[str] = None) -> List[Dict[str, Any]]:
    key = name_key(name)
    return sorted([s for s in load_progress(path)["sessoes"] if s.get("chave_nome") == key],
                  key=lambda s: s.get("registrado_em", ""))


def compare_with_previous(current: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evolução da sessão atual em relação à anterior (de preferência da mesma modalidade)."""
    earlier = [s for s in history if s.get("registrado_em", "") < current.get("registrado_em", "")
               and s.get("sessao") != current.get("sessao")]
    out: Dict[str, Any] = {
        "nome": current.get("nome"), "sessoes_no_historico": len(history),
        "historico": [{"data": s["registrado_em"][:10], "geral": s.get("geral"), "modalidade": s.get("modalidade")}
                      for s in history[-MAX_HISTORY_POINTS:]],
    }
    if not earlier:
        out["primeira_sessao"] = True
        return out
    same = [s for s in earlier if s.get("modalidade") == current.get("modalidade")]
    prev = (same or earlier)[-1]
    out["anterior"] = {"data": prev["registrado_em"][:10], "modalidade": prev.get("modalidade"),
                       "video": prev.get("video", ""), "mesma_modalidade": bool(same)}
    metrics = []
    for k, spec in METRICS.items():
        a, b = prev.get("metricas", {}).get(k), current.get("metricas", {}).get(k)
        if a is None or b is None:
            continue
        metrics.append({"metrica": k, "label": spec["label"], "antes": a, "agora": b, "delta": round(b - a, 1)})
    out["metricas"] = metrics
    out["geral"] = {"antes": prev.get("geral"), "agora": current.get("geral"),
                    "delta": round(float(current.get("geral") or 0) - float(prev.get("geral") or 0), 1)}
    goals = []
    for g in prev.get("metas", []):
        now = current.get("metricas", {}).get(g["metrica"])
        goals.append({"label": g["label"], "meta": g["meta"], "agora": now,
                      "atingida": None if now is None else bool(now >= g["meta"])})
    out["metas_anteriores"] = goals
    return out


def attach_progress(profile: Dict[str, Any], session_key: str, name: str, default_name: str = "",
                    video: str = "", source: str = "gravado", session_date: Optional[str] = None,
                    path: Optional[str] = None) -> Dict[str, Any]:
    """
    Grava a sessão do kendoca no histórico do nome e põe a evolução em profile["diagnosis"]["evolucao"]
    (tela, Markdown, PDF e resumo do sensei passam a mostrá-la). Devolve a evolução.
    """
    diag = profile.get("diagnosis") or {}
    if diag.get("achados") is None:
        return {}
    pillars = profile.get("pillars") or {}
    kendoka_id = str(profile.get("kendoka_id") or "KENSHI")
    try:
        rec = record_session(name, session_key, kendoka_id, diag.get("modality_key", ""), diag, pillars,
                             video=video, source=source, default_name=default_name, session_date=session_date,
                             path=path)
    except Exception as err:
        from src.utils.logger_manager import log_event
        log_event("WARNING", f"Histórico de evolução indisponível: {err}", "training_progress")
        rec = None
    if rec is None:
        evo = {"sem_nome": True} if is_default_name(name, default_name) else {}
    else:
        evo = compare_with_previous(rec, history_for(name, path))
    diag["evolucao"] = evo
    return evo


def evolution_lines(evo: Dict[str, Any]) -> List[str]:
    """Frases da evolução para o Markdown, o PDF e o resumo do sensei (vazio sem sessão anterior)."""
    if not evo or not evo.get("anterior"):
        return []
    prev, geral = evo["anterior"], evo.get("geral", {})
    lines = [f"Comparado com a sessão de {prev['data']}"
             + ("" if prev.get("mesma_modalidade") else " (outra modalidade)")
             + f": nota geral de {float(geral.get('antes') or 0):.0f} para {float(geral.get('agora') or 0):.0f}."]
    ups = [m for m in evo.get("metricas", []) if m["delta"] >= 3]
    downs = [m for m in evo.get("metricas", []) if m["delta"] <= -3]
    if ups:
        lines.append("Melhorou: " + "; ".join(f"{m['label']} de {m['antes']:.0f}% para {m['agora']:.0f}%" for m in ups) + ".")
    if downs:
        lines.append("Caiu: " + "; ".join(f"{m['label']} de {m['antes']:.0f}% para {m['agora']:.0f}%" for m in downs) + ".")
    metas = [g for g in evo.get("metas_anteriores", []) if g.get("atingida") is not None]
    if metas:
        hit = [g for g in metas if g["atingida"]]
        detail = "; ".join(f"{g['label']} {g['agora']:.0f}%, meta {g['meta']:.0f}%" for g in metas)
        lines.append(f"Metas da sessão anterior atingidas: {len(hit)} de {len(metas)} ({detail}).")
    return lines
