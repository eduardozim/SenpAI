"""
Resumo corrido do diagnóstico do modo de treinamento, como um sensei falando com o aluno.

Duas fontes:
- "modelo": texto montado por regras a partir do diagnóstico (sempre disponível, sem rede);
- "ia": o assistente LLM do projeto (Gemini ou OpenAI, chave em Configurações > Variáveis do Sistema)
  reescreve o resumo a partir SÓ dos fatos do diagnóstico.

Proteções do texto da IA:
- a IA recebe só métricas, instantes, causas, exercícios, plano e metas; o nome do kenshi não é enviado
  (o texto é escrito em "você");
- todo número citado no texto precisa existir nos fatos enviados; se a IA citar um número que não está
  lá (valor inventado), o texto é rejeitado e fica o do modelo;
- texto vazio, curto ou longo demais também é rejeitado.
"""

import re
import json
import datetime
from typing import Any, Dict, List, Optional

from src.analytics.training_feedback import SEVERITY_LABEL
from src.utils.logger_manager import log_event

MIN_CHARS = 200
MAX_CHARS = 2500


def _modality_name(diag: Dict[str, Any]) -> str:
    try:
        from src.analytics.training_analyzer import TRAINING_MODALITIES_METADATA
        return TRAINING_MODALITIES_METADATA.get(diag.get("modality_key", ""), {}).get("name", "treino")
    except Exception:
        return "treino"


def _lc(label: str) -> str:
    """Primeira letra minúscula, mantendo termos japoneses como Shisei e Ki-Ken-Tai-Ichi."""
    return label[:1].lower() + label[1:]


def _join(items: List[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " e " + items[-1]


def template_narrative(diag: Dict[str, Any]) -> str:
    """Resumo por regras: pontos fortes, o mais urgente (com a causa medida), exercícios e metas."""
    mod = _modality_name(diag)
    paras = []
    fortes = diag.get("pontos_fortes", [])[:2]
    if fortes:
        paras.append(f"Nesta sessão de {mod}, seus pontos mais fortes foram "
                     + _join([f"{_lc(s['label'])} ({s['valor']:.0f}%)" for s in fortes]) + ".")
    else:
        paras.append(f"Nesta sessão de {mod}, nenhuma métrica ficou acima da referência ainda: há bastante espaço para evoluir.")
    evo = _evolution_facts(diag)
    if evo:
        paras.append(" ".join(evo))
    achados = diag.get("achados", [])
    if achados:
        f = achados[0]
        txt = (f"O que mais precisa de atenção agora é {_lc(f['label'])}: {f['valor']:.0f}%, "
               f"quando a referência é {f['ref']:.0f}%. {f['o_que']}")
        medida = next((c for c in f.get("causas", []) if c.startswith("Medido:")), None)
        if medida:
            m = medida.replace("Medido: ", "").rstrip(".")
            txt += " " + m[:1].upper() + m[1:] + "."
        if len(achados) > 1:
            txt += " Também vale cuidar de " + _join([_lc(a["label"]) for a in achados[1:3]]) + "."
        paras.append(txt)
    else:
        paras.append("Todas as métricas medidas ficaram dentro da referência; o foco agora é manter a qualidade.")
    exs = diag.get("exercicios", [])
    if exs:
        first = exs[0]
        txt = f"Para a próxima semana, comece por {first['name']}: {first['dose']}"
        if len(exs) > 1:
            txt += ". Inclua também " + _join([e["name"] for e in exs[1:]])
        paras.append(txt + ".")
    metas = diag.get("metas", [])
    if metas:
        paras.append("Na próxima sessão, a meta é levar " + _join(
            [f"{_lc(m['label'])} de {m['atual']:.0f}% para {m['meta']:.0f}% ou mais" for m in metas[:2]]) + ".")
    return "\n\n".join(paras)


def narrative_facts(diag: Dict[str, Any]) -> Dict[str, Any]:
    """Só os fatos que a IA pode usar (sem nome do kenshi)."""
    return {
        "modalidade": _modality_name(diag),
        "pontos_fortes": [{"metrica": s["label"], "valor_pct": round(s["valor"]), "referencia_pct": round(s["ref"])}
                          for s in diag.get("pontos_fortes", [])[:4]],
        "pontos_de_atencao": [{
            "metrica": f["label"], "gravidade": SEVERITY_LABEL[f["gravidade"]], "valor_pct": round(f["valor"]),
            "referencia_pct": round(f["ref"]), "o_que_foi_visto": f["o_que"], "onde": f.get("onde", ""),
            "por_que_importa": f["por_que"], "causas_provaveis": f.get("causas", [])[:3],
        } for f in diag.get("achados", [])[:4]],
        "nao_medido": [f"{n['label']}: {n['motivo']}" for n in diag.get("nao_medidos", [])],
        "exercicios": [{"nome": e["name"], "motivo": e["motivo"], "dose": e["dose"]} for e in diag.get("exercicios", [])],
        "plano_da_semana": diag.get("plano", []),
        "metas": [m["texto"] for m in diag.get("metas", [])],
        "evolucao_desde_a_sessao_anterior": _evolution_facts(diag),
    }


def _evolution_facts(diag: Dict[str, Any]) -> List[str]:
    try:
        from src.analytics.training_progress import evolution_lines
        return evolution_lines(diag.get("evolucao") or {})
    except Exception:
        return []


def build_prompt(facts: Dict[str, Any]) -> str:
    return (
        "Você é um sensei de kendo experiente escrevendo para um aluno sobre a sessão de treino que ele acabou de "
        "gravar. Escreva em português do Brasil, falando direto com o aluno (\"você\"), em tom respeitoso, claro e "
        "encorajador, sem exageros.\n"
        "Regras obrigatórias:\n"
        "- Use SOMENTE os fatos do JSON abaixo. Não invente medidas, porcentagens, tempos, repetições nem exercícios.\n"
        "- Todo número que aparecer no texto tem que estar no JSON (copie como está). Se não tiver certeza, não use número.\n"
        "- Comece pelo ponto de atenção mais grave e explique por que ele importa no kendo; cite onde aconteceu, se houver.\n"
        "- Reconheça os pontos fortes em uma frase.\n"
        "- Se houver evolução desde a sessão anterior, diga o que melhorou, o que caiu e quantas metas foram atingidas.\n"
        "- Diga quais exercícios fazer primeiro, com a dose indicada, e qual é a meta para a próxima sessão.\n"
        "- Se algo não foi medido, não comente como se tivesse sido.\n"
        "- 3 a 5 parágrafos curtos, sem títulos, sem listas, sem saudação e sem assinatura.\n"
        f"Fatos da sessão (JSON):\n{json.dumps(facts, ensure_ascii=False)}\n\n"
        'Responda só JSON: {"texto": "..."}'
    )


def unsupported_numbers(text: str, facts: Dict[str, Any]) -> List[str]:
    """Números do texto que não aparecem nos fatos enviados (sinal de valor inventado)."""
    allowed = set(re.findall(r"\d+", json.dumps(facts, ensure_ascii=False)))
    return sorted({n for n in re.findall(r"\d+", text) if n not in allowed})


def ai_narrative(diag: Dict[str, Any], llm: Any = None) -> Dict[str, Any]:
    """
    Pede o resumo ao LLM. Devolve {"texto", "fonte": "ia"|"modelo", "gerado_em", "aviso"?}.
    Sem IA disponível, ou com resposta rejeitada, devolve o texto do modelo com o motivo em "aviso".
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    fallback = {"texto": template_narrative(diag), "fonte": "modelo", "gerado_em": now}
    if llm is None:
        try:
            from src.engine.web_research import get_llm
            llm = get_llm()
        except Exception:
            llm = None
    if llm is None:
        return {**fallback, "aviso": "IA desligada: informe a chave do Gemini ou da OpenAI em Configurações > Variáveis do Sistema."}
    facts = narrative_facts(diag)
    try:
        res = llm._call_remote_llm(build_prompt(facts)) or {}
    except Exception as err:
        log_event("WARNING", f"Resumo do diagnóstico pela IA falhou: {err}", "training_narrative")
        res = {}
    text = str(res.get("texto", "")).strip()
    if not text:
        return {**fallback, "aviso": "A IA não respondeu; mantido o texto do modelo."}
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return {**fallback, "aviso": f"Texto da IA com tamanho fora do esperado ({len(text)} caracteres); mantido o texto do modelo."}
    bad = unsupported_numbers(text, facts)
    if bad:
        log_event("WARNING", f"Resumo da IA rejeitado: números que não estão no diagnóstico ({', '.join(bad)}).", "training_narrative")
        return {**fallback, "aviso": f"A IA citou números que não estão no diagnóstico ({', '.join(bad)}); mantido o texto do modelo."}
    return {"texto": text, "fonte": "ia", "gerado_em": now, "provedor": str(getattr(llm, "provider", ""))}


def narrative_text(diag: Dict[str, Any]) -> Dict[str, Any]:
    """O resumo a exibir: o gravado no diagnóstico (da IA) ou o do modelo."""
    nar = diag.get("narrativa") or {}
    if nar.get("texto"):
        return nar
    return {"texto": template_narrative(diag), "fonte": "modelo"}
