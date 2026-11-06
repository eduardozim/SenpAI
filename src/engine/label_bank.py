"""
Banco Único de Rótulos do SenpAI (Pendência I11).

Toda informação que pode ensinar o sistema vira um RÓTULO neste banco, com ORIGEM e CONFIANÇA:
- gabarito ........ pontos oficiais da luta (I2)
- revisao ......... correção golpe a golpe na tela de análise (TP/FP/FN por revisor Dan ou Shinpan)
- excel ........... revisão importada de planilha
- curadoria ....... decisão na Fila de Curadoria Ativa (I12)
- bandeiras ....... pontos lidos das bandeiras dos shinpan (I7)
- modo_treino ..... correções de modalidade/trecho no Modo Treinamento (I13)
- internet ........ coleta automática (I15)

Cada rótulo afirma uma coisa ("claim") sobre um instante de um vídeo:
- "ponto"      → o golpe vira ponto (True) ou não (False)
- "golpe"      → houve tentativa de golpe (False = detecção espúria)
- "tecnica"    → MEN / KOTE / DO / TSUKI
- "modalidade" → luta ou uma das 14 modalidades (I13/I14)
- "regra"      → observação de regra (hansoku, golpe após o yame...) para a curadoria (I12)

O ciclo de treino (I8) lê daqui para treinar TODOS os modelos: filtro de golpes (I3), técnica (I9),
calibração de pesos (I4) e, depois, modalidade (I14). Lutas do Golden Benchmark nunca entram no treino.

As fontes antigas continuam funcionando: sync_from_sources() traz o gabarito e o feedback_dataset para
cá a cada ciclo (idempotente). Fontes novas gravam direto com add_label().
"""

import os
import re
import json
import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.logger_manager import log_event
from src.utils.test_isolation import is_test_mode, get_test_sandbox_dir
from src.utils.file_utils import safe_write_json, safe_load_json, _GLOBAL_FILE_IO_LOCK

DEFAULT_BANK_PATH = os.path.join("data", "label_bank.json")
CLAIMS = ("ponto", "golpe", "tecnica", "modalidade", "regra")
ORIGINS = ("gabarito", "revisao", "excel", "curadoria", "bandeiras", "modo_treino", "internet")
# Peso base de cada origem (multiplicado pelo peso do revisor quando houver um revisor humano)
ORIGIN_BASE_AUTHORITY = {
    "gabarito": 4.0, "revisao": 1.0, "excel": 1.0, "curadoria": 1.0,
    "bandeiras": 2.0, "modo_treino": 1.0, "internet": 0.5,
}
MATCH_TOLERANCE_SEC = 1.5
LUTA_MODALITY = "luta"   # rótulo de modalidade das lutas oficiais (as outras são as 14 modalidades de treino)
# Diferença mínima de autoridade para um rótulo vencer outro sem virar conflito para a curadoria
CONFLICT_MARGIN = 0.25
DERIVED_DECISION_CATEGORIES = {"rotulo_fraco"}
_UPLOAD_PREFIX = re.compile(r"^upload_\d+_")


def canonical_video_name(video: Any) -> str:
    """Nome do arquivo sem pasta e sem o prefixo 'upload_<n>_' que a tela acrescenta."""
    return _UPLOAD_PREFIX.sub("", os.path.basename(str(video or "")).strip())


def parse_time(value: Any) -> Optional[float]:
    """'00:13.600', '13.6s', '1:02:03' ou número → segundos (None se não der para ler)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().rstrip("s").replace(",", ".")
    if not text:
        return None
    try:
        secs = 0.0
        for part in text.split(":"):
            secs = secs * 60.0 + float(part)
        return secs
    except ValueError:
        return None


def get_bank_path() -> str:
    return os.path.join(get_test_sandbox_dir(), "label_bank.json") if is_test_mode() else DEFAULT_BANK_PATH


class LabelBank:
    def __init__(self, path: Optional[str] = None):
        self.path = path or get_bank_path()

    # ------------------------------------------------------------------
    # Armazenamento
    # ------------------------------------------------------------------
    def load(self) -> Dict[str, Dict[str, Any]]:
        data = safe_load_json(self.path, default={})
        if isinstance(data, dict):
            return data.get("rotulos", {})
        return {}

    def _save(self, labels: Dict[str, Dict[str, Any]]) -> None:
        safe_write_json(self.path, {"versao": 1, "rotulos": labels}, indent=1)

    @staticmethod
    def make_id(origin: str, video: str, claim: str, time_sec: Optional[float], kenshi: Optional[str], ref: str = "") -> str:
        t = "na" if time_sec is None else f"{time_sec:.2f}"
        return f"{origin}|{canonical_video_name(video)}|{claim}|{t}|{kenshi or '-'}|{ref}"

    def add_labels(self, labels: Iterable[Dict[str, Any]]) -> int:
        """Grava (ou atualiza) rótulos. Cada um precisa de origin, video_name e claim."""
        with _GLOBAL_FILE_IO_LOCK:
            bank = self.load()
            now = datetime.datetime.now().isoformat(timespec="seconds")
            n = 0
            for lb in labels:
                if lb.get("origin") not in ORIGINS or lb.get("claim") not in CLAIMS:
                    raise ValueError(f"Rótulo inválido: origem {lb.get('origin')!r}, afirmação {lb.get('claim')!r}")
                item = dict(lb)
                item["video_name"] = canonical_video_name(item["video_name"])
                item.setdefault("confidence", 1.0)
                item.setdefault("reviewer_weight", 1.0)
                item["id"] = item.get("id") or self.make_id(item["origin"], item["video_name"], item["claim"],
                                                              item.get("time_sec"), item.get("kenshi"), item.get("ref", ""))
                item.setdefault("created_at", bank.get(item["id"], {}).get("created_at", now))
                item["updated_at"] = now
                bank[item["id"]] = item
                n += 1
            self._save(bank)
            return n

    def add_label(self, **label) -> str:
        self.add_labels([label])
        return self.make_id(label["origin"], label["video_name"], label["claim"], label.get("time_sec"),
                            label.get("kenshi"), label.get("ref", "")) if not label.get("id") else label["id"]

    def remove_origin(self, origin: str, keep_ids: Optional[set] = None) -> int:
        """Remove os rótulos de uma origem que não estão mais na fonte (sincronização)."""
        bank = self.load()
        drop = [k for k, v in bank.items() if v.get("origin") == origin and (keep_ids is None or k not in keep_ids)]
        for k in drop:
            del bank[k]
        if drop:
            self._save(bank)
        return len(drop)

    def labels(self, video_name: Optional[str] = None, claim: Optional[str] = None) -> List[Dict[str, Any]]:
        out = list(self.load().values())
        if video_name is not None:
            name = canonical_video_name(video_name)
            out = [x for x in out if x["video_name"] == name]
        if claim is not None:
            out = [x for x in out if x["claim"] == claim]
        return out

    def stats(self) -> Dict[str, Any]:
        bank = self.load().values()
        by_origin: Dict[str, int] = {}
        by_claim: Dict[str, int] = {}
        for x in bank:
            by_origin[x["origin"]] = by_origin.get(x["origin"], 0) + 1
            by_claim[x["claim"]] = by_claim.get(x["claim"], 0) + 1
        return {"total": sum(by_origin.values()), "por_origem": by_origin, "por_afirmacao": by_claim,
                "videos": len({x["video_name"] for x in bank})}

    # ------------------------------------------------------------------
    # Autoridade e conflitos
    # ------------------------------------------------------------------
    @staticmethod
    def authority(label: Dict[str, Any]) -> float:
        return ORIGIN_BASE_AUTHORITY.get(label.get("origin"), 1.0) * float(label.get("reviewer_weight", 1.0)) * float(label.get("confidence", 1.0))

    # ------------------------------------------------------------------
    # Sincronização com as fontes existentes
    # ------------------------------------------------------------------
    def sync_from_sources(self, labels_path: Optional[str] = None, feedback_path: Optional[str] = None) -> Dict[str, int]:
        """
        Traz para o banco o gabarito (I2) e o feedback_dataset (revisão, Excel, curadoria). Idempotente:
        rótulos que sumiram da fonte saem do banco. Rótulos derivados do próprio gabarito
        (decision_category "rotulo_fraco") são ignorados para não contar o mesmo fato duas vezes.
        """
        from src.engine import weak_labels as wl
        from src.engine.mathematical_calibrator import BayesianCalibrationOptimizer
        counts = {"gabarito": 0, "revisao": 0, "excel": 0, "curadoria": 0, "ignorados_derivados": 0, "ignorados_sem_tempo": 0}

        # Gabarito: cada ponto oficial afirma "ponto=True" (e a técnica, quando informada)
        gab_labels = []
        gabarito = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
        for name, entry in gabarito.items():
            if not wl.is_entry_filled(entry):
                continue
            origin = "bandeiras" if str(entry.get("origem", "")).startswith("automatico") else "gabarito"
            gab_labels.append({"origin": origin, "video_name": name, "claim": "modalidade", "value": LUTA_MODALITY,
                               "time_sec": 0.0, "fim_s": None, "kenshi": None, "ref": "video", "confidence": 1.0})
            for p in wl.parse_official_points(entry):
                if not p["is_strike"] or p["time_sec"] is None:
                    continue
                kenshi = "AKA" if p["kenshi_id"] == "KENSHI_AKA" else ("SHIRO" if p["kenshi_id"] == "KENSHI_SHIRO" else None)
                base = {"origin": origin, "video_name": name, "time_sec": p["time_sec"], "kenshi": kenshi,
                        "ref": f"ponto{p['index']}", "confidence": 1.0, "pontos_completos": bool(entry.get("pontos_completos"))}
                gab_labels.append(dict(base, claim="ponto", value=True))
                tech = wl.normalize_technique(p.get("technique"))
                if tech and origin == "gabarito":
                    gab_labels.append(dict(base, claim="tecnica", value=tech))
        ids = {self.make_id(x["origin"], x["video_name"], x["claim"], x["time_sec"], x["kenshi"], x["ref"]) for x in gab_labels}
        self.remove_origin("gabarito", keep_ids=ids)
        self.remove_origin("bandeiras", keep_ids=ids)
        if gab_labels:
            self.add_labels(gab_labels)
        counts["gabarito"] = len(gab_labels)

        # Feedback: revisão golpe a golpe, Excel e curadoria
        from src.engine.feedback_manager import FeedbackManager
        fm = FeedbackManager(dataset_path=feedback_path) if feedback_path else FeedbackManager()
        fb_labels = []
        for e in fm.load_feedback():
            if e.get("decision_category") in DERIVED_DECISION_CATEGORIES:
                counts["ignorados_derivados"] += 1
                continue
            t = parse_time(e.get("timestamp"))
            if t is None:
                counts["ignorados_sem_tempo"] += 1
                continue
            label = str(e.get("label", "")).upper()
            category = str(e.get("decision_category", "")).upper()
            notes = str(e.get("notes", "")).lower()
            if "curadoria" in notes or str(e.get("id_event", "")).startswith("unc_") or category in ("IPPON", "INVALID_STRIKE", "NO_STRIKE"):
                origin = "curadoria"
            elif "excel" in notes or "planilha" in notes:
                origin = "excel"
            else:
                origin = "revisao"
            if label in ("TP", "CONFIRMED", "VALID", "IPPON", "FN", "INCLUDED"):
                claim, value = "ponto", True
            elif category == "NO_STRIKE":
                claim, value = "golpe", False
            elif label in ("FP", "REJECTED", "INVALID", "NO_IPPON"):
                claim, value = "ponto", False
            else:
                continue
            fb_labels.append({
                "origin": origin, "video_name": e.get("video_name", ""), "claim": claim, "value": value,
                "time_sec": t, "kenshi": None, "ref": str(e.get("id", "")),
                "reviewer_weight": BayesianCalibrationOptimizer.get_reviewer_weight(e),
                "is_shinpan": bool(e.get("is_shinpan_decision")),
                "strike_type": e.get("strike_type"), "sub_scores": e.get("sub_scores") or {},
                "total_score": e.get("total_score"), "fonte": "feedback_dataset",
            })
            counts[origin] += 1
        fb_ids = {self.make_id(x["origin"], x["video_name"], x["claim"], x["time_sec"], x["kenshi"], x["ref"]) for x in fb_labels}
        for origin in ("revisao", "excel", "curadoria"):
            bank = self.load()
            keep = {k for k in fb_ids} | {k for k, v in bank.items() if v.get("origin") == origin and v.get("fonte") != "feedback_dataset"}
            self.remove_origin(origin, keep_ids=keep)
        if fb_labels:
            self.add_labels(fb_labels)
        log_event("INFO", f"Banco de rótulos sincronizado: {counts}", "label_bank")
        return counts

    # ------------------------------------------------------------------
    # Aplicar as correções aos golpes detectados
    # ------------------------------------------------------------------
    def corrections_for(self, video_name: str) -> List[Dict[str, Any]]:
        """Rótulos de golpe (ponto/golpe/técnica) que não vêm do gabarito — eles corrigem os golpes detectados."""
        return [x for x in self.labels(video_name) if x["claim"] in ("ponto", "golpe", "tecnica")
                and x["origin"] not in ("gabarito", "bandeiras") and x.get("time_sec") is not None]

    def apply_corrections(self, items: List[Dict[str, Any]], corrections: List[Dict[str, Any]],
                          tolerance_sec: float = MATCH_TOLERANCE_SEC) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Sobrepõe as correções aos rótulos vindos do gabarito, pelo golpe mais próximo no tempo (e mesmo kenshi,
        se informado). Vence quem tem mais autoridade (o gabarito vale 4.0); empate apertado vira conflito.
        Devolve (itens corrigidos, conflitos).
        """
        from src.engine import weak_labels as wl
        out = [dict(it) for it in items]
        conflicts = []
        for c in sorted(corrections, key=lambda x: -self.authority(x)):
            best, best_dt = None, None
            for i, it in enumerate(out):
                dt = abs(float(it["time_sec"]) - float(c["time_sec"]))
                kenshi_ok = c.get("kenshi") in (None, "") or it.get("attacker_id", "").endswith(str(c["kenshi"]).upper())
                if dt <= tolerance_sec and kenshi_ok and (best_dt is None or dt < best_dt):
                    best, best_dt = i, dt
            if best is None:
                continue
            it = out[best]
            if c["claim"] == "tecnica":
                it["tecnica_corrigida"] = c["value"]
                continue
            new_label = wl.POSITIVE if (c["claim"] == "ponto" and c["value"]) else wl.NEG_CORRECTION
            current_auth = it.get("autoridade_rotulo", ORIGIN_BASE_AUTHORITY["gabarito"] if it["weak_label"] in (wl.POSITIVE, *wl.NEGATIVE_LABELS) else 0.0)
            new_auth = self.authority(c)
            if new_label == it["weak_label"]:
                it["autoridade_rotulo"] = max(current_auth, new_auth)
                continue
            if abs(new_auth - current_auth) < CONFLICT_MARGIN * max(new_auth, current_auth, 1e-9):
                conflicts.append({"video_name": c["video_name"], "time_sec": it["time_sec"], "atual": it["weak_label"],
                                  "correcao": new_label, "origem": c["origin"], "id": c["id"]})
                continue
            if new_auth > current_auth:
                it.update({"weak_label": new_label, "reason": f"corrigido por {c['origin']}",
                           "origem_rotulo": c["origin"], "autoridade_rotulo": new_auth})
        return out, conflicts

    def record_training_segments(self, video_name: str, segments: List[Dict[str, Any]], reviewer_weight: float = 1.0,
                                 whole_video_modality: Optional[str] = None) -> int:
        """
        Correções do Modo Treinamento (I13): cada trecho {start_time, end_time, modality_key} vira um rótulo
        "modalidade" (origem modo_treino), de onde o classificador luta × 14 modalidades (I14) aprende.
        Substitui as correções anteriores do mesmo vídeo (a edição mais recente é a que vale).
        Sem trechos, whole_video_modality rotula o vídeo inteiro.
        """
        name = canonical_video_name(video_name)
        bank = self.load()
        stale = [k for k, v in bank.items() if v.get("origin") == "modo_treino" and v.get("claim") == "modalidade" and v.get("video_name") == name]
        for k in stale:
            del bank[k]
        if stale:
            self._save(bank)
        labels = []
        for i, seg in enumerate(segments or []):
            start = parse_time(seg.get("start_time", seg.get("start_seconds")))
            end = parse_time(seg.get("end_time", seg.get("end_seconds")))
            key = seg.get("modality_key")
            if start is None or not key:
                continue
            labels.append({"origin": "modo_treino", "video_name": name, "claim": "modalidade", "value": key,
                           "time_sec": start, "fim_s": end, "kenshi": None, "ref": f"trecho{i}", "reviewer_weight": reviewer_weight})
        if not labels and whole_video_modality:
            labels.append({"origin": "modo_treino", "video_name": name, "claim": "modalidade", "value": whole_video_modality,
                           "time_sec": 0.0, "fim_s": None, "kenshi": None, "ref": "video", "reviewer_weight": reviewer_weight})
        if labels:
            self.add_labels(labels)
        log_event("INFO", f"Correções de modalidade registradas no banco: {len(labels)} trecho(s) de {name}.", "label_bank")
        return len(labels)

    def modality_labels(self) -> List[Dict[str, Any]]:
        """Rótulos de modalidade (luta ou uma das 14 modalidades) para o classificador da I14."""
        return sorted(self.labels(claim="modalidade"), key=lambda x: (x["video_name"], x.get("time_sec") or 0.0))

    def labeled_videos(self, labels_path: Optional[str] = None) -> List[str]:
        """Vídeos com algo a ensinar: gabarito preenchido ou alguma correção no banco."""
        from src.engine import weak_labels as wl
        gab = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
        names = {canonical_video_name(n) for n, e in gab.items() if wl.is_entry_filled(e)}
        names |= {x["video_name"] for x in self.load().values() if x["claim"] in ("ponto", "golpe", "tecnica")}
        ignored = {canonical_video_name(n) for n, e in gab.items() if wl.is_entry_ignored(e)}
        return sorted(names - ignored)


def build_training_reports(pipeline, videos_dir: str, labels_path: Optional[str] = None,
                           bank: Optional[LabelBank] = None, exclude_golden: bool = True) -> Dict[str, Any]:
    """
    Um relatório rotulado por vídeo, combinando o gabarito (I2) com TODAS as correções do banco.
    É a fonte única que o filtro (I3), a técnica (I9) e a calibração (I4) usam para treinar.
    Vídeos só com correções (sem gabarito) contribuem apenas com os golpes corrigidos.
    """
    from src.engine import weak_labels as wl
    from src.engine.active_learning import is_golden_video
    bank = bank or LabelBank()
    gabarito = wl.load_match_labels(labels_path or wl.DEFAULT_LABELS_PATH)["videos"]
    by_canonical = {canonical_video_name(n): (n, e) for n, e in gabarito.items()}
    reports, skipped, conflicts = [], [], []
    for name in bank.labeled_videos(labels_path):
        if exclude_golden and is_golden_video(name):
            skipped.append({"video_name": name, "motivo": "golden"})
            continue
        from src.engine.internet_collector import resolve_video_path
        video = resolve_video_path(videos_dir, by_canonical.get(name, (name, None))[0])  # inclui os vídeos da coleta (I15)
        if not os.path.isfile(video):
            skipped.append({"video_name": name, "motivo": "vídeo ausente"})
            continue
        entry = by_canonical.get(name, (name, None))[1]
        if not entry or not wl.is_entry_filled(entry):
            entry = {"pontos": [], "pontos_completos": False}
        report = wl.weak_label_video(pipeline, video, entry, invert_combatants=bool(entry.get("inverter_aka_shiro", False)))
        corrections = bank.corrections_for(name)
        for level in ("candidates", "events"):
            report[level]["items"], c = bank.apply_corrections(report[level]["items"], corrections)
            if level == "events":
                conflicts.extend(c)
        report["inverter_aka_shiro"] = bool(entry.get("inverter_aka_shiro", False))
        report["origem_gabarito"] = "automatico" if str(entry.get("origem", "")).startswith("automatico") else ("manual" if entry.get("pontos") or entry.get("pontos_completos") else "sem_gabarito")
        report["correcoes_aplicadas"] = len(corrections)
        reports.append(report)
    return {"reports": reports, "skipped": skipped, "conflitos": conflicts}


def optimizer_feedback(training: Dict[str, Any], bank: Optional[LabelBank] = None) -> List[Dict[str, Any]]:
    """
    Amostras para a calibração de pesos (I4): os golpes avaliados com rótulo conhecido (com as notas reais),
    mais as correções do banco cujo vídeo não pôde ser reanalisado (mas que trazem as notas do golpe).
    """
    from src.engine import weak_labels as wl
    bank = bank or LabelBank()
    out, replayed = [], set()
    for r in training["reports"]:
        replayed.add(canonical_video_name(r["video_name"]))
        for ev in r["events"]["items"]:
            if ev["weak_label"] == wl.POSITIVE:
                label = "TP"
            elif ev["weak_label"] in wl.NEGATIVE_LABELS:
                label = "FP"
            else:
                continue
            if not ev.get("sub_scores"):
                continue
            out.append({"label": label, "sub_scores": ev["sub_scores"], "strike_type": ev.get("strike_type"),
                        "video_name": r["video_name"], "reviewer_dan": "shinpan" if r.get("origem_gabarito") == "manual" else 4,
                        "timestamp": None, "origem": ev.get("origem_rotulo", "gabarito")})
    for lb in bank.labels(claim="ponto"):
        if lb["video_name"] in replayed or not lb.get("sub_scores"):
            continue
        out.append({"label": "TP" if lb["value"] else "FP", "sub_scores": lb["sub_scores"], "strike_type": lb.get("strike_type"),
                    "video_name": lb["video_name"], "reviewer_dan": "shinpan" if lb.get("is_shinpan") else lb.get("reviewer_weight", 1.0),
                    "timestamp": None, "origem": lb["origin"]})
    return out
