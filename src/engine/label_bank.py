"""
Banco Único de Rótulos do SenpAI (Pendência I11).

Toda informação que pode ensinar o sistema vira um RÓTULO neste banco, com ORIGEM e CONFIANÇA:
- gabarito ........ pontos oficiais da luta (I2) — decisão dos shinpan: CONTEXTO, nunca rótulo de validade
- revisao ......... correção golpe a golpe na tela de análise (TP/FP/FN por revisor Dan ou Shinpan)
- excel ........... revisão importada de planilha
- curadoria ....... decisão na Fila de Curadoria Ativa (I12)
- modo_treino ..... correções de modalidade/trecho no Modo Treinamento (I13)
- internet ........ coleta automática (I15)

Cada rótulo afirma uma coisa ("claim") sobre um instante de um vídeo:
- "ponto"      → o golpe vira ponto (True) ou não (False)
- "golpe"      → houve tentativa de golpe (False = detecção espúria)
- "tecnica"    → MEN / KOTE / DO / TSUKI
- "modalidade" → luta ou uma das 14 modalidades (I13/I14)
- "regra"      → observação de regra (hansoku, golpe após o yame...) para a curadoria (I12)

Validade do golpe (ponto ou não) vem SÓ de quem revisa o vídeo (kenshi e senseis, pesados pelo dan): revisão,
Excel e curadoria. A decisão dos shinpan (pontos do gabarito) fica registrada em cada golpe como
"rotulo_shinpan", vira pedido de revisão na curadoria e serve para medir a concordância shinpan × revisão,
mas não treina nem mede o modelo (os shinpan erram nos dois sentidos e o modelo aprenderia o viés deles).
A técnica do ponto oficial (MEN/KOTE/DO/TSUKI) continua valendo como técnica tentada.

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
ORIGINS = ("gabarito", "revisao", "excel", "curadoria", "modo_treino", "internet")
# Peso base de cada origem (multiplicado pelo peso do revisor quando houver um revisor humano: dan 1–8 vale o
# próprio dan, shinpan credenciado que revisa o vídeo 4,5, kyu/anônimo 0,8). Entre revisões que discordam vence
# o dan maior; diferença menor que CONFLICT_MARGIN vai para a curadoria.
# A decisão dos shinpan na luta (pontos do gabarito) NÃO tem autoridade sobre a validade do golpe: ela só
# aponta o que revisar (SHINPAN_ORIGINS). O peso dela aqui vale só para técnica e modalidade.
ORIGIN_BASE_AUTHORITY = {
    "gabarito": 3.0, "revisao": 1.0, "excel": 1.0, "curadoria": 1.0,
    "modo_treino": 1.0, "internet": 0.5,
}
# Origens que registram a decisão dos shinpan (contexto para revisão, nunca rótulo de validade)
SHINPAN_ORIGINS = ("gabarito",)
# Origens com revisão humana do vídeo: as únicas que dizem se o golpe é válido
REVIEW_ORIGINS = ("revisao", "excel", "curadoria")
MATCH_TOLERANCE_SEC = 1.5
# Modalidade: uma origem por vídeo, da mais específica para a menos
MODALITY_ORIGIN_PRIORITY = ("modo_treino", "gabarito", "internet")
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
            tipo = wl.entry_video_type(entry)
            if tipo not in (LUTA_MODALITY, wl.VIDEO_TYPE_DISCARD):
                # Treino ou exame (uma das modalidades): o vídeo inteiro vira rótulo de modalidade
                gab_labels.append({"origin": "gabarito", "video_name": name, "claim": "modalidade", "value": tipo,
                                   "time_sec": 0.0, "fim_s": None, "kenshi": None, "ref": "video", "confidence": 1.0})
                continue
            if tipo == LUTA_MODALITY and entry.get("tipo") and not wl.is_entry_filled(entry):
                # Marcado como luta, mas ainda sem pontos: já ensina a modalidade
                gab_labels.append({"origin": "gabarito", "video_name": name, "claim": "modalidade", "value": LUTA_MODALITY,
                                   "time_sec": 0.0, "fim_s": None, "kenshi": None, "ref": "video", "confidence": 1.0})
                continue
            if not wl.is_entry_filled(entry):
                continue
            origin = "gabarito"
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
                if tech and not str(entry.get("origem", "")).startswith("automatico"):  # gabarito automático antigo: técnica desconhecida
                    gab_labels.append(dict(base, claim="tecnica", value=tech))
        ids = {self.make_id(x["origin"], x["video_name"], x["claim"], x["time_sec"], x["kenshi"], x["ref"]) for x in gab_labels}
        self.remove_origin("gabarito", keep_ids=ids)
        self.remove_origin("bandeiras")  # origem antiga (detector de bandeiras removido)
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
        """Revisões de golpe (ponto/golpe/técnica) feitas por quem assistiu ao vídeo: a única fonte de validade."""
        return [x for x in self.labels(video_name) if x["claim"] in ("ponto", "golpe", "tecnica")
                and x["origin"] not in SHINPAN_ORIGINS and x.get("time_sec") is not None]

    def apply_corrections(self, items: List[Dict[str, Any]], corrections: List[Dict[str, Any]],
                          tolerance_sec: float = MATCH_TOLERANCE_SEC, base_authority: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Sobrepõe as correções aos rótulos vindos do gabarito, pelo golpe mais próximo no tempo (e mesmo kenshi,
        se informado). Vence quem tem mais autoridade; empate apertado vira conflito para a curadoria.
        `base_authority` é a autoridade do rótulo que o golpe já tem (0 em relabel_by_reviews: sem revisão,
        o golpe não tem rótulo). Devolve (itens corrigidos, conflitos).
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
            if c["claim"] == "golpe" and not c["value"]:
                it["nao_e_golpe"] = True
            base = ORIGIN_BASE_AUTHORITY["gabarito"] if base_authority is None else base_authority
            current_auth = it.get("autoridade_rotulo", base if it["weak_label"] in (wl.POSITIVE, *wl.NEGATIVE_LABELS) else 0.0)
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
        """
        Rótulos de modalidade (luta ou uma das modalidades de treino) para o classificador da I14.
        Shiai-geiko conta como luta. Em cada vídeo vale uma origem só, a mais específica: trechos corrigidos
        no Modo Treinamento > tipo escolhido no gabarito > título do vídeo da internet
        (ex.: a sua escolha no gabarito substitui a modalidade adivinhada pelo título).
        """
        from src.engine import weak_labels as wl
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for x in self.labels(claim="modalidade"):
            if x.get("value") in wl.LUTA_ALIASES:
                x = dict(x, value=LUTA_MODALITY)
            by_video.setdefault(x["video_name"], []).append(x)
        out = []
        for lbs in by_video.values():
            rank = lambda o: MODALITY_ORIGIN_PRIORITY.index(o) if o in MODALITY_ORIGIN_PRIORITY else len(MODALITY_ORIGIN_PRIORITY)  # noqa: E731
            best = min(rank(x.get("origin")) for x in lbs)
            out.extend(x for x in lbs if rank(x.get("origin")) == best)
        return sorted(out, key=lambda x: (x["video_name"], x.get("time_sec") or 0.0))

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
    Um relatório rotulado por vídeo. É a fonte única que o filtro (I3), a técnica (I9), a calibração (I4) e a
    referência dos golpes usam para treinar.
    A validade de cada golpe vem só das revisões (relabel_by_reviews); os pontos do gabarito ficam em
    "rotulo_shinpan" e geram pedidos de revisão (review_requests). Golpe sem revisão não treina.
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
        conflicts.extend(relabel_by_reviews(report, corrections, bank))
        report["inverter_aka_shiro"] = bool(entry.get("inverter_aka_shiro", False))
        report["origem_gabarito"] = "automatico" if str(entry.get("origem", "")).startswith("automatico") else ("manual" if entry.get("pontos") or entry.get("pontos_completos") else "sem_gabarito")
        report["correcoes_aplicadas"] = len(corrections)
        reports.append(report)
    return {"reports": reports, "skipped": skipped, "conflitos": conflicts}


def optimizer_feedback(training: Dict[str, Any], bank: Optional[LabelBank] = None) -> List[Dict[str, Any]]:
    """
    Amostras para a calibração de pesos (I4): os golpes revisados (com as notas reais), mais as revisões do
    banco cujo vídeo não pôde ser reanalisado (mas que trazem as notas do golpe). Golden fica de fora.
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
                        # Peso = autoridade da revisão (dan do revisor)
                        "video_name": r["video_name"], "reviewer_dan": float(ev.get("autoridade_rotulo") or 1.0),
                        "timestamp": None, "origem": ev.get("origem_rotulo", "revisao")})
    from src.engine.active_learning import is_golden_video
    for lb in bank.labels(claim="ponto"):
        if (lb["video_name"] in replayed or not lb.get("sub_scores") or lb["origin"] in SHINPAN_ORIGINS
                or is_golden_video(lb["video_name"])):
            continue
        out.append({"label": "TP" if lb["value"] else "FP", "sub_scores": lb["sub_scores"], "strike_type": lb.get("strike_type"),
                    "video_name": lb["video_name"], "reviewer_dan": "shinpan" if lb.get("is_shinpan") else lb.get("reviewer_weight", 1.0),
                    "timestamp": None, "origem": lb["origin"]})
    return out


# ----------------------------------------------------------------------
# Validade pela revisão (não pela decisão dos shinpan)
# ----------------------------------------------------------------------
def shinpan_said_point(item: Dict[str, Any], points_complete: bool) -> Optional[bool]:
    """O que os shinpan decidiram sobre o golpe: True (deram ponto), False (não deram) ou None (não se sabe)."""
    from src.engine import weak_labels as wl
    label = item.get("rotulo_shinpan")
    if label == wl.POSITIVE:
        return True
    if label == wl.NEG_NO_POINT or (points_complete and label == wl.AMBIGUOUS):
        return False
    return None


def relabel_by_reviews(report: Dict[str, Any], corrections: List[Dict[str, Any]],
                       bank: Optional["LabelBank"] = None) -> List[Dict[str, Any]]:
    """
    Troca, no relatório de weak_label_video, o rótulo vindo dos pontos oficiais pelo das revisões.
    - "rotulo_shinpan" guarda o que o casamento com os pontos do gabarito disse (contexto);
    - "weak_label" passa a ser POSITIVO/NEGATIVO_CORRECAO só onde alguém revisou, senão DESCONHECIDO;
    - revisões de ponto válido que nenhum golpe detectado encontrou viram "revisoes_sem_golpe" (perdidos);
    - "concordancia_shinpan": nos golpes revisados em que a decisão dos shinpan é conhecida, quantas batem.
    Devolve os conflitos entre revisões (para a curadoria).
    """
    from src.engine import weak_labels as wl
    bank = bank or LabelBank()
    complete = bool(report.get("pontos_completos"))
    conflicts: List[Dict[str, Any]] = []
    for level in ("candidates", "events"):
        items = []
        for it in report[level]["items"]:
            it = dict(it)
            it.setdefault("rotulo_shinpan", it["weak_label"])
            it.update({"weak_label": wl.UNKNOWN, "reason": "sem revisão"})
            it.pop("autoridade_rotulo", None)
            it.pop("origem_rotulo", None)
            items.append(it)
        items, c = bank.apply_corrections(items, corrections, base_authority=0.0)
        counts: Dict[str, int] = {}
        for it in items:
            counts[it["weak_label"]] = counts.get(it["weak_label"], 0) + 1
        report[level]["items"], report[level]["counts"] = items, counts
        if level == "events":
            conflicts.extend(c)
    events = report["events"]["items"]
    unmatched = 0
    for c in corrections:
        if c["claim"] != "ponto" or not c["value"]:
            continue
        hit = any(e["weak_label"] == wl.POSITIVE and abs(float(e["time_sec"]) - float(c["time_sec"])) <= MATCH_TOLERANCE_SEC
                  for e in events)
        unmatched += int(not hit)
    report["revisoes_sem_golpe"] = unmatched
    compared = agree = 0
    for e in events:
        said = shinpan_said_point(e, complete)
        if said is None or e["weak_label"] not in (wl.POSITIVE, wl.NEG_CORRECTION):
            continue
        compared += 1
        agree += int(said == (e["weak_label"] == wl.POSITIVE))
    report["concordancia_shinpan"] = {"comparados": compared, "concordam": agree}
    report["validade_por"] = "revisao"
    return conflicts


MAX_REVIEW_REQUESTS_PER_VIDEO = 10


def review_requests(report: Dict[str, Any], limit: int = MAX_REVIEW_REQUESTS_PER_VIDEO) -> List[Dict[str, Any]]:
    """
    Golpes que a decisão dos shinpan aponta e ninguém revisou ainda (depois de relabel_by_reviews):
    ponto dos shinpan, ponto dos shinpan que o SenpAI não detectou, e ponto do SenpAI que os shinpan não deram.
    Viram dúvidas "shinpan" na curadoria; a resposta (com o dan) é o rótulo.
    """
    from src.engine import weak_labels as wl
    complete = bool(report.get("pontos_completos"))
    events = report["events"]["items"]
    out = []
    for e in events:
        if e["weak_label"] != wl.UNKNOWN:
            continue
        said = shinpan_said_point(e, complete)
        if said is True:
            reason = "Os shinpan deram ponto neste golpe. Pela técnica, ele é válido?"
        elif said is False and e.get("predicted_valid"):
            reason = "O SenpAI deu ponto e os shinpan não. Pela técnica, o golpe é válido?"
        else:
            continue
        out.append({"time_sec": float(e["time_sec"]), "kenshi": e.get("attacker_id"), "strike_type": e.get("strike_type"),
                    "sub_scores": e.get("sub_scores") or {}, "total_score": e.get("total_score"), "motivo": reason,
                    "prioridade": 1.0 if said is True else 0.9})
    reviewed_times = [float(e["time_sec"]) for e in events if e["weak_label"] != wl.UNKNOWN]
    for p in report["events"].get("missed_points", []):
        if p.get("time_sec") is None or any(abs(t - p["time_sec"]) <= MATCH_TOLERANCE_SEC for t in reviewed_times):
            continue
        tech = wl.normalize_technique(p.get("technique")) or "MEN"
        out.append({"time_sec": float(p["time_sec"]), "kenshi": p.get("kenshi_id"), "strike_type": tech,
                    "sub_scores": {}, "total_score": None, "prioridade": 0.95,
                    "motivo": "Os shinpan deram ponto aqui, mas o SenpAI não detectou o golpe. Houve golpe válido?"})
    out.sort(key=lambda r: (-r["prioridade"], r["time_sec"]))
    return out[:limit]
