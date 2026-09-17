"""
Módulo de Gerenciamento de Feedback, Governança por Dan e Otimização Adaptativa por Reforço.
Gerencia a gravação de marcações (TP, FP, FN, edições e revisões por Dan), estatísticas e retreinamento do sistema.
"""

import json
import os
import re
import datetime
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse
from src.utils.logger_manager import log_event

SHINPAN_REV_KEY: str = "shinpan"
SHINPAN_NAME: str = "Decisão dos Shinpans"
SHINPAN_CALIBRATION_WEIGHT: float = 4.5  # Constante média equilibrada (mediana de 1º a 8º Dan)


class DuplicateShinpanReviewError(ValueError):
    """Exceção levantada ao tentar registrar uma segunda Decisão dos Shinpans para um mesmo link de vídeo."""
    pass


def normalize_video_identifier(url_or_name: Optional[str]) -> str:
    """
    Normaliza a URL ou o identificador de arquivo do vídeo para uma chave canônica única.
    Permite detectar o mesmo vídeo independentemente do formato do link ou parâmetros extras.

    Exemplos:
    - https://www.youtube.com/watch?v=ABC123xyz -> youtube:ABC123xyz
    - https://youtu.be/ABC123xyz -> youtube:ABC123xyz
    - https://www.youtube.com/shorts/ABC123xyz -> youtube:ABC123xyz
    - https://example.com/video.mp4?token=123 -> url:https://example.com/video.mp4
    - upload_172000000_fight.mp4 -> file:fight.mp4
    - fight.mp4 -> file:fight.mp4
    """
    if not url_or_name or not isinstance(url_or_name, str):
        return ""

    val = url_or_name.strip()
    if not val:
        return ""

    # Padrões do YouTube
    yt_patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{6,})",
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{6,})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{6,})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/live\/([a-zA-Z0-9_-]{6,})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{6,})",
    ]
    for pat in yt_patterns:
        m = re.search(pat, val, re.IGNORECASE)
        if m:
            return f"youtube:{m.group(1)}"

    # URLs HTTP/HTTPS genéricas
    if val.startswith("http://") or val.startswith("https://"):
        try:
            parsed = urlparse(val)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            return f"url:{clean_url.rstrip('/').lower()}"
        except Exception:
            return f"url:{val.lower()}"

    # Arquivo local / upload
    basename = os.path.basename(val)
    # Remove prefixo temporário 'upload_\d+_' se presente
    clean_basename = re.sub(r"^upload_\d+_", "", basename)
    return f"file:{clean_basename.lower()}"


def is_shinpan_reviewer(dan_val: Any) -> bool:
    if dan_val is None:
        return False
    if dan_val in [SHINPAN_REV_KEY, "shinpan", "Decisão dos Shinpans", -1, 10]:
        return True
    s = str(dan_val).strip().lower()
    return "shinpan" in s or "árbitro" in s or "arbitro" in s or "decisao dos shinpans" in s or "decisão dos shinpans" in s

DAN_NAMES: Dict[Any, str] = {
    1: "1º Dan (Shodan)",
    2: "2º Dan (Nidan)",
    3: "3º Dan (Sandan)",
    4: "4º Dan (Yondan)",
    5: "5º Dan (Godan)",
    6: "6º Dan (Rokudan)",
    7: "7º Dan (Nanadan)",
    8: "8º Dan (Hachidan)",
    SHINPAN_REV_KEY: SHINPAN_NAME,
    10: SHINPAN_NAME
}

DEFAULT_CALIBRATION_PROFILES: Dict[str, Any] = {
    "permissivo": {
        "name": "Iniciantes / Educacional (Permissivo)",
        "description": "Tolerância ampliada para feedback formativo com praticantes de níveis iniciais.",
        "min_total_score": 0.50,
        "weights": {"target_impact": 0.35, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.20},
        "sub_thresholds": {"target_impact": 0.45, "fumikomi_sync": 0.35, "posture": 0.35, "zanshin": 0.30}
    },
    "normal": {
        "name": "Treino Geral / Keiko (Normal)",
        "description": "Equilíbrio padrão para treinos do dia a dia e avaliações gerais de Keiko.",
        "min_total_score": 0.65,
        "weights": {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15},
        "sub_thresholds": {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45}
    },
    "rigido": {
        "name": "Campeonato / Audit de Dan (Rígido)",
        "description": "Alta exigência em Ki-Ken-Tai-Ichi e Zanshin. Recomendado para exames de graduação e torneios oficiais.",
        "min_total_score": 0.78,
        "weights": {"target_impact": 0.45, "fumikomi_sync": 0.25, "posture": 0.15, "zanshin": 0.15},
        "sub_thresholds": {"target_impact": 0.70, "fumikomi_sync": 0.60, "posture": 0.60, "zanshin": 0.55}
    }
}

class FeedbackManager:
    def __init__(
        self,
        dataset_path: str = "data/feedback_dataset.json",
        history_path: str = "data/training_history.json",
        profiles_path: str = "config/calibration_profiles.json",
        models_dir: str = "models",
        knowledge_base_path: str = "config/ai_knowledge_base.json",
        shinpan_registry_path: str = "data/shinpan_reviewed_videos.json",
        checkpoint_path: str = "data/auto_training_checkpoint.json"
    ):
        self.dataset_path = dataset_path
        self.history_path = history_path
        self.profiles_path = profiles_path
        self.models_dir = models_dir
        self.knowledge_base_path = knowledge_base_path
        self.shinpan_registry_path = shinpan_registry_path
        self.checkpoint_path = checkpoint_path
        self._ensure_files_exist()

    def _ensure_files_exist(self):
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        if not os.path.exists(self.dataset_path):
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        if not os.path.exists(self.history_path):
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

        os.makedirs(os.path.dirname(self.shinpan_registry_path), exist_ok=True)
        if not os.path.exists(self.shinpan_registry_path):
            with open(self.shinpan_registry_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

        os.makedirs(os.path.dirname(self.profiles_path), exist_ok=True)
        if not os.path.exists(self.profiles_path):
            with open(self.profiles_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CALIBRATION_PROFILES, f, indent=2, ensure_ascii=False)

    def load_feedback(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception:
                    return []
        return []

    def load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.history_path):
            with open(self.history_path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception:
                    return []
        return []

    def load_shinpan_reviewed_videos(self) -> List[Dict[str, Any]]:
        """
        Carrega a lista de links/arquivos de vídeos com Decisão dos Shinpans já registrada.
        Sincroniza retroativamente com registros de training_history.json e feedback_dataset.json caso algum
        vídeo histórico de Shinpans ainda não esteja presente no arquivo exclusivo.
        """
        entries: List[Dict[str, Any]] = []
        if os.path.exists(self.shinpan_registry_path):
            try:
                with open(self.shinpan_registry_path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                entries = []

        # Sincronização retroativa com training_history.json
        history = self.load_history()
        registered_ids = {e.get("video_identifier") for e in entries if e.get("video_identifier")}
        history_added = False

        for h in history:
            is_sh = bool(h.get("is_shinpan_decision") or is_shinpan_reviewer(h.get("reviewer_dan")))
            if is_sh:
                v_url = h.get("video_url") or h.get("streaming_url") or ""
                v_name = h.get("video_name") or ""
                canon_id = normalize_video_identifier(v_url) if v_url else normalize_video_identifier(v_name)
                if canon_id and canon_id not in registered_ids:
                    new_rec = {
                        "video_identifier": canon_id,
                        "video_url": v_url or v_name,
                        "video_name": v_name,
                        "session_id": h.get("id", ""),
                        "reviewed_at": h.get("timestamp", ""),
                        "reviewer_dan": SHINPAN_REV_KEY,
                        "reviewer_dan_name": SHINPAN_NAME,
                        "profile_key": h.get("profile_key", "normal"),
                        "items_count": h.get("items_count", 0)
                    }
                    entries.append(new_rec)
                    registered_ids.add(canon_id)
                    history_added = True

        # Sincronização retroativa complementar com feedback_dataset.json
        feedbacks = self.load_feedback()
        for fb in feedbacks:
            is_sh = bool(fb.get("is_shinpan_decision") or is_shinpan_reviewer(fb.get("reviewer_dan")))
            if is_sh:
                v_url = fb.get("video_url") or fb.get("streaming_url") or ""
                v_name = fb.get("video_name") or ""
                canon_id = normalize_video_identifier(v_url) if v_url else normalize_video_identifier(v_name)
                if canon_id and canon_id not in registered_ids:
                    new_rec = {
                        "video_identifier": canon_id,
                        "video_url": v_url or v_name,
                        "video_name": v_name,
                        "session_id": fb.get("id", ""),
                        "reviewed_at": fb.get("review_date", ""),
                        "reviewer_dan": SHINPAN_REV_KEY,
                        "reviewer_dan_name": SHINPAN_NAME,
                        "profile_key": fb.get("profile_key", "normal"),
                        "items_count": 1
                    }
                    entries.append(new_rec)
                    registered_ids.add(canon_id)
                    history_added = True

        if history_added and os.path.exists(os.path.dirname(self.shinpan_registry_path)):
            try:
                with open(self.shinpan_registry_path, "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        return entries

    def is_video_reviewed_by_shinpan(
        self,
        video_url: Optional[str] = None,
        video_name: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verifica se o link ou nome do vídeo já possui uma Decisão dos Shinpans homologada.
        Retorna (True, record) se já foi avaliado pelos Shinpans, ou (False, None) caso contrário.
        """
        cand_ids = set()
        if video_url:
            cand_ids.add(normalize_video_identifier(video_url))
        if video_name:
            cand_ids.add(normalize_video_identifier(video_name))
        cand_ids.discard("")

        if not cand_ids:
            return False, None

        entries = self.load_shinpan_reviewed_videos()
        for entry in entries:
            e_id = entry.get("video_identifier")
            if e_id and e_id in cand_ids:
                return True, entry
            if video_url and entry.get("video_url") and entry.get("video_url").strip().lower() == video_url.strip().lower():
                return True, entry
            if video_name and entry.get("video_name") and entry.get("video_name").strip().lower() == video_name.strip().lower():
                return True, entry

        return False, None

    def register_shinpan_review(
        self,
        video_identifier: str,
        video_url: str,
        video_name: str,
        session_id: str,
        reviewed_at: str,
        items_count: int,
        profile_key: str
    ) -> Dict[str, Any]:
        """
        Registra oficialmente o link/arquivo do vídeo na lista de Decisões dos Shinpans.
        """
        entries = self.load_shinpan_reviewed_videos()
        record = {
            "video_identifier": video_identifier,
            "video_url": video_url,
            "video_name": video_name,
            "session_id": session_id,
            "reviewed_at": reviewed_at,
            "reviewer_dan": SHINPAN_REV_KEY,
            "reviewer_dan_name": SHINPAN_NAME,
            "profile_key": profile_key,
            "items_count": items_count
        }

        idx_found = None
        for i, e in enumerate(entries):
            if e.get("video_identifier") == video_identifier:
                idx_found = i
                break
        if idx_found is not None:
            entries[idx_found] = record
        else:
            entries.append(record)

        with open(self.shinpan_registry_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

        log_event(
            "INFO",
            f"VÍDEO REGISTRADO EM DECISÃO DOS SHINPANS: Link/ID='{video_identifier}', Sessão='{session_id}', Ippons={items_count}",
            "feedback_manager"
        )
        return record

    def save_feedback(
        self,
        video_name: str,
        profile_key: str,
        event_id: str,
        label: str,  # "TP", "FP", "FN", "CONFIRMED", "EDITED", "INCLUDED"
        sub_scores: Optional[Dict[str, Any]] = None,
        total_score: float = 0.0,
        strike_type: str = "MEN",
        timestamp: str = "00:00.000",
        notes: str = "",
        reviewer_dan: Any = 1,
        is_edited: bool = False,
        is_included: bool = False,
        decision_category: str = "",
        is_shinpan_decision: Optional[bool] = None,
        video_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adiciona ou atualiza uma anotação de feedback no dataset com registro de Dan ou Decisão dos Shinpans.
        """
        data = self.load_feedback()
        is_shinpan = (
            is_shinpan_decision is True or
            is_shinpan_reviewer(reviewer_dan)
        )
        if is_shinpan:
            dan_val = SHINPAN_REV_KEY
            dan_name = SHINPAN_NAME
        else:
            try:
                dan_int = int(reviewer_dan)
                dan_val = max(1, min(8, dan_int))
            except Exception:
                dan_val = 1
            dan_name = DAN_NAMES.get(dan_val, f"{dan_val}º Dan")

        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        entry = {
            "id": f"{video_name}_{event_id}_{len(data)+1}",
            "id_event": event_id,
            "video_name": video_name,
            "video_url": video_url or "",
            "profile_key": profile_key,
            "label": label,
            "decision_category": decision_category,
            "strike_type": strike_type,
            "timestamp": timestamp,
            "total_score": total_score,
            "sub_scores": sub_scores or {},
            "notes": notes,
            "reviewer_dan": dan_val,
            "reviewer_dan_name": dan_name,
            "is_shinpan_decision": is_shinpan,
            "review_date": now_iso,
            "is_edited": is_edited,
            "is_included": is_included
        }

        updated = False
        for idx, item in enumerate(data):
            if item.get("video_name") == video_name and item.get("id_event") == event_id:
                entry["id"] = item.get("id", entry["id"])
                data[idx] = entry
                updated = True
                break

        if not updated:
            data.append(entry)

        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return entry

    def save_review_session(
        self,
        video_name: str,
        profile_key: str,
        reviewer_dan: Any,
        review_items: List[Dict[str, Any]],
        current_profile_config: Dict[str, Any],
        video_url: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Salva uma sessão de revisão de detecção gravada, atualiza os dados por Dan ou Shinpans,
        executa o retreinamento do modelo e grava no histórico de treinamentos.

        Regras de Governança:
        - Decisão dos Shinpans: Registra o link/arquivo do vídeo e NÃO permite 2 entradas com o mesmo link de vídeo.
        - Revisão por Dan (1º ao 8º Dan): NÃO possui restrição de entradas duplicadas.
        """
        is_shinpan = is_shinpan_reviewer(reviewer_dan)
        if is_shinpan:
            dan_val = SHINPAN_REV_KEY
            dan_name = SHINPAN_NAME

            # REGRA DE GOVERNANÇA: Bloqueio estrito de entradas duplicadas para Decisão dos Shinpans
            already_reviewed, prior_record = self.is_video_reviewed_by_shinpan(video_url=video_url, video_name=video_name)
            if already_reviewed and prior_record:
                p_date = prior_record.get("reviewed_at", "sessão anterior")
                p_sess = prior_record.get("session_id", "")
                p_url = prior_record.get("video_url") or video_url or video_name
                raise DuplicateShinpanReviewError(
                    f"Entrada duplicada bloqueada: O link/vídeo '{p_url}' já possui uma Decisão dos Shinpans registrada "
                    f"em {p_date} (Sessão: {p_sess}). "
                    f"Conforme as regras de governança, cada link de vídeo só pode receber 1 única Decisão dos Shinpans. "
                    f"Para múltiplas revisões ou estudos técnicos deste vídeo, utilize a Revisão por Dan (1º ao 8º Dan)."
                )
        else:
            # Revisão por Dan: Sem restrição para entradas duplicadas
            try:
                dan_int = int(reviewer_dan)
                dan_val = max(1, min(8, dan_int))
            except Exception:
                dan_val = 1
            dan_name = DAN_NAMES.get(dan_val, f"{dan_val}º Dan")

        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        saved_entries = []
        for item in review_items:
            item_url = video_url or item.get("video_url") or item.get("streaming_url") or ""
            entry = self.save_feedback(
                video_name=video_name,
                profile_key=profile_key,
                event_id=item.get("event_id", f"ev_{len(saved_entries)+1}"),
                label=item.get("label", "TP"),
                sub_scores=item.get("sub_scores", {}),
                total_score=item.get("total_score", 0.0),
                strike_type=item.get("strike_type", "MEN"),
                timestamp=item.get("timestamp", "00:00.000"),
                notes=item.get("notes", ""),
                reviewer_dan=dan_val,
                is_edited=item.get("is_edited", False),
                is_included=item.get("is_included", False),
                decision_category=str(item.get("decision_category") or item.get("category") or ""),
                is_shinpan_decision=is_shinpan,
                video_url=item_url
            )
            saved_entries.append(entry)

        # Recalibrar / retreinar o modelo com os novos dados
        new_config, opt_summary = self.optimize_profile_config(profile_key, current_profile_config)

        # Registrar o evento de treinamento no histórico
        history = self.load_history()
        session_id = f"train_{now_iso.replace(':', '').replace('-', '')}_{len(history)+1}"
        canon_id = normalize_video_identifier(video_url) if video_url else normalize_video_identifier(video_name)

        session_record = {
            "id": session_id,
            "timestamp": now_iso,
            "reviewer_dan": dan_val,
            "reviewer_dan_name": dan_name,
            "is_shinpan_decision": is_shinpan,
            "video_name": video_name,
            "video_url": video_url or "",
            "video_identifier": canon_id,
            "profile_key": profile_key,
            "items_count": len(saved_entries),
            "optimization_summary": opt_summary
        }
        history.append(session_record)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # Registra link na lista de vídeos com Decisão dos Shinpans
        if is_shinpan and canon_id:
            self.register_shinpan_review(
                video_identifier=canon_id,
                video_url=video_url or video_name,
                video_name=video_name,
                session_id=session_id,
                reviewed_at=now_iso,
                items_count=len(saved_entries),
                profile_key=profile_key
            )

        return new_config, session_record

    def get_stats(self, profile_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Retorna estatísticas de anotações (TP, FP, FN) gerais ou filtradas por perfil.
        """
        data = self.load_feedback()
        if profile_key:
            data = [d for d in data if d.get("profile_key") == profile_key]

        total = len(data)
        tp = sum(1 for d in data if d.get("label") in ["TP", "CONFIRMED"])
        fp = sum(1 for d in data if d.get("label") == "FP")
        fn = sum(1 for d in data if d.get("label") in ["FN", "INCLUDED"])

        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0

        return {
            "total_feedback": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision_pct": round(precision, 1),
            "recall_pct": round(recall, 1)
        }

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Calcula as métricas de governança para o Menu de Configurações:
        - Contador total de treinamentos realizados (Humanos Dan + Decisão dos Shinpans + IA).
        - Nível médio (Dan) dos treinamentos humanos (1º ao 8º Dan, isolado e sem distorção).
        - Tabela de quantidade de treinamentos agrupada por Dan (1º a 8º Dan) + Decisão dos Shinpans + Treinamentos Automatizados por IA.
        """
        history = self.load_history()
        data = self.load_feedback()

        total_trainings = len(history)

        dan_counts = {dan: 0 for dan in range(1, 9)}
        auto_trainings_count = 0
        shinpan_trainings_count = 0

        dan_sum = 0
        human_weight_count = 0

        if total_trainings > 0:
            for session in history:
                is_auto = (
                    session.get("is_auto_training", False) or
                    session.get("reviewer_dan") == 0 or
                    session.get("reviewer_dan_name") == "Treinamento Automático por IA (Web & Vídeo)" or
                    session.get("reviewer_dan_name") == "Treinamento Automático por IA" or
                    session.get("optimization_summary", {}).get("mode") == "auto_training_ai" or
                    str(session.get("id", "")).startswith("auto_train_") or
                    str(session.get("video_name", "")).startswith("AI_Auto_Trainer_")
                )
                is_shinpan = (
                    session.get("is_shinpan_decision", False) or
                    is_shinpan_reviewer(session.get("reviewer_dan")) or
                    session.get("reviewer_dan_name") in [SHINPAN_NAME, "Decisão dos Shinpans (Árbitros de Shiai)"]
                )
                if is_auto:
                    auto_trainings_count += 1
                elif is_shinpan:
                    shinpan_trainings_count += 1
                else:
                    dan = session.get("reviewer_dan", 1)
                    if isinstance(dan, int) and 1 <= dan <= 8:
                        dan_counts[dan] += 1
                        dan_sum += dan
                        human_weight_count += 1
        elif data:
            # Fallback para contar revisões se o histórico estiver vazio
            for item in data:
                if item.get("is_shinpan_decision") or is_shinpan_reviewer(item.get("reviewer_dan")):
                    shinpan_trainings_count += 1
                else:
                    dan = item.get("reviewer_dan", 1)
                    if isinstance(dan, int) and 1 <= dan <= 8:
                        dan_counts[dan] += 1
                        dan_sum += dan
                        human_weight_count += 1
            total_trainings = human_weight_count + shinpan_trainings_count

        avg_dan = (dan_sum / human_weight_count) if human_weight_count > 0 else 0.0
        avg_dan_round = round(avg_dan, 1)
        avg_dan_int = max(1, min(8, round(avg_dan))) if avg_dan > 0 else 1

        if human_weight_count > 0:
            avg_dan_label = f"{avg_dan_round}º Dan ({DAN_NAMES.get(avg_dan_int, '')})"
        elif shinpan_trainings_count > 0:
            avg_dan_label = "Decisão dos Shinpans (Árbitros de Shiai)"
        elif auto_trainings_count > 0:
            avg_dan_label = "Treinamento Automático por IA (Sem revisor humano)"
        else:
            avg_dan_label = "Nenhum treinamento"

        shinpan_items_count = sum(
            1 for d in data if d.get("is_shinpan_decision") or is_shinpan_reviewer(d.get("reviewer_dan"))
        )

        table_data = []
        for dan in range(1, 9):
            cnt = dan_counts[dan]
            pct = round((cnt / total_trainings) * 100, 1) if total_trainings > 0 else 0.0
            table_data.append({
                "Dan": f"{dan}º Dan",
                "Nome Graduação": DAN_NAMES[dan],
                "Quantidade Treinamentos": cnt,
                "Percentual (%)": f"{pct}%"
            })

        # Linha dedicada para Decisão dos Shinpans (Árbitros de Shiai)
        shinpan_pct = round((shinpan_trainings_count / total_trainings) * 100, 1) if total_trainings > 0 else 0.0
        table_data.append({
            "Dan": "⚖️ Shinpans",
            "Nome Graduação": "Decisão dos Shinpans (Árbitros de Shiai)",
            "Quantidade Treinamentos": shinpan_trainings_count,
            "Percentual (%)": f"{shinpan_pct}%"
        })

        # Linha dedicada para Treinamentos Automatizados por IA
        auto_pct = round((auto_trainings_count / total_trainings) * 100, 1) if total_trainings > 0 else 0.0
        table_data.append({
            "Dan": "🤖 IA",
            "Nome Graduação": "Treinamentos Automatizados (IA / Web & Vídeo)",
            "Quantidade Treinamentos": auto_trainings_count,
            "Percentual (%)": f"{auto_pct}%"
        })

        storage_info = self.get_training_storage_info()

        return {
            "total_trainings_count": total_trainings,
            "human_trainings_count": human_weight_count,
            "shinpan_trainings_count": shinpan_trainings_count,
            "shinpan_items_count": shinpan_items_count,
            "auto_trainings_count": auto_trainings_count,
            "average_dan_level": avg_dan_round,
            "average_dan_label": avg_dan_label,
            "total_review_items": len(data),
            "dan_distribution": table_data,
            "storage_info": storage_info
        }

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        """Formata uma quantidade de bytes em string legível (B, KB, MB, GB)."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    def get_training_storage_info(
        self,
        models_dir: Optional[str] = None,
        knowledge_base_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcula o espaço em disco ocupado atualmente pelo treinamento do sistema:
        - Datasets de Feedback e Marcações de Revisão (data/feedback_dataset.json)
        - Histórico de Treinamento e Sessões de Retreinamento (data/training_history.json)
        - Pesos de Modelos de Rede Neural de IA (models/ e raiz, ex: yolov8n-pose.pt)
        - Base de Conhecimento e Memória da IA (config/ai_knowledge_base.json)
        - Perfis de Calibração e Aprendizado Biomecânico (config/calibration_profiles.json)

        Retorna totais consolidados em bytes e formatados, métricas por categoria e arquivos detalhados.
        """
        target_models_dir = models_dir or getattr(self, "models_dir", "models")
        target_kb_path = knowledge_base_path or getattr(self, "knowledge_base_path", "config/ai_knowledge_base.json")

        files_detail: List[Dict[str, Any]] = []
        datasets_bytes = 0
        models_bytes = 0
        knowledge_bytes = 0

        # 1. Datasets & Histórico (data/)
        dataset_paths = [self.dataset_path, self.history_path, self.shinpan_registry_path]
        data_folder = os.path.dirname(self.dataset_path) or "data"
        seen_paths = set()

        for p in dataset_paths:
            norm_p = os.path.normpath(p)
            seen_paths.add(norm_p)
            if os.path.exists(p):
                sz = os.path.getsize(p)
                datasets_bytes += sz
                if p == self.dataset_path:
                    desc = "Dataset de Feedbacks & Marcações Dan"
                elif p == self.shinpan_registry_path:
                    desc = "Registro de Vídeos com Decisão dos Shinpans"
                else:
                    desc = "Histórico de Sessões de Retreinamento"
                files_detail.append({
                    "name": os.path.basename(p),
                    "path": p.replace("\\", "/"),
                    "category": "Datasets & Feedbacks",
                    "category_key": "datasets",
                    "bytes": sz,
                    "formatted": self._format_bytes(sz),
                    "description": desc,
                    "exists": True
                })
            else:
                files_detail.append({
                    "name": os.path.basename(p),
                    "path": p.replace("\\", "/"),
                    "category": "Datasets & Feedbacks",
                    "category_key": "datasets",
                    "bytes": 0,
                    "formatted": "0 B",
                    "description": "Arquivo não inicializado",
                    "exists": False
                })

        # Outros arquivos .json no diretório data (ex: backups, pacotes importados)
        if os.path.isdir(data_folder):
            try:
                for fname in sorted(os.listdir(data_folder)):
                    fpath = os.path.join(data_folder, fname)
                    norm_f = os.path.normpath(fpath)
                    if norm_f not in seen_paths and os.path.isfile(fpath) and fname.endswith(".json") and not fname.startswith("test_"):
                        sz = os.path.getsize(fpath)
                        datasets_bytes += sz
                        seen_paths.add(norm_f)
                        files_detail.append({
                            "name": fname,
                            "path": fpath.replace("\\", "/"),
                            "category": "Datasets & Feedbacks",
                            "category_key": "datasets",
                            "bytes": sz,
                            "formatted": self._format_bytes(sz),
                            "description": "Dados adicionais / Pacote importado",
                            "exists": True
                        })
            except Exception:
                pass

        # 2. Modelos de Rede Neural de IA (models/)
        model_seen = set()
        if os.path.isdir(target_models_dir):
            try:
                for fname in sorted(os.listdir(target_models_dir)):
                    fpath = os.path.join(target_models_dir, fname)
                    if os.path.isfile(fpath) and (
                        fname.endswith(".pt") or fname.endswith(".onnx") or fname.endswith(".engine") or fname.endswith(".bin")
                    ):
                        sz = os.path.getsize(fpath)
                        models_bytes += sz
                        model_seen.add(fname)
                        files_detail.append({
                            "name": fname,
                            "path": fpath.replace("\\", "/"),
                            "category": "Modelos de IA & Pesos Neurais",
                            "category_key": "models",
                            "bytes": sz,
                            "formatted": self._format_bytes(sz),
                            "description": "Rede Neural YOLOv8 / Pose Estimation",
                            "exists": True
                        })
            except Exception:
                pass

        # Se yolov8n-pose.pt estiver na raiz do projeto e ainda não contabilizado em models/
        root_model = "yolov8n-pose.pt"
        if "yolov8n-pose.pt" not in model_seen and os.path.isfile(root_model):
            sz = os.path.getsize(root_model)
            models_bytes += sz
            files_detail.append({
                "name": root_model,
                "path": root_model,
                "category": "Modelos de IA & Pesos Neurais",
                "category_key": "models",
                "bytes": sz,
                "formatted": self._format_bytes(sz),
                "description": "Rede Neural YOLOv8 / Pose Estimation",
                "exists": True
            })

        # 3. Base de Conhecimento e Calibração (config/)
        config_paths = [self.profiles_path, target_kb_path]
        for cp in config_paths:
            norm_cp = os.path.normpath(cp)
            if os.path.exists(cp):
                sz = os.path.getsize(cp)
                knowledge_bytes += sz
                desc = "Base de Conhecimento Técnico da IA (FIK/ZNKR)" if cp == target_kb_path else "Perfis de Calibração Biomecânica"
                files_detail.append({
                    "name": os.path.basename(cp),
                    "path": cp.replace("\\", "/"),
                    "category": "Conhecimento & Calibração",
                    "category_key": "knowledge_config",
                    "bytes": sz,
                    "formatted": self._format_bytes(sz),
                    "description": desc,
                    "exists": True
                })
            else:
                files_detail.append({
                    "name": os.path.basename(cp),
                    "path": cp.replace("\\", "/"),
                    "category": "Conhecimento & Calibração",
                    "category_key": "knowledge_config",
                    "bytes": 0,
                    "formatted": "0 B",
                    "description": "Configuração padrão inicial",
                    "exists": False
                })

        total_bytes = datasets_bytes + models_bytes + knowledge_bytes

        categories = {
            "datasets": {
                "name": "Datasets & Feedbacks",
                "folder": data_folder.replace("\\", "/") + "/",
                "bytes": datasets_bytes,
                "formatted": self._format_bytes(datasets_bytes),
                "description": "Feedbacks de marcações, histórico de sessões e registros de vídeos dos Shinpans"
            },
            "models": {
                "name": "Modelos de IA & Pesos Neurais",
                "folder": target_models_dir.replace("\\", "/") + "/",
                "bytes": models_bytes,
                "formatted": self._format_bytes(models_bytes),
                "description": "Pesos neurais do YOLOv8 Pose e modelos de IA do sistema"
            },
            "knowledge_config": {
                "name": "Conhecimento & Calibração",
                "folder": (os.path.dirname(self.profiles_path) or "config").replace("\\", "/") + "/",
                "bytes": knowledge_bytes,
                "formatted": self._format_bytes(knowledge_bytes),
                "description": "Memória técnica do Auto-Trainer (FIK/ZNKR) e perfis calibrados"
            }
        }

        return {
            "total_bytes": total_bytes,
            "total_formatted": self._format_bytes(total_bytes),
            "categories": categories,
            "files": files_detail
        }

    def reset_all_training_data(self) -> None:
        """
        Apaga todo o treinamento do sistema, limpando conjuntos de dados,
        registro de vídeos dos Shinpans e restaurando os perfis de calibração para a configuração padrão.
        """
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

        with open(self.shinpan_registry_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

        os.makedirs(os.path.dirname(self.profiles_path), exist_ok=True)
        with open(self.profiles_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CALIBRATION_PROFILES, f, indent=2, ensure_ascii=False)

        log_event("WARNING", "TREINAMENTO APAGADO (RESET): Todo o histórico de revisões, dataset de feedbacks, registro de vídeos dos Shinpans e calibrações foram restaurados ao estágio inicial de fábrica.", "feedback_manager")

    def _merge_ai_knowledge_base(
        self,
        imported_kb: Dict[str, Any],
        imported_checkpoint: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Mescla uma base de conhecimento importada diretamente no arquivo de configuração
        self.knowledge_base_path de forma cumulativa e segura.
        """
        if not isinstance(imported_kb, dict):
            return {"status": "skipped", "reason": "invalid_kb_format"}

        current_kb = {}
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
                    current_kb = json.load(f)
            except Exception:
                current_kb = {}

        if not current_kb:
            current_kb = json.loads(json.dumps(imported_kb))
            os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
            with open(self.knowledge_base_path, "w", encoding="utf-8") as f:
                json.dump(current_kb, f, indent=2, ensure_ascii=False)
            return {"status": "success", "imported": True, "created_new": True}

        # 1. Mesclar fontes indexadas
        existing_sources = current_kb.setdefault("sources", {})
        imported_sources = imported_kb.get("sources", {})
        sources_added = 0
        if isinstance(imported_sources, dict):
            for s_key, s_val in imported_sources.items():
                if s_key not in existing_sources and not any(
                    isinstance(es, dict) and es.get("title") == (s_val.get("title") if isinstance(s_val, dict) else "")
                    for es in existing_sources.values()
                ):
                    existing_sources[s_key] = s_val
                    sources_added += 1
        elif isinstance(imported_sources, list):
            for s_val in imported_sources:
                if isinstance(s_val, dict):
                    stitle = s_val.get("title", "")
                    skey = stitle.lower().replace(" ", "_")[:40] if stitle else f"src_imp_{random.randint(1000, 9999)}"
                    if skey not in existing_sources and not any(
                        isinstance(es, dict) and es.get("title") == stitle
                        for es in existing_sources.values()
                    ):
                        existing_sources[skey] = s_val
                        sources_added += 1

        current_kb["sources"] = existing_sources
        current_kb["total_web_sources_indexed"] = len(existing_sources)

        # 2. Mesclar parâmetros aprendidos por modalidade
        cur_learned = current_kb.setdefault("learned_parameters", {})
        cur_mods = cur_learned.setdefault("training_modalities", {})
        imp_learned = imported_kb.get("learned_parameters", {})
        imp_mods = imp_learned.get("training_modalities", {}) if isinstance(imp_learned, dict) else {}

        modalities_updated = 0
        if isinstance(imp_mods, dict):
            for mod_k, imp_data in imp_mods.items():
                if not isinstance(imp_data, dict):
                    continue
                if mod_k not in cur_mods:
                    cur_mods[mod_k] = json.loads(json.dumps(imp_data))
                    modalities_updated += 1
                else:
                    cur_m = cur_mods[mod_k]
                    cur_acc = float(cur_m.get("current_accuracy", 0.0))
                    imp_acc = float(imp_data.get("current_accuracy", 0.0))
                    if imp_acc > cur_acc:
                        cur_m["current_accuracy"] = imp_acc
                        cur_m["last_calibrated"] = imp_data.get("last_calibrated", cur_m.get("last_calibrated", ""))
                        cur_m["mastery_level"] = imp_data.get("mastery_level", cur_m.get("mastery_level", ""))
                        modalities_updated += 1

                    cur_principles = cur_m.setdefault("principles_learned", [])
                    imp_principles = imp_data.get("principles_learned", [])
                    if isinstance(imp_principles, list):
                        for p in imp_principles:
                            if p and p not in cur_principles:
                                cur_principles.append(p)

                    cur_web = cur_m.setdefault("web_sources", [])
                    imp_web = imp_data.get("web_sources", [])
                    if isinstance(imp_web, list):
                        for ws in imp_web:
                            if isinstance(ws, dict):
                                w_title = ws.get("title", "")
                                if w_title and not any(isinstance(cw, dict) and cw.get("title") == w_title for cw in cur_web):
                                    cur_web.append(ws)

                    cur_evo = cur_m.setdefault("evolution_log", [])
                    imp_evo = imp_data.get("evolution_log", [])
                    if isinstance(imp_evo, list):
                        for el in imp_evo:
                            if isinstance(el, dict):
                                el_note = el.get("note", "")
                                if el_note and not any(isinstance(ce, dict) and ce.get("note") == el_note for ce in cur_evo):
                                    cur_evo.append(el)

                    cur_m["sessions_count"] = max(int(cur_m.get("sessions_count", 0)), int(imp_data.get("sessions_count", 0)))

                    for param_key in ["movement_weight", "precision_weight", "constancy_weight", "cadence_tolerance_pct", "posture_strictness"]:
                        if param_key in imp_data and imp_acc >= cur_acc:
                            cur_m[param_key] = imp_data[param_key]

        # 3. Mesclar princípios gerais de Kendo
        cur_gen = cur_learned.setdefault("general_kendo_principles", [])
        imp_gen = imp_learned.get("general_kendo_principles", []) if isinstance(imp_learned, dict) else []
        if isinstance(imp_gen, list):
            for gp in imp_gen:
                if gp and gp not in cur_gen:
                    cur_gen.append(gp)

        # 4. Total de sessões e datas
        imp_sessions = int(imported_kb.get("training_sessions_completed", 0))
        cur_sessions = int(current_kb.get("training_sessions_completed", 0))
        current_kb["training_sessions_completed"] = max(cur_sessions, imp_sessions)
        current_kb["last_retrained_at"] = imported_kb.get("last_retrained_at", datetime.datetime.now().isoformat())

        os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
        with open(self.knowledge_base_path, "w", encoding="utf-8") as f:
            json.dump(current_kb, f, indent=2, ensure_ascii=False)

        # 5. Checkpoint
        if imported_checkpoint and isinstance(imported_checkpoint, dict):
            ckpt_path = getattr(self, "checkpoint_path", "data/auto_training_checkpoint.json")
            if not os.path.exists(ckpt_path):
                try:
                    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump(imported_checkpoint, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        return {
            "status": "success",
            "sources_added": sources_added,
            "modalities_updated": modalities_updated,
            "total_sources_now": len(existing_sources),
            "sessions_completed": current_kb["training_sessions_completed"]
        }

    def export_training_package(self, auto_trainer_instance: Optional[Any] = None) -> Dict[str, Any]:
        """
        Exporta o pacote de treinamento completo contendo:
        1. Todas as anotações e revisões por Dan (1º ao 8º Dan) e Shinpans;
        2. Histórico completo de treinamentos (humanos e automáticos);
        3. Registro de todos os vídeos e links de streaming com Decisão dos Shinpans homologada;
        4. Perfis calibrados do modelo de detecção;
        5. Base de conhecimento completa de IA (ai_knowledge_base: 14 modalidades, princípios, acurácia, fontes web);
        6. Checkpoint de auto-treinamento (se existir).
        """
        data = self.load_feedback()
        history = self.load_history()
        shinpan_videos = self.load_shinpan_reviewed_videos()
        metrics = self.get_training_metrics()

        profiles = DEFAULT_CALIBRATION_PROFILES.copy()
        if os.path.exists(self.profiles_path):
            try:
                with open(self.profiles_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
            except Exception:
                pass

        # Obter base de conhecimento da IA
        ai_kb = None
        if auto_trainer_instance is not None and hasattr(auto_trainer_instance, "load_knowledge_base"):
            try:
                ai_kb = auto_trainer_instance.load_knowledge_base()
            except Exception:
                pass

        if ai_kb is None and os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
                    ai_kb = json.load(f)
            except Exception:
                pass

        # Obter checkpoint de auto-treinamento
        auto_ckpt = None
        if auto_trainer_instance is not None and hasattr(auto_trainer_instance, "load_checkpoint"):
            try:
                auto_ckpt = auto_trainer_instance.load_checkpoint()
            except Exception:
                pass

        ckpt_path = getattr(self, "checkpoint_path", "data/auto_training_checkpoint.json")
        if auto_ckpt is None and os.path.exists(ckpt_path):
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    auto_ckpt = json.load(f)
            except Exception:
                pass

        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        # Contagem de sessões automáticas
        auto_trainings_count = sum(1 for h in history if h.get("is_auto_training") or str(h.get("video_name", "")).startswith("AI_Auto_Trainer_"))
        indexed_sources_count = len(ai_kb.get("sources", {})) if (ai_kb and isinstance(ai_kb.get("sources"), dict)) else 0

        pkg = {
            "system_name": "SenpAI",
            "package_version": "2.0",
            "exported_at": now_iso,
            "summary": {
                "total_trainings_performed": metrics["total_trainings_count"],
                "average_reviewer_dan": metrics["average_dan_level"],
                "average_dan_label": metrics["average_dan_label"],
                "total_review_entries": len(data),
                "shinpan_reviewed_videos_count": len(shinpan_videos),
                "auto_trainings_count": auto_trainings_count,
                "indexed_sources_count": indexed_sources_count
            },
            "review_items": data,
            "training_history": history,
            "shinpan_reviewed_videos": shinpan_videos,
            "calibration_profiles": profiles,
            "ai_knowledge_base": ai_kb,
            "auto_training_checkpoint": auto_ckpt
        }

        log_event(
            "INFO",
            f"PACOTE DE TREINAMENTO EXPORTADO: Pacote v2.0 gerado com {len(data)} itens de revisão, "
            f"{len(history)} sessões de treino, {len(shinpan_videos)} vídeos/links de streaming dos Shinpans e Base de Conhecimento de IA.",
            "feedback_manager"
        )
        return pkg

    def import_training_package(
        self,
        package_data: Any,
        auto_trainer_instance: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Importa um pacote de treinamento (v2.0, v1.0 ou lista de revisões) baixados anteriormente.
        Mescla revisões por Dan, histórico de treinamentos, links de streaming com Decisão dos Shinpans,
        perfis de calibração e a base de conhecimento de IA (treinamentos automáticos).
        """
        imported_items = []
        imported_history = []
        imported_shinpan_videos = []
        imported_profiles = {}
        imported_kb = None
        imported_ckpt = None

        if isinstance(package_data, list):
            for idx, item in enumerate(package_data):
                if isinstance(item, dict):
                    if "label" in item or "strike_type" in item or "sub_scores" in item:
                        imported_items.append(item)
                    elif "items_count" in item or "optimization_summary" in item:
                        imported_history.append(item)
                    else:
                        imported_items.append(item)
        elif isinstance(package_data, dict):
            if "review_items" in package_data and isinstance(package_data["review_items"], list):
                imported_items = package_data["review_items"]
            elif "label" in package_data or "strike_type" in package_data:
                imported_items = [package_data]

            if "training_history" in package_data and isinstance(package_data["training_history"], list):
                imported_history = package_data["training_history"]

            if "shinpan_reviewed_videos" in package_data and isinstance(package_data["shinpan_reviewed_videos"], list):
                imported_shinpan_videos = package_data["shinpan_reviewed_videos"]

            if "calibration_profiles" in package_data and isinstance(package_data["calibration_profiles"], dict):
                imported_profiles = package_data["calibration_profiles"]

            if "ai_knowledge_base" in package_data and isinstance(package_data["ai_knowledge_base"], dict):
                imported_kb = package_data["ai_knowledge_base"]
            elif "knowledge_base" in package_data and isinstance(package_data["knowledge_base"], dict):
                imported_kb = package_data["knowledge_base"]

            if "auto_training_checkpoint" in package_data and isinstance(package_data["auto_training_checkpoint"], dict):
                imported_ckpt = package_data["auto_training_checkpoint"]
            elif "checkpoint" in package_data and isinstance(package_data["checkpoint"], dict):
                imported_ckpt = package_data["checkpoint"]
        else:
            raise ValueError("Formato de arquivo JSON não reconhecido.")

        if not imported_items and not imported_history and not imported_profiles and not imported_shinpan_videos and not imported_kb:
            raise ValueError("O arquivo JSON não contém itens de revisão, histórico nem base de conhecimento válidos.")

        # Captura IDs de vídeos dos Shinpans já registrados ANTES da importação
        pre_registered_video_ids = {
            v.get("video_identifier")
            for v in self.load_shinpan_reviewed_videos()
            if v.get("video_identifier")
        }

        # 1. Carregar e mesclar revisões no dataset
        current_data = self.load_feedback()
        existing_ids = {item.get("id") for item in current_data if "id" in item and item.get("id")}

        new_added_count = 0
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        for idx, item in enumerate(imported_items):
            if not isinstance(item, dict):
                continue
            item_id = item.get("id") or f"imported_{item.get('video_name', 'vid')}_{item.get('timestamp', '00')}_{idx+1}"
            item["id"] = item_id

            if item_id not in existing_ids:
                if is_shinpan_reviewer(item.get("reviewer_dan")) or item.get("is_shinpan_decision") or item.get("reviewer_dan_name") == SHINPAN_NAME:
                    item["reviewer_dan"] = SHINPAN_REV_KEY
                    item["reviewer_dan_name"] = SHINPAN_NAME
                    item["is_shinpan_decision"] = True
                else:
                    if "reviewer_dan" not in item:
                        item["reviewer_dan"] = 1
                    if "reviewer_dan_name" not in item:
                        item["reviewer_dan_name"] = DAN_NAMES.get(item["reviewer_dan"], "1º Dan")
                if "review_date" not in item:
                    item["review_date"] = now_iso

                current_data.append(item)
                existing_ids.add(item_id)
                new_added_count += 1

        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)

        # 2. Carregar e mesclar histórico de treinamentos
        current_history = self.load_history()
        existing_hist_ids = {h.get("id") for h in current_history if "id" in h and h.get("id")}
        new_history_count = 0
        auto_trainings_imported = 0

        for h_item in imported_history:
            if not isinstance(h_item, dict):
                continue
            h_id = h_item.get("id") or f"train_imp_{len(current_history)+1}"
            h_item["id"] = h_id
            if h_item.get("is_auto_training") or str(h_item.get("video_name", "")).startswith("AI_Auto_Trainer_"):
                auto_trainings_imported += 1
            if h_id not in existing_hist_ids:
                current_history.append(h_item)
                existing_hist_ids.add(h_id)
                new_history_count += 1

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(current_history, f, indent=2, ensure_ascii=False)

        # 2.1. Mesclar vídeos e links de streaming com Decisão dos Shinpans
        current_shinpan_videos = self.load_shinpan_reviewed_videos()
        registered_video_ids = {v.get("video_identifier") for v in current_shinpan_videos if v.get("video_identifier")}

        for sv in imported_shinpan_videos:
            if isinstance(sv, dict):
                sv_id = sv.get("video_identifier") or normalize_video_identifier(sv.get("video_url") or sv.get("video_name"))
                if sv_id and sv_id not in registered_video_ids:
                    sv["video_identifier"] = sv_id
                    current_shinpan_videos.append(sv)
                    registered_video_ids.add(sv_id)

        # Sincronização adicional a partir do histórico importado
        for h in current_history:
            if h.get("is_shinpan_decision") or is_shinpan_reviewer(h.get("reviewer_dan")):
                v_url = h.get("video_url") or h.get("streaming_url") or ""
                v_name = h.get("video_name") or ""
                canon_id = normalize_video_identifier(v_url) if v_url else normalize_video_identifier(v_name)
                if canon_id and canon_id not in registered_video_ids:
                    current_shinpan_videos.append({
                        "video_identifier": canon_id,
                        "video_url": v_url or v_name,
                        "video_name": v_name,
                        "session_id": h.get("id", ""),
                        "reviewed_at": h.get("timestamp", ""),
                        "reviewer_dan": SHINPAN_REV_KEY,
                        "reviewer_dan_name": SHINPAN_NAME,
                        "profile_key": h.get("profile_key", "normal"),
                        "items_count": h.get("items_count", 0)
                    })
                    registered_video_ids.add(canon_id)

        # Sincronização adicional a partir dos itens de revisão importados
        for itm in current_data:
            if itm.get("is_shinpan_decision") or is_shinpan_reviewer(itm.get("reviewer_dan")):
                v_url = itm.get("video_url") or itm.get("streaming_url") or ""
                v_name = itm.get("video_name") or ""
                canon_id = normalize_video_identifier(v_url) if v_url else normalize_video_identifier(v_name)
                if canon_id and canon_id not in registered_video_ids:
                    current_shinpan_videos.append({
                        "video_identifier": canon_id,
                        "video_url": v_url or v_name,
                        "video_name": v_name,
                        "session_id": itm.get("id", ""),
                        "reviewed_at": itm.get("review_date", ""),
                        "reviewer_dan": SHINPAN_REV_KEY,
                        "reviewer_dan_name": SHINPAN_NAME,
                        "profile_key": itm.get("profile_key", "normal"),
                        "items_count": 1
                    })
                    registered_video_ids.add(canon_id)

        shinpan_imported_count = len([
            v for v in current_shinpan_videos
            if v.get("video_identifier") not in pre_registered_video_ids
        ])

        with open(self.shinpan_registry_path, "w", encoding="utf-8") as f:
            json.dump(current_shinpan_videos, f, indent=2, ensure_ascii=False)

        if imported_items and not imported_history:
            first_is_shinpan = is_shinpan_reviewer(imported_items[0].get("reviewer_dan")) or imported_items[0].get("is_shinpan_decision")
            first_dan = SHINPAN_REV_KEY if first_is_shinpan else imported_items[0].get("reviewer_dan", 1)
            first_name = SHINPAN_NAME if first_is_shinpan else DAN_NAMES.get(first_dan, "1º Dan")
            session_rec = {
                "id": f"train_imp_{now_iso.replace(':', '').replace('-', '')}_{len(current_history)+1}",
                "timestamp": now_iso,
                "reviewer_dan": first_dan,
                "reviewer_dan_name": first_name,
                "is_shinpan_decision": first_is_shinpan,
                "video_name": imported_items[0].get("video_name", "imported_package"),
                "video_url": imported_items[0].get("video_url") or imported_items[0].get("streaming_url") or "",
                "profile_key": imported_items[0].get("profile_key", "normal"),
                "items_count": len(imported_items),
                "optimization_summary": {"status": "success", "imported": True}
            }
            current_history.append(session_rec)
            new_history_count += 1

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(current_history, f, indent=2, ensure_ascii=False)

        # 3. Atualizar perfis de calibração
        profiles_to_use = DEFAULT_CALIBRATION_PROFILES.copy()
        if os.path.exists(self.profiles_path):
            try:
                with open(self.profiles_path, "r", encoding="utf-8") as f:
                    profiles_to_use = json.load(f)
            except Exception:
                pass

        if imported_profiles and isinstance(imported_profiles, dict):
            profiles_to_use.update(imported_profiles)

        for p_key in list(profiles_to_use.keys()):
            updated_p_cfg, _ = self.optimize_profile_config(p_key, profiles_to_use[p_key])
            profiles_to_use[p_key] = updated_p_cfg

        with open(self.profiles_path, "w", encoding="utf-8") as f:
            json.dump(profiles_to_use, f, indent=2, ensure_ascii=False)

        # 4. Mesclar Base de Conhecimento e Treinamentos Automáticos por IA
        kb_merge_res = {}
        kb_updated = False
        if imported_kb and isinstance(imported_kb, dict):
            if auto_trainer_instance is not None and hasattr(auto_trainer_instance, "merge_knowledge_data"):
                kb_merge_res = auto_trainer_instance.merge_knowledge_data(imported_kb, imported_checkpoint=imported_ckpt)
                kb_updated = True
            else:
                kb_merge_res = self._merge_ai_knowledge_base(imported_kb, imported_checkpoint=imported_ckpt)
                kb_updated = True

        updated_metrics = self.get_training_metrics()

        log_event(
            "INFO",
            f"PACOTE DE TREINAMENTO CARREGADO E RECALIBRADO: {new_added_count} novos itens de revisão, "
            f"{new_history_count} treinamentos, {shinpan_imported_count} vídeos/links de streaming dos Shinpans integrados "
            f"e Base de Conhecimento {'atualizada' if kb_updated else 'mantida'}. Nível Dan Médio: {updated_metrics['average_dan_label']}.",
            "feedback_manager"
        )

        return {
            "status": "success",
            "imported_items_count": len(imported_items),
            "new_items_added": new_added_count,
            "imported_trainings_count": new_history_count,
            "shinpan_videos_imported": shinpan_imported_count,
            "shinpan_videos_total": len(current_shinpan_videos),
            "auto_trainings_imported": auto_trainings_imported,
            "knowledge_base_updated": kb_updated,
            "knowledge_merge_summary": kb_merge_res,
            "total_trainings_now": updated_metrics["total_trainings_count"],
            "average_dan_now": updated_metrics["average_dan_label"]
        }

    def optimize_profile_config(self, profile_key: str, current_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Aplica otimização por reforço baseada nos feedbacks gravados para o perfil ativo.
        Pondera anotações por revisores de maior graduação Dan (1 a 8) e Decisão dos Shinpans
        com constante média equilibrada (SHINPAN_CALIBRATION_WEIGHT = 4.5).
        Recalibra min_total_score, sub_thresholds e os pesos de validação (weights).
        """
        feedback_list = [d for d in self.load_feedback() if d.get("profile_key") == profile_key or not d.get("profile_key")]

        fps = [d for d in feedback_list if d.get("label") == "FP"]
        tps = [d for d in feedback_list if d.get("label") in ["TP", "CONFIRMED"]]
        fns = [d for d in feedback_list if d.get("label") in ["FN", "INCLUDED"]]

        if not feedback_list:
            return current_config, {
                "status": "no_data",
                "message": "Nenhum feedback registrado para otimizar este perfil."
            }

        new_config = json.loads(json.dumps(current_config))
        weights = new_config.get("weights", {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15})
        sub_thresholds = new_config.get("sub_thresholds", {"target_impact": 0.60, "fumikomi_sync": 0.50, "posture": 0.50, "zanshin": 0.45})
        min_total = new_config.get("min_total_score", 0.65)

        changes_summary = []

        def _get_item_weight(item: Dict[str, Any]) -> float:
            if item.get("is_shinpan_decision") or is_shinpan_reviewer(item.get("reviewer_dan")):
                return SHINPAN_CALIBRATION_WEIGHT
            r_dan = item.get("reviewer_dan", 1)
            if isinstance(r_dan, (int, float)) and 1 <= r_dan <= 8:
                return float(r_dan)
            return 1.0

        # Ponderação e ajuste de limiares
        if fps:
            max_fp_total_score = max([d.get("total_score", 0.0) for d in fps]) / 100.0 if fps else 0.0
            if max_fp_total_score >= min_total:
                old_min = min_total
                min_total = min(0.90, max(min_total + 0.05, max_fp_total_score + 0.02))
                changes_summary.append(f"Elevação da Pontuação Mínima Global: {int(old_min*100)}% ➔ {int(min_total*100)}%")

            sub_keys = ["target_impact", "fumikomi_sync", "posture", "zanshin"]
            for skey in sub_keys:
                fp_items = [d for d in fps if "sub_scores" in d and skey in d.get("sub_scores", {})]
                tp_items = [d for d in tps if "sub_scores" in d and skey in d.get("sub_scores", {})]

                if fp_items:
                    fp_w_sum = sum(_get_item_weight(d) for d in fp_items)
                    tp_w_sum = sum(_get_item_weight(d) for d in tp_items)

                    avg_fp_sub = sum((d["sub_scores"][skey] / 100.0) * _get_item_weight(d) for d in fp_items) / (fp_w_sum if fp_w_sum > 0 else 1.0)
                    avg_tp_sub = (sum((d["sub_scores"][skey] / 100.0) * _get_item_weight(d) for d in tp_items) / (tp_w_sum if tp_w_sum > 0 else 1.0)) if tp_items else 0.80

                    if avg_tp_sub > avg_fp_sub:
                        old_sub = sub_thresholds.get(skey, 0.50)
                        target_new_sub = min(0.85, max(old_sub + 0.05, avg_fp_sub + 0.05))
                        sub_thresholds[skey] = round(target_new_sub, 2)
                        changes_summary.append(f"Reforço no Limiar de '{skey}': {int(old_sub*100)}% ➔ {int(target_new_sub*100)}%")

        elif fns and not fps:
            old_min = min_total
            min_total = max(0.40, min_total - 0.04)
            changes_summary.append(f"Suavização da Pontuação Mínima Global para capturar golpes perdidos: {int(old_min*100)}% ➔ {int(min_total*100)}%")

        # Recalibração adaptativa dos pesos dos 4 pilares (weights)
        if tps or fps:
            sub_keys = ["target_impact", "fumikomi_sync", "posture", "zanshin"]
            tp_weight_sums = {k: 0.0 for k in sub_keys}
            total_tp_w = 0.0

            for item in tps:
                w_factor = _get_item_weight(item)
                scores = item.get("sub_scores", {})
                has_scores = any(k in scores for k in sub_keys)
                total_tp_w += w_factor
                for k in sub_keys:
                    s_val = (scores.get(k, 80.0) if has_scores else 80.0) / 100.0
                    tp_weight_sums[k] += s_val * w_factor

            if total_tp_w > 0:
                raw_proportions = {k: tp_weight_sums[k] / total_tp_w for k in sub_keys}
                prop_sum = sum(raw_proportions.values())
                if prop_sum > 0:
                    target_w = {k: raw_proportions[k] / prop_sum for k in sub_keys}
                    alpha = min(0.20, 0.03 * (total_tp_w / 5.0))
                    new_w_calc = {}
                    for k in sub_keys:
                        cur_w = weights.get(k, 0.25)
                        new_w_calc[k] = (1.0 - alpha) * cur_w + alpha * target_w[k]

                    w_sum = sum(new_w_calc.values())
                    new_weights = {k: round(new_w_calc[k] / w_sum, 3) for k in sub_keys}
                    diff_sum = round(1.0 - sum(new_weights.values()), 3)
                    new_weights["target_impact"] = round(new_weights["target_impact"] + diff_sum, 3)

                    if any(abs(new_weights[k] - weights.get(k, 0.0)) >= 0.005 for k in sub_keys):
                        changes_summary.append(
                            f"Recalibração dos Pesos de Validação dos Golpes: "
                            f"Alvo={int(new_weights['target_impact']*100)}%, "
                            f"Fumikomi={int(new_weights['fumikomi_sync']*100)}%, "
                            f"Postura={int(new_weights['posture']*100)}%, "
                            f"Zanshin={int(new_weights['zanshin']*100)}%"
                        )
                        weights = new_weights

        new_config["min_total_score"] = round(min_total, 2)
        new_config["sub_thresholds"] = sub_thresholds
        new_config["weights"] = weights

        opt_stats = {
            "status": "success",
            "profile_key": profile_key,
            "fps_analyzed": len(fps),
            "tps_analyzed": len(tps),
            "fns_analyzed": len(fns),
            "changes": changes_summary if changes_summary else ["Parâmetros já otimizados para o conjunto de dados atual."]
        }

        return new_config, opt_stats

