"""
Módulo de Gerenciamento de Feedback, Governança por Dan e Otimização Adaptativa por Reforço.
Gerencia a gravação de marcações (TP, FP, FN, edições e revisões por Dan), estatísticas e retreinamento do sistema.
"""

import json
import os
import datetime
from typing import Dict, Any, List, Tuple
from src.utils.logger_manager import log_event

DAN_NAMES: Dict[int, str] = {
    1: "1º Dan (Shodan)",
    2: "2º Dan (Nidan)",
    3: "3º Dan (Sandan)",
    4: "4º Dan (Yondan)",
    5: "5º Dan (Godan)",
    6: "6º Dan (Rokudan)",
    7: "7º Dan (Nanadan)",
    8: "8º Dan (Hachidan)"
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
        profiles_path: str = "config/calibration_profiles.json"
    ):
        self.dataset_path = dataset_path
        self.history_path = history_path
        self.profiles_path = profiles_path
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

    def save_feedback(
        self,
        video_name: str,
        profile_key: str,
        event_id: str,
        label: str,  # "TP", "FP", "FN", "CONFIRMED", "EDITED", "INCLUDED"
        sub_scores: Dict[str, float] = None,
        total_score: float = 0.0,
        strike_type: str = "MEN",
        timestamp: str = "00:00.000",
        notes: str = "",
        reviewer_dan: int = 1,
        is_edited: bool = False,
        is_included: bool = False,
        decision_category: str = ""
    ) -> Dict[str, Any]:
        """
        Adiciona ou atualiza uma anotação de feedback no dataset com registro de Dan e categoria de decisão.
        """
        data = self.load_feedback()
        dan_val = max(1, min(8, int(reviewer_dan)))
        dan_name = DAN_NAMES.get(dan_val, f"{dan_val}º Dan")
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        entry = {
            "id": f"{video_name}_{event_id}_{len(data)+1}",
            "id_event": event_id,
            "video_name": video_name,
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
        reviewer_dan: int,
        review_items: List[Dict[str, Any]],
        current_profile_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Salva uma sessão de revisão de detecção gravada, atualiza os dados por Dan,
        executa o retreinamento do modelo e grava no histórico de treinamentos.
        """
        dan_val = max(1, min(8, int(reviewer_dan)))
        dan_name = DAN_NAMES.get(dan_val, f"{dan_val}º Dan")
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        saved_entries = []
        for item in review_items:
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
                decision_category=item.get("decision_category", item.get("category", ""))
            )
            saved_entries.append(entry)

        # Recalibrar / retreinar o modelo com os novos dados
        new_config, opt_summary = self.optimize_profile_config(profile_key, current_profile_config)

        # Registrar o evento de treinamento no histórico
        history = self.load_history()
        session_record = {
            "id": f"train_{now_iso.replace(':', '').replace('-', '')}_{len(history)+1}",
            "timestamp": now_iso,
            "reviewer_dan": dan_val,
            "reviewer_dan_name": dan_name,
            "video_name": video_name,
            "profile_key": profile_key,
            "items_count": len(saved_entries),
            "optimization_summary": opt_summary
        }
        history.append(session_record)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        return new_config, session_record

    def get_stats(self, profile_key: str = None) -> Dict[str, Any]:
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
        - Contador total de treinamentos realizados.
        - Nível médio (Dan) dos treinamentos.
        - Tabela de quantidade de treinamentos agrupada por Dan (1º a 8º Dan).
        """
        history = self.load_history()
        data = self.load_feedback()

        total_trainings = len(history)

        # Se houver histórico de treinamentos, usar o Dan dos treinamentos.
        # Caso haja apenas feedbacks diretos sem sessão explícita, agregar ambos.
        dan_counts = {dan: 0 for dan in range(1, 9)}

        dan_sum = 0
        weight_count = 0

        if total_trainings > 0:
            for session in history:
                dan = session.get("reviewer_dan", 1)
                if 1 <= dan <= 8:
                    dan_counts[dan] += 1
                    dan_sum += dan
                    weight_count += 1
        elif data:
            # Fallback para contar revisões se o histórico estiver vazio
            for item in data:
                dan = item.get("reviewer_dan", 1)
                if 1 <= dan <= 8:
                    dan_counts[dan] += 1
                    dan_sum += dan
                    weight_count += 1
            total_trainings = weight_count

        avg_dan = (dan_sum / weight_count) if weight_count > 0 else 0.0
        avg_dan_round = round(avg_dan, 1)
        avg_dan_int = max(1, min(8, int(round(avg_dan)))) if avg_dan > 0 else 1
        avg_dan_label = f"{avg_dan_round}º Dan ({DAN_NAMES.get(avg_dan_int, '')})" if avg_dan > 0 else "Nenhum treinamento"

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

        return {
            "total_trainings_count": total_trainings,
            "average_dan_level": avg_dan_round,
            "average_dan_label": avg_dan_label,
            "total_review_items": len(data),
            "dan_distribution": table_data
        }

    def reset_all_training_data(self) -> None:
        """
        Apaga todo o treinamento do sistema, limpando conjuntos de dados e
        restaurando os perfis de calibração para a configuração padrão original.
        """
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

        os.makedirs(os.path.dirname(self.profiles_path), exist_ok=True)
        with open(self.profiles_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CALIBRATION_PROFILES, f, indent=2, ensure_ascii=False)

        log_event("WARNING", "TREINAMENTO APAGADO (RESET): Todo o histórico de revisões, dataset de feedbacks e calibrações foram restaurados ao estágio inicial de fábrica.", "feedback_manager")

    def export_training_package(self) -> Dict[str, Any]:
        """
        Exporta o pacote de treinamento atual contendo todos os arquivos de revisão,
        marcações por Dan, histórico de treinamentos e perfis calibrados.
        """
        data = self.load_feedback()
        history = self.load_history()
        metrics = self.get_training_metrics()

        profiles = DEFAULT_CALIBRATION_PROFILES.copy()
        if os.path.exists(self.profiles_path):
            try:
                with open(self.profiles_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
            except Exception:
                pass

        now_iso = datetime.datetime.now().isoformat(timespec="seconds")

        pkg = {
            "system_name": "SenpAI",
            "package_version": "1.0",
            "exported_at": now_iso,
            "summary": {
                "total_trainings_performed": metrics["total_trainings_count"],
                "average_reviewer_dan": metrics["average_dan_level"],
                "average_dan_label": metrics["average_dan_label"],
                "total_review_entries": len(data)
            },
            "review_items": data,
            "training_history": history,
            "calibration_profiles": profiles
        }

        log_event("INFO", f"PACOTE DE TREINAMENTO EXPORTADO: Pacote gerado com {len(data)} itens de revisão e {len(history)} treinamentos registrados.", "feedback_manager")
        return pkg

    def import_training_package(self, package_data: Any) -> Dict[str, Any]:
        """
        Importa um pacote de treinamento ou lista de revisões baixados anteriormente.
        Mescla os arquivos de revisão, atualiza o histórico e recalibra os modelos.
        """
        imported_items = []
        imported_history = []
        imported_profiles = {}

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

            if "calibration_profiles" in package_data and isinstance(package_data["calibration_profiles"], dict):
                imported_profiles = package_data["calibration_profiles"]
        else:
            raise ValueError("Formato de arquivo JSON não reconhecido.")

        if not imported_items and not imported_history and not imported_profiles:
            raise ValueError("O arquivo JSON não contém itens de revisão nem histórico válidos.")

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
        for h_item in imported_history:
            if not isinstance(h_item, dict):
                continue
            h_id = h_item.get("id") or f"train_imp_{len(current_history)+1}"
            h_item["id"] = h_id
            if h_id not in existing_hist_ids:
                current_history.append(h_item)
                existing_hist_ids.add(h_id)
                new_history_count += 1

        if imported_items and not imported_history:
            session_rec = {
                "id": f"train_imp_{now_iso.replace(':', '').replace('-', '')}_{len(current_history)+1}",
                "timestamp": now_iso,
                "reviewer_dan": imported_items[0].get("reviewer_dan", 1),
                "reviewer_dan_name": DAN_NAMES.get(imported_items[0].get("reviewer_dan", 1), "1º Dan"),
                "video_name": imported_items[0].get("video_name", "imported_package"),
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

        updated_metrics = self.get_training_metrics()

        log_event("INFO", f"PACOTE DE TREINAMENTO CARREGADO E RECALIBRADO: {new_added_count} novos itens de revisão e {new_history_count} treinamentos integrados. Nível Dan Médio atual: {updated_metrics['average_dan_label']}.", "feedback_manager")

        return {
            "status": "success",
            "imported_items_count": len(imported_items),
            "new_items_added": new_added_count,
            "imported_trainings_count": new_history_count,
            "total_trainings_now": updated_metrics["total_trainings_count"],
            "average_dan_now": updated_metrics["average_dan_label"]
        }

    def optimize_profile_config(self, profile_key: str, current_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Aplica otimização por reforço baseada nos feedbacks gravados para o perfil ativo.
        Pondera anotações por revisores de maior graduação Dan.
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

        # Ponderação por Dan (Dan 1x a 8x peso)
        if fps:
            max_fp_total_score = max([d.get("total_score", 0.0) for d in fps]) / 100.0 if fps else 0.0
            if max_fp_total_score >= min_total:
                old_min = min_total
                min_total = min(0.90, max(min_total + 0.05, max_fp_total_score + 0.02))
                changes_summary.append(f"Elevação da Pontuação Mínima Global: {int(old_min*100)}% ➔ {int(min_total*100)}%")

            sub_keys = ["target_impact", "fumikomi_sync", "posture", "zanshin"]
            for skey in sub_keys:
                fp_sub_scores = [d.get("sub_scores", {}).get(skey, 100.0) / 100.0 for d in fps if "sub_scores" in d]
                tp_sub_scores = [d.get("sub_scores", {}).get(skey, 0.0) / 100.0 for d in tps if "sub_scores" in d]

                if fp_sub_scores:
                    avg_fp_sub = sum(fp_sub_scores) / len(fp_sub_scores)
                    avg_tp_sub = (sum(tp_sub_scores) / len(tp_sub_scores)) if tp_sub_scores else 0.80

                    if avg_tp_sub > avg_fp_sub:
                        old_sub = sub_thresholds.get(skey, 0.50)
                        target_new_sub = min(0.85, max(old_sub + 0.05, avg_fp_sub + 0.05))
                        sub_thresholds[skey] = round(target_new_sub, 2)
                        changes_summary.append(f"Reforço no Limiar de '{skey}': {int(old_sub*100)}% ➔ {int(target_new_sub*100)}%")

        elif fns and not fps:
            old_min = min_total
            min_total = max(0.40, min_total - 0.04)
            changes_summary.append(f"Suavização da Pontuação Mínima Global para capturar golpes perdidos: {int(old_min*100)}% ➔ {int(min_total*100)}%")

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

