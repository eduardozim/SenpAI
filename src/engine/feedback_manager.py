"""
Módulo de Gerenciamento de Feedback e Otimização Adaptativa por Reforço.
Gerencia a gravação de feedbacks (TP, FP, FN) e recalibra limiares do perfil selecionado.
"""

import json
import os
from typing import Dict, Any, List, Tuple

class FeedbackManager:
    def __init__(self, dataset_path: str = "data/feedback_dataset.json"):
        self.dataset_path = dataset_path
        self._ensure_dataset_exists()

    def _ensure_dataset_exists(self):
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        if not os.path.exists(self.dataset_path):
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def load_feedback(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
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
        label: str,  # "TP" (True Positive / Correct), "FP" (False Positive / Wrong), "FN" (False Negative / Missed)
        sub_scores: Dict[str, float] = None,
        total_score: float = 0.0,
        strike_type: str = "MEN",
        timestamp: str = "00:00.000",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Adiciona uma anotação de feedback ao dataset.
        """
        data = self.load_feedback()
        
        entry = {
            "id": f"{video_name}_{event_id}_{len(data)+1}",
            "video_name": video_name,
            "profile_key": profile_key,
            "label": label,  # TP, FP, FN
            "strike_type": strike_type,
            "timestamp": timestamp,
            "total_score": total_score,
            "sub_scores": sub_scores or {},
            "notes": notes
        }

        # Atualizar entrada existente se o mesmo event_id e video_name já tiverem anotação
        updated = False
        for idx, item in enumerate(data):
            if item.get("video_name") == video_name and item.get("id_event") == event_id:
                entry["id_event"] = event_id
                data[idx] = entry
                updated = True
                break
        
        if not updated:
            entry["id_event"] = event_id
            data.append(entry)

        with open(self.dataset_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return entry

    def get_stats(self, profile_key: str = None) -> Dict[str, Any]:
        """
        Retorna estatísticas de anotações (TP, FP, FN) gerais ou filtradas por perfil.
        """
        data = self.load_feedback()
        if profile_key:
            data = [d for d in data if d.get("profile_key") == profile_key]

        total = len(data)
        tp = sum(1 for d in data if d.get("label") == "TP")
        fp = sum(1 for d in data if d.get("label") == "FP")
        fn = sum(1 for d in data if d.get("label") == "FN")

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

    def optimize_profile_config(self, profile_key: str, current_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Aplica otimização por reforço baseada nos feedbacks gravados para o perfil ativo.
        Ajusta `min_total_score`, `sub_thresholds` e pesos para reduzir a taxa de Falsos Positivos.
        Retorna (nova_config, estatisticas_da_otimizacao).
        """
        feedback_list = [d for d in self.load_feedback() if d.get("profile_key") == profile_key or not d.get("profile_key")]
        
        fps = [d for d in feedback_list if d.get("label") == "FP"]
        tps = [d for d in feedback_list if d.get("label") == "TP"]
        fns = [d for d in feedback_list if d.get("label") == "FN"]

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

        # 1. Se há Falsos Positivos (golpes detectados indevidamente):
        # Aumentar os sub-thresholds e o min_total para eliminar falsos positivos
        if fps:
            max_fp_total_score = max([d.get("total_score", 0.0) for d in fps]) / 100.0 if fps else 0.0
            
            # Ajustar min_total_score para ficar acima da maioria dos FPs se possível
            if max_fp_total_score >= min_total:
                old_min = min_total
                # Eleva gradualmente o min_total com segurança (teto de 0.90)
                min_total = min(0.90, max(min_total + 0.05, max_fp_total_score + 0.02))
                changes_summary.append(f"Elevação da Pontuação Mínima Global: {int(old_min*100)}% ➔ {int(min_total*100)}%")

            # Analisar quais sub-critérios falharam nos TPs versus FPs
            sub_keys = ["target_impact", "fumikomi_sync", "posture", "zanshin"]
            for skey in sub_keys:
                fp_sub_scores = [d.get("sub_scores", {}).get(skey, 100.0) / 100.0 for d in fps if "sub_scores" in d]
                tp_sub_scores = [d.get("sub_scores", {}).get(skey, 0.0) / 100.0 for d in tps if "sub_scores" in d]

                if fp_sub_scores:
                    avg_fp_sub = sum(fp_sub_scores) / len(fp_sub_scores)
                    avg_tp_sub = (sum(tp_sub_scores) / len(tp_sub_scores)) if tp_sub_scores else 0.80

                    # Se a pontuação média do sub-critério nos TPs é significativamente maior do que nos FPs,
                    # podemos subir o sub_threshold para filtrar os FPs
                    if avg_tp_sub > avg_fp_sub:
                        old_sub = sub_thresholds.get(skey, 0.50)
                        target_new_sub = min(0.85, max(old_sub + 0.05, avg_fp_sub + 0.05))
                        sub_thresholds[skey] = round(target_new_sub, 2)
                        changes_summary.append(f"Reforço no Limiar de '{skey}': {int(old_sub*100)}% ➔ {int(target_new_sub*100)}%")

        # 2. Se há Falsos Negativos (golpes legítimos não pontuados):
        elif fns and not fps:
            # Tornar o perfil ligeiramente mais tolerante para capturar mais golpes
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
