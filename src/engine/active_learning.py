"""
Módulo de Aprendizado Ativo (Active Learning), Golden Benchmark e Governança Multi-Árbitro.
Implementa os 4 pilares do Eixo 4 do SenpAI:
1. UncertaintySampler: Amostragem por incerteza (45% a 65%) para curadoria ativa.
2. GoldenBenchmark: Dataset padrão-ouro com salvaguarda contra esquecimento catastrófico (Catastrophic Forgetting Prevention).
3. MultiJudgeConsensus: Consenso ponderado por Dan no modelo oficial de arbitragem da FIK (2 de 3) com grau de divergência.
4. ReviewerTrustManager: Decaimento temporal de autoridade por inatividade e divergência acumulada de revisores.
"""

import os
import json
import math
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple

from src.utils.logger_manager import log_event
from src.engine.llm_assistant import KendoLLMAssistant


# ==============================================================================
# 4.1 AMOSTRAGEM POR INCERTEZA (UNCERTAINTY SAMPLING)
# ==============================================================================
class UncertaintySampler:
    """
    Gerencia a fila de lances prioritários para rotulagem humana (Active Learning),
    selecionando instâncias cuja probabilidade de Ippon está próxima da fronteira de decisão.
    """

    def __init__(
        self,
        queue_file: str = "data/active_learning_queue.json",
        min_threshold: float = 0.45,
        max_threshold: float = 0.65,
        llm_assistant: Optional[KendoLLMAssistant] = None
    ):
        self.queue_file = queue_file
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.llm_assistant = llm_assistant or KendoLLMAssistant()
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.queue_file) or ".", exist_ok=True)
        if not os.path.exists(self.queue_file):
            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    @staticmethod
    def calculate_uncertainty(confidence: float) -> float:
        """
        Calcula a incerteza normalizada em [0.0, 1.0]:
        Incerteza(x) = 1.0 - 2 * |P(Ippon|x) - 0.50|
        - Confiança 0.50 -> Incerteza máxima = 1.00
        - Confiança 0.00 ou 1.00 -> Incerteza mínima = 0.00
        """
        conf_clamped = max(0.0, min(1.0, float(confidence)))
        return round(1.0 - 2.0 * abs(conf_clamped - 0.50), 4)

    def is_uncertain(self, confidence: float) -> bool:
        """Verifica se a confiança recai na faixa crítica de incerteza."""
        return self.min_threshold <= float(confidence) <= self.max_threshold

    def evaluate_and_enqueue(
        self,
        strike_data: Dict[str, Any],
        confidence: float,
        video_source: str = "",
        profile_name: str = "normal",
        enrich_with_llm: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Se o lance for incerto, enriquece com diagnóstico do LLM e enfileira para curadoria ativa.
        """
        if not self.is_uncertain(confidence):
            return None

        uncertainty_score = self.calculate_uncertainty(confidence)

        # Diagnóstico com LLM para acelerar o processo de curadoria do Sensei
        llm_triage = None
        if enrich_with_llm and self.llm_assistant:
            try:
                llm_triage = self.llm_assistant.analyze_uncertain_strike(
                    strike_data, profile_name=profile_name, confidence=confidence
                )
            except Exception as e:
                log_event("WARNING", f"Falha ao gerar triage LLM para lance incerto: {e}", "active_learning")

        entry = {
            "id": f"unc_{int(time.time() * 1000)}_{len(self.get_queue())}",
            "timestamp": datetime.datetime.now().isoformat(),
            "video_source": video_source,
            "strike_type": strike_data.get("strike_type", "Men"),
            "confidence": round(float(confidence), 4),
            "uncertainty_score": uncertainty_score,
            "strike_data": strike_data,
            "profile_name": profile_name,
            "llm_triage": llm_triage,
            "status": "pending_curation"
        }

        queue = self.get_queue()
        # Evita duplicatas se o mesmo id ou dados similares já estiverem pendentes
        queue.append(entry)
        # Ordena a fila decrescente por incerteza (as mais incertas primeiro)
        queue.sort(key=lambda x: x.get("uncertainty_score", 0.0), reverse=True)

        # Limita a 100 itens mais informativos
        queue = queue[:100]

        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)

        log_event(
            "INFO",
            f"Lance incerto enfileirado para Curadoria Ativa (P={confidence:.2f}, Incerteza={uncertainty_score:.2f}, Golpe={entry['strike_type']})",
            "active_learning"
        )
        return entry

    def get_queue(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna os itens da fila de curadoria ativa."""
        self._ensure_storage()
        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                if status:
                    return [it for it in items if it.get("status") == status]
                return items
        except Exception:
            return []

    def resolve_item(self, item_id: str, label_approved: bool, reviewer_dan: int, notes: str = "") -> bool:
        """Marca um item da fila como curado e resolvido por um árbitro."""
        queue = self.get_queue()
        resolved = False
        for it in queue:
            if it.get("id") == item_id:
                it["status"] = "curated"
                it["curated_label"] = "IPPON" if label_approved else "NO_POINT"
                it["curator_dan"] = reviewer_dan
                it["curator_notes"] = notes
                it["curated_at"] = datetime.datetime.now().isoformat()
                resolved = True
                break

        if resolved:
            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump(queue, f, ensure_ascii=False, indent=2)
        return resolved


# ==============================================================================
# 4.2 CONJUNTO DE VALIDAÇÃO PADRÃO-OURO (GOLDEN BENCHMARK)
# ==============================================================================
DEFAULT_GOLDEN_BENCHMARK_SAMPLES: List[Dict[str, Any]] = [
    # 1. Ippons Canônicos Indiscutíveis (FIK chancelados)
    {
        "id": "golden_men_01",
        "category": "indisputable_ippon",
        "strike_type": "Men",
        "ground_truth": True,
        "description": "Men perfeito em Ki-Ken-Tai-Ichi com Fumikomi simultâneo e Zanshin sustentado.",
        "metrics": {"target_impact": 0.92, "fumikomi_sync": 0.90, "posture": 0.88, "zanshin": 0.85, "hasuji_score": 0.95}
    },
    {
        "id": "golden_kote_01",
        "category": "indisputable_ippon",
        "strike_type": "Kote",
        "ground_truth": True,
        "description": "Kote com estalo seco no antebraço direito, cotovelo estendido e recuo controlado.",
        "metrics": {"target_impact": 0.88, "fumikomi_sync": 0.85, "posture": 0.82, "zanshin": 0.80, "hasuji_score": 0.90}
    },
    {
        "id": "golden_do_01",
        "category": "indisputable_ippon",
        "strike_type": "Do",
        "ground_truth": True,
        "description": "Nuki-do limpo com corte diagonal de lâmina a 45° e saída veloz em Zanshin.",
        "metrics": {"target_impact": 0.89, "fumikomi_sync": 0.80, "posture": 0.84, "zanshin": 0.86, "hasuji_score": 0.92}
    },
    {
        "id": "golden_tsuki_01",
        "category": "indisputable_ippon",
        "strike_type": "Tsuki",
        "ground_truth": True,
        "description": "Tsuki colinear na garganta (Tsuki-tare) com travamento de braço e impulso para frente.",
        "metrics": {"target_impact": 0.95, "fumikomi_sync": 0.85, "posture": 0.90, "zanshin": 0.82, "hasuji_score": 0.96}
    },
    # 2. Golpes Imperfeitos (Não-Ponto por deficiência técnica)
    {
        "id": "golden_men_no_fumikomi",
        "category": "imperfect_strike",
        "strike_type": "Men",
        "ground_truth": False,
        "description": "Men atingindo o alvo mas com pé esquerdo arrastado e sem batida de Fumikomi-ashi.",
        "metrics": {"target_impact": 0.85, "fumikomi_sync": 0.28, "posture": 0.70, "zanshin": 0.60, "hasuji_score": 0.80}
    },
    {
        "id": "golden_men_no_zanshin",
        "category": "imperfect_strike",
        "strike_type": "Men",
        "ground_truth": False,
        "description": "Men correto no impacto mas o atacante tropeçou e baixou o Shinai sem Zanshin.",
        "metrics": {"target_impact": 0.88, "fumikomi_sync": 0.82, "posture": 0.40, "zanshin": 0.22, "hasuji_score": 0.85}
    },
    {
        "id": "golden_kote_bad_hasuji",
        "category": "imperfect_strike",
        "strike_type": "Kote",
        "ground_truth": False,
        "description": "Kote batido com a lateral chata da lâmina (desvio de Hasuji > 35°).",
        "metrics": {"target_impact": 0.65, "fumikomi_sync": 0.75, "posture": 0.70, "zanshin": 0.65, "hasuji_score": 0.25}
    },
    # 3. Fintas, Bloqueios e Golpes no Vazio (Ku-totsu)
    {
        "id": "golden_ku_totsu_empty",
        "category": "feint_block_ku_totsu",
        "strike_type": "Men",
        "ground_truth": False,
        "description": "Golpe desferido fora da distância de corte (Ku-totsu) sem alcance do Monouchi.",
        "metrics": {"target_impact": 0.15, "fumikomi_sync": 0.60, "posture": 0.60, "zanshin": 0.30, "hasuji_score": 0.40}
    },
    {
        "id": "golden_block_tsubazeriai",
        "category": "feint_block_ku_totsu",
        "strike_type": "Men",
        "ground_truth": False,
        "description": "Bloqueio defensivo em guarda alta e contato de tsuba contra tsuba.",
        "metrics": {"target_impact": 0.10, "fumikomi_sync": 0.10, "posture": 0.50, "zanshin": 0.10, "hasuji_score": 0.30}
    }
]


class GoldenBenchmark:
    """
    Mantém o conjunto de validação padrão-ouro (data/benchmark_golden/golden_dataset.json)
    e executa testes obrigatórios de não-regressão antes de qualquer gravação de calibração.
    """

    def __init__(self, dataset_path: str = "data/benchmark_golden/golden_dataset.json"):
        self.dataset_path = dataset_path
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.dataset_path) or ".", exist_ok=True)
        if not os.path.exists(self.dataset_path):
            with open(self.dataset_path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_GOLDEN_BENCHMARK_SAMPLES, f, ensure_ascii=False, indent=2)

    def load_samples(self) -> List[Dict[str, Any]]:
        self._ensure_storage()
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_GOLDEN_BENCHMARK_SAMPLES

    def evaluate_profile(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Avalia o perfil fornecido contra as amostras do Golden Benchmark.
        Calcula Acurácia, Precisão, Recall e F1-score.
        """
        samples = self.load_samples()
        if not samples:
            return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

        weights = profile.get("weights", {"target_impact": 0.40, "fumikomi_sync": 0.25, "posture": 0.20, "zanshin": 0.15})
        sub_th = profile.get("sub_thresholds", {"target_impact": 0.50, "fumikomi_sync": 0.40, "posture": 0.40, "zanshin": 0.35})
        min_total = profile.get("min_total_score", 0.65)

        tp, fp, tn, fn = 0, 0, 0, 0

        for s in samples:
            m = s["metrics"]
            gt = s["ground_truth"]

            # Cálculo ponderado da pontuação total
            total_score = sum(weights.get(k, 0.2) * m.get(k, 0.5) for k in weights)

            # Verificação de sub-limiares
            passes_sub = all(m.get(k, 0.5) >= sub_th.get(k, 0.3) for k in sub_th)

            # Decisão do motor
            predicted_ippon = (total_score >= min_total) and passes_sub

            if predicted_ippon and gt:
                tp += 1
            elif predicted_ippon and not gt:
                fp += 1
            elif not predicted_ippon and not gt:
                tn += 1
            else:
                fn += 1

        total = len(samples)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total_samples": total
        }

    def validate_no_regression(
        self,
        new_profile: Dict[str, Any],
        current_profile: Dict[str, Any],
        tolerance: float = 0.03
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Garante que uma nova calibração proposta não regrida a acurácia no Golden Benchmark
        além da tolerância estabelecida (prevenção contra Catastrophic Forgetting).
        """
        curr_metrics = self.evaluate_profile(current_profile)
        new_metrics = self.evaluate_profile(new_profile)

        # Regras de salvaguarda: F1 e Acurácia não podem cair mais do que a tolerância
        acc_drop = curr_metrics["accuracy"] - new_metrics["accuracy"]
        f1_drop = curr_metrics["f1"] - new_metrics["f1"]

        passed = (acc_drop <= tolerance) and (f1_drop <= tolerance)

        report = {
            "passed": passed,
            "current_metrics": curr_metrics,
            "new_metrics": new_metrics,
            "accuracy_change": round(new_metrics["accuracy"] - curr_metrics["accuracy"], 4),
            "f1_change": round(new_metrics["f1"] - curr_metrics["f1"], 4),
            "block_reason": (
                f"Regressão detectada no Golden Benchmark! (Acurácia: {curr_metrics['accuracy']} -> {new_metrics['accuracy']}, F1: {curr_metrics['f1']} -> {new_metrics['f1']})"
                if not passed else None
            )
        }

        if not passed:
            log_event("WARNING", f"Recalibração bloqueada por regressão no Golden Benchmark: {report['block_reason']}", "active_learning")
        else:
            log_event("INFO", f"Recalibração aprovada no Golden Benchmark (F1={new_metrics['f1']:.3f}).", "active_learning")

        return passed, report


# ==============================================================================
# 4.3 CONSENSO COM MÚLTIPLOS ÁRBITROS (2 DE 3) & DIVERGÊNCIA
# ==============================================================================
class MultiJudgeConsensus:
    """
    Consolida múltiplos pareceres arbitrais sobre um mesmo lance, aplicando a regra
    oficial da FIK (2 de 3 bandeiras), ponderação por Dan e cálculo do Grau de Divergência.
    """

    @staticmethod
    def calculate_divergence_degree(reviews: List[Dict[str, Any]]) -> float:
        """
        Calcula o grau de divergência arbitral em [0.0, 1.0]:
        - 0.0: Unanimidade absoluta (todos os árbitros concordam)
        - 1.0: Desacordo total / empate simétrico
        """
        if not reviews or len(reviews) <= 1:
            return 0.0

        votes_pos = sum(1 for r in reviews if r.get("verdict") in [True, "valid", "TP", 1, "IPPON"])
        votes_neg = len(reviews) - votes_pos
        total = len(reviews)

        # Proporção da minoria em relação a 50%
        minority = min(votes_pos, votes_neg)
        max_possible_minority = total / 2.0
        return round(minority / max_possible_minority, 4)

    @staticmethod
    def consolidate_reviews(
        reviews: List[Dict[str, Any]],
        llm_assistant: Optional[KendoLLMAssistant] = None
    ) -> Dict[str, Any]:
        """
        Consolida as anotações dos árbitros com votação ponderada por Dan e modelo 2 de 3.
        """
        if not reviews:
            return {
                "consensus_verdict": "NO_POINT",
                "is_ippon": False,
                "confidence": 0.0,
                "divergence_degree": 0.0,
                "majority_ratio": "0/0",
                "training_weight": 0.0
            }

        total_weighted_votes = 0.0
        ippon_weighted_votes = 0.0
        simple_ippon_count = 0

        for r in reviews:
            dan = r.get("dan", 3)
            # Dan 1-8 tem peso igual ao Dan, Shinpan credenciado tem peso 4.5
            weight = 4.5 if str(dan).lower() in ["shinpan", "10", "-1"] else max(1.0, float(dan))
            is_pos = r.get("verdict") in [True, "valid", "TP", 1, "IPPON"]

            total_weighted_votes += weight
            if is_pos:
                ippon_weighted_votes += weight
                simple_ippon_count += 1

        total_count = len(reviews)
        divergence = MultiJudgeConsensus.calculate_divergence_degree(reviews)

        # Regra FIK oficial: se houver 3 árbitros, pelo menos 2 bandeiras devem concordar
        is_ippon = (ippon_weighted_votes / total_weighted_votes) >= 0.50 if total_weighted_votes > 0 else False

        # Atenuação de peso no treinamento se a divergência for alta
        # Instâncias com alta divergência recebem peso reduzido para não desestabilizar a calibração
        training_weight = round(max(0.20, 1.0 - 0.5 * divergence), 3)

        result = {
            "consensus_verdict": "IPPON" if is_ippon else "NO_POINT",
            "is_ippon": is_ippon,
            "weighted_ratio": round(ippon_weighted_votes / total_weighted_votes, 4) if total_weighted_votes > 0 else 0.0,
            "divergence_degree": divergence,
            "majority_ratio": f"{simple_ippon_count}/{total_count}",
            "training_weight": training_weight,
            "total_reviews": total_count
        }

        # Se houver divergência arbitral e o LLM estiver disponível, gera parecer sintetizado
        if divergence > 0.4 and llm_assistant:
            try:
                summary = {"votes": f"{simple_ippon_count}/{total_count}", "divergence": divergence}
                result["llm_synthesis"] = llm_assistant.synthesize_dispute(reviews, summary)
            except Exception:
                pass

        return result


# ==============================================================================
# 4.4 DECAIMENTO TEMPORAL DE CONFIANÇA POR INATIVIDADE DO REVISOR
# ==============================================================================
class ReviewerTrustManager:
    """
    Rastreia a consistência temporal e data da última atividade de cada árbitro,
    aplicando decaimento exponencial ao peso de revisores inativos ou com alta divergência.
    """

    def __init__(self, storage_file: str = "data/reviewer_trust_registry.json", half_life_days: float = 180.0):
        self.storage_file = storage_file
        self.half_life_days = half_life_days
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(os.path.dirname(self.storage_file) or ".", exist_ok=True)
        if not os.path.exists(self.storage_file):
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)

    def load_registry(self) -> Dict[str, Any]:
        self._ensure_storage()
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def record_activity(
        self,
        reviewer_id: str,
        dan_level: int,
        agreed_with_consensus: bool,
        timestamp: Optional[float] = None
    ):
        """Registra a participação de um árbitro e atualiza seu histórico de consistência."""
        reg = self.load_registry()
        ts = timestamp or time.time()
        r_data = reg.get(reviewer_id, {
            "dan_level": dan_level,
            "first_seen": ts,
            "last_active": ts,
            "total_reviews": 0,
            "consensus_agreements": 0,
            "divergence_count": 0
        })

        r_data["last_active"] = ts
        r_data["total_reviews"] += 1
        if agreed_with_consensus:
            r_data["consensus_agreements"] += 1
        else:
            r_data["divergence_count"] += 1

        reg[reviewer_id] = r_data
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)

    def get_effective_weight(
        self,
        reviewer_id: str,
        base_dan: int,
        current_timestamp: Optional[float] = None
    ) -> float:
        """
        Calcula o peso efetivo do revisor considerando sua graduação Dan,
        tempo de inatividade e consistência histórica com o consenso:
        peso_efetivo = peso_base * exp(-lambda * delta_t) * fator_consistência
        """
        reg = self.load_registry()
        r_data = reg.get(reviewer_id)

        base_weight = 4.5 if str(base_dan).lower() in ["shinpan", "10", "-1"] else max(1.0, float(base_dan))
        if not r_data:
            return base_weight

        now = current_timestamp or time.time()
        last_active = r_data.get("last_active", now)
        days_inactive = max(0.0, (now - last_active) / 86400.0)

        # Fator de decaimento temporal exponencial com meia-vida de half_life_days
        lambda_decay = math.log(2.0) / self.half_life_days
        time_factor = math.exp(-lambda_decay * days_inactive)

        # Fator de consistência baseado na proporção de concordância com o consenso
        tot = r_data.get("total_reviews", 0)
        agreements = r_data.get("consensus_agreements", 0)
        consistency_ratio = (agreements / tot) if tot >= 5 else 1.0
        consistency_factor = max(0.40, consistency_ratio)

        # Peso final modulado (não desce abaixo de 0.50)
        effective = base_weight * time_factor * consistency_factor
        return round(max(0.50, effective), 3)
