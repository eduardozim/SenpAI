from src.utils.test_isolation import get_safe_data_path, assert_not_production_write
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
DOUBT_TYPES = ("pontuacao", "golpe_filtro", "tecnica", "regra", "conflito", "bandeira", "modalidade", "padrao")
# Dúvidas sobre um golpe de luta analisada: o golpe vira um corte de ~10 s em data/videos (treino)
ANALYSIS_CLIP_DOUBTS = ("pontuacao", "golpe_filtro", "tecnica", "regra", "padrao")
DOUBT_LABELS = {
    "pontuacao": "Pontuação", "golpe_filtro": "É golpe?", "tecnica": "Técnica", "regra": "Regra",
    "conflito": "Conflito entre fontes", "bandeira": "Bandeiras", "modalidade": "Modalidade",
    "padrao": "Fora do padrão dos validados",
}
MAX_QUEUE_ITEMS = 100


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
        self.queue_file = get_safe_data_path(queue_file, 'test_active_learning_queue.json')
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
        enrich_with_llm: bool = True,
        clip_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Se o lance for incerto, enriquece com diagnóstico do LLM e enfileira para curadoria ativa.
        """
        if not self.is_uncertain(confidence):
            return None

        uncertainty_score = self.calculate_uncertainty(confidence)
        # Impacto: um golpe que o SenpAI deu como ponto mexe no placar
        impact = 1.0 if strike_data.get("is_valid") else 0.7

        # Diagnóstico com LLM para acelerar o processo de curadoria do Sensei
        llm_triage = None
        if enrich_with_llm and self.llm_assistant:
            try:
                llm_triage = self.llm_assistant.analyze_uncertain_strike(
                    strike_data, profile_name=profile_name, confidence=confidence
                )
            except Exception as e:
                log_event("WARNING", f"Falha ao gerar triage LLM para lance incerto: {e}", "active_learning")

        # Normalização de caminhos portáteis para servidor/máquina atual
        from src.utils.strike_clip_manager import get_cross_platform_basename
        if clip_path:
            try:
                rel = os.path.relpath(clip_path)
                if not rel.startswith(".."):
                    clip_path = rel.replace("\\", "/")
                else:
                    clip_path = clip_path.replace("\\", "/")
            except Exception:
                clip_path = clip_path.replace("\\", "/")

        if video_source:
            try:
                rel_v = os.path.relpath(video_source)
                if not rel_v.startswith(".."):
                    video_source = rel_v.replace("\\", "/")
                elif "senpai_uploads" in video_source or get_cross_platform_basename(video_source).startswith(("yt_", "upload_")):
                    video_source = get_cross_platform_basename(video_source)
            except Exception:
                pass

        return self.enqueue_doubt(
            doubt_type="pontuacao",
            video_source=video_source,
            strike_data=strike_data,
            uncertainty=uncertainty_score,
            impact=impact,
            reason=f"Nota do golpe na faixa de incerteza ({confidence * 100:.0f}%).",
            profile_name=profile_name,
            clip_path=clip_path,
            confidence=confidence,
            llm_triage=llm_triage,
        )

    def enqueue_doubt(
        self,
        doubt_type: str,
        video_source: str,
        strike_data: Dict[str, Any],
        uncertainty: float,
        reason: str,
        impact: float = 1.0,
        profile_name: str = "normal",
        clip_path: Optional[str] = None,
        confidence: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
        kenshi: Optional[str] = None,
        llm_triage: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fila de Curadoria de TODAS as dúvidas (Pendência I12). Tipos:
        pontuacao (nota do golpe), golpe_filtro (filtro de golpes), tecnica, regra (ai-uchi, golpe após o
        yame...), conflito (fontes do banco de rótulos discordam), bandeira, modalidade, padrao (golpe que destoa
        dos golpes validados da mesma técnica, pela referência do ciclo de treino).
        Prioridade = incerteza × impacto (as que mais ensinam primeiro). Não repete a mesma dúvida
        (mesmo vídeo, tipo e instante ±1 s) enquanto ela estiver pendente.
        """
        if doubt_type not in DOUBT_TYPES:
            raise ValueError(f"Tipo de dúvida desconhecido: {doubt_type}")
        original_source = video_source
        from src.utils.strike_clip_manager import get_cross_platform_basename
        if clip_path:
            try:
                rel = os.path.relpath(clip_path)
                clip_path = (rel if not rel.startswith("..") else clip_path).replace("\\", "/")
            except Exception:
                clip_path = clip_path.replace("\\", "/")
        if video_source:
            try:
                rel_v = os.path.relpath(video_source)
                if not rel_v.startswith(".."):
                    video_source = rel_v.replace("\\", "/")
                elif "senpai_uploads" in video_source or get_cross_platform_basename(video_source).startswith(("yt_", "upload_")):
                    video_source = get_cross_platform_basename(video_source)
            except Exception:
                pass

        kenshi = kenshi or strike_data.get("kenshi")
        strike_time = strike_data.get("timestamp_seconds")
        if strike_time is None and strike_data.get("impact_frame") is not None and strike_data.get("fps"):
            strike_time = float(strike_data["impact_frame"]) / float(strike_data["fps"])

        queue = self.get_queue()
        if strike_time is not None:
            src_base = get_cross_platform_basename(video_source or "")
            for it in queue:
                if (it.get("status") == "pending_curation" and it.get("tipo_duvida", "pontuacao") == doubt_type
                        and get_cross_platform_basename(it.get("video_source") or "") == src_base
                        and it.get("tempo_golpe_s") is not None and abs(float(it["tempo_golpe_s"]) - float(strike_time)) < 1.0):
                    return None

        priority = round(max(0.0, min(1.0, float(uncertainty))) * max(0.0, min(1.0, float(impact))), 4)
        entry = {
            "id": f"unc_{int(time.time() * 1000)}_{len(queue)}",
            "timestamp": datetime.datetime.now().isoformat(),
            "video_source": video_source,
            "clip_path": clip_path,
            "strike_type": strike_data.get("strike_type", "Men"),
            # Só números reais: a nota do golpe ou a probabilidade de um modelo (dúvidas de regra/conflito não têm)
            "confidence": None if confidence is None else round(float(confidence), 4),
            "uncertainty_score": round(float(uncertainty), 4),
            "prioridade": priority,
            "tipo_duvida": doubt_type,
            "motivo": reason,
            "opcoes": options or {},
            "kenshi": kenshi,
            "tempo_golpe_s": None if strike_time is None else round(float(strike_time), 3),
            "strike_data": strike_data,
            "profile_name": profile_name,
            "llm_triage": llm_triage,
            "status": "pending_curation"
        }

        # Se não foi fornecido clip_path, tentar gerar ou associar sob demanda
        if not entry.get("clip_path"):
            try:
                from src.utils.strike_clip_manager import get_or_create_strike_clip
                entry["clip_path"] = get_or_create_strike_clip(entry)
            except Exception:
                pass

        # Golpe em dúvida numa luta analisada: guarda um corte de ~10 s em data/videos para o treino
        # (recortes de bandeira, gabarito, ciclo). Não repete golpes a menos de 10 s de um corte já guardado.
        if doubt_type in ANALYSIS_CLIP_DOUBTS and strike_time is not None:
            try:
                from src.utils.video_library import save_analysis_clip
                saved = save_analysis_clip(original_source, float(strike_time))
                if saved:
                    entry["corte_treino"] = saved.replace("\\", "/")
            except Exception:
                pass

        queue.append(entry)
        # As dúvidas que mais ensinam primeiro (itens antigos sem prioridade usam a incerteza)
        queue.sort(key=lambda x: x.get("prioridade", x.get("uncertainty_score", 0.0)), reverse=True)
        queue = queue[:MAX_QUEUE_ITEMS]

        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False, indent=2)

        log_event("INFO", f"Dúvida '{doubt_type}' enfileirada para curadoria (prioridade {priority:.2f}): {reason}", "active_learning")
        return entry

    def get_queue(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna os itens da fila de curadoria ativa (ou histórico se status=='curated')."""
        self._ensure_storage()
        if status == "curated":
            history_file = os.path.join(os.path.dirname(self.queue_file) or ".", "active_learning_curated_history.json")
            if os.path.isfile(history_file):
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return []
            return []

        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                from src.utils.strike_clip_manager import get_cross_platform_basename
                for it in items:
                    c_p = it.get("clip_path")
                    if c_p and not os.path.isfile(c_p):
                        c_base = get_cross_platform_basename(c_p)
                        cand = os.path.join("data", "active_learning_clips", c_base)
                        if os.path.isfile(cand):
                            it["clip_path"] = cand.replace("\\", "/")
                    v_s = it.get("video_source")
                    if v_s and not os.path.isfile(v_s):
                        v_base = get_cross_platform_basename(v_s)
                        if "senpai_uploads" in str(v_s) or v_base.startswith("yt_") or v_base.startswith("upload_"):
                            it["video_source"] = v_base
                if status:
                    return [it for it in items if it.get("status") == status]
                return items
        except Exception:
            return []

    def get_history(self) -> List[Dict[str, Any]]:
        """Retorna os registros históricos arquivados de lances já curados."""
        history_file = os.path.join(os.path.dirname(self.queue_file) or ".", "active_learning_curated_history.json")
        if os.path.isfile(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def ensure_item_clip(self, item: Dict[str, Any]) -> Optional[str]:
        """Garante que o item possua um clipe de demonstração gerado/recortado em disco."""
        try:
            from src.utils.strike_clip_manager import get_or_create_strike_clip
            clip_path = get_or_create_strike_clip(item)
            if clip_path and item.get("clip_path") != clip_path:
                item["clip_path"] = clip_path
                # Atualizar na persistência
                queue = self.get_queue()
                for it in queue:
                    if it.get("id") == item.get("id"):
                        it["clip_path"] = clip_path
                        break
                with open(self.queue_file, "w", encoding="utf-8") as f:
                    json.dump(queue, f, ensure_ascii=False, indent=2)
            return clip_path
        except Exception as e:
            log_event("WARNING", f"Falha ao assegurar clipe do item {item.get('id')}: {e}", "active_learning")
            return None

    def resolve_item(
        self,
        item_id: str,
        label_approved: bool,
        reviewer_dan: int,
        notes: str = "",
        cleanup_clip: bool = True,
        decision: Optional[str] = None,
        *args,
        **kwargs
    ) -> bool:
        """
        Marca um item da fila como curado e resolvido por um árbitro:
        1. Apaga o arquivo de vídeo do golpe em disco.
        2. Salva o registro curado no histórico de auditoria com categorização explícita
           (IPPON: válido, INVALID_STRIKE: tentativa inválida, NO_STRIKE: nenhuma tentativa).
        3. Remove o item definitivamente da lista/fila de pendências ativas.
        """
        queue = self.get_queue()
        target_item = None
        remaining_queue = []
        target_id_str = item_id.strip()

        for it in queue:
            cur_id_str = str(it.get("id", "")).strip()
            if cur_id_str == target_id_str:
                target_item = it
            else:
                remaining_queue.append(it)

        # Fallback defensivo por índice posicional caso o ID tenha sido gerado como item_{idx}
        if not target_item and target_id_str.startswith("item_"):
            try:
                idx_val = int(target_id_str.split("_")[1])
                if 0 <= idx_val < len(queue):
                    target_item = queue[idx_val]
                    remaining_queue = [it for i, it in enumerate(queue) if i != idx_val]
            except Exception:
                pass

        if not target_item:
            log_event("WARNING", f"Tentativa de resolver item inexistente na fila de curadoria: {item_id}", "active_learning")
            return False

        # 1. Apagar o vídeo do golpe em disco
        clip_to_delete = target_item.get("clip_path")
        if cleanup_clip:
            try:
                from src.utils.strike_clip_manager import delete_strike_clip
                if clip_to_delete:
                    delete_strike_clip(clip_to_delete)
                cand_file = os.path.join("data", "active_learning_clips", f"{item_id}_corte.mp4")
                if os.path.isfile(cand_file):
                    delete_strike_clip(cand_file)
            except Exception as e:
                log_event("WARNING", f"Erro ao apagar clipe de demonstração pós-resolução: {e}", "active_learning")

        # 2. Registrar no histórico permanente de curadorias
        resolved_category = decision or ("IPPON" if label_approved else "INVALID_STRIKE")
        target_item["status"] = "discarded" if resolved_category in ("DISCARDED", "DISCARD") else "curated"
        target_item["decision"] = resolved_category
        target_item["curated_label"] = resolved_category
        target_item["curator_dan"] = reviewer_dan
        target_item["curator_notes"] = notes
        target_item["curated_at"] = datetime.datetime.now().isoformat()
        target_item["clip_deleted"] = True
        target_item["clip_path"] = None

        history_file = os.path.join(os.path.dirname(self.queue_file) or ".", "active_learning_curated_history.json")
        try:
            history_list = []
            if os.path.isfile(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    history_list = json.load(f)
            history_list.append(target_item)
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log_event("WARNING", f"Falha ao arquivar histórico permanente de curadoria: {e}", "active_learning")

        # 3. Remover definitivamente o item da fila ativa
        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump(remaining_queue, f, ensure_ascii=False, indent=2)

        log_event(
            "INFO",
            f"Item {item_id} resolvido ({target_item['curated_label']}): vídeo excluído e item removido da lista ativa.",
            "active_learning"
        )
        return True

    def discard_item(
        self,
        item_id: str,
        reviewer_dan: int = 5,
        notes: str = "Descartado: Resultado indeterminado",
        cleanup_clip: bool = True,
        *args,
        **kwargs
    ) -> bool:
        """Descarta um item da fila de curadoria sem afetar o treinamento (quando o resultado do lance é indeterminado)."""
        return self.resolve_item(
            item_id=item_id,
            label_approved=False,
            reviewer_dan=reviewer_dan,
            notes=notes,
            cleanup_clip=cleanup_clip,
            decision="DISCARDED",
            *args,
            **kwargs
        )

    def clear_queue(self, delete_clips: bool = True) -> Dict[str, Any]:
        """
        Limpa todos os itens da fila de curadoria ativa (pendências),
        opcionalmente excluindo os arquivos de vídeo correspondentes em disco.
        """
        self._ensure_storage()
        items = self.get_queue()
        deleted_clips_count = 0

        if delete_clips:
            try:
                from src.utils.strike_clip_manager import delete_strike_clip
                for it in items:
                    c_p = it.get("clip_path")
                    it_id = it.get("id")
                    if c_p:
                        try:
                            delete_strike_clip(c_p)
                            deleted_clips_count += 1
                        except Exception:
                            pass
                    if it_id:
                        cand = os.path.join("data", "active_learning_clips", f"{it_id}_corte.mp4")
                        if os.path.isfile(cand):
                            try:
                                delete_strike_clip(cand)
                                deleted_clips_count += 1
                            except Exception:
                                pass
            except Exception as e:
                log_event("WARNING", f"Falha parcial ao apagar clipes durante limpeza de fila: {e}", "active_learning")

        with open(self.queue_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

        log_event(
            "INFO",
            f"Fila de Curadoria Ativa limpa com sucesso: {len(items)} pendências removidas e {deleted_clips_count} clipes excluídos.",
            "active_learning"
        )
        return {"cleared_items": len(items), "clips_deleted": deleted_clips_count}

    def clear_all(self, include_history: bool = True, delete_clips: bool = True) -> Dict[str, Any]:
        """
        Limpa a fila de curadoria ativa, histórico de lances curados e clipes de vídeo em disco.
        """
        res = self.clear_queue(delete_clips=delete_clips)
        cleared_hist = 0

        if include_history:
            history_file = os.path.join(os.path.dirname(self.queue_file) or ".", "active_learning_curated_history.json")
            if os.path.isfile(history_file):
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        hist_items = json.load(f)
                        cleared_hist = len(hist_items)
                except Exception:
                    cleared_hist = 0
                try:
                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump([], f, ensure_ascii=False, indent=2)
                except Exception as e:
                    log_event("WARNING", f"Erro ao limpar histórico de curadoria: {e}", "active_learning")

        # Varredura em data/active_learning_clips para evitar arquivos órfãos
        if delete_clips:
            clips_dir = os.path.join("data", "active_learning_clips")
            if os.path.isdir(clips_dir):
                try:
                    for f_name in os.listdir(clips_dir):
                        if f_name.endswith((".mp4", ".avi", ".mov")):
                            full_p = os.path.join(clips_dir, f_name)
                            try:
                                os.remove(full_p)
                                res["clips_deleted"] = res.get("clips_deleted", 0) + 1
                            except Exception:
                                pass
                except Exception:
                    pass

        res["cleared_history"] = cleared_hist
        return res


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


GOLDEN_DATASET_DEFAULT_PATH = "data/benchmark_golden/golden_dataset.json"
GOLDEN_MATCHES_FILE = "golden_matches.json"
GOLDEN_HISTORY_FILE = "benchmark_history.jsonl"
REAL_MATCH_CATEGORY = "real_match"
# Abaixo disso a trava continua com as amostras sintéticas (com poucas amostras, 1 erro pesa demais)
MIN_REAL_SAMPLES_FOR_GATE = 30
GOLDEN_FROZEN_FIELDS = ("aka", "shiro", "pontos_completos", "inverter_aka_shiro", "pontos")


class GoldenBenchmark:
    """
    Conjunto de validação padrão-ouro do SenpAI (data/benchmark_golden/), único no sistema.

    - Lutas golden: lutas reais com gabarito completo, congeladas em golden_matches.json
      (cópia do gabarito no momento do congelamento) e nunca usadas no treino nem na curadoria.
    - Amostras (golden_dataset.json): golpes avaliados nessas lutas, com rótulo vindo do placar
      oficial (categoria "real_match"), mais as amostras sintéticas originais, que só valem
      enquanto houver menos de MIN_REAL_SAMPLES_FOR_GATE golpes reais.
    - Trava anti-regressão (validate_no_regression) antes de qualquer gravação de calibração.
    - Benchmark por luta (run_matches): placar oficial × SenpAI, com histórico.

    Todas as métricas usam a fonte única de src.engine.weak_labels
    (precision_recall_f1, match_metrics, aggregate_match_metrics).
    """

    def __init__(self, dataset_path: str = GOLDEN_DATASET_DEFAULT_PATH):
        self.dataset_path = get_safe_data_path(dataset_path, "test_golden_dataset.json")
        self.golden_dir = os.path.dirname(self.dataset_path) or "."
        self.matches_path = os.path.join(self.golden_dir, GOLDEN_MATCHES_FILE)
        self.history_path = os.path.join(self.golden_dir, GOLDEN_HISTORY_FILE)
        self._ensure_storage()

    def _ensure_storage(self):
        os.makedirs(self.golden_dir, exist_ok=True)
        if not os.path.exists(self.dataset_path):
            self._write_json(self.dataset_path, DEFAULT_GOLDEN_BENCHMARK_SAMPLES)

    @staticmethod
    def _write_json(path: str, data: Any) -> None:
        from src.utils.file_utils import safe_write_json
        safe_write_json(path, data, indent=2)

    # --------------------------------------------------------------------------
    # Amostras
    # --------------------------------------------------------------------------
    def _load_all_samples(self) -> List[Dict[str, Any]]:
        self._ensure_storage()
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return list(DEFAULT_GOLDEN_BENCHMARK_SAMPLES)

    def load_samples(self) -> List[Dict[str, Any]]:
        """Golpes reais quando já são suficientes (e têm ao menos 1 positivo); senão, as sintéticas."""
        samples = self._load_all_samples()
        real = [s for s in samples if s.get("category") == REAL_MATCH_CATEGORY]
        if len(real) >= MIN_REAL_SAMPLES_FOR_GATE and any(s.get("ground_truth") for s in real):
            return real
        return [s for s in samples if s.get("category") != REAL_MATCH_CATEGORY] or list(DEFAULT_GOLDEN_BENCHMARK_SAMPLES)

    def samples_source(self) -> Dict[str, Any]:
        """De onde vêm as amostras da trava: 'real' ou 'sintetico', com as contagens."""
        real = sum(1 for s in self._load_all_samples() if s.get("category") == REAL_MATCH_CATEGORY)
        in_use = self.load_samples()
        source = "real" if in_use and in_use[0].get("category") == REAL_MATCH_CATEGORY else "sintetico"
        return {"fonte": source, "golpes_reais": real, "minimo_reais": MIN_REAL_SAMPLES_FOR_GATE, "amostras_em_uso": len(in_use)}

    def replace_real_samples(self, real_samples: List[Dict[str, Any]]) -> None:
        """Troca as amostras reais pelas da última execução, preservando as sintéticas."""
        synthetic = [s for s in self._load_all_samples() if s.get("category") != REAL_MATCH_CATEGORY]
        assert_not_production_write(self.dataset_path, "GoldenBenchmark.replace_real_samples")
        self._write_json(self.dataset_path, synthetic + list(real_samples))

    @staticmethod
    def report_to_samples(report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Golpes avaliados de uma luta golden → amostras (notas 0..1, verdade pelo placar oficial)."""
        from src.engine import weak_labels as wl
        samples = []
        for ev in report["events"]["items"]:
            if ev["weak_label"] == wl.POSITIVE:
                truth = True
            elif ev["weak_label"] in wl.NEGATIVE_LABELS or ev["weak_label"] == wl.AMBIGUOUS:
                truth = False  # gabarito completo: ambíguo não é ponto novo
            else:
                continue
            sub = ev.get("sub_scores") or {}
            if not sub:
                continue
            metrics = {k: round(float(sub[k]) / 100.0, 4) for k in ("target_impact", "fumikomi_sync", "posture", "zanshin") if k in sub}
            if "hasuji" in sub:
                metrics["hasuji_score"] = round(float(sub["hasuji"]) / 100.0, 4)
            samples.append({
                "id": f"{report['video_name']}#{ev['attacker_id']}#{ev['impact_frame']}",
                "category": REAL_MATCH_CATEGORY,
                "strike_type": str(ev.get("strike_type") or "MEN").capitalize(),
                "ground_truth": truth,
                "description": f"{report['video_name']} — {ev['attacker_id']} em {ev['time_sec']:.1f}s ({ev['weak_label']})",
                "metrics": metrics,
            })
        return samples

    # --------------------------------------------------------------------------
    # Lutas congeladas
    # --------------------------------------------------------------------------
    def load_matches(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.isfile(self.matches_path):
            return {}
        try:
            with open(self.matches_path, "r", encoding="utf-8") as f:
                return json.load(f).get("lutas", {})
        except Exception:
            return {}

    def _save_matches(self, matches: Dict[str, Dict[str, Any]]) -> None:
        self._write_json(self.matches_path, {"versao": 1, "lutas": matches})

    def is_golden_video(self, video_name: str) -> bool:
        return os.path.basename(video_name) in self.load_matches()

    @staticmethod
    def _frozen_copy(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {k: json.loads(json.dumps(entry.get(k))) for k in GOLDEN_FROZEN_FIELDS if k in entry}

    def differs_from_gabarito(self, video_name: str, entry: Dict[str, Any]) -> bool:
        """True quando o gabarito foi editado depois do congelamento da luta."""
        frozen = self.load_matches().get(os.path.basename(video_name))
        if not frozen:
            return False
        current = self._frozen_copy(entry)
        return any(frozen.get(k) != current.get(k) for k in GOLDEN_FROZEN_FIELDS)

    def freeze_match(self, video_name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Congela (ou recongela) uma luta. Exige gabarito completo: sem ele os negativos não são confiáveis."""
        from src.engine import weak_labels as wl
        if not entry or not entry.get("pontos_completos"):
            raise ValueError("Só lutas com todos os pontos listados (pontos_completos) entram no golden.")
        if entry.get("ignorar"):
            raise ValueError("Vídeo marcado como \"não é luta\": não pode entrar no golden.")
        if str(entry.get("origem", "")).startswith("automatico"):
            raise ValueError("O golden só aceita gabarito humano: pontos lidos automaticamente (bandeiras) não entram.")
        wl.parse_official_points(entry)  # valida tempos e técnicas
        matches = self.load_matches()
        frozen = self._frozen_copy(entry)
        frozen["congelado_em"] = datetime.datetime.now().isoformat(timespec="seconds")
        matches[os.path.basename(video_name)] = frozen
        self._save_matches(matches)
        log_event("INFO", f"Luta congelada no golden benchmark: {os.path.basename(video_name)}", "active_learning")
        return frozen

    def unfreeze_match(self, video_name: str) -> bool:
        matches = self.load_matches()
        removed = matches.pop(os.path.basename(video_name), None) is not None
        if removed:
            self._save_matches(matches)
        return removed

    # --------------------------------------------------------------------------
    # Benchmark por luta
    # --------------------------------------------------------------------------
    def run_matches(
        self,
        pipeline,
        videos_dir: str,
        tolerance_sec: Optional[float] = None,
        progress: Optional[Any] = None,
        record_history: bool = True,
        update_samples: bool = True
    ) -> Dict[str, Any]:
        """
        Reanalisa as lutas golden (via cache de features) e mede o placar oficial × SenpAI.
        record_history/update_samples=False servem para comparar um modelo desafiante sem alterar o golden.
        """
        from src.engine import weak_labels as wl
        tol = wl.DEFAULT_TOLERANCE_SEC if tolerance_sec is None else tolerance_sec
        matches = self.load_matches()
        rows, samples, missing, failures = [], [], [], []
        names = sorted(matches)
        for idx, name in enumerate(names, start=1):
            if progress:
                progress(idx, len(names), name)
            video = os.path.join(videos_dir, name)
            if not os.path.isfile(video):
                missing.append(name)
                continue
            entry = matches[name]
            try:
                report = wl.weak_label_video(pipeline, video, entry, tolerance_sec=tol,
                                             invert_combatants=bool(entry.get("inverter_aka_shiro", False)))
            except Exception as e:
                failures.append({"video_name": name, "erro": str(e)})
                continue
            rows.append(report["metricas"])
            samples.extend(self.report_to_samples(report))

        result = {
            "executado_em": datetime.datetime.now().isoformat(timespec="seconds"),
            "perfil": getattr(getattr(pipeline, "calibrator", None), "active_config", {}).get("name", "normal"),
            "tolerancia_s": tol,
            "lutas": rows,
            "total": wl.aggregate_match_metrics(rows),
            "golpes_reais": len(samples),
            "videos_ausentes": missing,
            "falhas": failures,
        }
        if rows:
            if update_samples:
                self.replace_real_samples(samples)
            if record_history:
                with open(self.history_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({k: result[k] for k in ("executado_em", "perfil", "total", "golpes_reais")}, ensure_ascii=False) + "\n")
        return result

    def load_history(self) -> List[Dict[str, Any]]:
        if not os.path.isfile(self.history_path):
            return []
        rows = []
        with open(self.history_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        return rows

    # --------------------------------------------------------------------------
    # Trava anti-regressão (amostras)
    # --------------------------------------------------------------------------
    def evaluate_profile(self, profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Avalia o perfil fornecido contra as amostras do Golden Benchmark.
        Calcula Acurácia, Precisão, Recall e F1-score (fórmula única de weak_labels).
        """
        from src.engine.weak_labels import precision_recall_f1
        samples = self.load_samples()
        if not samples:
            return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

        from src.engine.calibrator import decide_strike

        tp, fp, tn, fn = 0, 0, 0, 0

        for s in samples:
            m = s["metrics"]
            gt = s["ground_truth"]

            # Mesma decisão da análise de luta (pesos por tipo de golpe, mistura do hasuji, só alvo/hasuji eliminam)
            predicted_ippon = decide_strike(
                m.get("target_impact", 0.5), m.get("fumikomi_sync", 0.5), m.get("posture", 0.5), m.get("zanshin", 0.5),
                config=profile, strike_type=s.get("strike_type", "MEN"), hasuji_score=m.get("hasuji_score")
            )["is_valid"]

            if predicted_ippon and gt:
                tp += 1
            elif predicted_ippon and not gt:
                fp += 1
            elif not predicted_ippon and not gt:
                tn += 1
            else:
                fn += 1

        total = len(samples)
        prf = precision_recall_f1(tp, fp, fn)
        # Indefinido (0/0) vira 0.0 aqui: a trava compara números
        return {
            "accuracy": round((tp + tn) / total, 4) if total > 0 else 0.0,
            "precision": prf["precisao"] or 0.0,
            "recall": prf["recall"] or 0.0,
            "f1": prf["f1"] or 0.0,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "total_samples": total,
            "source": "real" if samples[0].get("category") == REAL_MATCH_CATEGORY else "sintetico"
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


def is_golden_video(video_name: str) -> bool:
    """Atalho: a luta pertence ao golden benchmark (e por isso não pode virar treino)?"""
    try:
        return GoldenBenchmark().is_golden_video(video_name)
    except Exception:
        return False


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

        result: Dict[str, Any] = {
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
        self.storage_file = get_safe_data_path(storage_file, "test_reviewer_trust_registry.json")
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
