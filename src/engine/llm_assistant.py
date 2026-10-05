"""
Módulo de Integração com LLMs para Aprendizado Ativo e Captura de Movimentos no Kendo.
Permite ao SenpAI acelerar o aprendizado biomecânico e arbitral utilizando LLMs (Google Gemini,
endpoints OpenAI-compatíveis ou Motor Especialista de Regras FIK/AJKF integrado como fallback offline).

Funcionalidades principais:
1. Análise arbitral fundamentada de lances controversos e com alta incerteza.
2. Captura e rotulagem assistida de movimentos em vídeos e sessões de treino.
3. Síntese e desempate fundamentado de decisões divergentes de múltiplos árbitros.
"""

import os
import json
import re
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

from src.utils.logger_manager import log_event


class KendoLLMAssistant:
    """
    Assistente cognitivo baseado em LLM para aceleração do aprendizado em Kendo,
    análise de incerteza arbitral e rotulagem inteligente de movimentos.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
        provider: str = "auto"
    ):
        """
        Inicializa o assistente LLM.
        Se nenhuma chave de API for encontrada, opera em modo Especialista de Regras FIK (offline),
        garantindo funcionamento contínuo, seguro e sem falhas em qualquer ambiente.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name
        self.provider = provider

        # Se provider for auto, detecta pela chave ou convenção
        if self.provider == "auto":
            if os.environ.get("OPENAI_API_KEY") and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                self.provider = "openai"
            else:
                self.provider = "gemini"

    @property
    def is_online(self) -> bool:
        """Indica se há uma chave de API disponível para chamadas remotas de LLM."""
        return bool(self.api_key and len(self.api_key.strip()) > 8)

    # --------------------------------------------------------------------------
    # 1. ANÁLISE DE LANCES INCERTOS (ACTIVE LEARNING TRIAGE)
    # --------------------------------------------------------------------------
    def analyze_uncertain_strike(
        self,
        strike_data: Dict[str, Any],
        profile_name: str = "normal",
        confidence: float = 0.50
    ) -> Dict[str, Any]:
        """
        Avalia um lance em que o motor de visão teve incerteza (45% a 65% de confiança).
        Gera parecer técnico com embasamento no regulamento oficial da FIK/AJKF,
        identificando falhas no Ki-Ken-Tai-Ichi, Hasuji, Fumikomi ou Zanshin.
        """
        strike_type = strike_data.get("strike_type", "Men").capitalize()
        scores = strike_data.get("scores", {})
        sub_scores = strike_data.get("sub_scores", {})
        target = float(sub_scores.get("target_impact", scores.get("target_impact", 0.5)))
        fumi = float(sub_scores.get("fumikomi_sync", scores.get("fumikomi_sync", 0.5)))
        posture = float(sub_scores.get("posture", scores.get("posture", 0.5)))
        zanshin = float(sub_scores.get("zanshin", scores.get("zanshin", 0.5)))
        hasuji = float(strike_data.get("hasuji_score", 0.5))

        prompt = (
            f"Você é um árbitro internacional de Kendo chancelado pela FIK (8º Dan Hanshi).\n"
            f"Analise o seguinte golpe controverso avaliado pelo sistema computacional SenpAI com confiança incerta de {confidence*100:.1f}%:\n"
            f"- Tipo de golpe: {strike_type}\n"
            f"- Perfil arbitral: {profile_name}\n"
            f"- Impacto no alvo (Datotsu-bui): {target:.2f}/1.00\n"
            f"- Sincronismo Ki-Ken-Tai-Ichi / Fumikomi-ashi: {fumi:.2f}/1.00\n"
            f"- Postura (Shisei / Coluna ereta): {posture:.2f}/1.00\n"
            f"- Zanshin (Prontidão física e mental): {zanshin:.2f}/1.00\n"
            f"- Alinhamento da lâmina (Hasuji): {hasuji:.2f}/1.00\n\n"
            f"Retorne um JSON com a seguinte estrutura:\n"
            f"{{\n"
            f'  "verdict": "VALID_IPPON" | "NO_POINT" | "DOUBTFUL",\n'
            f'  "recommended_p_ippon": <float entre 0.0 e 1.0>,\n'
            f'  "primary_deficiency": "KI_KEN_TAI_ICHI" | "HASUJI" | "TARGET_IMPACT" | "POSTURE" | "ZANSHIN" | "NONE",\n'
            f'  "fik_rule_rationale": "<explicação técnica detalhada baseada no regulamento FIK>",\n'
            f'  "coaching_advice": "<recomendação prática para o Kenshi melhorar este golpe nos treinos>"\n'
            f"}}"
        )

        if self.is_online:
            res = self._call_remote_llm(prompt)
            if res:
                return res

        # Fallback especialista determinístico baseado nas regras oficiais da FIK
        return self._expert_fallback_uncertain_strike(strike_type, target, fumi, posture, zanshin, hasuji, confidence)

    # --------------------------------------------------------------------------
    # 2. CAPTURA E ROTULAGEM INTELIGENTE DE MOVIMENTOS EM VÍDEOS E TREINOS
    # --------------------------------------------------------------------------
    def assisted_movement_labeling(
        self,
        kinematic_summary: Dict[str, Any],
        context_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Recebe métricas de movimento capturadas de um vídeo ou sessão de treino
        (velocidade de pulso, ângulos de ombro/cotovelo, desvio de coluna, cadência de pés)
        e gera automaticamente rótulos preliminares, detecção de erros e tags de treinamento.
        """
        prompt = (
            f"Você é um especialista em biomecânica e pedagogia do Kendo moderno.\n"
            f"Interprete as métricas cinemáticas capturadas na sessão de vídeo:\n"
            f"{json.dumps(kinematic_summary, ensure_ascii=False, indent=2)}\n"
            f"Contexto fornecido pelo usuário/treino: {context_hint or 'Keiko geral'}.\n\n"
            f"Retorne um JSON estruturado com:\n"
            f"{{\n"
            f'  "detected_modality": "<Ashi-sabaki | Suburi | Kirikaeshi | Nihon Kendo Kata | Men-uchi | Kote-uchi | Do-uchi | Tsuki | Shiai>",\n'
            f'  "confidence": <float 0.0 a 1.0>,\n'
            f'  "execution_quality": "EXCELENTE" | "BOA" | "REGULAR" | "PRECISA_CORRECAO",\n'
            f'  "detected_biomechanical_flaws": ["<lista de falhas posturais ou temporais identificadas>"],\n'
            f'  "action_spotting_tags": ["<tags temporais para rotulagem automática do dataset>"],\n'
            f'  "pedagogical_focus": "<orientação pedagógica para a próxima série de repetições>"\n'
            f"}}"
        )

        if self.is_online:
            res = self._call_remote_llm(prompt)
            if res:
                return res

        return self._expert_fallback_movement_labeling(kinematic_summary, context_hint)

    # --------------------------------------------------------------------------
    # 3. SÍNTESE E DESEMPATE DE DISPUTAS MULTI-ÁRBITRO
    # --------------------------------------------------------------------------
    def synthesize_dispute(
        self,
        reviews: List[Dict[str, Any]],
        strike_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analisa as opiniões divergentes de árbitros (ex: 2 declaram Ippon e 1 declara Não-Ponto)
        e produz uma síntese de consenso embasada no regulamento da FIK.
        """
        prompt = (
            f"Você é o Diretor de Arbitragem (Shinpan-cho) da Federação Internacional de Kendo (FIK).\n"
            f"O seguinte lance teve opiniões divergentes entre os árbitros:\n"
            f"Lance avaliado: {json.dumps(strike_summary, ensure_ascii=False)}\n"
            f"Avaliações dos árbitros:\n{json.dumps(reviews, ensure_ascii=False, indent=2)}\n\n"
            f"Consolide o parecer arbitral final em JSON:\n"
            f"{{\n"
            f'  "final_ruling": "IPPON" | "NO_POINT",\n'
            f'  "majority_decision": "<ex: 2 de 3 votos>",\n'
            f'  "synthesized_rationale": "<parecer fundamentado ponderando a posição e graduação dos árbitros>",\n'
            f'  "divergence_cause": "<motivo da divergência arbitral>",\n'
            f'  "training_dataset_action": "INCLUDE_HIGH_PRIORITY" | "INCLUDE_ATTENUATED" | "DISCARD"\n'
            f"}}"
        )

        if self.is_online:
            res = self._call_remote_llm(prompt)
            if res:
                return res

        return self._expert_fallback_dispute_synthesis(reviews, strike_summary)

    # --------------------------------------------------------------------------
    # MÉTODOS DE COMUNICAÇÃO REMOTA HTTP COM LLM (GEMINI E OPENAI COMPATÍVEIS)
    # --------------------------------------------------------------------------
    def _call_remote_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Realiza chamada HTTP REST com timeout seguro e extração estrita de JSON."""
        try:
            import requests

            if self.provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "responseMimeType": "application/json"
                    }
                }
                resp = requests.post(url, json=payload, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        return self._parse_json_response(text)

            elif self.provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Você é um assistente especialista em Kendo FIK. Retorne estritamente JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return self._parse_json_response(text)

        except Exception as e:
            log_event("WARNING", f"Falha na comunicação com API LLM remota ({self.provider}): {e}. Utilizando motor especialista offline.", "llm_assistant")

        return None

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extrai bloco JSON de texto retornado pelo LLM."""
        if not text:
            return None
        text = text.strip()
        # Remove blocos de markdown ```json ... ``` se presentes
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            # Tenta encontrar primeiro { e último }
            m = re.search(r"(\{.*\})", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass
        return None

    # --------------------------------------------------------------------------
    # MOTORES ESPECIALISTAS DETERMINÍSTICOS OFFLINE (REGRAS OFICIAIS FIK / AJKF)
    # --------------------------------------------------------------------------
    def _expert_fallback_uncertain_strike(
        self,
        strike_type: str,
        target: float,
        fumi: float,
        posture: float,
        zanshin: float,
        hasuji: float,
        confidence: float
    ) -> Dict[str, Any]:
        """Motor especialista de regras de Kendo para análise de incerteza sem necessidade de rede."""
        scores_map = {
            "HASUJI": hasuji,
            "KI_KEN_TAI_ICHI": fumi,
            "TARGET_IMPACT": target,
            "POSTURE": posture,
            "ZANSHIN": zanshin
        }
        # Identifica a maior deficiência (menor score)
        primary_def = min(scores_map.keys(), key=lambda k: scores_map[k])
        lowest_score = scores_map[primary_def]

        reasons = {
            "HASUJI": "O ângulo do Shinai divergiu do fio de corte da lâmina (desvio de Hasuji). Segundo o Regulamento FIK (Artigo 12), o golpe deve atingir o Datotsu-bui com o Monouchi e fio correto.",
            "KI_KEN_TAI_ICHI": "Houve dessincronização entre o impacto da lâmina e a finalização do Fumikomi-ashi (quebra do princípio Ki-Ken-Tai-Ichi).",
            "TARGET_IMPACT": "O impacto ocorreu de raspão ou fora da zona regulamentar de pontuação (Datotsu-bui).",
            "POSTURE": "A coluna apresentou inclinação excessiva no momento do golpe, comprometendo a postura digna (Shisei).",
            "ZANSHIN": "Não houve demonstração sustentada de prontidão mental e física (Zanshin) após o ataque."
        }

        coaching = {
            "HASUJI": "Trabalhe Suburi focando no Tenouchi (aperto elástico com os dedos mínimo e anelar) para travar a lâmina em ângulo reto no momento do impacto.",
            "KI_KEN_TAI_ICHI": "Pratique Uchikomi-geiko com ênfase no arranque da perna esquerda (Hiki-tsuke) simultâneo à descida do Shinai.",
            "TARGET_IMPACT": "Ajuste o Maai (distância) inicial: evite desferir o golpe em Chikama onde o Shinai atinge a base em vez do Monouchi.",
            "POSTURE": "Mantenha o olhar no oponente (Metsuke) e o tronco ereto sem projetar a cabeça para frente durante o Fumikomi.",
            "ZANSHIN": "Após o golpe, mantenha a pressão (Seme) passando pelo oponente e virando com Chudan imediato sem baixar os braços."
        }

        avg_score = (target + fumi + posture + zanshin + hasuji) / 5.0
        if avg_score >= 0.62 and lowest_score >= 0.45:
            verdict = "VALID_IPPON"
            rec_p = min(0.85, avg_score + 0.10)
        elif avg_score < 0.50 or lowest_score < 0.35:
            verdict = "NO_POINT"
            rec_p = max(0.20, avg_score - 0.10)
        else:
            verdict = "DOUBTFUL"
            rec_p = avg_score

        return {
            "verdict": verdict,
            "recommended_p_ippon": round(rec_p, 3),
            "primary_deficiency": primary_def,
            "fik_rule_rationale": reasons.get(primary_def, "Critérios limítrofes de Ki-Ken-Tai-Ichi e Zanshin."),
            "coaching_advice": coaching.get(primary_def, "Concentre-se na execução coordenada do golpe."),
            "engine": "FIK-Expert-Deterministic-Offline"
        }

    def _expert_fallback_movement_labeling(
        self,
        metrics: Dict[str, Any],
        context_hint: Optional[str]
    ) -> Dict[str, Any]:
        """Classificador cinemático de movimentos de Kendo para rotulagem automática."""
        cadence = metrics.get("cadence_cpm", 0.0)
        wrist_vel = metrics.get("peak_wrist_speed", 0.0)
        spine_tilt = metrics.get("spine_tilt_deg", 5.0)
        strike_count = metrics.get("strike_count", 0)

        # Inferência de modalidade
        if cadence > 40 and strike_count > 5:
            modality = "Kirikaeshi" if cadence > 50 else "Suburi"
        elif wrist_vel > 1.2:
            modality = "Men-uchi"
        elif "ashi" in (context_hint or "").lower():
            modality = "Ashi-sabaki"
        else:
            modality = "Keiko Geral (Shiai)"

        flaws = []
        if spine_tilt > 10.0:
            flaws.append("Inclinação anterior excessiva da coluna vertebral (> 10°)")
        if metrics.get("fumikomi_delay_ms", 0.0) > 80.0:
            flaws.append("Atraso acentuado entre impacto e contato de pé no solo (Fumikomi)")
        if metrics.get("hasuji_deviation_deg", 0.0) > 15.0:
            flaws.append("Desvio lateral do ângulo do Shinai (corte de raspão)")

        quality = "EXCELENTE" if not flaws else ("BOA" if len(flaws) == 1 else "PRECISA_CORRECAO")

        return {
            "detected_modality": modality,
            "confidence": 0.88,
            "execution_quality": quality,
            "detected_biomechanical_flaws": flaws or ["Nenhuma falha crítica detectada"],
            "action_spotting_tags": [f"action_{modality.lower().replace(' ', '_')}", f"quality_{quality.lower()}"],
            "pedagogical_focus": f"Foco no controle postural e estabilidade em {modality}.",
            "engine": "Kinematic-Heuristic-Classifier"
        }

    def _expert_fallback_dispute_synthesis(
        self,
        reviews: List[Dict[str, Any]],
        strike_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Síntese arbitral de maioria qualificada baseada em Dan."""
        ippon_votes = sum(1 for r in reviews if r.get("verdict") in [True, "valid", "TP", 1, "IPPON"])
        total_votes = len(reviews)
        maj_ippon = ippon_votes >= (total_votes / 2.0)

        return {
            "final_ruling": "IPPON" if maj_ippon else "NO_POINT",
            "majority_decision": f"{ippon_votes} de {total_votes} árbitros",
            "synthesized_rationale": (
                f"Pela maioria arbitral ({ippon_votes}/{total_votes}), a ação preencheu os requisitos de Yuko-Datotsu "
                f"conforme as diretrizes da FIK, considerando os impactos registrados no alvo."
                if maj_ippon else
                f"A maioria dos árbitros ({total_votes - ippon_votes}/{total_votes}) não considerou o golpe válido "
                f"devido a deficiências em Ki-Ken-Tai-Ichi ou ausência de Zanshin regulamentar."
            ),
            "divergence_cause": "Percepção distinta da simultaneidade do impacto em relação ao Fumikomi-ashi.",
            "training_dataset_action": "INCLUDE_ATTENUATED" if (0 < ippon_votes < total_votes) else "INCLUDE_HIGH_PRIORITY",
            "engine": "FIK-Arbitral-Consensus-Engine"
        }

    # --------------------------------------------------------------------------
    # 4. EXTRAÇÃO DE RESTRIÇÕES BIOMECÂNICAS EM JSON SCHEMA (EIXO 2.1)
    # --------------------------------------------------------------------------
    def extract_physical_constraints(
        self,
        source_text_or_metadata: Union[str, Dict[str, Any]],
        source_title: str = "Diretriz Regulamentar de Kendo"
    ) -> Dict[str, Any]:
        """
        Extrai restrições físicas numéricas estruturadas em JSON Schema (Eixo 2.1).
        Converte texto não-estruturado de manuais FIK/AJKF e literatura técnica em limites
        acionáveis para o calibrador (ex: elbow_extension_impact_deg, spine_tilt_max_deg,
        fumikomi_hand_foot_window_ms, zanshin_duration_min_sec).
        """
        raw_text = source_text_or_metadata if isinstance(source_text_or_metadata, str) else json.dumps(source_text_or_metadata, ensure_ascii=False)
        src_meta = source_text_or_metadata if isinstance(source_text_or_metadata, dict) else {"title": source_title, "summary": raw_text}

        prompt = (
            f"Você é um cientista do esporte e árbitro especialista em Kendo da FIK/AJKF.\n"
            f"Extraia os limites físicos e parâmetros biomecânicos exatos do seguinte texto regulamentar/técnico:\n"
            f"Fonte: {source_title}\n"
            f"Conteúdo: {raw_text[:1500]}\n\n"
            f"Retorne OBRIGATORIAMENTE um JSON válido exatamente neste formato:\n"
            f"{{\n"
            f'  "concept": "men_strike_biomechanics | kote_strike_biomechanics | do_strike_biomechanics | tsuki_thrust_biomechanics | tenouchi_hasuji_core",\n'
            f'  "source": "{source_title}",\n'
            f'  "constraints": {{\n'
            f'    "elbow_extension_impact_deg": {{"min": 150.0, "max": 175.0, "ideal": 165.0}},\n'
            f'    "spine_tilt_max_deg": 8.5,\n'
            f'    "fumikomi_hand_foot_window_ms": {{"min": -45.0, "max": 30.0}},\n'
            f'    "zanshin_duration_min_sec": 0.80,\n'
            f'    "hasuji_max_deviation_deg": 12.0,\n'
            f'    "blade_contact_zone": "monouchi"\n'
            f'  }}\n'
            f"}}"
        )

        if self.is_online:
            res = self._call_remote_llm(prompt)
            if res and isinstance(res, dict) and "constraints" in res:
                return res

        # Fallback especialista determinístico do Eixo 2
        try:
            from src.engine.actionable_research import PhysicalConstraintExtractor
            extractor = PhysicalConstraintExtractor()
            return extractor.extract_from_source(src_meta)
        except Exception:
            return {
                "concept": "general_kendo_biomechanics",
                "source": source_title,
                "authority_tier": 2,
                "authority_name": "AJKF Referee Handbook",
                "constraints": {
                    "spine_tilt_max_deg": 8.5,
                    "fumikomi_hand_foot_window_ms": {"min": -45.0, "max": 30.0, "ideal": 0.0},
                    "zanshin_duration_min_sec": 0.80,
                    "hasuji_max_deviation_deg": 12.0,
                    "blade_contact_zone": "monouchi"
                }
            }

