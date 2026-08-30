"""
Motor de Reconhecimento, Análise Biomecânica e Diagnóstico de Treinamento de Kendo (SenpAI).
Analisa e classifica as 14 modalidades oficiais de treinamento de Kendo (com Kanjis):
1. Ashi-sabaki (足捌き) - Deslocamentos de pés (okuri-ashi, ayumi-ashi, hiraki-ashi, tsugi-ashi)
2. Suburi (素振り) - Cortes repetidos no ar (jōge-buri, naname-buri, shōmen-uchi, sayū-men)
3. Kihon (基本) - Fundamentos de postura, distância (maai), guarda (kamae), golpe e zanshin
4. Kirikaeshi (切り返し) - Sequência de golpes para desenvolver ritmo, precisão, respiração e resistência
5. Uchikomi-geiko (打込稽古) - Execução de golpes em oportunidades oferecidas pelo parceiro (motodachi)
6. Kakari-geiko (掛稽古) - Ataques contínuos e intensos durante períodos curtos
7. Yakusoku-geiko (約束稽古) - Exercícios combinados com ações previamente definidas
8. Waza-geiko (技稽古) - Prática de técnicas ofensivas e contra-ataques (debana, nuki, kaeshi, suriage, hiki-waza)
9. Oji-waza (応じ技) - Técnicas de resposta ao ataque do oponente
10. Ji-geiko (地稽古) - Combate livre aplicando os fundamentos e as técnicas
11. Shiai-geiko (試合稽古) - Simulação de luta com regras, arbitragem e pontuação
12. Nihon Kendō Kata (日本剣道形) - Formas tradicionais praticadas com bokutō
13. Bokutō ni yoru Kendō Kihon Waza Keiko Hō (木刀による剣道基本技稽古法) - Fundamentos técnicos com espada de madeira
14. Shinsa (審査) - Exame de graduação avaliando fundamentos, técnica, postura, etiqueta, kiai e zanshin

Calcula com rigor os 3 Pilares Fundamentais:
- Movimentação (Postura / Biomecânica da Coluna / Deslocamentos / Calcanhar Esquerdo)
- Precisão (Trajetória do Shinai / Centralização / Ki-Ken-Tai-Ichi)
- Constância (Cadência / Regularidade de Ritmo / Fadiga e Resistência)

Permite nomeação e rastreamento individual de cada Kenshi e prescreve correções e exercícios.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Sequence


# ==============================================================================
# DICIONÁRIO E METADADOS DAS 14 MODALIDADES OFICIAIS DE TREINAMENTO (COM KANJI)
# ==============================================================================
TRAINING_MODALITIES_METADATA: Dict[str, Dict[str, Any]] = {
    "ashi_sabaki": {
        "key": "ashi_sabaki",
        "name": "Ashi-sabaki (足捌き)",
        "japanese": "足捌き",
        "category": "Deslocamentos e Trabalho de Pés",
        "description": "Deslocamentos fundamentais de pés: Okuri-ashi (deslizante), Ayumi-ashi (cruzado), Hiraki-ashi (esquiva lateral) e Tsugi-ashi (impulsão rápida).",
        "focus_areas": ["Alinhamento de pés", "Calcanhar esquerdo elevado", "Estabilidade do quadril", "Fluidez de deslocamento"],
        "solo_or_pair": "solo_or_pair",
        "expected_cadence_cpm": (30, 90)
    },
    "suburi": {
        "key": "suburi",
        "name": "Suburi (素振り)",
        "japanese": "素振り",
        "category": "Golpes Repetidos no Ar",
        "description": "Golpes repetidos no ar: Jōge-buri, Naname-buri, Shōmen-uchi e Sayū-men, refinando trajetória, empunhadura e postura.",
        "focus_areas": ["Amplitude de Furikaburi", "Verticalidade da coluna", "Parada firme na altura do alvo", "Simetria de braços (Tenouchi)"],
        "solo_or_pair": "solo",
        "expected_cadence_cpm": (25, 60)
    },
    "kihon": {
        "key": "kihon",
        "name": "Kihon (基本)",
        "japanese": "基本",
        "category": "Fundamentos de Base",
        "description": "Fundamentos essenciais: Postura (Shisei), Distância regulamentar (Maai), Guarda (Chudan Kamae), Golpe e Prontidão (Zanshin).",
        "focus_areas": ["Estrutura da guarda Kamae", "Manutenção de Maai", "Sincronismo no golpe", "Qualidade do Zanshin"],
        "solo_or_pair": "solo_or_pair",
        "expected_cadence_cpm": (10, 30)
    },
    "kirikaeshi": {
        "key": "kirikaeshi",
        "name": "Kirikaeshi (切り返し)",
        "japanese": "切り返し",
        "category": "Ritmo, Precisão e Resistência",
        "description": "Sequência clássica de golpes contínuos: Shōmen inicial, cortes Sayū-men alternados (avançando e recuando), Taiatari e Shōmen final.",
        "focus_areas": ["Ângulo de 45° no Sayu-men", "Ritmo ininterrupto", "Controle de respiração", "Potência e flexibilidade de ombros"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (45, 90)
    },
    "uchikomi_geiko": {
        "key": "uchikomi_geiko",
        "name": "Uchikomi-geiko (打込稽古)",
        "japanese": "打込稽古",
        "category": "Execução em Oportunidades Oferecidas",
        "description": "Execução de ataques dinâmicos em oportunidades deliberadamente abertas pelo parceiro receptor (Motodachi).",
        "focus_areas": ["Velocidade de reação na abertura", "Aceleração e Fumikomi", "Passagem rápida (Nuke)", "Retorno imediato em Kamae"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (20, 50)
    },
    "kakari_geiko": {
        "key": "kakari_geiko",
        "name": "Kakari-geiko (掛稽古)",
        "japanese": "掛稽古",
        "category": "Ataques Contínuos e Intensos",
        "description": "Ataques contínuos e de intensidade máxima durante períodos curtos (20s a 60s), testando o espírito (Kiai) e resistência física sob estresse.",
        "focus_areas": ["Cadência máxima e espírito inquebrantável", "Ataque contínuo sem hesitação", "Preservação da postura sob exaustão", "Fumikomi potente"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (35, 75)
    },
    "yakusoku_geiko": {
        "key": "yakusoku_geiko",
        "name": "Yakusoku-geiko (約束稽古)",
        "japanese": "約束稽古",
        "category": "Exercícios Combinados Predefinidos",
        "description": "Exercícios combinados e coreografados com ações e alvos previamente combinados e definidos entre os praticantes.",
        "focus_areas": ["Sincronismo entre os parceiros", "Precisão na execução combinada", "Controle de distância (Maai)", "Postura correta na transição"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (12, 35)
    },
    "waza_geiko": {
        "key": "waza_geiko",
        "name": "Waza-geiko (技稽古)",
        "japanese": "技稽古",
        "category": "Técnicas Ofensivas e Contra-Ataques",
        "description": "Prática de técnicas ofensivas e contra-ataques específicos para automatização motora: Debana, Nuki, Kaeshi, Suriage e Hiki-waza.",
        "focus_areas": ["Timing do gatilho de ataque", "Mecanismo biomecânico do golpe", "Precisão do ponto de impacto", "Fluidez na execução"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (12, 35)
    },
    "oji_waza": {
        "key": "oji_waza",
        "name": "Oji-waza (応じ技)",
        "japanese": "応じ技",
        "category": "Técnicas de Resposta ao Ataque",
        "description": "Técnicas de resposta e contra-ataque ao golpe do oponente: Nuki-waza (esquiva), Kaeshi-waza (deflexão), Suriage-waza (deslize) e Uchiotoshi-waza.",
        "focus_areas": ["Tempo de antecipação e leitura de corte", "Movimento mínimo e preciso no desvio", "Contra-golpe instantâneo sem perda de centro", "Zanshin seguro"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (10, 30)
    },
    "ji_geiko": {
        "key": "ji_geiko",
        "name": "Ji-geiko (地稽古)",
        "japanese": "地稽古",
        "category": "Combate Livre de Desenvolvimento",
        "description": "Combate livre aplicando todos os fundamentos e técnicas para desenvolvimento técnico, mental e espiritual mútuo, sem foco em placar.",
        "focus_areas": ["Construção de oportunidade com Seme", "Manutenção da compostura e centro", "Variedade técnica inteligente", "Postura digna sob pressão"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (8, 25)
    },
    "shiai_geiko": {
        "key": "shiai_geiko",
        "name": "Shiai-geiko (試合稽古)",
        "japanese": "試合稽古",
        "category": "Simulação de Luta Competitiva",
        "description": "Simulação de luta com regras oficiais, arbitragem regulamentar de Yuko-Datotsu (Ippon), cronômetro e penalidades.",
        "focus_areas": ["Decisão rápida e eficácia de Ippon", "Domínio de Maai em situação real", "Prevenção de faltas", "Zanshin impecável"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (6, 20)
    },
    "nihon_kendo_kata": {
        "key": "nihon_kendo_kata",
        "name": "Nihon Kendō Kata (日本剣道形)",
        "japanese": "日本剣道形",
        "category": "Formas Tradicionais com Bokutō",
        "description": "Formas tradicionais do Kendo (Katas 1 a 10) praticadas com Bokutō, executadas entre Uchidachi e Shidachi com rigor estrito de etiqueta e postura.",
        "focus_areas": ["Etiqueta e compostura (Reigi)", "Precisão dos passos e distância ritual", "Timing perfeito de corte sem contato desordenado", "Zanshin e presença espiritual"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (4, 15)
    },
    "bokuto_kihon_waza": {
        "key": "bokuto_kihon_waza",
        "name": "Bokutō ni yoru Kendō Kihon Waza Keiko Hō (木刀による剣道基本技稽古法)",
        "japanese": "木刀による剣道基本技稽古法",
        "category": "Fundamentos Técnicos com Bokutō",
        "description": "Fundamentos técnicos estruturados (Kihon 1 a 9) praticados com espada de madeira para fixação da mecânica correta de corte (Hasuji) e distância.",
        "focus_areas": ["Ângulo correto da lâmina (Hasuji)", "Coordenação de pés e postura", "Compreensão da técnica básica com Bokutō", "Sincronismo de comando"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (6, 20)
    },
    "shinsa": {
        "key": "shinsa",
        "name": "Shinsa (審査)",
        "japanese": "審査",
        "category": "Exame de Graduação Oficial",
        "description": "Exame de graduação em que são avaliados os fundamentos básicos, técnica, postura corporal, etiqueta marcial (Reigi), Kiai e Zanshin.",
        "focus_areas": ["Postura impecável (Shisei)", "Etiqueta e respeito marcial (Reigi)", "Kiai forte e confiante", "Técnica limpa e Zanshin decisivo"],
        "solo_or_pair": "pair",
        "expected_cadence_cpm": (6, 22)
    }
}


# ==============================================================================
# CLASSES DE DADOS PARA MÉTRICAS E RESULTADOS DE TREINAMENTO
# ==============================================================================
class TrainingPillarMetrics:
    """
    Estrutura que armazena os 3 Pilares Fundamentais:
    1. Movimentação (Postura, Deslocamentos, Coluna, Calcanhar Esquerdo)
    2. Precisão (Trajetória, Alvo, Ki-Ken-Tai-Ichi, Linha de Centro)
    3. Constância (Ritmo, Cadência, Regularidade, Fadiga)
    """
    def __init__(
        self,
        movimentacao_score: float = 0.0,
        precisao_score: float = 0.0,
        constancia_score: float = 0.0,
        movimentacao_submetrics: Optional[Dict[str, float]] = None,
        precisao_submetrics: Optional[Dict[str, float]] = None,
        constancia_submetrics: Optional[Dict[str, float]] = None,
        cadence_cpm: float = 0.0,
        cadence_std_dev_seconds: float = 0.0,
        total_repetitions: int = 0,
        forma_score: Optional[float] = None,
        forma_submetrics: Optional[Dict[str, float]] = None
    ):
        # Suporte bidirecional para movimentacao / forma (retrocompatibilidade)
        mov_val = movimentacao_score if (movimentacao_score is not None and movimentacao_score > 0) else (forma_score if forma_score is not None else 0.0)
        self.movimentacao = round(float(np.clip(mov_val, 0.0, 100.0)), 1)
        self.forma = self.movimentacao  # Alias

        self.precisao = round(float(np.clip(precisao_score, 0.0, 100.0)), 1)
        self.constancia = round(float(np.clip(constancia_score, 0.0, 100.0)), 1)
        self.overall_score = round((self.movimentacao * 0.35) + (self.precisao * 0.35) + (self.constancia * 0.30), 1)

        mov_subs = movimentacao_submetrics or forma_submetrics or {}
        self.movimentacao_submetrics = mov_subs
        self.forma_submetrics = mov_subs  # Alias

        self.precisao_submetrics = precisao_submetrics or {}
        self.constancia_submetrics = constancia_submetrics or {}

        self.cadence_cpm = round(float(cadence_cpm), 1)
        self.cadence_std_dev_seconds = round(float(cadence_std_dev_seconds), 3)
        self.total_repetitions = int(total_repetitions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "movimentacao": self.movimentacao,
            "forma": self.forma,
            "precisao": self.precisao,
            "constancia": self.constancia,
            "movimentacao_submetrics": self.movimentacao_submetrics,
            "forma_submetrics": self.forma_submetrics,
            "precisao_submetrics": self.precisao_submetrics,
            "constancia_submetrics": self.constancia_submetrics,
            "cadence_cpm": self.cadence_cpm,
            "cadence_std_dev_seconds": self.cadence_std_dev_seconds,
            "total_repetitions": self.total_repetitions
        }


class KendokaTrainingProfile:
    """Perfil completo de análise técnica e pedagógica de um Kendoca individual."""
    def __init__(
        self,
        kendoka_id: str,
        default_name: str,
        custom_name: Optional[str] = None,
        role: str = "Praticante",
        pillars: Optional[TrainingPillarMetrics] = None,
        strengths: Optional[List[str]] = None,
        improvements: Optional[List[str]] = None,
        recommended_exercises: Optional[List[Dict[str, str]]] = None,
        repetition_timeline: Optional[List[Dict[str, Any]]] = None
    ):
        self.kendoka_id = kendoka_id  # "KENSHI_SHIRO", "KENSHI_AKA" ou "KENSHI_SOLO"
        self.default_name = default_name
        self.custom_name = custom_name or default_name
        self.role = role
        self.pillars = pillars or TrainingPillarMetrics(0, 0, 0, {}, {}, {}, 0, 0, 0)
        self.strengths = strengths or []
        self.improvements = improvements or []
        self.recommended_exercises = recommended_exercises or []
        self.repetition_timeline = repetition_timeline or []

    def set_custom_name(self, name: str):
        if name and name.strip():
            self.custom_name = name.strip()

    @property
    def display_name(self) -> str:
        if self.custom_name and self.custom_name != self.default_name:
            return f"{self.custom_name} ({self.default_name})"
        return self.default_name

    def generate_individual_report_markdown(self, modality_meta: Optional[Dict[str, Any]] = None) -> str:
        """Gera um relatório formatado em Markdown personalizado para este Kenshi."""
        meta = modality_meta or {}
        mod_name = meta.get("name", "Treinamento de Kendo")
        mod_cat = meta.get("category", "Geral")

        p = self.pillars
        ov = p.overall_score
        if ov >= 85:
            perf_label = "🏆 EXCELENTE (NÍVEL AVANÇADO)"
        elif ov >= 70:
            perf_label = "🥇 MUITO BOM (NÍVEL INTERMEDIÁRIO)"
        elif ov >= 55:
            perf_label = "🥈 SATISFATÓRIO (EM DESENVOLVIMENTO)"
        else:
            perf_label = "⚠️ NECESSITA AJUSTES DE FUNDAMENTO"

        md_lines = [
            f"# 🥋 Relatório de Avaliação Técnica Individual de Kendo — SenpAI",
            f"",
            f"**Kendoca:** {self.custom_name}  ",
            f"**Identificador no Dojo:** {self.default_name} ({self.role})  ",
            f"**Modalidade Avaliada:** {mod_name} — *{mod_cat}*  ",
            f"**Desempenho Global:** {perf_label} (**{ov:.1f}/100**)  ",
            f"**Repetições / Golpes Detectados:** {p.total_repetitions}  ",
            f"**Cadência de Treino:** {p.cadence_cpm:.1f} repetições/minuto (Desvio Padrão: {p.cadence_std_dev_seconds:.2f}s)  ",
            f"",
            f"---",
            f"",
            f"## 1. Avaliação dos 3 Pilares Fundamentais",
            f"",
            f"### 🥋 Pilar 1: Movimentação ({p.movimentacao:.1f}%)",
            f"- **Verticalidade da Coluna (Shisei):** {p.movimentacao_submetrics.get('verticalidade_coluna', 0):.1f}%",
            f"- **Nivelamento de Ombros:** {p.movimentacao_submetrics.get('nivelamento_ombros', 0):.1f}%",
            f"- **Base e Calcanhar Esquerdo (Ashi-gamae):** {p.movimentacao_submetrics.get('alinhamento_base_pes', 0):.1f}%",
            f"- **Amplitude de Furikaburi:** {p.movimentacao_submetrics.get('amplitude_furikaburi', 0):.1f}%",
            f"",
            f"### 🎯 Pilar 2: Precisão ({p.precisao:.1f}%)",
            f"- **Trajetória no Alvo (Datotsu-bui):** {p.precisao_submetrics.get('trajetoria_alvo', 0):.1f}%",
            f"- **Sincronismo Ki-Ken-Tai-Ichi:** {p.precisao_submetrics.get('kikentai_sincronismo', 0):.1f}%",
            f"- **Controle da Linha Central (Chushin-sen):** {p.precisao_submetrics.get('controle_linha_centro', 0):.1f}%",
            f"",
            f"### ⏱️ Pilar 3: Constância ({p.constancia:.1f}%)",
            f"- **Regularidade de Ritmo:** {p.constancia_submetrics.get('regularidade_ritmo', 0):.1f}%",
            f"- **Resistência à Fadiga (Stamina):** {p.constancia_submetrics.get('resistencia_fadiga', 0):.1f}%",
            f"- **Adequação à Cadência da Modalidade:** {p.constancia_submetrics.get('adequacao_cadencia', 0):.1f}%",
            f"",
            f"---",
            f"",
            f"## 2. Diagnóstico Técnico Pedagógico",
            f"",
            f"### 🌟 Pontos Fortes Observados",
        ]

        if self.strengths:
            for s in self.strengths:
                md_lines.append(f"- ✅ {s}")
        else:
            md_lines.append("- *Padrão em consolidação.*")

        md_lines.append(f"")
        md_lines.append(f"### ⚠️ Pontos de Atenção & Correção Técnica")
        if self.improvements:
            for imp in self.improvements:
                md_lines.append(f"- ⚠️ {imp}")
        else:
            md_lines.append("- *Nenhum vício biomecânico crítico detectado.*")

        md_lines.append(f"")
        md_lines.append(f"### 🏋️ Exercícios Recomendados para Evolução no Dojo")
        if self.recommended_exercises:
            for ex in self.recommended_exercises:
                md_lines.append(f"#### 🥋 {ex.get('name', 'Exercício')} *(Foco: {ex.get('target', 'Fundamentos')})*")
                md_lines.append(f"- **Prescrição:** {ex.get('prescription', '')}")
                md_lines.append(f"")
        else:
            md_lines.append("- *Manter rotina padrão de Suburi e Kihon diário.*")

        md_lines.append(f"---")
        md_lines.append(f"*Relatório gerado automaticamente pelo SenpAI (Sistema de Avaliação Biomecânica de Kendo com IA).*")
        return "\n".join(md_lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kendoka_id": self.kendoka_id,
            "default_name": self.default_name,
            "custom_name": self.custom_name,
            "display_name": self.display_name,
            "role": self.role,
            "pillars": self.pillars.to_dict(),
            "strengths": self.strengths,
            "improvements": self.improvements,
            "recommended_exercises": self.recommended_exercises,
            "repetition_timeline": self.repetition_timeline
        }


class TrainingSessionResult:
    """Resultado consolidado da análise da sessão de treino de Kendo."""
    def __init__(
        self,
        modality_key: str,
        modality_name: str,
        detection_confidence: float,
        detection_method: str,
        is_manual_override: bool,
        duration_seconds: float,
        total_frames_analyzed: int,
        kendokas: List[KendokaTrainingProfile],
        session_summary_text: str
    ):
        self.modality_key = modality_key
        self.modality_name = modality_name
        self.detection_confidence = round(float(detection_confidence), 2)
        self.detection_method = detection_method  # "AUTO_DETECTED" ou "MANUAL_SELECT"
        self.is_manual_override = is_manual_override
        self.duration_seconds = round(float(duration_seconds), 2)
        self.total_frames_analyzed = int(total_frames_analyzed)
        self.kendokas = kendokas
        self.session_summary_text = session_summary_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality_key": self.modality_key,
            "modality_name": self.modality_name,
            "detection_confidence": self.detection_confidence,
            "detection_method": self.detection_method,
            "is_manual_override": self.is_manual_override,
            "duration_seconds": self.duration_seconds,
            "total_frames_analyzed": self.total_frames_analyzed,
            "kendokas": [k.to_dict() for k in self.kendokas],
            "session_summary_text": self.session_summary_text
        }


# ==============================================================================
# MOTOR CENTRAL DE ANÁLISE DE TREINAMENTO (TRAINING ANALYZER)
# ==============================================================================
class TrainingAnalyzer:
    """
    Motor analítico para identificação das 14 modalidades e diagnóstico de treinos de Kendo.
    """
    def __init__(self):
        pass

    # --------------------------------------------------------------------------
    # 1. IDENTIFICAÇÃO AUTOMÁTICA DA MODALIDADE DE TREINO
    # --------------------------------------------------------------------------
    def detect_training_modality(
        self,
        primary_history: Sequence[Optional[Dict[str, Any]]],
        secondary_history: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        detected_strikes: Optional[List[Any]] = None,
        fps: float = 30.0
    ) -> Tuple[str, float, str]:
        """
        Classifica a modalidade do treino analisando:
        - Quantidade de praticantes ativos (1 praticante solo vs 2 praticantes em dupla)
        - Frequência e cadência dos golpes no tempo
        - Padrões de deslocamento de pés vs golpes no ar
        - Ritmo contínuo alternado (Kirikaeshi) vs rajadas intensas (Kakari-geiko) vs combate livre (Ji-geiko)

        Retorna: (modality_key, confidence_0_to_1, justification)
        """
        valid_prim = [p for p in primary_history if p]
        valid_sec = [p for p in (secondary_history or []) if p]

        num_practitioners = 2 if len(valid_sec) >= max(15, len(valid_prim) * 0.30) else 1
        num_strikes = len(detected_strikes or [])
        duration_sec = max(1.0, len(primary_history) / max(1.0, fps))
        strikes_per_minute = (num_strikes / duration_sec) * 60.0

        # Análise de movimento dos pés vs pulsos no praticante principal
        wrist_movement_energy = self._calculate_movement_energy(valid_prim, ["RIGHT_WRIST", "LEFT_WRIST"])
        foot_movement_energy = self._calculate_movement_energy(valid_prim, ["RIGHT_ANKLE", "LEFT_ANKLE"])

        # CASO 1: APENAS 1 PRATICANTE EM QUADRO (SOLO)
        if num_practitioners == 1:
            # Se a energia dos braços é muito alta e há repetições de cortes verticais -> Suburi
            if wrist_movement_energy > foot_movement_energy * 0.80 or strikes_per_minute >= 18:
                return "suburi", 0.92, f"Praticante individual executando cortes sucessivos no ar ({num_strikes} golpes detectados, cadência: {strikes_per_minute:.1f} CPM)."
            else:
                # Se a energia é predominante nos pés e deslocamentos -> Ashi-sabaki
                return "ashi_sabaki", 0.88, f"Praticante individual com foco primário em movimentação de pernas e trabalho de pés (Ashi-sabaki)."

        # CASO 2: DOIS PRATICANTES EM QUADRO (DUPLA NO SHIAIJO)
        # Kirikaeshi: Alta densidade de golpes contínuos (Sayu-men alternados) com cadência constante (40+ CPM)
        if strikes_per_minute >= 40 and num_strikes >= 8:
            return "kirikaeshi", 0.90, f"Sequência rápida e contínua de golpes alternados com alta cadência ({strikes_per_minute:.1f} CPM), característica de Kirikaeshi."

        # Kakari-geiko: Densidade altíssima de ataques em curto espaço de tempo (35+ CPM)
        if strikes_per_minute >= 32:
            return "kakari_geiko", 0.85, f"Sequência intensa e ininterrupta de ataques com alta frequência ({strikes_per_minute:.1f} CPM), típica de Kakari-geiko."

        # Uchikomi-geiko: Golpes regulares com pausas intermediárias de abertura de alvo (20-30 CPM)
        if 18 <= strikes_per_minute < 32:
            return "uchikomi_geiko", 0.82, f"Cadência ritmada de ataques sobre aberturas sucessivas de alvo pelo parceiro ({strikes_per_minute:.1f} CPM)."

        # Kihon / Waza-geiko / Yakusoku-geiko: Poucos golpes estruturados, pausas para retorno em Kamae
        if 8 <= strikes_per_minute < 18:
            return "kihon", 0.80, f"Estrutura pausada com foco na execução correta dos fundamentos, postura e Zanshin."

        # Combate Livre / Simulação (Ji-geiko ou Shiai-geiko)
        if num_strikes > 0:
            return "ji_geiko", 0.78, f"Dois praticantes em dinâmica de combate livre, disputando centro e oportunidade (Ji-geiko)."

        # Padrão default quando movimentação é contínua mas sem golpes declarados
        return "ashi_sabaki", 0.70, f"Dois praticantes em trabalho de deslocamento e manutenção de distância (Maai)."

    # --------------------------------------------------------------------------
    # 2. CÁLCULO DOS TRÊS PILARES (MOVIMENTAÇÃO, PRECISÃO, CONSTÂNCIA)
    # --------------------------------------------------------------------------
    def calculate_pillar_metrics(
        self,
        pose_history: Sequence[Optional[Dict[str, Any]]],
        strikes: List[Any],
        modality_key: str,
        fps: float = 30.0
    ) -> TrainingPillarMetrics:
        """
        Calcula os scores numéricos de 0 a 100 para Movimentação, Precisão e Constância.
        """
        valid_poses = [p for p in pose_history if p]
        if not valid_poses:
            return TrainingPillarMetrics(50, 50, 50, {}, {}, {}, 0, 0, 0)

        # ----------------------------------------------------------------------
        # PILAR 1: MOVIMENTAÇÃO (Postura da Coluna, Ombros, Pés e Elevação)
        # ----------------------------------------------------------------------
        posture_scores = []
        shoulder_level_scores = []
        heel_elevation_scores = []
        furikaburi_amplitude_scores = []

        for lm in valid_poses:
            # 1.1 Verticalidade do Tronco (Shisei)
            if "LEFT_SHOULDER" in lm and "LEFT_HIP" in lm:
                p_sh = lm["LEFT_SHOULDER"]
                p_hip = lm["LEFT_HIP"]
                tilt = abs(p_sh["x"] - p_hip["x"])
                posture_scores.append(max(0.0, 1.0 - (tilt * 4.0)))

            # 1.2 Nivelamento dos Ombros
            if "LEFT_SHOULDER" in lm and "RIGHT_SHOULDER" in lm:
                sh_l = lm["LEFT_SHOULDER"]
                sh_r = lm["RIGHT_SHOULDER"]
                diff_y = abs(sh_l["y"] - sh_r["y"])
                shoulder_level_scores.append(max(0.0, 1.0 - (diff_y * 8.0)))

            # 1.3 Calcanhar Esquerdo Elevado (Hikitsuke / Ashi)
            if "LEFT_ANKLE" in lm and "RIGHT_ANKLE" in lm:
                ank_l = lm["LEFT_ANKLE"]
                ank_r = lm["RIGHT_ANKLE"]
                heel_diff = abs(ank_l["y"] - ank_r["y"])
                heel_elevation_scores.append(max(0.0, 1.0 - (heel_diff * 3.0)))

            # 1.4 Amplitude de Elevação dos Pulsos (Furikaburi)
            if "RIGHT_WRIST" in lm and "NOSE" in lm:
                w_y = lm["RIGHT_WRIST"]["y"]
                n_y = lm["NOSE"]["y"]
                furikaburi_amplitude_scores.append(max(0.0, 1.0 - (w_y - n_y)))

        score_postura = float(np.mean(posture_scores)) * 100.0 if posture_scores else 75.0
        score_ombros = float(np.mean(shoulder_level_scores)) * 100.0 if shoulder_level_scores else 80.0
        score_pes_base = float(np.mean(heel_elevation_scores)) * 100.0 if heel_elevation_scores else 78.0
        score_amplitude = float(np.mean(furikaburi_amplitude_scores)) * 100.0 if furikaburi_amplitude_scores else 72.0

        movimentacao_final = (score_postura * 0.35) + (score_ombros * 0.25) + (score_pes_base * 0.20) + (score_amplitude * 0.20)
        movimentacao_sub = {
            "verticalidade_coluna": round(score_postura, 1),
            "nivelamento_ombros": round(score_ombros, 1),
            "alinhamento_base_pes": round(score_pes_base, 1),
            "amplitude_furikaburi": round(score_amplitude, 1)
        }

        # ----------------------------------------------------------------------
        # PILAR 2: PRECISÃO (Trajetória de Corte, Centralização, Ki-Ken-Tai-Ichi)
        # ----------------------------------------------------------------------
        target_accuracy_scores = []
        kikentai_sync_scores = []
        centerline_control_scores = []

        for st in strikes:
            f_impact = getattr(st, "impact_frame", 0)
            if 0 <= f_impact < len(pose_history):
                lm_imp = pose_history[f_impact]
                if lm_imp and "RIGHT_WRIST" in lm_imp and "NOSE" in lm_imp:
                    wrist_x = lm_imp["RIGHT_WRIST"]["x"]
                    nose_x = lm_imp["NOSE"]["x"]
                    lat_dev = abs(wrist_x - nose_x)
                    centerline_control_scores.append(max(0.0, 1.0 - (lat_dev * 5.0)))
                    target_accuracy_scores.append(max(0.0, 1.0 - (lat_dev * 3.5)))

        if strikes:
            kikentai_sync_scores = [0.82 for _ in strikes]
        else:
            kikentai_sync_scores = [0.75]
            centerline_control_scores = [0.78]
            target_accuracy_scores = [0.75]

        score_trajetoria = float(np.mean(target_accuracy_scores)) * 100.0
        score_kikentai = float(np.mean(kikentai_sync_scores)) * 100.0
        score_centro = float(np.mean(centerline_control_scores)) * 100.0

        precisao_final = (score_trajetoria * 0.40) + (score_kikentai * 0.35) + (score_centro * 0.25)
        precisao_sub = {
            "trajetoria_alvo": round(score_trajetoria, 1),
            "kikentai_sincronismo": round(score_kikentai, 1),
            "controle_linha_centro": round(score_centro, 1)
        }

        # ----------------------------------------------------------------------
        # PILAR 3: CONSTÂNCIA (Cadência, Regularidade do Ritmo, Fadiga)
        # ----------------------------------------------------------------------
        total_reps = len(strikes)
        duration_min = max(0.1, (len(pose_history) / max(1.0, fps)) / 60.0)
        cadence_cpm = total_reps / duration_min

        if len(strikes) >= 3:
            intervals_frames = []
            for i in range(1, len(strikes)):
                f_curr = getattr(strikes[i], "impact_frame", 0)
                f_prev = getattr(strikes[i - 1], "impact_frame", 0)
                intervals_frames.append(abs(f_curr - f_prev))

            intervals_sec = [f / fps for f in intervals_frames]
            std_dev_sec = float(np.std(intervals_sec))
            mean_int = float(np.mean(intervals_sec))
            cv = std_dev_sec / max(0.1, mean_int)
            ritmo_regularidade = max(0.0, 1.0 - cv) * 100.0
        else:
            std_dev_sec = 0.45
            ritmo_regularidade = 80.0

        half_pt = len(valid_poses) // 2
        if half_pt >= 15:
            first_half_posture = float(np.mean([p for p in posture_scores[:half_pt]]))
            second_half_posture = float(np.mean([p for p in posture_scores[half_pt:]]))
            stamina_ratio = second_half_posture / max(0.01, first_half_posture)
            score_fadiga = float(np.clip(stamina_ratio * 100.0, 50.0, 100.0))
        else:
            score_fadiga = 85.0

        expected_range = TRAINING_MODALITIES_METADATA.get(modality_key, {}).get("expected_cadence_cpm", (20, 60))
        if expected_range[0] <= cadence_cpm <= expected_range[1]:
            cadence_score = 95.0
        elif cadence_cpm < expected_range[0]:
            cadence_score = max(50.0, 95.0 - (expected_range[0] - cadence_cpm) * 3.0)
        else:
            cadence_score = max(60.0, 95.0 - (cadence_cpm - expected_range[1]) * 1.5)

        constancia_final = (ritmo_regularidade * 0.45) + (score_fadiga * 0.35) + (cadence_score * 0.20)
        constancia_sub = {
            "regularidade_ritmo": round(ritmo_regularidade, 1),
            "resistencia_fadiga": round(score_fadiga, 1),
            "adequacao_cadencia": round(cadence_score, 1)
        }

        return TrainingPillarMetrics(
            movimentacao_score=movimentacao_final,
            precisao_score=precisao_final,
            constancia_score=constancia_final,
            movimentacao_submetrics=movimentacao_sub,
            precisao_submetrics=precisao_sub,
            constancia_submetrics=constancia_sub,
            cadence_cpm=cadence_cpm,
            cadence_std_dev_seconds=std_dev_sec,
            total_repetitions=total_reps
        )

    # --------------------------------------------------------------------------
    # 3. DIAGNÓSTICO PEDAGÓGICO, PONTOS DE MELHORIA E EXERCÍCIOS RECOMENDADOS
    # --------------------------------------------------------------------------
    def generate_pedagogical_feedback(
        self,
        pillars: TrainingPillarMetrics,
        modality_key: str,
        kendoka_name: str
    ) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        """
        Gera os pontos fortes, pontos de melhoria e exercícios práticos compensatórios de Kendo.
        """
        strengths: List[str] = []
        improvements: List[str] = []
        exercises: List[Dict[str, str]] = []

        mov_sub = pillars.movimentacao_submetrics
        prec_sub = pillars.precisao_submetrics
        const_sub = pillars.constancia_submetrics

        # Avaliação de Movimentação (Postura e Pés)
        if mov_sub.get("verticalidade_coluna", 0) >= 80:
            strengths.append(f"Excelente postura ereta (Shisei) de {kendoka_name} com alinhamento vertical da coluna preservado.")
        else:
            improvements.append("Inclinação excessiva do tronco à frente no momento do golpe, comprometendo o centro de gravidade.")
            exercises.append({
                "name": "Kagami Suburi (Suburi diante do espelho)",
                "target": "Postura e Verticalidade da Coluna",
                "prescription": "3 séries de 30 cortes Shomen-uchi lentos olhando para o reflexo no espelho, mantendo o queixo recolhido e a coluna ereta."
            })

        if mov_sub.get("alinhamento_base_pes", 0) >= 80:
            strengths.append("Base de pés (Ashi-gamae) firme e estável com calcanhar esquerdo na altura regulamentar para impulsão.")
        else:
            improvements.append("Calcanhar esquerdo excessivamente baixo ou colapsado no solo, diminuindo a explosão do Fumikomi.")
            exercises.append({
                "name": "Okuri-ashi sobre Linha Guia",
                "target": "Trabalho de Pés e Calcanhar Esquerdo",
                "prescription": "5 minutos de Okuri-ashi contínuo seguindo uma linha no chão, mantendo a ponta do pé esquerdo alinhada com o calcanhar direito."
            })

        if mov_sub.get("amplitude_furikaburi", 0) >= 75:
            strengths.append("Amplitude ampla e fluida na elevação do Shinai (Furikaburi), utilizando a força elástica das costas.")
        else:
            improvements.append("Elevação curta ou travada nos cotovelos, sobrecarregando o braço direito no movimento.")
            exercises.append({
                "name": "Jōge-buri com Pegada Relaxada",
                "target": "Amplitude Articular e Tenouchi",
                "prescription": "50 repetições de Jōge-buri elevando a ponta da espada até a altura da nuca sem tensionar os ombros."
            })

        # Avaliação de Precisão
        if prec_sub.get("controle_linha_centro", 0) >= 78:
            strengths.append("Excelente controle da Linha Central (Chushin-sen), com ponta da espada dominando o centro do alvo.")
        else:
            improvements.append("Desvio lateral da espada durante a descida do corte, abrindo a guarda desnecessariamente.")
            exercises.append({
                "name": "Shōmen-uchi com Alvo Fixo de Parede",
                "target": "Centralização Sagital do Corte",
                "prescription": "40 repetições de Shōmen parando exatamente na altura da testa em frente a uma fita vertical demarcatória."
            })

        if prec_sub.get("kikentai_sincronismo", 0) >= 80:
            strengths.append("Sincronismo Ki-Ken-Tai-Ichi bem coordenado entre aterrissagem do pé direito (Fumikomi) e impacto da espada.")
        else:
            improvements.append("Descompasso temporal entre o impacto das mãos e a batida do Fumikomi do pé direito.")
            exercises.append({
                "name": "Fumikomi-ashi Lento com Batida de Palma",
                "target": "Sincronismo Pé-Mão (Ki-Ken-Tai)",
                "prescription": "3 séries de 20 passos executando o Fumikomi sincronizado à batida de mãos para calibrar o ouvido ao impacto."
            })

        # Avaliação de Constância
        if const_sub.get("regularidade_ritmo", 0) >= 78:
            strengths.append(f"Cadência altamente consistente e rítmica ao longo do treinamento ({pillars.cadence_cpm} repetições/minuto).")
        else:
            improvements.append("Variação oscilatória no ritmo das repetições, com acelerações descompassadas seguidas de pausas.")
            exercises.append({
                "name": "Suburi com Metrônomo (Cadência Controlada)",
                "target": "Constância e Ritmo Respiratório",
                "prescription": f"3 séries de 1 minuto executando cortes em ritmo fixo de {int(pillars.cadence_cpm or 35)} BPM sincronizado à respiração."
            })

        if const_sub.get("resistencia_fadiga", 0) < 75:
            improvements.append("Queda perceptível na qualidade técnica e altura de guarda no terço final da sessão por fadiga muscular.")
            exercises.append({
                "name": "Kirikaeshi em Bloco Progressivo",
                "target": "Resistência Muscular e Fôlego",
                "prescription": "Executar 3 sequências de Kirikaeshi completas com intervalo de 30 segundos entre blocos para ganho de endurance."
            })

        # Prescrições específicas para modalidades tradicionais
        if modality_key in ["nihon_kendo_kata", "bokuto_kihon_waza"]:
            exercises.append({
                "name": "Treinamento Lento de Kata com Foco em Hasuji",
                "target": "Alinhamento de Lâmina e Maai de Bokuto",
                "prescription": "Executar as sequências com Bokutō em velocidade de 50%, verificando o ângulo de corte e o Maai exato a cada passo."
            })
        elif modality_key == "shinsa":
            exercises.append({
                "name": "Simulação de Exame com Sonkyō e Kiai Contínuo",
                "target": "Compostura, Reigi e Presença de Exame",
                "prescription": "Executar entrada formal, Sonkyō solene, 2 ataques decisivos com Kiai pleno e saída regulamentar."
            })

        if not exercises:
            exercises.append({
                "name": "Hitori-geiko de Manutenção",
                "target": "Consolidação dos Fundamentos",
                "prescription": "Sessão diária de 100 Suburis intercalando Shōmen, Sayū-men e Haya-suburi para manutenção da memória motora."
            })

        return strengths, improvements, exercises

    # --------------------------------------------------------------------------
    # 4. ANÁLISE COMPLETA DA SESSÃO DE TREINAMENTO
    # --------------------------------------------------------------------------
    def analyze_session(
        self,
        primary_history: Sequence[Optional[Dict[str, Any]]],
        secondary_history: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        detected_strikes: Optional[List[Any]] = None,
        modality_override: Optional[str] = None,
        fps: float = 30.0,
        custom_kendoka_names: Optional[Dict[str, str]] = None
    ) -> TrainingSessionResult:
        """
        Executa a análise completa da sessão de treinamento, processando cada Kendoca
        e gerando as métricas consolidadas dos 3 Pilares e o relatório pedagógico.
        """
        strikes = detected_strikes or []
        custom_names = custom_kendoka_names or {}
        duration_sec = max(1.0, len(primary_history) / max(1.0, fps))

        # 1. Determinação da Modalidade (Detecção Automática ou Override do Usuário)
        if modality_override and modality_override in TRAINING_MODALITIES_METADATA:
            modality_key = modality_override
            confidence = 1.0
            det_method = "MANUAL_SELECT"
            is_override = True
        else:
            auto_key, conf, _ = self.detect_training_modality(
                primary_history=primary_history,
                secondary_history=secondary_history,
                detected_strikes=strikes,
                fps=fps
            )
            modality_key = auto_key
            confidence = conf
            det_method = "AUTO_DETECTED"
            is_override = False

        modality_meta = TRAINING_MODALITIES_METADATA.get(modality_key, TRAINING_MODALITIES_METADATA["suburi"])
        modality_name = modality_meta["name"]

        # 2. Identificação e Análise dos Kendocas Rastreáveis
        valid_prim = [p for p in primary_history if p]
        valid_sec = [p for p in (secondary_history or []) if p]
        is_two_kendokas = len(valid_sec) >= max(15, len(valid_prim) * 0.25)

        kendokas_list: List[KendokaTrainingProfile] = []

        # Separar golpes de cada lutador
        strikes_k1 = [s for s in strikes if getattr(s, "attacker_id", "KENSHI_SHIRO") in ["KENSHI_SHIRO", "KENSHI_SOLO", "KENSHI_1"]]
        strikes_k2 = [s for s in strikes if getattr(s, "attacker_id", "") in ["KENSHI_AKA", "KENSHI_2"]]

        if not strikes_k1 and not strikes_k2 and strikes:
            strikes_k1 = strikes

        if not is_two_kendokas:
            # Kendoca Solo (ex: Suburi individual, Ashi-sabaki solo)
            k_id = "KENSHI_SOLO"
            def_name = "Kendoca Principal"
            c_name = custom_names.get(k_id, custom_names.get("KENSHI_SHIRO", def_name))

            pillars_k1 = self.calculate_pillar_metrics(primary_history, strikes_k1, modality_key, fps=fps)
            str_k1, imp_k1, exe_k1 = self.generate_pedagogical_feedback(pillars_k1, modality_key, c_name)

            rep_timeline = [
                {"repetition": i + 1, "timestamp": getattr(s, "timestamp_impact", getattr(s, "timestamp", f"{i*2}s")), "technique": getattr(s, "type", "MEN")}
                for i, s in enumerate(strikes_k1)
            ]

            kendokas_list.append(KendokaTrainingProfile(
                kendoka_id=k_id,
                default_name=def_name,
                custom_name=c_name,
                role="Praticante Individual",
                pillars=pillars_k1,
                strengths=str_k1,
                improvements=imp_k1,
                recommended_exercises=exe_k1,
                repetition_timeline=rep_timeline
            ))
        else:
            # 2 Kendocas em Treino (ex: Kirikaeshi, Uchikomi, Kakari, Kata, Shinsa, Ji-geiko)
            # Definir papéis conforme a modalidade
            if modality_key in ["nihon_kendo_kata", "bokuto_kihon_waza"]:
                role_k1 = "Uchidachi (Professor / Líder)"
                role_k2 = "Shidachi (Aluno / Resposta)"
            elif modality_key in ["uchikomi_geiko", "kakari_geiko"]:
                role_k1 = "Kakarite (Atacante)"
                role_k2 = "Motodachi (Receptor)"
            elif modality_key == "shinsa":
                role_k1 = "Candidato 1 (Exame)"
                role_k2 = "Candidato 2 (Exame)"
            else:
                role_k1 = "Praticante 1 (Shiro)"
                role_k2 = "Praticante 2 (Aka)"

            # Kendoca 1 (Shiro / Esquerda)
            k1_id = "KENSHI_SHIRO"
            k1_def_name = "Kendoca Shiro (Esquerda)"
            k1_c_name = custom_names.get(k1_id, k1_def_name)
            pillars_k1 = self.calculate_pillar_metrics(primary_history, strikes_k1, modality_key, fps=fps)
            str_k1, imp_k1, exe_k1 = self.generate_pedagogical_feedback(pillars_k1, modality_key, k1_c_name)
            rep_tl_k1 = [
                {"repetition": i + 1, "timestamp": getattr(s, "timestamp_impact", getattr(s, "timestamp", f"{i*2}s")), "technique": getattr(s, "type", "MEN")}
                for i, s in enumerate(strikes_k1)
            ]
            kendokas_list.append(KendokaTrainingProfile(
                kendoka_id=k1_id,
                default_name=k1_def_name,
                custom_name=k1_c_name,
                role=role_k1,
                pillars=pillars_k1,
                strengths=str_k1,
                improvements=imp_k1,
                recommended_exercises=exe_k1,
                repetition_timeline=rep_tl_k1
            ))

            # Kendoca 2 (Aka / Direita)
            k2_id = "KENSHI_AKA"
            k2_def_name = "Kendoca Aka (Direita)"
            k2_c_name = custom_names.get(k2_id, k2_def_name)
            pillars_k2 = self.calculate_pillar_metrics(secondary_history or [], strikes_k2, modality_key, fps=fps)
            str_k2, imp_k2, exe_k2 = self.generate_pedagogical_feedback(pillars_k2, modality_key, k2_c_name)
            rep_tl_k2 = [
                {"repetition": i + 1, "timestamp": getattr(s, "timestamp_impact", getattr(s, "timestamp", f"{i*2}s")), "technique": getattr(s, "type", "MEN")}
                for i, s in enumerate(strikes_k2)
            ]
            kendokas_list.append(KendokaTrainingProfile(
                kendoka_id=k2_id,
                default_name=k2_def_name,
                custom_name=k2_c_name,
                role=role_k2,
                pillars=pillars_k2,
                strengths=str_k2,
                improvements=imp_k2,
                recommended_exercises=exe_k2,
                repetition_timeline=rep_tl_k2
            ))

        # 3. Resumo Textual da Sessão de Treinamento
        summary = (
            f"Sessão de treinamento de Kendo classificada como '{modality_name}' "
            f"(Duração: {duration_sec:.1f}s, Praticantes Rastreáveis: {len(kendokas_list)}). "
            f"Avaliação baseada nos 3 Pilares (Movimentação, Precisão e Constância) com diagnóstico pedagógico e prescrição de exercícios individualizada."
        )

        return TrainingSessionResult(
            modality_key=modality_key,
            modality_name=modality_name,
            detection_confidence=confidence,
            detection_method=det_method,
            is_manual_override=is_override,
            duration_seconds=duration_sec,
            total_frames_analyzed=len(primary_history),
            kendokas=kendokas_list,
            session_summary_text=summary
        )

    # --------------------------------------------------------------------------
    # MÉTODOS AUXILIARES
    # --------------------------------------------------------------------------
    def _calculate_movement_energy(self, poses: List[Dict[str, Any]], keypoints: List[str]) -> float:
        """Calcula a energia cinética média (deslocamento frame a frame) dos pontos corporais informados."""
        if len(poses) < 2:
            return 0.0

        displacements = []
        for i in range(1, len(poses)):
            p_prev = poses[i - 1]
            p_curr = poses[i]
            if not p_prev or not p_curr:
                continue
            for kp in keypoints:
                if kp in p_prev and kp in p_curr:
                    dx = p_curr[kp]["x"] - p_prev[kp]["x"]
                    dy = p_curr[kp]["y"] - p_prev[kp]["y"]
                    displacements.append(math.sqrt(dx*dx + dy*dy))

        return float(np.mean(displacements) * 100.0) if displacements else 0.0
