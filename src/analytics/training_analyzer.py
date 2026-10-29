"""
Motor de Reconhecimento, Análise Biomecânica e Diagnóstico de Treinamento de Kendo (SenpAI).
Analisa e classifica as 14 modalidades oficiais de treinamento de Kendo (com Kanjis):
1. Ashi-sabaki (足捌き) - Deslocamentos de pés (okuri-ashi, ayumi-ashi, hiraki-ashi, tsugi-ashi)
2. Suburi (素振り) - Cortes repetidos no ar (jōge-buri, naname-buri, shōmen-uchi, sayū-men)
3. Kihon (基本) - Fundamentos de postura, distância (maai), guarda (kamae), golpe e zanshin
4. Kirikaeshi (切り返し) - Sequência clássica de golpes contínuos
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

import re
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
        "category": "Sequência clássica de golpes contínuos",
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
# FUNÇÕES UTILITÁRIAS DE TEMPO (MM:SS) PARA SESSÕES E BLOCOS DE MODALIDADE
# ==============================================================================
def format_training_mmss(seconds: float) -> str:
    """Formata segundos em string MM:SS (ou HH:MM:SS se >= 3600s)."""
    sec = max(0, round(float(seconds or 0.0)))
    hrs = sec // 3600
    mins = (sec % 3600) // 60
    rem = sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{rem:02d}"
    return f"{mins:02d}:{rem:02d}"


def parse_training_mmss(value: Any, default: float = 0.0) -> float:
    """Converte string 'MM:SS', 'HH:MM:SS', '12.5s' ou número em segundos (float)."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    raw = str(value).strip()
    if not raw:
        return default
    if ":" in raw:
        parts = [p.strip() for p in raw.split(":")]
        try:
            if len(parts) == 2:
                return max(0.0, int(parts[0]) * 60.0 + float(parts[1]))
            elif len(parts) == 3:
                return max(0.0, int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2]))
        except Exception:
            return default
    clean = re.sub(r"[^\d.]", "", raw)
    try:
        return max(0.0, float(clean)) if clean else default
    except Exception:
        return default


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
        self.movement_score: float = self.movimentacao  # Alias em inglês

        self.precisao = round(float(np.clip(precisao_score, 0.0, 100.0)), 1)
        self.precision_score: float = self.precisao  # Alias em inglês

        self.constancia = round(float(np.clip(constancia_score, 0.0, 100.0)), 1)
        self.constancy_score: float = self.constancia  # Alias em inglês

        self.overall_score = round((self.movimentacao * 0.35) + (self.precisao * 0.35) + (self.constancia * 0.30), 1)

        mov_subs = movimentacao_submetrics or forma_submetrics or {}
        self.movimentacao_submetrics = mov_subs
        self.forma_submetrics = mov_subs  # Alias

        self.precisao_submetrics = precisao_submetrics or {}
        self.constancia_submetrics = constancia_submetrics or {}

        self.cadence_cpm = round(float(cadence_cpm), 1)
        self.cadence_std_dev_seconds = round(float(cadence_std_dev_seconds), 3)
        self.total_repetitions = total_repetitions

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
        repetition_timeline: Optional[List[Dict[str, Any]]] = None,
        kinesthetic_deviations: Optional[Dict[str, Any]] = None,
        kinesthetic_insights: Optional[List[str]] = None,
        presence_frames: int = 0,
        presence_seconds: float = 0.0,
        presence_intervals: Optional[List[Tuple[int, int]]] = None,
        reentry_count: int = 0,
        thumbnail: Optional[Any] = None,
        modality_breakdown: Optional[List[Dict[str, Any]]] = None
    ):
        self.kendoka_id = kendoka_id  # "KENSHI_01", "KENSHI_02", "KENSHI_SHIRO", etc.
        self.default_name = default_name
        self.custom_name = custom_name or default_name
        self.role = role
        self.pillars = pillars or TrainingPillarMetrics(0, 0, 0, {}, {}, {}, 0, 0, 0)
        self.strengths = strengths or []
        self.improvements = improvements or []
        self.recommended_exercises = recommended_exercises or []
        self.repetition_timeline = repetition_timeline or []
        self.kinesthetic_deviations = kinesthetic_deviations or {}
        self.kinesthetic_insights = kinesthetic_insights or []
        self.presence_frames = presence_frames
        self.presence_seconds = round(float(presence_seconds), 1)
        self.presence_intervals = presence_intervals or []
        self.reentry_count = reentry_count
        self.thumbnail = thumbnail
        self.modality_breakdown = modality_breakdown or []

    def set_custom_name(self, name: str):
        if name and name.strip():
            self.custom_name = name.strip()

    @property
    def display_name(self) -> str:
        if self.custom_name and self.custom_name != self.default_name:
            return f"{self.custom_name} ({self.default_name})"
        return self.default_name

    def generate_individual_report_markdown(
        self,
        modality_meta: Optional[Dict[str, Any]] = None,
        modality_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Gera um relatório formatado em Markdown personalizado para este Kenshi, incluindo blocos por modalidade."""
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

        # Extrair blocos de modalidade para este praticante (de modality_blocks ou self.modality_breakdown)
        kendoka_blocks: List[Dict[str, Any]] = []
        if modality_blocks:
            for mb in modality_blocks:
                mb_dict = mb.to_dict() if hasattr(mb, "to_dict") else (mb if isinstance(mb, dict) else {})
                for k_item in mb_dict.get("kendokas", []):
                    if (
                        k_item.get("kendoka_id") == self.kendoka_id
                        or k_item.get("custom_name") == self.custom_name
                    ):
                        kendoka_blocks.append({
                            "modality_key": mb_dict.get("modality_key", ""),
                            "modality_name": mb_dict.get("modality_name", ""),
                            "modality_category": mb_dict.get("modality_category", ""),
                            "intervals_label": mb_dict.get("intervals_label", ""),
                            "total_duration_seconds": mb_dict.get("total_duration_seconds", 0.0),
                            "kendoka_profile": k_item
                        })
                        break
        elif self.modality_breakdown:
            kendoka_blocks = list(self.modality_breakdown)

        md_lines = [
            f"# 🥋 Relatório de Avaliação Técnica Individual de Kendo — SenpAI",
            f"",
            f"**Kendoca:** {self.custom_name}  ",
            f"**Identificador no Dojo:** {self.default_name} ({self.role})  ",
            f"**Modalidade Predominante Avaliada:** {mod_name} — *{mod_cat}*  ",
            f"**Desempenho Global:** {perf_label} (**{ov:.1f}/100**)  ",
            f"**Repetições / Golpes Detectados:** {p.total_repetitions}  ",
            f"**Cadência de Treino:** {p.cadence_cpm:.1f} repetições/minuto (Desvio Padrão: {p.cadence_std_dev_seconds:.2f}s)  ",
            f"",
            f"---",
            f"",
            f"## 1. Avaliação Consolidada dos 3 Pilares Fundamentais",
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
            f"## 2. Diagnóstico Técnico Pedagógico Geral",
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

        # Seção dedicada separando por bloco de modalidade de treinamento
        if kendoka_blocks:
            md_lines.append(f"---")
            md_lines.append(f"")
            md_lines.append(f"## 3. Avaliação Separada por Bloco de Modalidade de Treinamento")
            md_lines.append(f"")
            for b_idx, kb in enumerate(kendoka_blocks, start=1):
                b_mod_name = kb.get("modality_name", "Modalidade")
                b_mod_cat = kb.get("modality_category", "Geral")
                b_iv_label = kb.get("intervals_label", "")
                b_dur = float(kb.get("total_duration_seconds", 0.0))
                k_prof = kb.get("kendoka_profile", kb)
                k_pil = k_prof.get("pillars", {})
                b_ov = float(k_pil.get("overall_score", 0.0))
                b_mov = float(k_pil.get("movimentacao", k_pil.get("forma", 0.0)))
                b_prec = float(k_pil.get("precisao", 0.0))
                b_const = float(k_pil.get("constancia", 0.0))
                b_reps = int(k_pil.get("total_repetitions", 0))
                b_cpm = float(k_pil.get("cadence_cpm", 0.0))

                md_lines.append(f"### 📦 Bloco de Modalidade {b_idx}: {b_mod_name} (*{b_mod_cat}*)")
                if b_iv_label:
                    md_lines.append(f"- **Sessões / Intervalos Concentrados:** `{b_iv_label}` (Duração acumulada: {b_dur:.1f}s)")
                md_lines.append(
                    f"- **Métricas no Bloco:** Score Geral **{b_ov:.1f}/100** | "
                    f"Movimentação: **{b_mov:.1f}%** | Precisão: **{b_prec:.1f}%** | Constância: **{b_const:.1f}%** | "
                    f"Repetições: **{b_reps}** ({b_cpm:.1f} CPM)"
                )
                b_str = k_prof.get("strengths", [])
                if b_str:
                    md_lines.append(f"- **Destaques na Modalidade:** " + "; ".join(b_str[:3]))
                b_imp = k_prof.get("improvements", [])
                if b_imp:
                    md_lines.append(f"- **Ajustes na Modalidade:** " + "; ".join(b_imp[:3]))
                b_ex = k_prof.get("recommended_exercises", [])
                if b_ex:
                    ex_names = [e.get("name", "") for e in b_ex if isinstance(e, dict) and e.get("name")]
                    if ex_names:
                        md_lines.append(f"- **Drills Específicos:** " + ", ".join(ex_names[:3]))
                md_lines.append(f"")

        # Modelagem do Estilo Individual & Baseline Cinestésico (Eixo 6)
        if self.kinesthetic_insights or self.kinesthetic_deviations:
            md_lines.append(f"---")
            md_lines.append(f"")
            md_lines.append(f"## 4. Análise Comparativa com o Baseline Cinestésico Individual (Eixo 6)")
            md_lines.append(f"")
            md_lines.append(f"O SenpAI compara a execução atual com o estilo e padrão habitual histórico do praticante:")
            if self.kinesthetic_insights:
                for ins in self.kinesthetic_insights:
                    md_lines.append(f"- 🧬 {ins}")
            else:
                md_lines.append(f"- 🧬 *Padrão biomecânico perfeitamente alinhado com o baseline histórico do praticante.*")

        md_lines.append(f"")
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
            "repetition_timeline": self.repetition_timeline,
            "kinesthetic_deviations": self.kinesthetic_deviations,
            "kinesthetic_insights": self.kinesthetic_insights,
            "presence_frames": self.presence_frames,
            "presence_seconds": self.presence_seconds,
            "presence_intervals": self.presence_intervals,
            "reentry_count": self.reentry_count,
            "modality_breakdown": self.modality_breakdown
        }


class TrainingSubSession:
    """
    Representa uma sessão (segmento temporal cronológico) identificada dentro da mesma sequência de treinamento.
    Exemplo: 00:00 a 03:00 -> Kirikaeshi; 03:00 a 05:00 -> Waza-geiko.
    """
    def __init__(
        self,
        segment_index: int,
        start_frame: int,
        end_frame: int,
        start_seconds: float,
        end_seconds: float,
        modality_key: str,
        modality_name: str,
        modality_category: str = "",
        detection_confidence: float = 0.85,
        detection_method: str = "AUTO_DETECTED",
        justification: str = "",
        total_repetitions: int = 0,
        kendokas: Optional[List[Any]] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None
    ):
        self.segment_index = segment_index
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_seconds = round(float(start_seconds), 2)
        self.end_seconds = round(float(end_seconds), 2)
        self.start_timestamp = start_timestamp or format_training_mmss(self.start_seconds)
        self.end_timestamp = end_timestamp or format_training_mmss(self.end_seconds)
        self.duration_seconds = round(max(0.1, self.end_seconds - self.start_seconds), 2)
        self.modality_key = modality_key
        self.modality_name = modality_name
        self.modality_category = modality_category
        self.detection_confidence = round(float(detection_confidence), 2)
        self.detection_method = detection_method
        self.justification = justification
        self.total_repetitions = total_repetitions
        self.kendokas = kendokas or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_seconds": self.duration_seconds,
            "modality_key": self.modality_key,
            "modality_name": self.modality_name,
            "modality_category": self.modality_category,
            "detection_confidence": self.detection_confidence,
            "detection_method": self.detection_method,
            "justification": self.justification,
            "total_repetitions": self.total_repetitions,
            "kendokas": [
                k.to_dict() if hasattr(k, "to_dict") else k
                for k in self.kendokas
            ]
        }


class TrainingModalityBlock:
    """
    Concentra todas as sessões de uma mesma modalidade de treinamento identificadas na sequência,
    consolidando intervalos temporais, repetições, fundamentação técnica e métricas dos 3 Pilares por Kendoca.
    """
    def __init__(
        self,
        block_index: int,
        modality_key: str,
        modality_name: str,
        modality_category: str = "",
        description: str = "",
        focus_areas: Optional[List[str]] = None,
        learned_principles: Optional[List[str]] = None,
        web_sources: Optional[List[Dict[str, str]]] = None,
        intervals: Optional[List[Dict[str, Any]]] = None,
        intervals_label: str = "",
        total_duration_seconds: float = 0.0,
        total_repetitions: int = 0,
        detection_confidence: float = 0.85,
        detection_method: str = "AUTO_DETECTED",
        justification: str = "",
        kendokas: Optional[List[Any]] = None
    ):
        self.block_index = block_index
        self.modality_key = modality_key
        self.modality_name = modality_name
        self.modality_category = modality_category
        self.description = description
        self.focus_areas = focus_areas or []
        self.learned_principles = learned_principles or []
        self.web_sources = web_sources or []
        self.intervals = intervals or []
        if not intervals_label and self.intervals:
            intervals_label = ", ".join(
                f"{iv.get('start_timestamp', '00:00')} a {iv.get('end_timestamp', '00:00')}"
                for iv in self.intervals
            )
        self.intervals_label = intervals_label
        self.total_duration_seconds = round(float(total_duration_seconds), 2)
        self.total_repetitions = total_repetitions
        self.detection_confidence = round(float(detection_confidence), 2)
        self.detection_method = detection_method
        self.justification = justification
        self.kendokas = kendokas or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_index": self.block_index,
            "modality_key": self.modality_key,
            "modality_name": self.modality_name,
            "modality_category": self.modality_category,
            "description": self.description,
            "focus_areas": self.focus_areas,
            "learned_principles": self.learned_principles,
            "web_sources": self.web_sources,
            "intervals": self.intervals,
            "intervals_label": self.intervals_label,
            "session_count": len(self.intervals),
            "total_duration_seconds": self.total_duration_seconds,
            "total_repetitions": self.total_repetitions,
            "detection_confidence": self.detection_confidence,
            "detection_method": self.detection_method,
            "justification": self.justification,
            "kendokas": [
                k.to_dict() if hasattr(k, "to_dict") else k
                for k in self.kendokas
            ]
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
        session_summary_text: str,
        justification: str = "",
        learned_principles: Optional[List[str]] = None,
        modality_category: str = "",
        web_sources: Optional[List[Dict[str, str]]] = None,
        sub_sessions: Optional[List[Any]] = None,
        modality_blocks: Optional[List[Any]] = None
    ):
        self.modality_key = modality_key
        self.modality_name = modality_name
        self.detection_confidence = round(float(detection_confidence), 2)
        self.detection_method = detection_method  # "AUTO_DETECTED" ou "MANUAL_SELECT"
        self.is_manual_override = is_manual_override
        self.duration_seconds = round(float(duration_seconds), 2)
        self.total_frames_analyzed = total_frames_analyzed
        self.kendokas = kendokas
        self.session_summary_text = session_summary_text
        self.justification = justification
        self.learned_principles = learned_principles or []
        self.modality_category = modality_category
        self.web_sources = web_sources or []
        self.sub_sessions = sub_sessions or []
        self.modality_blocks = modality_blocks or []

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
            "session_summary_text": self.session_summary_text,
            "justification": self.justification,
            "learned_principles": self.learned_principles,
            "modality_category": self.modality_category,
            "web_sources": self.web_sources,
            "sub_sessions": [
                s.to_dict() if hasattr(s, "to_dict") else s
                for s in self.sub_sessions
            ],
            "modality_blocks": [
                b.to_dict() if hasattr(b, "to_dict") else b
                for b in self.modality_blocks
            ]
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
        fps: float = 30.0,
        has_sonkyo: bool = False
    ) -> Tuple[str, float, str]:
        """
        Classifica a modalidade do treino analisando:
        - Quantidade de praticantes ativos (1 praticante solo vs 2 praticantes em dupla)
        - Presença de Sonkyō inicial/final (indicativo de Shiai-geiko, Shinsa ou Kata)
        - Frequência e cadência dos golpes no tempo
        - Padrões de deslocamento de pés vs golpes no ar
        - Ritmo contínuo alternado (Kirikaeshi) vs rajadas intensas (Kakari-geiko) vs combate livre (Ji-geiko)

        Retorna: (modality_key, confidence_0_to_1, justification)
        """
        valid_prim = [p for p in primary_history if p]
        valid_sec = [p for p in (secondary_history or []) if p]

        num_practitioners = 2 if len(valid_sec) >= max(5, len(valid_prim) * 0.25) else 1
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
        min_kiri_strikes = 8 if duration_sec >= 25.0 else max(3, int(duration_sec * 0.35))
        if strikes_per_minute >= 40 and num_strikes >= min_kiri_strikes:
            return "kirikaeshi", 0.90, f"Sequência rápida e contínua de golpes alternados com alta cadência ({strikes_per_minute:.1f} CPM), característica de Kirikaeshi."

        # Kakari-geiko: Densidade altíssima de ataques em curto espaço de tempo (32+ CPM)
        if strikes_per_minute >= 32:
            return "kakari_geiko", 0.85, f"Sequência intensa e ininterrupta de ataques com alta frequência ({strikes_per_minute:.1f} CPM), típica de Kakari-geiko."

        # Uchikomi-geiko: Golpes regulares com pausas intermediárias de abertura de alvo (18-32 CPM)
        if 18 <= strikes_per_minute < 32:
            return "uchikomi_geiko", 0.82, f"Cadência ritmada de ataques sobre aberturas sucessivas de alvo pelo parceiro ({strikes_per_minute:.1f} CPM)."

        # Shiai-geiko ou Shinsa com Sonkyō detectado e cadência típica de luta / exame
        if has_sonkyo and 6 <= strikes_per_minute < 20:
            return "shiai_geiko", 0.88, f"Dois praticantes em combate com presença de Sonkyō regulamentar e disputa por Ippon ({strikes_per_minute:.1f} CPM), característico de Shiai-geiko."

        if has_sonkyo and strikes_per_minute < 6:
            return "shinsa", 0.84, f"Execução com Sonkyō formal, etiqueta e pausas estruturadas, padrão de exame de graduação (Shinsa) ou Nihon Kendō Kata."

        # Kihon / Waza-geiko / Yakusoku-geiko: Poucos golpes estruturados, pausas para retorno em Kamae
        if 8 <= strikes_per_minute < 18:
            strike_types = {
                str(getattr(s, "type", s.get("technique", "MEN") if isinstance(s, dict) else "MEN")).upper()
                for s in (detected_strikes or [])
            }
            if len(strike_types) >= 2 or foot_movement_energy >= 0.45:
                return "waza_geiko", 0.84, f"Prática técnica estruturada de golpes ofensivos e contra-ataques (Waza-geiko, cadência: {strikes_per_minute:.1f} CPM)."
            return "kihon", 0.80, f"Estrutura pausada com foco na execução correta dos fundamentos, postura e Zanshin."

        # Combate Livre / Simulação (Ji-geiko)
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
        fps: float = 30.0,
        presence_duration_seconds: Optional[float] = None,
        frame_offset: int = 0
    ) -> TrainingPillarMetrics:
        """
        Calcula os scores numéricos de 0 a 100 para Movimentação, Precisão e Constância.
        presence_duration_seconds: quando informado, calcula o CPM pelo tempo real de presença.
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
            f_raw = st.get("impact_frame", 0) if isinstance(st, dict) else getattr(st, "impact_frame", 0)
            f_impact = int(f_raw or 0) - (frame_offset or 0)
            if not (0 <= f_impact < len(pose_history)) and (0 <= int(f_raw or 0) < len(pose_history)):
                f_impact = int(f_raw or 0)
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
            if not centerline_control_scores:
                centerline_control_scores = [0.80]
            if not target_accuracy_scores:
                target_accuracy_scores = [0.78]
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
        if presence_duration_seconds is not None and presence_duration_seconds > 0:
            duration_min = max(0.1, presence_duration_seconds / 60.0)
        else:
            duration_min = max(0.1, (len(pose_history) / max(1.0, fps)) / 60.0)
        cadence_cpm = total_reps / duration_min

        if len(strikes) >= 3:
            intervals_frames = []
            for i in range(1, len(strikes)):
                s_curr = strikes[i]
                s_prev = strikes[i - 1]
                f_curr = s_curr.get("impact_frame", 0) if isinstance(s_curr, dict) else getattr(s_curr, "impact_frame", 0)
                f_prev = s_prev.get("impact_frame", 0) if isinstance(s_prev, dict) else getattr(s_prev, "impact_frame", 0)
                intervals_frames.append(abs(int(f_curr or 0) - int(f_prev or 0)))

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
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        has_sonkyo: bool = False,
        session_segments_override: Optional[List[Dict[str, Any]]] = None
    ) -> TrainingSessionResult:
        """
        Executa a análise completa da sessão de treinamento, processando cada Kendoca,
        identificando sessões de treinamento dentro da mesma sequência e concentrando
        os resultados em blocos por modalidade de treinamento.
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
            justification = f"Modalidade selecionada manualmente pelo usuário: {TRAINING_MODALITIES_METADATA[modality_key]['name']}."
        else:
            auto_key, conf, just = self.detect_training_modality(
                primary_history=primary_history,
                secondary_history=secondary_history,
                detected_strikes=strikes,
                fps=fps,
                has_sonkyo=has_sonkyo
            )
            modality_key = auto_key
            confidence = conf
            det_method = "AUTO_DETECTED"
            is_override = False
            justification = just

        modality_meta = TRAINING_MODALITIES_METADATA.get(modality_key, TRAINING_MODALITIES_METADATA["suburi"])
        modality_name = modality_meta["name"]
        modality_cat = modality_meta.get("category", "Geral")

        # Obter princípios acumulados e fontes de conhecimento do Kendo
        learned_principles, web_sources = self._get_modality_knowledge(modality_key)

        # 2. Identificação e Análise dos Kendocas Rastreáveis
        valid_prim = [p for p in primary_history if p]
        valid_sec = [p for p in (secondary_history or []) if p]
        is_two_kendokas = len(valid_sec) >= max(5, len(valid_prim) * 0.25)

        kendokas_list: List[KendokaTrainingProfile] = []

        # Separar golpes de cada lutador
        strikes_k1 = [s for s in strikes if getattr(s, "attacker_id", "KENSHI_SHIRO") in ["KENSHI_SHIRO", "KENSHI_SOLO", "KENSHI_1", "KENSHI_01"]]
        strikes_k2 = [s for s in strikes if getattr(s, "attacker_id", "") in ["KENSHI_AKA", "KENSHI_2", "KENSHI_02"]]

        if not strikes_k1 and not strikes_k2 and strikes:
            strikes_k1 = strikes

        histories_map: Dict[str, List[Optional[Dict[str, Any]]]] = {}
        strikes_map: Dict[str, List[Any]] = {}
        default_names_map: Dict[str, str] = {}
        roles_map: Dict[str, str] = {}

        if not is_two_kendokas:
            # Kendoca Solo (ex: Suburi individual, Ashi-sabaki solo)
            k_id = "KENSHI_SOLO"
            def_name = "Kendoca Principal"
            c_name = custom_names.get(k_id, custom_names.get("KENSHI_SHIRO", def_name))
            role_solo = "Praticante Individual"

            histories_map[k_id] = list(primary_history)
            strikes_map[k_id] = list(strikes_k1)
            default_names_map[k_id] = def_name
            roles_map[k_id] = role_solo

            pillars_k1 = self.calculate_pillar_metrics(primary_history, strikes_k1, modality_key, fps=fps)
            str_k1, imp_k1, exe_k1 = self.generate_pedagogical_feedback(pillars_k1, modality_key, c_name)

            # Eixo 6: Integração com o Baseline Cinestésico Individual
            kin_devs_k1 = {}
            kin_ins_k1 = []
            try:
                from src.analytics.kenshi_style_model import KinestheticProfileManager
                kin_mgr = KinestheticProfileManager()
                prof_k1 = kin_mgr.get_or_create_profile(k_id, display_name=c_name)
                eval_k1 = prof_k1.evaluate_strike_against_baseline(
                    strike_type="MEN",
                    observed_spine_tilt_deg=pillars_k1.movimentacao_submetrics.get("verticalidade_coluna")
                )
                kin_devs_k1 = eval_k1.get("deviations", {})
                kin_ins_k1 = eval_k1.get("insights", [])

                kin_mgr.record_training_session(
                    kenshi_id=k_id,
                    display_name=c_name,
                    cadence_cpm=pillars_k1.cadence_cpm,
                    mean_spine_tilt=pillars_k1.movimentacao_submetrics.get("verticalidade_coluna")
                )
            except Exception:
                pass

            rep_timeline = self._build_repetition_timeline(strikes_k1, fps=fps)

            kendokas_list.append(KendokaTrainingProfile(
                kendoka_id=k_id,
                default_name=def_name,
                custom_name=c_name,
                role=role_solo,
                pillars=pillars_k1,
                strengths=str_k1,
                improvements=imp_k1,
                recommended_exercises=exe_k1,
                repetition_timeline=rep_timeline,
                kinesthetic_deviations=kin_devs_k1,
                kinesthetic_insights=kin_ins_k1,
                presence_frames=len(valid_prim),
                presence_seconds=round(len(valid_prim) / max(1.0, fps), 1)
            ))
        else:
            # 2 Kendocas em Treino (ex: Kirikaeshi, Uchikomi, Kakari, Kata, Shinsa, Ji-geiko)
            role_k1, role_k2 = self._get_pair_roles_for_modality(modality_key)

            # Kendoca 1 (Shiro / Esquerda)
            k1_id = "KENSHI_SHIRO"
            k1_def_name = "Kendoca Shiro (Esquerda)"
            k1_c_name = custom_names.get(k1_id, k1_def_name)
            histories_map[k1_id] = list(primary_history)
            strikes_map[k1_id] = list(strikes_k1)
            default_names_map[k1_id] = k1_def_name
            roles_map[k1_id] = role_k1

            pillars_k1 = self.calculate_pillar_metrics(primary_history, strikes_k1, modality_key, fps=fps)
            str_k1, imp_k1, exe_k1 = self.generate_pedagogical_feedback(pillars_k1, modality_key, k1_c_name)
            rep_tl_k1 = self._build_repetition_timeline(strikes_k1, fps=fps)

            kin_devs_k1 = {}
            kin_ins_k1 = []
            try:
                from src.analytics.kenshi_style_model import KinestheticProfileManager
                kin_mgr = KinestheticProfileManager()
                prof_k1 = kin_mgr.get_or_create_profile(k1_id, display_name=k1_c_name)
                eval_k1 = prof_k1.evaluate_strike_against_baseline(
                    strike_type="MEN",
                    observed_spine_tilt_deg=pillars_k1.movimentacao_submetrics.get("verticalidade_coluna")
                )
                kin_devs_k1 = eval_k1.get("deviations", {})
                kin_ins_k1 = eval_k1.get("insights", [])

                kin_mgr.record_training_session(
                    kenshi_id=k1_id,
                    display_name=k1_c_name,
                    cadence_cpm=pillars_k1.cadence_cpm,
                    mean_spine_tilt=pillars_k1.movimentacao_submetrics.get("verticalidade_coluna")
                )
            except Exception:
                pass

            kendokas_list.append(KendokaTrainingProfile(
                kendoka_id=k1_id,
                default_name=k1_def_name,
                custom_name=k1_c_name,
                role=role_k1,
                pillars=pillars_k1,
                strengths=str_k1,
                improvements=imp_k1,
                recommended_exercises=exe_k1,
                repetition_timeline=rep_tl_k1,
                kinesthetic_deviations=kin_devs_k1,
                kinesthetic_insights=kin_ins_k1,
                presence_frames=len(valid_prim),
                presence_seconds=round(len(valid_prim) / max(1.0, fps), 1)
            ))

            # Kendoca 2 (Aka / Direita)
            k2_id = "KENSHI_AKA"
            k2_def_name = "Kendoca Aka (Direita)"
            k2_c_name = custom_names.get(k2_id, k2_def_name)
            histories_map[k2_id] = list(secondary_history or [])
            strikes_map[k2_id] = list(strikes_k2)
            default_names_map[k2_id] = k2_def_name
            roles_map[k2_id] = role_k2

            pillars_k2 = self.calculate_pillar_metrics(secondary_history or [], strikes_k2, modality_key, fps=fps)
            str_k2, imp_k2, exe_k2 = self.generate_pedagogical_feedback(pillars_k2, modality_key, k2_c_name)
            rep_tl_k2 = self._build_repetition_timeline(strikes_k2, fps=fps)

            kin_devs_k2 = {}
            kin_ins_k2 = []
            try:
                from src.analytics.kenshi_style_model import KinestheticProfileManager
                kin_mgr = KinestheticProfileManager()
                prof_k2 = kin_mgr.get_or_create_profile(k2_id, display_name=k2_c_name)
                eval_k2 = prof_k2.evaluate_strike_against_baseline(
                    strike_type="MEN",
                    observed_spine_tilt_deg=pillars_k2.movimentacao_submetrics.get("verticalidade_coluna")
                )
                kin_devs_k2 = eval_k2.get("deviations", {})
                kin_ins_k2 = eval_k2.get("insights", [])

                kin_mgr.record_training_session(
                    kenshi_id=k2_id,
                    display_name=k2_c_name,
                    cadence_cpm=pillars_k2.cadence_cpm,
                    mean_spine_tilt=pillars_k2.movimentacao_submetrics.get("verticalidade_coluna")
                )
            except Exception:
                pass

            kendokas_list.append(KendokaTrainingProfile(
                kendoka_id=k2_id,
                default_name=k2_def_name,
                custom_name=k2_c_name,
                role=role_k2,
                pillars=pillars_k2,
                strengths=str_k2,
                improvements=imp_k2,
                recommended_exercises=exe_k2,
                repetition_timeline=rep_tl_k2,
                kinesthetic_deviations=kin_devs_k2,
                kinesthetic_insights=kin_ins_k2,
                presence_frames=len(valid_sec),
                presence_seconds=round(len(valid_sec) / max(1.0, fps), 1)
            ))

        # 3. Segmentação de Sub-Sessões na Sequência e Concentração por Bloco de Modalidade
        sub_sessions = self.segment_training_sequence(
            kenshi_histories=histories_map,
            detected_strikes_by_kenshi=strikes_map,
            fps=fps,
            modality_override=modality_override,
            session_segments_override=session_segments_override,
            custom_kendoka_names=custom_names,
            has_sonkyo=has_sonkyo,
            default_names_by_id=default_names_map,
            roles_by_id=roles_map
        )
        modality_blocks = self.build_modality_blocks(
            sub_sessions=sub_sessions,
            kenshi_histories=histories_map,
            detected_strikes_by_kenshi=strikes_map,
            global_kendokas=kendokas_list,
            fps=fps,
            custom_kendoka_names=custom_names,
            default_names_by_id=default_names_map,
            roles_by_id=roles_map
        )

        # Se múltiplas modalidades foram detectadas/definidas na sequência, enriquecer resumo e modalidade principal
        if modality_blocks and not is_override:
            dominant_block = max(modality_blocks, key=lambda b: b.total_duration_seconds)
            modality_key = dominant_block.modality_key
            modality_name = dominant_block.modality_name
            modality_cat = dominant_block.modality_category
            learned_principles = dominant_block.learned_principles or learned_principles
            web_sources = dominant_block.web_sources or web_sources

        # 4. Resumo Textual da Sessão de Treinamento
        if len(sub_sessions) > 1 or len(modality_blocks) > 1:
            seq_desc = "; ".join(
                f"{s.start_timestamp} a {s.end_timestamp} {s.modality_name}"
                for s in sub_sessions
            )
            blocks_desc = ", ".join(
                f"{b.modality_name} ({b.intervals_label})"
                for b in modality_blocks
            )
            summary = (
                f"Sequência de treinamento de Kendo com {len(sub_sessions)} sessões identificadas "
                f"({seq_desc}), concentradas em {len(modality_blocks)} bloco(s) de modalidade: {blocks_desc}. "
                f"Duração total: {duration_sec:.1f}s | Praticantes rastreáveis: {len(kendokas_list)}."
            )
        else:
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
            session_summary_text=summary,
            justification=justification,
            learned_principles=learned_principles,
            modality_category=modality_cat,
            web_sources=web_sources,
            sub_sessions=sub_sessions,
            modality_blocks=modality_blocks
        )

    # --------------------------------------------------------------------------
    # MÉTODOS AUXILIARES E DE SEGMENTAÇÃO TEMPORAL / BLOCOS DE MODALIDADE
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

    def _get_modality_knowledge(self, modality_key: str) -> Tuple[List[str], List[Dict[str, str]]]:
        """Retorna princípios técnicos aprendidos e fontes web para uma modalidade."""
        modality_meta = TRAINING_MODALITIES_METADATA.get(modality_key, TRAINING_MODALITIES_METADATA["suburi"])
        modality_name = modality_meta["name"]
        try:
            from src.engine.auto_trainer import get_modality_learned_knowledge
            learned_info = get_modality_learned_knowledge(modality_key)
        except Exception:
            learned_info = {}

        learned_principles = learned_info.get("principles_learned", [])
        if not learned_principles:
            focus_str = ", ".join(modality_meta.get("focus_areas", ["Fundamentos", "Ki-Ken-Tai-Ichi"]))
            learned_principles = [
                f"Princípio Técnico ({modality_name}): Domínio de {focus_str}.",
                "Alinhamento postural (Shisei) e prontidão marcial (Zanshin)."
            ]
        web_sources = learned_info.get("web_sources", [])
        return learned_principles, web_sources

    def _get_pair_roles_for_modality(self, modality_key: str) -> Tuple[str, str]:
        """Define os papéis técnicos dos 2 praticantes conforme a modalidade."""
        if modality_key in ["nihon_kendo_kata", "bokuto_kihon_waza"]:
            return "Uchidachi (Professor / Líder)", "Shidachi (Aluno / Resposta)"
        elif modality_key in ["uchikomi_geiko", "kakari_geiko", "kirikaeshi"]:
            return "Kakarite (Atacante)", "Motodachi (Receptor)"
        elif modality_key == "shinsa":
            return "Candidato 1 (Exame)", "Candidato 2 (Exame)"
        return "Praticante 1 (Shiro)", "Praticante 2 (Aka)"

    def _extract_strike_frame_and_sec(self, strike: Any, fps: float = 30.0, fallback_idx: int = 0) -> Tuple[int, float]:
        """Extrai o frame de impacto e o tempo em segundos de um golpe (objeto Strike ou dict)."""
        fps_safe = max(1.0, float(fps or 30.0))
        if isinstance(strike, dict):
            if "impact_frame" in strike and strike["impact_frame"] is not None:
                f_idx = int(strike["impact_frame"])
                return f_idx, round(f_idx / fps_safe, 2)
            if "timestamp_seconds" in strike and strike["timestamp_seconds"] is not None:
                sec = float(strike["timestamp_seconds"])
                return round(sec * fps_safe), round(sec, 2)
            ts_raw = strike.get("timestamp_impact", strike.get("timestamp"))
            if ts_raw is not None:
                sec = parse_training_mmss(ts_raw, default=fallback_idx * 2.0)
                return round(sec * fps_safe), round(sec, 2)
            sec = fallback_idx * 2.0
            return round(sec * fps_safe), round(sec, 2)
        else:
            f_val = getattr(strike, "impact_frame", None)
            if f_val is not None:
                f_idx = int(f_val)
                return f_idx, round(f_idx / fps_safe, 2)
            ts_raw = getattr(strike, "timestamp_impact", getattr(strike, "timestamp", None))
            if ts_raw is not None:
                sec = parse_training_mmss(ts_raw, default=fallback_idx * 2.0)
                return round(sec * fps_safe), round(sec, 2)
            sec = fallback_idx * 2.0
            return round(sec * fps_safe), round(sec, 2)

    def _build_repetition_timeline(self, strikes: List[Any], fps: float = 30.0) -> List[Dict[str, Any]]:
        """Constrói a linha do tempo de repetições com timestamp formatado e timestamp_seconds numérico."""
        timeline: List[Dict[str, Any]] = []
        for i, s in enumerate(strikes):
            f_idx, sec = self._extract_strike_frame_and_sec(s, fps=fps, fallback_idx=i)
            if isinstance(s, dict):
                ts_label = s.get("timestamp", format_training_mmss(sec))
                tech = s.get("technique", s.get("type", "MEN"))
            else:
                ts_label = getattr(s, "timestamp_impact", getattr(s, "timestamp", f"{sec:.1f}s"))
                tech = getattr(s, "type", "MEN")
            timeline.append({
                "repetition": i + 1,
                "timestamp": ts_label,
                "timestamp_seconds": sec,
                "impact_frame": f_idx,
                "technique": tech
            })
        return timeline

    def segment_training_sequence(
        self,
        kenshi_histories: Dict[str, List[Optional[Dict[str, Any]]]],
        detected_strikes_by_kenshi: Optional[Dict[str, List[Any]]] = None,
        fps: float = 30.0,
        modality_override: Optional[str] = None,
        session_segments_override: Optional[List[Dict[str, Any]]] = None,
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        kenshis_metadata: Optional[List[Dict[str, Any]]] = None,
        has_sonkyo: bool = False,
        default_names_by_id: Optional[Dict[str, str]] = None,
        roles_by_id: Optional[Dict[str, str]] = None
    ) -> List[TrainingSubSession]:
        """
        Identifica as sessões de treinamento dentro da mesma sequência (ex: 00:00 a 03:00 Kirikaeshi,
        03:00 a 05:00 Waza-geiko), seja automaticamente por janelas temporais + fusão de blocos adjacentes,
        seja através de definição/edição manual de intervalos (session_segments_override).
        """
        fps_safe = max(1.0, float(fps or 30.0))
        strikes_map = detected_strikes_by_kenshi or {}
        custom_names = custom_kendoka_names or {}
        def_names = default_names_by_id or {}
        roles = roles_by_id or {}

        total_frames = max([len(h) for h in kenshi_histories.values()]) if kenshi_histories else 1
        total_frames = max(1, total_frames)
        total_duration_sec = max(1.0, total_frames / fps_safe)

        sorted_kenshis = sorted(
            kenshi_histories.keys(),
            key=lambda k: len([p for p in kenshi_histories[k] if p]),
            reverse=True
        )
        primary_k_id = sorted_kenshis[0] if sorted_kenshis else "KENSHI_01"
        primary_hist = kenshi_histories.get(primary_k_id, [])
        secondary_hist = kenshi_histories.get(sorted_kenshis[1], []) if len(sorted_kenshis) > 1 else None

        # Mapear todos os golpes com seus frames e segundos
        indexed_strikes_by_k: Dict[str, List[Tuple[int, float, Any]]] = {}
        all_indexed_strikes: List[Tuple[int, float, Any]] = []
        for k_id, s_list in strikes_map.items():
            k_indexed = []
            for idx_s, s_obj in enumerate(s_list or []):
                f_idx, sec = self._extract_strike_frame_and_sec(s_obj, fps=fps_safe, fallback_idx=idx_s)
                k_indexed.append((f_idx, sec, s_obj))
                all_indexed_strikes.append((f_idx, sec, s_obj))
            indexed_strikes_by_k[k_id] = k_indexed

        raw_segments: List[Dict[str, Any]] = []

        # CASO A: Segmentos definidos/editados manualmente pelo usuário (session_segments_override)
        if session_segments_override:
            for seg_spec in session_segments_override:
                if not isinstance(seg_spec, dict):
                    continue
                s_sec = (
                    float(seg_spec["start_seconds"])
                    if "start_seconds" in seg_spec and seg_spec["start_seconds"] is not None
                    else parse_training_mmss(
                        seg_spec.get("start_timestamp", seg_spec.get("start_time", seg_spec.get("start", "00:00"))),
                        default=0.0
                    )
                )
                e_sec = (
                    float(seg_spec["end_seconds"])
                    if "end_seconds" in seg_spec and seg_spec["end_seconds"] is not None
                    else parse_training_mmss(
                        seg_spec.get("end_timestamp", seg_spec.get("end_time", seg_spec.get("end", format_training_mmss(total_duration_sec)))),
                        default=total_duration_sec
                    )
                )
                if e_sec <= s_sec:
                    e_sec = max(s_sec + 1.0, total_duration_sec)

                s_frame = max(0, round(s_sec * fps_safe))
                e_frame = max(s_frame + 1, round(e_sec * fps_safe))

                req_mod = str(seg_spec.get("modality_key", "auto") or "auto").strip()
                p_slice = primary_hist[min(s_frame, len(primary_hist)):min(e_frame, len(primary_hist))]
                s_slice = secondary_hist[min(s_frame, len(secondary_hist)):min(e_frame, len(secondary_hist))] if secondary_hist else None
                seg_strikes = [st for f_i, sc_i, st in all_indexed_strikes if s_sec <= sc_i < e_sec or (s_sec == 0 and sc_i == 0) or (abs(e_sec - total_duration_sec) < 0.5 and sc_i <= e_sec)]

                if req_mod in TRAINING_MODALITIES_METADATA:
                    mod_key = req_mod
                    conf = 1.0
                    det_m = "MANUAL_SELECT"
                    mod_title = TRAINING_MODALITIES_METADATA[mod_key]["name"]
                    just = f"Sessão de {format_training_mmss(s_sec)} a {format_training_mmss(e_sec)} definida como {mod_title}."
                else:
                    mod_key, conf, just = self.detect_training_modality(
                        primary_history=p_slice if p_slice else primary_hist,
                        secondary_history=s_slice if s_slice else secondary_hist,
                        detected_strikes=seg_strikes,
                        fps=fps_safe,
                        has_sonkyo=has_sonkyo
                    )
                    det_m = "AUTO_DETECTED"

                raw_segments.append({
                    "start_frame": s_frame,
                    "end_frame": e_frame,
                    "start_seconds": s_sec,
                    "end_seconds": e_sec,
                    "start_timestamp": seg_spec.get("start_timestamp") or format_training_mmss(s_sec),
                    "end_timestamp": seg_spec.get("end_timestamp") or format_training_mmss(e_sec),
                    "modality_key": mod_key,
                    "confidence": conf,
                    "detection_method": det_m,
                    "justification": just
                })

        # CASO B: Modalidade única fixada manualmente sem sub-segmentos customizados
        elif modality_override and modality_override in TRAINING_MODALITIES_METADATA:
            mod_title = TRAINING_MODALITIES_METADATA[modality_override]["name"]
            raw_segments.append({
                "start_frame": 0,
                "end_frame": total_frames,
                "start_seconds": 0.0,
                "end_seconds": total_duration_sec,
                "start_timestamp": "00:00",
                "end_timestamp": format_training_mmss(total_duration_sec),
                "modality_key": modality_override,
                "confidence": 1.0,
                "detection_method": "MANUAL_SELECT",
                "justification": f"Modalidade selecionada manualmente para a sequência: {mod_title}."
            })

        # CASO C: Segmentação automática de sessões ao longo da sequência
        else:
            if total_duration_sec < 20.0:
                all_st = [st for _, _, st in all_indexed_strikes]
                mod_key, conf, just = self.detect_training_modality(
                    primary_history=primary_hist,
                    secondary_history=secondary_hist,
                    detected_strikes=all_st,
                    fps=fps_safe,
                    has_sonkyo=has_sonkyo
                )
                raw_segments.append({
                    "start_frame": 0,
                    "end_frame": total_frames,
                    "start_seconds": 0.0,
                    "end_seconds": total_duration_sec,
                    "start_timestamp": "00:00",
                    "end_timestamp": format_training_mmss(total_duration_sec),
                    "modality_key": mod_key,
                    "confidence": conf,
                    "detection_method": "AUTO_DETECTED",
                    "justification": just
                })
            else:
                # Janela base divisora exata de minutos (10s para vídeos <= 60s, 15s para <= 360s, 30s para vídeos longos)
                if total_duration_sec <= 60.0:
                    win_sec = 10.0
                elif total_duration_sec <= 360.0:
                    win_sec = 15.0
                else:
                    win_sec = 30.0
                win_frames = max(round(win_sec * fps_safe), 15)

                windows: List[Dict[str, Any]] = []
                cur_f = 0
                while cur_f < total_frames:
                    nxt_f = min(total_frames, cur_f + win_frames)
                    # Se o restante final for muito curto (< 40% da janela) e já houver janelas, anexa a esta janela
                    if (total_frames - nxt_f) < int(win_frames * 0.40):
                        nxt_f = total_frames

                    w_s_sec = cur_f / fps_safe
                    w_e_sec = nxt_f / fps_safe
                    p_slice = primary_hist[cur_f:nxt_f]
                    s_slice = secondary_hist[cur_f:nxt_f] if secondary_hist else None
                    w_strikes = [
                        st for f_i, sc_i, st in all_indexed_strikes
                        if cur_f <= f_i < nxt_f or (nxt_f == total_frames and f_i == total_frames)
                    ]

                    w_mod, w_conf, w_just = self.detect_training_modality(
                        primary_history=p_slice,
                        secondary_history=s_slice,
                        detected_strikes=w_strikes,
                        fps=fps_safe,
                        has_sonkyo=has_sonkyo and (cur_f == 0 or nxt_f == total_frames)
                    )
                    windows.append({
                        "start_frame": cur_f,
                        "end_frame": nxt_f,
                        "start_seconds": w_s_sec,
                        "end_seconds": w_e_sec,
                        "modality_key": w_mod,
                        "confidence": w_conf,
                        "justification": w_just
                    })
                    cur_f = nxt_f

                # Suavização de micro-oscilações isoladas (janela única cercada por modalidades iguais)
                if len(windows) >= 3:
                    for idx_w in range(1, len(windows) - 1):
                        prev_mod = windows[idx_w - 1]["modality_key"]
                        next_mod = windows[idx_w + 1]["modality_key"]
                        curr_mod = windows[idx_w]["modality_key"]
                        if prev_mod == next_mod and curr_mod != prev_mod:
                            windows[idx_w]["modality_key"] = prev_mod

                # Fusão de janelas adjacentes com a mesma modalidade
                merged_windows: List[Dict[str, Any]] = []
                for w in windows:
                    if not merged_windows or merged_windows[-1]["modality_key"] != w["modality_key"]:
                        merged_windows.append(dict(w))
                    else:
                        merged_windows[-1]["end_frame"] = w["end_frame"]
                        merged_windows[-1]["end_seconds"] = w["end_seconds"]

                # Reavaliar cada segmento fundido completo para obter confiança e justificativa precisas
                for mw in merged_windows:
                    sf = mw["start_frame"]
                    ef = mw["end_frame"]
                    s_sec = sf / fps_safe
                    e_sec = ef / fps_safe
                    p_slice = primary_hist[sf:ef]
                    s_slice = secondary_hist[sf:ef] if secondary_hist else None
                    m_strikes = [
                        st for f_i, sc_i, st in all_indexed_strikes
                        if sf <= f_i < ef or (ef == total_frames and f_i == total_frames)
                    ]
                    m_mod, m_conf, m_just = self.detect_training_modality(
                        primary_history=p_slice,
                        secondary_history=s_slice,
                        detected_strikes=m_strikes,
                        fps=fps_safe,
                        has_sonkyo=has_sonkyo and (sf == 0 or ef == total_frames)
                    )
                    # Preserva a modalidade fundida se a reavaliação for compatível, ou usa a modalidade reavaliada
                    final_mod = m_mod if m_mod in TRAINING_MODALITIES_METADATA else mw["modality_key"]
                    raw_segments.append({
                        "start_frame": sf,
                        "end_frame": ef,
                        "start_seconds": s_sec,
                        "end_seconds": e_sec,
                        "start_timestamp": format_training_mmss(s_sec),
                        "end_timestamp": format_training_mmss(e_sec),
                        "modality_key": final_mod,
                        "confidence": m_conf,
                        "detection_method": "AUTO_DETECTED",
                        "justification": m_just
                    })

                # Segunda passagem de fusão caso a reavaliação tenha igualado segmentos vizinhos
                compact_segments: List[Dict[str, Any]] = []
                for seg in raw_segments:
                    if not compact_segments or compact_segments[-1]["modality_key"] != seg["modality_key"]:
                        compact_segments.append(seg)
                    else:
                        compact_segments[-1]["end_frame"] = seg["end_frame"]
                        compact_segments[-1]["end_seconds"] = seg["end_seconds"]
                        compact_segments[-1]["end_timestamp"] = seg["end_timestamp"]
                        compact_segments[-1]["confidence"] = round(
                            (compact_segments[-1]["confidence"] + seg["confidence"]) / 2.0, 2
                        )
                raw_segments = compact_segments

        # Construir objetos TrainingSubSession avaliando os Kendocas em cada segmento
        sub_sessions_list: List[TrainingSubSession] = []
        for idx_seg, seg in enumerate(raw_segments, start=1):
            sf = int(seg["start_frame"])
            ef = int(seg["end_frame"])
            s_sec = float(seg["start_seconds"])
            e_sec = float(seg["end_seconds"])
            seg_dur = max(0.1, e_sec - s_sec)
            mod_k = seg["modality_key"]
            mod_meta = TRAINING_MODALITIES_METADATA.get(mod_k, TRAINING_MODALITIES_METADATA["suburi"])
            mod_name = mod_meta["name"]
            mod_cat = mod_meta.get("category", "Geral")

            seg_kendokas: List[KendokaTrainingProfile] = []
            seg_total_reps = 0

            for k_idx, k_id in enumerate(sorted_kenshis):
                full_hist = kenshi_histories.get(k_id, [])
                if sf < len(full_hist):
                    k_slice = full_hist[sf:min(ef, len(full_hist))]
                else:
                    k_slice = [p for p in full_hist if p][:max(1, round(seg_dur * fps_safe))]

                valid_k_slice = [p for p in k_slice if p]
                k_indexed = indexed_strikes_by_k.get(k_id, [])
                k_seg_strikes = [
                    st for f_i, sc_i, st in k_indexed
                    if s_sec <= sc_i < e_sec
                    or (idx_seg == len(raw_segments) and abs(sc_i - e_sec) <= 0.5)
                ]
                seg_total_reps += len(k_seg_strikes)

                # Presença no segmento
                if valid_k_slice:
                    pres_sec = max(0.1, min(seg_dur, len(valid_k_slice) / fps_safe))
                    # Se o praticante esteve presente na maior parte dos frames amostrados do segmento, usa a duração do segmento
                    if len(k_slice) > 0 and (len(valid_k_slice) / len(k_slice)) >= 0.70:
                        pres_sec = seg_dur
                else:
                    pres_sec = seg_dur

                def_n = def_names.get(k_id, f"Kenshi {k_id.replace('KENSHI_', '')}")
                c_n = custom_names.get(k_id, def_n)
                role_n = roles.get(k_id, f"Praticante #{k_idx + 1}")

                pillars_seg = self.calculate_pillar_metrics(
                    pose_history=k_slice if valid_k_slice else full_hist,
                    strikes=k_seg_strikes,
                    modality_key=mod_k,
                    fps=fps_safe,
                    presence_duration_seconds=pres_sec,
                    frame_offset=sf
                )
                str_seg, imp_seg, exe_seg = self.generate_pedagogical_feedback(pillars_seg, mod_k, c_n)
                rep_tl_seg = self._build_repetition_timeline(k_seg_strikes, fps=fps_safe)

                seg_kendokas.append(KendokaTrainingProfile(
                    kendoka_id=k_id,
                    default_name=def_n,
                    custom_name=c_n,
                    role=role_n,
                    pillars=pillars_seg,
                    strengths=str_seg,
                    improvements=imp_seg,
                    recommended_exercises=exe_seg,
                    repetition_timeline=rep_tl_seg,
                    presence_frames=len(valid_k_slice),
                    presence_seconds=round(pres_sec, 1),
                    presence_intervals=[(sf, ef)]
                ))

            sub_sessions_list.append(TrainingSubSession(
                segment_index=idx_seg,
                start_frame=sf,
                end_frame=ef,
                start_seconds=s_sec,
                end_seconds=e_sec,
                start_timestamp=seg.get("start_timestamp") or format_training_mmss(s_sec),
                end_timestamp=seg.get("end_timestamp") or format_training_mmss(e_sec),
                modality_key=mod_k,
                modality_name=mod_name,
                modality_category=mod_cat,
                detection_confidence=seg["confidence"],
                detection_method=seg["detection_method"],
                justification=seg["justification"],
                total_repetitions=seg_total_reps,
                kendokas=seg_kendokas
            ))

        return sub_sessions_list

    def build_modality_blocks(
        self,
        sub_sessions: List[TrainingSubSession],
        kenshi_histories: Dict[str, List[Optional[Dict[str, Any]]]],
        detected_strikes_by_kenshi: Optional[Dict[str, List[Any]]] = None,
        global_kendokas: Optional[List[KendokaTrainingProfile]] = None,
        fps: float = 30.0,
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        default_names_by_id: Optional[Dict[str, str]] = None,
        roles_by_id: Optional[Dict[str, str]] = None
    ) -> List[TrainingModalityBlock]:
        """
        Agrupa e concentra todas as sub-sessões por modalidade de treinamento (modality_key),
        consolidando os intervalos de tempo, repetições, fundamentação técnica e os 3 Pilares
        de cada Kendoca no bloco de cada modalidade.
        """
        if not sub_sessions:
            return []

        fps_safe = max(1.0, float(fps or 30.0))
        strikes_map = detected_strikes_by_kenshi or {}
        custom_names = custom_kendoka_names or {}
        def_names = default_names_by_id or {}
        roles = roles_by_id or {}

        sorted_kenshis = sorted(
            kenshi_histories.keys(),
            key=lambda k: len([p for p in kenshi_histories[k] if p]),
            reverse=True
        )

        indexed_strikes_by_k: Dict[str, List[Tuple[int, float, Any]]] = {}
        for k_id, s_list in strikes_map.items():
            indexed_strikes_by_k[k_id] = [
                (*self._extract_strike_frame_and_sec(s_obj, fps=fps_safe, fallback_idx=idx_s), s_obj)
                for idx_s, s_obj in enumerate(s_list or [])
            ]

        # Agrupar sub-sessões preservando a ordem de aparição de cada modalidade
        grouped_by_mod: Dict[str, List[TrainingSubSession]] = {}
        mod_order: List[str] = []
        for ss in sub_sessions:
            m_key = ss.modality_key
            if m_key not in grouped_by_mod:
                grouped_by_mod[m_key] = []
                mod_order.append(m_key)
            grouped_by_mod[m_key].append(ss)

        blocks: List[TrainingModalityBlock] = []
        breakdown_by_kendoka_id: Dict[str, List[Dict[str, Any]]] = {k_id: [] for k_id in sorted_kenshis}

        for b_idx, mod_k in enumerate(mod_order, start=1):
            ss_list = grouped_by_mod[mod_k]
            mod_meta = TRAINING_MODALITIES_METADATA.get(mod_k, TRAINING_MODALITIES_METADATA["suburi"])
            mod_name = mod_meta["name"]
            mod_cat = mod_meta.get("category", "Geral")
            mod_desc = mod_meta.get("description", "")
            mod_focus = list(mod_meta.get("focus_areas", []))
            learned_principles, web_sources = self._get_modality_knowledge(mod_k)

            intervals: List[Dict[str, Any]] = []
            total_dur = 0.0
            total_reps = 0
            confs: List[float] = []
            methods: List[str] = []
            justifications: List[str] = []

            for ss in ss_list:
                intervals.append({
                    "segment_index": ss.segment_index,
                    "start_frame": ss.start_frame,
                    "end_frame": ss.end_frame,
                    "start_seconds": ss.start_seconds,
                    "end_seconds": ss.end_seconds,
                    "start_timestamp": ss.start_timestamp,
                    "end_timestamp": ss.end_timestamp,
                    "duration_seconds": ss.duration_seconds,
                    "total_repetitions": ss.total_repetitions
                })
                total_dur += ss.duration_seconds
                total_reps += ss.total_repetitions
                confs.append(ss.detection_confidence)
                methods.append(ss.detection_method)
                if ss.justification and ss.justification not in justifications:
                    justifications.append(ss.justification)

            intervals_label = ", ".join(
                f"{iv['start_timestamp']} a {iv['end_timestamp']}"
                for iv in intervals
            )
            avg_conf = round(float(np.mean(confs)), 2) if confs else 0.85
            block_method = "MANUAL_SELECT" if "MANUAL_SELECT" in methods else "AUTO_DETECTED"
            block_just = " | ".join(justifications) if justifications else f"Bloco concentrado da modalidade {mod_name}."

            # Consolidar métricas de cada Kendoca especificamente neste bloco de modalidade
            block_kendokas: List[KendokaTrainingProfile] = []
            for k_idx, k_id in enumerate(sorted_kenshis):
                full_hist = kenshi_histories.get(k_id, [])
                k_indexed = indexed_strikes_by_k.get(k_id, [])

                concat_poses: List[Optional[Dict[str, Any]]] = []
                concat_strikes: List[Any] = []
                k_intervals_frames: List[Tuple[int, int]] = []
                k_pres_sec = 0.0

                for ss in ss_list:
                    sf, ef = ss.start_frame, ss.end_frame
                    s_sec, e_sec = ss.start_seconds, ss.end_seconds
                    if sf < len(full_hist):
                        sl = full_hist[sf:min(ef, len(full_hist))]
                    else:
                        sl = [p for p in full_hist if p][:max(1, round(ss.duration_seconds * fps_safe))]
                    concat_poses.extend(sl)
                    k_intervals_frames.append((sf, ef))

                    valid_sl = [p for p in sl if p]
                    if valid_sl and len(sl) > 0 and (len(valid_sl) / len(sl)) >= 0.70:
                        k_pres_sec += ss.duration_seconds
                    elif valid_sl:
                        k_pres_sec += min(ss.duration_seconds, len(valid_sl) / fps_safe)
                    else:
                        k_pres_sec += ss.duration_seconds

                    for f_i, sc_i, st in k_indexed:
                        if s_sec <= sc_i < e_sec or (ss.segment_index == len(sub_sessions) and abs(sc_i - e_sec) <= 0.5):
                            concat_strikes.append(st)

                k_pres_sec = max(0.1, k_pres_sec)
                def_n = def_names.get(k_id, f"Kenshi {k_id.replace('KENSHI_', '')}")
                c_n = custom_names.get(k_id, def_n)
                if len(sorted_kenshis) == 2 and k_id in ["KENSHI_SHIRO", "KENSHI_AKA"]:
                    r_k1, r_k2 = self._get_pair_roles_for_modality(mod_k)
                    role_n = r_k1 if k_id == "KENSHI_SHIRO" else r_k2
                else:
                    role_n = roles.get(k_id, f"Praticante #{k_idx + 1}")

                pillars_blk = self.calculate_pillar_metrics(
                    pose_history=concat_poses if [p for p in concat_poses if p] else full_hist,
                    strikes=concat_strikes,
                    modality_key=mod_k,
                    fps=fps_safe,
                    presence_duration_seconds=k_pres_sec
                )
                str_blk, imp_blk, exe_blk = self.generate_pedagogical_feedback(pillars_blk, mod_k, c_n)
                rep_tl_blk = self._build_repetition_timeline(concat_strikes, fps=fps_safe)

                k_prof_blk = KendokaTrainingProfile(
                    kendoka_id=k_id,
                    default_name=def_n,
                    custom_name=c_n,
                    role=role_n,
                    pillars=pillars_blk,
                    strengths=str_blk,
                    improvements=imp_blk,
                    recommended_exercises=exe_blk,
                    repetition_timeline=rep_tl_blk,
                    presence_frames=len([p for p in concat_poses if p]),
                    presence_seconds=round(k_pres_sec, 1),
                    presence_intervals=k_intervals_frames
                )
                block_kendokas.append(k_prof_blk)

                breakdown_by_kendoka_id.setdefault(k_id, []).append({
                    "block_index": b_idx,
                    "modality_key": mod_k,
                    "modality_name": mod_name,
                    "modality_category": mod_cat,
                    "intervals_label": intervals_label,
                    "total_duration_seconds": round(total_dur, 2),
                    "kendoka_profile": k_prof_blk.to_dict()
                })

            blocks.append(TrainingModalityBlock(
                block_index=b_idx,
                modality_key=mod_k,
                modality_name=mod_name,
                modality_category=mod_cat,
                description=mod_desc,
                focus_areas=mod_focus,
                learned_principles=learned_principles,
                web_sources=web_sources,
                intervals=intervals,
                intervals_label=intervals_label,
                total_duration_seconds=total_dur,
                total_repetitions=total_reps,
                detection_confidence=avg_conf,
                detection_method=block_method,
                justification=block_just,
                kendokas=block_kendokas
            ))

        # Vincular o breakdown de cada modalidade aos perfis globais dos Kendocas
        if global_kendokas:
            for gk in global_kendokas:
                gk.modality_breakdown = breakdown_by_kendoka_id.get(gk.kendoka_id, [])

        return blocks

    # --------------------------------------------------------------------------
    # 5. ANÁLISE DE SESSÃO MULTI-KENSHI (MODO DE TREINAMENTO DEDICADO)
    # --------------------------------------------------------------------------
    def analyze_multi_session(
        self,
        kenshi_histories: Dict[str, List[Optional[Dict[str, Any]]]],
        detected_strikes_by_kenshi: Optional[Dict[str, List[Any]]] = None,
        modality_override: Optional[str] = None,
        fps: float = 30.0,
        custom_kendoka_names: Optional[Dict[str, str]] = None,
        kenshis_metadata: Optional[List[Dict[str, Any]]] = None,
        session_segments_override: Optional[List[Dict[str, Any]]] = None
    ) -> TrainingSessionResult:
        """
        Executa a análise para múltiplos Kenshis (N >= 1) no modo dedicado de treinamento.
        Cada praticante é avaliado com base no seu tempo real de presença no enquadramento,
        identificando as sessões dentro da mesma sequência e concentrando os blocos por modalidade.
        """
        strikes_map = detected_strikes_by_kenshi or {}
        custom_names = custom_kendoka_names or {}
        meta_by_id = {m.get("kenshi_id"): m for m in (kenshis_metadata or [])}

        total_session_frames = max([len(h) for h in kenshi_histories.values()]) if kenshi_histories else 1
        duration_sec = max(1.0, total_session_frames / max(1.0, fps))

        # Determinar modalidade: usa o Kenshi com mais frames válidos como referência
        sorted_kenshis = sorted(
            kenshi_histories.keys(),
            key=lambda k: len([p for p in kenshi_histories[k] if p]),
            reverse=True
        )
        primary_k_id = sorted_kenshis[0] if sorted_kenshis else "KENSHI_01"
        primary_hist = kenshi_histories.get(primary_k_id, [])
        secondary_hist = kenshi_histories.get(sorted_kenshis[1], []) if len(sorted_kenshis) > 1 else None

        all_strikes_flat = []
        for s_list in strikes_map.values():
            all_strikes_flat.extend(s_list)

        if modality_override and modality_override in TRAINING_MODALITIES_METADATA:
            modality_key = modality_override
            confidence = 1.0
            det_method = "MANUAL_SELECT"
            is_override = True
            justification = f"Modalidade selecionada manualmente: {TRAINING_MODALITIES_METADATA[modality_key]['name']}."
        else:
            modality_key, confidence, justification = self.detect_training_modality(
                primary_history=primary_hist,
                secondary_history=secondary_hist,
                detected_strikes=all_strikes_flat,
                fps=fps,
                has_sonkyo=False
            )
            det_method = "AUTO_DETECTED"
            is_override = False

        modality_meta = TRAINING_MODALITIES_METADATA.get(modality_key, TRAINING_MODALITIES_METADATA["suburi"])
        modality_name = modality_meta["name"]
        modality_cat = modality_meta.get("category", "Geral")
        learned_principles, web_sources = self._get_modality_knowledge(modality_key)

        kendokas_list: List[KendokaTrainingProfile] = []
        default_names_map: Dict[str, str] = {}
        roles_map: Dict[str, str] = {}

        for idx, k_id in enumerate(sorted_kenshis):
            hist = kenshi_histories[k_id]
            valid_poses = [p for p in hist if p]
            meta = meta_by_id.get(k_id, {})

            presence_frames = meta.get("presence_frames", len(valid_poses))
            presence_sec = max(0.1, presence_frames / max(1.0, fps))
            reentry_count = meta.get("reentry_count", 0)
            presence_intervals = meta.get("presence_intervals", [])
            thumbnail = meta.get("thumbnail")

            def_name = f"Kenshi {k_id.replace('KENSHI_', '')}"
            c_name = custom_names.get(k_id, def_name)
            role_label = f"Praticante #{idx + 1}"
            k_strikes = strikes_map.get(k_id, [])

            default_names_map[k_id] = def_name
            roles_map[k_id] = role_label

            # Cálculo de pilares com duração justa baseada na presença
            pillars = self.calculate_pillar_metrics(
                pose_history=hist,
                strikes=k_strikes,
                modality_key=modality_key,
                fps=fps,
                presence_duration_seconds=presence_sec
            )

            strengths, improvements, exercises = self.generate_pedagogical_feedback(
                pillars,
                modality_key,
                c_name
            )

            rep_timeline = self._build_repetition_timeline(k_strikes, fps=fps)

            kendokas_list.append(KendokaTrainingProfile(
                kendoka_id=k_id,
                default_name=def_name,
                custom_name=c_name,
                role=role_label,
                pillars=pillars,
                strengths=strengths,
                improvements=improvements,
                recommended_exercises=exercises,
                repetition_timeline=rep_timeline,
                presence_frames=presence_frames,
                presence_seconds=presence_sec,
                presence_intervals=presence_intervals,
                reentry_count=reentry_count,
                thumbnail=thumbnail
            ))

        # Segmentação de Sub-Sessões na Sequência e Concentração em Blocos de Modalidade
        sub_sessions = self.segment_training_sequence(
            kenshi_histories=kenshi_histories,
            detected_strikes_by_kenshi=strikes_map,
            fps=fps,
            modality_override=modality_override,
            session_segments_override=session_segments_override,
            custom_kendoka_names=custom_names,
            kenshis_metadata=kenshis_metadata,
            has_sonkyo=False,
            default_names_by_id=default_names_map,
            roles_by_id=roles_map
        )
        modality_blocks = self.build_modality_blocks(
            sub_sessions=sub_sessions,
            kenshi_histories=kenshi_histories,
            detected_strikes_by_kenshi=strikes_map,
            global_kendokas=kendokas_list,
            fps=fps,
            custom_kendoka_names=custom_names,
            default_names_by_id=default_names_map,
            roles_by_id=roles_map
        )

        if modality_blocks and not is_override:
            dominant_block = max(modality_blocks, key=lambda b: b.total_duration_seconds)
            modality_key = dominant_block.modality_key
            modality_name = dominant_block.modality_name
            modality_cat = dominant_block.modality_category
            learned_principles = dominant_block.learned_principles or learned_principles
            web_sources = dominant_block.web_sources or web_sources

        if len(sub_sessions) > 1 or len(modality_blocks) > 1:
            seq_desc = "; ".join(
                f"{s.start_timestamp} a {s.end_timestamp} {s.modality_name}"
                for s in sub_sessions
            )
            blocks_desc = ", ".join(
                f"{b.modality_name} ({b.intervals_label})"
                for b in modality_blocks
            )
            summary = (
                f"Sequência de treinamento Multi-Kenshi com {len(sub_sessions)} sessões identificadas "
                f"({seq_desc}), concentradas em {len(modality_blocks)} bloco(s) de modalidade: {blocks_desc}. "
                f"Duração total: {duration_sec:.1f}s | Praticantes rastreáveis: {len(kendokas_list)}."
            )
        else:
            summary = (
                f"Sessão de treinamento de Kendo classificada como '{modality_name}' "
                f"(Duração: {duration_sec:.1f}s, Praticantes Rastreáveis: {len(kendokas_list)}). "
                f"Rastreamento Multi-Kenshi com Re-ID via Zekken, Tenugui, Tare lateral e Biometria."
            )

        return TrainingSessionResult(
            modality_key=modality_key,
            modality_name=modality_name,
            detection_confidence=confidence,
            detection_method=det_method,
            is_manual_override=is_override,
            duration_seconds=duration_sec,
            total_frames_analyzed=total_session_frames,
            kendokas=kendokas_list,
            session_summary_text=summary,
            justification=justification,
            learned_principles=learned_principles,
            modality_category=modality_cat,
            web_sources=web_sources,
            sub_sessions=sub_sessions,
            modality_blocks=modality_blocks
        )


def merge_kendoka_profiles_dict(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    merged_name: str
) -> Dict[str, Any]:
    """
    Mescla os dados e métricas analíticas de dois perfis de Kendocas (reidentificação manual).
    Consolida histórico de repetições, linha do tempo, tempos de presença, reentradas e recalcula os 3 Pilares.
    """
    merged = dict(p1)
    merged["custom_name"] = merged_name
    merged["display_name"] = merged_name

    # 1. Presença e Reentradas
    pres_1 = float(p1.get("presence_seconds", 0.0))
    pres_2 = float(p2.get("presence_seconds", 0.0))
    combined_pres_sec = round(pres_1 + pres_2, 1)
    merged["presence_seconds"] = combined_pres_sec

    frames_1 = int(p1.get("presence_frames", 0))
    frames_2 = int(p2.get("presence_frames", 0))
    merged["presence_frames"] = frames_1 + frames_2

    reentries_1 = int(p1.get("reentry_count", 0))
    reentries_2 = int(p2.get("reentry_count", 0))
    merged["reentry_count"] = reentries_1 + reentries_2 + 1

    intervals_1 = list(p1.get("presence_intervals", []))
    intervals_2 = list(p2.get("presence_intervals", []))
    merged["presence_intervals"] = sorted(
        intervals_1 + intervals_2,
        key=lambda iv: iv[0] if isinstance(iv, (list, tuple)) and len(iv) > 0 else 0
    )

    # 2. Linha do Tempo e Repetições
    def _extract_ts(item: Any) -> float:
        if isinstance(item, dict):
            if "timestamp_seconds" in item:
                try:
                    return float(item["timestamp_seconds"])
                except Exception:
                    pass
            ts_val = str(item.get("timestamp", "0"))
            clean = re.sub(r"[^\d.]", "", ts_val)
            try:
                return float(clean) if clean else 0.0
            except Exception:
                pass
        return 0.0

    raw_tl1 = list(p1.get("repetition_timeline", []))
    raw_tl2 = list(p2.get("repetition_timeline", []))
    combined_timeline = sorted(raw_tl1 + raw_tl2, key=_extract_ts)
    for i, itm in enumerate(combined_timeline):
        if isinstance(itm, dict):
            itm["repetition"] = i + 1
            itm["repetition_number"] = i + 1
    merged["repetition_timeline"] = combined_timeline

    reps_1 = int(p1.get("pillars", {}).get("total_repetitions", len(raw_tl1)))
    reps_2 = int(p2.get("pillars", {}).get("total_repetitions", len(raw_tl2)))
    total_reps = len(combined_timeline) if combined_timeline else (reps_1 + reps_2)

    # 3. Cadência e Desvio Padrão
    if len(combined_timeline) >= 2:
        ts_list = [_extract_ts(it) for it in combined_timeline]
        diffs = [ts_list[j+1] - ts_list[j] for j in range(len(ts_list)-1) if ts_list[j+1] >= ts_list[j]]
        cadence_std = round(float(np.std(diffs)), 2) if diffs else 0.0
    else:
        std1 = float(p1.get("pillars", {}).get("cadence_std_dev_seconds", 0.0))
        std2 = float(p2.get("pillars", {}).get("cadence_std_dev_seconds", 0.0))
        cadence_std = round((std1 + std2) / 2.0, 2)

    if combined_pres_sec > 0 and total_reps > 0:
        cadence_cpm = round((total_reps / combined_pres_sec) * 60.0, 1)
    else:
        cpm1 = float(p1.get("pillars", {}).get("cadence_cpm", 0.0))
        cpm2 = float(p2.get("pillars", {}).get("cadence_cpm", 0.0))
        cadence_cpm = round((cpm1 + cpm2) / 2.0, 1)

    # 4. Pilares (Médias Ponderadas por Repetições ou Presença)
    w1 = max(1.0, float(reps_1) if reps_1 > 0 else (pres_1 if pres_1 > 0 else 1.0))
    w2 = max(1.0, float(reps_2) if reps_2 > 0 else (pres_2 if pres_2 > 0 else 1.0))

    def _weighted(val1: float, val2: float) -> float:
        return round((float(val1) * w1 + float(val2) * w2) / (w1 + w2), 1)

    pil1 = p1.get("pillars", {})
    pil2 = p2.get("pillars", {})

    mov1 = float(pil1.get("movimentacao", pil1.get("forma", 0.0)))
    mov2 = float(pil2.get("movimentacao", pil2.get("forma", 0.0)))
    merged_mov = _weighted(mov1, mov2)

    prec1 = float(pil1.get("precisao", 0.0))
    prec2 = float(pil2.get("precisao", 0.0))
    merged_prec = _weighted(prec1, prec2)

    const1 = float(pil1.get("constancia", 0.0))
    const2 = float(pil2.get("constancia", 0.0))
    merged_const = _weighted(const1, const2)

    merged_overall = round(merged_mov * 0.40 + merged_prec * 0.35 + merged_const * 0.25, 1)

    # Submétricas
    merged_sub_mov = {}
    sub_m1 = pil1.get("movimentacao_submetrics", {})
    sub_m2 = pil2.get("movimentacao_submetrics", {})
    for k in set(list(sub_m1.keys()) + list(sub_m2.keys())):
        merged_sub_mov[k] = _weighted(sub_m1.get(k, 0.0), sub_m2.get(k, 0.0))

    merged_sub_prec = {}
    sub_p1 = pil1.get("precisao_submetrics", {})
    sub_p2 = pil2.get("precisao_submetrics", {})
    for k in set(list(sub_p1.keys()) + list(sub_p2.keys())):
        merged_sub_prec[k] = _weighted(sub_p1.get(k, 0.0), sub_p2.get(k, 0.0))

    merged_sub_const = {}
    sub_c1 = pil1.get("constancia_submetrics", {})
    sub_c2 = pil2.get("constancia_submetrics", {})
    for k in set(list(sub_c1.keys()) + list(sub_c2.keys())):
        merged_sub_const[k] = _weighted(sub_c1.get(k, 0.0), sub_c2.get(k, 0.0))

    merged["pillars"] = {
        "overall_score": merged_overall,
        "movimentacao": merged_mov,
        "forma": merged_mov,
        "precisao": merged_prec,
        "constancia": merged_const,
        "cadence_cpm": cadence_cpm,
        "total_repetitions": total_reps,
        "cadence_std_dev_seconds": cadence_std,
        "movimentacao_submetrics": merged_sub_mov,
        "precisao_submetrics": merged_sub_prec,
        "constancia_submetrics": merged_sub_const,
    }

    # 5. Forças, Melhorias e Exercícios
    s1 = list(p1.get("strengths", []))
    s2 = list(p2.get("strengths", []))
    merged["strengths"] = list(dict.fromkeys(s1 + s2))

    imp1 = list(p1.get("improvements", []))
    imp2 = list(p2.get("improvements", []))
    merged["improvements"] = list(dict.fromkeys(imp1 + imp2))

    ex1 = list(p1.get("recommended_exercises", []))
    ex2 = list(p2.get("recommended_exercises", []))
    ex_seen = set()
    merged_exercises = []
    for ex in (ex1 + ex2):
        if isinstance(ex, dict):
            ex_name = ex.get("name")
            if ex_name and ex_name not in ex_seen:
                ex_seen.add(ex_name)
                merged_exercises.append(ex)
        elif ex not in ex_seen:
            ex_seen.add(ex)
            merged_exercises.append(ex)
    merged["recommended_exercises"] = merged_exercises

    ins1 = list(p1.get("kinesthetic_insights", []))
    ins2 = list(p2.get("kinesthetic_insights", []))
    merged["kinesthetic_insights"] = list(dict.fromkeys(ins1 + ins2))

    dev1 = dict(p1.get("kinesthetic_deviations", {}))
    dev2 = dict(p2.get("kinesthetic_deviations", {}))
    merged["kinesthetic_deviations"] = {**dev1, **dev2}

    return merged


def differentiate_kendoka_names(
    kendokas: List[Dict[str, Any]],
    existing_id: str,
    source_id: str,
    base_name: str
) -> Tuple[str, str]:
    """
    Gera nomes diferenciados com sufixos sequenciais (Ex.: 'Nome (1)', 'Nome (2)')
    quando o usuário opta por NÃO mesclar os dados de kendocas com mesmo nome.
    """
    clean_base = re.sub(r"\s*\(\d+\)$", "", base_name).strip()

    used_numbers = set()
    existing_num = None

    for k_item in kendokas:
        c_name = str(k_item.get("custom_name") or k_item.get("default_name", "")).strip()
        m = re.match(r"^" + re.escape(clean_base) + r"\s*\((\d+)\)$", c_name, flags=re.IGNORECASE)
        if m:
            num = int(m.group(1))
            used_numbers.add(num)
            if k_item.get("kendoka_id") == existing_id:
                existing_num = num

    if existing_num is None:
        name_existing = f"{clean_base} (1)"
        used_numbers.add(1)
    else:
        name_existing = f"{clean_base} ({existing_num})"

    next_num = 2
    while next_num in used_numbers:
        next_num += 1

    name_source = f"{clean_base} ({next_num})"
    return name_existing, name_source


def get_kendoka_id_aliases(k_id: str, fallback_index: Optional[int] = None) -> List[str]:
    """
    Retorna as chaves equivalentes para o ID de um Kendoca:
    - Se k_id possui número explícito (ex: 'KENSHI_04'), retorna exclusivamente ['KENSHI_04', 'KENSHI_4'],
      jamais usando o índice da lista ordenada (evitando colisão com 'KENSHI_03').
    - Se k_id é simbólico (ex: 'KENSHI_SHIRO' ou 'KENSHI_AKA' no modo de dupla), inclui também
      o alias posicional correspondente ('KENSHI_01'/'KENSHI_1' ou 'KENSHI_02'/'KENSHI_2').
    """
    clean_id = (k_id or "").strip()
    if not clean_id and fallback_index is not None:
        clean_id = f"KENSHI_{fallback_index + 1:02d}"
    if not clean_id:
        return []
    aliases = [clean_id]
    m_num = re.match(r"^KENSHI_0*(\d+)$", clean_id, flags=re.IGNORECASE)
    if m_num:
        num = int(m_num.group(1))
        for cand in (f"KENSHI_{num:02d}", f"KENSHI_{num}"):
            if cand not in aliases:
                aliases.append(cand)
    elif fallback_index is not None:
        num_fb = fallback_index + 1
        for cand in (f"KENSHI_{num_fb:02d}", f"KENSHI_{num_fb}"):
            if cand not in aliases:
                aliases.append(cand)
    return aliases


def sync_kendoka_names_in_training_data(
    train_data: Dict[str, Any],
    custom_names: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Sincroniza nomes customizados dos Kendocas bidirecionalmente entre custom_names e
    train_data['kendokas'], train_data['sub_sessions'] e train_data['modality_blocks'].
    Usa estritamente os aliases do próprio kendoka_id (sem usar o índice da lista ordenada),
    evitando que a ordenação por tempo de presença sobrescreva nomes entre Kenshis distintos.
    """
    if custom_names is None:
        custom_names = {}
    if not isinstance(train_data, dict):
        return custom_names

    # Limpa eventuais entradas espúrias em custom_names onde um ID aponta para o nome padrão de outro ID
    for ck in list(custom_names.keys()):
        cv = (custom_names.get(ck) or "").strip()
        m_ck = re.match(r"^KENSHI_0*(\d+)$", ck.strip(), flags=re.IGNORECASE)
        m_cv = re.match(r"^(?:Kenshi|Kendoca)\s+0*(\d+)$", cv, flags=re.IGNORECASE)
        if m_ck and m_cv and int(m_ck.group(1)) != int(m_cv.group(1)):
            custom_names.pop(ck, None)

    def _apply_to_kendokas_list(k_list: Any, update_map: bool = False) -> None:
        if not isinstance(k_list, list):
            return
        for idx_k, k_item in enumerate(k_list):
            if not isinstance(k_item, dict):
                continue
            k_id = str(k_item.get("kendoka_id", "")).strip() or f"KENSHI_{idx_k + 1:02d}"
            def_n = str(k_item.get("default_name") or f"Kenshi {k_id.replace('KENSHI_', '')}").strip()
            aliases = get_kendoka_id_aliases(k_id, fallback_index=idx_k)

            new_name: Optional[str] = None
            for al in aliases:
                cand_val = custom_names.get(al)
                if cand_val and cand_val.strip():
                    new_name = cand_val.strip()
                    break

            if not new_name:
                raw_cust = str(k_item.get("custom_name") or "").strip()
                # Auto-cura caso custom_name tenha recebido o rótulo padrão de outro Kenshi
                m_kid = re.match(r"^KENSHI_0*(\d+)$", k_id, flags=re.IGNORECASE)
                m_rc = re.match(r"^(?:Kenshi|Kendoca)\s+0*(\d+)$", raw_cust, flags=re.IGNORECASE)
                if m_kid and m_rc and int(m_kid.group(1)) != int(m_rc.group(1)):
                    raw_cust = def_n
                new_name = raw_cust or def_n

            if new_name and new_name.strip():
                clean_n = new_name.strip()
                k_item["custom_name"] = clean_n
                k_item["display_name"] = f"{clean_n} ({def_n})" if clean_n != def_n else def_n
                if update_map:
                    for al in aliases:
                        custom_names[al] = clean_n

    _apply_to_kendokas_list(train_data.get("kendokas"), update_map=True)

    for ss in train_data.get("sub_sessions", []):
        if isinstance(ss, dict):
            _apply_to_kendokas_list(ss.get("kendokas"), update_map=False)

    for mb in train_data.get("modality_blocks", []):
        if isinstance(mb, dict):
            _apply_to_kendokas_list(mb.get("kendokas"), update_map=False)

    for gk in train_data.get("kendokas", []):
        if isinstance(gk, dict):
            for bd in gk.get("modality_breakdown", []):
                if isinstance(bd, dict):
                    if isinstance(bd.get("kendoka_profile"), dict):
                        _apply_to_kendokas_list([bd["kendoka_profile"]], update_map=False)
                    elif "custom_name" in bd or "kendoka_id" in bd:
                        _apply_to_kendokas_list([bd], update_map=False)

    return custom_names


def merge_kendokas_in_training_data(
    train_data: Dict[str, Any],
    target_id: str,
    source_id: str,
    merged_name: str
) -> Optional[Dict[str, Any]]:
    """
    Mescla o perfil source_id dentro de target_id em train_data['kendokas'],
    propagando também para train_data['sub_sessions'] e train_data['modality_blocks'].
    """
    if not isinstance(train_data, dict):
        return None

    def _merge_in_list(k_list: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        t_idx = None
        s_idx = None
        for idx, k_item in enumerate(k_list):
            if isinstance(k_item, dict):
                if k_item.get("kendoka_id") == target_id:
                    t_idx = idx
                elif k_item.get("kendoka_id") == source_id:
                    s_idx = idx
        if t_idx is not None and s_idx is not None and t_idx != s_idx:
            merged_p = merge_kendoka_profiles_dict(k_list[t_idx], k_list[s_idx], merged_name)
            new_l = []
            for idx, k_item in enumerate(k_list):
                if idx == t_idx:
                    new_l.append(merged_p)
                elif idx == s_idx:
                    continue
                else:
                    new_l.append(k_item)
            return merged_p, new_l
        elif s_idx is not None and t_idx is None:
            # Se no sub-bloco apenas source_id existia, converte para target_id com o nome mesclado
            conv = dict(k_list[s_idx])
            conv["kendoka_id"] = target_id
            conv["custom_name"] = merged_name
            conv["display_name"] = merged_name
            k_list[s_idx] = conv
            return conv, k_list
        return None, k_list

    main_list = train_data.get("kendokas", [])
    merged_main, updated_main = _merge_in_list(main_list)
    train_data["kendokas"] = updated_main

    for ss in train_data.get("sub_sessions", []):
        if isinstance(ss, dict) and isinstance(ss.get("kendokas"), list):
            _, ss["kendokas"] = _merge_in_list(ss["kendokas"])

    for mb in train_data.get("modality_blocks", []):
        if isinstance(mb, dict) and isinstance(mb.get("kendokas"), list):
            _, mb["kendokas"] = _merge_in_list(mb["kendokas"])

    merged_map = train_data.setdefault("merged_id_map", {})
    if isinstance(merged_map, dict):
        merged_map[source_id] = target_id
        # Se outros IDs apontavam para source_id, redireciona para target_id
        for k_m, v_m in list(merged_map.items()):
            if v_m == source_id:
                merged_map[k_m] = target_id

    return merged_main


def sync_kendoka_names_in_events(
    events: Optional[List[Dict[str, Any]]],
    custom_names: Optional[Dict[str, str]],
    merged_id_map: Optional[Dict[str, str]] = None
) -> None:
    """
    Sincroniza os nomes personalizados (e IDs mesclados) diretamente nos eventos
    de golpes da linha do tempo do vídeo (res['events']).
    """
    if not events or not isinstance(events, list):
        return
    names_map = custom_names or {}
    m_map = merged_id_map or {}
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ev_info = ev.get("event_info")
        if not isinstance(ev_info, dict):
            continue
        raw_id = str(ev_info.get("attacker_id") or "").strip()
        if not raw_id:
            continue
        eff_id = m_map.get(raw_id, raw_id)
        ev_info["attacker_id"] = eff_id
        resolved_name = None
        for al in get_kendoka_id_aliases(eff_id):
            val = names_map.get(al)
            if val and val.strip():
                resolved_name = val.strip()
                break
        if resolved_name:
            ev_info["attacker_name"] = resolved_name
            st_type = ev_info.get("type", "MEN")
            ts_imp = ev_info.get("timestamp_impact", ev_info.get("timestamp", "00:00"))
            ev["diagnostic_report"] = f"🥋 Golpe {st_type} executado por {resolved_name} ({ts_imp})."


def recalculate_blocks_from_segments(
    train_data: Optional[Dict[str, Any]] = None,
    session_segments: Optional[List[Dict[str, Any]]] = None,
    fps: float = 30.0,
    training_data: Optional[Dict[str, Any]] = None,
    segments_override: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Recalcula sub_sessions e modality_blocks em um dicionário train_data já existente
    com base nos intervalos de sessão informados pelo usuário (ex.: 00:00 a 03:00 Kirikaeshi,
    03:00 a 05:00 Waza-geiko), separando repetições e recalculando cadência, Constância
    e feedback pedagógico por bloco de modalidade.
    """
    if train_data is None:
        train_data = training_data if isinstance(training_data, dict) else {}
    if session_segments is None:
        session_segments = segments_override or []
    if not isinstance(train_data, dict) or not session_segments:
        return train_data or {}

    fps_safe = max(1.0, float(fps or 30.0))
    total_dur = max(1.0, float(train_data.get("duration_seconds", 60.0)))
    global_kendokas = train_data.get("kendokas", [])
    analyzer = TrainingAnalyzer()

    def _extract_ts_sec(rep_item: Dict[str, Any], idx_fallback: int = 0) -> float:
        if "timestamp_seconds" in rep_item and rep_item["timestamp_seconds"] is not None:
            try:
                return float(rep_item["timestamp_seconds"])
            except Exception:
                pass
        if "impact_frame" in rep_item and rep_item["impact_frame"] is not None:
            try:
                return float(rep_item["impact_frame"]) / fps_safe
            except Exception:
                pass
        return parse_training_mmss(rep_item.get("timestamp", f"{idx_fallback * 2}s"), default=idx_fallback * 2.0)

    def _eval_kendoka_in_intervals(
        gk_dict: Dict[str, Any],
        intervals_sec: List[Tuple[float, float]],
        mod_key: str
    ) -> Dict[str, Any]:
        base_pil = gk_dict.get("pillars", {})
        mov_score = float(base_pil.get("movimentacao", base_pil.get("forma", 78.0)))
        prec_score = float(base_pil.get("precisao", 76.0))
        mov_sub = dict(base_pil.get("movimentacao_submetrics", {}))
        prec_sub = dict(base_pil.get("precisao_submetrics", {}))
        base_const_sub = dict(base_pil.get("constancia_submetrics", {}))

        raw_tl = gk_dict.get("repetition_timeline", [])
        filtered_reps: List[Dict[str, Any]] = []
        rep_times: List[float] = []
        for idx_r, r_item in enumerate(raw_tl):
            if not isinstance(r_item, dict):
                continue
            t_s = _extract_ts_sec(r_item, idx_r)
            for iv_s, iv_e in intervals_sec:
                if iv_s <= t_s < iv_e or (abs(iv_e - total_dur) < 0.5 and abs(t_s - iv_e) <= 0.5):
                    r_copy = dict(r_item)
                    r_copy["timestamp_seconds"] = round(t_s, 2)
                    filtered_reps.append(r_copy)
                    rep_times.append(t_s)
                    break

        for i_r, r_c in enumerate(filtered_reps, start=1):
            r_c["repetition"] = i_r

        iv_dur = max(0.1, sum(max(0.1, e - s) for s, e in intervals_sec))
        reps_count = len(filtered_reps)
        # Se a timeline original estava vazia mas total_repetitions > 0, estima proporcionalmente à duração
        if not raw_tl and int(base_pil.get("total_repetitions", 0)) > 0:
            reps_count = max(1, round(int(base_pil.get("total_repetitions", 0)) * (iv_dur / max(1.0, total_dur))))

        cpm = round((reps_count / max(0.1, iv_dur / 60.0)), 1)
        if len(rep_times) >= 3:
            diffs = [abs(rep_times[i] - rep_times[i - 1]) for i in range(1, len(rep_times))]
            std_dev = round(float(np.std(diffs)), 3)
            mean_d = float(np.mean(diffs))
            cv = std_dev / max(0.1, mean_d)
            ritmo_reg = round(max(0.0, 1.0 - cv) * 100.0, 1)
        else:
            std_dev = float(base_pil.get("cadence_std_dev_seconds", 0.35))
            ritmo_reg = float(base_const_sub.get("regularidade_ritmo", 80.0))

        fadiga_score = float(base_const_sub.get("resistencia_fadiga", 85.0))
        expected_range = TRAINING_MODALITIES_METADATA.get(mod_key, {}).get("expected_cadence_cpm", (20, 60))
        if expected_range[0] <= cpm <= expected_range[1]:
            cad_score = 95.0
        elif cpm < expected_range[0]:
            cad_score = max(50.0, 95.0 - (expected_range[0] - cpm) * 3.0)
        else:
            cad_score = max(60.0, 95.0 - (cpm - expected_range[1]) * 1.5)

        const_final = round((ritmo_reg * 0.45) + (fadiga_score * 0.35) + (cad_score * 0.20), 1)
        const_sub = {
            "regularidade_ritmo": round(ritmo_reg, 1),
            "resistencia_fadiga": round(fadiga_score, 1),
            "adequacao_cadencia": round(cad_score, 1)
        }

        pillars_obj = TrainingPillarMetrics(
            movimentacao_score=mov_score,
            precisao_score=prec_score,
            constancia_score=const_final,
            movimentacao_submetrics=mov_sub,
            precisao_submetrics=prec_sub,
            constancia_submetrics=const_sub,
            cadence_cpm=cpm,
            cadence_std_dev_seconds=std_dev,
            total_repetitions=reps_count
        )

        k_id = gk_dict.get("kendoka_id", "KENSHI_01")
        def_n = gk_dict.get("default_name", k_id)
        c_n = gk_dict.get("custom_name", def_n)
        role_n = gk_dict.get("role", "Praticante")
        str_l, imp_l, exe_l = analyzer.generate_pedagogical_feedback(pillars_obj, mod_key, c_n)

        prof_obj = KendokaTrainingProfile(
            kendoka_id=k_id,
            default_name=def_n,
            custom_name=c_n,
            role=role_n,
            pillars=pillars_obj,
            strengths=str_l,
            improvements=imp_l,
            recommended_exercises=exe_l,
            repetition_timeline=filtered_reps,
            presence_frames=round(iv_dur * fps_safe),
            presence_seconds=round(iv_dur, 1),
            presence_intervals=[(round(s * fps_safe), round(e * fps_safe)) for s, e in intervals_sec]
        )
        return prof_obj.to_dict()

    # 1. Construir sub_sessions
    sub_sessions_dicts: List[Dict[str, Any]] = []
    for idx_seg, seg_spec in enumerate(session_segments, start=1):
        s_sec = (
            float(seg_spec["start_seconds"])
            if "start_seconds" in seg_spec and seg_spec["start_seconds"] is not None
            else parse_training_mmss(
                seg_spec.get("start_timestamp", seg_spec.get("start_time", seg_spec.get("start", "00:00"))),
                default=0.0
            )
        )
        e_sec = (
            float(seg_spec["end_seconds"])
            if "end_seconds" in seg_spec and seg_spec["end_seconds"] is not None
            else parse_training_mmss(
                seg_spec.get("end_timestamp", seg_spec.get("end_time", seg_spec.get("end", format_training_mmss(total_dur)))),
                default=total_dur
            )
        )
        if e_sec <= s_sec:
            e_sec = max(s_sec + 1.0, total_dur)

        mod_k = str(seg_spec.get("modality_key", train_data.get("modality_key", "suburi"))).strip()
        if mod_k not in TRAINING_MODALITIES_METADATA:
            mod_k = train_data.get("modality_key", "suburi")
        mod_meta = TRAINING_MODALITIES_METADATA.get(mod_k, TRAINING_MODALITIES_METADATA["suburi"])
        mod_name = mod_meta["name"]
        mod_cat = mod_meta.get("category", "Geral")

        seg_k_dicts = []
        seg_reps = 0
        for gk in global_kendokas:
            if isinstance(gk, dict):
                kd = _eval_kendoka_in_intervals(gk, [(s_sec, e_sec)], mod_k)
                seg_reps += int(kd.get("pillars", {}).get("total_repetitions", 0))
                seg_k_dicts.append(kd)

        ss_obj = TrainingSubSession(
            segment_index=idx_seg,
            start_frame=round(s_sec * fps_safe),
            end_frame=round(e_sec * fps_safe),
            start_seconds=s_sec,
            end_seconds=e_sec,
            start_timestamp=seg_spec.get("start_timestamp") or format_training_mmss(s_sec),
            end_timestamp=seg_spec.get("end_timestamp") or format_training_mmss(e_sec),
            modality_key=mod_k,
            modality_name=mod_name,
            modality_category=mod_cat,
            detection_confidence=float(seg_spec.get("detection_confidence", 1.0)),
            detection_method=str(seg_spec.get("detection_method", "MANUAL_SELECT")),
            justification=seg_spec.get("justification") or f"Sessão de {format_training_mmss(s_sec)} a {format_training_mmss(e_sec)} classificada como {mod_name}.",
            total_repetitions=seg_reps,
            kendokas=seg_k_dicts
        )
        sub_sessions_dicts.append(ss_obj.to_dict())

    # 2. Concentrar em modality_blocks
    grouped_ss: Dict[str, List[Dict[str, Any]]] = {}
    mod_order: List[str] = []
    for ss_d in sub_sessions_dicts:
        mk = ss_d["modality_key"]
        if mk not in grouped_ss:
            grouped_ss[mk] = []
            mod_order.append(mk)
        grouped_ss[mk].append(ss_d)

    modality_blocks_dicts: List[Dict[str, Any]] = []
    breakdown_by_kid: Dict[str, List[Dict[str, Any]]] = {}

    for b_idx, mk in enumerate(mod_order, start=1):
        ss_group = grouped_ss[mk]
        mod_meta = TRAINING_MODALITIES_METADATA.get(mk, TRAINING_MODALITIES_METADATA["suburi"])
        mod_name = mod_meta["name"]
        mod_cat = mod_meta.get("category", "Geral")
        learned_p, web_s = analyzer._get_modality_knowledge(mk)

        intervals_list = []
        intervals_sec_tuples: List[Tuple[float, float]] = []
        b_dur = 0.0
        b_reps = 0
        for ssd in ss_group:
            intervals_list.append({
                "segment_index": ssd["segment_index"],
                "start_frame": ssd["start_frame"],
                "end_frame": ssd["end_frame"],
                "start_seconds": ssd["start_seconds"],
                "end_seconds": ssd["end_seconds"],
                "start_timestamp": ssd["start_timestamp"],
                "end_timestamp": ssd["end_timestamp"],
                "duration_seconds": ssd["duration_seconds"],
                "total_repetitions": ssd["total_repetitions"]
            })
            intervals_sec_tuples.append((ssd["start_seconds"], ssd["end_seconds"]))
            b_dur += float(ssd["duration_seconds"])
            b_reps += int(ssd["total_repetitions"])

        iv_label = ", ".join(f"{iv['start_timestamp']} a {iv['end_timestamp']}" for iv in intervals_list)

        blk_kendokas = []
        for gk in global_kendokas:
            if isinstance(gk, dict):
                kd_blk = _eval_kendoka_in_intervals(gk, intervals_sec_tuples, mk)
                blk_kendokas.append(kd_blk)
                kid = gk.get("kendoka_id", "")
                breakdown_by_kid.setdefault(kid, []).append({
                    "block_index": b_idx,
                    "modality_key": mk,
                    "modality_name": mod_name,
                    "modality_category": mod_cat,
                    "intervals_label": iv_label,
                    "total_duration_seconds": round(b_dur, 2),
                    "kendoka_profile": kd_blk
                })

        mb_obj = TrainingModalityBlock(
            block_index=b_idx,
            modality_key=mk,
            modality_name=mod_name,
            modality_category=mod_cat,
            description=mod_meta.get("description", ""),
            focus_areas=list(mod_meta.get("focus_areas", [])),
            learned_principles=learned_p,
            web_sources=web_s,
            intervals=intervals_list,
            intervals_label=iv_label,
            total_duration_seconds=b_dur,
            total_repetitions=b_reps,
            detection_confidence=1.0,
            detection_method="MANUAL_SELECT",
            justification=f"Bloco concentrado da modalidade {mod_name} ({iv_label}).",
            kendokas=blk_kendokas
        )
        modality_blocks_dicts.append(mb_obj.to_dict())

    for gk in global_kendokas:
        if isinstance(gk, dict):
            gk["modality_breakdown"] = breakdown_by_kid.get(gk.get("kendoka_id", ""), [])

    train_data["sub_sessions"] = sub_sessions_dicts
    train_data["modality_blocks"] = modality_blocks_dicts

    if modality_blocks_dicts:
        dom = max(modality_blocks_dicts, key=lambda b: float(b.get("total_duration_seconds", 0.0)))
        train_data["modality_key"] = dom["modality_key"]
        train_data["modality_name"] = dom["modality_name"]
        train_data["modality_category"] = dom["modality_category"]
        seq_str = "; ".join(f"{s['start_timestamp']} a {s['end_timestamp']} {s['modality_name']}" for s in sub_sessions_dicts)
        blk_str = ", ".join(f"{b['modality_name']} ({b['intervals_label']})" for b in modality_blocks_dicts)
        train_data["session_summary_text"] = (
            f"Sequência de treinamento com {len(sub_sessions_dicts)} sessões definidas "
            f"({seq_str}), concentradas em {len(modality_blocks_dicts)} bloco(s) de modalidade: {blk_str}."
        )

    return train_data



