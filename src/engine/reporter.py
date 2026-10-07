"""
Gerador de Feedback & Diagnóstico Técnico e Avaliação em Kendo.
Transforma as métricas técnicas e de calibração em relatórios e críticas construtivas em português.
"""

from typing import Dict, Any

class DiagnosticReporter:
    STRIKE_NAMES = {
        "MEN": "メ MEN",
        "KOTE": "コ KOTE",
        "DO": "ド DO",
        "TSUKI": "ツ TSUKI",
        "メ MEN": "メ MEN",
        "コ KOTE": "コ KOTE",
        "ド DO": "ド DO",
        "ツ TSUKI": "ツ TSUKI",
    }

    @staticmethod
    def format_strike_name(strike_type: str) -> str:
        """Formata o nome da técnica/golpe com Katakana para exibição."""
        if not strike_type:
            return ""
        st_clean = str(strike_type).strip()
        return DiagnosticReporter.STRIKE_NAMES.get(st_clean.upper(), DiagnosticReporter.STRIKE_NAMES.get(st_clean, st_clean))

    @staticmethod
    def generate_strike_report(event_info: Dict[str, Any], evaluation: Dict[str, Any], fumikomi_offset_ms: float) -> str:
        """
        Gera a crítica textual detalhada para um golpe analisado.
        """
        raw_strike_type = event_info.get("type", "MEN")
        strike_type = DiagnosticReporter.format_strike_name(raw_strike_type)
        timestamp = event_info["timestamp"]
        attacker_name = event_info.get("attacker_name", "Kenshi Aka")
        is_valid = evaluation["is_valid"]
        score = evaluation["total_score"]
        min_req = evaluation["min_required"]
        sub = evaluation["sub_scores"]
        
        status_symbol = "✅ GOLPE VÁLIDO (IPPON)" if is_valid else "❌ GOLPE INVÁLIDO"

        lines = []
        lines.append(f"### Golpe: {strike_type} ({timestamp}) - {attacker_name} - Status: {status_symbol}")
        lines.append(f"**Pontuação Geral:** {score}% (Limiar Exigido: {min_req}%)")
        lines.append("")
        lines.append("**Detalhamento dos Critérios de Ki-Ken-Tai-Ichi:**")
        
        # 1. Alvo / Impacto
        if sub["target_impact"] >= 70:
            lines.append(f"- ✅ **Impacto no Alvo ({sub['target_impact']}%):** Excelente precisão no ponto de contato do {strike_type}.")
        else:
            lines.append(f"- ⚠️ **Impacto no Alvo ({sub['target_impact']}%):** O Shinai não atingiu o centro exato da região do {strike_type}.")

        # 2. Fumikomi / Sincronismo Pé-Mão
        if sub["fumikomi_sync"] >= 65:
            lines.append(f"- ✅ **Fumikomi-ashi / Sincronismo ({sub['fumikomi_sync']}%):** Mãos e pé direito atingiram o solo em perfeito alinhamento.")
        else:
            deslocamento = f"{abs(fumikomi_offset_ms):.0f}ms " + ("após" if fumikomi_offset_ms > 0 else "antes")
            lines.append(f"- ⚠️ **Fumikomi-ashi ({sub['fumikomi_sync']}%):** Desalinhamento no Ki-Ken-Tai-Ichi. O pé direito bateu {deslocamento} do impacto da mão.")

        # 3. Postura
        if sub["posture"] >= 65:
            lines.append(f"- ✅ **Postura Corporal ({sub['posture']}%):** Tronco alinhado e verticalidade preservada durante o golpe.")
        else:
            lines.append(f"- ⚠️ **Postura Corporal ({sub['posture']}%):** Houve inclinação excessiva do tronco para frente no momento do impacto.")

        # 4. Zanshin
        if sub["zanshin"] >= 60:
            lines.append(f"- ✅ **Zanshin ({sub['zanshin']}%):** Praticante manteve alerta e controle de guarda após a execução.")
        else:
            lines.append(f"- ⚠️ **Zanshin ({sub['zanshin']}%):** Perda de postura ou desaceleração abrupta imediatamente após o golpe.")

        # 5. Hasuji (5° Pilar do Eixo 3.2)
        if "hasuji" in sub:
            h_score = sub["hasuji"]
            if h_score >= 65:
                lines.append(f"- ✅ **Hasuji / Ângulo da Lâmina ({h_score}%):** Corte alinhado com o fio do Shinai no plano regulamentar.")
            else:
                lines.append(f"- ⚠️ **Hasuji / Ângulo da Lâmina ({h_score}%):** Lâmina desalinhada; golpe desferido com a lateral ou costas do Shinai.")

        # Seme e Contrataque (Eixo 3.3 e 3.4)
        if "seme" in sub:
            s_score = sub["seme"]
            if s_score >= 60:
                lines.append(f"- ✅ **Seme / Pressão Pré-Golpe ({s_score}%):** Domínio do centro (Chushin-sen) e iniciativa ofensiva.")
            else:
                lines.append(f"- ⚠️ **Seme / Pressão Pré-Golpe ({s_score}%):** Falta de pressão prévia ou ataque desferido a partir de recuo.")

        if event_info.get("is_counterattack"):
            lines.append(f"- ⚔️ **Técnica de Contrataque:** Reconhecido como {event_info.get('counterattack_type', 'Oji-waza')}.")

        # Perspectiva e Enquadramento da Câmera (Eixo 5)
        if "camera_angle_deg" in event_info:
            c_angle = event_info["camera_angle_deg"]
            c_cat = event_info.get("camera_category", "LATERAL")
            c_qual = event_info.get("camera_quality_score", 85.0)
            lines.append(f"- 📹 **Enquadramento de Câmera ({c_cat} ~{c_angle:.0f}° - Qualidade {c_qual:.0f}%):** Compensação geométrica de perspectiva aplicada.")

        # Modelagem do Estilo Individual do Kenshi (Eixo 6)
        if "kinesthetic_insights" in event_info and event_info["kinesthetic_insights"]:
            for ins in event_info["kinesthetic_insights"]:
                lines.append(f"- 🧬 **Estilo Individual:** {ins}")

        lines.append("")
        if not is_valid:
            lines.append("**O que faltou para o Ponto Válido:**")
            failed = evaluation.get("failed_subcriteria", [])
            if "KU_TOTSU_VAZIO" in failed:
                lines.append(" 1. Golpe no vazio (Ku-totsu): Fora do alcance de combate (Maai) sem contato com o oponente.")
            if "ALVO_FORA" in failed or sub["target_impact"] < 60:
                lines.append(" 2. Focar o contato com a parte correta do Shinai (Datotsu-bu) no centro do alvo.")
            if "SEM_FUMIKOMI" in failed or sub["fumikomi_sync"] < 60:
                lines.append(" 3. Sincronizar a pisada forte do pé direito exatamente no instante do corte.")
            if "POSTURA_INCLINADA" in failed or sub["posture"] < 60:
                lines.append(" 4. Manter a coluna ereta sem projetar o ombro excessivamente à frente.")
            if "SEM_ZANSHIN" in failed or sub["zanshin"] < 50:
                lines.append(" 5. Sustentar a guarda e prontidão (Zanshin) após ultrapassar/finalizar o golpe.")
            if "HASUJI_INCORRETO" in failed:
                lines.append(" 6. Corrigir o Hasuji: Garantir que o plano da lâmina corresponda ao ângulo de corte regulamentar.")

        return "\n".join(lines)
