"""
Gerenciador e Motor de Avaliação de Treinamento e Aprendizado de Kendo em Tempo Real (SenpAI).
Processa fluxos de vídeo ao vivo (Webcam e RTSP/IP), executando:
- Identificação contínua ou guiada das 14 modalidades oficiais de treinamento de Kendo (com Kanji);
- Rastreamento biomecânico e cálculo dinâmico dos 3 Pilares (Movimentação, Precisão, Constância);
- Contagem automática de repetições / cortes executados com cadência rítmica (CPM);
- Feedbacks pedagógicos e alertas posturais instantâneos (Shisei, ombros, calcanhar);
- Consolidação e geração de relatórios de sessão ao vivo (Markdown e JSON).
"""

import time
import html
import math
from typing import Dict, List, Any, Optional, Tuple, Sequence

from src.analytics.training_analyzer import (
    TrainingAnalyzer,
    TRAINING_MODALITIES_METADATA,
    TrainingPillarMetrics
)


class LiveTrainingSessionManager:
    """
    Gerenciador de estado e avaliação para sessões de Treinamento de Kendo em Tempo Real.
    """

    def __init__(
        self,
        modality_override: Optional[str] = None,
        kendoka_name: str = "Kendoka Praticante",
        target_dan: int = 3,
        training_analyzer: Optional[TrainingAnalyzer] = None
    ):
        self.analyzer = training_analyzer or TrainingAnalyzer()
        self.modality_override = modality_override if modality_override and modality_override != "auto" else None
        self.kendoka_name = kendoka_name.strip() or "Kendoka Praticante"
        self.target_dan = target_dan

        self.start_time = time.time()
        self.frame_count = 0
        self.rep_count = 0
        self.rep_history: List[Dict[str, Any]] = []

        # Modalidade
        self.current_modality_key = self.modality_override or "suburi"
        self.modality_detection_conf = 0.90 if self.modality_override else 0.50
        self.is_auto_detected = (self.modality_override is None)

        # Pilares
        self.movement_score = 75.0
        self.precision_score = 75.0
        self.constancy_score = 75.0
        self.overall_score = 75.0
        self.pillar_records: List[Tuple[float, float, float]] = []

        # Estado da máquina de repetição (IDLE -> FURIKABURI -> STRIKE -> RECOVERY)
        self._strike_phase = "IDLE"
        self._last_strike_timestamp = 0.0
        self._last_wrist_y = 0.5
        self._furikaburi_peak_y = 0.5

        # Feedbacks
        self.latest_feedback = "🥋 Mantenha a postura Chudan Kamae estável e inicie as repetições."
        self.feedback_severity = "info"  # "success", "warning", "info"

    def process_live_frame(
        self,
        live_pose_histories: List[List[Optional[Dict[str, Any]]]],
        fps: float = 30.0,
        current_frame_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Processa um passo do fluxo ao vivo para detecção de repetições,
        cálculo dos 3 Pilares e geração de feedback pedagógico.
        """
        self.frame_count += 1
        elapsed = max(0.001, time.time() - self.start_time)
        fps = fps if fps > 0 else 30.0

        # Identificar landmarks primários (da câmera 1 ou primeira câmera com pose)
        primary_history = live_pose_histories[0] if live_pose_histories and len(live_pose_histories) > 0 else []
        sec_history = live_pose_histories[1] if len(live_pose_histories) > 1 else []
        active_landmarks: Optional[Dict[str, Any]] = None
        for p in reversed(primary_history):
            if p:
                active_landmarks = p
                break

        # 1. Identificação periódica da modalidade caso seja seleção automática (a cada 40 quadros)
        if self.is_auto_detected and self.frame_count % 40 == 0 and len(primary_history) >= 20:
            try:
                m_key, m_conf, _ = self.analyzer.detect_training_modality(
                    primary_history=primary_history[-90:],
                    secondary_history=sec_history[-90:] if sec_history else [],
                    fps=fps
                )
                self.current_modality_key = m_key
                self.modality_detection_conf = m_conf
            except Exception:
                pass

        # 2. Avaliação biomecânica do frame ativo e transições de corte/repetição
        new_rep_detected = False
        rep_event_info = None

        if active_landmarks:
            # Coletar pontos chave
            rw = active_landmarks.get("RIGHT_WRIST")
            lw = active_landmarks.get("LEFT_WRIST")
            rs = active_landmarks.get("RIGHT_SHOULDER")
            ls = active_landmarks.get("LEFT_SHOULDER")
            rh = active_landmarks.get("RIGHT_HIP")
            lh = active_landmarks.get("LEFT_HIP")
            la = active_landmarks.get("LEFT_ANKLE")

            # Média da altura dos punhos e dos ombros (coordenadas Y normalizadas: 0 é o topo, 1 é o piso)
            if rw and lw and rs and ls:
                wrist_y = (rw["y"] + lw["y"]) / 2.0
                shoulder_y = (rs["y"] + ls["y"]) / 2.0
                shoulder_x = (rs["x"] + ls["x"]) / 2.0
                hip_x = ((rh["x"] + lh["x"]) / 2.0) if (rh and lh) else shoulder_x

                # 2.1 Análise postural instantânea (Shisei e Nivelamento dos Ombros)
                tilt_posture = abs(shoulder_x - hip_x)
                shoulder_diff_y = abs(rs["y"] - ls["y"])

                # Postura da coluna (Shisei)
                posture_frame_score = max(0.0, min(100.0, (1.0 - tilt_posture * 4.0) * 100.0))
                # Nivelamento dos ombros
                level_frame_score = max(0.0, min(100.0, (1.0 - shoulder_diff_y * 5.0) * 100.0))

                instant_mov = (posture_frame_score * 0.6) + (level_frame_score * 0.4)

                # Máquina de estados de golpe / repetição de treino
                now_sec = time.time()
                time_since_last_strike = now_sec - self._last_strike_timestamp

                # FASE 1: FURIKABURI (Elevação dos braços acima dos ombros / cabeça)
                if wrist_y < (shoulder_y - 0.08):
                    if self._strike_phase != "FURIKABURI":
                        self._strike_phase = "FURIKABURI"
                        self._furikaburi_peak_y = wrist_y
                    else:
                        self._furikaburi_peak_y = min(self._furikaburi_peak_y, wrist_y)

                # FASE 2: UCHI / IMPACTO (Descida rápida após furikaburi)
                elif self._strike_phase == "FURIKABURI" and wrist_y >= (shoulder_y - 0.02):
                    if time_since_last_strike >= 0.35:
                        self.rep_count += 1
                        self._last_strike_timestamp = now_sec
                        self._strike_phase = "RECOVERY"
                        new_rep_detected = True

                        ts_formatted = self._format_timestamp(elapsed)
                        rep_quality = int(instant_mov * 0.5 + 45.0)

                        # Feedback pontual
                        if tilt_posture < 0.04 and shoulder_diff_y < 0.03:
                            self.latest_feedback = f"✨ Repetição #{self.rep_count}: Excelente Shisei ereto e ombros nivelados ({rep_quality}%)."
                            self.feedback_severity = "success"
                            eval_badge = "✅ Excelente"
                        elif tilt_posture >= 0.07:
                            self.latest_feedback = f"⚠️ Repetição #{self.rep_count}: Atenção à coluna inclinada para a frente. Mantenha o Shisei vertical!"
                            self.feedback_severity = "warning"
                            eval_badge = "⚠️ Inclinação"
                        elif shoulder_diff_y >= 0.05:
                            self.latest_feedback = f"⚠️ Repetição #{self.rep_count}: Ombro desalinhado no corte. Firme o Tenouchi de ambas as mãos!"
                            self.feedback_severity = "warning"
                            eval_badge = "⚠️ Ombros Desiguais"
                        else:
                            self.latest_feedback = f"🎯 Repetição #{self.rep_count}: Golpe completado com bom alinhamento ({rep_quality}%)."
                            self.feedback_severity = "info"
                            eval_badge = "🎯 Bom Ritmo"

                        rep_event_info = {
                            "rep_number": self.rep_count,
                            "timestamp": ts_formatted,
                            "elapsed_seconds": round(elapsed, 2),
                            "quality_score": rep_quality,
                            "posture_score": int(posture_frame_score),
                            "status": eval_badge,
                            "feedback": self.latest_feedback
                        }
                        self.rep_history.insert(0, rep_event_info)

                # FASE 3: RETORNO AO KAMAE
                elif self._strike_phase == "RECOVERY" and wrist_y >= shoulder_y:
                    if time_since_last_strike >= 0.2:
                        self._strike_phase = "IDLE"

        # 3. Recalcular métricas consolidadas dos 3 Pilares periodicamente (a cada 15 quadros)
        if self.frame_count % 15 == 0 and len(primary_history) >= 15:
            try:
                pm = self.analyzer.calculate_pillar_metrics(
                    pose_history=primary_history[-90:],
                    strikes=[],
                    modality_key=self.current_modality_key,
                    fps=fps
                )
                self.movement_score = float(pm.movement_score)
                self.precision_score = float(pm.precision_score)

                # Constância baseada na regularidade do intervalo entre as repetições
                cadence_cpm = self.get_current_cadence_cpm()
                expected_min, expected_max = self.get_expected_cadence_range()
                if self.rep_count >= 2 and cadence_cpm > 5.0:
                    if expected_min <= cadence_cpm <= expected_max:
                        const_sc = 85.0 + min(15.0, (cadence_cpm - expected_min) * 0.5)
                    else:
                        dist = min(abs(cadence_cpm - expected_min), abs(cadence_cpm - expected_max))
                        const_sc = max(40.0, 80.0 - dist * 1.5)
                else:
                    const_sc = float(pm.constancy_score)

                self.constancy_score = float(const_sc)
                self.overall_score = round(
                    (self.movement_score * 0.40) +
                    (self.precision_score * 0.35) +
                    (self.constancy_score * 0.25),
                    1
                )
                self.pillar_records.append((self.movement_score, self.precision_score, self.constancy_score))
            except Exception:
                pass

        meta = TRAINING_MODALITIES_METADATA.get(self.current_modality_key, TRAINING_MODALITIES_METADATA["suburi"])

        return {
            "modality_key": self.current_modality_key,
            "modality_name": meta.get("name", "Suburi"),
            "modality_category": meta.get("category", "Treino de Kendo"),
            "modality_confidence": int(self.modality_detection_conf * 100),
            "is_auto_detected": self.is_auto_detected,
            "rep_count": self.rep_count,
            "cadence_cpm": round(self.get_current_cadence_cpm(), 1),
            "movement_score": round(self.movement_score, 1),
            "precision_score": round(self.precision_score, 1),
            "constancy_score": round(self.constancy_score, 1),
            "overall_score": round(self.overall_score, 1),
            "latest_feedback": self.latest_feedback,
            "feedback_severity": self.feedback_severity,
            "new_rep_detected": new_rep_detected,
            "rep_event": rep_event_info,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_formatted": self._format_timestamp(elapsed)
        }

    def get_current_cadence_cpm(self) -> float:
        """Calcula a taxa de repetições por minuto (CPM) atual."""
        elapsed_min = max(0.05, (time.time() - self.start_time) / 60.0)
        return float(self.rep_count / elapsed_min)

    def get_expected_cadence_range(self) -> Tuple[float, float]:
        """Obtém a faixa esperada de cadência (CPM) para a modalidade atual."""
        meta = TRAINING_MODALITIES_METADATA.get(self.current_modality_key, {})
        expected = meta.get("expected_cadence_cpm", (20, 60))
        return float(expected[0]), float(expected[1])

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Converte segundos para MM:SS.mmm."""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{mins:02d}:{secs:02d}.{millis:03d}"

    def render_live_hud_html(self) -> str:
        """Gera o HTML estilizado do HUD de Treinamento ao Vivo para o painel."""
        meta = TRAINING_MODALITIES_METADATA.get(self.current_modality_key, TRAINING_MODALITIES_METADATA["suburi"])
        mod_name = meta.get("name", "Suburi (素振り)")
        mod_cat = meta.get("category", "Golpes no Ar")
        cpm = self.get_current_cadence_cpm()

        # Cores para cada pilar
        def get_pill_color(val: float) -> str:
            if val >= 80:
                return "#22C55E"
            elif val >= 65:
                return "#38BDF8"
            elif val >= 50:
                return "#F59E0B"
            return "#EF4444"

        col_mov = get_pill_color(self.movement_score)
        col_prec = get_pill_color(self.precision_score)
        col_const = get_pill_color(self.constancy_score)
        col_over = get_pill_color(self.overall_score)

        feedback_bg = {
            "success": "rgba(34, 197, 94, 0.15)",
            "warning": "rgba(245, 158, 11, 0.15)",
            "info": "rgba(99, 102, 241, 0.15)"
        }.get(self.feedback_severity, "rgba(99, 102, 241, 0.15)")

        feedback_border = {
            "success": "#22C55E",
            "warning": "#F59E0B",
            "info": "#6366F1"
        }.get(self.feedback_severity, "#6366F1")

        feedback_text_col = {
            "success": "#4ADE80",
            "warning": "#FDE047",
            "info": "#C7D2FE"
        }.get(self.feedback_severity, "#C7D2FE")

        detect_tag = "🔍 IA Auto" if self.is_auto_detected else "⚙️ Manual"

        return f"""
        <div style="background: #090D16; border: 1.5px solid #6366F1; border-radius: 12px; padding: 12px 14px; margin-bottom: 12px; box-shadow: 0 4px 18px rgba(99, 102, 241, 0.25);">
            <!-- Cabeçalho da Modalidade -->
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E293B; padding-bottom: 8px; margin-bottom: 10px;">
                <div>
                    <span style="color: #A5B4FC; font-size: 10.5px; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase;">🎓 MODALIDADE AO VIVO</span>
                    <div style="color: #FFFFFF; font-size: 16px; font-weight: 900; font-family: monospace;">{html.escape(mod_name)}</div>
                    <div style="color: #94A3B8; font-size: 11px;">{html.escape(mod_cat)}</div>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(99, 102, 241, 0.22); color: #C7D2FE; border: 1px solid #6366F1; padding: 2px 8px; border-radius: 9999px; font-size: 11px; font-weight: 700;">
                        {detect_tag} ({int(self.modality_detection_conf * 100)}%)
                    </span>
                    <div style="color: #38BDF8; font-size: 11px; font-weight: 600; margin-top: 3px;">🥋 {html.escape(self.kendoka_name)}</div>
                </div>
            </div>

            <!-- Contador de Repetições e Cadência -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
                <div style="background: rgba(99, 102, 241, 0.12); border: 1.5px solid #6366F1; border-radius: 8px; padding: 8px 10px; text-align: center;">
                    <div style="color: #A5B4FC; font-size: 10px; font-weight: 800; letter-spacing: 0.5px;">REPETIÇÕES / GOLPES</div>
                    <div style="color: #FFFFFF; font-size: 32px; font-weight: 900; font-family: monospace; line-height: 1.1; margin: 2px 0;">{self.rep_count}</div>
                    <div style="color: #94A3B8; font-size: 10px;">Execuções detectadas</div>
                </div>
                <div style="background: rgba(56, 189, 248, 0.10); border: 1.5px solid #38BDF8; border-radius: 8px; padding: 8px 10px; text-align: center;">
                    <div style="color: #7DD3FC; font-size: 10px; font-weight: 800; letter-spacing: 0.5px;">CADÊNCIA RÍTMICA</div>
                    <div style="color: #38BDF8; font-size: 32px; font-weight: 900; font-family: monospace; line-height: 1.1; margin: 2px 0;">{cpm:.0f}</div>
                    <div style="color: #94A3B8; font-size: 10px;">Cortes / Minuto (CPM)</div>
                </div>
            </div>

            <!-- Barras dos 3 Pilares -->
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #1E293B; border-radius: 8px; padding: 8px 10px; margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
                    <span style="color: #E2E8F0; font-weight: 600;">🏃 Movimentação (Shisei & Pés):</span>
                    <span style="color: {col_mov}; font-weight: 800;">{self.movement_score:.1f}%</span>
                </div>
                <div style="background: #1E293B; border-radius: 4px; height: 6px; overflow: hidden; margin-bottom: 6px;">
                    <div style="background: {col_mov}; width: {min(100.0, max(0.0, self.movement_score))}%; height: 100%;"></div>
                </div>

                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
                    <span style="color: #E2E8F0; font-weight: 600;">🎯 Precisão (Ki-Ken-Tai-Ichi):</span>
                    <span style="color: {col_prec}; font-weight: 800;">{self.precision_score:.1f}%</span>
                </div>
                <div style="background: #1E293B; border-radius: 4px; height: 6px; overflow: hidden; margin-bottom: 6px;">
                    <div style="background: {col_prec}; width: {min(100.0, max(0.0, self.precision_score))}%; height: 100%;"></div>
                </div>

                <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
                    <span style="color: #E2E8F0; font-weight: 600;">⏱️ Constância (Ritmo & Fadiga):</span>
                    <span style="color: {col_const}; font-weight: 800;">{self.constancy_score:.1f}%</span>
                </div>
                <div style="background: #1E293B; border-radius: 4px; height: 6px; overflow: hidden; margin-bottom: 4px;">
                    <div style="background: {col_const}; width: {min(100.0, max(0.0, self.constancy_score))}%; height: 100%;"></div>
                </div>
            </div>

            <!-- Banner de Alerta / Feedback Pedagógico Instantâneo -->
            <div style="background: {feedback_bg}; border: 1px solid {feedback_border}; border-radius: 8px; padding: 7px 10px; font-size: 12px; color: {feedback_text_col}; font-weight: 600;">
                {html.escape(self.latest_feedback)}
            </div>
        </div>
        """

    def generate_final_session_report(self) -> Dict[str, Any]:
        """Consolida as métricas finais e gera os relatórios em Markdown e JSON."""
        elapsed = max(1.0, time.time() - self.start_time)
        meta = TRAINING_MODALITIES_METADATA.get(self.current_modality_key, TRAINING_MODALITIES_METADATA["suburi"])

        avg_mov = float(sum(p[0] for p in self.pillar_records) / max(1, len(self.pillar_records))) if self.pillar_records else self.movement_score
        avg_prec = float(sum(p[1] for p in self.pillar_records) / max(1, len(self.pillar_records))) if self.pillar_records else self.precision_score
        avg_const = float(sum(p[2] for p in self.pillar_records) / max(1, len(self.pillar_records))) if self.pillar_records else self.constancy_score
        avg_overall = round((avg_mov * 0.40) + (avg_prec * 0.35) + (avg_const * 0.25), 1)

        cpm_final = round(self.rep_count / (elapsed / 60.0), 1)

        # Diagnósticos pedagógicos
        strengths = []
        improvements = []
        drills = []

        if avg_mov >= 75.0:
            strengths.append("Excelente postura vertical da coluna (Shisei) e estabilidade de tronco mantida.")
        else:
            improvements.append("Oscilação ou inclinação anterior do tronco durante os movimentos.")
            drills.append("Praticar Suburi diante de espelho focando na verticalidade das costas e ombros relaxados.")

        if avg_prec >= 75.0:
            strengths.append("Ótimo controle de empunhadura Tenouchi e alinhamento do golpe no centro do alvo.")
        else:
            improvements.append("Desalinhamento de ombros ou altura irregular na parada final do Shinai.")
            drills.append("Executar 50 repetições de Shōmen-uchi pausado conferindo a parada exata na altura dos olhos.")

        if avg_const >= 75.0:
            strengths.append(f"Cadência rítmica consistente com média de {cpm_final} repetições por minuto.")
        else:
            improvements.append("Irregularidade no ritmo de repetição entre os golpes com desaceleração prematura.")
            drills.append("Exercício de Kakari-geiko e Suburi com metrônomo para condicionamento da constância respiratória.")

        markdown_report = f"""# 🥋 SenpAI - Relatório de Treinamento em Tempo Real
**Data:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Kendoka:** {self.kendoka_name}  
**Modalidade:** {meta.get('name', 'Suburi')} ({meta.get('category', 'Treino')})  
**Duração da Sessão:** {elapsed:.1f} segundos ({elapsed/60.0:.1f} min)  
**Total de Repetições Detectadas:** {self.rep_count}  
**Cadência Média:** {cpm_final} CPM (Cortes por Minuto)  

---

## 📊 Avaliação dos 3 Pilares Fundamentais
- **🏃 Movimentação:** {avg_mov:.1f} / 100
- **🎯 Precisão:** {avg_prec:.1f} / 100
- **⏱️ Constância:** {avg_const:.1f} / 100
- **⭐ Nota Geral Consolidada:** **{avg_overall:.1f} / 100**

---

## 💡 Diagnóstico Pedagógico
### Pontos Fortes:
{chr(10).join(f"- ✅ {s}" for s in strengths) if strengths else "- Em consolidação técnica inicial."}

### Pontos de Atenção & Melhoria:
{chr(10).join(f"- ⚠️ {i}" for i in improvements) if improvements else "- Nenhuma falha biomecânica grave detectada."}

### Prescrições de Exercícios Recomendados:
{chr(10).join(f"- 🥋 {d}" for d in drills) if drills else "- Manter a rotina habitual de Keiko."}

---
*Relatório gerado automaticamente pelo SenpAI em estrita conformidade com os manuais oficiais da FIK e AJKF/ZNKR.*
"""

        return {
            "kendoka_name": self.kendoka_name,
            "modality_key": self.current_modality_key,
            "modality_name": meta.get("name", "Suburi"),
            "modality_category": meta.get("category", "Treino"),
            "duration_seconds": round(elapsed, 2),
            "total_reps": self.rep_count,
            "cadence_cpm": cpm_final,
            "average_movement": round(avg_mov, 1),
            "average_precision": round(avg_prec, 1),
            "average_constancy": round(avg_const, 1),
            "overall_score": avg_overall,
            "strengths": strengths,
            "improvements": improvements,
            "prescribed_drills": drills,
            "rep_history": self.rep_history,
            "markdown_report": markdown_report
        }
