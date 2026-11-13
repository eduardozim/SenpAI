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

import numpy as np

from src.analytics.training_analyzer import (
    TrainingAnalyzer,
    TRAINING_MODALITIES_METADATA,
    TrainingPillarMetrics,
    format_training_mmss
)
from src.analytics.training_feedback import (
    FUMIKOMI_MODALITIES, KIKENTAI_MIN_MEASURED_FRACTION, build_diagnosis, diagnosis_markdown, legacy_lists,
    measure_kikentai,
)

# Quanto do histórico o Ki-Ken-Tai-Ichi espera depois do impacto para medir o pé (a janela é ±0,25 s)
KIKENTAI_WAIT_SECONDS = 0.3
RECENT_REPS_FOR_HUD = 10


class LiveTrainingSessionManager:
    """
    Gerenciador de estado e avaliação para sessões de Treinamento de Kendo em Tempo Real.
    """

    def __init__(
        self,
        modality_override: Optional[str] = None,
        kendoka_name: str = "Kendoka Praticante",
        target_dan: Optional[int] = 3,
        training_analyzer: Optional[TrainingAnalyzer] = None
    ):
        self.analyzer = training_analyzer or TrainingAnalyzer()
        self.modality_override = modality_override if modality_override and modality_override != "auto" else None
        self.kendoka_name = kendoka_name.strip() or "Kendoka Praticante"
        self.target_dan = target_dan if target_dan is not None else 3

        self.start_time = time.time()
        self.frame_count = 0
        self.rep_count = 0
        self.rep_history: List[Dict[str, Any]] = []

        # Modalidade
        self.current_modality_key = self.modality_override or "suburi"
        self.modality_detection_conf = 0.90 if self.modality_override else 0.50
        self.is_auto_detected = (self.modality_override is None)
        self.modality_timeline_segments: List[Dict[str, Any]] = [
            {
                "modality_key": self.current_modality_key,
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "confidence": self.modality_detection_conf,
                "rep_count": 0,
                "pillar_records": []
            }
        ]

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

        # Medidas por repetição (mesmo formato do modo gravado) para o diagnóstico detalhado
        self.rep_details: List[Dict[str, Any]] = []
        # Golpes esperando os quadros seguintes para medir o pé do fumikomi: (índice em rep_details, pose do impacto)
        self._pending_kikentai: List[Tuple[int, Dict[str, Any], float]] = []
        # Submétricas de movimentação medidas periodicamente (média no fim da sessão)
        self._movement_sub_records: List[Dict[str, float]] = []
        self.last_diagnosis: Dict[str, Any] = {}

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

        # Ki-Ken-Tai-Ichi dos golpes recentes, medido quando já há quadros suficientes depois do impacto
        self._resolve_pending_kikentai(primary_history, fps)

        # 1. Identificação periódica da modalidade caso seja seleção automática (a cada 40 quadros)
        if self.is_auto_detected and self.frame_count % 40 == 0 and len(primary_history) >= 20:
            try:
                m_key, m_conf, _ = self.analyzer.detect_training_modality(
                    primary_history=primary_history[-90:],
                    secondary_history=sec_history[-90:] if sec_history else [],
                    fps=fps
                )
                prev_key = self.current_modality_key
                self.current_modality_key = m_key
                self.modality_detection_conf = m_conf
                if self.modality_timeline_segments:
                    self.modality_timeline_segments[-1]["end_seconds"] = round(elapsed, 2)
                if m_key != prev_key:
                    self.modality_timeline_segments.append({
                        "modality_key": m_key,
                        "start_seconds": round(elapsed, 2),
                        "end_seconds": round(elapsed, 2),
                        "confidence": m_conf,
                        "rep_count": 0,
                        "pillar_records": []
                    })
            except Exception:
                pass
        elif self.modality_timeline_segments:
            self.modality_timeline_segments[-1]["end_seconds"] = round(elapsed, 2)

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
                        if self.modality_timeline_segments:
                            self.modality_timeline_segments[-1]["rep_count"] = int(self.modality_timeline_segments[-1].get("rep_count", 0)) + 1
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
                        self._record_rep_details(active_landmarks, elapsed, posture_frame_score, fps)

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
                self.movement_score = float(pm.movimentacao)
                self._movement_sub_records.append(dict(pm.movimentacao_submetrics))
                # Precisão dos golpes recentes (sem golpes, a janela de poses não tem o que medir)
                recent_prec = self._precision_from_reps(self.rep_details[-RECENT_REPS_FOR_HUD:])
                self.precision_score = float(recent_prec[0]) if recent_prec else float(pm.precisao)

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
                    const_sc = float(pm.constancia)

                self.constancy_score = float(const_sc)
                self.overall_score = round(
                    (self.movement_score * 0.40) +
                    (self.precision_score * 0.35) +
                    (self.constancy_score * 0.25),
                    1
                )
                self.pillar_records.append((self.movement_score, self.precision_score, self.constancy_score))
                if self.modality_timeline_segments:
                    self.modality_timeline_segments[-1]["pillar_records"].append(
                        (self.movement_score, self.precision_score, self.constancy_score)
                    )
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

    # ------------------------------------------------------------------------------------------------------
    # Medidas por repetição (mesmas fórmulas do modo gravado, em training_analyzer.calculate_pillar_metrics)
    # ------------------------------------------------------------------------------------------------------
    def _record_rep_details(self, lm: Dict[str, Any], elapsed: float, posture_score: float, fps: float) -> None:
        rep: Dict[str, Any] = {"rep": self.rep_count, "tempo_s": round(elapsed, 2), "postura": round(posture_score, 1)}
        rw, nose = lm.get("RIGHT_WRIST"), lm.get("NOSE")
        if rw and nose:
            lat_dev = abs(rw["x"] - nose["x"])
            rep["centro"] = round(max(0.0, 1.0 - lat_dev * 5.0) * 100.0, 1)
            rep["alvo"] = round(max(0.0, 1.0 - lat_dev * 3.5) * 100.0, 1)
            rep["furikaburi"] = round(float(np.clip(1.0 - (self._furikaburi_peak_y - nose["y"]), 0.0, 1.0)) * 100.0, 1)
        self.rep_details.append(rep)
        if self.current_modality_key in FUMIKOMI_MODALITIES:
            self._pending_kikentai.append((len(self.rep_details) - 1, lm, time.time()))

    def _resolve_pending_kikentai(self, history: Sequence[Optional[Dict[str, Any]]], fps: float) -> None:
        if not self._pending_kikentai:
            return
        now = time.time()
        still_waiting = []
        for rep_idx, impact_lm, t_impact in self._pending_kikentai:
            if now - t_impact < KIKENTAI_WAIT_SECONDS:
                still_waiting.append((rep_idx, impact_lm, t_impact))
                continue
            # A pose do impacto é localizada pela identidade: o histórico é cortado e deslocado durante a sessão
            idx = next((i for i in range(len(history) - 1, -1, -1) if history[i] is impact_lm), None)
            res = measure_kikentai(history, idx, fps) if idx is not None else None
            if res is None:
                continue
            offset_ms, score = res
            rep = self.rep_details[rep_idx]
            rep["kikentai_ms"], rep["kikentai"] = offset_ms, score
            if abs(offset_ms) > 60:
                quando = "depois" if offset_ms > 0 else "antes"
                self.latest_feedback = (f"⏱️ Repetição #{rep['rep']}: o pé chegou {abs(offset_ms):.0f} ms {quando} "
                                        f"da espada. Pé e espada juntos (Ki-Ken-Tai-Ichi)!")
                self.feedback_severity = "warning"
        self._pending_kikentai = still_waiting

    def _precision_from_reps(self, reps: List[Dict[str, Any]],
                             modality_key: Optional[str] = None) -> Optional[Tuple[float, Dict[str, float], List[Dict[str, str]]]]:
        """(nota de Precisão, submétricas, não medidos) a partir das repetições; None sem golpes medidos."""
        modality_key = modality_key or self.current_modality_key
        alvo = [r["alvo"] for r in reps if r.get("alvo") is not None]
        centro = [r["centro"] for r in reps if r.get("centro") is not None]
        if not alvo or not centro:
            return None
        subs = {"trajetoria_alvo": round(float(np.mean(alvo)), 1), "controle_linha_centro": round(float(np.mean(centro)), 1)}
        kkt = [r["kikentai"] for r in reps if r.get("kikentai") is not None]
        nao_medidos: List[Dict[str, str]] = []
        kkt_min = max(2, int(np.ceil(KIKENTAI_MIN_MEASURED_FRACTION * len(reps))))
        if modality_key in FUMIKOMI_MODALITIES and len(kkt) >= kkt_min:
            subs["kikentai_sincronismo"] = round(float(np.mean(kkt)), 1)
            score = subs["trajetoria_alvo"] * 0.40 + subs["kikentai_sincronismo"] * 0.35 + subs["controle_linha_centro"] * 0.25
        else:
            score = (subs["trajetoria_alvo"] * 0.40 + subs["controle_linha_centro"] * 0.25) / 0.65
            motivo = ("não se aplica a esta modalidade (sem fumikomi)" if modality_key not in FUMIKOMI_MODALITIES else
                      f"o pé do fumikomi foi medido em {len(kkt)} de {len(reps)} golpes (mínimo {kkt_min}): "
                      "pés fora da imagem ou sem fumikomi visível")
            nao_medidos.append({"metrica": "kikentai_sincronismo", "label": "Sincronismo Ki-Ken-Tai-Ichi", "motivo": motivo})
        return round(score, 1), subs, nao_medidos

    def _session_pillars(self, elapsed: float) -> TrainingPillarMetrics:
        """Métricas da sessão inteira, no mesmo formato do modo gravado, para o diagnóstico detalhado."""
        reps = self.rep_details
        # Movimentação: média das medições periódicas da janela de poses
        mov_sub: Dict[str, float] = {}
        for key in ("verticalidade_coluna", "nivelamento_ombros", "alinhamento_base_pes", "amplitude_furikaburi"):
            vals = [r[key] for r in self._movement_sub_records if isinstance(r.get(key), (int, float))]
            if vals:
                mov_sub[key] = round(float(np.mean(vals)), 1)
        if mov_sub:
            mov = (mov_sub.get("verticalidade_coluna", 75.0) * 0.35 + mov_sub.get("nivelamento_ombros", 80.0) * 0.25
                   + mov_sub.get("alinhamento_base_pes", 78.0) * 0.20 + mov_sub.get("amplitude_furikaburi", 72.0) * 0.20)
        else:
            mov = self.movement_score
        # Precisão: golpes da sessão
        prec_res = self._precision_from_reps(reps)
        if prec_res:
            prec, prec_sub, nao_medidos = prec_res
        else:
            prec, prec_sub = self.precision_score, {}
            nao_medidos = [{"metrica": "trajetoria_alvo", "label": "Trajetória no alvo", "motivo": "nenhum golpe com o pulso visível"},
                           {"metrica": "controle_linha_centro", "label": "Controle da linha central", "motivo": "nenhum golpe com o pulso visível"},
                           {"metrica": "kikentai_sincronismo", "label": "Sincronismo Ki-Ken-Tai-Ichi", "motivo": "nenhum golpe detectado"}]
        # Constância: intervalos entre golpes, queda de postura ao longo da sessão e cadência da modalidade
        times = [r["tempo_s"] for r in reps]
        const_sub: Dict[str, float] = {}
        std_dev = 0.0
        if len(times) >= 3:
            gaps = np.diff(times)
            std_dev = float(np.std(gaps))
            const_sub["regularidade_ritmo"] = round(max(0.0, 1.0 - std_dev / max(0.1, float(np.mean(gaps)))) * 100.0, 1)
        postures = [r["postura"] for r in reps if r.get("postura") is not None]
        if len(postures) >= 6:
            half = len(postures) // 2
            ratio = float(np.mean(postures[half:])) / max(0.01, float(np.mean(postures[:half])))
            const_sub["resistencia_fadiga"] = round(float(np.clip(ratio * 100.0, 50.0, 100.0)), 1)
        cpm = self.rep_count / max(0.05, elapsed / 60.0)
        lo, hi = self.get_expected_cadence_range()
        if self.rep_count >= 2:
            if lo <= cpm <= hi:
                const_sub["adequacao_cadencia"] = 95.0
            elif cpm < lo:
                const_sub["adequacao_cadencia"] = round(max(50.0, 95.0 - (lo - cpm) * 3.0), 1)
            else:
                const_sub["adequacao_cadencia"] = round(max(60.0, 95.0 - (cpm - hi) * 1.5), 1)
        if const_sub:
            w = {"regularidade_ritmo": 0.45, "resistencia_fadiga": 0.35, "adequacao_cadencia": 0.20}
            const = sum(const_sub[k] * w[k] for k in const_sub) / sum(w[k] for k in const_sub)
        else:
            const = self.constancy_score
        return TrainingPillarMetrics(
            movimentacao_score=mov, precisao_score=prec, constancia_score=const,
            movimentacao_submetrics=mov_sub, precisao_submetrics=prec_sub, constancia_submetrics=const_sub,
            cadence_cpm=cpm, cadence_std_dev_seconds=std_dev, total_repetitions=self.rep_count,
            rep_details=list(reps), nao_medidos=nao_medidos,
        )

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
        if self.modality_timeline_segments:
            self.modality_timeline_segments[-1]["end_seconds"] = round(elapsed, 2)
        meta = TRAINING_MODALITIES_METADATA.get(self.current_modality_key, TRAINING_MODALITIES_METADATA["suburi"])

        avg_mov = float(sum(p[0] for p in self.pillar_records) / max(1, len(self.pillar_records))) if self.pillar_records else self.movement_score
        avg_prec = float(sum(p[1] for p in self.pillar_records) / max(1, len(self.pillar_records))) if self.pillar_records else self.precision_score
        avg_const = float(sum(p[2] for p in self.pillar_records) / max(1, len(self.pillar_records))) if self.pillar_records else self.constancy_score
        avg_overall = round((avg_mov * 0.40) + (avg_prec * 0.35) + (avg_const * 0.25), 1)

        cpm_final = round(self.rep_count / (elapsed / 60.0), 1)

        # Construir sub_sessions e modality_blocks concentrados para a sessão ao vivo
        sub_sessions: List[Dict[str, Any]] = []
        grouped_blocks: Dict[str, Dict[str, Any]] = {}
        block_order: List[str] = []

        for idx_seg, seg in enumerate(self.modality_timeline_segments):
            s_sec = float(seg.get("start_seconds", 0.0))
            e_sec = max(s_sec + 0.1, float(seg.get("end_seconds", elapsed)))
            dur_seg = round(e_sec - s_sec, 2)
            m_key = str(seg.get("modality_key") or self.current_modality_key)
            s_meta = TRAINING_MODALITIES_METADATA.get(m_key, meta)
            s_recs = seg.get("pillar_records", [])
            s_mov = float(sum(r[0] for r in s_recs) / len(s_recs)) if s_recs else avg_mov
            s_prec = float(sum(r[1] for r in s_recs) / len(s_recs)) if s_recs else avg_prec
            s_const = float(sum(r[2] for r in s_recs) / len(s_recs)) if s_recs else avg_const
            s_ov = round((s_mov * 0.40) + (s_prec * 0.35) + (s_const * 0.25), 1)
            s_reps = int(seg.get("rep_count", 0))
            s_start_ts = format_training_mmss(s_sec)
            s_end_ts = format_training_mmss(e_sec)

            sub_sessions.append({
                "segment_index": idx_seg + 1,
                "start_seconds": round(s_sec, 2),
                "end_seconds": round(e_sec, 2),
                "start_timestamp": s_start_ts,
                "end_timestamp": s_end_ts,
                "duration_seconds": dur_seg,
                "modality_key": m_key,
                "modality_name": s_meta.get("name", m_key),
                "modality_category": s_meta.get("category", "Treino"),
                "detection_confidence": float(seg.get("confidence", self.modality_detection_conf)),
                "detection_method": "AUTO_DETECTED" if self.is_auto_detected else "MANUAL_OVERRIDE",
                "total_repetitions": s_reps,
                "average_movement": round(s_mov, 1),
                "average_precision": round(s_prec, 1),
                "average_constancy": round(s_const, 1),
                "overall_score": s_ov
            })

            if m_key not in grouped_blocks:
                block_order.append(m_key)
                grouped_blocks[m_key] = {
                    "modality_key": m_key,
                    "modality_name": s_meta.get("name", m_key),
                    "modality_category": s_meta.get("category", "Treino"),
                    "intervals": [],
                    "total_duration_seconds": 0.0,
                    "total_repetitions": 0,
                    "weighted_mov": 0.0,
                    "weighted_prec": 0.0,
                    "weighted_const": 0.0
                }
            gb = grouped_blocks[m_key]
            gb["intervals"].append({
                "segment_index": idx_seg + 1,
                "start_timestamp": s_start_ts,
                "end_timestamp": s_end_ts,
                "start_seconds": round(s_sec, 2),
                "end_seconds": round(e_sec, 2),
                "duration_seconds": dur_seg,
                "repetitions": s_reps
            })
            gb["total_duration_seconds"] = round(gb["total_duration_seconds"] + dur_seg, 2)
            gb["total_repetitions"] += s_reps
            gb["weighted_mov"] += s_mov * dur_seg
            gb["weighted_prec"] += s_prec * dur_seg
            gb["weighted_const"] += s_const * dur_seg

        modality_blocks: List[Dict[str, Any]] = []
        for b_idx, m_key in enumerate(block_order):
            gb = grouped_blocks[m_key]
            tot_d = max(0.1, float(gb["total_duration_seconds"]))
            b_mov = round(gb["weighted_mov"] / tot_d, 1)
            b_prec = round(gb["weighted_prec"] / tot_d, 1)
            b_const = round(gb["weighted_const"] / tot_d, 1)
            b_ov = round((b_mov * 0.40) + (b_prec * 0.35) + (b_const * 0.25), 1)
            int_label = " + ".join(
                f"{iv['start_timestamp']} a {iv['end_timestamp']}" for iv in gb["intervals"]
            )
            modality_blocks.append({
                "block_index": b_idx + 1,
                "modality_key": m_key,
                "modality_name": gb["modality_name"],
                "modality_category": gb["modality_category"],
                "intervals": gb["intervals"],
                "intervals_label": int_label,
                "session_count": len(gb["intervals"]),
                "total_duration_seconds": gb["total_duration_seconds"],
                "total_repetitions": gb["total_repetitions"],
                "average_movement": b_mov,
                "average_precision": b_prec,
                "average_constancy": b_const,
                "overall_score": b_ov
            })

        # Diagnóstico detalhado (o mesmo do modo gravado), na modalidade em que a sessão passou mais tempo
        diag_modality = (max(modality_blocks, key=lambda b: b["total_duration_seconds"])["modality_key"]
                         if modality_blocks else self.current_modality_key)
        diag_meta = TRAINING_MODALITIES_METADATA.get(diag_modality, meta)
        prev_key = self.current_modality_key
        self.current_modality_key = diag_modality   # cadência e Precisão avaliadas na modalidade do diagnóstico
        try:
            session_pillars = self._session_pillars(elapsed)
        finally:
            self.current_modality_key = prev_key
        diagnosis = build_diagnosis(session_pillars, diag_modality, diag_meta)
        self.last_diagnosis = diagnosis
        strengths, improvements, exercises = legacy_lists(diagnosis)
        drills = [f"{e['name']} — {e['dose']}" for e in exercises]
        # As notas da sessão passam a ser as do diagnóstico (medidas nos golpes, não na janela de poses)
        avg_mov, avg_prec, avg_const = session_pillars.movimentacao, session_pillars.precisao, session_pillars.constancia
        avg_overall = round((avg_mov * 0.40) + (avg_prec * 0.35) + (avg_const * 0.25), 1)

        blocks_md_lines = []
        for mb in modality_blocks:
            blocks_md_lines.append(
                f"- **Bloco #{mb['block_index']} — {mb['modality_name']}** ({mb['intervals_label']} | {mb['total_duration_seconds']:.1f}s | {mb['total_repetitions']} reps): "
                f"Geral **{mb['overall_score']:.1f}/100** (Mov: {mb['average_movement']:.1f} | Prec: {mb['average_precision']:.1f} | Const: {mb['average_constancy']:.1f})"
            )
        blocks_md_section = "\n".join(blocks_md_lines) if blocks_md_lines else "- Sessão única consolidada."

        markdown_report = f"""# 🥋 SenpAI - Relatório de Treinamento em Tempo Real
**Data:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Kendoka:** {self.kendoka_name}  
**Modalidade Principal:** {meta.get('name', 'Suburi')} ({meta.get('category', 'Treino')})  
**Duração da Sessão:** {elapsed:.1f} segundos ({elapsed/60.0:.1f} min)  
**Total de Repetições Detectadas:** {self.rep_count}  
**Cadência Média:** {cpm_final} CPM (Cortes por Minuto)  

---

## 🗂️ Blocos Concentrados por Modalidade de Treinamento
{blocks_md_section}

---

## 📊 Avaliação dos 3 Pilares Fundamentais
- **🏃 Movimentação:** {avg_mov:.1f} / 100
- **🎯 Precisão:** {avg_prec:.1f} / 100
- **⏱️ Constância:** {avg_const:.1f} / 100
- **⭐ Nota Geral Consolidada:** **{avg_overall:.1f} / 100**

---

## 💡 Diagnóstico Pedagógico ({diag_meta.get('name', diag_modality)})
{chr(10).join(diagnosis_markdown(diagnosis))}

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
            "recommended_exercises": exercises,
            "diagnosis": diagnosis,
            "pillars": session_pillars.to_dict(),
            "kendoka_id": "AO_VIVO",
            "session_id": f"aovivo_{int(self.start_time * 1000)}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.start_time)),
            "rep_details": list(self.rep_details),
            "rep_history": self.rep_history,
            "sub_sessions": sub_sessions,
            "modality_blocks": modality_blocks,
            "markdown_report": markdown_report
        }
