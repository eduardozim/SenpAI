"""
Rastreador Dedicado de Múltiplos Kenshis para o Modo de Treinamento e Aprendizado (SenpAI).
Isolado do CombatantTracker de Shiai (combate 1v1).
Características principais:
- Rastreamento simultâneo de N praticantes no dojo.
- Filtro anti-fantasmas (estágio TENTATIVE -> CONFIRMED) para evitar sobre-alocação de IDs por detecções transitórias.
- Janela de oclusão ampliada (90 frames / ~3.0s) com âncora de posição amortecida.
- Associação em cascata: Tracks Ativos (espacial) -> Tracks Perdidos (reassociação rápida) -> Tracks Dormentes (Re-ID multimodal).
- Re-identificação com limiar adaptativo para vídeos reais com compressão/movimento.
- Linha do tempo de presença por Kenshi para cálculo justo de métricas e cadência (CPM).
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

from src.vision.kenshi_reid import (
    KenshiAppearanceSignature,
    KenshiAppearanceSnapshot,
    KenshiReIDExtractor
)


class TrainingTrackState:
    """Representa a máquina de estados e histórico de um Kenshi no treinamento."""
    def __init__(self, kenshi_id: str, initial_landmarks: Dict[str, Any], initial_frame_idx: int, is_confirmed: bool = True):
        self.kenshi_id = kenshi_id # ex: "KENSHI_01" ou "TENTATIVE_X"
        self.display_name = f"Kenshi {kenshi_id.replace('KENSHI_', '')}" if not kenshi_id.startswith("TENTATIVE") else "Detectando..."
        self.status = "ACTIVE" # "ACTIVE", "LOST", "DORMANT"
        self.is_confirmed = is_confirmed
        self.hit_streak = 1
        self.lost_frames_count = 0
        self.first_seen_frame = initial_frame_idx
        self.last_seen_frame = initial_frame_idx

        # Histórico temporal alinhado ao frame do vídeo (None quando fora de cena)
        self.history: List[Optional[Dict[str, Any]]] = []
        self.presence_mask: List[bool] = [] # True se presente no frame, False se ausente

        # Últimas coordenadas conhecidas
        self.last_center_x, self.last_center_y = self._get_center(initial_landmarks)
        self.vx = 0.0
        self.vy = 0.0
        self.last_valid_landmarks = initial_landmarks

        # Assinatura de Re-ID
        self.signature = KenshiAppearanceSignature(kenshi_id)
        self.reentry_count = 0
        self.last_reentry_frame: Optional[int] = None
        self.reid_match_confidence: float = 1.0

        # Histórico de intervalos de presença: [(frame_in, frame_out), ...]
        self.presence_intervals: List[Tuple[int, int]] = []
        self._current_interval_start: Optional[int] = initial_frame_idx if is_confirmed else None

    @staticmethod
    def _get_center(landmarks: Dict[str, Any]) -> Tuple[float, float]:
        l_hip = landmarks.get("LEFT_HIP")
        r_hip = landmarks.get("RIGHT_HIP")
        if l_hip and r_hip:
            return (l_hip["x"] + r_hip["x"]) / 2.0, (l_hip["y"] + r_hip["y"]) / 2.0
        nose = landmarks.get("NOSE")
        if nose:
            return nose["x"], nose["y"]
        return 0.5, 0.5

    def update_active(self, landmarks: Dict[str, Any], snap: KenshiAppearanceSnapshot, frame_idx: int):
        """Atualiza estado ativo do Kenshi com amortecimento de velocidade para evitar picos por jitter."""
        cx, cy = self._get_center(landmarks)
        if self.last_center_x is not None:
            raw_vx = cx - self.last_center_x
            raw_vy = cy - self.last_center_y
            self.vx = 0.70 * self.vx + 0.30 * float(np.clip(raw_vx, -0.08, 0.08))
            self.vy = 0.70 * self.vy + 0.30 * float(np.clip(raw_vy, -0.08, 0.08))
        self.last_center_x = cx
        self.last_center_y = cy
        self.last_valid_landmarks = landmarks
        self.last_seen_frame = frame_idx
        self.lost_frames_count = 0
        self.hit_streak += 1
        self.status = "ACTIVE"

        self.signature.update(snap)

    def mark_lost(self):
        """Marca como perdido temporariamente (oclusão breve ou movimento rápido)."""
        if self.status == "ACTIVE":
            self.status = "LOST"
        self.lost_frames_count += 1
        self.hit_streak = 0
        # Decaimento de inércia para ancorar firmemente na última posição observada
        self.vx *= 0.70
        self.vy *= 0.70

    def mark_dormant(self, frame_idx: int):
        """Marca como dormente (saiu do enquadramento). Registra fim do intervalo de presença."""
        if self.status != "DORMANT":
            self.status = "DORMANT"
            if self._current_interval_start is not None:
                self.presence_intervals.append((self._current_interval_start, self.last_seen_frame))
                self._current_interval_start = None

    def mark_reactivated(self, frame_idx: int, confidence: float):
        """Reativa o Kenshi após retorno ao enquadramento."""
        self.status = "ACTIVE"
        self.lost_frames_count = 0
        self.hit_streak = 1
        self.reentry_count += 1
        self.last_reentry_frame = frame_idx
        self.reid_match_confidence = confidence
        if self._current_interval_start is None:
            self._current_interval_start = frame_idx

    def finalize(self, total_frames: int):
        """Finaliza registros ao término da sessão."""
        if self._current_interval_start is not None:
            self.presence_intervals.append((self._current_interval_start, self.last_seen_frame))
            self._current_interval_start = None


class TrainingMultiKenshiTracker:
    """
    Rastreador Multi-Kenshi de Treino do SenpAI.
    """
    def __init__(
        self,
        max_lost_frames: int = 90,           # ~3.0s a 30fps para tolerar oclusões e passadas de suburi
        spatial_dist_threshold: float = 0.44,# Tolerância espacial para passos e movimentações
        reid_similarity_threshold: float = 0.46, # Limiar adaptativo multimodal para retorno ao dojo
        min_hits: int = 3                    # Mínimo de confirmações consecutivas para validar novo Kenshi
    ):
        self.max_lost_frames = max_lost_frames
        self.spatial_threshold = spatial_dist_threshold
        self.reid_threshold = reid_similarity_threshold
        self.min_hits = min_hits

        self.tracks: Dict[str, TrainingTrackState] = {} # Apenas tracks confirmados (KENSHI_XX)
        self.tentative_tracks: List[TrainingTrackState] = [] # Tracks provisórios em validação
        self.next_kenshi_num = 1
        self.tentative_counter = 0
        self.frame_idx = 0

        # Galeria persistente de Kenshis dormentes que já passaram pelo treino
        self.dormant_gallery: Dict[str, KenshiAppearanceSignature] = {}

        # Estatísticas da sessão
        self.total_reidentifications = 0
        self.reidentification_events: List[Dict[str, Any]] = []

    def _allocate_kenshi_id(self) -> str:
        k_id = f"KENSHI_{self.next_kenshi_num:02d}"
        self.next_kenshi_num += 1
        return k_id

    def process_frame(
        self,
        candidates: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Processa candidatos detectados no frame atual através de cascata hierárquica:
        1. Associação espacial direta com tracks ATIVOS (Hungarian algorithm).
        2. Reassociação imediata com tracks PERDIDOS (LOST <= max_lost_frames) por âncora de posição.
        3. Re-identificação multimodal contra tracks DORMENTES (Kenshis que saíram e retornaram).
        4. Filtro de tracks provisórios (TENTATIVE): descarta ruídos rápidos e confirma kenshis estáveis.
        5. Atualização de históricos alinhados e desduplicação espacial.
        """
        self.frame_idx += 1
        current_frame_results: Dict[str, Optional[Dict[str, Any]]] = {}

        cand_snapshots = [
            KenshiReIDExtractor.extract_snapshot(frame, lm)
            for lm in candidates
        ]
        cand_centers = [TrainingTrackState._get_center(lm) for lm in candidates]

        matched_tracks = set()
        matched_candidates = set()

        # ---------------------------------------------------------------------
        # FASE 1: Associação com Tracks ATIVOS (Confirmados atualmente em cena)
        # ---------------------------------------------------------------------
        active_ids = [t_id for t_id, t in self.tracks.items() if t.status == "ACTIVE"]
        if active_ids and candidates:
            cost_matrix = np.zeros((len(active_ids), len(candidates)), dtype=np.float32)
            for i, t_id in enumerate(active_ids):
                t = self.tracks[t_id]
                pred_x = t.last_center_x + t.vx
                pred_y = t.last_center_y + t.vy
                for j, (cx, cy) in enumerate(cand_centers):
                    cost_matrix[i, j] = float(np.hypot(pred_x - cx, pred_y - cy))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.spatial_threshold:
                    t_id = active_ids[r]
                    matched_tracks.add(t_id)
                    matched_candidates.add(c)
                    track = self.tracks[t_id]
                    track.update_active(candidates[c], cand_snapshots[c], self.frame_idx)
                    current_frame_results[t_id] = candidates[c]

        # ---------------------------------------------------------------------
        # FASE 2: Reassociação Imediata com Tracks PERDIDOS Recentemente (LOST)
        # Impede que oclusões transitórias gerem novos IDs duplicados
        # ---------------------------------------------------------------------
        unmatched_cand_indices = [j for j in range(len(candidates)) if j not in matched_candidates]
        lost_ids = [t_id for t_id, t in self.tracks.items() if t.status == "LOST" and t.lost_frames_count <= self.max_lost_frames]

        if lost_ids and unmatched_cand_indices:
            lost_cost = np.zeros((len(lost_ids), len(unmatched_cand_indices)), dtype=np.float32)
            for i, t_id in enumerate(lost_ids):
                t = self.tracks[t_id]
                for j, c_idx in enumerate(unmatched_cand_indices):
                    cx, cy = cand_centers[c_idx]
                    lost_cost[i, j] = float(np.hypot(t.last_center_x - cx, t.last_center_y - cy))

            r_lost, c_lost = linear_sum_assignment(lost_cost)
            for r, c in zip(r_lost, c_lost):
                if lost_cost[r, c] <= 0.42:
                    t_id = lost_ids[r]
                    real_c_idx = unmatched_cand_indices[c]
                    matched_tracks.add(t_id)
                    matched_candidates.add(real_c_idx)
                    track = self.tracks[t_id]
                    track.update_active(candidates[real_c_idx], cand_snapshots[real_c_idx], self.frame_idx)
                    current_frame_results[t_id] = candidates[real_c_idx]

        # ---------------------------------------------------------------------
        # FASE 3: Re-identificação Multimodal contra Kenshis DORMENTES (Retorno)
        # ---------------------------------------------------------------------
        still_unmatched = [j for j in range(len(candidates)) if j not in matched_candidates]
        dormant_tracks = [t for t in self.tracks.values() if t.status == "DORMANT"]

        if dormant_tracks and still_unmatched:
            for c_idx in list(still_unmatched):
                cand_snap = cand_snapshots[c_idx]
                cand_sig = KenshiAppearanceSignature("CANDIDATE")
                cand_sig.update(cand_snap)

                best_match_id = None
                best_score = 0.0
                second_score = 0.0
                best_details = {}

                for d_track in dormant_tracks:
                    score, details = d_track.signature.compute_similarity(cand_sig)
                    if score > best_score:
                        second_score = best_score
                        best_score = score
                        best_match_id = d_track.kenshi_id
                        best_details = details
                    elif score > second_score:
                        second_score = score

                # Aceitação de Re-ID com limiar adaptativo
                if best_match_id is not None and best_score >= self.reid_threshold:
                    matched_candidates.add(c_idx)
                    still_unmatched.remove(c_idx)
                    re_track = self.tracks[best_match_id]
                    re_track.mark_reactivated(self.frame_idx, best_score)
                    re_track.update_active(candidates[c_idx], cand_snap, self.frame_idx)
                    current_frame_results[best_match_id] = candidates[c_idx]

                    self.total_reidentifications += 1
                    self.reidentification_events.append({
                        "kenshi_id": best_match_id,
                        "frame_idx": self.frame_idx,
                        "score": round(best_score, 3),
                        "details": best_details
                    })
                    dormant_tracks = [t for t in self.tracks.values() if t.status == "DORMANT"]

        # ---------------------------------------------------------------------
        # FASE 4: Gestão de Candidatos Não Associados & Tracks Tentative (Anti-Fantasmas)
        # ---------------------------------------------------------------------
        new_tentatives: List[TrainingTrackState] = []
        for c_idx in still_unmatched:
            cx, cy = cand_centers[c_idx]
            cand_lm = candidates[c_idx]
            cand_snap = cand_snapshots[c_idx]

            matched_tentative = None
            best_tent_dist = 0.32
            for tent in self.tentative_tracks:
                d = float(np.hypot(tent.last_center_x - cx, tent.last_center_y - cy))
                if d < best_tent_dist:
                    best_tent_dist = d
                    matched_tentative = tent

            if matched_tentative is not None:
                matched_tentative.update_active(cand_lm, cand_snap, self.frame_idx)
                # No início do vídeo (primeiros 12 frames), confirmação mais ágil
                req_hits = 1 if self.frame_idx <= 12 else self.min_hits
                if matched_tentative.hit_streak >= req_hits:
                    # Promover para track CONFIRMADO!
                    new_id = self._allocate_kenshi_id()
                    matched_tentative.kenshi_id = new_id
                    matched_tentative.display_name = f"Kenshi {new_id.replace('KENSHI_', '')}"
                    matched_tentative.is_confirmed = True
                    matched_tentative._current_interval_start = matched_tentative.first_seen_frame
                    self.tracks[new_id] = matched_tentative
                    current_frame_results[new_id] = cand_lm
                else:
                    new_tentatives.append(matched_tentative)
            else:
                if self.frame_idx <= 12:
                    # Nos primeiros frames do vídeo, os Kenshis em cena já nascem confirmados
                    new_id = self._allocate_kenshi_id()
                    new_track = TrainingTrackState(new_id, cand_lm, self.frame_idx, is_confirmed=True)
                    new_track.signature.update(cand_snap)
                    self.tracks[new_id] = new_track
                    current_frame_results[new_id] = cand_lm
                else:
                    self.tentative_counter += 1
                    t_name = f"TENTATIVE_{self.tentative_counter}"
                    tent_state = TrainingTrackState(t_name, cand_lm, self.frame_idx, is_confirmed=False)
                    tent_state.signature.update(cand_snap)
                    new_tentatives.append(tent_state)

        # Manter apenas tentativas que foram observadas recentemente (descarta ruídos em até 3 frames)
        self.tentative_tracks = [t for t in new_tentatives if (self.frame_idx - t.last_seen_frame) <= 3]

        # ---------------------------------------------------------------------
        # FASE 5: Atualizar status dos tracks confirmados que não tiveram detecção
        # ---------------------------------------------------------------------
        for t_id, track in self.tracks.items():
            if t_id not in current_frame_results:
                current_frame_results[t_id] = None
                track.mark_lost()
                if track.lost_frames_count > self.max_lost_frames:
                    # Kenshi saiu fisicamente do enquadramento da câmera
                    track.mark_dormant(self.frame_idx)
                    self.dormant_gallery[t_id] = track.signature

        # ---------------------------------------------------------------------
        # FASE 6: Histórico alinhado por Kenshi
        # ---------------------------------------------------------------------
        for t_id, track in self.tracks.items():
            lm_val = current_frame_results.get(t_id)
            track.history.append(lm_val)
            track.presence_mask.append(lm_val is not None)

        return current_frame_results

    def finish_session(self, total_frames: int):
        """Finaliza a sessão e consolida dados de presença."""
        for track in self.tracks.values():
            track.finalize(total_frames)

    def get_session_summary(self) -> Dict[str, Any]:
        """Gera sumário completo dos Kenshis confirmados no treinamento."""
        kenshis_data = []
        for t_id, track in self.tracks.items():
            presence_frames = sum(1 for p in track.presence_mask if p)
            kenshis_data.append({
                "kenshi_id": track.kenshi_id,
                "display_name": track.display_name,
                "presence_frames": presence_frames,
                "reentry_count": track.reentry_count,
                "presence_intervals": track.presence_intervals,
                "thumbnail": track.signature.best_thumbnail
            })

        return {
            "total_kenshis_detected": len(self.tracks),
            "total_reidentifications": self.total_reidentifications,
            "reidentification_events": self.reidentification_events,
            "kenshis": kenshis_data
        }

    def draw_overlay(
        self,
        frame: np.ndarray,
        current_results: Dict[str, Optional[Dict[str, Any]]],
        custom_names: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Desenha os esqueletos com as cores atribuídas a cada Kenshi rastreado no treino,
        indicando seu ID, nome (se configurado), reentradas por Re-ID e banner informativo de sessão.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        palette = [
            (246, 130, 59),  # Ciano/Azul claro (BGR)
            (59, 214, 115),  # Verde esmeralda
            (37, 99, 235),   # Laranja/Vermelho quente
            (217, 70, 239),  # Magenta/Roxo
            (245, 158, 11),  # Azul royal
            (14, 165, 233),  # Âmbar
            (168, 85, 247),  # Púrpura
            (74, 222, 128),  # Verde lima
        ]

        from src.vision.pose_detector import CombatantVisualizer

        for idx_track, (k_id, lm) in enumerate(current_results.items()):
            if lm is not None:
                track = self.tracks.get(k_id)
                col = palette[idx_track % len(palette)]
                display_name = track.display_name if track else k_id
                c_name = (custom_names or {}).get(k_id, display_name)
                label_txt = f"{c_name}"
                if track and track.reentry_count > 0:
                    label_txt += f" [Re-ID x{track.reentry_count}]"

                CombatantVisualizer.draw_skeleton(out, lm, color=col, label=label_txt)

        # Banner de Treinamento
        active_now = sum(1 for lm in current_results.values() if lm is not None)
        cv2.rectangle(out, (10, 10), (w - 10, 48), (15, 23, 42), -1)
        cv2.rectangle(out, (10, 10), (w - 10, 48), (99, 102, 241), 1)
        txt_banner = f"MODO TREINAMENTO SENPAI | Kenshis em Cena: {active_now} | Total na Sessao: {len(self.tracks)} | Re-IDs: {self.total_reidentifications}"
        cv2.putText(out, txt_banner, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        return out
