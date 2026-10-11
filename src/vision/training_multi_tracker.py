"""
Rastreador Dedicado de Múltiplos Kenshis para o Modo de Treinamento e Aprendizado (SenpAI).
Isolado do CombatantTracker de Shiai (combate 1v1).
Características principais:
- Rastreamento simultâneo de N praticantes no dojo.
- Gerenciamento de ciclo de vida de tracks: ACTIVE, LOST (oclusão breve) e DORMANT (saiu do enquadramento).
- Re-identificação automática (Re-ID) quando um Kenshi retorna ao enquadramento (via Zekken, Tenugui, Tare lateral e Biometria).
- Linha do tempo de presença por Kenshi para cálculo justo de métricas e cadência (CPM).
"""

import time
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
    def __init__(self, kenshi_id: str, initial_landmarks: Dict[str, Any], initial_frame_idx: int):
        self.kenshi_id = kenshi_id # ex: "KENSHI_01"
        self.display_name = f"Kenshi {kenshi_id.replace('KENSHI_', '')}"
        self.status = "ACTIVE" # "ACTIVE", "LOST", "DORMANT"
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
        self._current_interval_start: Optional[int] = initial_frame_idx

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
        """Atualiza estado ativo do Kenshi."""
        cx, cy = self._get_center(landmarks)
        if self.last_center_x is not None:
            self.vx = 0.6 * self.vx + 0.4 * (cx - self.last_center_x)
            self.vy = 0.6 * self.vy + 0.4 * (cy - self.last_center_y)
        self.last_center_x = cx
        self.last_center_y = cy
        self.last_valid_landmarks = landmarks
        self.last_seen_frame = frame_idx
        self.status = "ACTIVE"

        self.signature.update(snap)

    def mark_lost(self, max_lost_frames: int = 45):
        """Marca como perdido temporariamente (oclusão breve)."""
        if self.status == "ACTIVE":
            self.status = "LOST"

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
        self.reentry_count += 1
        self.last_reentry_frame = frame_idx
        self.reid_match_confidence = confidence
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
        max_lost_frames: int = 40,      # ~1.3s a 30fps antes de considerar saída do enquadramento
        reid_similarity_threshold: float = 0.62, # Limiar de similaridade multimodal para re-ID
        reid_margin_threshold: float = 0.08      # Margem exigida sobre o 2º melhor candidato para desempate
    ):
        self.max_lost_frames = max_lost_frames
        self.reid_threshold = reid_similarity_threshold
        self.reid_margin = reid_margin_threshold

        self.tracks: Dict[str, TrainingTrackState] = {} # kenshi_id -> TrainingTrackState
        self.next_kenshi_num = 1
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
        Processa candidatos detectados no frame atual:
        1. Extrai snapshots visuais e biométricos.
        2. Associa aos tracks ACTIVE/LOST via distância espacial ponderada.
        3. Para candidatos não associados, testa Re-ID contra a galeria DORMANT.
        4. Cria novos tracks se não for Kenshi retornando.
        5. Atualiza máquina de estados (ACTIVE -> LOST -> DORMANT).
        Retorna dicionário: {kenshi_id: landmarks_no_frame_ou_None}
        """
        self.frame_idx += 1
        current_frame_results: Dict[str, Optional[Dict[str, Any]]] = {}

        # 1. Extração de snapshots dos candidatos
        cand_snapshots = [
            KenshiReIDExtractor.extract_snapshot(frame, lm)
            for lm in candidates
        ]

        active_track_ids = [k_id for k_id, t in self.tracks.items() if t.status in ["ACTIVE", "LOST"]]

        matched_tracks = set()
        matched_candidates = set()

        # 2. Associação por Kuhn-Munkres (Hungarian) para tracks atualmente na cena
        if active_track_ids and candidates:
            cost_matrix = np.zeros((len(active_track_ids), len(candidates)), dtype=np.float32)

            for i, t_id in enumerate(active_track_ids):
                track = self.tracks[t_id]
                pred_x = track.last_center_x + track.vx
                pred_y = track.last_center_y + track.vy

                for j, cand_lm in enumerate(candidates):
                    cx, cy = TrainingTrackState._get_center(cand_lm)
                    spatial_dist = np.hypot(pred_x - cx, pred_y - cy)

                    # Bônus de similaridade biométrica se disponível
                    bio_dist = 0.0
                    cand_bio = cand_snapshots[j].biometric_vector
                    if track.signature.biometric_vector is not None and cand_bio is not None:
                        bio_dist = float(np.linalg.norm(track.signature.biometric_vector - cand_bio))

                    # Custo total (distância ponderada)
                    cost = spatial_dist * 0.75 + bio_dist * 0.25
                    cost_matrix[i, j] = cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Limiar máximo aceitável de deslocamento frame a frame (~0.35 do enquadramento)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.38:
                    t_id = active_track_ids[r]
                    matched_tracks.add(t_id)
                    matched_candidates.add(c)
                    track = self.tracks[t_id]
                    track.update_active(candidates[c], cand_snapshots[c], self.frame_idx)
                    current_frame_results[t_id] = candidates[c]

        # 3. Tratar candidatos não associados -> Verificar Re-ID contra Kenshis DORMANT
        unmatched_candidates = [j for j in range(len(candidates)) if j not in matched_candidates]

        for cand_idx in unmatched_candidates:
            cand_lm = candidates[cand_idx]
            cand_snap = cand_snapshots[cand_idx]

            # Criar assinatura temporária do candidato
            cand_sig = KenshiAppearanceSignature("CANDIDATE")
            cand_sig.update(cand_snap)

            best_match_id = None
            best_score = 0.0
            second_score = 0.0
            best_details = {}

            # Buscar melhor correspondência na galeria de kenshis que saíram do enquadramento
            dormant_tracks = [t for t in self.tracks.values() if t.status == "DORMANT"]
            for d_track in dormant_tracks:
                score, details = d_track.signature.compute_similarity(cand_sig)
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_match_id = d_track.kenshi_id
                    best_details = details
                elif score > second_score:
                    second_score = score

            # Re-identificação aceita se score >= limiar e margem suficiente sobre o segundo
            is_reidentified = (
                best_match_id is not None and
                best_score >= self.reid_threshold and
                (best_score - second_score >= self.reid_margin or len(dormant_tracks) == 1)
            )

            if is_reidentified:
                # Kenshi retornou ao enquadramento!
                re_track = self.tracks[best_match_id]
                re_track.mark_reactivated(self.frame_idx, best_score)
                re_track.update_active(cand_lm, cand_snap, self.frame_idx)
                current_frame_results[best_match_id] = cand_lm

                self.total_reidentifications += 1
                self.reidentification_events.append({
                    "kenshi_id": best_match_id,
                    "frame_idx": self.frame_idx,
                    "score": round(best_score, 3),
                    "details": best_details
                })
            else:
                # Novo Kenshi entrando no treinamento
                new_id = self._allocate_kenshi_id()
                new_track = TrainingTrackState(new_id, cand_lm, self.frame_idx)
                new_track.signature.update(cand_snap)
                self.tracks[new_id] = new_track
                current_frame_results[new_id] = cand_lm

        # 4. Tratar tracks que não foram observados no frame atual
        for t_id, track in self.tracks.items():
            if t_id not in current_frame_results:
                current_frame_results[t_id] = None

                if track.status == "ACTIVE":
                    track.mark_lost()
                elif track.status == "LOST":
                    frames_since_seen = self.frame_idx - track.last_seen_frame
                    if frames_since_seen > self.max_lost_frames:
                        # Kenshi saiu do enquadramento da câmera
                        track.mark_dormant(self.frame_idx)
                        self.dormant_gallery[t_id] = track.signature

        # 5. Adicionar aos históricos alinhados de cada track
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
        """Gera sumário completo dos Kenshis rastreados no treinamento."""
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
