"""
Módulo de Assinatura Visual e Re-identificação (Re-ID) de Kenshis para Treinamento (SenpAI).
Especializado para cenários de Dojo e Treino de Kendo (sem fitas de Shiai Aka/Shiro):
1. Zekken (frente do Tare: nome e dojo - visão frontal)
2. Tenugui (sob o Men, visto por trás da cabeça / nuca - visão dorsal)
3. Identificação lateral do Tare (dojo e kenshi nas abas laterais - visão de perfil)
4. Biometria corporal (altura normalizada corrigida por perspectiva, proporções anatômicas)
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def _crop_safe(image: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int) -> Optional[np.ndarray]:
    """Recorta uma região da imagem com verificação estrita de limites."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    xmin = max(0, min(w - 1, xmin))
    xmax = max(0, min(w, xmax))
    ymin = max(0, min(h - 1, ymin))
    ymax = max(0, min(h, ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    crop = image[ymin:ymax, xmin:xmax]
    return crop if crop.size > 0 else None


def _extract_color_histogram(crop: np.ndarray, bins: Tuple[int, int] = (16, 16)) -> Optional[np.ndarray]:
    """Extrai histograma 2D H-S normalizado."""
    if crop is None or crop.size < 36:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins[0], bins[1]], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def _compare_histograms(hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]) -> float:
    """Calcula similaridade de correlação de histogramas (0.0 a 1.0)."""
    if hist1 is None or hist2 is None or len(hist1) != len(hist2):
        return 0.0
    # Similaridade de correlação: [-1, 1] mapeada para [0, 1]
    val = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
    if np.isnan(val):
        return 0.0
    return float(np.clip((val + 1.0) / 2.0, 0.0, 1.0))


def _extract_texture_edge_profile(crop: np.ndarray) -> Optional[np.ndarray]:
    """Extrai perfil de gradiente/densidade de bordas (característico de inscrições em tecido/zekken)."""
    if crop is None or crop.size < 36:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32))
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    cv2.normalize(mag, mag, 0, 1, cv2.NORM_MINMAX)
    # Amostragem em 8 regiões para assinatura leve e invariante a ruído
    blocks = [mag[r*8:(r+1)*8, c*8:(c+1)*8].mean() for r in range(4) for c in range(4)]
    return np.array(blocks, dtype=np.float32)


def _compare_feature_vectors(v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
    """Similaridade por cosseno entre vetores (0.0 a 1.0)."""
    if v1 is None or v2 is None or len(v1) == 0:
        return 0.0
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
    return float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))


class KenshiAppearanceSnapshot:
    """Representa a assinatura de aparência extraída de um único frame para um Kenshi."""
    def __init__(
        self,
        orientation: str = "UNKNOWN", # "FRONT", "BACK", "SIDE_LEFT", "SIDE_RIGHT"
        zekken_hist: Optional[np.ndarray] = None,
        zekken_texture: Optional[np.ndarray] = None,
        tenugui_hist: Optional[np.ndarray] = None,
        tenugui_texture: Optional[np.ndarray] = None,
        side_tare_hist: Optional[np.ndarray] = None,
        side_tare_texture: Optional[np.ndarray] = None,
        biometric_vector: Optional[np.ndarray] = None,
        quality_score: float = 0.5,
        best_crop: Optional[np.ndarray] = None
    ):
        self.orientation = orientation
        self.zekken_hist = zekken_hist
        self.zekken_texture = zekken_texture
        self.tenugui_hist = tenugui_hist
        self.tenugui_texture = tenugui_texture
        self.side_tare_hist = side_tare_hist
        self.side_tare_texture = side_tare_texture
        self.biometric_vector = biometric_vector
        self.quality_score = quality_score
        self.best_crop = best_crop


class KenshiAppearanceSignature:
    """
    Assinatura acumulada ao longo do tempo para um Kenshi.
    Combina múltiplas observações para construir um perfil robusto de re-identificação.
    """
    def __init__(self, kenshi_id: str):
        self.kenshi_id = kenshi_id
        self.sample_count = 0

        # Médias acumuladas das pistas
        self.zekken_hist: Optional[np.ndarray] = None
        self.zekken_texture: Optional[np.ndarray] = None
        self.tenugui_hist: Optional[np.ndarray] = None
        self.tenugui_texture: Optional[np.ndarray] = None
        self.side_tare_hist: Optional[np.ndarray] = None
        self.side_tare_texture: Optional[np.ndarray] = None
        self.biometric_vector: Optional[np.ndarray] = None

        self.best_thumbnail: Optional[np.ndarray] = None
        self.best_quality: float = 0.0

    def update(self, snap: KenshiAppearanceSnapshot):
        """Atualiza a assinatura acumulada com uma nova observação ponderada pela qualidade."""
        if snap.quality_score < 0.25:
            return

        alpha = 0.35 if self.sample_count > 0 else 1.0

        def blend_vec(curr: Optional[np.ndarray], new_v: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if new_v is None:
                return curr
            if curr is None:
                return new_v.copy()
            return (1.0 - alpha) * curr + alpha * new_v

        if snap.zekken_hist is not None:
            self.zekken_hist = blend_vec(self.zekken_hist, snap.zekken_hist)
            self.zekken_texture = blend_vec(self.zekken_texture, snap.zekken_texture)

        if snap.tenugui_hist is not None:
            self.tenugui_hist = blend_vec(self.tenugui_hist, snap.tenugui_hist)
            self.tenugui_texture = blend_vec(self.tenugui_texture, snap.tenugui_texture)

        if snap.side_tare_hist is not None:
            self.side_tare_hist = blend_vec(self.side_tare_hist, snap.side_tare_hist)
            self.side_tare_texture = blend_vec(self.side_tare_texture, snap.side_tare_texture)

        if snap.biometric_vector is not None:
            self.biometric_vector = blend_vec(self.biometric_vector, snap.biometric_vector)

        if snap.best_crop is not None and snap.quality_score > self.best_quality:
            self.best_thumbnail = snap.best_crop.copy()
            self.best_quality = snap.quality_score

        self.sample_count += 1

    def compute_similarity(self, candidate: "KenshiAppearanceSignature") -> Tuple[float, Dict[str, float]]:
        """
        Calcula a similaridade multimodal ponderada entre a assinatura acumulada e um candidato.
        Renormaliza os pesos dinamicamente com base nas pistas disponíveis no momento.
        """
        details = {}
        total_weight = 0.0
        weighted_score = 0.0

        # 1. Pista: Zekken Frontal (35% se disponível)
        if self.zekken_hist is not None and candidate.zekken_hist is not None:
            sim_c = _compare_histograms(self.zekken_hist, candidate.zekken_hist)
            sim_t = _compare_feature_vectors(self.zekken_texture, candidate.zekken_texture)
            s_zekken = 0.55 * sim_c + 0.45 * sim_t
            w_zekken = 0.35
            weighted_score += s_zekken * w_zekken
            total_weight += w_zekken
            details["zekken"] = round(s_zekken, 3)

        # 2. Pista: Tenugui Posterior (25% se disponível)
        if self.tenugui_hist is not None and candidate.tenugui_hist is not None:
            sim_c = _compare_histograms(self.tenugui_hist, candidate.tenugui_hist)
            sim_t = _compare_feature_vectors(self.tenugui_texture, candidate.tenugui_texture)
            s_tenugui = 0.60 * sim_c + 0.40 * sim_t
            w_tenugui = 0.25
            weighted_score += s_tenugui * w_tenugui
            total_weight += w_tenugui
            details["tenugui"] = round(s_tenugui, 3)

        # 3. Pista: Tare Lateral (15% se disponível)
        if self.side_tare_hist is not None and candidate.side_tare_hist is not None:
            sim_c = _compare_histograms(self.side_tare_hist, candidate.side_tare_hist)
            sim_t = _compare_feature_vectors(self.side_tare_texture, candidate.side_tare_texture)
            s_sidetare = 0.55 * sim_c + 0.45 * sim_t
            w_sidetare = 0.15
            weighted_score += s_sidetare * w_sidetare
            total_weight += w_sidetare
            details["side_tare"] = round(s_sidetare, 3)

        # 4. Pista: Biometria e Porte Corporal (25% se disponível)
        if self.biometric_vector is not None and candidate.biometric_vector is not None:
            # Similaridade exponencial suave da distância euclidiana entre vetores biométricos
            diff = float(np.linalg.norm(self.biometric_vector - candidate.biometric_vector))
            s_bio = float(np.exp(-1.5 * diff))
            w_bio = 0.25
            weighted_score += s_bio * w_bio
            total_weight += w_bio
            details["biometric"] = round(s_bio, 3)

        if total_weight < 0.10:
            return 0.0, details

        final_score = float(weighted_score / total_weight)
        details["final_score"] = round(final_score, 3)
        return final_score, details


class KenshiReIDExtractor:
    """
    Extrator de assinaturas visuais e biométricas a partir de um frame e landmarks de pose.
    """
    @staticmethod
    def determine_orientation(landmarks: Dict[str, Any]) -> str:
        """Determina se o kenshi está de frente, de costas ou de perfil."""
        nose = landmarks.get("NOSE")
        l_ear = landmarks.get("LEFT_EAR")
        r_ear = landmarks.get("RIGHT_EAR")
        l_sh = landmarks.get("LEFT_SHOULDER")
        r_sh = landmarks.get("RIGHT_SHOULDER")

        if l_sh and r_sh:
            shoulder_width = abs(l_sh["x"] - r_sh["x"])
            # Se ombros estiverem quase sobrepostos no eixo X, é perfil
            if shoulder_width < 0.055:
                return "SIDE_LEFT" if (nose and nose["x"] < l_sh["x"]) else "SIDE_RIGHT"

        # Se nariz estiver claramente visível e entre os ombros -> Frente
        if nose and nose.get("visibility", 0) > 0.45:
            return "FRONT"

        # Se orelhas visíveis e nariz não -> Costas
        if (l_ear and l_ear.get("visibility", 0) > 0.4) or (r_ear and r_ear.get("visibility", 0) > 0.4):
            if not nose or nose.get("visibility", 0) < 0.35:
                return "BACK"

        return "FRONT" # Default neutro

    @staticmethod
    def extract_biometrics(landmarks: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extrai vetor biométrico invariante à escala normalizada do Kenshi:
        [proporção_tronco_pernas, largura_ombros_relativa, largura_quadril_relativa, altura_estimada]
        """
        l_sh = landmarks.get("LEFT_SHOULDER")
        r_sh = landmarks.get("RIGHT_SHOULDER")
        l_hip = landmarks.get("LEFT_HIP")
        r_hip = landmarks.get("RIGHT_HIP")
        l_ank = landmarks.get("LEFT_ANKLE")
        r_ank = landmarks.get("RIGHT_ANKLE")
        nose = landmarks.get("NOSE")

        if not (l_sh and r_sh and l_hip and r_hip):
            return None

        sh_y = (l_sh["y"] + r_sh["y"]) / 2.0
        hip_y = (l_hip["y"] + r_hip["y"]) / 2.0
        torso_h = abs(hip_y - sh_y)
        if torso_h < 0.02:
            return None

        shoulder_w = abs(l_sh["x"] - r_sh["x"]) / torso_h
        hip_w = abs(l_hip["x"] - r_hip["x"]) / torso_h

        legs_h = torso_h
        total_h = torso_h * 2.2
        if l_ank is not None or r_ank is not None:
            if l_ank is not None and r_ank is not None:
                ank_y = (l_ank["y"] + r_ank["y"]) / 2.0
            elif l_ank is not None:
                ank_y = l_ank["y"]
            else:
                ank_y = r_ank["y"]  # type: ignore[index]
            legs_h = max(0.05, abs(ank_y - hip_y))
            head_y = nose["y"] if nose else max(0.0, sh_y - torso_h * 0.4)
            total_h = abs(ank_y - head_y)

        ratio_torso_legs = torso_h / max(0.01, legs_h)

        return np.array([
            float(np.clip(ratio_torso_legs, 0.2, 2.5)),
            float(np.clip(shoulder_w, 0.4, 2.8)),
            float(np.clip(hip_w, 0.3, 2.2)),
            float(np.clip(total_h, 0.1, 1.0))
        ], dtype=np.float32)

    @classmethod
    def extract_snapshot(
        cls,
        frame: Optional[np.ndarray],
        landmarks: Dict[str, Any]
    ) -> KenshiAppearanceSnapshot:
        """Extrai todos os descritores visuais e biométricos para um dado frame."""
        orientation = cls.determine_orientation(landmarks)
        bio_vec = cls.extract_biometrics(landmarks)

        if frame is None or frame.size == 0:
            return KenshiAppearanceSnapshot(orientation=orientation, biometric_vector=bio_vec, quality_score=0.4)

        h, w = frame.shape[:2]

        l_sh = landmarks.get("LEFT_SHOULDER")
        r_sh = landmarks.get("RIGHT_SHOULDER")
        l_hip = landmarks.get("LEFT_HIP")
        r_hip = landmarks.get("RIGHT_HIP")
        nose = landmarks.get("NOSE")
        l_ear = landmarks.get("LEFT_EAR")
        r_ear = landmarks.get("RIGHT_EAR")

        zekken_h = None
        zekken_t = None
        tenugui_h = None
        tenugui_t = None
        side_tare_h = None
        side_tare_t = None
        best_crop = None
        quality = 0.5

        # 1. Recorte do Zekken (quando voltado de frente ou neutro)
        if orientation in ["FRONT", "SIDE_LEFT", "SIDE_RIGHT"] and l_hip and r_hip and l_sh and r_sh:
            hip_mid_x = (l_hip["x"] + r_hip["x"]) / 2.0
            hip_mid_y = (l_hip["y"] + r_hip["y"]) / 2.0
            sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
            torso_h = abs(hip_mid_y - sh_mid_y)

            # Zekken fica na frente do Tare (logo abaixo da cintura até ~40% do tronco para baixo)
            zw = int(abs(l_hip["x"] - r_hip["x"]) * w * 0.90)
            zw = max(18, zw)
            zh = int(torso_h * h * 0.55)
            zh = max(18, zh)

            zx1 = int(hip_mid_x * w - zw // 2)
            zy1 = int(hip_mid_y * h)
            zx2 = zx1 + zw
            zy2 = zy1 + zh

            z_crop = _crop_safe(frame, zx1, zy1, zx2, zy2)
            if z_crop is not None:
                zekken_h = _extract_color_histogram(z_crop)
                zekken_t = _extract_texture_edge_profile(z_crop)
                best_crop = z_crop
                quality += 0.25

        # 2. Recorte do Tenugui (quando de costas / nuca visível)
        if orientation in ["BACK", "SIDE_LEFT", "SIDE_RIGHT"]:
            ref_ear_y = None
            if l_ear and r_ear:
                ear_mid_x = (l_ear["x"] + r_ear["x"]) / 2.0
                ref_ear_y = (l_ear["y"] + r_ear["y"]) / 2.0
            elif l_ear:
                ear_mid_x = l_ear["x"]
                ref_ear_y = l_ear["y"]
            elif r_ear:
                ear_mid_x = r_ear["x"]
                ref_ear_y = r_ear["y"]
            else:
                ear_mid_x = (l_sh["x"] + r_sh["x"]) / 2.0 if (l_sh and r_sh) else 0.5
                ref_ear_y = (l_sh["y"] + r_sh["y"]) / 2.0 - 0.15 if (l_sh and r_sh) else None

            if ref_ear_y is not None:
                tw = int(0.12 * w)
                th = int(0.10 * h)
                tx1 = int(ear_mid_x * w - tw // 2)
                ty1 = int(ref_ear_y * h - th * 0.3)
                tx2 = tx1 + tw
                ty2 = ty1 + th

                t_crop = _crop_safe(frame, tx1, ty1, tx2, ty2)
                if t_crop is not None:
                    tenugui_h = _extract_color_histogram(t_crop)
                    tenugui_t = _extract_texture_edge_profile(t_crop)
                    if best_crop is None:
                        best_crop = t_crop
                    quality += 0.20

        # 3. Recorte Lateral do Tare (quando em perfil)
        if orientation in ["SIDE_LEFT", "SIDE_RIGHT"] and l_hip and r_hip and l_sh and r_sh:
            hip_mid_y = (l_hip["y"] + r_hip["y"]) / 2.0
            sh_mid_y = (l_sh["y"] + r_sh["y"]) / 2.0
            torso_h = abs(hip_mid_y - sh_mid_y)
            chosen_hip = l_hip if orientation == "SIDE_LEFT" else r_hip

            sw = int(0.14 * w)
            sh_box = int(torso_h * h * 0.45)
            sx1 = int(chosen_hip["x"] * w - sw // 2)
            sy1 = int(hip_mid_y * h)
            sx2 = sx1 + sw
            sy2 = sy1 + sh_box

            s_crop = _crop_safe(frame, sx1, sy1, sx2, sy2)
            if s_crop is not None:
                side_tare_h = _extract_color_histogram(s_crop)
                side_tare_t = _extract_texture_edge_profile(s_crop)
                quality += 0.15

        return KenshiAppearanceSnapshot(
            orientation=orientation,
            zekken_hist=zekken_h,
            zekken_texture=zekken_t,
            tenugui_hist=tenugui_h,
            tenugui_texture=tenugui_t,
            side_tare_hist=side_tare_h,
            side_tare_texture=side_tare_t,
            biometric_vector=bio_vec,
            quality_score=min(1.0, quality),
            best_crop=best_crop
        )
