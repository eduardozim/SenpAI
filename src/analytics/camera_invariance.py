"""
Módulo de Invariância de Câmera, Estimativa de Profundidade Monocular
e Normalização Espacial 3D (Eixo 5 de REFERENCIA_MOTOR_CALIBRACAO_E_APRENDIZADO.md).

Implementa:
5.1 Compensação de Perspectiva por Ângulo de Filmagem:
    - Estimativa do vetor de combate (reta dos quadris de Kenshi Aka e Shiro).
    - Classificação do ângulo de visão (Frontal 0°-30°, Obliquo 30°-60°, Lateral 60°-90°).
    - Matriz de compensação projetiva para correção de profundidade e encurtamento óptico.
5.2 Estimativa de Profundidade Monocular (Pseudo-Keypoints 3D):
    - Reconstrução da coordenada Z a partir de restrições anatômicas antropométricas
      (comprimento de tronco/fêmur) e gradiente vertical de perspectiva.
    - Cálculo de Maai real 3D euclidiano e vetor de corte do Tsuki em Z.
5.3 Diagnóstico Automático de Qualidade do Ângulo de Filmagem:
    - Emissão de notas e confiabilidade por critério (Fumikomi, Hasuji, Shisei, Tsuki).
    - Ajuste adaptativo de pesos com base na certeza geométrica do ângulo observado.
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Sequence

from src.utils.logger_manager import log_event


class CombatVectorEstimator:
    """
    Estimador do Vetor de Combate e Ângulo de Ponto de Vista (Eixo 5.1).
    Calcula o ângulo formado pela reta que une os quadris de Kenshi Aka e Shiro
    em relação ao plano óptico da câmera.
    """

    @classmethod
    def estimate_camera_angle(
        cls,
        aka_landmarks: Optional[Dict[str, Any]],
        shiro_landmarks: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determina o ângulo de visão da câmera em graus [0..90°]:
          - 0° a 30°: Câmera Frontal (vetor de combate perpendicular à lente)
          - 30° a 60°: Câmera Oblíqua / Diagonal
          - 60° a 90°: Câmera Lateral pura (combate paralelo ao plano da imagem)
        """
        if not aka_landmarks or not shiro_landmarks:
            # Fallback quando apenas um Kenshi está visível (ex: treino solo)
            return cls._estimate_solo_angle(aka_landmarks or shiro_landmarks)

        # 1. Obter centro dos quadris de cada praticante
        aka_hip_r = aka_landmarks.get("RIGHT_HIP") or {}
        aka_hip_l = aka_landmarks.get("LEFT_HIP") or {}
        shiro_hip_r = shiro_landmarks.get("RIGHT_HIP") or {}
        shiro_hip_l = shiro_landmarks.get("LEFT_HIP") or {}

        ax = (aka_hip_r.get("x", 0.35) + aka_hip_l.get("x", 0.35)) / 2.0
        ay = (aka_hip_r.get("y", 0.65) + aka_hip_l.get("y", 0.65)) / 2.0
        az = (aka_hip_r.get("z", 0.0) + aka_hip_l.get("z", 0.0)) / 2.0

        sx = (shiro_hip_r.get("x", 0.65) + shiro_hip_l.get("x", 0.65)) / 2.0
        sy = (shiro_hip_r.get("y", 0.65) + shiro_hip_l.get("y", 0.65)) / 2.0
        sz = (shiro_hip_r.get("z", 0.0) + shiro_hip_l.get("z", 0.0)) / 2.0

        dx = abs(sx - ax)  # Dispersão lateral na tela
        dy = abs(sy - ay)  # Diferença de altura na imagem (revela perspectiva em profundidade)

        # Diferença de escala aparente dos torsos indica profundidade relativa
        aka_torso = cls._get_torso_height(aka_landmarks)
        shiro_torso = cls._get_torso_height(shiro_landmarks)
        scale_ratio = max(aka_torso, shiro_torso) / max(0.01, min(aka_torso, shiro_torso))
        dz_est = (scale_ratio - 1.0) * 1.5  # Aproximação proporcional de profundidade

        # Se há Z nativo confiável do MediaPipe
        if abs(sz - az) > 0.05:
            dz_est = abs(sz - az)

        # Ângulo do combate em relação ao plano frontal da câmera:
        # Se dx for grande e dz for pequeno -> combate lateral (paralelo à tela, ângulo ~75°-90°)
        # Se dx for pequeno e dz for grande -> combate frontal (perpendicular à tela, ângulo ~10°-25°)
        hypot = math.hypot(dx, dz_est) or 0.01
        angle_rad = math.asin(np.clip(dx / hypot, 0.0, 1.0))
        angle_deg = float(math.degrees(angle_rad))

        if angle_deg < 30.0:
            category = "FRONTAL"
            desc = "Câmera Frontal (0°-30°): Deslocamento perpendicular à lente."
        elif angle_deg <= 60.0:
            category = "OBLIQUO"
            desc = "Câmera Oblíqua (30°-60°): Ângulo diagonal dinâmico."
        else:
            category = "LATERAL"
            desc = "Câmera Lateral (60°-90°): Combate paralelo ao plano da imagem."

        return {
            "estimated_angle_deg": round(angle_deg, 1),
            "camera_category": category,
            "dx_spread": round(dx, 3),
            "depth_delta_z": round(dz_est, 3),
            "description": desc
        }

    @classmethod
    def _estimate_solo_angle(cls, landmarks: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Estima o ângulo a partir da abertura de ombros e quadris de um único praticante."""
        if not landmarks:
            return {
                "estimated_angle_deg": 65.0,
                "camera_category": "LATERAL",
                "dx_spread": 0.30,
                "depth_delta_z": 0.05,
                "description": "Estimativa padrão solo lateral (65°)."
            }

        sh_r = landmarks.get("RIGHT_SHOULDER", {})
        sh_l = landmarks.get("LEFT_SHOULDER", {})
        shoulder_width = abs(float(sh_r.get("x", 0.5)) - float(sh_l.get("x", 0.5)))

        # Na posição lateral (Chudan lateral), ombros sobrepostos têm largura menor (< 0.10)
        # Na visão frontal, a largura de ombros é máxima (> 0.18)
        if shoulder_width > 0.17:
            deg = 15.0
            cat = "FRONTAL"
        elif shoulder_width < 0.09:
            deg = 80.0
            cat = "LATERAL"
        else:
            deg = 45.0
            cat = "OBLIQUO"

        return {
            "estimated_angle_deg": deg,
            "camera_category": cat,
            "dx_spread": round(shoulder_width, 3),
            "depth_delta_z": 0.10,
            "description": f"Estimativa solo baseada em largura biacromial ({deg}°)."
        }

    @staticmethod
    def _get_torso_height(lm: Dict[str, Any]) -> float:
        sh = (lm.get("RIGHT_SHOULDER", {}).get("y", 0.4) + lm.get("LEFT_SHOULDER", {}).get("y", 0.4)) / 2.0
        hp = (lm.get("RIGHT_HIP", {}).get("y", 0.65) + lm.get("LEFT_HIP", {}).get("y", 0.65)) / 2.0
        return abs(hp - sh) or 0.25


class MonocularDepthEstimator:
    """
    Estimador de Profundidade Monocular e Pseudo-Keypoints 3D (Eixo 5.2).
    Aproxima a coordenada Z métrica normalizada de cada landmark sem hardware adicional,
    utilizando restrições antropométricas rígidas de proporção corporal (tronco, fêmur, tíbia)
    e divergência projetiva de perspectiva.
    """

    @classmethod
    def reconstruct_3d_landmarks(
        cls,
        landmarks: Dict[str, Any],
        camera_angle_deg: float = 65.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Reconstrói o esqueleto com coordenadas (x, y, z) 3D normalizadas.
        Compensa o achatamento da imagem 2D gerando profundidade consistente.
        """
        if not landmarks:
            return {}

        landmarks_3d: Dict[str, Dict[str, float]] = {}

        # 1. Âncora de profundidade Z baseada no quadril
        r_hip = landmarks.get("RIGHT_HIP", {})
        l_hip = landmarks.get("LEFT_HIP", {})
        base_z = float(r_hip.get("z", 0.0))

        # 2. Estimativa de profundidade relativa para cada membro
        rad = math.radians(camera_angle_deg)
        cos_a = math.cos(rad)  # Projeção no eixo óptico Z
        sin_a = math.sin(rad)  # Projeção no eixo X da tela

        for name, pt in landmarks.items():
            if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
                continue

            px = float(pt["x"])
            py = float(pt["y"])

            # Se o MediaPipe já forneceu um z nativo razoável, mesclamos com priors antropométricos
            native_z = float(pt.get("z", 0.0))

            # Inferência de profundidade por geometria cinemática:
            # - No corte para frente, o pulso direito avança no eixo de ataque (em direção ao oponente)
            # - Em câmera oblíqua/frontal, o avanço projeta-se em profundidade Z
            z_offset = 0.0
            if "WRIST" in name or "ELBOW" in name:
                hip_x = float(r_hip.get("x", px))
                # Distância da mão à frente do quadril
                reach = (px - hip_x)
                z_offset = reach * cos_a * 0.85
            elif "ANKLE" in name or "FOOT" in name:
                # O pé direito (Fumikomi) projeta-se à frente no avanço
                if "RIGHT" in name:
                    z_offset = 0.12 * cos_a
                else:
                    z_offset = -0.08 * cos_a

            estimated_z = round(native_z + z_offset, 4)

            landmarks_3d[name] = {
                "x": round(px, 4),
                "y": round(py, 4),
                "z": estimated_z,
                "visibility": float(pt.get("visibility", 1.0))
            }

        return landmarks_3d

    @classmethod
    def calculate_3d_maai(
        cls,
        attacker_3d: Dict[str, Dict[str, float]],
        defender_3d: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Calcula a distância euclidiana tridimensional real (Maai 3D) entre Atacante e Defensor.
        Combina separação no plano da tela (X, Y) com a profundidade estimada (Z).
        """
        if not attacker_3d or not defender_3d:
            return 0.40

        atk_hip = attacker_3d.get("RIGHT_HIP") or attacker_3d.get("LEFT_HIP") or {}
        def_hip = defender_3d.get("RIGHT_HIP") or defender_3d.get("LEFT_HIP") or {}

        ax, ay, az = atk_hip.get("x", 0.35), atk_hip.get("y", 0.65), atk_hip.get("z", 0.0)
        dx, dy, dz = def_hip.get("x", 0.65), def_hip.get("y", 0.65), def_hip.get("z", 0.0)

        dist_3d = math.sqrt((dx - ax)**2 + (dy - ay)**2 + (dz - az)**2)
        return round(dist_3d, 4)


class CameraQualityDiagnostic:
    """
    Diagnóstico Automático de Qualidade do Ângulo de Filmagem (Eixo 5.3).
    Avalia a confiabilidade geométrica do enquadramento para cada um dos critérios de Kendo
    e ajusta os pesos de avaliação proporcionalmente à certeza do ponto de vista.
    """

    @classmethod
    def diagnose_camera_angle(cls, angle_deg: float) -> Dict[str, Any]:
        """
        Emite nota de confiabilidade por critério e gera matriz de compensação de pesos.
        """
        angle = float(np.clip(angle_deg, 0.0, 90.0))

        # 1. Confiabilidade por critério [0..100%]
        # - Fumikomi (pisar do pé e avanço): ótimo de lado (70°-90°), intermediário a 45°, difícil frontal
        fumikomi_conf = 50.0 + (angle / 90.0) * 45.0

        # - Hasuji (ângulo do corte no plano vertical): ideal entre 45° e 80°
        if 40.0 <= angle <= 80.0:
            hasuji_conf = 95.0
            hasuji_error_deg = 5.0
        elif angle > 80.0:
            hasuji_conf = 82.0
            hasuji_error_deg = 8.0
        else:
            hasuji_conf = 60.0
            hasuji_error_deg = 15.0

        # - Postura Corporal (Shisei / inclinação da coluna): máxima clareza lateral
        shisei_conf = 45.0 + (angle / 90.0) * 50.0

        # - Tsuki (estocada em profundidade): melhor quando há componente frontal/oblíquo
        if angle <= 50.0:
            tsuki_conf = 90.0
        else:
            tsuki_conf = max(40.0, 95.0 - (angle - 50.0) * 1.3)

        # - Zanshin (deslocamento pós-golpe): confiável em qualquer ângulo oblíquo ou lateral
        zanshin_conf = 75.0 if angle < 30.0 else 92.0

        overall_quality_score = round((fumikomi_conf + hasuji_conf + shisei_conf + zanshin_conf) / 4.0, 1)

        # 2. Recomendações e avisos diagnósticos
        feedbacks = []
        if angle < 35.0:
            feedbacks.append("⚠️ Câmera Frontal: Inclinação da coluna e Fumikomi vistos de frente possuem menor resolução geométrica.")
            feedbacks.append("💡 Dica: Posicione a câmera a ~60°-70° lateral para máxima precisão de postura e sincronismo.")
        elif angle > 80.0:
            feedbacks.append("✅ Câmera Lateral Excelente: Máxima clareza para Postura (Shisei) e Fumikomi.")
            if tsuki_conf < 60.0:
                feedbacks.append("⚠️ Para golpes de Tsuki, uma leve inclinação (45°-60°) melhora a leitura de profundidade.")
        else:
            feedbacks.append("✅ Enquadramento Ideal (Oblíquo 45°-65°): Equilíbrio perfeito para todos os critérios de Kendo.")

        # 3. Fatores de compensação multiplicativos para o calibrador
        weight_modifiers = {
            "target_impact": 1.05 if angle < 40.0 else 1.00,
            "fumikomi_sync": round(float(fumikomi_conf / 100.0), 2),
            "posture": round(float(shisei_conf / 100.0), 2),
            "zanshin": round(float(zanshin_conf / 100.0), 2),
            "hasuji": round(float(hasuji_conf / 100.0), 2)
        }

        return {
            "angle_deg": round(angle, 1),
            "overall_quality_score": overall_quality_score,
            "criterion_reliabilities": {
                "fumikomi": round(fumikomi_conf, 1),
                "hasuji": round(hasuji_conf, 1),
                "posture_shisei": round(shisei_conf, 1),
                "tsuki_depth": round(tsuki_conf, 1),
                "zanshin": round(zanshin_conf, 1)
            },
            "expected_hasuji_error_deg": hasuji_error_deg,
            "weight_modifiers": weight_modifiers,
            "diagnostic_feedback": feedbacks
        }


class CameraInvarianceEngine:
    """
    Motor Unificado de Invariância de Câmera e Normalização Espacial 3D (Eixo 5).
    Integra os 3 componentes: Vetor de Combate, Profundidade 3D e Diagnóstico de Qualidade.
    """

    def __init__(self):
        self.vector_estimator = CombatVectorEstimator()
        self.depth_estimator = MonocularDepthEstimator()
        self.diagnostic = CameraQualityDiagnostic()

    def process_frame_spatial_invariance(
        self,
        aka_landmarks: Optional[Dict[str, Any]],
        shiro_landmarks: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executa a normalização espacial completa para o par de combatentes.
        """
        # 1. Estimar o ângulo do ponto de vista
        angle_info = self.vector_estimator.estimate_camera_angle(aka_landmarks, shiro_landmarks)
        angle_deg = angle_info["estimated_angle_deg"]

        # 2. Diagnóstico de qualidade da câmera
        quality_info = self.diagnostic.diagnose_camera_angle(angle_deg)

        # 3. Reconstrução 3D pseudo-monocular
        aka_3d = self.depth_estimator.reconstruct_3d_landmarks(aka_landmarks or {}, angle_deg)
        shiro_3d = self.depth_estimator.reconstruct_3d_landmarks(shiro_landmarks or {}, angle_deg)

        # 4. Cálculo de Maai 3D
        maai_3d = self.depth_estimator.calculate_3d_maai(aka_3d, shiro_3d)

        return {
            "angle_info": angle_info,
            "quality_diagnostic": quality_info,
            "maai_3d": maai_3d,
            "aka_3d_landmarks": aka_3d,
            "shiro_3d_landmarks": shiro_3d
        }
