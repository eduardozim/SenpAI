"""
Utilitário para geração de vídeos de demonstração sintética de Kendo.
Gera um arquivo .mp4 simulando movimentos de luta para testes do pipeline.
"""

import cv2
import numpy as np
import os

def generate_demo_kendo_video(output_path: str = "demo_kendo_match.mp4", duration_sec: int = 4, fps: int = 30) -> str:
    """
    Cria um vídeo sintético simulando um praticante realizando um golpe de Men.
    """
    width, height = 640, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Parâmetros de animação do boneco
    center_x, center_y = width // 2, height // 2 + 50

    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40 # Fundo escuro do Dojo

        # Simular elevação e descida dos braços (Golpe de Men por volta do frame 45)
        progress = i / total_frames
        
        # Animação da posição das mãos (Men golpeia no frame 45)
        if i < 30:
            # Kamae (postura inicial)
            hand_y = center_y - 30
            foot_x = center_x
        elif i < 45:
            # Furikaburi (elevação)
            hand_y = center_y - 120
            foot_x = center_x + 10
        elif i < 55:
            # Impacto Men
            hand_y = center_y - 100
            foot_x = center_x + 30 # Fumikomi passo a frente
        else:
            # Zanshin / Retorno
            hand_y = center_y - 50
            foot_x = center_x + 40

        # Desenhar figura simplificada (Cabeça, Tronco, Pernas, Braços)
        head_pos = (center_x, center_y - 150)
        hip_pos = (center_x, center_y)
        r_hand = (center_x + 20, hand_y)
        l_hand = (center_x - 20, hand_y + 10)
        r_foot = (foot_x + 20, center_y + 120)
        l_foot = (center_x - 30, center_y + 120)

        # Cabeça (Men)
        cv2.circle(frame, head_pos, 25, (200, 200, 200), -1)
        # Tronco (Keikogi / Do)
        cv2.line(frame, head_pos, hip_pos, (255, 255, 255), 4)
        # Braços
        cv2.line(frame, (center_x, center_y - 120), r_hand, (0, 255, 255), 4)
        cv2.line(frame, (center_x, center_y - 120), l_hand, (0, 255, 255), 4)
        # Shinai (Espada)
        shinai_tip = (r_hand[0] + 40, r_hand[1] - 80)
        cv2.line(frame, r_hand, shinai_tip, (0, 200, 255), 3)
        # Pernas
        cv2.line(frame, hip_pos, r_foot, (180, 180, 180), 4)
        cv2.line(frame, hip_pos, l_foot, (180, 180, 180), 4)

        # Texto informativo no vídeo
        cv2.putText(frame, f"SenpAI Demo - Frame {i}/{total_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    return output_path
