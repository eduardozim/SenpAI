"""
Shinpanai - CLI Principal para Execução de Análise de Lutas de Kendo.
"""

import sys
import os
import argparse

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.pipeline import ShinpanaiPipeline
from src.utils.demo_generator import generate_demo_kendo_video

def main():
    parser = argparse.ArgumentParser(description="Shinpanai - AI Kendo Match Analysis System")
    parser.add_argument("--video", type=str, help="Caminho para o arquivo de vídeo de luta (.mp4, .avi)")
    parser.add_argument("--output", type=str, default="output_annotated.mp4", help="Caminho para salvar o vídeo anotado")
    parser.add_argument("--profile", type=str, default="normal", choices=["rigido", "normal", "permissivo"],
                        help="Perfil de calibração de arbitragem (rigido, normal, permissivo)")
    parser.add_argument("--demo", action="store_true", help="Gera um vídeo sintético de demonstração e executa o teste")

    args = parser.parse_args()

    video_path = args.video

    if args.demo or not video_path:
        print("[Shinpanai] Nenhum vídeo fornecido ou modo --demo selecionado. Gerando vídeo de teste sintético...")
        video_path = generate_demo_kendo_video("demo_kendo_match.mp4")
        print(f"[Shinpanai] Vídeo sintético criado em: {video_path}")

    print(f"[Shinpanai] Iniciando processamento do vídeo: {video_path}")
    print(f"[Shinpanai] Aplicando perfil de calibração: '{args.profile}'")

    pipeline = ShinpanaiPipeline(calibration_profile=args.profile)
    
    def on_progress(p):
        print(f"\rProgress: {int(p * 100)}%", end="", flush=True)

    result = pipeline.process_video(video_path, output_video_path=args.output, progress_callback=on_progress)
    print("\n[Shinpanai] Processamento concluído!")
    print("=" * 60)
    print(f"Vídeo: {result['video_path']}")
    print(f"Duração: {result['duration_seconds']}s ({result['total_frames']} frames)")
    print(f"Perfil de Arbitragem Aplicado: {result['profile_applied']}")
    print(f"Golpes Detectados: {result['events_detected_count']}")
    print("=" * 60)

    for idx, ev_data in enumerate(result["events"]):
        print(f"\n--- GOLPE #{idx+1} ---")
        print(ev_data["diagnostic_report"])
        print("-" * 60)

if __name__ == "__main__":
    main()
