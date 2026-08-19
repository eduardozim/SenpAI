"""
SenpAI - CLI Principal para Execução de Análise de Lutas de Kendo.
Suporta os 3 Modos Principais de Operação:
1. 'recorded' (Modo de Arbitragem Gravada)
2. 'training' (Modo de Treinamento & Aprendizado)
3. 'realtime' (Modo de Detecção em Tempo Real)
"""

import sys
import os
import argparse

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.pipeline import SenpAIPipeline
from src.utils.demo_generator import generate_demo_kendo_video
from src.engine.feedback_manager import FeedbackManager
from src.utils.settings_manager import get_processing_device
from src.utils.hardware import validate_and_setup_gpu_requirements

def main():
    default_device = get_processing_device()
    parser = argparse.ArgumentParser(description="SenpAI - AI Kendo Match Analysis System")
    parser.add_argument("--video", type=str, help="Caminho para o arquivo de vídeo de luta (.mp4, .avi)")
    parser.add_argument("--output", type=str, default="output_annotated.mp4", help="Caminho para salvar o vídeo anotado")
    parser.add_argument("--profile", type=str, default="normal", choices=["permissivo", "normal", "rigido"],
                        help="Perfil de calibração de arbitragem (permissivo, normal, rigido)")
    parser.add_argument("--mode", type=str, default="recorded", choices=["recorded", "training", "realtime", "user", "learning"],
                        help="Modo de operação: 'recorded' (Arbitragem Gravada), 'training' (Treinamento & Aprendizado) ou 'realtime' (Detecção em Tempo Real)")
    parser.add_argument("--device", type=str, default=default_device, choices=["cpu", "gpu"],
                        help="Dispositivo de processamento: 'cpu' (somente CPU) ou 'gpu' (GPU NVIDIA quando disponível)")
    parser.add_argument("--optimize-profile", action="store_true",
                        help="Executa otimização por reforço no perfil selecionado usando o dataset de feedback registrado")
    parser.add_argument("--demo", action="store_true", help="Gera um vídeo sintético de demonstração e executa o teste")

    args = parser.parse_args()

    # Mapeamento para nomes padronizados
    mode_map = {
        "user": "recorded",
        "learning": "training",
        "recorded": "recorded",
        "training": "training",
        "realtime": "realtime"
    }
    active_mode = mode_map.get(args.mode, "recorded")

    feedback_mgr = FeedbackManager()

    if args.optimize_profile:
        print(f"[SenpAI - Treinamento] Otimizando o perfil '{args.profile}' com base no histórico de feedback...")
        pipeline_temp = SenpAIPipeline(calibration_profile=args.profile, device_preference=args.device)
        curr_cfg = pipeline_temp.calibrator.active_config
        new_cfg, opt_stats = feedback_mgr.optimize_profile_config(args.profile, curr_cfg)
        
        if opt_stats["status"] == "no_data":
            print(f"[SenpAI - Treinamento] {opt_stats['message']}")
        else:
            pipeline_temp.calibrator.update_and_save_profile(args.profile, new_cfg)
            print(f"[SenpAI - Treinamento] Perfil '{args.profile}' recalibrado com sucesso!")
            for chg in opt_stats["changes"]:
                print(f"  - {chg}")
        return

    video_path = args.video

    if args.demo or not video_path:
        print("[SenpAI] Nenhum vídeo fornecido ou modo --demo selecionado. Gerando vídeo de teste sintético...")
        video_path = generate_demo_kendo_video("demo_kendo_match.mp4")
        print(f"[SenpAI] Vídeo sintético criado em: {video_path}")

    print(f"[SenpAI] Modo de Operação: '{active_mode.upper()}'")
    print(f"[SenpAI] Aplicando perfil de calibração: '{args.profile}'")
    print(f"[SenpAI] Preferência de Hardware Solicitada: '{args.device.upper()}'")

    if args.device == "gpu":
        gpu_val = validate_and_setup_gpu_requirements(auto_install=True)
        print(f"[SenpAI - Hardware Check] {gpu_val['message']}")

    pipeline = SenpAIPipeline(calibration_profile=args.profile, device_preference=args.device)
    print(f"[SenpAI] Status de Hardware: {pipeline.device_status_message}")
    
    def on_progress(p):
        print(f"\rProgress: {int(p * 100)}%", end="", flush=True)

    result = pipeline.process_video(video_path, output_video_path=args.output, progress_callback=on_progress)
    print("\n[SenpAI] Processamento concluído!")
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
