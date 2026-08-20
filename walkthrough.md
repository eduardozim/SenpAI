# Walkthrough de Implementação - SenpAI (AI Kendo Referee & Analytical System)

Implementamos a estrutura completa do sistema **SenpAI** para detecção de golpes em vídeos de lutas de Kendo, avaliação biomecânica dos critérios de *Yuko-datotsu*, crítica textual explicativa e motor de calibração de sensibilidade ajustável.

---

## 🚀 Arquivos Criados e Estrutura do Projeto

* [`config/calibration_profiles.json`](file:///d:/Projetos/SenpAI/Dev/config/calibration_profiles.json): Definição dos perfis de calibração (Rígido, Normal, Permissivo) e pesos de Ki-Ken-Tai-Ichi.
* [`src/vision/pose_detector.py`](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py): Rastreamento de keypoints 3D (33 landmarks) via MediaPipe Pose.
* [`src/vision/shinai_tracker.py`](file:///d:/Projetos/SenpAI/Dev/src/vision/shinai_tracker.py): Projeção vetorial do Kensen (ponta da espada) e zonas de impacto (*Men*, *Kote*, *Do*, *Tsuki*).
* [`src/analytics/event_spotter.py`](file:///d:/Projetos/SenpAI/Dev/src/analytics/event_spotter.py): Algoritmo adaptativo de detecção de eventos e classificação temporal de golpes na luta.
* [`src/analytics/biomechanics.py`](file:///d:/Projetos/SenpAI/Dev/src/analytics/biomechanics.py): Cálculo numérico de precisão do alvo, sincronismo pé-mão (*Fumikomi*), alinhamento da coluna (*Postura*) e *Zanshin*.
* [`src/engine/calibrator.py`](file:///d:/Projetos/SenpAI/Dev/src/engine/calibrator.py): Motor de calibração de sensibilidade com suporte a limiares dinâmicos e customização manual.
* [`src/engine/reporter.py`](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py): Gerador de diagnósticos em português detalhando o status e correções necessárias.
* [`src/pipeline.py`](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py): Pipeline orquestrador end-to-end de vídeo.
* [`main.py`](file:///d:/Projetos/SenpAI/Dev/main.py): CLI interativa para execução do sistema.
* [`app.py`](file:///d:/Projetos/SenpAI/Dev/app.py): Interface Web Interativa em Streamlit com painel de calibração e visualização de vídeos.

---

## 🧪 Testes e Validação Realizados

1. **Ambiente Virtual**: Python 3.11 com aceleração opcional GPU NVIDIA CUDA e MediaPipe Pose.
2. **Execução CLI (`main.py --demo` e `run_tests.py`)**:
   - Suíte de **44 testes automatizados** aprovados com 100% de sucesso.
   - Relatório descritivo estruturado por teste gerado em `logs/senpai_test_report.log`.
3. **Placar Oficial Sanbon-Shobu**:
   - Marcação estrita com caracteres Katakana individuais: `メ` (Men), `コ` (Kote), `ド` (Do) e `ツ` (Tsuki).
4. **Linha do Tempo de Eventos**:
   - Destaque visual nos golpes que geraram marcação de ponto válido (Ippon): `メ MEN`, `コ KOTE`, `ド DO` e `ツ TSUKI`.

---

## 💻 Como Utilizar a Aplicação

### 1. Interface Web (Dashboard Interativo)
Para abrir o painel web com sliders de calibração e reprodutor de vídeo anotado:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

### 2. Linha de Comando (CLI)
Para analisar qualquer vídeo de luta de Kendo gravado:
```powershell
.\.venv\Scripts\python.exe main.py --video "caminho/do/video.mp4" --profile normal
```
Ou para rodar o teste de demonstração rápida:
```powershell
.\.venv\Scripts\python.exe main.py --demo
```
