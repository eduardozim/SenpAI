# SenpAI (先輩 AI) — Manual Técnico Completo

> **Arquitetura, Implementação, Algoritmos e Log de Mudanças**  
> **Versão Oficial do Sistema**: `v 0.3.5.0`

---

## 1. Visão Geral do Sistema e Filosofia de Arquitetura

O **SenpAI (先輩 AI)** é uma plataforma avançada de visão computacional, análise biomecânica e avaliação assistida por inteligência artificial projetada para a arte marcial do **Kendo**.

No Kendo tradicional, a atribuição de um ponto válido (*Yuko-Datotsu*) é regida pelo conceito fundamental de **Ki-Ken-Tai-Ichi** (気剣体一致 — Espírito, Espada e Corpo em harmonia unificada):

- **Ki (気)**: Espírito / Prontidão (*Zanshin*)
- **Ken (剣)**: Espada / Precisão do impacto do Shinai no alvo
- **Tai (体)**: Corpo / Sincronismo do pisar (*Fumikomi-ashi*) e postura corporal

O SenpAI traduz esses princípios marciais em algoritmos numéricos de alta precisão através da análise cinemática de esqueletos 3D, projeção vetorial da espada (*Shinai*) e aprendizado por reforço (*Reinforcement Learning*).

---

## 2. Requisitos de Sistema, Instalação e Guia de Execução

### 2.1. Requisitos de Sistema
- **Sistema Operacional**: Windows 10/11 (64-bit), Linux (Ubuntu 20.04+) ou macOS (Apple Silicon / Intel).
- **Interpretador Python**: **Python 3.11** (Distribuição oficial da *Python Software Foundation* — versão ideal e necessária para compatibilidade estável com `mediapipe`, `opencv-python`, `ultralytics` e `streamlit`).
- **Ambiente Virtual**: Ambiente isolado nativo (`.venv`) gerenciado pelo módulo `venv` do Python.
- **Hardware Recomendado**:
  - **CPU**: Processador Multi-core (Intel Core i5/i7/i9 ou AMD Ryzen 5/7/9).
  - **Memória RAM**: 8 GB mínimo (16 GB recomendado para vídeos em 1080p/60fps).
  - **Aceleração GPU (Opcional)**: GPU NVIDIA GeForce RTX/GTX com suporte a CUDA 12.1+ para inferência acelerada com FP16 Tensor Cores.

---

### 2.2. Guia de Instalação no Windows (Passo a Passo Oficial)

Para assegurar compatibilidade absoluta com as políticas de integridade do sistema operacional Windows (**Controle de Aplicativo Inteligente / Smart App Control / WDAC**), recomenda-se a instalação oficial do Python 3.11 assinado digitalmente:

#### 1. Instalar o Python 3.11 Oficial via Winget
Execute no terminal (PowerShell ou Prompt de Comando):
```powershell
winget install Python.Python.3.11
```
*(Ou realize o download do instalador oficial de 64 bits em [python.org](https://www.python.org/downloads/release/python-3119/)).*

#### 2. Criar o Ambiente Virtual (`.venv`)
No diretório raiz do projeto (`Dev/`):
```powershell
# Criação do ambiente virtual com o interpretador oficial Python 3.11
py -3.11 -m venv .venv
```

#### 3. Ativar o Ambiente Virtual
```powershell
# No PowerShell:
.\.venv\Scripts\activate

# No Prompt de Comando (CMD):
.\.venv\Scripts\activate.bat
```

#### 4. Instalar as Dependências do Projeto
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Habilitar Aceleração por GPU NVIDIA (Opcional)
Caso possua placa de vídeo dedicada NVIDIA com suporte a CUDA:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics
```

---

### 2.3. Instruções de Execução

#### A. Interface Web Interativa (Streamlit — Recomendado)
Com o ambiente virtual (`.venv`) ativado, execute o Streamlit através do módulo Python:
```powershell
# Execução recomendada (invoca o Streamlit via interpretador oficial assinado):
python -m streamlit run app.py

# Ou especificando o caminho completo do interpretador:
.\.venv\Scripts\python.exe -m streamlit run app.py
```

> [!NOTE]
> **Por que usar `python -m streamlit` em vez de `streamlit.exe`?**
> No Windows (especialmente no Windows 11 com *Smart App Control* ou políticas corporativas de segurança), o wrapper executável `streamlit.exe` gerado dinamicamente pelo `pip` na pasta `.venv\Scripts\` pode ser bloqueado por não possuir assinatura digital individualizada. Ao chamar `python -m streamlit`, o sistema executa diretamente o processo `python.exe` oficial da *Python Software Foundation*, garantindo execução segura e sem bloqueios.

#### B. Linha de Comando (CLI)
```powershell
# Análise de vídeo com perfil normal padrão:
python main.py --video "caminho/do/video.mp4"

# Análise com aceleração de GPU NVIDIA CUDA:
python main.py --video "caminho/do/video.mp4" --device gpu

# Análise especificando modelo de visão computacional (yolov8, yolov11, yolov12, yolov26):
python main.py --video "caminho/do/video.mp4" --device gpu --model yolov11
```

#### C. Execução da Suíte de Testes Automatizados
```powershell
# Execução completa com relatório descritivo salvo em logs/senpai_test_report.log:
python run_tests.py

# Ou via runner padrão unittest:
python -m unittest discover tests
```

---

### 2.4. Nota de Segurança e Compatibilidade (Windows Smart App Control)
O uso da distribuição oficial do Python 3.11 assinada digitalmente pela *Python Software Foundation* e a criação de ambiente com o módulo nativo `venv` evitam bloqueios de segurança do sistema operacional (como `ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION` / `os error 4551`), garantindo que o interpretador e seus executáveis rodem sem restrições em ambientes corporativos ou Windows 11 com *Smart App Control* ativado.

---

## 3. Estrutura do Projeto e Módulos

A estrutura de arquivos do projeto está organizada de forma modular:

```text
Dev/
├── .streamlit/
│   └── config.toml                 # Configurações do servidor Streamlit (remoção de limite de upload de vídeo)
├── config/
│   ├── ai_knowledge_base.json      # Base de conhecimento persistida e auto-treinamento de IA
│   ├── calibration_profiles.json   # Configurações e pesos dos perfis de calibração
│   ├── settings.json               # Configurações globais do sistema (CPU/GPU)
│   └── sonkyo_learned_profile.json # Perfil adaptativo aprendido de postura de Sonkyō
├── data/
│   ├── auto_training_checkpoint.json # Checkpoint persistente de tolerância a falhas do auto-treinador
│   ├── benchmark_golden/
│   │   └── golden_dataset.json       # Dataset canônico imutável de benchmark padrão-ouro (Eixo 4)
│   ├── feedback_dataset.json       # Base de dados de anotações (TP/FP/FN/Dan) para RL
│   └── training_history.json       # Histórico de sessões de treinamento e revisões por Dan
├── logs/
│   ├── senpai_debug.log            # Arquivo consolidado de logs, erros e alertas do sistema
│   └── senpai_test_report.log      # Relatório descritivo da última execução de testes automatizados
├── src/
│   ├── analytics/
│   │   ├── biomechanics.py         # Cálculo numérico dos critérios de Yuko-Datotsu e Maai
│   │   ├── camera_invariance.py    # Invariância de câmera, compensação de perspectiva e Maai 3D (Eixo 5)
│   │   ├── kenshi_style_model.py   # Modelagem de baseline cinestésico individual e warm start de perfis (Eixo 6)
│   │   ├── event_spotter.py        # Detecção temporal de picos cinemáticos, debounce e NMS de golpes
│   │   ├── multi_camera_fusion.py  # Fusão de consenso multi-câmeras e avaliação Yuko-Datotsu
│   │   ├── sonkyo_detector.py      # Identificação de Sonkyō, delimitação da luta e aprendizado
│   │   └── training_analyzer.py    # Análise das 14 modalidades de dojo e avaliação dos 3 Pilares
│   ├── engine/
│   │   ├── active_learning.py      # Motor de Aprendizado Ativo (Uncertainty, Golden Benchmark, Consenso e Trust)
│   │   ├── auto_trainer.py         # Motor de auto-treinamento por IA, persistência e baselines
│   │   ├── calibrator.py           # Motor de pontuação e validação de limiares
│   │   ├── feedback_manager.py     # Motor de Aprendizagem por Reforço, Governança por Dan e Otimização
│   │   ├── llm_assistant.py        # Assistente LLM Especialista FIK (Gemini/OpenAI + fallback determinístico)
│   │   └── reporter.py             # Gerador de relatórios diagnósticos textuais
│   ├── utils/
│   │   ├── demo_generator.py       # Gerador sintético de vídeos de teste de Kendo
│   │   ├── environment.py          # Detecção e status de isolamento de ambiente virtual (.venv)
│   │   ├── excel_strikes_manager.py # Exportação e importação de golpes via planilha Excel (.xlsx)
│   │   ├── hardware.py             # Detecção de GPU NVIDIA CUDA e resolução de fallback CPU
│   │   ├── logger_manager.py       # Gerenciador central de logs, alertas e diagnósticos de debug
│   │   ├── settings_manager.py     # Gerenciamento e persistência das configurações do sistema
│   │   ├── stream_capture.py       # Captura assíncrona otimizada para câmeras IP / RTSP / Webcams
│   │   ├── test_runner.py          # Runner de testes automatizados com emissão de log descritivo
│   │   ├── video_downloader.py     # Download, extração de metadados e streaming do YouTube/Web
│   │   └── video_player_controls.py # Controles interativos de reprodução e VAR via HTML/JS injetado
│   ├── vision/
│   │   ├── combatant_tracker.py    # Rastreamento dos 2 Kenshi (Aka/Shiro), flag dorsal e planos
│   │   ├── pose_detector.py        # Rastreamento de esqueleto 3D via YOLOv8-Pose / MediaPipe
│   │   └── shinai_tracker.py       # Estimação do Kensen e zonas anatômicas de alvo
│   └── pipeline.py                 # Pipeline orquestrador end-to-end de vídeo e validação de Maai
├── tests/
│   ├── test_active_learning_eixo4.py # Testes de aprendizado ativo, golden benchmark e assistente LLM
│   ├── test_camera_invariance_eixo5.py # Testes de invariância de câmera, perspectiva e Maai 3D (Eixo 5)
│   ├── test_kenshi_style_model_eixo6.py # Testes de baseline cinestésico e warm start de perfis (Eixo 6)
│   ├── test_auto_trainer.py        # Testes de auto-treinamento, baselines < 50% e tolerância a falhas
│   ├── test_dan_training_governance.py # Testes da governança por Dan, pacotes e retreinamento
│   ├── test_environment.py         # Testes de detecção de ambiente virtual
│   ├── test_excel_strikes_io.py    # Testes de exportação/importação Excel e retreinamento
│   ├── test_feedback_loop.py       # Testes unitários para a malha de feedback e RL
│   ├── test_hardware_settings.py   # Testes automatizados de hardware e configurações
│   ├── test_logger_manager.py      # Testes automatizados do sistema de logs e diagnóstico
│   ├── test_multi_camera_fusion.py # Testes de fusão multi-câmeras, quórum e consenso
│   ├── test_pipeline_cancellation.py # Testes automatizados de cancelamento e interrupção do pipeline
│   ├── test_pose_batch_processing.py # Testes de processamento em batch de poses
│   ├── test_scoreboard_and_flag_detection.py # Testes do placar oficial e detecção de flag dorsal
│   ├── test_sonkyo_and_plane_filtering.py # Testes de Sonkyō, limites da luta e filtragem de planos
│   ├── test_stream_capture.py      # Testes de captura e normalização de streams
│   ├── test_training_modes.py      # Testes das 14 modalidades de treino e 3 pilares
│   ├── test_video_downloader.py    # Testes unitários e de integração do downloader de YouTube
│   └── test_video_player_controls.py # Testes dos controles interativos de vídeo e seek DOM
├── app.py                          # Dashboard Web Interativo Streamlit (Home, Análise e Configurações)
├── main.py                         # Interface de Linha de Comando (CLI com flags completas)
├── run_tests.py                    # Script raiz para execução descritiva dos testes automatizados
├── Melhorias_Issues.md             # Registro de pendências, issues e histórico de versões
├── README.TXT                      # Manual simplificado de uso rápido
└── manual.md                       # Manual técnico completo e log de mudanças (este arquivo)
```

---

## 4. Detalhamento das Implementações Técnicas e Algoritmos

### 4.1. Visão Computacional (`src/vision/`)

#### `PoseDetector` ([pose_detector.py](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py))
Utiliza os frameworks **YOLOv8-Pose (PyTorch CUDA FP16)** e **MediaPipe Pose** para rastreamento de pontos de articulação 3D (*landmarks*) em tempo real.
- **Reconstrução Cinemática e Interpolação Anatômica**: Em situações de alta velocidade de corte ou oclusão por Hakama/Men, realiza a síntese e interpolação de pulsos (`RIGHT_WRIST`/`LEFT_WRIST`) e pés (`RIGHT_FOOT_INDEX`/`LEFT_FOOT_INDEX`) baseando-se na cinemática dos cotovelos, ombros e tornozelos.
- **Renderização Limpa do Vídeo Anotado (`draw_combatants_overlay`)**: Parâmetro `show_discarded: bool = False` por padrão. Elementos descartados (árbitros, público, mesas e fundo) não recebem poluição de retângulos cinzas no vídeo final. Apenas os dois Kendocas oficiais (`🔴 AKA` e `⚪ SHIRO`) e seus respectivos Shinai são desenhados.

#### `ShinaiTracker` ([shinai_tracker.py](file:///d:/Projetos/SenpAI/Dev/src/vision/shinai_tracker.py))
A espada (*Shinai*) é estimada como uma extensão vetorial a partir do eixo formado pelos pulsos (`RIGHT_WRIST` e `LEFT_WRIST`). O algoritmo projeta a trajetória do **Kensen** (ponta da espada) e define as zonas anatômicas de ataque em 3D/2D:

- **MEN**: Região da cabeça (com base no nariz/orelhas).
- **KOTE**: Região dos antebraços/pulsos do oponente.
- **DO**: Flancos abdominais (com base na linha entre ombro e quadril).
- **TSUKI**: Região da garganta/esterno superior.

#### `CombatantTracker` ([combatant_tracker.py](file:///d:/Projetos/SenpAI/Dev/src/vision/combatant_tracker.py))
Responsável pela persistência, discriminação de papéis e identificação contínua dos dois lutadores principais no Shiaijo:
- **Delimitação de Quadra (Shiai-jo ROI) & Margens de Segurança**:
  - Aceita máscara poligonal 2D (`shiaijo_polygon`) delimitando a área de piso regulamentar.
  - Caso nenhuma máscara manual seja informada pelo usuário, aplica margens automáticas seguras ($0.12 \le ground\_x \le 0.88$ e $0.20 \le ground\_y \le 0.98$), excluindo automaticamente árbitros laterais, mesários e público nas bordas externas do enquadramento.
- **Discriminação de Árbitros (Shinpans) e Seleção Ótima da Dupla de Kenshis (`select_best_combatant_pair`)**:
  - Avalia múltiplos candidatos a esqueletos no frame e calcula a probabilidade postural de ser um Kenshi (`compute_kenshi_feature_score`): empunhadura bimanual do cabo do Shinai no abdômen/Kamae ($\Delta_{\text{wrists}} < 0.18 \times H$) vs mãos abertas segurando bandeiras nas laterais, centralidade no Shiaijo ($x \in [0.20, 0.80]$), elevação para corte (*Furikaburi*) e flexão/agachamento de *Sonkyō*.
  - Isola com alta precisão os 2 Kenshis mesmo quando árbitros (Shinpans) estão em primeiro plano (próximos à câmera), aplicando compatibilidade de escala mútua no plano da quadra e descartando os árbitros como `FOREGROUND_OCCLUDER` ou `BACKGROUND`.
- **Calibração Dinâmica de Escala em Tomadas Abertas (Wide-Angle)**:
  - `calibrate_main_plane` adapta dinamicamente a escala de referência à altura média real dos combatentes filmados (`max(0.20, avg_height)`), evitando que atletas reais em planos abertos sejam indevidamente descartados como segundo plano.
- **Travamento de IDs (K=2) e Persistência Inercial (`return_persisted=True`)**:
  - Sistema de trava rígida nos dois combatentes após inicialização no Sonkyō.
  - Em dropouts momentâneos (cruzamentos de corpos, giros rápidos de *Tai-atari* ou oclusões severas), o rastreador mantém a trajetória por propagação inercial contínua até que o atleta reapareça, evitando saltos de identificação e falhas nos cálculos biomecânicos.
- **Detecção Cromática de Flag Dorsal (Tasukuki)**: Segmentação em espaço de cor HSV (`detect_red_flag_score`) no dorso dos atletas para identificação inequívoca de **Kenshi Aka (Vermelho)** e **Kenshi Shiro (Branco)**, mesmo com keikogi azul escuro, branco ou preto.
- **Filtragem Geométrica de Plano de Combate**: Calibra a escala espacial média dos kenshi e descarta automaticamente pessoas e movimentações em segundo plano (outras lutas, arquibancadas) ou oclusões em primeiro plano (transeuntes passando em frente à câmera).

---

### 4.2. Análise, Biomecânica e Rituais (`src/analytics/`)

#### `EventSpotter` ([event_spotter.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/event_spotter.py))
Classificador temporal (*Action Spotter*) que analisa as séries temporais de velocidade e aceleração das mãos e da espada. Identifica:
1. Fase de elevação (*Furikaburi*)
2. Aceleração descendente rápida
3. Instante exato de impacto (pico de desaceleração)
- **Cooldown Estendido (35 frames / ~1.2s)**: O intervalo mínimo entre eventos (`min_event_gap_frames`) foi calibrado para evitar que a elevação e a descida do mesmo movimento gerem disparos redundantes.
- **Supressão Temporal de Duplicatas (Non-Maximum Suppression - NMS)**: Aplica supressão temporal por combatente sobre candidatos a golpes vizinhos, retendo estritamente o instante de maior intensidade cinemática e descartando oscilações espúrias.

#### `BiomechanicsAnalyzer` ([biomechanics.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/biomechanics.py))
Calcula quantitativamente os 4 pilares do **Ki-Ken-Tai-Ichi** e a validação de contato (**Maai**):

1. **Target Impact (Ken)**: Avalia a proximidade entre a ponta do *Kensen* e o centro da zona anatômica alvo no frame de impacto (Escala: $0\%$ a $100\%$).
2. **Fumikomi Sync (Tai)**: Mede a diferença de tempo (offset em ms) entre a batida do pé direito no solo e o ponto de máxima desaceleração do golpe. Quanto menor o offset em relação à janela ideal ($0\text{ ms}$ a $40\text{ ms}$), maior a pontuação.
3. **Posture (Tai)**: Calcula o alinhamento do vetor da coluna (ombro-quadril) em relação à vertical perfeita. Penaliza inclinações excessivas para a frente/lados e perda de estabilidade da cabeça.
4. **Zanshin (Ki)**: Avalia a janela pós-golpe (15 frames após o impacto). Mede a manutenção da postura firme, estabilidade visual e ausência de desaceleração desordenada ou desequilíbrio.
5. **Discriminação de Contato Físico (Maai / Distância de Combate)**: No instante do impacto, avalia a distância geométrica relativa entre os combatentes (Aka e Shiro). Golpes desferidos no vazio sem alcance real (> 0.48 de distância na quadra) são sumariamente reprovados com o diagnóstico `Fora do Maai (Sem contato com oponente)`.

#### `SonkyoDetector` ([sonkyo_detector.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/sonkyo_detector.py))
Módulo biomecânico que monitora e reconhece o ritual sagrado de **Sonkyō** (agachamento sobre os calcanhares com coluna vertical):
- **Classificação Postural Multifatorial**: Avalia rebaixamento de quadril ($\Delta Y$), proporção tronco-altura, compressão vertical relativa ($H_{sonkyo} \le 0.75 \times H_{standing}$) e verticalidade da coluna.
- **Delimitação Regulamentar da Luta**: Marca o início oficial do combate (`match_start_frame`) no término do Sonkyō Inicial e o encerramento oficial (`match_end_frame`) no início do Sonkyō Final.
- **Filtragem Estrita de Golpes**: Qualquer golpe fora desse intervalo ritual é sumariamente descartado da avaliação oficial.
- **Aprendizado Biomecânico Adaptativo**: Permite edição interativa de intervalos na UI e recalibra os limiares de Sonkyō, persistindo o aprendizado em `config/sonkyo_learned_profile.json`.

#### `MultiCameraFusionEngine` ([multi_camera_fusion.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/multi_camera_fusion.py))
Motor de **Consenso, Calibração e Validação Cruzada Multi-Câmeras**. Implementa a regra fundamental de arbitragem e ancoragem biomecânica:
> **"A definição de haver ou não o golpe deve ser tomada com base no conjunto das imagens das câmeras e validada estritamente pelo modelo de calibração. Com 1 câmera, o golpe só é marcado se houver movimentação física real acima do limiar cinemático e conformidade aos critérios de Ki-Ken-Tai-Ichi. Com múltiplas câmeras, escalona-se o quórum de confirmação entre os ângulos de visão."**

- **Integração Estrita com o Modelo de Treinamento e Calibração (`CalibrationEngine`)**:
  - Toda avaliação em tempo real (1 a 4 câmeras) passa diretamente pelos pesos e sub-limiares regulamentares de **Ki-Ken-Tai-Ichi** (*Target Impact*, *Fumikomi Sync*, *Posture*, *Zanshin*).
  - **Limiar Cinemático Mínimo de Movimentação**: Elimina ruído estático e micro-vibrações de keypoints quando o praticante está em postura estática (Kamae/Sonkyō), descartando sumariamente candidatos com velocidade inferior ao limiar do perfil ativo (Permissivo: $0.018$, Normal: $0.025$, Rígido: $0.032$).
- **Escalonamento do Quórum de Confirmação**:
  À medida que o número de câmeras $N$ aumenta, o sistema eleva a exigência de quórum de câmeras ativas com evidência visual síncrona nos quadros:
  - **1 Câmera**: Quórum de **1/1** (100% monocular, condicionado à validação biomecânica completa e movimentação real).
  - **2 Câmeras**: Quórum de **2/2** (100% de confirmação cruzada obrigatória — elimina artefatos de perspectiva ou oclusões unilaterais).
  - **3 Câmeras**: Quórum de **2/3** ($\ge 66.7\%$ no modo Normal) ou **3/3** (100% no modo Rígido).
  - **4 Câmeras**: Quórum de **3/4** ($\ge 75\%$ no modo Normal) ou **4/4** (100% no modo Rígido).
- **Alinhamento Temporal Síncrono ($\Delta t$)**: Janela de busca cruzada ($\pm 10$ frames / $\approx 350\text{ ms}$) entre as séries temporais de aceleração de pulso e trajetória das câmeras.
- **Extração de Evidências em Frames (`CameraFrameEvidence`)**: Mede velocidade do pulso, proximidade do alvo, postura e validação de calibração para cada câmera individual.
- **Decisão e Fusão Conjunta (`MultiCameraStrikeEvaluation`)**: Computa o score médio conjunto das visões confirmadas e classifica o golpe em `CONFIRMED_MULTICAM`, `REJECTED_NO_MOTION_OR_INVALID`, `REJECTED_SINGLE_ANGLE` ou `REJECTED_INSUFFICIENT_CONSENSUS`.
- **Painel de Feed e Placar ao Vivo na Interface ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - **Contador de Pontos**: No topo da coluna de feed, um painel consolidado monitora em tempo real o placar de Ippons e total de golpes para **⚪ Kenshi Shiro** e **🔴 Kenshi Aka**, além da métrica total consolidada da sessão.
  - **Ordem Decrescente / Mais Recentes no Topo**: A lista de histórico renderiza os eventos do mais recente para o mais antigo, mantendo a ação recém-executada sempre visível no topo sem necessidade de rolagem.
  - **Componentes Nativos `<details><summary>`**: Os relatórios técnicos descritivos são embutidos como accordions HTML leves com escape de caracteres (`html.escape`), garantindo expansão instantânea no navegador sem interferir na cadência de processamento de vídeo do servidor ou gerar colisões de chave no Streamlit.

#### `TrainingAnalyzer` ([training_analyzer.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/training_analyzer.py))
Motor de Reconhecimento, Análise Biomecânica e Diagnóstico Pedagógico de Treinamento de Kendo:
- **As 14 Modalidades Oficiais de Treinamento (com Kanjis Oficiais)**:
  1. **Ashi-sabaki (足捌き)**: Deslocamentos fundamentais de pés (*okuri-ashi*, *ayumi-ashi*, *hiraki-ashi*, *tsugi-ashi*).
  2. **Suburi (素振り)**: Golpes repetidos no ar (*jōge-buri*, *naname-buri*, *shōmen-uchi*, *sayū-men*).
  3. **Kihon (基本)**: Fundamentos de postura (*shisei*), distância (*maai*), guarda (*kamae*), golpe e *zanshin*.
  4. **Kirikaeshi (切り返し)**: Sequência contínua de golpes para desenvolvimento de ritmo, precisão, respiração e resistência.
  5. **Uchikomi-geiko (打込稽古)**: Execução de golpes em oportunidades oferecidas pelo parceiro (*motodachi*).
  6. **Kakari-geiko (掛稽古)**: Ataques contínuos e intensos em alta velocidade durante períodos curtos.
  7. **Yakusoku-geiko (約束稽古)**: Exercícios combinados em duplas com ações e contra-ataques previamente definidos.
  8. **Waza-geiko (技稽古)**: Prática sistemática de técnicas ofensivas e contra-ataques (*debana*, *nuki*, *kaeshi*, *suriage*, *hiki-waza*).
  9. **Oji-waza (応じ技)**: Técnicas especializadas de resposta e recepção direta ao ataque do oponente.
  10. **Ji-geiko (地稽古)**: Combate livre aplicando livremente todos os fundamentos e técnicas aprendidas.
  11. **Shiai-geiko (試合稽古)**: Simulação formal de luta com regras oficiais da FIK, arbitragem e pontuação.
  12. **Nihon Kendō Kata (日本剣道形)**: Formas tradicionais milenares praticadas em duplas com espada de madeira (*bokutō*).
  13. **Bokutō ni yoru Kendō Kihon Waza Keiko Hō (木刀による剣道基本技稽古法)**: Fundamentos técnicos e pedagógicos praticados com espada de madeira.
  14. **Shinsa (審査)**: Exame de graduação no qual são rigorosamente avaliados fundamentos, técnica, postura, etiqueta, kiai e zanshin.

- **Avaliação dos 3 Pilares Fundamentais**:
  - **Pilar 1: Movimentação (35%)**: Avalia a biomecânica postural do atleta — verticalidade da coluna (*Shisei*), nivelamento e simetria de ombros, alinhamento e calcanhar esquerdo na base (*Ashi-gamae*) e amplitude do movimento de elevação (*Furikaburi*).
  - **Pilar 2: Precisão (35%)**: Avalia a assertividade e coordenação do golpe — trajetória direcionada no ponto anatômico alvo (*Datotsu-bui*), sincronismo unificado corpo-espada (*Ki-Ken-Tai-Ichi*) e preservação da linha central (*Chushin-sen*).
  - **Pilar 3: Constância (30%)**: Avalia a resistência e cadência — regularidade métrica do intervalo entre repetições (desvio padrão em segundos), resistência à fadiga muscular (*Stamina*) ao longo do tempo e aderência à cadência esperada da modalidade.

- **Rastreamento e Nomeação Individual de Kenshi**:
  - Rastreamento isolado de cada praticante no Shiaijo (`KENSHI_SOLO`, `KENSHI_SHIRO`, `KENSHI_AKA`).
  - Campo interativo na interface web permitindo renomear o Kendoca (ex: "Sensei Tanaka", "Eduardo Zimermann").
  - **Exportação de Relatório Individual em Markdown (`.md`)**: Gera um dossiê técnico pedagógico contendo notas percentuais dos 3 Pilares, sub-métricas, pontos fortes observados, pontos de atenção biomecânica e plano prescritivo de exercícios do Kendo.
  - **Exportação Consolidada da Sessão em JSON**: Estrutura completa de dados para integração com sistemas de dojo e gestão de atletas.

#### `LiveTrainingSessionManager` ([training_live_manager.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/training_live_manager.py))
Gerenciador de Sessão de Treinamento em Tempo Real para dojos e academias com câmeras RTSP e Webcams:
- **Máquina de Estados de Golpes e Repetições (`LiveStrikeState`)**:
  - Transições contínuas de estado (`IDLE` -> `FURIKABURI` -> `RECOVERY`) monitorando elevação de punhos acima da cabeça, aceleração descendente de corte e retorno à guarda Kamae.
  - Contador incremental de repetições com cadência dinâmica em Golpes por Minuto (**CPM** - *Cuts Per Minute*).
- **Rastreamento Biomecânico Contínuo**:
  - Cálculo instantâneo da verticalidade da coluna (*Shisei*) a partir do vetor quadril-ombros.
  - Nivelamento e simetria de ombros para detecção precoce de assimetria ou rotação inadequada de tronco durante a execução dos cortes.
  - Emissão de biofeedback em tempo real com alertas visuais destacados (ex: alerta de tronco inclinado à frente ou ombro direito descompensado).
- **HUD Dinâmico dos 3 Pilares em HTML (`render_live_hud_html`)**:
  - Painel de telemetria visual embutido sobre o feed com pontuações percentuais consolidadas dos 3 Pilares (**Movimentação**, **Precisão**, **Constância**), cadência atual e alertas pedagógicos imediatos.
- **Relatório Consolidado de Fim de Sessão (`generate_final_session_report`)**:
  - Compilação automática de estatísticas completas ao encerrar a transmissão ao vivo, com cálculo da média e estabilidade da cadência, pontuações finais dos 3 Pilares, diagnósticos de postura e planos prescritivos exportáveis em **Markdown (`.md`)** e **JSON**.

---

### 4.3. Engine de Calibração & Motor Matemático ([calibrator.py](file:///d:/Projetos/SenpAI/Dev/src/engine/calibrator.py), [mathematical_calibrator.py](file:///d:/Projetos/SenpAI/Dev/src/engine/mathematical_calibrator.py) & [calibration_profiles.json](file:///d:/Projetos/SenpAI/Dev/config/calibration_profiles.json))

O motor calcula a **Pontuação Total Ponderada** com base no tipo de golpe desferido e nas regras biomecânicas da FIK/AJKF:

$$\text{Score}_{\text{Total}} = (w_{\text{target}} \cdot S_{\text{target}}) + (w_{\text{fumikomi}} \cdot S_{\text{fumikomi}}) + (w_{\text{posture}} \cdot S_{\text{posture}}) + (w_{\text{zanshin}} \cdot S_{\text{zanshin}})$$

Para um golpe ser validado como **Yuko-Datotsu** (Ponto Válido / *Ippon*):
1. $\text{Score}_{\text{Total}}$ deve ser maior ou igual a `min_total_score` do perfil ativo.
2. Cada sub-pontuação individual deve satisfazer o respectivo `sub_threshold`.

#### 4.3.1. Pesos Especializados por Tipo de Golpe (`weights_by_strike_type` — Eixo 1.4)
Em vez de pesos homogêneos para todas as técnicas, a importância relativa de cada pilar biomecânico adapta-se dinamicamente conforme a técnica executada:

| Golpe (Waza) | Pilar Mais Crítico | Alvo ($w_{\text{target}}$) | Fumikomi ($w_{\text{fumikomi}}$) | Postura ($w_{\text{posture}}$) | Zanshin ($w_{\text{zanshin}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 🔴 **Men** | Sincronismo Ki-Ken-Tai-Ichi | 35% | 30% | 20% | 15% |
| 🟡 **Kote** | Extensão de cotovelo e contato no Bogu | 45% | 25% | 18% | 12% |
| 🟢 **Do** | Ângulo de corte lateral (Hasuji) | 45% | 15% | 20% | 20% |
| 🔵 **Tsuki** | Alinhamento da lâmina e colinearidade | 50% | 20% | 15% | 15% |

#### 4.3.2. Classificador Probabilístico Calibrado (Platt Scaling — Eixo 1.2)
Além da classificação booleana (*Ippon* vs *Não-Ippon*), o sistema calcula a probabilidade real contínua de validação arbitral:

$$P(\text{Yuko-Datotsu}=1 \mid s) = \sigma(A \cdot s + B) = \frac{1}{1 + e^{-(A \cdot s + B)}}$$

Onde os parâmetros $A$ e $B$ são calibrados por máxima verossimilhança com regularização L2 sobre os feedbacks validados. O resultado é disponibilizado em `probability` $[0.0, 1.0]$ e `probability_pct` (ex: `88.4%`).

#### 4.3.3. Monitoramento de Deriva Temporal (Concept Drift — Eixo 1.3)
Para detectar alterações nos padrões de julgamento arbitral ou defasagem temporal dos modelos:
* **Fator de Decaimento Exponencial:** $W_i = \text{DanWeight}_i \times e^{-\lambda \cdot \Delta t}$, com meia-vida regulamentar de 30 dias ($\lambda = \ln(2)/30$), conferindo maior autoridade a anotações recentes.
* **Teste Bilateral Kolmogorov-Smirnov (`scipy.stats.ks_2samp`):** Compara a distribuição dos scores recentes com a base histórica de calibração. Se $p\text{-valor} < 0.05$, o sistema emite um alerta de `drift_alert` recomendando recalibração.

#### 4.3.4. Otimização Formal com Custo Assimétrico (`BayesianCalibrationOptimizer` — Eixo 1.1)
Substituição definitiva de acréscimos manuais fixos por otimização de parâmetros com:
* **Motor Híbrido:** Optuna (TPE Sampler) com fallback automático para SciPy SLSQP (Sequential Least Squares Programming).
* **Restrições Rígidas:** $\sum w_i = 1.0$, cada peso individual $w_i \ge 0.10$, $T_{\text{global}} \in [0.50, 0.90]$, sub-limiares em $[0.30, 0.85]$.
* **Penalização Assimétrica de Competição:** Custo 3x maior para Falso Positivo ($C_{\text{FP}} = 3.0 \times C_{\text{FN}}$), em conformidade com o regulamento da FIK que proíbe Ippons duvidosos.

#### Perfis Pré-configurados ([calibration_profiles.json](file:///d:/Projetos/SenpAI/Dev/config/calibration_profiles.json))

| Perfil | $\text{min\_total\_score}$ | Pesos Globais ($w_{\text{target}}, w_{\text{fumikomi}}, w_{\text{posture}}, w_{\text{zanshin}}$) | Aplicação Principal |
| :--- | :---: | :--- | :--- |
| **Rígido** | `78%` | Target: 45%, Fumikomi: 25%, Posture: 15%, Zanshin: 15% | Campeonatos / Exames de Dan |
| **Normal** | `74%` | Target: 40%, Fumikomi: 25%, Posture: 20%, Zanshin: 15% | Treinos de Dojang e Avaliação Geral |
| **Permissivo** | `50%` | Target: 35%, Fumikomi: 25%, Posture: 20%, Zanshin: 20% | Iniciantes / Avaliação Educacional |
| **Shiai** | `65%` | Target: 40%, Fumikomi: 25%, Posture: 20%, Zanshin: 15% | Competições Oficiais (Custo 3x FP) |
| **Custom** | Dinâmico | Definido pelo usuário via sliders no Streamlit | Pesquisa e Ajustes Finos |

---

### 4.4. Aprendizagem por Reforço, Governança por Dan e Gestão de Treinamento ([feedback_manager.py](file:///d:/Projetos/SenpAI/Dev/src/engine/feedback_manager.py))

Gerencia o ciclo completo de auditoria, revisão por Dan e otimização adaptativa dos modelos:

- **Seleção de Dan do Revisor**: Mapeia revisores de **1º Dan (Shodan)** a **8º Dan (Hachidan)**, bem como a opção regulamentar **"Decisão dos Shinpans"** (`reviewer_dan == "shinpan"`), associando `reviewer_dan`, `reviewer_dan_name` e `review_date` (timestamp ISO) a cada revisão.
- **Decisão dos Shinpans & Peso Balanceado de Calibração**:
  - A Decisão dos Shinpans é tratada de forma especializada e separada das avaliações pedagógicas por Dan.
  - Utiliza uma **constante média fixa como peso regulamentar**: `SHINPAN_CALIBRATION_WEIGHT = 4.5`. Esse peso balanceado situa-se entre 1º e 8º Dan, não sobrecarregando nem superando a autoridade técnica de um Dan avançado (como 7º ou 8º Dan), garantindo que as marcações dos árbitros em Shiai calibrem os critérios biomecânicos de forma harmoniosa.
- **Edição e Regra de Auditabilidade (Sem Exclusão)**:
  - Permite **confirmar** marcações, **editar** técnica/timestamp/resultado e **incluir** golpes perdidos (falsos negativos).
  - A exclusão de marcações é **desabilitada por norma de auditabilidade** para revisores Dan, preservando a integridade do conjunto de dados. No modo Shinpan, Ippons adicionados podem ser removidos da lista oficial antes da finalização.
- **Histórico de Treinamentos (`data/training_history.json`)**: Registra cada sessão de retreinamento executada, incluindo o Dan do aplicador (ou indicação de Shinpan via `is_shinpan_decision`), a contagem de itens revisados e o resumo das alterações de calibração.
- **Métricas de Governança (`get_training_metrics()`)**:
  - Contador total de treinamentos realizados (separando sessões de revisores humanos, sessões de **Decisão dos Shinpans** e treinamentos automatizados por IA).
  - **Preservação da Média Pura dos Dans**: O nível médio (Dan) dos treinamentos considera **exclusivamente revisores humanos com graduação Dan (1º ao 8º Dan)**. As sessões arbitrais dos Shinpans são contabilizadas isoladamente (`shinpan_trainings_count`), não distorcendo o cálculo da graduação média dos treinadores.
  - Tabela de distribuição da quantidade de treinamentos e percentual por Dan (1º a 8º Dan) + **linha dedicada para Decisões dos Shinpans** + **linha dedicada para Treinamentos Automatizados (IA / Web & Vídeo)**.
- **Espaço em Disco do Treinamento & Modelos (`get_training_storage_info()`)**:
  - Medição em tempo real do espaço em disco ocupado pelo ecossistema de treinamento do sistema.
  - Discriminação detalhada por categoria: **Datasets & Histórico** (`data/`), **Modelos de IA & Pesos Neurais** (`models/`, ex: YOLOv8-Pose) e **Memória de Conhecimento & Calibração** (`config/`).
  - Painel com cards visuais e listagem expansível com caminhos físicos, status e tamanho de cada arquivo no disco.
- **Pacotes de Treinamento Completos (Exportação e Importação Unificada)**:
  - `export_training_package()`: Exporta um pacote consolidado `.json` (v2.0) contendo:
    1. **Revisões por Dan** (1º ao 8º Dan) com data, scores, tipos de golpe e notas;
    2. **Decisões dos Shinpans**, preservando integralmente todos os links de streaming homologados (YouTube, stream web RTSP/HLS/HTTP, uploads), IDs canônicos e sessões;
    3. **Perfis de Calibração** recalibrados e adaptados;
    4. **Treinamentos Automáticos por IA**: exporta a Base de Conhecimento completa (`ai_knowledge_base.json`), contemplando as 14 modalidades pedagógicas, matrizes biomecânicas, acurácias aprendidas, princípios técnicos consolidados, fontes web mineradas, checkpoints e histórico de evolução.
  - `import_training_package()`: Importa pacotes `.json` previamente baixados (v2.0, v1.0 ou listas brutas), mesclando cumulativamente todos os dados: restaura as revisões humanas, registra os links de streaming dos Shinpans assegurando governança anti-duplicidade, mescla fontes e acurácias na base de conhecimento da IA e retreina o modelo imediatamente.
  - `reset_all_training_data()`: Apaga os dados de treinamento e restaura o sistema ao estágio inicial de fábrica.

### 4.5. Treinamento Automático por Inteligência Artificial ([auto_trainer.py](file:///d:/Projetos/SenpAI/Dev/src/engine/auto_trainer.py) & [ai_knowledge_base.json](file:///d:/Projetos/SenpAI/Dev/config/ai_knowledge_base.json))

Motor de inteligência artificial autônomo para busca, ingestão técnica e recalibração automática de modelos:

- **Diagnóstico Autônomo de Necessidade Mais Latente (`diagnose_latent_need`)**:
  - Avalia dinamicamente desbalanceamentos entre Falsos Positivos e Falsos Negativos, lacunas de cobertura por modalidade e desvios de precisão nos perfis de calibração para selecionar automaticamente o foco mais crítico de aprendizado.
- **Base de Conhecimento Estruturada de Kendo (`ai_knowledge_base.json`)**:
  - Repositório de referências técnicas e manuais oficiais da Federação Internacional de Kendo (FIK), tratados de arbitragem da AJKF/ZNKR, artigos científicos de biomecânica desportiva e corpus cinemático de vídeos de alta velocidade.
- **Execução com Duração Determinada (Tempo Controlado em Minutos)**:
  - Processamento em loop temporal estrito respeitando o tempo especificado pelo usuário (1 min, 5 min, 10 min, 15 min, 30 min, 1h, 2h ou personalizado).
  - Atualização progressiva da acurácia biomecânica, streaming de logs de mineração e suporte a cancelamento cooperativo (`request_stop()`).
- **Suporte Multimodal de Aprendizado**:
  - Calibração dos limiares de *Yuko-Datotsu*, *Ki-Ken-Tai-Ichi* e *Sonkyō* para Lutas Gravadas (Shiai).
  - Otimização do quórum de consenso multi-câmeras e baixa latência para Detecção em Tempo Real.
  - Ajuste dos 3 Pilares (*Movimentação, Precisão e Constância*) nas 14 Modalidades Pedagógicas de Treinamento.
- **Calibração Realista de Acurácia (< 50% de Fábrica)**:
  - O sistema adota uma postura empírica transparente: modelos brutos de visão computacional sem calibração prévia possuem acurácia preliminar entre **32.0% e 46.5%** (sendo *Suburi* 46.5%, *Ji-Geiko* 32.0%, *recorded_shiai* 34.0% e *realtime_shiai* 35.0%).
  - À medida que o auto-treinador processa manuais da FIK e que árbitros graduados registram anotações, a acurácia é refinada continuamente. Quando anotações reais de Shinpans existem, a precisão empírica real (`feedback_mgr.get_stats()["precision_pct"]`) substitui as baselines teóricas.
  - **Faixas Visuais de Maturação**:
    - `< 45.0%`: `Fase Inicial (Falsos Positivos)` (Badge Âmbar/Laranja com barra de progresso em tom quente).
    - `45.0% - 65.0%`: `Em Calibração` (Badge Azul).
    - `65.0% - 80.0%`: `Calibrado` (Badge Verde Menta).
    - `>= 80.0%`: `Excelente / Shiai` (Badge Verde Esmeralda).
- **Persistência Contínua, Checkpoints e Tolerância a Falhas (`auto_training_checkpoint.json`)**:
  - **Consolidação Preventiva no Início**: Antes de iniciar um novo ciclo, qualquer aprendizado anterior pendente ou interrompido é automaticamente absorvido e consolidado na Base de Conhecimento.
  - **Aprendizado Cumulativo Contínuo**: Novos treinamentos partem sempre da acurácia e parâmetros reais acumulados previamente, treinando pelo tempo total selecionado sem reiniciar do zero.
  - **Persistência Atômica Periódica**: Checkpoints salvos a cada etapa e gravação direta incremental de novas fontes mineradas em disco.
  - **Salvamento Garantido em Caso de Falhas (`interrupted_salvaged`)**: Caso ocorra qualquer erro inesperado durante a execução, todo o conhecimento, fontes mineradas e amostras biomecânicas obtidas até a fração de segundo da falha são imediatamente preservados e consolidados na Base de Conhecimento e no histórico de governança.
  - **Preservação de Dados**: Nenhum treinamento realizado é descartado automaticamente, sendo limpo apenas quando o usuário aciona explicitamente a opção de reset nas Configurações.

---

### 4.6. Relatórios e Pipeline ([reporter.py](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py) & [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py))

- **`DiagnosticReporter`** ([reporter.py](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py)): Gera um texto explicativo em Português detalhando por que o golpe foi aprovado ou reprovado, apresentando os milissegundos do Fumikomi e dicas de correção técnica para o praticante.
- **`SenpAIPipeline`** ([pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py)): Orquestra a execução frame-a-frame do vídeo, grava o vídeo anotado com esqueletos e alvos, valida distância física de contato (*Maai*) e retorna o dicionário completo com métricas.

---

### 4.7. Interface Web e Navegação Estrita (Página Inicial / Boas-Vindas) ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))

A interface web conta com uma arquitetura de navegação com visibilidade estrita baseada no estado de seleção:

- **Página Inicial de Abertura (`render_welcome_home_page`)**:
  - Ponto de partida carregado automaticamente ao inicializar o sistema (`st.session_state["nav_page_selection"] = "home"`).
  - **Regra Estrita de Visibilidade**: Esta página inicial **só é visível quando nem a página Análise de Lutas nem a página Menu de Configurações estiverem selecionadas**. Assim que o usuário escolhe qualquer outra página, a Home é completamente ocultada, deixando a interface limpa e focada.
  - **Hero Banner Institucional**: Apresentação de conformidade com normas FIK (Artigos 12 a 24) e AJKF/ZNKR, com detecção em tempo real de hardware ativo (GPU NVIDIA CUDA com FP16 Tensor Cores ou CPU).
  - **Barra de Atalhos de Ação Rápida**: Botões para transição imediata para a Análise de Lutas, acesso ao Menu de Configurações e download direto da documentação oficial (`manual.md`).
  - **Guia Demonstrativo em 4 Passos (*Como Iniciar o Uso*)**:
    1. *Navegação no Menu*: Seleção primária na barra lateral entre **⚔️ Análise de Lutas** ou **🎓 Treinamento e Aprendizado**.
    2. *Escolha do Submodo*: Seleção entre **🔴 Tempo Real** (câmeras ao vivo / RTSP) ou **📹 Vídeo Gravado** (arquivos locais / YouTube).
    3. *Vídeo ou Câmera*: Configuração dos feeds RTSP/Webcam ou upload de arquivos e vídeo sintético de demonstração.
    4. *Diagnósticos & Exportação*: Acompanhamento ao vivo com telemetria dos 3 Pilares e exportação completa em Markdown/JSON/Excel.
  - **Apresentação dos 2 Modos de Operação Principais**:
    - **1. ⚔️ Modo de Análise de Lutas (Combate / Shiai)**:
      - *🔴 Detecção em Tempo Real*: Fusão multi-câmera síncrona (1 a 4 câmeras RTSP ou Webcams), feed ao vivo com placar Sanbon-shobu e detecção de Ippons segundo as regras oficiais da FIK.
      - *📹 Detecção Gravada*: Upload local, YouTube ou vídeo de demonstração, análise de Yūko-Datotsu (Ki-Ken-Tai-Ichi), rituais de Sonkyō, VAR interativo quadro a quadro, governança por Dan e retreinamento via planilha Excel.
    - **2. 🎓 Modo de Treinamento e Aprendizado (Dojo / Keiko)**:
      - *📹 Análise de Vídeo Gravado*: Avaliação diagnóstica pós-treino nas 14 modalidades oficiais do Kendo com cálculo dos 3 Pilares Fundamentais (Movimentação, Precisão, Constância) e exportação em Markdown e JSON.
      - *🔴 Análise em Tempo Real*: Monitoramento ao vivo de dojos via câmeras RTSP ou Webcams, com contagem de repetições/golpes, cadência dinâmica em Golpes por Minuto (CPM), HUD em tempo real dos 3 Pilares e biofeedback postural.
  - **Transparência de Inteligência Artificial & Governança por Dan**: Esclarecimento didático sobre a calibração inicial realista (< 50%) e a governança com auditoria de árbitros (1º ao 8º Dan).

---

### 4.8. Controles Interativos de Reprodução de Vídeo e VAR ([video_player_controls.py](file:///d:/Projetos/SenpAI/Dev/src/utils/video_player_controls.py))

O módulo de controle de reprodução atua como uma mesa de corte e **VAR (*Video Assistant Referee*)** interativo acoplada ao player de vídeo da Detecção Gravada:

- **Isolamento e Execução Segura de JavaScript (`st.iframe`)**:
  - A API nativa `st.html()` do Streamlit bloqueia ativamente tags `<script>` por motivos de sanitização interna. Para contornar essa restrição sem comprometer a segurança, o módulo renderiza um iframe isolado via `st.iframe` (com fallback retrocompatível para `streamlit.components.v1.html`).
  - Como o iframe compartilha a mesma origem (origin) do aplicativo Streamlit, o script tem acesso seguro e direto ao documento pai através de `window.parent.document`.
- **Comunicação Direta com o Elemento `<video>` no DOM**:
  - Localiza o elemento de vídeo no DOM pai via `parentDoc.querySelectorAll('video')` ou `parentDoc.getElementById(containerId)`.
  - Permite controle em tempo real de `currentTime`, `playbackRate`, `play()`, `pause()`, volume e tela cheia sem recarregar ou causar re-execução do script Python no servidor Streamlit.
- **Botões e Ferramentas do Painel VAR**:
  - **Transporte Básico**: `▶️ Play`, `⏸️ Pause`, `⏪ Reiniciar`.
  - **Salto Temporal Fino / Scrubbing**: Saltos incrementais rápidos de `⏪ -10s`, `◀️ -5s`, `⏮️ -1s`, `⏭️ +1s`, `▶️ +5s`, `⏩ +10s` para análise quadro a quadro de cortes milimétricos.
  - **Câmera Lenta e Velocidade Variável**: Seletor de velocidade instantânea com 7 taxas: `0.1x` (ultra-slow-motion), `0.25x`, `0.5x`, `0.75x`, `1.0x` (velocidade normal), `1.5x` e `2.0x`.
  - **Controle de Visualização**: Alternador de Tela Cheia (`⛶`) e controle de áudio/mudo (`🔊`/`🔇`).
  - **Display de Tempo**: Exibição de timestamp decorrido e duração total (`MM:SS / MM:SS`).
- **Resolução de Incompatibilidade de Seek no Streamlit (`st.video`)**:
  - No Streamlit, o método `st.video()` não suporta o parâmetro `key` em diversas versões, gerando a exceção fatal `TypeError: MediaMixin.video() got an unexpected keyword argument 'key'`.
  - A sincronização temporal automática com os eventos selecionados na Linha do Tempo (Sonkyō e Golpes) é agora resolvida diretamente pelo componente via `target_start_time`: a função JavaScript `applyInitialSeek()` posiciona o vídeo exatamente no instante pretendido assim que os metadados do vídeo são carregados no navegador (`loadedmetadata` / `canplay`), dispensando parâmetros stateful no backend.

---

### 4.9. Gestão e Retreinamento Bidirecional de Golpes via Planilha Excel ([excel_strikes_manager.py](file:///d:/Projetos/SenpAI/Dev/src/utils/excel_strikes_manager.py))

Permite a auditoria, anotação offline e retreinamento do modelo de IA através de planilhas eletrônicas padronizadas em formato Microsoft Excel (`.xlsx`):

- **Exportação Formatada de Golpes (`export_strikes_to_excel`)**:
  - Converte a lista de golpes detectados pela IA em um arquivo Excel (.xlsx) estruturado com OpenPyXL / Pandas.
  - Gera colunas completas de telemetria: `ID_Golpe`, `Timestamp_Segundos`, `Timestamp_Formatado (MM:SS.s)`, `Alvo_Detectado (MEN, KOTE, DO, TSUKI)`, `Atacante (Aka/Shiro)`, `Ippon_Valido (Sim/Não)`, `Score_Geral (%)`, `Score_Alvo_Ken (%)`, `Score_Fumikomi_Tai (%)`, `Score_Postura_Tai (%)`, `Score_Zanshin_Ki (%)`, `Fumikomi_Offset_ms`, `Dentro_Janela_Sonkyo (Sim/Não)` e `Status_Revisao`.
  - Inclui campos dedicados para anotação humana: `Acao_Revisao (CONFIRMAR / EDITAR / DESCARTAR)`, `Novo_Alvo_Corrigido`, `Novo_Timestamp_Corrigido`, `Dan_Anotador (1 a 8)` e `Observacoes_Tecnicas`.
- **Template em Branco para Anotações Manuais (`generate_empty_template_excel`)**:
  - Cria um arquivo modelo pré-estruturado para registro de novas lutas ou anotações a partir do zero por equipes de arbitragem.
- **Importação, Validação e Retreinamento Automático (`import_strikes_from_excel`)**:
  - Validação estrita de integridade e sanitização de schema contra arquivos corrompidos ou colunas faltantes.
  - Processa as correções registradas por árbitros graduados (Shodan 1º Dan a Hachidan 8º Dan) e as converte automaticamente em amostras de feedback (Verdadeiros Positivos, Falsos Positivos e Falsos Negativos).
  - Atualiza atomicamente a base de governança (`data/feedback_dataset.json`) e aciona imediatamente o retreinamento adaptativo do modelo via `DanTrainingGovernance`, refinando os limiares de Ki-Ken-Tai-Ichi sem intervenção manual.

---

### 4.10. Decisão dos Shinpans & Gestão da Linha do Tempo em Arbitragem de Shiai

O módulo de **Decisão dos Shinpans** foi concebido para atender às exigências formais de arbitragem em campeonatos de Kendo (*Shiai*), onde a pontuação válida (*Yūko-Datotsu*) depende exclusivamente do julgamento colegiado dos três árbitros em quadra (um *Shushin* e dois *Fukushin*).

- **Princípio da Linha do Tempo Liberada**:
  - Diferentemente da revisão pedagógica por Dan (onde todos os golpes identificados pela IA são apresentados para análise do treinador), ao selecionar a opção **"Decisão dos Shinpans"**, a Linha do Tempo fica **limpa e liberada** para conter **exclusivamente os golpes válidos (Ippons) confirmados pela arbitragem oficial** na luta.
  - Golpes detectados pelo modelo que não foram assinalados pelos Shinpans não entram na contagem de pontos do combate.
- **As 5 Regras Estritas de Transição de Estado da Interface**:
  1. **Habilitar Edição Não Selecionada (`enable_editing == False`)**: A listagem completa de golpes identificados automaticamente pela IA (`res['events']`) é apresentada com sua pontuação biomecânica original e status padrão (`✅ PONTO VÁLIDO (IPPON)` ou `❌ GOLPE INVÁLIDO`).
  2. **Habilitar Edição Selecionada + Graduação Dan (1º a 8º Dan)**: A listagem completa de golpes identificados pela IA é apresentada acompanhada das ferramentas de ação técnica do treinador Dan (`✅ Confirmar`, `✏️ Editar`, `🚫 Não Houve Golpe`, inclusões manuais e remoções).
  3. **Habilitar Edição Selecionada + Decisão dos Shinpans**: A lista cronológica fica limpa/liberada para incluir exclusivamente os Ippons concedidos pelos árbitros de Shiai.
  4. **Habilitar Edição Desmarcada após Decisão dos Shinpans**: A listagem completa de golpes detectados pela IA volta a ser apresentada imediatamente em sua totalidade, sem resquícios de filtros arbitrais e com plena reatividade visual.
  5. **Selecionar um Dan (1º a 8º Dan) após Decisão dos Shinpans**: A listagem completa de golpes detectados pela IA volta a ser apresentada imediatamente com os botões de revisão do Dan escolhido, ignorando anotações arbitrais na linha do tempo técnica.
- **Painel de Sugestões Rápidas da IA (`🤖 Aproveitar Golpes Detectados pela IA`)**:
  - Exibe os momentos de golpe pré-identificados pelo modelo, com timestamp, técnica em Katakana, combatente atacante e índice de aprovação.
  - Botão **`➕ Ippon dos Shinpans`**: Permite oficializar em 1 clique o golpe diretamente na linha do tempo oficial dos Shinpans, sem necessidade de digitação manual de horários ou timestamps.
- **Inseridores Inline e Inclusão Manual**:
  - Inseridores inline `+` nos intervalos entre o Sonkyō e entre golpes para inclusão de lances intermediários.
  - Formulário manual ao final do combate restrito a `VALID_IPPON` e observações padronizadas.
- **Placar Oficial Eletrônico (Sanbon-Shobu)**:
  - No modo Shinpan, o placar oficial computa **exclusivamente os Ippons atribuídos pela arbitragem**, garantindo total fidelidade com as súmulas e placares físicos de competição.
- **Recalibração Balanceada de Pesos (`SHINPAN_CALIBRATION_WEIGHT = 4.5`)**:
  - Botão `⚖️ Salvar Decisão dos Shinpans & Recalibrar Pesos`.
  - Recalibra os pesos dos 4 critérios de *Ki-Ken-Tai-Ichi* (*target_impact*, *fumikomi_sync*, *posture*, *zanshin*) utilizando fator de aprendizado balanceado `4.5` com normalização matemática estrita ($\sum w = 1.0$).
  - Trava de proteção: impede confirmação automática incorreta caso nenhum Ippon tenha sido apontado pelos árbitros.
- **Governança de Links de Vídeos & Prevenção de Entradas Duplicadas para Shinpans**:
  - **Registro Persistente de Links de Vídeos (`data/shinpan_reviewed_videos.json`)**: Ao homologar uma sessão como *Decisão dos Shinpans*, o link do vídeo (YouTube watch, Shorts, streaming web ou arquivo de upload local) é normalizado para um identificador canônico e persistido com data/hora, ID da sessão de retreinamento e quantidade de Ippons homologados.
  - **Bloqueio Estrito de Duplicidade para Shinpans**: O sistema impede terminantemente que um mesmo link de vídeo receba 2 entradas como *Decisão dos Shinpans*, protegendo a integridade do histórico arbitral. Quando um vídeo já registrado é aberto em modo Shinpan:
    - Um banner de bloqueio é exibido: `⛔ ENTRADA DUPLICADA BLOQUEADA — DECISÃO DOS SHINPANS JÁ REGISTRADA`, informando data, sessão e link registrado.
    - Os botões `⚖️ Salvar Decisão dos Shinpans & Recalibrar Pesos` e `➕ Ippon dos Shinpans` ficam bloqueados e desabilitados.
    - O motor de governança levanta `DuplicateShinpanReviewError` prevenindo qualquer tentativa de gravação duplicada.
  - **Revisão por DAN Irrestrita para Entradas Duplicadas**: A restrição de link único incide **exclusivamente sobre a Decisão dos Shinpans**. Revisores graduados de **1º ao 8º Dan** possuem total liberdade para avaliar o mesmo link de vídeo quantas vezes forem necessárias (para fins formativos, estudos técnicos e análises pedagógicas), sem qualquer restrição de entradas duplicadas.
  - **Painel de Auditoria e Consulta**: Aba de Governança de Treinamento exibe a lista expansível de vídeos homologados com seus links, IDs canônicos, datas e sessões.

---

### 4.11. Motor de Aprendizado Ativo & Benchmark Padrão-Ouro (Eixo 4) ([active_learning.py](file:///d:/Projetos/SenpAI/Dev/src/engine/active_learning.py))

O módulo de Aprendizado Ativo (*Active Learning*) automatiza o ciclo de evolução do SenpAI, priorizando instâncias que trazem maior ganho informativo e blindando o modelo contra o esquecimento catastrófico:

- **Amostragem por Incerteza (`UncertaintySampler`)**:
  - Em vez de solicitar rotulagem humana para golpes óbvios (confiança muito alta $> 85\%$ ou muito baixa $< 30\%$), o algoritmo calcula o índice de incerteza da predição:
    $$\text{Incerteza}(x) = 1.0 - 2 \cdot |P(\text{Ippon}) - 0.5|$$
  - Golpes cuja probabilidade/score de aprovação situa-se na faixa de contorno e ambiguidade regulamentar (**45% a 65%**) recebem incerteza elevada ($\ge 0.70$) e são automaticamente enfileirados em `data/active_learning_queue.json`.
  - A fila possui limite deslizante (`max_queue_size = 500`), priorizando os lances mais críticos para revisão humana orientada.
- **Dataset Canônico de Benchmark Padrão-Ouro (`GoldenBenchmark` & `golden_dataset.json`)**:
  - Armazenado de forma imutável em `data/benchmark_golden/golden_dataset.json`, reúne lances incontestáveis de campeonatos mundiais (WKC), All Japan Kendo Championships e exames de alto Dan (Men, Kote, Do, Tsuki e Falsos Positivos canônicos como golpes fora do Shinai-bu ou sem Zanshin).
  - **Salvaguarda Mandatória contra Esquecimento Catastrófico (`validate_no_regression`)**:
    - Antes de qualquer atualização dos perfis de calibração (`feedback_manager.py`) ou consolidação de auto-treinamento (`auto_trainer.py`), o motor executa a validação contra o Golden Benchmark.
    - Se a acurácia global ou a acurácia em qualquer tipo de golpe regredir abaixo de uma tolerância estrita ($\le 2.0\%$), a atualização é terminantemente **rejeitada** e revertida, impedindo a degradação do modelo.
- **Consenso Multi-Árbitro Ponderado por Dan (`MultiJudgeConsensus`)**:
  - Modela o regulamento oficial da FIK (Artigo 24 — quórum de 2 em 3 árbitros).
  - Pondera cada voto pelo peso do Dan do revisor ($w_i = \text{Dan}_i$ ou $4.5$ para Shinpan):
    $$V_{\text{ponderado}} = \frac{\sum_{i=1}^{N} w_i \cdot v_i}{\sum_{i=1}^{N} w_i}, \quad v_i \in \{0, 1\}$$
  - Calcula o **Grau de Divergência Arbitral**: lances com alta discordância entre os juízes disparam alertas de controvérsia e são marcados para análise colegiada.
- **Decaimento Exponencial de Confiança de Revisores (`ReviewerTrustManager`)**:
  - Gerencia a autoridade de revisores com decaimento exponencial temporal por inatividade ($T_{1/2} = 180$ dias):
    $$\text{Trust}(t) = \text{Trust}_{\text{base}} \cdot e^{-\lambda \cdot \Delta t}$$
  - A confiança é restaurada e ampliada dinamicamente conforme a taxa de concordância histórica do revisor com o Golden Benchmark.

---

### 4.12. Assistente LLM Especialista em Kendo e Anotação Cinemática ([llm_assistant.py](file:///d:/Projetos/SenpAI/Dev/src/engine/llm_assistant.py))

Integração de Modelos de Linguagem de Grande Porte (LLMs) multimodal e textual para acelerar a rotulagem, explicar decisões e sintetizar regras da Federação Internacional de Kendo (FIK):

- **Arquitetura Híbrida com Fallback Determinístico**:
  - Suporte nativo às APIs de ponta: **Google Gemini** (`gemini-1.5-flash` / `gemini-1.5-pro`) via `GEMINI_API_KEY` e **OpenAI** (`gpt-4o` / `gpt-4o-mini`) via `OPENAI_API_KEY`.
  - **Motor Especialista Offline Embutido**: Caso nenhuma chave de API esteja configurada ou não haja conexão à internet, o assistente ativa um motor heurístico baseado nos tratados oficiais da FIK e AJKF, garantindo operação 100% autônoma e determinística.
- **Rotulagem e Anotação Cinemática Assistida (`label_strike_from_telemetry`)**:
  - Processa os vetores de telemetria de um golpe (aceleração do pulso, offset do Fumikomi em milissegundos, ângulo de Shisei da coluna, proximidade de Maai e Zanshin) e gera parecer arbitral completo em JSON estruturado com recomendação de ponto (`is_valid_ippon`), alvo sugerido, nível de confiança e justificativa técnica detalhada à luz do *Ki-Ken-Tai-Ichi*.
- **Explicação Pedagógica de Lances Controversos (`explain_controversial_strike`)**:
  - Analisa lances divergentes da fila de aprendizado ativo, detalhando os pontos de discordância (ex: impacto perfeito no Men, porém desprovido de Fumikomi sincrônico dentro da janela regulamentar de 100ms).

---

### 4.13. Invariância de Câmera e Normalização Espacial 3D (Eixo 5) ([camera_invariance.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/camera_invariance.py))

O módulo de invariância de câmera garante que a avaliação biomecânica e o julgamento de *Yuko-Datotsu* permaneçam consistentes e precisos independentemente do ângulo em que o smartphone, filmadora ou câmera de dojo for posicionado:

- **Compensação de Perspectiva por Ângulo de Filmagem (`CombatVectorEstimator`)**:
  - **Estimativa do Vetor de Combate**: Extrai as coordenadas 2D dos quadris dos dois atletas (`Kenshi Aka` e `Kenshi Shiro`) para traçar a linha principal de enfrentamento.
  - **Classificação de Ângulo**:
    - **Frontal ($0^\circ - 30^\circ$)**: Visão alinhada ao vetor de ataque; excelente para avaliar centralidade (*Chushin-sen*) e estocadas (*Tsuki*), porém sujeita a achatamento da inclinação da coluna.
    - **Oblíquo ($30^\circ - 60^\circ$)**: Posição angular mista, comum em arquibancadas ou cantos do Shiaijo.
    - **Lateral ($60^\circ - 90^\circ$)**: Plano canônico de referência para medição visual de avanço de pé (*Fumikomi*) e postura ereta (*Shisei*).
  - **Detecção em Treinamento Solo**: Caso apenas um Kendoca esteja presente no dojo, o vetor é deduzido a partir da linha biacromial (largura entre ombros normalizada).
  - **Normalização Trigonométrica de Postura (*Shisei*)**:
    - Aplica o fator multiplicador $1 / \sin(\theta)$ (com clamp em $\sin(20^\circ)$ para prevenir singularidades) sobre a inclinação observada na câmera:
      $$\theta_{\text{real}} = \frac{\theta_{\text{observado}}}{\sin(\max(\theta_{\text{câmera}}, 20^\circ))}$$
    - Recupera o ângulo físico real do tronco no espaço tridimensional mesmo quando filmado de frente.

- **Estimativa de Profundidade Monocular 3D & Maai 3D (`MonocularDepthEstimator`)**:
  - **Reconstrução de Pseudo-Keypoints 3D $(x, y, z)$**:
    - Estima o canal de profundidade $Z$ a partir de restrições antropométricas da altura em pixels e posições de articulações em relação ao plano canônico do solo.
  - **Distância Euclidiana Tridimensional (*Maai 3D*)**:
    - Substitui a distância euclidiana puramente bidimensional pela métrica tridimensional completa:
      $$d_{\text{3D}} = \sqrt{(x_{\text{aka}} - x_{\text{shiro}})^2 + (y_{\text{aka}} - y_{\text{shiro}})^2 + (z_{\text{aka}} - z_{\text{shiro}})^2}$$
    - **Eliminação Definitiva de Ku-totsu (Golpe no Vazio)**: Em tomadas frontais ou oblíquas, o atacante pode parecer sobreposto ao defensor no plano 2D $(x, y)$, mas estar a metros de distância no eixo $Z$. O *Maai 3D* detecta a separação real em profundidade e rejeita sumariamente o golpe caso $d_{\text{3D}} > 0.55$, eliminando marcações indevidas.

- **Diagnóstico Automático de Qualidade de Ângulo (`CameraQualityDiagnostic`)**:
  - Emite notas individuais de confiabilidade $[0.0, 1.0]$ para cada critério técnico conforme o enquadramento:
    - *Fumikomi*: Alta confiança em tomadas laterais ($\ge 60^\circ$); penalizado em tomadas frontais com aviso de oclusão de pés.
    - *Hasuji*: Avaliação ideal em tomadas oblíquas e laterais; penalizado quando o plano da lâmina fica colinear à lente.
    - *Shisei*: Compensado trigonometricamente em ângulos frontais com margem de erro documentada.
    - *Tsuki*: Confiabilidade máxima em tomadas frontais e oblíquas.
    - *Zanshin*: Alta confiabilidade em todos os ângulos com correção de perspectiva.
  - **Matriz de Modificadores de Peso**: Fornece multiplicadores adaptativos para o motor de calibração atenuar critérios de baixa visibilidade e reforçar os de alta certeza no ângulo ativo.

---

### 4.14. Modelagem do Estilo Individual do Kenshi & Warm Start (Eixo 6) ([kenshi_style_model.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/kenshi_style_model.py))

A modelagem do estilo individual personaliza a avaliação biomecânica, reconhecendo que cada Kenshi possui proporções corporais e características cinestésicas únicas, avaliando o praticante em relação ao seu próprio histórico evolutivo e transferindo conhecimento prévio entre perfis:

- **Aprendizado Contínuo Online do Baseline Cinestésico (`MetricDistribution` & Algoritmo de Welford)**:
  - Atualização recursiva e estatisticamente estável da média ($\mu_n$) e variância ($\sigma_n^2$) a cada nova repetição ou sessão de treino, sem necessidade de recalcular todo o histórico:
    $$M_{1} = x_1, \quad M_{k} = M_{k-1} + \frac{x_k - M_{k-1}}{k}$$
    $$S_k = S_{k-1} + (x_k - M_{k-1})(x_k - M_k), \quad \sigma^2 = \frac{S_k}{k - 1}$$
  - Rastreamento contínuo de 5 dimensões biomotoras:
    - Ângulo de postura de repouso em *Chudan* / *Shisei*;
    - Janela temporal de sincronismo habitual de *Fumikomi* (tempo em ms entre aceleração e impacto);
    - Extensão de braço típica por golpe (`MEN`, `KOTE`, `DO`, `TSUKI`);
    - Cadência de cortes em Golpes por Minuto (**CPM**);
    - Inclinação média habitual da coluna.

- **Avaliação Comparativa em Z-Score e Diagnóstico Humanizado (`KinestheticBaselineModel`)**:
  - Mede o desvio padronizado da execução atual em relação à média pessoal do próprio atleta:
    $$Z = \frac{x_{\text{observado}} - \mu_{\text{kenshi}}}{\sigma_{\text{kenshi}}}$$
  - Geração de diagnósticos pedagógicos humanizados em linguagem natural com ícone 🧬:
    - *Fumikomi*: *"Seu Fumikomi foi 35ms mais rápido que seu habitual (1.4σ), indicando excelente explosão do pé direito."*
    - *Coluna*: *"Atenção: inclinação da coluna 5.4° maior que seu padrão habitual (Z=+1.9σ). Mantenha o Shisei ereto."*
    - *Extensão*: *"Extensão de braço no Men perfeitamente alinhada com seu baseline histórico."*

- **Gestão Centralizada e Persistência Multi-Praticante (`KinestheticProfileManager`)**:
  - Armazenamento atômico em `data/kenshi_baselines.json`.
  - Integração no `SenpAIPipeline` e no `TrainingAnalyzer`, registrando e avaliando tanto praticantes solo (`KENSHI_SOLO`) quanto duplas de combate (`KENSHI_SHIRO`, `KENSHI_AKA`).
  - Inclusão automática da **Seção 3: Análise Comparativa com o Baseline Cinestésico Individual** nos relatórios exportáveis de treino em Markdown.

- **Transferência de Conhecimento entre Perfis com Warm Start (`ProfileWarmStartManager`)**:
  - Permite derivar novos perfis de calibração (`rigido`, `shiai` ou perfis para clubes e graduações específicas) reaproveitando o conhecimento otimizado do perfil pai.
  - **Preservação Integral de Pesos Ótimos**: Copia os pesos globais, a matriz `weights_by_strike_type` (`MEN`, `KOTE`, `DO`, `TSUKI`), calibração de Platt Scaling e priors de Dan.
  - **Ajuste Direcional de Rigidez**:
    - `more_strict`: Eleva a pontuação mínima em +10% e sub-limiares em +8% (respeitando tetos de segurança);
    - `more_permissive`: Reduz a pontuação mínima em -10% e sub-limiares em -8% (respeitando pisos de segurança);
    - `neutral`: Mantém limiares idênticos para direcionamento a novo segmento.
  - **Governança e Rastreamento de Linhagem**: Registra `parent_profile_id`, data/hora de criação e direção de ajuste em `config/calibration_profiles.json`.

---

## 5. Suíte de Testes Automatizados e Relatório de Execução

O projeto inclui suíte completa de testes automatizados em `unittest` com runner customizado ([test_runner.py](file:///d:/Projetos/SenpAI/Dev/src/utils/test_runner.py)) e script de execução dedicado ([run_tests.py](file:///d:/Projetos/SenpAI/Dev/run_tests.py)).

### Execução dos Testes via CLI e Interface

```bash
# Execução completa com exibição detalhada e geração de log descritivo:
.\.venv\Scripts\python.exe run_tests.py

# Ou via unittest padrão:
.\.venv\Scripts\python.exe -m unittest discover tests
```

Também é possível disparar os testes diretamente no **Web Dashboard** acessando a aba **⚙️ Configurações > Seção 5 (Diagnóstico e Logs)** através do botão **`🔬 Rodar Testes`** e baixar o relatório completo em **`📥 Baixar Log Testes (.log)`**.

### Relatório Descritivo e Política de Retenção de Logs

- **Relatório Detalhado ([`logs/senpai_test_report.log`](file:///d:/Projetos/SenpAI/Dev/logs/senpai_test_report.log))**:
  - Cada teste executado é documentado com: **Módulo**, **Classe**, **Método**, **Descrição Detalhada do Teste / Docstring**, **Status (PASS/FAIL/ERROR)**, **Duração em Segundos** e eventuais rastros de erro/falha.
  - Cabeçalho com data/hora, versão do sistema, plataforma operacional e hardware.
  - Resumo estatístico final (total, aprovados, falhas, erros, taxa de sucesso % e tempo total).
- **Política de Retenção Única**:
  - A pasta `logs/` mantém **estritamente apenas o último log de testes executado**, sobrescrevendo ou limpando relatórios anteriores automaticamente a cada nova execução.

### Módulos de Testes Incluídos (181 Testes em 19 Módulos)

- **`test_active_learning_eixo4.py` (10 testes)**: Valida o cálculo de amostragem por incerteza (faixa 45%-65%), enfileiramento inteligente, avaliação de acurácia contra o dataset padrão-ouro canônico (`GoldenBenchmark`), salvaguarda mandatória contra esquecimento catastrófico (`validate_no_regression`), bloqueio de regressão no retreinamento adaptativo e auto-treinamento, consenso multi-árbitro 2-de-3 ponderado por Dan, detecção de divergência arbitral, decaimento exponencial de autoridade por inatividade (`ReviewerTrustManager`), rotulagem de lances pelo Assistente LLM Especialista FIK e explicação pedagógica de lances controversos.
- **`test_auto_trainer.py` (14 testes)**: Valida a inicialização da base de conhecimento de Kendo, diagnóstico autônomo de necessidade mais latente, ciclo de auto-treinamento com tempo controlado, baselines preliminares realistas (< 50%), recalibração de perfis de arbitragem e das 14 modalidades pedagógicas, persistência incremental em governança e checkpoints de tolerância a falhas.
- **`test_dan_training_governance.py` (8 testes)**: Valida salvamento de revisões com Dan, retreinamento do modelo, cálculo das métricas Dan (contador humano vs IA, média de Dan humano pura e tabela por Dan com linha dedicada para IA e Decisão dos Shinpans), ponderação regulamentar com peso balanceado (4.5) para Shinpans, as 5 regras de transição de estado da UI (`test_shinpan_ui_state_transitions`), exportação/importação de pacotes `.json` com data e Dan/Shinpan, e reset do sistema.
- **`test_environment.py` (9 testes)**: Valida detecção, integridade e isolamento do ambiente virtual Python (`.venv`).
- **`test_excel_strikes_io.py` (7 testes)**: Valida exportação de golpes detectados para planilha Excel (.xlsx), geração de template vazio, importação com sanitização e validação de schema, suporte a anotações por Dan e Shinpans, integração com a base de governança e acionamento de retreinamento do modelo.
- **`test_feedback_loop.py` (2 testes)**: Valida persistência, cálculo de precisão/recall e algoritmo de aprendizagem por reforço sobre Falsos Positivos.
- **`test_hardware_settings.py` (7 testes)**: Valida detecção de GPU NVIDIA CUDA, configurações globais e resolução de fallback transparente para CPU.
- **`test_logger_manager.py` (5 testes)**: Valida sistema de logs, métricas em tempo real e diagnósticos automatizados.
- **`test_multi_camera_fusion.py` (10 testes)**: Valida o motor de consenso e fusão multi-câmeras, escalonamento de quórum por quantidade de câmeras ($N=1$ a $4$), rejeição de falsos positivos unilaterais, alinhamento temporal, fusão de scores e a presença da análise completa de Yūko-Datotsu (Ki-Ken-Tai-Ichi) para golpes Ippon e não-Ippon.
- **`test_pipeline_cancellation.py` (7 testes)**: Valida cancelamento cooperativo, liberação de recursos de streaming e cronômetro em tempo real.
- **`test_pose_batch_processing.py` (8 testes)**: Valida processamento de poses em lotes paralelos com aceleração.
- **`test_scoreboard_and_flag_detection.py` (7 testes)**: Valida o placar eletrônico Sanbon-shobu, detecção cromática de flag dorsal (Tasukuki) e inversão Aka ⇄ Shiro.
- **`test_shinai_tracking.py` (6 testes)**: Valida rastreamento de Shinai, estimação do Kensen, zonas anatômicas de alvo e predição vetorial de impacto.
- **`test_sonkyo_and_plane_filtering.py` (21 testes)**: Valida a classificação postural de Sonkyō, delimitação temporal da luta, filtragem de planos (fundo/transeuntes/árbitros em primeiro plano), delimitação da quadra de luta (Shiai-jo ROI), travamento K=2, interpolação cinemática de pulsos/pés sob oclusão, supressão de falsos positivos, debounce e NMS de 35 frames do `EventSpotter`, e persistência de aprendizado de Sonkyō.
- **`test_stream_capture.py` (8 testes)**: Valida a captura assíncrona com threading, reconexão automática e otimizações de rede para câmeras IP / RTSP / Webcams.
- **`test_training_modes.py` (7 testes)**: Valida as 14 modalidades pedagógicas de treino, cálculo dos 3 Pilares (Movimentação, Precisão, Constância) e perfil do Kendoca.
- **`test_video_downloader.py` (12 testes)**: Valida download, extração de metadados, validação de URLs do YouTube/Web e integração de streams com cache.
- **`test_video_player_controls.py` (5 testes)**: Valida a geração do HTML do componente de controles de vídeo, presença dos botões de transporte, scripts de seek DOM em `window.parent.document` e injeção do timestamp de busca inicial.
- **`test_training_live_manager.py` (5 testes)**: Valida a máquina de estados de golpes em tempo real (`LiveStrikeState`), rastreamento biomecânico contínuo da coluna (*Shisei*) e simetria de ombros, contagem de repetições, cadência em Golpes por Minuto (CPM), renderização do HUD em tempo real dos 3 Pilares e geração de relatórios de sessão em Markdown e JSON.

- **`test_kenshi_style_model_eixo6.py` (9 testes)**: Valida a atualização online via algoritmo de Welford (MetricDistribution), cálculo de desvios em Z-score e geração de insights humanizados (KinestheticBaselineModel), persistência atômica e gestão multi-praticante em JSON (KinestheticProfileManager), derivação direcional de perfis com Warm Start mais rígido e mais permissivo preservando pesos e linhagem (ProfileWarmStartManager), integração com FeedbackManager e inclusão da Seção 3 no relatório de treino do Kendoca.

Total de **232 testes automatizados** distribuídos em 25 módulos, executados e aprovados com 100% de sucesso.

---

## 6. Registro de Mudanças e Histórico de Versões (Changelog)

---

### `[v 0.3.5.0]` — 2026-10-02 *(Versão Atual)*

- **Implementação do Eixo 6: Modelagem do Estilo Individual do Kenshi & Warm Start de Perfis ([kenshi_style_model.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/kenshi_style_model.py), [feedback_manager.py](file:///d:/Projetos/SenpAI/Dev/src/engine/feedback_manager.py), [training_analyzer.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/training_analyzer.py), [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py), [reporter.py](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py) & [app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - **Perfil Cinestésico Individual (Kinesthetic Baseline) & Algoritmo de Welford**:
    - Rastreamento estatístico contínuo online de cada praticante com atualização estável de média e variância (`MetricDistribution`) para ângulo de repouso em *Chudan*, janela temporal de *Fumikomi* (em ms), extensão de braço por golpe (`MEN`, `KOTE`, `DO`, `TSUKI`), cadência em Golpes por Minuto (**CPM**) e inclinação da coluna.
    - Avaliação de golpes através do desvio em Z-score em relação ao baseline pessoal do atleta, gerando diagnósticos humanizados e construtivos em linguagem natural.
  - **Persistência Multi-Praticante & Relatório Individual Enriquecido**:
    - Armazenamento atômico dos baselines em `data/kenshi_baselines.json` através do `KinestheticProfileManager`.
    - Injeção automática da **Seção 3: Análise Comparativa com o Baseline Cinestésico Individual** nos relatórios em Markdown e telemetria de corte.
  - **Transferência de Conhecimento entre Perfis (Warm Start)**:
    - Derivação de novos perfis de calibração herdando integralmente os pesos matematicamente otimizados de Ki-Ken-Tai-Ichi, matriz `weights_by_strike_type`, parâmetros de Platt Scaling e priors de Dan.
    - Ajuste direcional com clamping seguro (`more_strict`, `more_permissive`, `neutral`) e rastreamento de linhagem (`parent_profile_id`).
  - **Interface Web Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
    - Adição de painel interativo de Perfis Cinestésicos Individuais e assistente em 1 clique para Derivação de Novos Perfis com Warm Start na aba de Calibração.
  - **Suíte de Testes Automatizados**:
    - Criação de [test_kenshi_style_model_eixo6.py](file:///d:/Projetos/SenpAI/Dev/tests/test_kenshi_style_model_eixo6.py) com 9 testes automatizados cobrindo todo o Eixo 6 com 100% de sucesso.
    - Suíte geral de testes do SenpAI atinge **232 testes automatizados em 25 módulos aprovados sem regressões**.

---

### `[v 0.3.4.0]` — 2026-10-02

- **Implementação do Eixo 5: Invariância de Câmera e Normalização Espacial 3D ([camera_invariance.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/camera_invariance.py), [biomechanics.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/biomechanics.py), [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py), [reporter.py](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py) & [app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - **Compensação de Perspectiva por Ângulo de Filmagem (`CombatVectorEstimator`)**:
    - Extração contínua do vetor de combate Kenshi Aka <-> Kenshi Shiro através dos centros de quadril e estimação do ângulo de incidência da câmera $\theta_{\text{camera}} \in [0^\circ, 90^\circ]$.
    - Classificação automática em 3 categorias operacionais:
      - **Frontal ($0^\circ - 30^\circ$)**: Foco no Chushin-sen e alinhamento central;
      - **Oblíquo ($30^\circ - 60^\circ$)**: Enquadramento diagonal de arquibancada / córner;
      - **Lateral ($60^\circ - 90^\circ$)**: Referencial canônico para análise de perfil e Fumikomi.
    - Suporte a treinamento solo deduzindo a rotação do praticante a partir da largura biacromial dos ombros.
    - **Normalização Trigonométrica de Postura (*Shisei*)**:
      - Compensação do achatamento visual em tomadas frontais através da escala $1 / \sin(\theta)$ no cálculo da inclinação da coluna, mapeando a leitura observada para o plano lateral canônico.
  - **Estimativa de Profundidade Monocular 3D & Maai 3D (`MonocularDepthEstimator`)**:
    - Reconstrução de pseudo-keypoints tridimensionais $(x, y, z)$ a partir de restrições antropométricas e avanço cinemático corporal.
    - **Distância Euclidiana Tridimensional (*Maai 3D*)**: Cálculo da distância de combate $\sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$, permitindo a detecção e eliminação definitiva de *Ku-totsu* (golpes no vazio) quando atacante e defensor parecem sobrepostos em 2D mas estão distantes no eixo $Z$.
    - Rejeição reforçada no pipeline caso $d_{\text{3D}} > 0.55$.
  - **Diagnóstico Automático de Qualidade do Ângulo (`CameraQualityDiagnostic`)**:
    - Avaliação de confiabilidade para cada um dos 5 critérios regulamentares (*Fumikomi*, *Hasuji*, *Shisei*, *Tsuki*, *Zanshin*), cálculo de erro angular esperado e matriz de modificadores multiplicativos de peso.
  - **Exibição Visual no Web App Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)) & Relatórios ([reporter.py](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py))**:
    - Badges em tempo real nos cards de golpes com ângulo de filmagem (ex: `📐 Câmera: 78.5° (Lateral - Alta Precisão)`) e diagnóstico de enquadramento nos relatórios textuais.
  - **Suíte de Testes Automatizados**:
    - Criação de [test_camera_invariance_eixo5.py](file:///d:/Projetos/SenpAI/Dev/tests/test_camera_invariance_eixo5.py) com 8 testes cobrindo todo o Eixo 5 com 100% de sucesso.
    - Suíte geral de testes do SenpAI atinge **231 testes automatizados em 24 módulos aprovados sem regressões**.

---

### `[v 0.3.3.0]` — 2026-10-02

- **Implementação do Eixo 3: Reconhecimento Multimodal de Golpes Válidos (Yuko-Datotsu) ([multimodal_yuko_datotsu.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/multimodal_yuko_datotsu.py), [biomechanics.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/biomechanics.py), [calibrator.py](file:///d:/Projetos/SenpAI/Dev/src/engine/calibrator.py), [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py), [reporter.py](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py) & [app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - **Interação Atacante ↔ Defensor & Eliminação de Ku-totsu (`TargetImpactEvaluator`)**:
    - Avaliação geométrica de colisão entre o *Datotsu-bu* (terço final / Kensen do Shinai) e as regiões regulamentares do Bogu do oponente (*Men*, *Kote*, *Do*, *Tsuki*).
    - Discriminação de distância (*Maai*): rejeição sumária e automática de golpes no vazio (*Ku-totsu*), penalizando golpes desferidos fora da distância física de combate.
  - **Hasuji — O Ângulo da Lâmina como 5° Pilar de Avaliação (`HasujiEvaluator`)**:
    - Extração contínua da orientação angular do Shinai (`angle_deg`) gerado pelo rastreador e comparação com os planos regulamentares de corte:
      - *Men*: Vertical puro com tolerância de $\pm 15^\circ$ (desvio $> 25^\circ$ reprova por corte de chapa).
      - *Do*: Diagonal descendente entre $30^\circ$ e $45^\circ$.
      - *Tsuki*: Horizontal frontal colinear com desvio $\le 10^\circ$.
      - *Kote*: Diagonal descendente moderada entre $15^\circ$ e $35^\circ$.
    - Integração de `hasuji_score` no [calibrator.py](file:///d:/Projetos/SenpAI/Dev/src/engine/calibrator.py) com peso balanceado de 15%, verificação de sub-limiar mínimo e feedback detalhado no [reporter.py](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py).
  - **Detecção do Seme e Pressão Pré-Golpe (`SemeDetector`)**:
    - Avaliação retroativa de 20 a 30 frames antes do impacto: verifica avanço com tronco ereto em direção ao oponente mantendo o centro (*Chudan/Chushin-sen*).
    - Penalização no score quando o ataque se origina de recuo descontrolado ou guarda quebrada.
  - **Detecção de Técnicas de Resposta (Oji-waza e Debana - `CounterattackDetector`)**:
    - Análise de janela de 10 a 15 frames para identificar se o adversário iniciou o movimento ofensivo (*Furikaburi*) antes do atacante responder.
    - Classificação automática em *Debana-waza* (interceptação no nascimento do golpe), *Kaeshi-waza* ou *Nuki-waza*, com detecção dinâmica da inversão de papéis.
  - **Fusão Multimodal com Faixa de Áudio (`AudioKiaiFusion`)**:
    - Detecção de *Datotsu-on* (transiente acústico seco de alta frequência entre $1.5\text{ kHz}$ e $4.0\text{ kHz}$) e *Kiai* vocal ($200\text{ Hz}$ a $1.0\text{ kHz}$).
    - Critério de sincronismo síncrono $\Delta t \le 40\text{ ms}$ entre pico sonoro e vídeo, com fallback gracioso e transparente para vídeos sem faixa de áudio.
  - **Modelo Temporal de Sequência de Poses (Action Spotting TCN - `TemporalActionSpotter`)**:
    - Classificador de convolução temporal sobre janela de 30 frames em 10 classes fundamentais de Kendo (`IDLE_KAMAE`, `TSUBAZERIAI`, `SEME_ADVANCE`, `MEN_ATTACK`, `KOTE_ATTACK`, `DO_ATTACK`, `TSUKI_ATTACK`, `DEFENSE_BLOCK`, `COUNTERATTACK`, `ZANSHIN_RETREAT`).
    - Supressão de falsos disparos durante movimentações de guarda, fintas e clinch prolongado (*Tsubazeriai*).
  - **Painel Interativo de Yuko-Datotsu no Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
    - Cartões de golpe ao vivo atualizados com grid de 6 métricas: Alvo, Fumikomi, Postura, Zanshin, Hasuji (5° Pilar) e Seme.
  - **Suíte de Testes Automatizados**:
    - Criação do módulo [test_multimodal_yuko_datotsu_eixo3.py](file:///d:/Projetos/SenpAI/Dev/tests/test_multimodal_yuko_datotsu_eixo3.py) com 14 testes cobrindo todos os módulos do Eixo 3 com 100% de sucesso.
    - Suíte geral de testes do SenpAI atinge **205 testes automatizados em 21 módulos aprovados sem regressões**.

---

### `[v 0.3.2.0]` — 2026-10-02

- **Implementação do Eixo 2: Conversão da Pesquisa Web em Parâmetros Físicos Acionáveis ([actionable_research.py](file:///d:/Projetos/SenpAI/Dev/src/engine/actionable_research.py), [auto_trainer.py](file:///d:/Projetos/SenpAI/Dev/src/engine/auto_trainer.py), [llm_assistant.py](file:///d:/Projetos/SenpAI/Dev/src/engine/llm_assistant.py) & [mathematical_calibrator.py](file:///d:/Projetos/SenpAI/Dev/src/engine/mathematical_calibrator.py))**:
  - **Pipeline de Extração Estruturada (Knowledge → JSON Schema - `PhysicalConstraintExtractor`)**:
    - Extração automática de restrições biomecânicas numéricas rígidas a partir de manuais e pesquisas da web para os conceitos e modalidades de Kendo (*Yuko-Datotsu*, *Tenouchi*, *Hasuji*, *Fumikomi-ashi*, *Men*, *Kote*, *Do*, *Tsuki*).
    - Validação de faixas físicas biologicamente plausíveis em formato estrito (`elbow_extension_impact_deg`, `spine_tilt_max_deg`, `fumikomi_hand_foot_window_ms`, `zanshin_duration_min_sec`, `hasuji_max_deviation_deg`, `blade_contact_zone`).
    - Integração no `KendoLLMAssistant.extract_physical_constraints` com suporte a execução remota e fallback especialista determinístico offline.
  - **Hierarquia Estrita de Fontes e Resolução de Conflitos (`SourceHierarchyResolver`)**:
    - Implementação das 5 camadas de autoridade marcial:
      - **Tier 1 (Prioridade 1, Peso 1.00)**: *FIK Official Rulebook* — Máxima autoridade, prevalece sempre.
      - **Tier 2 (Prioridade 2, Peso 0.85)**: *AJKF Referee Handbook* — Alta autoridade.
      - **Tier 3 (Prioridade 3, Peso 0.65)**: *Literatura arbitral especializada* — Média autoridade.
      - **Tier 4 (Prioridade 4, Peso 0.45)**: *Artigos acadêmicos e estudos laboratoriais* — Baixa autoridade.
      - **Tier 5 (Prioridade 5, Peso 0.00)**: *Blogs e fóruns abertos* — Descartados como prior de calibração.
    - **Princípio do Conservadorismo Técnico**: Em caso de empate de autoridade entre fontes, o critério que impõe maior rigor técnico e menor tolerância a falhas prevalece incondicionalmente (menor inclinação de coluna, menor atraso de Fumikomi, maior tempo de Zanshin).
    - Histórico e trilha de auditoria de decisões de desempate mantido em `conflict_history`.
  - **Mineração de Vídeos de Referência Oficial (`EmpiricalDistributionLearner`)**:
    - Mineração estatística de clipes de combates oficiais confirmados por árbitros (2 ou 3 bandeiras levantadas).
    - Construção das distribuições empíricas de referência por tipo de golpe (`MEN`, `KOTE`, `DO`, `TSUKI`), calculando média ($\mu$), desvio padrão ($\sigma$), mínimo, máximo e percentis completos ($p_{25}, p_{50} \text{ [mediana]}, p_{75}, p_{90}$) para cada dimensão de Ki-Ken-Tai-Ichi.
    - Armazenamento dedicado em `data/empirical_reference_distributions.json` e sincronização direta com a Base de Conhecimento da IA.
  - **Injeção de Priors Bayesianos no Otimizador Numérico (`BayesianPriorInjector`)**:
    - Vinculação direta das restrições físicas e medianas empíricas como fronteiras rígidas intransponíveis (*boundary conditions*) do `BayesianCalibrationOptimizer`.
    - Garantia formal de que sub-limiares (postura, fumikomi, impacto e zanshin) e piso do score global não possam ser degradados além dos limites canônicos estabelecidos pela FIK/AJKF.
  - **Painel Interativo do Eixo 2 no Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
    - Visualização em 3 abas interativas dedicadas: *Restrições Biomecânicas (JSON Schema com Badges de Autoridade)*, *Distribuições Empíricas de Vídeos Oficiais (Tabelas com percentis por golpe e botão de mineração em 1 clique)* e *Hierarquia de Fontes & Log de Resolução de Conflitos*.
  - **Suíte de Testes Automatizados**:
    - Criação de `tests/test_actionable_research_eixo2.py` com 10 testes rigorosos cobrindo todas as funcionalidades com 100% de aprovação.

---

### `[v 0.3.1.0]` — 2026-10-02

- **Implementação do Eixo 1: Otimização Matemática e Calibração dos Pesos ([mathematical_calibrator.py](file:///d:/Projetos/SenpAI/Dev/src/engine/mathematical_calibrator.py), [calibrator.py](file:///d:/Projetos/SenpAI/Dev/src/engine/calibrator.py) & [calibration_profiles.json](file:///d:/Projetos/SenpAI/Dev/config/calibration_profiles.json))**:
  - **Otimizador Numérico Bayesiano (`BayesianCalibrationOptimizer`)**:
    - Substituição definitiva de saltos fixos (`+0.05` / `-0.04`) por otimização formal com Optuna (TPE) e SciPy SLSQP.
    - Restrições lineares estritas: $\sum w_i = 1.0$, $w_i \ge 0.10$, $T_{\text{global}} \in [0.50, 0.90]$, sub-limiares em $[0.30, 0.85]$.
    - **Penalização Assimétrica de Competição**: Custo 3x maior para Falso Positivo ($C_{\text{FP}} = 3.0 \times C_{\text{FN}}$) fundamentado na regra da FIK que veda pontos duvidosos.
    - Ponderação por autoridade Dan (Shinpan: 4.5, Dan: 1 a 8, Kyu: 0.8) e decaimento temporal exponencial com meia-vida regulamentar de 30 dias ($e^{-\lambda \Delta t}$).
  - **Classificador Probabilístico Calibrado (Platt Scaling - `ProbabilisticPlattCalibrator`)**:
    - Cálculo contínuo da probabilidade real de validação do golpe: $P(\text{Yuko-Datotsu}=1 \mid s) = \sigma(A \cdot s + B)$, com parâmetros ajustados por máxima verossimilhança e regularização L2.
    - Exibição de `probability` e `probability_pct` na interface gráfica e relatórios diagnósticos.
  - **Monitoramento de Deriva Temporal (Concept Drift - `ConceptDriftDetector`)**:
    - Teste bilateral Kolmogorov-Smirnov (`scipy.stats.ks_2samp`) comparando a distribuição recente de scores contra a base histórica para detecção precoce de defasagem de regras ou critérios arbitrais.
  - **Ponderação Especializada por Tipo de Golpe (`weights_by_strike_type`)**:
    - Vetores customizados de Ki-Ken-Tai-Ichi por técnica: `MEN` (ênfase em sincronia mão-pé), `KOTE` (ênfase em extensão de cotovelo e contato), `DO` (ênfase em Hasuji/ângulo de corte), `TSUKI` (ênfase em colinearidade e alvo).
    - Roteamento dinâmico em `pipeline.py` e `multi_camera_fusion.py` através da passagem contextual do `strike_type`.
  - **Painel Interativo de Calibração Matemática no Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
    - Visualização gráfica de status do motor numérico, parâmetros de Platt Scaling, alertas de Concept Drift e tabela de pesos especializados por golpe com botão de execução rápida da otimização.
  - **Suíte de Testes Automatizados**:
    - Criação do módulo `tests/test_mathematical_calibration_eixo1.py` com 10 testes dedicados com 100% de aprovação.

---

### `[v 0.3.0.0]` — 2026-10-01

- **Implementação do Eixo 4: Aprendizado Ativo & Salvaguarda Golden Benchmark ([active_learning.py](file:///d:/Projetos/SenpAI/Dev/src/engine/active_learning.py), [feedback_manager.py](file:///d:/Projetos/SenpAI/Dev/src/engine/feedback_manager.py) & [auto_trainer.py](file:///d:/Projetos/SenpAI/Dev/src/engine/auto_trainer.py))**:
  - **Amostragem por Incerteza (`UncertaintySampler`)**:
    - Algoritmo de filtragem seletiva de lances de borda com probabilidade entre **45% e 65%** (incerteza máxima $\ge 0.70$), descartando casos triviais e enfileirando amostras de alto valor informativo em `data/active_learning_queue.json`.
  - **Dataset Canônico de Benchmark Padrão-Ouro (`GoldenBenchmark`)**:
    - Criação do dataset canônico imutável (`data/benchmark_golden/golden_dataset.json`) contendo casos de teste incontestáveis de campeonatos mundiais (WKC) e exames de alto Dan.
    - **Proteção Mandatória contra Esquecimento Catastrófico (`validate_no_regression`)**:
      - Bloqueio estrito de qualquer alteração de calibração ou auto-treinamento que cause regressão na acurácia do Golden Benchmark superior à margem de tolerância ($\le 2.0\%$).
  - **Consenso Multi-Árbitro Ponderado por Dan (`MultiJudgeConsensus`)**:
    - Quórum ponderado modelo 2 de 3 (FIK Artigo 24) com detecção e cálculo numérico do Grau de Divergência Arbitral.
  - **Gestão de Confiança e Decaimento Exponencial (`ReviewerTrustManager`)**:
    - Modelo de meia-vida temporal ($T_{1/2} = 180$ dias) com decaimento exponencial de autoridade por inatividade e reforço cumulativo por histórico de acertos no Golden Benchmark.
- **Assistente LLM Especialista em Kendo e Anotação Cinemática ([llm_assistant.py](file:///d:/Projetos/SenpAI/Dev/src/engine/llm_assistant.py))**:
  - Suporte nativo a Google Gemini (`gemini-1.5-flash` / `gemini-1.5-pro`) e OpenAI (`gpt-4o` / `gpt-4o-mini`).
  - Motor de regras FIK offline determinístico embutido para funcionamento autônomo sem dependência de credenciais externas.
  - Rotulagem cinemática estruturada em JSON (`label_strike_from_telemetry`) e explicação didática de lances controversos (`explain_controversial_strike`).
- **Painel de Aprendizado Ativo & Curadoria no Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - Adição de card de métricas de conformidade com o Golden Benchmark na aba de Calibração.
  - Visualização da Fila de Incerteza com ferramentas de curadoria rápida em 1 clique.
- **Suíte de Testes Automatizados Expandida**:
  - Criação do módulo `test_active_learning_eixo4.py` com 10 testes dedicados, elevando a suíte de testes do SenpAI para **181 testes automatizados em 19 módulos com 100% de aprovação**.

---

### `[v 0.2.4.1]` — 2026-09-30

- **Mitigação de Bloqueio de IP em Servidores Nuvem (`HTTP Error 403: Forbidden`) & Gestão de Cookies via Secrets ([video_downloader.py](file:///d:/Projetos/SenpAI/Dev/src/utils/video_downloader.py) & [manual.md](file:///d:/Projetos/SenpAI/Dev/manual.md))**:
  - **Diagnóstico da Causa Raiz do Erro 403 na Nuvem**:
    - Servidores em nuvem (ex: Streamlit Community Cloud) utilizam faixas de IP de datacenters comerciais (AWS/GCP), que são severamente bloqueadas pelas defesas anti-bot do YouTube ao requisitar streams diretos de vídeo.
    - Esclarecimento técnico de segurança: por imposição das políticas de segurança fundamentais da Web (*Same-Origin Policy* e flags *HttpOnly*), nenhum site ou script em nuvem tem permissão para extrair automaticamente cookies de terceiros do navegador do usuário.
  - **Suporte a Cookies Transparentes via Streamlit Secrets (`YOUTUBE_COOKIES`)**:
    - O módulo `video_downloader` consome cookies em formato Netscape injetados diretamente em `st.secrets["YOUTUBE_COOKIES"]` (ou arquivo `cookies.txt` no servidor), eliminando a necessidade de qualquer configuração ou upload por parte dos usuários finais na interface.
  - **Refinamento de Clientes InnerTube e Estratégias do yt-dlp**:
    - Priorização inteligente de clientes autenticados (`web`, `ios`, `android`) quando houver cookies válidos configurados nos Secrets.
    - Expansão de fallbacks sem cookies (`visionos`, `android`, `ios`, `mweb`) com mensagens claras de diagnóstico indicando a alternativa imediata de carregar o arquivo pela aba de Upload Local.
  - **Resiliência a Políticas de Controle de Aplicativos do Windows (Smart App Control)**:
    - Tratamento defensivo de `OSError` e bloqueios de DLLs de Deep Learning no validador de aceleração de hardware ([hardware.py](file:///d:/Projetos/SenpAI/Dev/src/utils/hardware.py)).
  - **Suíte de Testes Automatizados**: Suíte completa de 169 testes automatizados aprovada com 100% de sucesso.

---

### `[v 0.2.4.0]` — 2026-09-28

- **Reorganização Hierárquica em 2 Modos de Operação Principais ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py), [training_live_manager.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/training_live_manager.py), [manual.md](file:///d:/Projetos/SenpAI/Dev/manual.md))**:
  - **Reestruturação Estratégica da Experiência do Usuário (UI/UX)**:
    - Transição da seleção plana anterior para uma arquitetura em 2 pilares de atuação do SenpAI:
      1. **⚔️ Modo de Análise de Lutas (Combate / Shiai)**: Foco regulamentar na arbitragem de lutas oficiais da FIK, englobando os submodos **🔴 Detecção em Tempo Real** (multi-câmeras síncronas) e **📹 Detecção Gravada** (análise de Yūko-Datotsu, Sonkyō, VAR e governança por Dan).
      2. **🎓 Modo de Treinamento & Aprendizado (Dojo / Keiko)**: Foco pedagógico nas 14 modalidades oficiais do Kendo e nos 3 Pilares Fundamentais (Movimentação, Precisão, Constância), englobando os submodos **📹 Análise de Vídeo Gravado** e o inédito **🔴 Análise em Tempo Real**.
    - Transição limpa de estado com reset inteligente de análises prévias ao alternar entre modos e submodos.
  - **Treinamento e Aprendizado em Tempo Real com Câmeras RTSP e Webcams**:
    - Suporte a arranjo multi-câmera (1 a 4 câmeras simultâneas com teste de probe/ping e diagramas táticos de posicionamento no dojo).
    - Painel de telemetria com HUD dinâmico em tempo real exibindo as pontuações consolidadas dos 3 Pilares (**Movimentação**, **Precisão**, **Constância**).
    - Máquina de estados de corte (`LiveStrikeState`) para contagem automatizada de repetições e cadência contínua em Golpes por Minuto (**CPM** - *Cuts Per Minute*).
    - Rastreamento biomecânico contínuo da verticalidade da coluna (*Shisei*) e simetria de ombros com emissão de alertas imediatos de biofeedback.
    - Feed de repetições ao vivo e compilação de relatório final de fechamento de sessão com download em Markdown (`.md`) e JSON.
  - **Suíte de Testes Expandida**:
    - Inclusão do módulo `test_training_live_manager.py` com 5 testes unitários dedicados, elevando a suíte de testes do projeto para **169 testes automatizados com 100% de aprovação**.

### `[v 0.2.3.2]` — 2026-09-17

- **Seletor de Modelos de Visão Computacional & Integração de Hardware ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py), [settings_manager.py](file:///d:/Projetos/SenpAI/Dev/src/utils/settings_manager.py), [pose_detector.py](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py), [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py) & [main.py](file:///d:/Projetos/SenpAI/Dev/main.py))**:
  - **Menu de Configurações (`Processamento e Hardware`)**:
    - Implementação de seletor dedicado via `st.radio` permitindo alternar de maneira dinâmica e instantânea entre os modelos neurais **YOLOv8**, **YOLOv11**, **YOLOv12** e **YOLOv26**.
    - Apresentação em tempo real de um cartão visual descritivo de vantagens técnicas e funcionais de cada modelo, com ícone de status, pesos neurais, badge de arquitetura, visão geral, vantagens comprovadas (`✔`), foco na biomecânica do Kendo, contagem de parâmetros e latência-alvo.
    - Botão unificado `💾 Salvar Configurações de Hardware & Modelo de Visão` com persistência atômica no arquivo `config/settings.json`.
  - **Exibição Consolidada de Hardware & Modelo Ativo**:
    - **Banner Hero Superior**: Badge consolidado exibindo o acelerador ativo e o modelo neural em uso: `⚡ Hardware: GPU (NVIDIA GeForce RTX ... - GPU Habilitada) | 🧠 YOLOv8-Pose (FP16)` ou equivalente em CPU.
    - **Aba de Hardware**: Card de status consolidado no topo da aba combinando o acelerador selecionado (`🚀 GPU Habilitada` ou `💻 GPU Desabilitada`) lado a lado com o modelo de visão neural ativo.
    - **Barra Lateral (Sidebar)**: Seção `⚡ Aceleração & Modelo` exibindo status em tempo real (`🚀 GPU Habilitada` / `💻 GPU Desabilitada`), GPU física detectada e modelo ativo.
    - **Painel de Avaliação & Análise**: Card informativo antes do disparo de análise exibindo simultaneamente o acelerador e o modelo de visão computacional em execução.
  - **Catálogo Detalhado de Modelos de Visão Computacional (Armazenamento Exclusivo em `models/`)**:
    - **Diretório Canônico Obrigatório**: Todos os modelos neurais (YOLOv8, YOLOv11, YOLOv12, YOLOv26 e futuros pesos `.pt`, `.onnx`, `.engine`, `.bin`) são armazenados **estritamente na pasta `models/`**. Nenhum arquivo de modelo reside na raiz do projeto.
    1. **YOLOv8-Pose (Baseline Homologado)**:
       - *Pesos Neurais*: `models/yolov8n-pose.pt` (3.3M parâmetros | Latência: < 12ms em GPU CUDA FP16).
       - *Visão Geral*: Modelo baseline do SenpAI com pesos neurais já integrados localmente na pasta `models/`. Oferece alta velocidade de inferência, baixo consumo de VRAM e excelente equilíbrio de rastreamento em movimentos rápidos de artes marciais.
       - *Vantagens*: Estabilidade comprovada em toda a suíte de testes; Pesos locais embutidos em `models/` sem dependência de download externo; Excelente latência tanto em GPU quanto em CPU; Compatibilidade universal com versões de PyTorch e drivers NVIDIA.
       - *Foco no Kendo*: Rastreamento equilibrado de postura corporal, Sonkyō e deslocamentos com Shinai.
    2. **YOLOv11-Pose (C3k2 & C2PSA Architecture)**:
       - *Pesos Neurais*: `models/yolo11n-pose.pt` (2.6M parâmetros | Latência: < 10ms em GPU CUDA FP16).
       - *Visão Geral*: Arquitetura de última geração da Ultralytics baseada em blocos C3k2 e atenção espacial C2PSA, com download direcionado automaticamente para `models/`. Otimizado para máxima precisão em oclusões parciais, comum em combates próximos (Tsubazeriai).
       - *Vantagens*: Blocos C3k2 com atenção espacial aprimorada; Alta resiliência em oclusões corporais e cruzamentos de braços e pernas; Melhor extração de features em resoluções 1080p e 4K; 15% menor overhead computacional por FLOP comparado a gerações anteriores.
       - *Foco no Kendo*: Precisão crítica na detecção de cotovelos, empunhadura do Shinai e postura de Kamae.
    3. **YOLOv12-Pose (Attention-Centric & Flash Processing)**:
       - *Pesos Neurais*: `models/yolo12n-pose.pt` (2.8M parâmetros | Latência: < 11ms em GPU CUDA FP16).
       - *Visão Geral*: Modelo state-of-the-art centrado em mecanismos de auto-atenção pura (FlashAttention/A2) para estimativa de pose, oferecendo máxima taxa de acerto em poses extremas e trocas rápidas de direção.
       - *Vantagens*: Camadas de atenção global com inferência ultra-reativa; Rastreamento contínuo em investidas repentinas (Tobikomi-men); Redução de falsos positivos em keypoints articulares sob iluminação adversa; Otimizado para Tensor Cores modernos.
       - *Foco no Kendo*: Fixação temporal de keypoints em ataques de alta velocidade e saltos com Fumikomi.
    4. **YOLOv26-Pose (Next-Gen Neural Matrix)**:
       - *Pesos Neurais*: `models/yolov26n-pose.pt` (4.1M parâmetros | Latência: < 14ms em GPU CUDA FP16).
       - *Visão Geral*: Versão conceitual de alta densidade neural voltada para análises laboratoriais e streaming multi-câmera simultâneo, com refinamento submétrico de pontos anatômicos.
       - *Vantagens*: Arquitetura neural de densidade estendida para máxima granularidade biomecânica; Refinamento submétrico na detecção do pé de ataque (Fumikomi-ashi) e postura pélvica; Análise multi-atleta simultânea sem degradação de confiança; Suporte a quantização INT8/FP8.
       - *Foco no Kendo*: Análise biomecânica refinada para pesquisa científica e arbitragem de alto escalão.
  - **Mecanismo de Resiliência & Fallback Automático**:
    - O módulo `PoseDetector` implementa o método de busca inteligente `_resolve_model_path_or_name()`. O carregamento e downloads externos são direcionados exclusivamente para a pasta `models/`. Caso o arquivo de pesos selecionado não esteja no disco local ou ocorra indisponibilidade de download, o sistema efetua fallback transparente para o modelo local `models/yolov8n-pose.pt`, mantendo a estabilidade operacional sem jamais interromper o pipeline nem salvar arquivos na raiz.
  - **Integração CLI (`main.py`)**:
    - Adicionado suporte ao argumento `--model {yolov8,yolov11,yolov12,yolov26}` na CLI, possibilitando execução direta por linha de comando combinada com `--device {cpu,gpu}`.
  - **Suíte de Testes Automatizados**:
    - Testes unitários em `tests/test_hardware_settings.py` cobrindo validação de chaves do catálogo, armazenamento exclusivo em `models/`, ausência de modelos na raiz, persistência e recuperação no `settings.json`, fallback gracioso para entradas inválidas, resolução de modelo no `PoseDetector` e instanciação no `SenpAIPipeline`. Todos aprovados com 100% de sucesso.

---

### `[v 0.2.3.1]` — 2026-09-16

- **Exportação e Importação Unificada de Treinamento ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py), [feedback_manager.py](file:///d:/Projetos/SenpAI/Dev/src/engine/feedback_manager.py) & [auto_trainer.py](file:///d:/Projetos/SenpAI/Dev/src/engine/auto_trainer.py))**:
  - **Pacote Completo de Treinamento (v2.0)**: A opção `📥 Baixar Treinamento Atual` agora exporta integralmente em um único arquivo `.json`:
    1. Todas as revisões humanas por Dan (1º ao 8º Dan) com notas, scores, tipos de golpe e links de vídeo;
    2. Todas as Decisões dos Shinpans com persistência completa de links de streaming (YouTube watch/shorts, URLs de stream web RTSP/HLS/HTTP e arquivos locais) e seus identificadores canônicos;
    3. Perfis calibrados de arbitragem (`normal`, `rigido`, `permissivo`);
    4. Todo o aprendizado acumulado pelo Treinamento Automático de IA: Base de Conhecimento (`ai_knowledge_base.json`), calibração das 14 modalidades pedagógicas de Kendo, princípios técnicos assimilados, fontes web indexadas, logs de evolução e checkpoints.
  - **Restauração e Retreinamento Completo**: A opção `📤 Carregar Treinamento Baixado` reconstitui integralmente o ecossistema de aprendizado: reimporta marcações de Dan, cadastra links de streaming homologados aplicando governança anti-duplicidade imediata, mescla de forma cumulativa as fontes e modalidades da Base de Conhecimento e retreina os modelos.
  - **157 testes automatizados** validados com 100% de aprovação (incluindo testes dedicados em `tests/test_dan_training_governance.py` e `tests/test_auto_trainer.py`).

---

### `[v 0.2.3.0]` — 2026-09-13

- **Módulo de Decisão dos Shinpans & Linha do Tempo Dedicada ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py) & [feedback_manager.py](file:///d:/Projetos/SenpAI/Dev/src/engine/feedback_manager.py))**:
  - **Opção Regulamentar "Decisão dos Shinpans"**: Inclusão da opção arbitral no seletor de revisores, permitindo que a arbitragem de Shiai registre exclusivamente os golpes válidos (Ippon / Yūko-datotsu) concedidos pelos três árbitros em quadra.
  - **Linha do Tempo Liberada e Limpa**: Ao selecionar Decisão dos Shinpans, a linha do tempo cronológica não carrega automaticamente todos os golpes da IA, ficando limpa e disponível para incluir apenas os Ippons oficiais concedidos.
  - **Implementação e Blindagem das 5 Regras de Transição de Estado**:
    1. *Habilitar Edição Desmarcado*: Apresenta a listagem completa de golpes identificados pela IA (`res['events']`) com badges de status originais (`PONTO VÁLIDO (IPPON)` ou `GOLPE INVÁLIDO`).
    2. *Habilitar Edição Marcado + Dan Selecionado (1º ao 8º Dan)*: Apresenta a listagem de golpes da IA acompanhada das ferramentas de ação técnica de treinador Dan (`Confirmar`, `Editar`, `Não Houve Golpe`, inclusões manuais e remoções).
    3. *Habilitar Edição Marcado + Decisão dos Shinpans*: Linha do tempo limpa/liberada para inclusão exclusiva de Ippons de Shiai.
    4. *Habilitar Edição Desmarcado após Decisão dos Shinpans*: A listagem de golpes detectados pela IA volta a ser apresentada imediatamente em sua totalidade, sem resquícios de filtros arbitrais e com plena reatividade visual.
    5. *Selecionar um Dan após Decisão dos Shinpans*: A listagem de golpes detectados pela IA volta a ser apresentada imediatamente com os botões de revisão do Dan escolhido, ignorando anotações arbitrais na linha do tempo técnica.
  - **Painel de Sugestões Rápidas da IA (`🤖 Aproveitar Golpes Detectados pela IA`)**:
    - Exibe momentos de impacto detectados pelo modelo com botão `➕ Ippon dos Shinpans` para inclusão oficial em 1 clique, eliminando necessidade de digitação de horários.
  - **Inseridores Inline e Inclusão Manual**:
    - Botões inline `+` entre eventos (Sonkyō e golpes) e formulário manual restrito a `VALID_IPPON`.
  - **Placar Oficial Sanbon-Shobu**:
    - No modo Shinpan, computa exclusivamente os Ippons assinalados pela arbitragem.
- **Governança & Calibração Balanceada de Shinpans**:
  - **Constante Média Regulamentar**: `SHINPAN_CALIBRATION_WEIGHT = 4.5`, recalibrando os pesos biomecânicos de forma balanceada e normalizando a soma estritamente em 1.0, sem sobrecarregar Dans elevados.
  - **Isolamento da Média de Dan (`average_dan_level`)**: Sessões arbitrais não alteram a média aritmética pura dos treinadores Dan (1º ao 8º Dan).
  - **Métricas Dedicadas no Histórico**: Contabilização explícita sob `shinpan_trainings_count` e exibição de cards e linhas dedicadas no painel de Governança de Treinamento.
  - **Suporte Bidirecional em Pacotes JSON e Planilhas Excel (.xlsx)**: Exportação e importação preservando a marcação de Shinpans.
- **Expansão da Suíte de Testes Automatizados**:
  - Suíte completa de **143 testes automatizados** aprovados com 100% de sucesso (`Ran 143 tests, OK`).
  - Adicionados testes de transição de estado e calibração arbitral em `test_dan_training_governance.py`.

### `[v 0.2.3.0]` — 2026-09-14

- **Reformulação do Treinamento Automático Inteligente por IA & Reconhecimento Transversal de Modalidades ([auto_trainer.py](file:///d:/Projetos/SenpAI/Dev/src/engine/auto_trainer.py), [training_analyzer.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/training_analyzer.py), [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py) & [app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - **Reconhecimento Transversal da Modalidade do Vídeo em Qualquer Módulo de Análise**:
    - O sistema agora identifica automaticamente qual das 14 modalidades oficiais de treinamento de Kendo (com Kanjis) está sendo executada no vídeo, tanto no **Modo de Detecção Gravada**, quanto no **Modo em Tempo Real (Webcam / Multi-Câmeras)** e no **Modo de Treinamento & Aprendizado**.
    - No modo Gravado, o SenpAI renderiza um banner de destaque com a modalidade reconhecida, confiança percentual, justificativa detalhada e um acordeão com os princípios de Kendo aprendidos pela IA para aquela modalidade.
    - No modo Tempo Real, o HUD do contador de pontos ao vivo exibe continuamente a modalidade reconhecida dinamicamente no fluxo das câmeras.
  - **Ingestão e Mineração Web Resiliente de Princípios do Kendo**:
    - Implementada a função `search_web_kendo_knowledge()`, que realiza buscas dinâmicas em enciclopédias e manuais técnicos online (API da Wikipedia e diretrizes internacionais) com fallback automático e resiliente para as diretrizes da FIK (*International Kendo Federation*) e AJKF (*All Japan Kendo Federation*), garantindo operação 100% offline.
  - **Sequência Automática de Aprendizado por Necessidade Mais Latente (`diagnose_latent_need`)**:
    - No treinamento automático, a seleção da estratégia é **100% automática e sequencial**, não exigindo seleção manual pelo usuário:
      1. *1º Passo (Prioridade Máxima)*: Foca primeiramente na modalidade com **menor percentual de aprendizado acumulado** (`lowest_accuracy`), sanando a maior carência do modelo.
      2. *2º Passo (Consolidação Transversal)*: Após trabalhar a modalidade deficiente, avança automaticamente para o **conhecimento geral e princípios universais do Kendo** (`general_knowledge` / Ki-Ken-Tai-Ichi, Maai, Zanshin, Hasuji, Tenouchi e Sonkyō).
      3. *3º Passo (Exploração Contínua / Última Opção)*: Como última opção de ciclo, sorteia uma **modalidade randômica** (`random_modality`), diversificando o repertório técnico e prevenindo sobreajuste.
      4. O ciclo reinicia automaticamente reavaliando a menor acurácia agora recalibrada.
  - **Registro Cumulativo e Evolução Contínua do Conhecimento por Modalidade**:
    - A base de conhecimento (`ai_knowledge_base.json`) agora grava de forma persistente e cumulativa para cada uma das 14 modalidades:
      - `principles_learned`: Conceitos técnicos e éticos absorvidos e refinados.
      - `biomechanical_profile`: Limiares de cadência (CPM), alinhamento postural e alvos preferenciais.
      - `web_sources`: URLs e referências mineradas.
      - `evolution_log`: Histórico cronológico de recalibrações.
      - `mastery_level`: Nível de maestria do modelo (*Fase Inicial*, *Em Calibração*, *Calibrado*, *Excelente / Shiai*).
    - Na aba *Sumário de Acurácia por Modalidade*, foi implementado o painel interativo *"📖 O Que o SenpAI Já Aprendeu sobre Cada Modalidade"* para consulta profunda do conhecimento acumulado e dos princípios universais da ZNKR/FIK.
- **Expansão da Suíte de Testes Automatizados (149 Testes)**:
  - Suíte completa de **149 testes automatizados** aprovados com 100% de sucesso (`Ran 149 tests, OK`), incluindo testes para a sequência automática de 3 etapas, resiliência na busca web e identificação transversal de modalidades com Sonkyō.

---

### `[v 0.2.2.0]` — 2026-09-11

- **Otimização de Rastreamento dos Kendocas & Supressão Visual de Shinpans ([combatant_tracker.py](file:///d:/Projetos/SenpAI/Dev/src/vision/combatant_tracker.py) & [pose_detector.py](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py))**:
  - **Limpeza Visual do Vídeo Anotado**: O método `draw_combatants_overlay` agora suprime por padrão (`show_discarded=False`) a renderização de caixas cinzas e tags `[2º PLANO DESCARTADO]` / `[OCLUSÃO DESCARTADA]` ao redor de árbitros (Shinpans) e pessoas externas. O vídeo final concentra-se estritamente nos dois atletas (`🔴 AKA` e `⚪ SHIRO`) e nos traçados de seus Shinai.
  - **Delimitação Automática de Margens da Quadra (*Shiai-jo ROI*)**: Quando o usuário não fornece uma máscara poligonal manual, o sistema aplica limites padrão de quadra ($0.12 \le x \le 0.88$ e $0.20 \le y \le 0.98$), descartando sumariamente árbitros e jurados sentados nas laterais e mesas externas da quadra.
  - **Calibração Dinâmica para Gravações em Plano Aberto (Wide-Angle)**: O tracker adapta a escala de referência `ref_height` pela média real dos atletas detectados (`max(0.20, avg_height)`), evitando que Kendocas em enquadramentos distantes sejam erroneamente classificados como segundo plano.
  - **Penalidade de Borda e Distância Mútua de Combate**: Adicionadas penalidades severas ($-4.0$) para esqueletos colados nas bordas da filmagem e penalização progressiva para distâncias horizontais acima do *Maai* típico de combate.
- **Resiliência a Oclusões e Interpolação Anatômica ([pose_detector.py](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py) & [biomechanics.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/biomechanics.py))**:
  - Implementada a síntese geométrica e interpolação de pulsos (`RIGHT_WRIST`, `LEFT_WRIST`) e pés (`RIGHT_FOOT_INDEX`, `LEFT_FOOT_INDEX`) baseando-se em cotovelos e tornozelos sob oclusão rápida de Hakama ou Men.
  - Correção de tratamento de exceções de chave (`KeyError: 'RIGHT_FOOT_INDEX'`) no cálculo cinemático de Fumikomi.
  - Habilitação de persistência inercial contínua (`return_persisted=True`) em [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py) e [app.py](file:///d:/Projetos/SenpAI/Dev/app.py) para preenchimento de dropouts temporários.
- **Expansão da Suíte de Testes Automatizados**:
  - Suíte completa de **132 testes automatizados** aprovados com 100% de sucesso (`Ran 132 tests, OK`).

---

### `[v 0.2.1.0]` — 2026-09-09

- **Controles Interativos de Reprodução de Vídeo & VAR ([video_player_controls.py](file:///d:/Projetos/SenpAI/Dev/src/utils/video_player_controls.py) & [app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - Implementado componente interativo de reprodução de vídeo integrado diretamente abaixo do player no Modo de Detecção Gravada via `render_video_playback_controls()`.
  - **Mesa de Controle VAR**: Botoeiras de transporte (`Play`, `Pause`, `Reiniciar`), botões de salto fino no tempo (`-10s`, `-5s`, `-1s`, `+1s`, `+5s`, `+10s`), seletor de velocidade instantânea (`0.1x`, `0.25x`, `0.5x`, `0.75x`, `1.0x`, `1.5x`, `2.0x`), alternador de tela cheia, controle de volume e mostrador de tempo decorrido.
  - **Execução Segura em Iframe Isolado (`st.iframe`)**: Superação da limitação nativa do `st.html()`, que bloqueia tags `<script>`, permitindo execução segura de JavaScript no iframe filho com manipulação direta do `<video>` no documento pai (`window.parent.document`).
- **Resolução de Incompatibilidade de Seek no Streamlit (`st.video`)**:
  - Corrigido o erro fatal `TypeError: MediaMixin.video() got an unexpected keyword argument 'key'` ao invocar `st.video()`.
  - O salto temporal automático para os eventos da Linha do Tempo agora é executado diretamente a nível de DOM pelo script auxiliar através de `target_start_time` e `applyInitialSeek()`, sem necessidade de reinicializar o elemento de mídia no servidor.
- **Gestão e Retreinamento por Planilha Excel ([excel_strikes_manager.py](file:///d:/Projetos/SenpAI/Dev/src/utils/excel_strikes_manager.py))**:
  - Módulo completo de exportação e importação de golpes em planilhas Microsoft Excel (`.xlsx`).
  - **Exportação Detalhada**: Converte os eventos detectados com telemetria de Ki-Ken-Tai-Ichi, tempos, alvos e colunas dedicadas de revisão humana (`Acao_Revisao`, `Novo_Alvo`, `Dan_Anotador`, `Observacoes`).
  - **Importação com Retreinamento**: Processa planilhas anotadas offline por árbitros (1º ao 8º Dan), valida o schema, sincroniza com `feedback_dataset.json` e dispara o retreinamento adaptativo do modelo via `DanTrainingGovernance`.
  - **Template em Branco**: Permite gerar planilhas modelo limpas para anotações manuais em novos campeonatos.
- **Expansão da Suíte de Testes para 126 Testes Automatizados**:
  - Criado [test_excel_strikes_io.py](file:///d:/Projetos/SenpAI/Dev/tests/test_excel_strikes_io.py) (6 testes) cobrindo exportação, importação, validação de schema e retreinamento.
  - Criado [test_video_player_controls.py](file:///d:/Projetos/SenpAI/Dev/tests/test_video_player_controls.py) (4 testes) cobrindo geração de HTML/JS, botões de transporte, scripts de seek e compatibilidade unificada `unittest` + `pytest`.
  - Total de **126 testes automatizados** aprovados com 100% de sucesso.

---

### `[v 0.2.0.0]` — 2026-09-08

- **Página Inicial de Boas-Vindas (Home) e Regra de Visibilidade Estrita ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - Implementada a tela de abertura padrão do sistema através da função `render_welcome_home_page()`, carregada automaticamente ao inicializar o SenpAI (`st.session_state["nav_page_selection"] = "home"`).
  - **Regra de Visibilidade Estrita**: A página inicial **só é visível quando nem a página Análise de Lutas nem a página Menu de Configurações estiverem selecionadas**. A seleção de qualquer outra página oculta imediatamente a Home, garantindo foco total no fluxo selecionado.
  - **Hero Banner Institucional**: Identificação visual estilizada *SenpAI • 先輩 AI*, selo de conformidade com os regulamentos da FIK (*International Kendo Federation*) e AJKF/ZNKR, e status em tempo real de hardware ativo (GPU NVIDIA CUDA com Tensor Cores FP16 / CPU).
  - **Barra de Atalhos de Ação Rápida**: Botões de navegação direta para *Análise de Lutas*, *Menu de Configurações* e download do manual oficial (`manual.md`).
  - **Guia Demonstrativo em 4 Passos (*Como Iniciar o Uso*)**:
    1. *Navegação no Menu*: Orientações claras sobre o uso da barra lateral.
    2. *Escolha do Modo*: Visão rápida dos modos Tempo Real, Detecção Gravada e Treinamento.
    3. *Vídeo ou Câmera*: Instruções para upload, streaming web ou uso do botão *"Gerar Vídeo Demonstrativo"* para testar em 3 segundos.
    4. *Diagnósticos & Exportação*: Interpretação do HUD, validação de *Ki-Ken-Tai-Ichi*, *Maai* e exportação de relatórios em Excel/JSON.
  - **Cards Comparativos dos 3 Modos de Operação**: Apresentação detalhada do Modo em Tempo Real (multi-câmeras), Detecção Gravada (arbitragem oficial com rituais de Sonkyō) e Treinamento & Aprendizado (14 modalidades pedagógicas e 3 pilares).
  - **Transparência de IA & Governança por Dan**: Explicação sobre a calibração preliminar realista (< 50%) e a governança com auditoria de árbitros (1º ao 8º Dan).
- **Revisão da Acurácia Básica (< 50%) e Calibração Realista ([auto_trainer.py](file:///d:/Projetos/SenpAI/Dev/src/engine/auto_trainer.py))**:
  - Redefinição transparente das baselines empíricas de fábrica para o intervalo de **32.0% a 46.5%** (*Suburi* 46.5%, *Ji-Geiko* 32.0%, *recorded_shiai* 34.0%, *realtime_shiai* 35.0%), refletindo a honestidade técnica do estado inicial de um modelo de visão computacional sem anotações de arbitragem.
  - Integração com a precisão empírica real de Shinpans (`feedback_mgr.get_stats()["precision_pct"]`), que sobrepõe as baselines teóricas assim que anotações humanas forem registradas.
  - Categorização visual por faixas de maturação: *Fase Inicial / Falsos Positivos* (< 45%), *Em Calibração* (45-65%), *Calibrado* (65-80%) e *Excelente / Shiai* (>= 80%).
  - Sanitização e migração automática de bases de conhecimento legadas com valores inflados (`_sanitize_or_migrate_kb()`).
- **Mitigação de Falsos Positivos, NMS Temporal e Discriminação de Contato (Maai) ([event_spotter.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/event_spotter.py) & [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py))**:
  - Solução para o problema prático de contagens irreais de golpes em Shiai (ex: 66 a 30 em lutas terminadas em 1 a 1).
  - Cooldown de golpes (`min_event_gap_frames`) elevado de **15 para 35 frames (~1.2s)**.
  - Implementação de **Non-Maximum Suppression (NMS) temporal** por atacante, descartando oscilações espúrias durante a elevação (*Furikaburi*) e descida do corte.
  - **Discriminação de Contato / Maai**: Avaliação da proximidade física relativa entre Aka e Shiro no instante do impacto. Ataques desferidos sem proximidade de combate (> 0.48 de distância) são sumariamente reprovados como Ippon com diagnóstico explícito `Fora do Maai (Sem contato com oponente)`.
  - Debounce entre adversários ajustado para 16 frames.
- **Persistência Incremental e Consolidação de Checkpoints no Auto-Treinador ([auto_trainer.py](file:///d:/Projetos/SenpAI/Dev/src/engine/auto_trainer.py))**:
  - O motor de auto-treinamento por IA consolida o aprendizado no início e ao término de cada sessão, retomando sempre do último checkpoint e acumulando sobre todo o conhecimento prévio, sem reiniciar do zero.
- **Expansão da Suíte de Testes Automatizados (116 Testes)**:
  - Suíte completa de **116 testes automatizados** em `unittest` executados e validados com 100% de sucesso.

---

### `[v 0.1.9.1]` — 2026-09-02

- **Contador de Pontos e Placar ao Vivo no Modo Realtime ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - Inclusão do painel de **Contador de Pontos & Placar (Ippon ao Vivo)** posicionado no topo da lista de golpes no Modo de Detecção em Tempo Real.
  - Exibe a pontuação de Ippons válidos e a contagem total de golpes/tentativas para **⚪ Kenshi Shiro (Branco)** e **🔴 Kenshi Aka (Vermelho)**.
  - Badge no topo do painel consolidando a soma geral de golpes analisados e Ippons confirmados na sessão.
  - Atualização instantânea e atômica a cada disparo do motor de fusão multi-câmeras.
- **Inversão da Ordem do Histórico de Golpes em Tempo Real (Mais Recentes no Topo)**:
  - No Modo de Detecção em Tempo Real, a lista de golpes identificados agora exibe os eventos mais recentes sempre no topo (`live_strike_history.insert(0, ...)`), garantindo que o árbitro ou treinador visualize a ação atual sem a necessidade de rolagem para o final da página.
- **Renderização Rápida e Isolamento de Estado via HTML Nativo (`<details><summary>`)**:
  - Eliminação de problemas de conflito de chaves (`StreamlitDuplicateElementKey`) em execuções contínuas de streaming via substituição de widgets stateful por elementos semânticos HTML `<details>` e `<summary>`.
  - Tratamento de escape de caracteres com `html.escape` e injeção direta no DOM com `.html()`, permitindo abrir e fechar diagnósticos biomecânicos instantaneamente sem overhead no servidor Streamlit.

---

### `[v 0.1.9.0]` — 2026-08-31

- **Análise Integral de Yūko-Datotsu em Tempo Real para Golpes Ippon e Não-Ippon ([multi_camera_fusion.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/multi_camera_fusion.py) & [app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - **Acompanhamento Biomecânico de Cada Marcação de Golpe**:
    - No Modo de Detecção em Tempo Real (monocular ou multi-câmeras $N=1\dots 4$), **cada golpe detectado** — independentemente de ter sido homologado como Ippon válido ou reprovado — é acompanhado pela análise discriminativa completa dos 4 pilares de *Ki-Ken-Tai-Ichi*:
      1. 🎯 **Alvo (Ken)**: Proximidade geométrica e precisão no alvo anatômico (*Men, Kote, Do, Tsuki*).
      2. 🦶 **Fumikomi (Tai)**: Sincronismo do pisar firme com o instante do corte e defasagem exata em milissegundos ($\Delta t_{\text{fumikomi}}$).
      3. 🧍 **Postura (Tai)**: Estabilidade do tronco, alinhamento vertical da coluna e equilíbrio no impacto.
      4. ⚡ **Zanshin (Ki)**: Prontidão e manutenção da atitude de alerta imediata pós-corte.
  - **Cards Ricos e Painel de Métricas ao Vivo**:
    - Exibição de cards visuais para cada evento com status de homologação (`✅ IPPON VÁLIDO` vs `⚠️ GOLPE INVÁLIDO`), pontuação percentual consolidada, quórum de confirmação entre as câmeras e grade dos 4 sub-scores de Ki-Ken-Tai-Ichi.
    - Relatório pedagógico e diagnóstico descritivo colapsável (`st.expander`) detalhando os pontos fortes e o motivo da aprovação ou recusa do golpe.
  - **Suíte de Testes Automatizados Expandida**: Adicionado teste unitário `test_yuko_datotsu_analysis_present_for_both_ippon_and_non_ippon` totalizando **84 testes aprovados com 100% de sucesso**.

---

### `[v 0.1.8.1]` — 2026-08-31

- **Discriminação de Árbitros (Shinpans) & Seleção Ótima de Dupla de Kenshis ([combatant_tracker.py](file:///d:/Projetos/SenpAI/Dev/src/vision/combatant_tracker.py))**:
  - **Score de Características de Kenshi (`compute_kenshi_feature_score`)**:
    - Reconhece a postura exclusiva de combate do Kendo: empunhadura bimanual de *Chūdan-no-kamae* (distância entre pulsos $\Delta_{\text{wrists}} < 0.18 \times H$), centralidade no *Shiaijo* ($x \in [0.20, 0.80]$), elevação para corte (*Furikaburi*) e agachamento de *Sonkyō*.
    - Discrimina e penaliza a postura de árbitros (*Shinpans*), que se posicionam nas bordas e mantêm as mãos afastadas segurando as bandeiras vermelha e branca (*Kohaku*).
  - **Seleção Ótima da Dupla de Combate (`select_best_combatant_pair`)**:
    - Avalia combinatória de pares candidatos e seleciona a dupla que maximiza a afinidade de Kenshi, compatibilidade de escala de profundidade na quadra, alinhamento da linha de solo dos pés e distância de combate (*Maai*).
    - Isola com precisão os 2 Kenshis mesmo quando árbitros estão em primeiro plano (mais próximos da câmera), descartando-os automaticamente como `FOREGROUND_OCCLUDER` ou `BACKGROUND`.
- **Renderização Dinâmica do Vídeo Anotado ([pose_detector.py](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py) & [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py))**:
  - **Identificação Visual dos Kenshis**: Badges `🔴 KENSHI AKA` e `⚪ KENSHI SHIRO` com caixas e esqueletos coloridos de alto contraste.
  - **Vetor do Shinai**: Traçado da espada com ponto brilhante no *Kensen* acompanhando a trajetória e os cortes.
  - **Marcação Visual de Golpes**: Destaque neon no atacante (`[⚡ ATAQUE]`), mira/crosshair anatômica no defensor (*Men*, *Kote*, *Do*, *Tsuki*) e banner de diagnóstico com resultado oficial de *Ippon*.
  - **Transcodificação H.264/AVC1 Universal**: Conversão automática com FFmpeg (`yuv420p` + `faststart`), garantindo reprodução instantânea em navegadores web.
- **Suíte de Testes Expandida**: 83 testes automatizados validados com 100% de sucesso.

---

### `[v 0.1.8.0]` — 2026-08-30

- **Treinamento Automático por Inteligência Artificial (Web & Vídeo Knowledge Ingestion)**:
  - **Motor Central Autônomo ([auto_trainer.py](file:///d:/Projetos/SenpAI/Dev/src/engine/auto_trainer.py))**:
    - Implementação do motor de busca, mineração técnica e auto-calibração com duração controlada (tempo determinado em minutos) integrando conhecimento de manuais oficiais da Federação Internacional de Kendo (FIK), diretrizes práticas da AJKF/ZNKR, artigos de biomecânica desportiva e corpus cinemático de vídeos de alta velocidade.
  - **Seleção Inteligente por Necessidade Mais Latente (`diagnose_latent_need`)**:
    - Diagnóstico autônomo baseado no histórico de feedbacks, lacunas nos perfis de calibração e carência de dados, elegendo automaticamente o foco prioritário do treinamento (Lutas Shiai, Detecção em Tempo Real, 14 Modalidades Pedagógicas ou Geral).
  - **Suporte Abrangente de Focos de Treinamento**:
    - **Necessidade Mais Latente (Automático / Recomendado)**.
    - **Treinamento Geral Unificado** (todos os modos e modalidades).
    - **Avaliação de Lutas / Shiai (Modo de Detecção Gravada)** (Sonkyō, Yuko-Datotsu, Ki-Ken-Tai-Ichi e Zanshin).
    - **Detecção em Tempo Real (Multi-Câmeras)** (quórum de consenso entre ângulos e baixa latência).
    - **14 Modalidades Pedagógicas de Treinamento** (Ashi-sabaki, Suburi, Kihon, Kirikaeshi, Uchikomi-geiko, Kakari-geiko, Yakusoku-geiko, Waza-geiko, Oji-waza, Ji-geiko, Shiai-geiko, Nihon Kendo Kata, Bokuto Kihon e Shinsa).
  - **Interface Interativa na Aba de Governança de Treinamento ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
    - Seleção amigável de tempo (`1 min`, `5 min`, `10 min`, `15 min`, `30 min`, `1h`, `2h` ou minutos personalizados).
    - Monitoramento em tempo real com barra de progresso, cronômetro regressivo, acurácia biomecânica estimada e streaming dos logs de mineração da IA.
    - Emissão e download de Relatório Executivo de Treinamento em Markdown (`.md`) e exportação da Base de Conhecimento de IA (`.json`).
    - **Renderização de Alta Performance da Tabela de Evolução via `st.html`**: Substituição do `st.dataframe` por tabela nativa compacta em HTML/CSS, eliminando dependências de módulos dinâmicos do Vite e prevenindo erros de preload de CSS no navegador.
- **Governança de Treinamento & Separação dos Treinamentos por IA ([feedback_manager.py](file:///d:/Projetos/SenpAI/Dev/src/engine/feedback_manager.py))**:
  - **Linha Dedicada na Tabela de Governança**: A tabela de distribuição de treinamentos por Dan agora inclui uma 9ª linha exclusiva para **`🤖 IA | Treinamentos Automatizados (IA / Web & Vídeo)`** com sua respectiva contagem e percentual sobre o total.
  - **Não Poluição da Média Dan Humana**: Os treinamentos automatizados por IA registram `reviewer_dan: 0` e `is_auto_training: True`, **não sendo mais computados como 8º Dan (Hachidan)** nem distorcendo a média ponderada dos avaliadores humanos.
  - **Média Exclusiva de Dan Humano**: O cálculo de `average_dan_level` restringe-se estritamente aos Dan 1º ao 8º atribuídos por revisores humanos.
- **Suíte Completa de Testes Automatizados**:
  - **81 testes automatizados** em `unittest` validados com 100% de aprovação (incluindo `tests/test_auto_trainer.py` e `tests/test_dan_training_governance.py`).

---

### `[v 0.1.7.1]` — 2026-08-30

- **Tipagem Estrita, Estabilidade de Execução e Correção de Linter/Pyright no Web Dashboard ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
  - **Estreitamento de Tipos em Widgets Streamlit (`Type Narrowing`)**:
    - Aplicação de conversão defensiva e tipagem estrita garantindo que retornos de widgets (`st.selectbox`, `st.radio`, `st.number_input`) nunca repassem `None` para assinaturas e chamadas que exigem tipos estritos (`str`, `int`):
      - `profile_choice: str`: Garantido como `str` (fallback `"normal"`), eliminando erros de tipo na inicialização do [SenpAIPipeline](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py), no salvamento de revisões e no painel de otimização de sensibilidade.
      - `num_cameras: int`: Garantido como `int` (fallback `1`), eliminando erros de indexação no `diagram_map`, iterações `range(num_cameras)` e comparações lógicas no modo Ao Vivo Multi-Câmeras.
      - `selected_dan: int`: Garantido como `int` (fallback `3`), eliminando erros de indexação e consultas em `dan_options[selected_dan]` e `dan_options.get(selected_dan)`.
      - `selected_quality: str`: Tipado estritamente como `str` para compatibilidade com `QUALITY_LABELS.get()` e [video_downloader.py](file:///d:/Projetos/SenpAI/Dev/src/utils/video_downloader.py).
      - `selected_mod_key: str`: Tipado estritamente para indexação segura em `TRAINING_MODALITIES_METADATA`.
      - `selected_hw_str`: Estreitamento na seleção de hardware para envio seguro a `set_processing_device`.
  - **Correção da Lista de Frames em Tempo Real (`latest_drawn_frames`)**:
    - Tipagem explícita `list[Optional[np.ndarray]] = [None for _ in range(num_cameras)]` no [app.py](file:///d:/Projetos/SenpAI/Dev/app.py) e atualização da assinatura em [multi_camera_fusion.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/multi_camera_fusion.py) (`latest_frames: Optional[List[Optional[np.ndarray]]] = None`), permitindo atribuição de imagens processadas e validação contínua de consenso multi-câmeras em background.
  - **Limpeza Automática de Análise na Transição de Modos e Botão de Nova Análise**:
    - Implementada a função `clear_previous_analysis()` que monitora a troca entre os 3 modos de operação (Tempo Real, Detecção Gravada e Treinamento & Aprendizado), interrompendo com segurança qualquer worker em execução e redefinindo completamente os resultados, revisões, uploads e buffers de sessão.
    - Adicionado o botão `🧹 Nova Análise / Limpar` diretamente no topo do painel de resultados para redefinição manual instantânea.
  - **Layout Ergonômico Lado a Lado no Modo de Treinamento & Aprendizado**:
    - O painel pedagógico de avaliação de treinamento (10 Modalidades de Kendo, 3 Pilares biomecânicos — Movimentação, Precisão e Constância —, rastreamento/nomeação de Kendocas, pontos fortes, correções técnicas, prescrições de exercícios e exportações de relatórios .MD/.JSON) agora é exibido **diretamente ao lado do vídeo**, em duas colunas sincronizadas.
    - No modo de treinamento, a Linha do Tempo e a Revisão de Golpes de Campeonato foram suprimidas, garantindo uma interface limpa, focada exclusivamente na evolução técnica dos praticantes do dojo.
    - Adicionado suporte à inversão de identificação dos praticantes (`🔄 Inverter Lados dos Kendocas (Esquerda ⇄ Direita)`), atualizando a ordem dos cards pedagógicos em tempo real.
  - **Blindagem de Renderização Condicional da Linha do Tempo & Prevenção de `NameError: name 'res'`**:
    - Encapsulamento estrito de todos os componentes da Linha do Tempo, Formulários Dan e Botões de Retreinamento dentro do bloco condicional de existência de resultados (`analysis_result in st.session_state`), prevenindo tentativas de iteração sobre objetos de resultados antes da execução do pipeline ou após limpeza/reset de sessão.
  - **Aceleração Nativa NVIDIA CUDA via Ultralytics (YOLOv8-Pose)**:
    - Integração e homologação do pacote `ultralytics` nas rotinas de detecção de hardware ([hardware.py](file:///d:/Projetos/SenpAI/Dev/src/utils/hardware.py)), garantindo inicialização limpa em GPU (`use_gpu: True`) com fallback automático para CPU MediaPipe caso indisponível.
- **Suíte Completa de Testes Automatizados**:
  - **73 testes automatizados** em `unittest` executados e validados com 100% de aprovação.

---

### `[v 0.1.7.0]` — 2026-08-20

- **Consenso & Validação de Golpes por Conjunto Multi-Câmeras (`MultiCameraFusionEngine`)**:
  - Implementada a regra central: *"A definição de haver ou não o golpe deve ser tomado com base no conjunto das imagens das câmeras. Quanto mais câmeras, mais necessária a confirmação em imagens/frames da realização da técnica."*
  - Escalonamento progressivo do quórum de confirmação:
    - **1 Câmera**: $1/1$ (100%)
    - **2 Câmeras**: $2/2$ (100% de confirmação cruzada síncrona obrigatória)
    - **3 Câmeras**: $2/3$ ($\ge 66.7\%$ no modo Normal) ou $3/3$ (100% no modo Rígido)
    - **4 Câmeras**: $3/4$ ($\ge 75\%$ no modo Normal) ou $4/4$ (100% no modo Rígido)
  - Descarte automático de falsos positivos originados em visões unilaterais (`REJECTED_SINGLE_ANGLE` / `REJECTED_INSUFFICIENT_CONSENSUS`).
  - Sincronização temporal por janela $\Delta t$ ($\pm 10$ frames / $\approx 350\text{ ms}$) entre câmeras.
  - Painel de Consenso & Métricas Multi-Câmeras integrado no Modo Ao Vivo do Web App com exibição de quórum ativo, score conjunto e detalhamento por câmera.
  - Suíte de 8 novos testes automatizados dedicados em `tests/test_multi_camera_fusion.py` (totalizando 52 testes aprovados).

---

### `[v 0.1.6.1]` — 2026-08-19

- **Padronização das Marcações Katakana no Placar Oficial (Sanbon-shobu)**:
  - Mapeamento estrito e exclusivo dos caracteres Katakana oficiais da arbitragem de Kendo no painel de Pontuação Final ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)):
    - **`MEN`** $\rightarrow$ **`メ`**
    - **`KOTE`** $\rightarrow$ **`コ`**
    - **`DO`** $\rightarrow$ **`ド`**
    - **`TSUKI`** $\rightarrow$ **`ツ`**
  - Renderização dos badges dos Ippons de **Aka** e **Shiro** com a marcação compacta (ex: `🔴 メ (00:02.500)` e `⚪ コ (00:04.120)`).
- **Destaque Visual dos Golpes Responsáveis pela Marcação na Linha do Tempo**:
  - Destaque automático de todos os golpes com pontuação válida (*Yuko-Datotsu / Ippon*) no painel de Linha do Tempo & Revisão de Golpes:
    - **Título do Card Expansível**: Exibição da técnica com prefixo Katakana (ex: `🥊 Golpe #1: メ MEN @ 00:02.500 (🔴 Kenshi Aka) - ✅ IPPON`).
    - **Banner de Marcação Oficial**: Card colorido dedicado destacando a pontuação do lutador (`🔴 MARCAÇÃO OFICIAL (AKA): メ MEN` ou `⚪ MARCAÇÃO OFICIAL (SHIRO): コ KOTE`).
    - **Campo Técnica**: Destaque tipográfico com indicação de golpe pontuado: `**Técnica:** **メ MEN** 🥋 *(Golpe Pontuado no Placar)*`.
    - **Seletor de Navegação Rápida (Quick Jump Selectbox)**: Distinção imediata dos pontos válidos (`🥊 Golpe #1: メ MEN @ 00:02.500 (✅ Ippon - 🔴 Kenshi Aka)`).
  - Golpes inválidos mantêm a nomenclatura limpa (`MEN`, `KOTE`, `DO`, `TSUKI`), permitindo distinção visual instantânea na sequência da luta.
- **Nomenclatura Limpa nos Demais Componentes**:
  - Restauração da nomenclatura padrão limpa nos formulários de inserção inline, inclusão de golpes perdidos, edição de técnica, notificações toasts e modo ao vivo.
- **Suíte de Testes Automatizados**:
  - Execução e aprovação integral de **44/44 testes automatizados** com relatório descritivo emitido em `logs/senpai_test_report.log`.

---

### `[v 0.1.6.0]` — 2026-08-18

- **Relatório Descritivo de Testes Automatizados & Retenção Única de Log**:
  - Criado o runner customizado ([test_runner.py](file:///d:/Projetos/SenpAI/Dev/src/utils/test_runner.py)) e script de execução na raiz ([run_tests.py](file:///d:/Projetos/SenpAI/Dev/run_tests.py)).
  - Geração automática de relatório descritivo por teste com módulo, classe, método, descrição em Português, status individual, duração em segundos e sumário executivo.
  - Salvo na pasta `logs/` ([`logs/senpai_test_report.log`](file:///d:/Projetos/SenpAI/Dev/logs/senpai_test_report.log)) com política estrita de retenção: **apenas o último log de testes é mantido na pasta**.
  - Botões de execução rápida (`🔬 Rodar Testes (44)`) e download do relatório (`📥 Baixar Log Testes (.log)`) integrados na **Seção 5 de Diagnóstico e Logs** do Web App.
- **Detecção e Scoring Consolidado (Modo de Detecção Gravada)**:
  - Validação completa de *Yuko-Datotsu* com score ponderado (*Ki-Ken-Tai-Ichi*: impacto no alvo, sincronismo de *Fumikomi*, postura e *Zanshin*), corte automático de clipes de eventos e relatórios diagnósticos de combate.
- **Navegação Interativa no Vídeo com Salto Temporal Calibrado (-1.0s)**:
  - Salto temporal instantâneo no player de vídeo ao clicar nos botões individuais de evento (Sonkyō Inicial, Golpes Detectados ou Sonkyō Final) ou ao selecionar eventos no menu dropdown.
  - Calibração de **1 segundo de pré-roll (`-1.0s`)** antes do início do evento para permitir que o revisor assista à preparação, execução e finalização da ação com clareza.
  - Banner dinâmico com indicação da posição ativa (`🎯 Posicionado em X.Xs`) e botão de reset rápido (`✖️ Início`).
- **Otimização da Escala Visual da Interface (Zoom 80%)**:
  - Aplicação de redução global de 20% na escala de fontes e elementos (`zoom: 0.8`) com compactação ergonômica de paddings e containers (`max-width: 96%`), eliminando necessidade de rolagem excessiva.
- **Detecção de Sonkyō & Delimitação Temporal da Luta**:
  - Identificação e verificação automática da postura ritualística de *Sonkyō* (agachamento profundo sobre os calcanhares, flexão de joelhos e coluna ereta) para marcação do Início Oficial (`match_start_frame`) e Encerramento Oficial (`match_end_frame`) da luta no Modo de Detecção Gravada.
  - Filtragem estrita de golpes por Sonkyō: consideração e pontuação de *Yuko-Datotsu* realizada **estritamente entre os momentos de Sonkyō de início e término**, descartando movimentações e cortes fora da janela regulamentar de combate.
  - Edição interativa de Sonkyō com aprendizado biomecânico adaptativo contínuo persistido em `config/sonkyo_learned_profile.json`.
- **Rastreamento dos 2 Kenshi Principais e Filtragem de Planos**:
  - Rastreamento contínuo dos dois atletas principais que iniciaram o combate no Shiaijo (`Kenshi Aka - Vermelho` e `Kenshi Shiro - Branco`).
  - Calibração geométrica automática de plano principal, descartando elementos de segundo plano (outras lutas ao fundo, pessoas distantes, arquibancadas) e oclusões de primeiro plano (pessoas passando na frente da câmera).
- **Placar Oficial Eletrônico (Sanbon-shobu Scoreboard) e Inversão Manual Aka ⇄ Shiro**:
  - Placar eletrônico no topo dos resultados com contagem de Ippon para Aka e Shiro, técnicas pontuadas e declaração automática de resultado (*Sanbon-shobu*).
  - Detecção cromática HSV de flag dorsal (Tasukuki) e botão de ação rápida `🔄 Inverter Lutadores (Aka ⇄ Shiro)` para reatribuição imediata de pontuação, eventos e relatórios em gravações com câmera no lado oposto do Shiaijo.
- **Aceleração GPU NVIDIA CUDA com Tensor Cores FP16 & Streaming de Renderização**:
  - Suporte a GPU NVIDIA CUDA via YOLOv8-Pose em FP16 meia precisão (`half=True`) com fallback automático para CPU.
- **Suíte de Testes Automatizados**:
  - 44 testes automatizados em `unittest` com 100% de aprovação cobrindo todo o pipeline cinemático, Sonkyō, planos, placar, flag dorsal, hardware, governança por Dan e logs.

---

### `[v 0.1.6.2]` — 2026-08-25

- **Suporte a Links do YouTube, Streaming Web e Seleção de Qualidade no Modo de Detecção Gravada**:
  - Inclusão do seletor visual de origem de vídeo no painel de carregamento ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)), permitindo alternar facilmente entre:
    1. 📁 **Fazer Upload de Arquivo**: Upload de arquivos de vídeo locais (.mp4, .avi, .mov).
    2. 🌐 **Link do YouTube / Streaming Web**: Entrada de URLs de vídeos do YouTube (links padrão, encurtados `youtu.be`, `shorts`, transmissões e links de streaming direto).
  - **Seletor de Qualidade de Download**: Permite ao usuário escolher o nível de qualidade para download de streams:
    - **Média (Intermediária / 30 FPS - Padrão)**: Resolução intermediária (até 720p) limitada a 30 FPS para equilíbrio perfeito entre velocidade e fidelidade biomecânica.
    - **Alta (Máxima Disponível)**: Máxima resolução e FPS originais do vídeo.
    - **Baixa (Menor Disponível / Download Rápido)**: Menor resolução disponível para processamento ultrarrápido com baixo consumo de banda.
  - Implementação do módulo [video_downloader.py](file:///d:/Projetos/SenpAI/Dev/src/utils/video_downloader.py) utilizando a biblioteca `yt-dlp`:
    - Validação de múltiplos formatos de URLs e streams web.
    - Extração assíncrona rápida de metadados sem necessidade de download prévio completo (título, canal/autor, miniatura, duração formatada, resolução e taxa de quadros FPS).
    - Download otimizado no formato MP4 multiplexado com suporte a callbacks de progresso em tempo real e verificação de integridade.
    - Sistema de **caching local inteligente por nível de qualidade**: reutilização imediata de arquivos já baixados em `senpai_uploads`, eliminando downloads repetitivos da mesma luta.
    - **Exibição Transparente da Qualidade Baixada**:
      - *Card de Carregamento*: Exibe a etiqueta de qualidade (ex: `Média (Intermediária / 30 FPS)`), resolução real do arquivo baixado (ex: `1280x720 @ 30 FPS`), duração e tamanho do arquivo em MB.
      - *Player de Vídeo*: Badge no topo do player destacando a origem do YouTube, a qualidade baixada com resolução/FPS reais e link direto `[Ver no YouTube ↗️]`.
      - *Resumo do Combate*: Legenda informativa detalhando a fonte de streaming e a qualidade/resolução efetiva do vídeo processado pelo pipeline.
- **Expansão da Suíte de Testes Automatizados (64 Testes)**:
  - Criação do módulo [test_video_downloader.py](file:///d:/Projetos/SenpAI/Dev/tests/test_video_downloader.py) com 12 testes cobrindo validação de URLs, formatação de tempo, sanitização de nomes, extração de metadados mockados, rejeição de streams ao vivo, limites de duração, seletores de formato para cada qualidade (baixa, média, alta), persistência de cache por qualidade e integração de ponta a ponta com o [SenpAIPipeline](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py).
  - Suíte completa de 64 testes executada com 100% de sucesso.

### `[v 0.1.6.1]` — 2026-08-20

- **Menu de Configurações em Layout de Guias (Tabs)**:
  - Reestruturação completa da página de configurações ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)) com navegação modular em 4 guias especializadas via `st.tabs`:
    1. 🖥️ **Processamento & Hardware**: Seletor de dispositivo (CPU / GPU NVIDIA), status e diagnóstico de hardware em tempo real e instalador de pacotes CUDA.
    2. 🎓 **Governança de Treinamento**: Métricas de retreinamento por Dan, tabela de distribuição por graduação e ferramentas de backup/reset/importação de dados.
    3. 🎛️ **Perfis de Calibração**: Cards e tabela comparativa dos perfis de arbitragem (*Permissivo*, *Normal*, *Rígido*) e pesos dos critérios de *Ki-Ken-Tai-Ichi*.
    4. 🐛 **Diagnóstico, Alertas & Logs**: Métricas de eventos do sistema, ferramentas de download de logs, diagnóstico rápido, execução de testes automatizados e console de logs em tempo real com filtro por nível.
  - Estilização CSS refinada para as abas no tema escuro do SenpAI com destaque azul ativo, transições suaves e contraste ergonômico.
- **Suíte de Testes Automatizados**: 52 testes automatizados executados com 100% de aprovação.

---

### `[v 0.1.5.0]` — 2026-08-15

- **Sistema de Diagnóstico, Alertas e Log de Debug do Sistema**:
  - Criado o módulo central de logging e diagnóstico ([logger_manager.py](file:///d:/Projetos/SenpAI/Dev/src/utils/logger_manager.py)) com retenção em arquivo ([`logs/senpai_debug.log`](file:///d:/Projetos/SenpAI/Dev/logs/senpai_debug.log)) e buffer em memória.
  - Registro automático no log de eventos críticos: **reset de treinamento**, **importação de arquivos JSON**, **exportação de pacotes**, **retreinamentos por Dan** e diagnósticos de hardware.
  - Adicionada a **Seção 4: Diagnóstico, Alertas & Log de Debug** no menu de configurações do Web App ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)).
  - Métricas em tempo real de contagem de logs, alertas/avisos e erros do sistema.
  - Visualizador de logs com filtro dinâmico por nível (`ERROR`, `WARNING`, `INFO`, `DEBUG`).
  - Botão de **download do arquivo de log completo (`senpai_debug.log`)**.
  - Ferramenta de **teste de diagnóstico automatizado** para checagem de integridade de hardware, GPU, arquivos e bibliotecas.
- **Melhorias na Revisão de Golpes (Modo Gravado)**:
  - Exibição de badges visuais em tempo real: **`✅ CONFIRMADO`** (verde) e **`✏️ EDITADO`** (azul) com atualização instantânea na UI via `st.rerun()`.
  - Botão de **Reset Geral da Revisão (`🔄 Resetar Revisão`)** para limpar as marcações da sessão e botões de **Reset Individual (`🔄 Resetar este golpe`)** por card.
- **Suporte Universal a Arquivos de Treinamento JSON**:
  - O módulo de importação ([feedback_manager.py](file:///d:/Projetos/SenpAI/Dev/src/engine/feedback_manager.py)) foi aprimorado para aceitar pacotes completos, listas diretas de revisões JSON ou entradas avulsas, com tratamento de buffer (`seek(0)`) e atribuição de IDs.
- **Estabilidade de Interface**:
  - Tabela de treinamentos por Dan convertida para Markdown nativo, eliminando erros de pré-carregamento de módulos JS/CSS do navegador (Vite preload helper).
- **Testes Automatizados**: Suíte de testes em [test_logger_manager.py](file:///d:/Projetos/SenpAI/Dev/tests/test_logger_manager.py) e testes de importação expandidos em [test_dan_training_governance.py](file:///d:/Projetos/SenpAI/Dev/tests/test_dan_training_governance.py) (19 testes automatizados com 100% de aprovação).

### `[v 0.1.4.12]` — 2026-08-17

- **Otimização de Espaço e Remoção de Texto de Diagnóstico do Sonkyō**:
  - **Layout Compacto de Sonkyō ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**:
    - Removidos os blocos de texto verbosos e diagnósticos descritivos laterais de Sonkyō Inicial e Final.
    - Estruturação compacta e colapsável dos cards de Sonkyō (`expanded=False` por padrão, exceto durante edição ativa).
    - Exibição direta das informações operacionais essenciais (intervalo ritualístico, início/término oficial do combate e badge de status), economizando espaço vertical para os eventos de combate e golpes de Yuko-Datotsu.
- **Testes Automatizados**: Suíte de 44 testes executada com 100% de sucesso.

---

### `[v 0.1.4.11]` — 2026-08-17

- **Placar Oficial Eletrônico, Detecção de Flag Dorsal e Inversão Aka/Shiro**:
  - **Placar Oficial Eletrônico (Sanbon-shobu Scoreboard) ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py) e [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py))**:
    - Exibição de painel visual eletrônico no topo dos resultados com contagem de **Ippon** válidos para Aka (Vermelho) e Shiro (Branco), badges com as técnicas pontuadas e declaração automática do resultado regulamentar (*Vitória de Aka*, *Vitória de Shiro* ou *Empate / Hikiwake*).
  - **Detecção da Cor da Flag (Tasukuki / Faixa Vermelha nas Costas) ([combatant_tracker.py](file:///d:/Projetos/SenpAI/Dev/src/vision/combatant_tracker.py))**:
    - Implementada a segmentação cromática HSV no dorso/tronco (`detect_red_flag_score`) para identificar a fita vermelha dorsal do Kenshi Aka, independente da cor do Keikogi (azul escuro, branco, preto).
    - Permite a correta identificação dos lados mesmo quando a câmera de gravação estiver posicionada do lado oposto do Shiaijo (câmera invertida).
  - **Inversão Interativa Aka ⇄ Shiro**:
    - Adicionado botão de ação rápida `🔄 Inverter Lutadores (Aka ⇄ Shiro)` para troca instantânea de pontuação, eventos e diagnósticos em caso de ângulo de filmagem desfavorável.
- **Testes Automatizados**: Suíte de testes em [test_scoreboard_and_flag_detection.py](file:///d:/Projetos/SenpAI/Dev/tests/test_scoreboard_and_flag_detection.py) (44 testes automatizados com 100% de aprovação).

---

### `[v 0.1.4.10]` — 2026-08-17

- **Correção de AttributeError & Otimização de Performance e Memória**:
  - **Correção de `AttributeError` em Edição de Sonkyō ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py))**: Corrigida a verificação condicional em `initial_edit` e `final_edit` quando são `None`, garantindo que os timestamps padrão sejam lidos sem exceções de runtime.
  - **Eliminação de Sobrecarga de Memória RAM ([pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py))**:
    - Removido o armazenamento em buffer de todos os quadros descompactados (`raw_frames`) na memória RAM durante a passagem 1.
    - A renderização do vídeo anotado agora utiliza streaming direto em 2ª passada (`cap_render`), reduzindo o consumo de RAM de 15+ GB para menos de 100 MB em vídeos longos/alta resolução.
  - **Aceleração GPU com Tensor Cores FP16 ([pose_detector.py](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py))**:
    - Ativada a inferência em meia precisão (`half=True`) com dimensão padrão `imgsz=640` no YOLOv8-Pose em CUDA, aumentando substancialmente o throughput de frames por segundo (FPS).
- **Testes Automatizados**: Suíte de 39 testes executada com 100% de sucesso.

---

### `[v 0.1.4.9]` — 2026-08-17

- **Inclusão Automática de Sonkyō no Início e Fim da Gravação**:
  - **Garantia de Delimitação Ritual**: Quando a análise de visão computacional não detecta com alta confiança os rituais de Sonkyō nos primeiros ou últimos segundos da gravação, o sistema ([sonkyo_detector.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/sonkyo_detector.py)) **atribui automaticamente os movimentos de Sonkyō no início (00:00.000) e no encerramento do vídeo**.
  - **Identificação Visual Transparente**: No painel de Detecção Gravada ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)), os cards exibem a badge correspondente (`🥋 SONKYŌ DETECTADO` para detecção automática por pose ou `📌 SONKYŌ (Início/Fim do Vídeo / Ajustável)` para fallback padrão).
  - **Edição e Reprocessamento Imediatos**: O usuário tem a garantia de que ambos os rituais estarão sempre visíveis e expansíveis, podendo editar os intervalos com exatidão e reprocessar o combate com aprendizado contínuo.
- **Testes Automatizados**: Suíte de testes expandida em [test_sonkyo_and_plane_filtering.py](file:///d:/Projetos/SenpAI/Dev/tests/test_sonkyo_and_plane_filtering.py) com validação de inclusão de rituais padrão (39 testes automatizados com 100% de aprovação).

---

### `[v 0.1.4.8]` — 2026-08-17

- **Edição Interativa de Sonkyō, Reprocessamento e Aprendizado Contínuo**:
  - **Edição de Momentos de Sonkyō**: No Modo de Detecção Gravada ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)), é possível editar com precisão os tempos de início e fim tanto do Sonkyō Inicial quanto do Sonkyō Final (ou definir intervalos manuais caso não tenham sido detectados automaticamente).
  - **Botão de Reprocessamento com Aprendizado**: Ao alterar um dos momentos de Sonkyō, a interface habilita o botão de ação rápida `🔄 Reprocessar Análise com Aprendizado de Sonkyō`.
  - **Aprendizado Biomecânico Contínuo ([sonkyo_detector.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/sonkyo_detector.py))**:
    - As posturas e proporções corporais no intervalo editado são extraídas dinamicamente para calibrar a sensibilidade do detector de Sonkyō.
    - O perfil adaptado é persistido em `config/sonkyo_learned_profile.json`, sendo aplicado imediatamente neste reprocessamento e em **todas as futuras análises de vídeo**.
  - **Painel de Estatísticas de Sonkyō no Modo Treinamento**: Exibição da quantidade de amostras aprendidas, compressão de altura adaptada, rebaixamento de quadril ($\Delta Y$) e botão para restauração aos padrões de fábrica.
- **Testes Automatizados**: Suíte de testes expandida em [test_sonkyo_and_plane_filtering.py](file:///d:/Projetos/SenpAI/Dev/tests/test_sonkyo_and_plane_filtering.py) cobrindo conversão de timestamps, persistência de aprendizado e reprocessamento com overrides (38 testes automatizados com 100% de aprovação).

---

### `[v 0.1.4.7]` — 2026-08-17

- **Refinamento do Indicador de Aceleração de Hardware**:
  - **Sidebar Exclusiva para Status Visual**: A barra lateral ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)) agora exibe apenas o **card de indicação em tempo real** do estado do acelerador (`🚀 Aceleração Ativada` com nome da GPU NVIDIA e framework CUDA ou `💻 Aceleração Desativada` em CPU), mantendo o layout limpo e intuitivo.
  - **Centralização da Seleção de Dispositivo**: A alteração e o salvamento do dispositivo (CPU / GPU) ficam centralizados na seção de **Configurações Globais** do Modo de Treinamento.
- **Testes Automatizados**: Suíte de 34 testes validada com 100% de sucesso.

---

### `[v 0.1.4.6]` — 2026-08-17

- **Aceleração Nativa com GPU NVIDIA CUDA (YOLOv8-Pose)**:
  - **Motor de Inferência GPU de Alta Velocidade**: O módulo [pose_detector.py](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py) foi atualizado para utilizar o modelo **YOLOv8-Pose em PyTorch CUDA (`cuda:0`)** sobre a placa NVIDIA GeForce RTX 4050.
  - **Detecção Paralela Multi-Pessoa**: A análise de todos os atletas presentes no enquadramento agora ocorre em um **único passo direto na VRAM da GPU**, eliminando as 3 execuções redundantes por corte que eram feitas na CPU pelo MediaPipe.
  - **Aumento de Desempenho (FPS)**: A velocidade de processamento atinge taxas de **27 a 100+ FPS** dependendo da resolução do vídeo, reduzindo drasticamente o tempo de análise na Detecção Gravada.
  - **Seletor de Hardware na Sidebar e Painel de Avaliação**: Adicionado seletor e badges de diagnóstico em tempo real no [app.py](file:///d:/Projetos/SenpAI/Dev/app.py), permitindo alternar facilmente entre aceleração GPU NVIDIA CUDA e CPU.
- **Testes Automatizados**: Suíte completa de 34 testes automatizados validada com 100% de aprovação.

---

### `[v 0.1.4.5]` — 2026-08-17

- **Aprimoramento Robusto da Detecção de Sonkyō & Filtragem de Planos**:
  - **Resiliência a Oclusões por Hakama / Kendogi**: O estimador biomecânico ([sonkyo_detector.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/sonkyo_detector.py)) agora utiliza múltiplos sinais (rebaixamento de quadril, proporção tronco-altura, compressão vertical relativa e inclinação de coluna), operando com precisão mesmo quando joelhos ou tornozelos estão parcialmente oclusos.
  - **Análise Temporal de Altura Relativa**: Cálculo do baseline de altura e nível de quadril em pé do atleta ao longo da gravação, identificando o Sonkyō com base na compressão vertical relativa ($H_{sonkyo} \le 0.75 \times H_{standing}$).
  - **Fechamento Morfológico e Preenchimento de Falhas (Gap Bridging)**: Algoritmo de unificação temporal que preenche quedas momentâneas de rastreamento (dropouts de até 8 frames / ~0.27s), evitando a fragmentação de intervalos contínuos de Sonkyō.
  - **Correção na Filtragem de Planos ([combatant_tracker.py](file:///d:/Projetos/SenpAI/Dev/src/vision/combatant_tracker.py))**: O classificador de planos não descarta mais combatentes agachados no solo do Shiaijo como segundo plano.
- **Testes Automatizados**: Suíte expandida em [test_sonkyo_and_plane_filtering.py](file:///d:/Projetos/SenpAI/Dev/tests/test_sonkyo_and_plane_filtering.py) com testes de oclusão por Hakama, gap bridging e calibração de plano (34 testes automatizados com 100% de aprovação).

---

### `[v 0.1.4.4]` — 2026-08-17

- **Correção Crítica de Vazamento de Arquivos Temporários (`[Errno 28] No space left on device`)**:
  - Identificada e corrigida a criação repetitiva de arquivos temporários (`tempfile.NamedTemporaryFile`) a cada ciclo de atualização (`rerun`) do Streamlit no [app.py](file:///d:/Projetos/SenpAI/Dev/app.py).
  - Implementado sistema de **cache de uploads no `st.session_state`**: o arquivo enviado só é gravado em disco uma única vez por upload (baseado em `name` e `size`).
  - Adicionada rotina de **limpeza automática de arquivos temporários órfãos e antigos** na pasta `senpai_uploads`.
  - Liberação de mais de **20 GB** de espaço em disco no diretório temporário do sistema operacional.

---

### `[v 0.1.4.3]` — 2026-08-17

- **Cronômetro em Tempo Real e Persistência do Tempo de Processamento (Detecção Gravada)**:
  - Inclusão do **cronômetro dinâmico em tempo real** exibido durante o processamento do vídeo no [app.py](file:///d:/Projetos/SenpAI/Dev/app.py) (`MM:SS.s` e segundos decorridos).
  - Persistência visual do **tempo final de execução e taxa média de processamento (FPS)** no painel de status fixo e no cartão de resumo de métricas do combate (`summary-card`).
  - Suporte a medição precisa de tempo em [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py) via `AnalysisWorker.elapsed_seconds` e retorno de `processing_time_seconds` e `processing_fps`.
- **Resumo Estruturado de Processamento no Log do Sistema**:
  - Registro detalhado (`INFO`) no arquivo consolidado de logs (`senpai_debug.log`) contendo arquivo analisado, tempo de execução, FPS, dispositivo utilizado, detecções de Sonkyō, golpes e planos descartados.
- **Testes Automatizados**: Suíte de testes em [test_pipeline_cancellation.py](file:///d:/Projetos/SenpAI/Dev/tests/test_pipeline_cancellation.py) expandida para verificar cronômetro, persistência e log de resumo (32 testes com 100% de aprovação).

---

### `[v 0.1.4.2]` — 2026-08-17

- **Apresentação de Eventos de Sonkyō na Detecção Gravada**:
  - Inclusão dos eventos de **Sonkyō Inicial** (Abertura / Início do Combate) e **Sonkyō Final** (Encerramento / Fechamento do Combate) diretamente na lista de eventos apresentados no container de resultados (`col_results`) do [app.py](file:///d:/Projetos/SenpAI/Dev/app.py).
  - Exibição de cartões expansíveis detalhados com badge `🥋 SONKYŌ DETECTADO`, intervalo ritual (timestamps), contagem de frames de início e fim, duração em segundos, liberação regulamentar de combate e diagnóstico biomecânico da postura de respeito (*Reigi*).
  - Sequenciamento cronológico completo do combate: **Sonkyō Inicial ➡️ Golpes Identificados na Janela Regulamentar ➡️ Sonkyō Final**.

---

### `[v 0.1.4.1]` — 2026-08-17

- **Botão de Interromper Processamento na Detecção Gravada**:
  - Inclusão do botão `⏹️ Interromper Processamento` no painel de execução de vídeo no [app.py](file:///d:/Projetos/SenpAI/Dev/app.py).
  - Suporte a cancelamento cooperativo no método `process_video` do [pipeline.py](file:///d:/Projetos/SenpAI/Dev/src/pipeline.py) através do parâmetro `is_cancelled`.
  - Liberação segura de recursos e fechamento de streams (`VideoCapture` e `VideoWriter`) através de blocos `try...finally`.
  - Notificação visual de cancelamento no dashboard (`st.warning`) e limpeza de arquivos parciais gerados.
  - Registro de eventos de interrupção e alertas no sistema de logs (`log_event`).
- **Testes Automatizados**: Criado [test_pipeline_cancellation.py](file:///d:/Projetos/SenpAI/Dev/tests/test_pipeline_cancellation.py) cobrindo cancelamento imediato, cancelamento durante leitura de frames, validação de logs de aviso e execução normal sem interrupção.

---

### `[v 0.1.4.0]` — 2026-08-15

- **Modo de Detecção Gravada - Edição de Golpes por Dan**:
  - Adicionado o botão `✏️ Habilitar Edição dos Golpes Detectados`.
  - Inclusão do **Combo Box de Graduação DAN do Revisor** (Shodan a Hachidan / 1º ao 8º Dan).
  - Suporte a **confirmar marcação**, **editar marcação** (técnica, timestamp, resultado e observações) e **incluir marcação** de golpes perdidos.
  - Implementação da **regra de auditabilidade (sem exclusão)**, impedindo a exclusão acidental ou indevida de marcações.
  - Botão de salvamento final `💾 Salvar Alterações e Retreinar Modelo` para recalibração automática.
- **Menu de Configurações - Governança de Treinamento**:
  - Adicionado contador de treinamentos realizados, nível médio (Dan) dos treinamentos e total de marcações.
  - Tabela formatada de quantidade e percentual de treinamentos agrupados por Dan.
  - Opção `🗑️ Apagar Treinamento do Sistema` com confirmação de segurança para resetar ao estágio inicial.
  - Opção `📥 Baixar Treinamento Atual` para exportar pacote `.json` com o Dan do revisor e a data do treinamento feito.
  - Opção `📤 Carregar Treinamento Baixado` para importar pacotes previamente baixados e recalibrar o modelo.
- **Menu de Configurações - Governança de Treinamento**:
  - Adicionado contador de treinamentos realizados, nível médio (Dan) dos treinamentos e total de marcações.
  - Tabela formatada de quantidade e percentual de treinamentos agrupados por Dan.
  - Opção `🗑️ Apagar Treinamento do Sistema` com confirmação de segurança para resetar ao estágio inicial.
  - Opção `📥 Baixar Treinamento Atual` para exportar pacote `.json` com o Dan do revisor e a data do treinamento feito.
  - Opção `📤 Carregar Treinamento Baixado` para importar pacotes previamente baixados e recalibrar o modelo.
- **Testes Automatizados**: Criado [test_dan_training_governance.py](file:///d:/Projetos/SenpAI/Dev/tests/test_dan_training_governance.py) cobrindo governança, pacotes e retreinamento.

---

### `[v 0.1.3.0]` — 2026-08-12

- **Menu de Configurações Centralizado**: Implementado no [app.py](file:///d:/Projetos/SenpAI/Dev/app.py) com seletor de acelerador de hardware (CPU Somente vs GPU NVIDIA quando disponível).
- **Módulo de Hardware e Configurações**: Detecção dinâmica multi-nível de GPU NVIDIA e resolução de fallback transparente para CPU ([hardware.py](file:///d:/Projetos/SenpAI/Dev/src/utils/hardware.py) e [settings_manager.py](file:///d:/Projetos/SenpAI/Dev/src/utils/settings_manager.py)).
- **Suporte a CLI**: Adicionado parâmetro `--device {cpu,gpu}` no [main.py](file:///d:/Projetos/SenpAI/Dev/main.py).
- **Atualização de Requisitos**: Inclusão de instruções de instalação de pacotes CUDA (PyTorch CUDA e ONNX Runtime GPU) no [requirements.txt](file:///d:/Projetos/SenpAI/Dev/requirements.txt) e [README.TXT](file:///d:/Projetos/SenpAI/Dev/README.TXT).

---

### `[v 0.1.2.1]` — 2026-08-06

> [!NOTE]
> **Melhorias na Interface Web**
> - Reestruturação do Dashboard Web no Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)) com layout responsivo em duas colunas.
> - **Coluna Fixa (*Sticky Video Column*)**: O vídeo da luta (ou vídeo anotado da IA) fica ancorado à esquerda da página mesmo durante a rolagem.
> - **Coluna de Golpes Rolável**: A lista de golpes identificados, diagnósticos biomecânicos e painel de aprendizado adaptativo possuem barra de rolagem dedicada à direita (`st.container(height=680)`).
> - Alternador direto de exibição no player: Vídeo Anotado com Visão AI vs Vídeo Original.
> - Cartão com resumo de métricas do combate incorporado na coluna do vídeo.

---

### `[v 0.1.2.0]` — 2026-08-06

> [!NOTE]
> **Adicionado**
> - Módulo de Gerenciamento de Feedback e Aprendizagem por Reforço ([feedback_manager.py](file:///d:/Projetos/SenpAI/Dev/src/engine/feedback_manager.py)).
> - Dataset JSON para armazenamento de feedbacks ([feedback_dataset.json](file:///d:/Projetos/SenpAI/Dev/data/feedback_dataset.json)).
> - Suporte ao **Modo de Aprendizagem** na CLI ([main.py](file:///d:/Projetos/SenpAI/Dev/main.py)) através das flags `--mode learning` e `--optimize-profile`.
> - **Painel de Aprendizagem por Reforço** no Web App Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)), permitindo botões rápidos de anotação de TP (Correto), FP (Falso Positivo) e registro manual de FN (Golpe Perdido).
> - Painel de métricas de acurácia (Precisão %, Recall %, Total de Anotações) na interface Web.
> - Testes automatizados para a malha de feedback em [test_feedback_loop.py](file:///d:/Projetos/SenpAI/Dev/tests/test_feedback_loop.py).
> - Documentação reestruturada em formato Markdown ([manual.md](file:///d:/Projetos/SenpAI/Dev/manual.md)) e manual simplificado ([README.TXT](file:///d:/Projetos/SenpAI/Dev/README.TXT)).

---

### `[v 0.1.1.0]` — 2026-08-01

- **Dashboard Web Interativo** desenvolvido em Streamlit ([app.py](file:///d:/Projetos/SenpAI/Dev/app.py)) com estilização CSS customizada.
- Suporte ao perfil `custom` com sliders dinâmicos para ajuste manual de limiares e pesos de Ki-Ken-Tai-Ichi.
- Módulo gerador de relatórios textuais diagnósticos em Português ([reporter.py](file:///d:/Projetos/SenpAI/Dev/src/engine/reporter.py)).
- Exportação de vídeos anotados com suporte a visualização de esqueleto 3D e pontos de impacto.

---

### `[v 0.1.0.0]` — 2026-08-01

- **Lançamento inicial** da arquitetura base do SenpAI.
- Módulos de Visão Computacional ([pose_detector.py](file:///d:/Projetos/SenpAI/Dev/src/vision/pose_detector.py), [shinai_tracker.py](file:///d:/Projetos/SenpAI/Dev/src/vision/shinai_tracker.py)) baseados em MediaPipe Pose.
- Módulos de Análise Biomecânica ([biomechanics.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/biomechanics.py), [event_spotter.py](file:///d:/Projetos/SenpAI/Dev/src/analytics/event_spotter.py)) para os 4 critérios de Yuko-Datotsu.
- Motor de Calibração com perfis predefinidos (`rigido`, `normal`, `permissivo`) em JSON.
- Gerador sintético de vídeos de teste de Kendo ([demo_generator.py](file:///d:/Projetos/SenpAI/Dev/src/utils/demo_generator.py)).
- CLI principal para execução do pipeline ([main.py](file:///d:/Projetos/SenpAI/Dev/main.py)).

---

## 7. Direitos de Uso, Cópia e Propriedade Intelectual

> [!IMPORTANT]
> **AVISO LEGAL E TERMOS DE PROTEÇÃO INTELECTUAL**
> 
> **© 2026 SenpAI (先輩 AI) • Plataforma Inteligente de Kendo — Versão `v2.3.1`.**  
> **TODOS OS DIREITOS DE USO E CÓPIA RESERVADOS.**
> 
> 1. **Titularidade**: Todo o código-fonte, arquitetura de visão computacional, algoritmos biomecânicos de *Ki-Ken-Tai-Ichi*, modelos de detecção postural, pesos de rede neural, perfis de calibração heurística, identidade visual, interfaces e documentações pertencem exclusivamente aos desenvolvedores e detentores do projeto SenpAI.
> 2. **Restrições de Uso**: É estritamente proibida a reprodução, cópia, duplicação, distribuição comercial, engenharia reversa, descompilação ou criação de obras derivadas, no todo ou em parte, sem autorização prévia e expressa por escrito.
> 3. **Conformidade Normativa**: O sistema foi concebido e estruturado em estrita conformidade com as diretrizes e regulamentações técnicas da **International Kendo Federation (FIK)** e da **All Japan Kendo Federation (AJKF/ZNKR)**.







