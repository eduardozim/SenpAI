# ShinpanAI (審判 AI) — Manual Técnico Completo

> **Arquitetura, Implementação, Algoritmos e Log de Mudanças**

---

## 1. Visão Geral do Sistema e Filosofia de Arquitetura

O **ShinpanAI (審判 AI)** é uma plataforma avançada de visão computacional, análise biomecânica e arbitragem assistida por inteligência artificial projetada para a arte marcial do **Kendo**.

No Kendo tradicional, a atribuição de um ponto válido (*Yuko-Datotsu*) é regida pelo conceito fundamental de **Ki-Ken-Tai-Ichi** (気剣体一致 — Espírito, Espada e Corpo em harmonia unificada):

- **Ki (気)**: Espírito / Prontidão (*Zanshin*)
- **Ken (剣)**: Espada / Precisão do impacto do Shinai no alvo
- **Tai (体)**: Corpo / Sincronismo do pisar (*Fumikomi-ashi*) e postura corporal

O ShinpanAI traduz esses princípios marciais em algoritmos numéricos de alta precisão através da análise cinemática de esqueletos 3D, projeção vetorial da espada (*Shinai*) e aprendizado por reforço (*Reinforcement Learning*).

---

## 2. Estrutura do Projeto e Módulos

A estrutura de arquivos do projeto está organizada de forma modular:

```text
Dev/
├── config/
│   ├── calibration_profiles.json   # Configurações e pesos dos perfis de arbitragem
│   ├── settings.json               # Configurações globais do sistema (CPU/GPU)
│   └── sonkyo_learned_profile.json # Perfil adaptativo aprendido de postura de Sonkyō
├── data/
│   ├── feedback_dataset.json       # Base de dados de anotações (TP/FP/FN/Dan) para RL
│   └── training_history.json       # Histórico de sessões de treinamento e revisões por Dan
├── logs/
│   └── shinpanai_debug.log         # Arquivo consolidado de logs, erros e alertas do sistema
├── src/
│   ├── analytics/
│   │   ├── biomechanics.py         # Cálculo numérico dos critérios de Yuko-Datotsu
│   │   ├── event_spotter.py        # Detecção temporal de picos cinemáticos e golpes
│   │   └── sonkyo_detector.py      # Identificação de Sonkyō, delimitação da luta e aprendizado
│   ├── engine/
│   │   ├── calibrator.py           # Motor de pontuação e validação de limiares
│   │   ├── feedback_manager.py     # Motor de Aprendizagem por Reforço, Governança por Dan e Otimização
│   │   └── reporter.py             # Gerador de relatórios diagnósticos textuais
│   ├── utils/
│   │   ├── demo_generator.py       # Gerador sintético de vídeos de teste de Kendo
│   │   ├── hardware.py             # Detecção de GPU NVIDIA e resolução de fallback CPU
│   │   ├── logger_manager.py       # Gerenciador central de logs, alertas e diagnósticos de debug
│   │   └── settings_manager.py     # Gerenciamento e persistência das configurações do sistema
│   ├── vision/
│   │   ├── combatant_tracker.py    # Rastreamento dos 2 Kenshi (Aka/Shiro), flag dorsal e planos
│   │   ├── pose_detector.py        # Rastreamento de esqueleto 3D via YOLOv8-Pose / MediaPipe
│   │   └── shinai_tracker.py       # Estimação do Kensen e zonas anatômicas de alvo
│   └── pipeline.py                 # Pipeline orquestrador end-to-end de vídeo e renderização
├── tests/
│   ├── test_dan_training_governance.py # Testes da governança por Dan, pacotes e retreinamento
│   ├── test_feedback_loop.py       # Testes unitários para a malha de feedback e RL
│   ├── test_hardware_settings.py   # Testes automatizados de hardware e configurações
│   ├── test_logger_manager.py      # Testes automatizados do sistema de logs e diagnóstico
│   ├── test_pipeline_cancellation.py # Testes automatizados de cancelamento e interrupção do pipeline
│   ├── test_scoreboard_and_flag_detection.py # Testes do placar oficial e detecção de flag dorsal
│   └── test_sonkyo_and_plane_filtering.py # Testes de Sonkyō, limites da luta e filtragem de planos
├── app.py                          # Dashboard Web Interativo em Streamlit (com HUD, Placar e Configurações)
├── main.py                         # Interface de Linha de Comando (CLI com flags completas)
├── Melhorias_Issues.md             # Registro de pendências, issues e histórico de versões
├── README.TXT                      # Manual simplificado de uso rápido
└── manual.md                       # Manual técnico completo e log de mudanças (este arquivo)
```

---

## 3. Detalhamento das Implementações Técnicas e Algoritmos

### 3.1. Visão Computacional (`src/vision/`)

#### `PoseDetector` ([pose_detector.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/pose_detector.py))
Utiliza o framework **MediaPipe Pose** para rastrear 33 pontos de articulação 3D (*landmarks*) em tempo real por frame. Extrai coordenadas normalizadas $(x, y, z)$ e pontos em pixels $(px, py)$ para pulso, cotovelo, ombro, quadril, joelho, tornozelo, pé, nariz e orelhas.

#### `ShinaiTracker` ([shinai_tracker.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/shinai_tracker.py))
A espada (*Shinai*) é estimada como uma extensão vetorial a partir do eixo formado pelos pulsos (`RIGHT_WRIST` e `LEFT_WRIST`). O algoritmo projeta a trajetória do **Kensen** (ponta da espada) e define as zonas anatômicas de ataque em 3D/2D:

- **MEN**: Região da cabeça (com base no nariz/orelhas).
- **KOTE**: Região dos antebraços/pulsos do oponente.
- **DO**: Flancos abdominais (com base na linha entre ombro e quadril).
- **TSUKI**: Região da garganta/esterno superior.

#### `CombatantTracker` ([combatant_tracker.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/combatant_tracker.py))
Responsável pela persistência e identificação contínua dos dois lutadores principais no Shiaijo:
- **Detecção Cromática de Flag Dorsal (Tasukuki)**: Segmentação em espaço de cor HSV (`detect_red_flag_score`) no dorso dos atletas para identificação inequívoca de **Kenshi Aka (Vermelho)** e **Kenshi Shiro (Branco)**, mesmo com keikogi azul escuro, branco ou preto.
- **Filtragem Geométrica de Plano de Combate**: Calibra a escala espacial média dos kenshi e descarta automaticamente pessoas e movimentações em segundo plano (outras lutas, arquibancadas) ou oclusões em primeiro plano (transeuntes passando em frente à câmera).

---

### 3.2. Análise, Biomecânica e Rituais (`src/analytics/`)

#### `EventSpotter` ([event_spotter.py](file:///d:/Projetos/ShinpanAI/Dev/src/analytics/event_spotter.py))
Classificador temporal (*Action Spotter*) que analisa as séries temporais de velocidade e aceleração das mãos e da espada. Identifica:
1. Fase de elevação (*Furikaburi*)
2. Aceleração descendente rápida
3. Instante exato de impacto (pico de desaceleração)

#### `BiomechanicsAnalyzer` ([biomechanics.py](file:///d:/Projetos/ShinpanAI/Dev/src/analytics/biomechanics.py))
Calcula quantitativamente os 4 pilares do **Ki-Ken-Tai-Ichi**:

1. **Target Impact (Ken)**: Avalia a proximidade entre a ponta do *Kensen* e o centro da zona anatômica alvo no frame de impacto (Escala: $0\%$ a $100\%$).
2. **Fumikomi Sync (Tai)**: Mede a diferença de tempo (offset em ms) entre a batida do pé direito no solo e o ponto de máxima desaceleração do golpe. Quanto menor o offset em relação à janela ideal ($0\text{ ms}$ a $40\text{ ms}$), maior a pontuação.
3. **Posture (Tai)**: Calcula o alinhamento do vetor da coluna (ombro-quadril) em relação à vertical perfeita. Penaliza inclinações excessivas para a frente/lados e perda de estabilidade da cabeça.
4. **Zanshin (Ki)**: Avalia a janela pós-golpe (15 frames após o impacto). Mede a manutenção da postura firme, estabilidade visual e ausência de desaceleração desordenada ou desequilíbrio.

#### `SonkyoDetector` ([sonkyo_detector.py](file:///d:/Projetos/ShinpanAI/Dev/src/analytics/sonkyo_detector.py))
Módulo biomecânico que monitora e reconhece o ritual sagrado de **Sonkyō** (agachamento sobre os calcanhares com coluna vertical):
- **Classificação Postural Multifatorial**: Avalia rebaixamento de quadril ($\Delta Y$), proporção tronco-altura, compressão vertical relativa ($H_{sonkyo} \le 0.75 \times H_{standing}$) e verticalidade da coluna.
- **Delimitação Regulamentar da Luta**: Marca o início oficial do combate (`match_start_frame`) no término do Sonkyō Inicial e o encerramento oficial (`match_end_frame`) no início do Sonkyō Final.
- **Filtragem Estrita de Golpes**: Qualquer golpe fora desse intervalo ritual é sumariamente descartado da arbitragem oficial.
- **Aprendizado Biomecânico Adaptativo**: Permite edição interativa de intervalos na UI e recalibra os limiares de Sonkyō, persistindo o aprendizado em `config/sonkyo_learned_profile.json`.

---

### 3.3. Engine de Calibração ([calibrator.py](file:///d:/Projetos/ShinpanAI/Dev/src/engine/calibrator.py) & [calibration_profiles.json](file:///d:/Projetos/ShinpanAI/Dev/config/calibration_profiles.json))

O motor calcula a **Pontuação Total Ponderada**:

$$\text{Score}_{\text{Total}} = (w_{\text{target}} \cdot S_{\text{target}}) + (w_{\text{fumikomi}} \cdot S_{\text{fumikomi}}) + (w_{\text{posture}} \cdot S_{\text{posture}}) + (w_{\text{zanshin}} \cdot S_{\text{zanshin}})$$

Para um golpe ser validado como **Yuko-Datotsu** (Ponto Válido / *Ippon*):
1. $\text{Score}_{\text{Total}}$ deve ser maior ou igual a `min_total_score` do perfil ativo.
2. Cada sub-pontuação individual deve satisfazer o respectivo `sub_threshold`.

#### Perfis Pré-configurados ([calibration_profiles.json](file:///d:/Projetos/ShinpanAI/Dev/config/calibration_profiles.json))

| Perfil | $\text{min\_total\_score}$ | Pesos ($w_{\text{target}}, w_{\text{fumikomi}}, w_{\text{posture}}, w_{\text{zanshin}}$) | Aplicação Principal |
| :--- | :---: | :--- | :--- |
| **Rígido** | `82%` | Target: 35%, Fumikomi: 25%, Posture: 20%, Zanshin: 20% | Campeonatos / Exames de Dan |
| **Normal** | `65%` | Target: 40%, Fumikomi: 25%, Posture: 20%, Zanshin: 15% | Treinos de Dojang e Arbitragem Geral |
| **Permissivo** | `45%` | Target: 55%, Fumikomi: 20%, Posture: 15%, Zanshin: 10% | Iniciantes / Avaliação Educacional |
| **Custom** | Dinâmico | Definido pelo usuário via sliders no Streamlit | Pesquisa e Ajustes Finos |

---

### 3.4. Aprendizagem por Reforço, Governança por Dan e Gestão de Treinamento ([feedback_manager.py](file:///d:/Projetos/ShinpanAI/Dev/src/engine/feedback_manager.py))

Gerencia o ciclo completo de auditoria, revisão por Dan e otimização adaptativa dos modelos:

- **Seleção de Dan do Revisor**: Mapeia revisores de **1º Dan (Shodan)** a **8º Dan (Hachidan)**, associando `reviewer_dan`, `reviewer_dan_name` e `review_date` (timestamp ISO) a cada revisão.
- **Edição e Regra de Auditabilidade (Sem Exclusão)**:
  - Permite **confirmar** marcações, **editar** técnica/timestamp/resultado e **incluir** golpes perdidos (falsos negativos).
  - A exclusão de marcações é **desabilitada por norma de auditabilidade**, preservando a integridade do conjunto de dados.
- **Histórico de Treinamentos (`data/training_history.json`)**: Registra cada sessão de retreinamento executada, incluindo o Dan do aplicador, a contagem de itens revisados e o resumo das alterações de calibração.
- **Métricas de Governança (`get_training_metrics()`)**:
  - Contador total de treinamentos realizados.
  - Nível médio (Dan) dos treinamentos (ex: `4.0º Dan (Yondan)`).
  - Tabela de distribuição da quantidade de treinamentos e percentual por Dan (1º a 8º Dan).
- **Pacotes de Treinamento (Exportação e Importação)**:
  - `export_training_package()`: Exporta um arquivo `.json` contendo todas as marcações com o Dan do revisor e as datas dos treinamentos realizados.
  - `import_training_package()`: Importa arquivos `.json` previamente baixados, mesclando dados e recalibrando o modelo automaticamente.
  - `reset_all_training_data()`: Apaga os dados de treinamento e restaura o sistema ao estágio inicial.

---

### 3.5. Relatórios e Pipeline ([reporter.py](file:///d:/Projetos/ShinpanAI/Dev/src/engine/reporter.py) & [pipeline.py](file:///d:/Projetos/ShinpanAI/Dev/src/pipeline.py))

- **`DiagnosticReporter`** ([reporter.py](file:///d:/Projetos/ShinpanAI/Dev/src/engine/reporter.py)): Gera um texto explicativo em Português detalhando por que o golpe foi aprovado ou reprovado, apresentando os milissegundos do Fumikomi e dicas de correção técnica para o praticante.
- **`ShinpanAIPipeline`** ([pipeline.py](file:///d:/Projetos/ShinpanAI/Dev/src/pipeline.py)): Orquestra a execução frame-a-frame do vídeo, grava o vídeo anotado com esqueletos e alvos, e retorna o dicionário completo com métricas.

---

## 4. Suíte de Testes Automatizados e Relatório de Execução

O projeto inclui suíte completa de testes automatizados em `unittest` com runner customizado ([test_runner.py](file:///d:/Projetos/ShinpanAI/Dev/src/utils/test_runner.py)) e script de execução dedicado ([run_tests.py](file:///d:/Projetos/ShinpanAI/Dev/run_tests.py)).

### Execução dos Testes via CLI e Interface

```bash
# Execução completa com exibição detalhada e geração de log descritivo:
.\.venv\Scripts\python.exe run_tests.py

# Ou via unittest padrão:
.\.venv\Scripts\python.exe -m unittest discover tests
```

Também é possível disparar os testes diretamente no **Web Dashboard** acessando a aba **⚙️ Configurações > Seção 5 (Diagnóstico e Logs)** através do botão **`🔬 Rodar Testes (44)`** e baixar o relatório completo em **`📥 Baixar Log Testes (.log)`**.

### Relatório Descritivo e Política de Retenção de Logs

- **Relatório Detalhado ([`logs/shinpanai_test_report.log`](file:///d:/Projetos/ShinpanAI/Dev/logs/shinpanai_test_report.log))**:
  - Cada teste executado é documentado com: **Módulo**, **Classe**, **Método**, **Descrição Detalhada do Teste / Docstring**, **Status (PASS/FAIL/ERROR)**, **Duração em Segundos** e eventuais rastros de erro/falha.
  - Cabeçalho com data/hora, versão do sistema, plataforma operacional e hardware.
  - Resumo estatístico final (total, aprovados, falhas, erros, taxa de sucesso % e tempo total).
- **Política de Retenção Única**:
  - A pasta `logs/` mantém **estritamente apenas o último log de testes executado**, sobrescrevendo ou limpando relatórios anteriores automaticamente a cada nova execução.

### Módulos de Testes Incluídos (44 Testes)

- **`test_dan_training_governance.py`**: Valida salvamento de revisões com Dan, retreinamento do modelo, cálculo das métricas Dan (contador, média e tabela por Dan), exportação/importação de pacotes `.json` com data e Dan, e reset do sistema.
- **`test_feedback_loop.py`**: Valida salvamento, persistência, cálculo de precisão/recall e algoritmo de aprendizagem por reforço sobre Falsos Positivos.
- **`test_hardware_settings.py`**: Valida detecção de GPU NVIDIA, configurações globais e resolução de fallback transparente para CPU.
- **`test_logger_manager.py`**: Valida sistema de logs, métricas em tempo real e diagnósticos automatizados.
- **`test_pipeline_cancellation.py`**: Valida cancelamento cooperativo, liberação de recursos de streaming e cronômetro em tempo real.
- **`test_scoreboard_and_flag_detection.py`**: Valida o placar eletrônico Sanbon-shobu, detecção cromática de flag dorsal (Tasukuki) e inversão Aka ⇄ Shiro.
- **`test_sonkyo_and_plane_filtering.py`**: Valida a classificação postural de Sonkyō, delimitação temporal da luta, filtragem de planos (fundo/transeuntes) e persistência de aprendizado de Sonkyō.

Total de **44 testes automatizados** executados e aprovados com 100% de sucesso.

---

## 5. Registro de Mudanças e Histórico de Versões (Changelog)

---

### `[v1.6.0]` — 2026-08-18 *(Versão Atual)*

- **Relatório Descritivo de Testes Automatizados & Retenção Única de Log**:
  - Criado o runner customizado ([test_runner.py](file:///d:/Projetos/ShinpanAI/Dev/src/utils/test_runner.py)) e script de execução na raiz ([run_tests.py](file:///d:/Projetos/ShinpanAI/Dev/run_tests.py)).
  - Geração automática de relatório descritivo por teste com módulo, classe, método, descrição em Português, status individual, duração em segundos e sumário executivo.
  - Salvo na pasta `logs/` ([`logs/shinpanai_test_report.log`](file:///d:/Projetos/ShinpanAI/Dev/logs/shinpanai_test_report.log)) com política estrita de retenção: **apenas o último log de testes é mantido na pasta**.
  - Botões de execução rápida (`🔬 Rodar Testes (44)`) e download do relatório (`📥 Baixar Log Testes (.log)`) integrados na **Seção 5 de Diagnóstico e Logs** do Web App.
- **Detecção e Scoring Consolidado (Modo de Arbitragem Gravada)**:
  - Validação completa de *Yuko-Datotsu* com score ponderado (*Ki-Ken-Tai-Ichi*: impacto no alvo, sincronismo de *Fumikomi*, postura e *Zanshin*), corte automático de clipes de eventos e relatórios diagnósticos de combate.
- **Navegação Interativa no Vídeo com Salto Temporal Calibrado (-1.0s)**:
  - Salto temporal instantâneo no player de vídeo ao clicar nos botões individuais de evento (Sonkyō Inicial, Golpes Detectados ou Sonkyō Final) ou ao selecionar eventos no menu dropdown.
  - Calibração de **1 segundo de pré-roll (`-1.0s`)** antes do início do evento para permitir que o árbitro assista à preparação, execução e finalização da ação com clareza.
  - Banner dinâmico com indicação da posição ativa (`🎯 Posicionado em X.Xs`) e botão de reset rápido (`✖️ Início`).
- **Otimização da Escala Visual da Interface (Zoom 80%)**:
  - Aplicação de redução global de 20% na escala de fontes e elementos (`zoom: 0.8`) com compactação ergonômica de paddings e containers (`max-width: 96%`), eliminando necessidade de rolagem excessiva.
- **Detecção de Sonkyō & Delimitação Temporal da Luta**:
  - Identificação e verificação automática da postura ritualística de *Sonkyō* (agachamento profundo sobre os calcanhares, flexão de joelhos e coluna ereta) para marcação do Início Oficial (`match_start_frame`) e Encerramento Oficial (`match_end_frame`) da luta no Modo de Arbitragem Gravada.
  - Filtragem estrita de golpes por Sonkyō: consideração e pontuação de *Yuko-Datotsu* realizada **estritamente entre os momentos de Sonkyō de início e término**, descartando movimentações e cortes fora da janela regulamentar de combate.
  - Edição interativa de Sonkyō com aprendizado biomecânico adaptativo contínuo persistido em `config/sonkyo_learned_profile.json`.
- **Rastreamento dos 2 Kenshi Principais e Filtragem de Planos**:
  - Rastreamento contínuo dos dois atletas principais que iniciaram o combate no Shiaijo (`Kenshi Aka - Vermelho` e `Kenshi Shiro - Branco`).
  - Calibração geométrica automática de plano principal, descartando elementos de segundo plano (outras lutas ao fundo, árbitros distantes, arquibancadas) e oclusões de primeiro plano (pessoas passando na frente da câmera).
- **Placar Oficial de Arbitragem (Sanbon-shobu Scoreboard) e Inversão Manual Aka ⇄ Shiro**:
  - Placar eletrônico no topo dos resultados com contagem de Ippon para Aka e Shiro, técnicas pontuadas e declaração automática de resultado (*Sanbon-shobu*).
  - Detecção cromática HSV de flag dorsal (Tasukuki) e botão de ação rápida `🔄 Inverter Lutadores (Aka ⇄ Shiro)` para reatribuição imediata de pontuação, eventos e relatórios em gravações com câmera no lado oposto do Shiaijo.
- **Aceleração GPU NVIDIA CUDA com Tensor Cores FP16 & Streaming de Renderização**:
  - Suporte a GPU NVIDIA CUDA via YOLOv8-Pose em FP16 meia precisão (`half=True`) com fallback automático para CPU.
  - Streaming direto de renderização em 2ª passada no pipeline de gravação de vídeo anotado, reduzindo o consumo de memória RAM de 15+ GB para menos de 100 MB.
- **Suíte de Testes Automatizados**:
  - 44 testes automatizados em `unittest` com 100% de aprovação cobrindo todo o pipeline cinemático, Sonkyō, planos, placar, flag dorsal, hardware, governança por Dan e logs.

---

### `[v1.5.0]` — 2026-08-15

- **Sistema de Diagnóstico, Alertas e Log de Debug do Sistema**:
  - Criado o módulo central de logging e diagnóstico ([logger_manager.py](file:///d:/Projetos/ShinpanAI/Dev/src/utils/logger_manager.py)) com retenção em arquivo ([`logs/shinpanai_debug.log`](file:///d:/Projetos/ShinpanAI/Dev/logs/shinpanai_debug.log)) e buffer em memória.
  - Registro automático no log de eventos críticos: **reset de treinamento**, **importação de arquivos JSON**, **exportação de pacotes**, **retreinamentos por Dan** e diagnósticos de hardware.
  - Adicionada a **Seção 4: Diagnóstico, Alertas & Log de Debug** no menu de configurações do Web App ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py)).
  - Métricas em tempo real de contagem de logs, alertas/avisos e erros do sistema.
  - Visualizador de logs com filtro dinâmico por nível (`ERROR`, `WARNING`, `INFO`, `DEBUG`).
  - Botão de **download do arquivo de log completo (`shinpanai_debug.log`)**.
  - Ferramenta de **teste de diagnóstico automatizado** para checagem de integridade de hardware, GPU, arquivos e bibliotecas.
- **Melhorias na Revisão de Golpes (Modo Gravado)**:
  - Exibição de badges visuais em tempo real: **`✅ CONFIRMADO`** (verde) e **`✏️ EDITADO`** (azul) com atualização instantânea na UI via `st.rerun()`.
  - Botão de **Reset Geral da Revisão (`🔄 Resetar Revisão`)** para limpar as marcações da sessão e botões de **Reset Individual (`🔄 Resetar este golpe`)** por card.
- **Suporte Universal a Arquivos de Treinamento JSON**:
  - O módulo de importação ([feedback_manager.py](file:///d:/Projetos/ShinpanAI/Dev/src/engine/feedback_manager.py)) foi aprimorado para aceitar pacotes completos, listas diretas de revisões JSON ou entradas avulsas, com tratamento de buffer (`seek(0)`) e atribuição de IDs.
- **Estabilidade de Interface**:
  - Tabela de treinamentos por Dan convertida para Markdown nativo, eliminando erros de pré-carregamento de módulos JS/CSS do navegador (Vite preload helper).
- **Testes Automatizados**: Suíte de testes em [test_logger_manager.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_logger_manager.py) e testes de importação expandidos em [test_dan_training_governance.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_dan_training_governance.py) (19 testes automatizados com 100% de aprovação).

### `[v1.4.12]` — 2026-08-17

- **Otimização de Espaço e Remoção de Texto de Diagnóstico do Sonkyō**:
  - **Layout Compacto de Sonkyō ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py))**:
    - Removidos os blocos de texto verbosos e diagnósticos descritivos laterais de Sonkyō Inicial e Final.
    - Estruturação compacta e colapsável dos cards de Sonkyō (`expanded=False` por padrão, exceto durante edição ativa).
    - Exibição direta das informações operacionais essenciais (intervalo ritualístico, início/término oficial do combate e badge de status), economizando espaço vertical para os eventos de combate e golpes de Yuko-Datotsu.
- **Testes Automatizados**: Suíte de 44 testes executada com 100% de sucesso.

---

### `[v1.4.11]` — 2026-08-17

- **Placar Oficial de Arbitragem, Detecção de Flag Dorsal e Inversão Aka/Shiro**:
  - **Placar Oficial de Arbitragem (Sanbon-shobu Scoreboard) ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py) e [pipeline.py](file:///d:/Projetos/ShinpanAI/Dev/src/pipeline.py))**:
    - Exibição de painel visual eletrônico no topo dos resultados com contagem de **Ippon** válidos para Aka (Vermelho) e Shiro (Branco), badges com as técnicas pontuadas e declaração automática do resultado regulamentar (*Vitória de Aka*, *Vitória de Shiro* ou *Empate / Hikiwake*).
  - **Detecção da Cor da Flag (Tasukuki / Faixa Vermelha nas Costas) ([combatant_tracker.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/combatant_tracker.py))**:
    - Implementada a segmentação cromática HSV no dorso/tronco (`detect_red_flag_score`) para identificar a fita vermelha dorsal do Kenshi Aka, independente da cor do Keikogi (azul escuro, branco, preto).
    - Permite a correta identificação dos lados mesmo quando a câmera de gravação estiver posicionada do lado oposto do Shiaijo (câmera invertida).
  - **Inversão Interativa Aka ⇄ Shiro**:
    - Adicionado botão de ação rápida `🔄 Inverter Lutadores (Aka ⇄ Shiro)` para troca instantânea de pontuação, eventos e diagnósticos em caso de ângulo de filmagem desfavorável.
- **Testes Automatizados**: Suíte de testes em [test_scoreboard_and_flag_detection.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_scoreboard_and_flag_detection.py) (44 testes automatizados com 100% de aprovação).

---

### `[v1.4.10]` — 2026-08-17

- **Correção de AttributeError & Otimização de Performance e Memória**:
  - **Correção de `AttributeError` em Edição de Sonkyō ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py))**: Corrigida a verificação condicional em `initial_edit` e `final_edit` quando são `None`, garantindo que os timestamps padrão sejam lidos sem exceções de runtime.
  - **Eliminação de Sobrecarga de Memória RAM ([pipeline.py](file:///d:/Projetos/ShinpanAI/Dev/src/pipeline.py))**:
    - Removido o armazenamento em buffer de todos os quadros descompactados (`raw_frames`) na memória RAM durante a passagem 1.
    - A renderização do vídeo anotado agora utiliza streaming direto em 2ª passada (`cap_render`), reduzindo o consumo de RAM de 15+ GB para menos de 100 MB em vídeos longos/alta resolução.
  - **Aceleração GPU com Tensor Cores FP16 ([pose_detector.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/pose_detector.py))**:
    - Ativada a inferência em meia precisão (`half=True`) com dimensão padrão `imgsz=640` no YOLOv8-Pose em CUDA, aumentando substancialmente o throughput de frames por segundo (FPS).
- **Testes Automatizados**: Suíte de 39 testes executada com 100% de sucesso.

---

### `[v1.4.9]` — 2026-08-17

- **Inclusão Automática de Sonkyō no Início e Fim da Gravação**:
  - **Garantia de Delimitação Ritual**: Quando a análise de visão computacional não detecta com alta confiança os rituais de Sonkyō nos primeiros ou últimos segundos da gravação, o sistema ([sonkyo_detector.py](file:///d:/Projetos/ShinpanAI/Dev/src/analytics/sonkyo_detector.py)) **atribui automaticamente os movimentos de Sonkyō no início (00:00.000) e no encerramento do vídeo**.
  - **Identificação Visual Transparente**: No painel de Arbitragem Gravada ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py)), os cards exibem a badge correspondente (`🥋 SONKYŌ DETECTADO` para detecção automática por pose ou `📌 SONKYŌ (Início/Fim do Vídeo / Ajustável)` para fallback padrão).
  - **Edição e Reprocessamento Imediatos**: O árbitro tem a garantia de que ambos os rituais estarão sempre visíveis e expansíveis, podendo editar os intervalos com exatidão e reprocessar o combate com aprendizado contínuo.
- **Testes Automatizados**: Suíte de testes expandida em [test_sonkyo_and_plane_filtering.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_sonkyo_and_plane_filtering.py) com validação de inclusão de rituais padrão (39 testes automatizados com 100% de aprovação).

---

### `[v1.4.8]` — 2026-08-17

- **Edição Interativa de Sonkyō, Reprocessamento e Aprendizado Contínuo**:
  - **Edição de Momentos de Sonkyō**: No Modo de Arbitragem Gravada ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py)), o árbitro agora pode editar com precisão os tempos de início e fim tanto do Sonkyō Inicial quanto do Sonkyō Final (ou definir intervalos manuais caso não tenham sido detectados automaticamente).
  - **Botão de Reprocessamento com Aprendizado**: Ao alterar um dos momentos de Sonkyō, a interface habilita o botão de ação rápida `🔄 Reprocessar Arbitragem com Aprendizado de Sonkyō`.
  - **Aprendizado Biomecânico Contínuo ([sonkyo_detector.py](file:///d:/Projetos/ShinpanAI/Dev/src/analytics/sonkyo_detector.py))**:
    - As posturas e proporções corporais no intervalo editado são extraídas dinamicamente para calibrar a sensibilidade do detector de Sonkyō.
    - O perfil adaptado é persistido em `config/sonkyo_learned_profile.json`, sendo aplicado imediatamente neste reprocessamento e em **todas as futuras análises de vídeo**.
  - **Painel de Estatísticas de Sonkyō no Modo Treinamento**: Exibição da quantidade de amostras aprendidas, compressão de altura adaptada, rebaixamento de quadril ($\Delta Y$) e botão para restauração aos padrões de fábrica.
- **Testes Automatizados**: Suíte de testes expandida em [test_sonkyo_and_plane_filtering.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_sonkyo_and_plane_filtering.py) cobrindo conversão de timestamps, persistência de aprendizado e reprocessamento com overrides (38 testes automatizados com 100% de aprovação).

---

### `[v1.4.7]` — 2026-08-17

- **Refinamento do Indicador de Aceleração de Hardware**:
  - **Sidebar Exclusiva para Status Visual**: A barra lateral ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py)) agora exibe apenas o **card de indicação em tempo real** do estado do acelerador (`🚀 Aceleração Ativada` com nome da GPU NVIDIA e framework CUDA ou `💻 Aceleração Desativada` em CPU), mantendo o layout limpo e intuitivo.
  - **Centralização da Seleção de Dispositivo**: A alteração e o salvamento do dispositivo (CPU / GPU) ficam centralizados na seção de **Configurações Globais** do Modo de Treinamento.
- **Testes Automatizados**: Suíte de 34 testes validada com 100% de sucesso.

---

### `[v1.4.6]` — 2026-08-17

- **Aceleração Nativa com GPU NVIDIA CUDA (YOLOv8-Pose)**:
  - **Motor de Inferência GPU de Alta Velocidade**: O módulo [pose_detector.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/pose_detector.py) foi atualizado para utilizar o modelo **YOLOv8-Pose em PyTorch CUDA (`cuda:0`)** sobre a placa NVIDIA GeForce RTX 4050.
  - **Detecção Paralela Multi-Pessoa**: A análise de todos os atletas presentes no enquadramento agora ocorre em um **único passo direto na VRAM da GPU**, eliminando as 3 execuções redundantes por corte que eram feitas na CPU pelo MediaPipe.
  - **Aumento de Desempenho (FPS)**: A velocidade de processamento atinge taxas de **27 a 100+ FPS** dependendo da resolução do vídeo, reduzindo drasticamente o tempo de análise na Arbitragem Gravada.
  - **Seletor de Hardware na Sidebar e Painel de Arbitragem**: Adicionado seletor e badges de diagnóstico em tempo real no [app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py), permitindo alternar facilmente entre aceleração GPU NVIDIA CUDA e CPU.
- **Testes Automatizados**: Suíte completa de 34 testes automatizados validada com 100% de aprovação.

---

### `[v1.4.5]` — 2026-08-17

- **Aprimoramento Robusto da Detecção de Sonkyō & Filtragem de Planos**:
  - **Resiliência a Oclusões por Hakama / Kendogi**: O estimador biomecânico ([sonkyo_detector.py](file:///d:/Projetos/ShinpanAI/Dev/src/analytics/sonkyo_detector.py)) agora utiliza múltiplos sinais (rebaixamento de quadril, proporção tronco-altura, compressão vertical relativa e inclinação de coluna), operando com precisão mesmo quando joelhos ou tornozelos estão parcialmente oclusos.
  - **Análise Temporal de Altura Relativa**: Cálculo do baseline de altura e nível de quadril em pé do atleta ao longo da gravação, identificando o Sonkyō com base na compressão vertical relativa ($H_{sonkyo} \le 0.75 \times H_{standing}$).
  - **Fechamento Morfológico e Preenchimento de Falhas (Gap Bridging)**: Algoritmo de unificação temporal que preenche quedas momentâneas de rastreamento (dropouts de até 8 frames / ~0.27s), evitando a fragmentação de intervalos contínuos de Sonkyō.
  - **Correção na Filtragem de Planos ([combatant_tracker.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/combatant_tracker.py))**: O classificador de planos não descarta mais combatentes agachados no solo do Shiaijo como segundo plano.
- **Testes Automatizados**: Suíte expandida em [test_sonkyo_and_plane_filtering.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_sonkyo_and_plane_filtering.py) com testes de oclusão por Hakama, gap bridging e calibração de plano (34 testes automatizados com 100% de aprovação).

---

### `[v1.4.4]` — 2026-08-17

- **Correção Crítica de Vazamento de Arquivos Temporários (`[Errno 28] No space left on device`)**:
  - Identificada e corrigida a criação repetitiva de arquivos temporários (`tempfile.NamedTemporaryFile`) a cada ciclo de atualização (`rerun`) do Streamlit no [app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py).
  - Implementado sistema de **cache de uploads no `st.session_state`**: o arquivo enviado só é gravado em disco uma única vez por upload (baseado em `name` e `size`).
  - Adicionada rotina de **limpeza automática de arquivos temporários órfãos e antigos** na pasta `shinpanai_uploads`.
  - Liberação de mais de **20 GB** de espaço em disco no diretório temporário do sistema operacional.

---

### `[v1.4.3]` — 2026-08-17

- **Cronômetro em Tempo Real e Persistência do Tempo de Processamento (Arbitragem Gravada)**:
  - Inclusão do **cronômetro dinâmico em tempo real** exibido durante o processamento do vídeo no [app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py) (`MM:SS.s` e segundos decorridos).
  - Persistência visual do **tempo final de execução e taxa média de processamento (FPS)** no painel de status fixo e no cartão de resumo de métricas do combate (`summary-card`).
  - Suporte a medição precisa de tempo em [pipeline.py](file:///d:/Projetos/ShinpanAI/Dev/src/pipeline.py) via `AnalysisWorker.elapsed_seconds` e retorno de `processing_time_seconds` e `processing_fps`.
- **Resumo Estruturado de Processamento no Log do Sistema**:
  - Registro detalhado (`INFO`) no arquivo consolidado de logs (`shinpanai_debug.log`) contendo arquivo analisado, tempo de execução, FPS, dispositivo utilizado, detecções de Sonkyō, golpes e planos descartados.
- **Testes Automatizados**: Suíte de testes em [test_pipeline_cancellation.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_pipeline_cancellation.py) expandida para verificar cronômetro, persistência e log de resumo (32 testes com 100% de aprovação).

---

### `[v1.4.2]` — 2026-08-17

- **Apresentação de Eventos de Sonkyō na Arbitragem Gravada**:
  - Inclusão dos eventos de **Sonkyō Inicial** (Abertura / Início do Combate) e **Sonkyō Final** (Encerramento / Fechamento do Combate) diretamente na lista de eventos apresentados no container de resultados (`col_results`) do [app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py).
  - Exibição de cartões expansíveis detalhados com badge `🥋 SONKYŌ DETECTADO`, intervalo ritual (timestamps), contagem de frames de início e fim, duração em segundos, liberação regulamentar de combate e diagnóstico biomecânico da postura de respeito (*Reigi*).
  - Sequenciamento cronológico completo do combate: **Sonkyō Inicial ➡️ Golpes Identificados na Janela Regulamentar ➡️ Sonkyō Final**.

---

### `[v1.4.1]` — 2026-08-17

- **Botão de Interromper Processamento na Arbitragem Gravada**:
  - Inclusão do botão `⏹️ Interromper Processamento` no painel de execução de vídeo no [app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py).
  - Suporte a cancelamento cooperativo no método `process_video` do [pipeline.py](file:///d:/Projetos/ShinpanAI/Dev/src/pipeline.py) através do parâmetro `is_cancelled`.
  - Liberação segura de recursos e fechamento de streams (`VideoCapture` e `VideoWriter`) através de blocos `try...finally`.
  - Notificação visual de cancelamento no dashboard (`st.warning`) e limpeza de arquivos parciais gerados.
  - Registro de eventos de interrupção e alertas no sistema de logs (`log_event`).
- **Testes Automatizados**: Criado [test_pipeline_cancellation.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_pipeline_cancellation.py) cobrindo cancelamento imediato, cancelamento durante leitura de frames, validação de logs de aviso e execução normal sem interrupção.

---

### `[v1.4.0]` — 2026-08-15

- **Modo de Arbitragem Gravada - Edição de Golpes por Dan**:
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
- **Testes Automatizados**: Criado [test_dan_training_governance.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_dan_training_governance.py) cobrindo governança, pacotes e retreinamento.

---

### `[v1.3.0]` — 2026-08-12

- **Menu de Configurações Centralizado**: Implementado no [app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py) com seletor de acelerador de hardware (CPU Somente vs GPU NVIDIA quando disponível).
- **Módulo de Hardware e Configurações**: Detecção dinâmica multi-nível de GPU NVIDIA e resolução de fallback transparente para CPU ([hardware.py](file:///d:/Projetos/ShinpanAI/Dev/src/utils/hardware.py) e [settings_manager.py](file:///d:/Projetos/ShinpanAI/Dev/src/utils/settings_manager.py)).
- **Suporte a CLI**: Adicionado parâmetro `--device {cpu,gpu}` no [main.py](file:///d:/Projetos/ShinpanAI/Dev/main.py).
- **Atualização de Requisitos**: Inclusão de instruções de instalação de pacotes CUDA (PyTorch CUDA e ONNX Runtime GPU) no [requirements.txt](file:///d:/Projetos/ShinpanAI/Dev/requirements.txt) e [README.TXT](file:///d:/Projetos/ShinpanAI/Dev/README.TXT).

---

### `[v1.2.1]` — 2026-08-06

> [!NOTE]
> **Melhorias na Interface Web**
> - Reestruturação do Dashboard Web no Streamlit ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py)) com layout responsivo em duas colunas.
> - **Coluna Fixa (*Sticky Video Column*)**: O vídeo da luta (ou vídeo anotado da IA) fica ancorado à esquerda da página mesmo durante a rolagem.
> - **Coluna de Golpes Rolável**: A lista de golpes identificados, diagnósticos biomecânicos e painel de aprendizado adaptativo possuem barra de rolagem dedicada à direita (`st.container(height=680)`).
> - Alternador direto de exibição no player: Vídeo Anotado com Visão AI vs Vídeo Original.
> - Cartão com resumo de métricas do combate incorporado na coluna do vídeo.

---

### `[v1.2.0]` — 2026-08-06

> [!NOTE]
> **Adicionado**
> - Módulo de Gerenciamento de Feedback e Aprendizagem por Reforço ([feedback_manager.py](file:///d:/Projetos/ShinpanAI/Dev/src/engine/feedback_manager.py)).
> - Dataset JSON para armazenamento de feedbacks ([feedback_dataset.json](file:///d:/Projetos/ShinpanAI/Dev/data/feedback_dataset.json)).
> - Suporte ao **Modo de Aprendizagem** na CLI ([main.py](file:///d:/Projetos/ShinpanAI/Dev/main.py)) através das flags `--mode learning` e `--optimize-profile`.
> - **Painel de Aprendizagem por Reforço** no Web App Streamlit ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py)), permitindo botões rápidos de anotação de TP (Correto), FP (Falso Positivo) e registro manual de FN (Golpe Perdido).
> - Painel de métricas de acurácia (Precisão %, Recall %, Total de Anotações) na interface Web.
> - Testes automatizados para a malha de feedback em [test_feedback_loop.py](file:///d:/Projetos/ShinpanAI/Dev/tests/test_feedback_loop.py).
> - Documentação reestruturada em formato Markdown ([manual.md](file:///d:/Projetos/ShinpanAI/Dev/manual.md)) e manual simplificado ([README.TXT](file:///d:/Projetos/ShinpanAI/Dev/README.TXT)).

---

### `[v1.1.0]` — 2026-08-01

- **Dashboard Web Interativo** desenvolvido em Streamlit ([app.py](file:///d:/Projetos/ShinpanAI/Dev/app.py)) com estilização CSS customizada.
- Suporte ao perfil `custom` com sliders dinâmicos para ajuste manual de limiares e pesos de Ki-Ken-Tai-Ichi.
- Módulo gerador de relatórios textuais diagnósticos em Português ([reporter.py](file:///d:/Projetos/ShinpanAI/Dev/src/engine/reporter.py)).
- Exportação de vídeos anotados com suporte a visualização de esqueleto 3D e pontos de impacto.

---

### `[v1.0.0]` — 2026-07-25

- **Lançamento inicial** da arquitetura base do ShinpanAI.
- Módulos de Visão Computacional ([pose_detector.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/pose_detector.py), [shinai_tracker.py](file:///d:/Projetos/ShinpanAI/Dev/src/vision/shinai_tracker.py)) baseados em MediaPipe Pose.
- Módulos de Análise Biomecânica ([biomechanics.py](file:///d:/Projetos/ShinpanAI/Dev/src/analytics/biomechanics.py), [event_spotter.py](file:///d:/Projetos/ShinpanAI/Dev/src/analytics/event_spotter.py)) para os 4 critérios de Yuko-Datotsu.
- Motor de Calibração com perfis predefinidos (`rigido`, `normal`, `permissivo`) em JSON.
- Gerador sintético de vídeos de teste de Kendo ([demo_generator.py](file:///d:/Projetos/ShinpanAI/Dev/src/utils/demo_generator.py)).
- CLI principal para execução do pipeline ([main.py](file:///d:/Projetos/ShinpanAI/Dev/main.py)).






