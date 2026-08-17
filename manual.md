# Shinpanai (審判 AI) — Manual Técnico Completo

> **Arquitetura, Implementação, Algoritmos e Log de Mudanças**

---

## 1. Visão Geral do Sistema e Filosofia de Arquitetura

O **Shinpanai (審判 AI)** é uma plataforma avançada de visão computacional, análise biomecânica e arbitragem assistida por inteligência artificial projetada para a arte marcial do **Kendo**.

No Kendo tradicional, a atribuição de um ponto válido (*Yuko-Datotsu*) é regida pelo conceito fundamental de **Ki-Ken-Tai-Ichi** (気剣体一致 — Espírito, Espada e Corpo em harmonia unificada):

- **Ki (気)**: Espírito / Prontidão (*Zanshin*)
- **Ken (剣)**: Espada / Precisão do impacto do Shinai no alvo
- **Tai (体)**: Corpo / Sincronismo do pisar (*Fumikomi-ashi*) e postura corporal

O Shinpanai traduz esses princípios marciais em algoritmos numéricos de alta precisão através da análise cinemática de esqueletos 3D, projeção vetorial da espada (*Shinai*) e aprendizado por reforço (*Reinforcement Learning*).

---

## 2. Estrutura do Projeto e Módulos

A estrutura de arquivos do projeto está organizada de forma modular:

```text
Dev/
├── config/
│   ├── calibration_profiles.json   # Configurações e pesos dos perfis de arbitragem
│   └── settings.json               # Configurações globais do sistema (CPU/GPU)
├── data/
│   ├── feedback_dataset.json       # Base de dados de anotações (TP/FP/FN/Dan) para RL
│   └── training_history.json       # Histórico de sessões de treinamento e revisões por Dan
├── logs/
│   └── shinpanai_debug.log         # Arquivo consolidado de logs, erros e alertas do sistema
├── src/
│   ├── analytics/
│   │   ├── biomechanics.py         # Cálculo numérico dos critérios de Yuko-Datotsu
│   │   └── event_spotter.py        # Detecção temporal de picos cinemáticos e golpes
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
│   │   ├── pose_detector.py        # Rastreamento de esqueleto 3D via MediaPipe
│   │   └── shinai_tracker.py       # Estimação do Kensen e zonas anatômicas de alvo
│   └── pipeline.py                 # Pipeline orquestrador end-to-end de vídeo
├── tests/
│   ├── test_dan_training_governance.py # Testes automatizados da governança por Dan, pacotes e retreinamento
│   ├── test_feedback_loop.py       # Suíte de testes unitários para a malha de feedback
│   ├── test_hardware_settings.py   # Testes automatizados de hardware e configurações
│   ├── test_logger_manager.py      # Testes automatizados do sistema de logs e diagnóstico
│   ├── test_pipeline_cancellation.py # Testes automatizados de cancelamento e interrupção do pipeline
│   └── test_sonkyo_and_plane_filtering.py # Testes de sonkyo e filtragem de planos
├── app.py                          # Dashboard Web Interativo em Streamlit (com Edição por Dan e Configurações)
├── main.py                         # Interface de Linha de Comando (CLI com flag --device)
├── Melhorias_Issues.md             # Registro de pendências e visão de versão final
├── README.TXT                      # Manual simplificado de uso rápido
└── manual.md                       # Manual técnico completo e log de mudanças (este arquivo)
```

---

## 3. Detalhamento das Implementações Técnicas e Algoritmos

### 3.1. Visão Computacional (`src/vision/`)

#### `PoseDetector` ([pose_detector.py](file:///d:/Projetos/Shinpanai/Dev/src/vision/pose_detector.py))
Utiliza o framework **MediaPipe Pose** para rastrear 33 pontos de articulação 3D (*landmarks*) em tempo real por frame. Extrai coordenadas normalizadas $(x, y, z)$ e pontos em pixels $(px, py)$ para pulso, cotovelo, ombro, quadril, joelho, tornozelo, pé, nariz e orelhas.

#### `ShinaiTracker` ([shinai_tracker.py](file:///d:/Projetos/Shinpanai/Dev/src/vision/shinai_tracker.py))
A espada (*Shinai*) é estimada como uma extensão vetorial a partir do eixo formado pelos pulsos (`RIGHT_WRIST` e `LEFT_WRIST`). O algoritmo projeta a trajetória do **Kensen** (ponta da espada) e define as zonas anatômicas de ataque em 3D/2D:

- **MEN**: Região da cabeça (com base no nariz/orelhas).
- **KOTE**: Região dos antebraços/pulsos do oponente.
- **DO**: Flancos abdominais (com base na linha entre ombro e quadril).
- **TSUKI**: Região da garganta/esterno superior.

---

### 3.2. Análise e Biomecânica (`src/analytics/`)

#### `EventSpotter` ([event_spotter.py](file:///d:/Projetos/Shinpanai/Dev/src/analytics/event_spotter.py))
Classificador temporal (*Action Spotter*) que analisa as séries temporais de velocidade e aceleração das mãos e da espada. Identifica:
1. Fase de elevação (*Furikaburi*)
2. Aceleração descendente rápida
3. Instante exato de impacto (pico de desaceleração)

#### `BiomechanicsAnalyzer` ([biomechanics.py](file:///d:/Projetos/Shinpanai/Dev/src/analytics/biomechanics.py))
Calcula quantitativamente os 4 pilares do **Ki-Ken-Tai-Ichi**:

1. **Target Impact (Ken)**: Avalia a proximidade entre a ponta do *Kensen* e o centro da zona anatômica alvo no frame de impacto (Escala: $0\%$ a $100\%$).
2. **Fumikomi Sync (Tai)**: Mede a diferença de tempo (offset em ms) entre a batida do pé direito no solo e o ponto de máxima desaceleração do golpe. Quanto menor o offset em relação à janela ideal ($0\text{ ms}$ a $40\text{ ms}$), maior a pontuação.
3. **Posture (Tai)**: Calcula o alinhamento do vetor da coluna (ombro-quadril) em relação à vertical perfeita. Penaliza inclinações excessivas para a frente/lados e perda de estabilidade da cabeça.
4. **Zanshin (Ki)**: Avalia a janela pós-golpe (15 frames após o impacto). Mede a manutenção da postura firme, estabilidade visual e ausência de desaceleração desordenada ou desequilíbrio.

---

### 3.3. Engine de Calibração ([calibrator.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/calibrator.py) & [calibration_profiles.json](file:///d:/Projetos/Shinpanai/Dev/config/calibration_profiles.json))

O motor calcula a **Pontuação Total Ponderada**:

$$\text{Score}_{\text{Total}} = (w_{\text{target}} \cdot S_{\text{target}}) + (w_{\text{fumikomi}} \cdot S_{\text{fumikomi}}) + (w_{\text{posture}} \cdot S_{\text{posture}}) + (w_{\text{zanshin}} \cdot S_{\text{zanshin}})$$

Para um golpe ser validado como **Yuko-Datotsu** (Ponto Válido / *Ippon*):
1. $\text{Score}_{\text{Total}}$ deve ser maior ou igual a `min_total_score` do perfil ativo.
2. Cada sub-pontuação individual deve satisfazer o respectivo `sub_threshold`.

#### Perfis Pré-configurados ([calibration_profiles.json](file:///d:/Projetos/Shinpanai/Dev/config/calibration_profiles.json))

| Perfil | $\text{min\_total\_score}$ | Pesos ($w_{\text{target}}, w_{\text{fumikomi}}, w_{\text{posture}}, w_{\text{zanshin}}$) | Aplicação Principal |
| :--- | :---: | :--- | :--- |
| **Rígido** | `82%` | Target: 35%, Fumikomi: 25%, Posture: 20%, Zanshin: 20% | Campeonatos / Exames de Dan |
| **Normal** | `65%` | Target: 40%, Fumikomi: 25%, Posture: 20%, Zanshin: 15% | Treinos de Dojang e Arbitragem Geral |
| **Permissivo** | `45%` | Target: 55%, Fumikomi: 20%, Posture: 15%, Zanshin: 10% | Iniciantes / Avaliação Educacional |
| **Custom** | Dinâmico | Definido pelo usuário via sliders no Streamlit | Pesquisa e Ajustes Finos |

---

### 3.4. Aprendizagem por Reforço, Governança por Dan e Gestão de Treinamento ([feedback_manager.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/feedback_manager.py))

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

### 3.5. Relatórios e Pipeline ([reporter.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/reporter.py) & [pipeline.py](file:///d:/Projetos/Shinpanai/Dev/src/pipeline.py))

- **`DiagnosticReporter`** ([reporter.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/reporter.py)): Gera um texto explicativo em Português detalhando por que o golpe foi aprovado ou reprovado, apresentando os milissegundos do Fumikomi e dicas de correção técnica para o praticante.
- **`ShinpanaiPipeline`** ([pipeline.py](file:///d:/Projetos/Shinpanai/Dev/src/pipeline.py)): Orquestra a execução frame-a-frame do vídeo, grava o vídeo anotado com esqueletos e alvos, e retorna o dicionário completo com métricas.

---

## 4. Suíte de Testes Automatizados

O projeto inclui testes automatizados em `unittest` para validar o pipeline cinemático, hardware e governança por Dan.

### Comando para Execução dos Testes

```bash
.\.venv\Scripts\python.exe -m unittest discover tests
```

### Testes Incluídos

- **`test_dan_training_governance.py`**: Valida a salvamento de revisões com Dan, retreinamento do modelo, cálculo das métricas Dan (contador, média e tabela por Dan), exportação/importação de pacotes `.json` com data e Dan, e reset do sistema.
- **`test_feedback_loop.py`**: Valida o salvamento, persistência, cálculo de precisão/recall e algoritmo de aprendizagem por reforço sobre Falsos Positivos.
- **`test_hardware_settings.py`**: Valida detecção de GPU NVIDIA, configurações globais e resolução de fallback transparente para CPU.

---

## 5. Registro de Mudanças e Histórico de Versões (Changelog)

---

### `[v1.5.0]` — 2026-08-15 *(Versão Atual)*

- **Sistema de Diagnóstico, Alertas e Log de Debug do Sistema**:
  - Criado o módulo central de logging e diagnóstico ([logger_manager.py](file:///d:/Projetos/Shinpanai/Dev/src/utils/logger_manager.py)) com retenção em arquivo ([`logs/shinpanai_debug.log`](file:///d:/Projetos/Shinpanai/Dev/logs/shinpanai_debug.log)) e buffer em memória.
  - Registro automático no log de eventos críticos: **reset de treinamento**, **importação de arquivos JSON**, **exportação de pacotes**, **retreinamentos por Dan** e diagnósticos de hardware.
  - Adicionada a **Seção 4: Diagnóstico, Alertas & Log de Debug** no menu de configurações do Web App ([app.py](file:///d:/Projetos/Shinpanai/Dev/app.py)).
  - Métricas em tempo real de contagem de logs, alertas/avisos e erros do sistema.
  - Visualizador de logs com filtro dinâmico por nível (`ERROR`, `WARNING`, `INFO`, `DEBUG`).
  - Botão de **download do arquivo de log completo (`shinpanai_debug.log`)**.
  - Ferramenta de **teste de diagnóstico automatizado** para checagem de integridade de hardware, GPU, arquivos e bibliotecas.
- **Melhorias na Revisão de Golpes (Modo Gravado)**:
  - Exibição de badges visuais em tempo real: **`✅ CONFIRMADO`** (verde) e **`✏️ EDITADO`** (azul) com atualização instantânea na UI via `st.rerun()`.
  - Botão de **Reset Geral da Revisão (`🔄 Resetar Revisão`)** para limpar as marcações da sessão e botões de **Reset Individual (`🔄 Resetar este golpe`)** por card.
- **Suporte Universal a Arquivos de Treinamento JSON**:
  - O módulo de importação ([feedback_manager.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/feedback_manager.py)) foi aprimorado para aceitar pacotes completos, listas diretas de revisões JSON ou entradas avulsas, com tratamento de buffer (`seek(0)`) e atribuição de IDs.
- **Estabilidade de Interface**:
  - Tabela de treinamentos por Dan convertida para Markdown nativo, eliminando erros de pré-carregamento de módulos JS/CSS do navegador (Vite preload helper).
- **Testes Automatizados**: Suíte de testes em [test_logger_manager.py](file:///d:/Projetos/Shinpanai/Dev/tests/test_logger_manager.py) e testes de importação expandidos em [test_dan_training_governance.py](file:///d:/Projetos/Shinpanai/Dev/tests/test_dan_training_governance.py) (19 testes automatizados com 100% de aprovação).

### `[v1.4.6]` — 2026-08-17

- **Aceleração Nativa com GPU NVIDIA CUDA (YOLOv8-Pose)**:
  - **Motor de Inferência GPU de Alta Velocidade**: O módulo [pose_detector.py](file:///d:/Projetos/Shinpanai/Dev/src/vision/pose_detector.py) foi atualizado para utilizar o modelo **YOLOv8-Pose em PyTorch CUDA (`cuda:0`)** sobre a placa NVIDIA GeForce RTX 4050.
  - **Detecção Paralela Multi-Pessoa**: A análise de todos os atletas presentes no enquadramento agora ocorre em um **único passo direto na VRAM da GPU**, eliminando as 3 execuções redundantes por corte que eram feitas na CPU pelo MediaPipe.
  - **Aumento de Desempenho (FPS)**: A velocidade de processamento atinge taxas de **27 a 100+ FPS** dependendo da resolução do vídeo, reduzindo drasticamente o tempo de análise na Arbitragem Gravada.
  - **Seletor de Hardware na Sidebar e Painel de Arbitragem**: Adicionado seletor e badges de diagnóstico em tempo real no [app.py](file:///d:/Projetos/Shinpanai/Dev/app.py), permitindo alternar facilmente entre aceleração GPU NVIDIA CUDA e CPU.
- **Testes Automatizados**: Suíte completa de 34 testes automatizados validada com 100% de aprovação.

---

### `[v1.4.5]` — 2026-08-17

- **Aprimoramento Robusto da Detecção de Sonkyō & Filtragem de Planos**:
  - **Resiliência a Oclusões por Hakama / Kendogi**: O estimador biomecânico ([sonkyo_detector.py](file:///d:/Projetos/Shinpanai/Dev/src/analytics/sonkyo_detector.py)) agora utiliza múltiplos sinais (rebaixamento de quadril, proporção tronco-altura, compressão vertical relativa e inclinação de coluna), operando com precisão mesmo quando joelhos ou tornozelos estão parcialmente oclusos.
  - **Análise Temporal de Altura Relativa**: Cálculo do baseline de altura e nível de quadril em pé do atleta ao longo da gravação, identificando o Sonkyō com base na compressão vertical relativa ($H_{sonkyo} \le 0.75 \times H_{standing}$).
  - **Fechamento Morfológico e Preenchimento de Falhas (Gap Bridging)**: Algoritmo de unificação temporal que preenche quedas momentâneas de rastreamento (dropouts de até 8 frames / ~0.27s), evitando a fragmentação de intervalos contínuos de Sonkyō.
  - **Correção na Filtragem de Planos ([combatant_tracker.py](file:///d:/Projetos/Shinpanai/Dev/src/vision/combatant_tracker.py))**: O classificador de planos não descarta mais combatentes agachados no solo do Shiaijo como segundo plano.
- **Testes Automatizados**: Suíte expandida em [test_sonkyo_and_plane_filtering.py](file:///d:/Projetos/Shinpanai/Dev/tests/test_sonkyo_and_plane_filtering.py) com testes de oclusão por Hakama, gap bridging e calibração de plano (34 testes automatizados com 100% de aprovação).

---

### `[v1.4.4]` — 2026-08-17

- **Correção Crítica de Vazamento de Arquivos Temporários (`[Errno 28] No space left on device`)**:
  - Identificada e corrigida a criação repetitiva de arquivos temporários (`tempfile.NamedTemporaryFile`) a cada ciclo de atualização (`rerun`) do Streamlit no [app.py](file:///d:/Projetos/Shinpanai/Dev/app.py).
  - Implementado sistema de **cache de uploads no `st.session_state`**: o arquivo enviado só é gravado em disco uma única vez por upload (baseado em `name` e `size`).
  - Adicionada rotina de **limpeza automática de arquivos temporários órfãos e antigos** na pasta `shinpanai_uploads`.
  - Liberação de mais de **20 GB** de espaço em disco no diretório temporário do sistema operacional.

---

### `[v1.4.3]` — 2026-08-17

- **Cronômetro em Tempo Real e Persistência do Tempo de Processamento (Arbitragem Gravada)**:
  - Inclusão do **cronômetro dinâmico em tempo real** exibido durante o processamento do vídeo no [app.py](file:///d:/Projetos/Shinpanai/Dev/app.py) (`MM:SS.s` e segundos decorridos).
  - Persistência visual do **tempo final de execução e taxa média de processamento (FPS)** no painel de status fixo e no cartão de resumo de métricas do combate (`summary-card`).
  - Suporte a medição precisa de tempo em [pipeline.py](file:///d:/Projetos/Shinpanai/Dev/src/pipeline.py) via `AnalysisWorker.elapsed_seconds` e retorno de `processing_time_seconds` e `processing_fps`.
- **Resumo Estruturado de Processamento no Log do Sistema**:
  - Registro detalhado (`INFO`) no arquivo consolidado de logs (`shinpanai_debug.log`) contendo arquivo analisado, tempo de execução, FPS, dispositivo utilizado, detecções de Sonkyō, golpes e planos descartados.
- **Testes Automatizados**: Suíte de testes em [test_pipeline_cancellation.py](file:///d:/Projetos/Shinpanai/Dev/tests/test_pipeline_cancellation.py) expandida para verificar cronômetro, persistência e log de resumo (32 testes com 100% de aprovação).

---

### `[v1.4.2]` — 2026-08-17

- **Apresentação de Eventos de Sonkyō na Arbitragem Gravada**:
  - Inclusão dos eventos de **Sonkyō Inicial** (Abertura / Início do Combate) e **Sonkyō Final** (Encerramento / Fechamento do Combate) diretamente na lista de eventos apresentados no container de resultados (`col_results`) do [app.py](file:///d:/Projetos/Shinpanai/Dev/app.py).
  - Exibição de cartões expansíveis detalhados com badge `🥋 SONKYŌ DETECTADO`, intervalo ritual (timestamps), contagem de frames de início e fim, duração em segundos, liberação regulamentar de combate e diagnóstico biomecânico da postura de respeito (*Reigi*).
  - Sequenciamento cronológico completo do combate: **Sonkyō Inicial ➡️ Golpes Identificados na Janela Regulamentar ➡️ Sonkyō Final**.

---

### `[v1.4.1]` — 2026-08-17

- **Botão de Interromper Processamento na Arbitragem Gravada**:
  - Inclusão do botão `⏹️ Interromper Processamento` no painel de execução de vídeo no [app.py](file:///d:/Projetos/Shinpanai/Dev/app.py).
  - Suporte a cancelamento cooperativo no método `process_video` do [pipeline.py](file:///d:/Projetos/Shinpanai/Dev/src/pipeline.py) através do parâmetro `is_cancelled`.
  - Liberação segura de recursos e fechamento de streams (`VideoCapture` e `VideoWriter`) através de blocos `try...finally`.
  - Notificação visual de cancelamento no dashboard (`st.warning`) e limpeza de arquivos parciais gerados.
  - Registro de eventos de interrupção e alertas no sistema de logs (`log_event`).
- **Testes Automatizados**: Criado [test_pipeline_cancellation.py](file:///d:/Projetos/Shinpanai/Dev/tests/test_pipeline_cancellation.py) cobrindo cancelamento imediato, cancelamento durante leitura de frames, validação de logs de aviso e execução normal sem interrupção.

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
- **Testes Automatizados**: Criado [test_dan_training_governance.py](file:///d:/Projetos/Shinpanai/Dev/tests/test_dan_training_governance.py) cobrindo governança, pacotes e retreinamento.

---

### `[v1.3.0]` — 2026-08-12

- **Menu de Configurações Centralizado**: Implementado no [app.py](file:///d:/Projetos/Shinpanai/Dev/app.py) com seletor de acelerador de hardware (CPU Somente vs GPU NVIDIA quando disponível).
- **Módulo de Hardware e Configurações**: Detecção dinâmica multi-nível de GPU NVIDIA e resolução de fallback transparente para CPU ([hardware.py](file:///d:/Projetos/Shinpanai/Dev/src/utils/hardware.py) e [settings_manager.py](file:///d:/Projetos/Shinpanai/Dev/src/utils/settings_manager.py)).
- **Suporte a CLI**: Adicionado parâmetro `--device {cpu,gpu}` no [main.py](file:///d:/Projetos/Shinpanai/Dev/main.py).
- **Atualização de Requisitos**: Inclusão de instruções de instalação de pacotes CUDA (PyTorch CUDA e ONNX Runtime GPU) no [requirements.txt](file:///d:/Projetos/Shinpanai/Dev/requirements.txt) e [README.TXT](file:///d:/Projetos/Shinpanai/Dev/README.TXT).

---

### `[v1.2.1]` — 2026-08-06

> [!NOTE]
> **Melhorias na Interface Web**
> - Reestruturação do Dashboard Web no Streamlit ([app.py](file:///d:/Projetos/Shinpanai/Dev/app.py)) com layout responsivo em duas colunas.
> - **Coluna Fixa (*Sticky Video Column*)**: O vídeo da luta (ou vídeo anotado da IA) fica ancorado à esquerda da página mesmo durante a rolagem.
> - **Coluna de Golpes Rolável**: A lista de golpes identificados, diagnósticos biomecânicos e painel de aprendizado adaptativo possuem barra de rolagem dedicada à direita (`st.container(height=680)`).
> - Alternador direto de exibição no player: Vídeo Anotado com Visão AI vs Vídeo Original.
> - Cartão com resumo de métricas do combate incorporado na coluna do vídeo.

---

### `[v1.2.0]` — 2026-08-06

> [!NOTE]
> **Adicionado**
> - Módulo de Gerenciamento de Feedback e Aprendizagem por Reforço ([feedback_manager.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/feedback_manager.py)).
> - Dataset JSON para armazenamento de feedbacks ([feedback_dataset.json](file:///d:/Projetos/Shinpanai/Dev/data/feedback_dataset.json)).
> - Suporte ao **Modo de Aprendizagem** na CLI ([main.py](file:///d:/Projetos/Shinpanai/Dev/main.py)) através das flags `--mode learning` e `--optimize-profile`.
> - **Painel de Aprendizagem por Reforço** no Web App Streamlit ([app.py](file:///d:/Projetos/Shinpanai/Dev/app.py)), permitindo botões rápidos de anotação de TP (Correto), FP (Falso Positivo) e registro manual de FN (Golpe Perdido).
> - Painel de métricas de acurácia (Precisão %, Recall %, Total de Anotações) na interface Web.
> - Testes automatizados para a malha de feedback em [test_feedback_loop.py](file:///d:/Projetos/Shinpanai/Dev/tests/test_feedback_loop.py).
> - Documentação reestruturada em formato Markdown ([manual.md](file:///d:/Projetos/Shinpanai/Dev/manual.md)) e manual simplificado ([README.TXT](file:///d:/Projetos/Shinpanai/Dev/README.TXT)).

---

### `[v1.1.0]` — 2026-08-01

- **Dashboard Web Interativo** desenvolvido em Streamlit ([app.py](file:///d:/Projetos/Shinpanai/Dev/app.py)) com estilização CSS customizada.
- Suporte ao perfil `custom` com sliders dinâmicos para ajuste manual de limiares e pesos de Ki-Ken-Tai-Ichi.
- Módulo gerador de relatórios textuais diagnósticos em Português ([reporter.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/reporter.py)).
- Exportação de vídeos anotados com suporte a visualização de esqueleto 3D e pontos de impacto.

---

### `[v1.0.0]` — 2026-07-25

- **Lançamento inicial** da arquitetura base do Shinpanai.
- Módulos de Visão Computacional ([pose_detector.py](file:///d:/Projetos/Shinpanai/Dev/src/vision/pose_detector.py), [shinai_tracker.py](file:///d:/Projetos/Shinpanai/Dev/src/vision/shinai_tracker.py)) baseados em MediaPipe Pose.
- Módulos de Análise Biomecânica ([biomechanics.py](file:///d:/Projetos/Shinpanai/Dev/src/analytics/biomechanics.py), [event_spotter.py](file:///d:/Projetos/Shinpanai/Dev/src/analytics/event_spotter.py)) para os 4 critérios de Yuko-Datotsu.
- Motor de Calibração com perfis predefinidos (`rigido`, `normal`, `permissivo`) em JSON.
- Gerador sintético de vídeos de teste de Kendo ([demo_generator.py](file:///d:/Projetos/Shinpanai/Dev/src/utils/demo_generator.py)).
- CLI principal para execução do pipeline ([main.py](file:///d:/Projetos/Shinpanai/Dev/main.py)).






