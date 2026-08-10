# Shinpanai (審判 AI) — Manual Técnico Completo

> **Arquitetura, Implementação, Algoritmos e Log de Mudanças**

---

## 1. Visão Geral do Sistema e Filosofia de Arquitetura

O **Shinpanai (審判 AI)** é uma plataforma avançada de visão computacional, análise biomecânica e arbitragem assistida por inteligência artificial projetada para a arte marcial do **Kendo**.

No Kendo tradicional, a atribuição de um ponto válido (*Yuko-Datotsu*) é regida pelo conceito fundamental de **Ki-Ken-Tai-Ichi** (気剣体一致 — Espírito, Espada e Corpo em harmonia unificada):

- **Ki (気)**: Espírito / Prontidão (*Zanshin*)
- **Ken (剣)**: Espada / Precisão do impacto do Shinai no alvo
- **Tai (体)**: Corpo / Sincronismo do pisar (*Fumikomi-ashi*) e postura corporal

O Shinpanai traduz esses princípios marciais milenares em algoritmos numéricos de alta precisão através da análise cinemática de esqueletos 3D, projeção vetorial da espada (*Shinai*) e aprendizado por reforço (*Reinforcement Learning*).

---

## 2. Estrutura do Projeto e Módulos

A estrutura de arquivos do projeto está organizada de forma modular:

```text
Dev/
├── config/
│   └── calibration_profiles.json   # Configurações e pesos dos perfis de arbitragem
├── data/
│   └── feedback_dataset.json       # Base de dados de anotações (TP/FP/FN) para RL
├── src/
│   ├── analytics/
│   │   ├── biomechanics.py         # Cálculo numérico dos critérios de Yuko-Datotsu
│   │   └── event_spotter.py        # Detecção temporal de picos cinemáticos e golpes
│   ├── engine/
│   │   ├── calibrator.py           # Motor de pontuação e validação de limiares
│   │   ├── feedback_manager.py     # Motor de Aprendizagem por Reforço e Otimização
│   │   └── reporter.py             # Gerador de relatórios diagnósticos textuais
│   ├── utils/
│   │   └── demo_generator.py       # Gerador sintético de vídeos de teste de Kendo
│   ├── vision/
│   │   ├── pose_detector.py        # Rastreamento de esqueleto 3D via MediaPipe
│   │   └── shinai_tracker.py       # Estimação do Kensen e zonas anatômicas de alvo
│   └── pipeline.py                 # Pipeline orquestrador end-to-end de vídeo
├── tests/
│   └── test_feedback_loop.py       # Suíte de testes unitários para a malha de feedback
├── app.py                          # Dashboard Web Interativo em Streamlit
├── main.py                         # Interface de Linha de Comando (CLI)
├── Melhorias&Issues.txt            # Registro de pendências e melhorias futuras
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

### 3.4. Aprendizagem por Reforço e Feedback ([feedback_manager.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/feedback_manager.py))

Gerencia o ciclo de vida de feedback do usuário para otimização adaptativa:

- **Persistência**: Grava anotações no dataset JSON ([feedback_dataset.json](file:///d:/Projetos/Shinpanai/Dev/data/feedback_dataset.json)) contendo: `label` (`TP`, `FP`, `FN`), `profile_key`, `sub_scores`, `total_score`, `strike_type` e `timestamp`.
- **Método `get_stats()`**: Calcula precisão (*Precision*) e revocação (*Recall*) percentuais.
- **Método `optimize_profile_config()`**: Algoritmo de Aprendizagem por Reforço / Calibração Adaptativa:
  - **Se houver Falsos Positivos (FP)**: Analisa o Score Total máximo dos FPs e eleva o `min_total_score` de forma segura (até o teto de 90%). Além disso, analisa as médias dos sub-critérios de TPs vs FPs e reforça os `sub_thresholds` das métricas onde os FPs mais falham.
  - **Se houver Falsos Negativos (FN) sem FPs**: Suaviza a pontuação mínima global (redução de 4%) para capturar golpes válidos perdidos.

---

### 3.5. Relatórios e Pipeline ([reporter.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/reporter.py) & [pipeline.py](file:///d:/Projetos/Shinpanai/Dev/src/pipeline.py))

- **`DiagnosticReporter`** ([reporter.py](file:///d:/Projetos/Shinpanai/Dev/src/engine/reporter.py)): Gera um texto explicativo em Português detalhando por que o golpe foi aprovado ou reprovado, apresentando os milissegundos do Fumikomi e dicas de correção técnica para o praticante.
- **`ShinpanaiPipeline`** ([pipeline.py](file:///d:/Projetos/Shinpanai/Dev/src/pipeline.py)): Orquestra a execução frame-a-frame do vídeo, grava o vídeo anotado com esqueletos e alvos, e retorna o dicionário completo com métricas.

---

## 4. Suíte de Testes Automatizados

O projeto inclui testes automatizados em `unittest` para validar a integridade do ciclo de feedback e aprendizado adaptativo no arquivo [test_feedback_loop.py](file:///d:/Projetos/Shinpanai/Dev/tests/test_feedback_loop.py).

### Comando para Execução dos Testes

```bash
.\.venv\Scripts\python.exe -m unittest tests/test_feedback_loop.py
```

### Testes Incluídos

- `test_save_and_load_feedback`: Valida o salvamento, persistência e leitura das anotações no dataset.
- `test_optimize_profile_on_false_positives`: Garante que o motor de RL ajusta os limiares para eliminar Falsos Positivos detectados.

---

## 5. Registro de Mudanças e Histórico de Versões (Changelog)

### `[v1.2.1]` — 2026-08-06 *(Versão Atual)*

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

---

## 6. Roadmap e Pendências Conhecidas (Melhorias & Issues)

Com base no planejamento do projeto ([Melhorias&Issues.md](file:///d:/Projetos/Shinpanai/Dev/Melhorias&Issues.md)), a versão final será composta por **3 Nodos Principais de Operação**:

1. **Modo de Treinamento**: Dicas de melhorias técnicas, exercícios, avaliação de exame de graduação (Kyu/Dan) e revisão por reforço com seleção de graduação (Dan) do revisor.
2. **Modo de Arbitragem Gravada (Modo Atual)**: Análise biomecânica, arbitragem assistida em vídeos gravados e revisão por reforço (RL) com seleção de graduação (Dan) do revisor para calibração dinâmica.
3. **Modo de Detecção em Tempo Real**: Arbitragem e sinalização ao vivo via webcam/câmeras de transmissão com suporte a RTCP.

### 🚀 Outras Melhorias Futuras

1. **Menu de Configurações Centralizado**: Painel dedicado para centralizar preferências do sistema, calibrações, seleção do modo de processamento (**CPU exclusivo** vs **GPU quando presente**) e streams de câmera.
2. **Identificação Individualizada dos Kenshi**: Distinção autônoma entre Aka / Shiro na luta.
3. **Expandir Testes Automatizados**: Auto-testes da aplicação.

### ✅ Funcionalidades Concluídas

1. **Modo de Arbitragem Gravada**: Implementação base e consolidação do modo atual.
2. **Layout Responsivo com Vídeo Fixo (Sticky) e Relatório Rolável**: Implementado na v1.2.1 em [app.py](file:///d:/Projetos/Shinpanai/Dev/app.py).

### 🐛 Issues & Performance

1. **Aceleração via GPU**: Suporte a GPU (CUDA / TensorRT / PyTorch) no MediaPipe Pose para elevar a taxa de FPS durante o processamento de vídeos em alta resolução.
2. **Marcações em Vídeo**: Ajustes na renderização do vídeo com marcações anatômicas.

---



