# SenpAI (先輩 AI) ⚔️

> **Sistema Avançado de Avaliação e Análise Biomecânica de Kendo com Inteligência Artificial**  
> **Versão Oficial**: `v 0.4.2.0` • **Python**: `3.11` • **Licença**: Todos os Direitos Reservados

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8-Pose](https://img.shields.io/badge/YOLOv8-Pose-00FFFF.svg)](https://ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-00C7B7.svg)](https://developers.google.com/mediapipe)
[![Tests 490 Passing](https://img.shields.io/badge/Tests-490%20Passing-success.svg)](file:///d:/Projetos/SenpAI/Dev/tests)

---

## 🥋 Visão Geral e Filosofia Marcial

O **SenpAI (先輩 AI)** é uma plataforma pioneira que une visão computacional de alta performance, cinemática 3D e aprendizado por reforço para avaliar combates e treinos de **Kendo** em estrita conformidade com as diretrizes da **International Kendo Federation (FIK)** e **All Japan Kendo Federation (AJKF/ZNKR)**.

O sistema analisa o conceito regulamentar de **Yūko-Datotsu** (ponto válido) através dos pilares do **Ki-Ken-Tai-Ichi** (気剣体一致):

- **Ki (気 - Espírito)**: Prontidão e intenção contínua (*Zanshin*);
- **Ken (剣 - Espada)**: Alvo atingido (*Datotsu-bui*), fio da lâmina (*Hasuji*) e lâmina válida (*Monouchi*);
- **Tai (体 - Corpo)**: Sincronismo do impacto do pé (*Fumikomi-ashi*), postura da coluna e distância combativa (*Maai*).

---

## 🚀 Principais Módulos do Sistema

### 1. ⚔️ Modo de Análise de Lutas (Shiai)
- **Rastreador Dedicado de Combate**: Acompanhamento focado nos 2 atletas (Aka e Shiro), identificação automática por fitas dorsais (*Tasuki*) e inversão manual interativa.
- **Placar Eletrônico Sanbon-shobu**: Registro de Ippons oficiais, Nihon-me e Hantei.
- **Detecção de Sonkyō e Validação Espacial**: Delimitação automática do Shiai-jo com matriz homográfica perspectiva.

### 2. 🥋 Modo de Treinamento Pedagógico & Multi-Kenshi
- **Rastreador Multi-Kenshi**: Acompanhamento paralelo de múltiplos praticantes simultâneos no Dojo.
- **Re-identificação Multimodal (KenshiReID)**: Reconhece atletas pelo Tare frontal, Tenugui posterior e proporções anatômicas mesmo após saírem e retornarem à cena.
- **Métricas Individuais de Rendimento**: Cadência real (CPM por tempo de presença efetivo), estamina e consistência biomecânica nas 14 modalidades oficiais.

### 3. 🏛️ Governança de Treinamento & Conhecimento Acumulado
- **Estrutura Modular de Governança**: Gestão de espaço em disco, distribuição de revisões por graduação Dan e gerenciamento unificado do dataset (apagar, exportar e restaurar via `.zip` com validação de hash SHA-256).
- **Modalidades de Treinamento & Acurácia**: Tabela desacoplada com métrica de acurácia de detecção/classificação calibrada por modalidade.
- **Corpus Técnico de Referência**: Separação transparente entre Manuais Canônicos de fábrica (FIK, AJKF e Estudo Biomecânico) e fontes mineradas na web.
- **Gabarito de Lutas & Ciclos de Treino**: Anotações oficiais integradas na governança e histórico contínuo de ciclos de treinamento.
- **"O que o SenpAI já aprendeu"**: Painel do conhecimento acumulado (princípios universais ZNKR/FIK, perfis biomecânicos de tolerância e cadência, e fontes catalogadas).

### 4. 🌐 Coleta Contínua na Internet & Mineração de Diretrizes
- **Varredura Periódica Autônoma**: Busca e catalogação contínua de vídeos no YouTube (via yt-dlp) e diretrizes técnicas na web (Wikipedia, DuckDuckGo).
- **Atualização Reativa em Tempo Real**: Progresso, tempo decorrido, percentual de avanço e estimativa de término atualizados a cada 3 segundos via fragment do Streamlit (`@st.fragment`), sem necessidade de atualização manual.
- **Download Inteligente & Filtros de Disco**: Aplicação estrita de limites de armazenamento e verificação de licenças/duração.

### 5. ⚖️ Curadoria Ativa (Active Learning) & Banco Único de Rótulos
- **Banco Único de Rótulos (`LabelBank`)**: Consolidação estruturada de rótulos humanos, fontes web, gabarito e detecções automáticas.
- **Fila de Incerteza Arbitral (45% a 65%)**: Amostras controversas isoladas com reprodução visual do corte real do golpe para decisão de Senseis.
- **Detector de Bandeiras de Árbitros (`FlagDetector`)**: Extração de decisões arbitrais em vídeo para rotulagem automática de pontos e dúvidas.
- **Salvaguarda do Golden Benchmark**: Bloqueio de qualquer recalibração que cause regressão no conjunto de ouro.

### 6. 🔐 Configurações Globais & Variáveis do Sistema
- **Aba "Variáveis do Sistema"**: Gestão isolada de chaves de API (Google Gemini, OpenAI, SerpApi) em `config/api_keys.local.json` (fora do controle de versão e dos pacotes de exportação).
- **Resiliência Multiplataforma**: Camada de I/O atômica e thread-safe com locks reentrantes e retries exponenciais, eliminando travamentos de acesso negado no Windows.

---

## 💻 Requisitos e Instalação

### Requisitos:
- **Python**: `3.11` (Distribuição oficial recomendada).
- **Ambiente Virtual**: `.venv` isolado.
- **Sistemas Operacionais**: Windows 10/11, Linux (Ubuntu/Debian) ou macOS.
- **Hardware**: CPU multi-core (Intel i5/i7/i9 ou AMD Ryzen). Opcional: GPU NVIDIA com suporte a CUDA para aceleração FP16 no YOLOv8-Pose.

### Passo a Passo de Instalação:

1. **Clonar o Repositório**:
   ```bash
   git clone https://github.com/eduardozim/SenpAI.git
   cd SenpAI/Dev
   ```

2. **Criar e Ativar o Ambiente Virtual**:
   - **Windows**:
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     python3.11 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar Dependências**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🎬 Como Executar

### Interface Gráfica Interativa (Streamlit - Recomendado):
```bash
python -m streamlit run app.py
```
O navegador abrirá automaticamente em `http://localhost:8501`.

### Linha de Comando (CLI):
```bash
python main.py --video caminho/da/luta.mp4 --profile normal --output anotado.mp4
```

### Executar Testes Automatizados:
```bash
python run_tests.py
# ou: pytest tests/
```

---

## 🌟 Novidades da Versão `v 0.4.2.0` — Atualização do Método de Treinamento

- **Validade dos golpes pelas revisões**: a decisão dos shinpan só aponta golpes a revisar; quem valida é a revisão de kenshi e senseis, pesada pelo dan. O detector de bandeiras foi removido.
- **Referência dos Golpes Validados**: distribuição real das notas dos pilares por técnica no lugar do antigo padrão de números fixos, com retorno por pilar na análise.
- **Pré-processamento das lutas da internet**: os golpes em dúvida das lutas baixadas vão para a Fila de Curadoria, e as respostas viram treino.
- **Fila de Curadoria na Governança de Treinamento**: transferida de Perfis de Calibração para logo após o Ciclo de Treino; o Eixo 4 passa a se chamar Golden Benchmark.
- **Correção do golpe na curadoria**: seleção "Qual golpe foi?" pré-marcada com o golpe identificado; se a tentativa foi de outro golpe, o árbitro corrige e decide Ippon ou Golpe Inválido, e a correção também ensina o classificador de técnica.

## 🌟 Novidades da Versão `v 0.4.1.0`

- **Reestruturação Completa da Governança de Treinamento**:
  - Painel de Revisão por Dan, métricas de graduação e gestão de espaço em disco consolidados.
  - Tabela desacoplada de Modalidades de Treinamento com métrica de acurácia individual calibrada.
  - Separação estrita entre Manuais Canônicos de fábrica e fontes web mineradas dinamicamente.
  - Integração do Gabarito de Lutas diretamente na Governança de Treinamento.
  - Renomeação oficial para "Ciclo de Treino" e inclusão do painel *"O que o SenpAI já aprendeu"* com visão geral e por modalidade.
- **Coleta Contínua na Internet com Atualização Automática (Fragment 3s)**:
  - Busca e download programado de vídeos e diretrizes na web com barra de progresso, tempo decorrido e término previsto.
  - Atualização automática da interface a cada 3 segundos via `@st.fragment`, sem intervenção manual.
  - Silenciamento inteligente de avisos repetitivos de runtime JavaScript do `yt-dlp`.
- **Aba Dedicada "Variáveis do Sistema"**:
  - Nova aba no menu de configurações para cadastro de chaves de API do Google Gemini, OpenAI e SerpApi com isolamento em `config/api_keys.local.json`.
- **Resiliência de I/O e Escrita Atômica no Windows (`file_utils.py`)**:
  - Eliminação definitiva de erros de colisão de arquivos temporários (`[WinError 5] Acesso negado`) através de `safe_replace` e `safe_write_json` com retries exponenciais, locks de thread e fallback de cópia direta.
- **Banco Único de Rótulos & Detecção de Bandeiras**:
  - Implementação do `LabelBank` unificando anotações humanas, mineradas e automáticas.
  - Detecção computadorizada de bandeiras de árbitros (`FlagDetector`) para rotulagem automática de lutas.
- **Expansão da Suíte de Testes para 490 Testes Automatizados**:
  - Auditoria completa com 490 testes em 50 módulos cobrindo todo o ecossistema com 100% de aprovação.

---

### Versão Anterior: `v 0.4.0.0`
- **Limpeza Automática & Otimização de Armazenamento**: Gerenciamento proativo do diretório de uploads e arquivos temporários (`data/tmp/senpai_uploads`), liberando espaço automaticamente na inicialização e pós-processamento, além de controle manual por botão de 1 clique nas configurações.
- **Compatibilidade Web Universal de Vídeo HTML5**: Transcodificação otimizada para contêiner MP4 (H.264 / AAC / YUV420p com flag `+faststart`) aplicada de forma transparente para vídeos anotados e originais, prevenindo erros de decodificação (`Video source error`) em navegadores.
- **Auditoria Dinâmica de Testes**: Contador dinâmico em tempo de execução para a suíte de testes.
- **Mensagens Humanizadas de Streaming**: Feedback claro e informativo para streams ao vivo e links remotos.

---

### Versão Anterior: `v 0.3.8.0`
- **Identificação de Múltiplas Sessões & Agrupamento por Bloco de Modalidade**: Reconhecimento inteligente de diferentes sessões de treino dentro de uma mesma gravação contínua.
- **Nitidez Aprimorada nas Etiquetas & Renderização HD**: Etiquetas TrueType de alto contraste, anti-colisão vertical e codificação de alta definição FFmpeg.
- **Re-renderização Dinâmica do Vídeo Anotado**: Diálogo interativo de renomeação de atletas sobre o vídeo com cache de poses sem re-inferência.

Para especificações detalhadas, formulação matemática e guia completo dos regulamentos FIK, consulte o arquivo [manual.md](file:///d:/Projetos/SenpAI/Dev/manual.md).

---

## 📜 Propriedade Intelectual e Licença

**© 2026 SenpAI (先輩 AI) • Todos os Direitos Reservados.**  
Este software, sua arquitetura, algoritmos biomecânicos e modelos neurais são propriedade intelectual protegida por leis de direitos autorais. É proibida a reprodução, engenharia reversa ou comercialização não autorizada.
