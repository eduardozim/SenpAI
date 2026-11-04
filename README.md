# SenpAI (先輩 AI) ⚔️

> **Sistema Avançado de Avaliação e Análise Biomecânica de Kendo com Inteligência Artificial**  
> **Versão Oficial**: `v 0.4.0.0` • **Python**: `3.11` • **Licença**: Todos os Direitos Reservados

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![YOLOv8-Pose](https://img.shields.io/badge/YOLOv8-Pose-00FFFF.svg)](https://ultralytics.com/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-00C7B7.svg)](https://developers.google.com/mediapipe)
[![Tests](https://img.shields.io/badge/Tests-Passing-success.svg)](file:///d:/Projetos/SenpAI/Dev/tests)

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
- **Métricas Individuais de Rendimento**: Cadência real (CPM por tempo de presença efetivo), estamina e consistência biomecânica.

### 3. ⚖️ Curadoria Ativa (Active Learning) & Governança por Dan
- **Fila de Incerteza Arbitral (45% a 65%)**: Amostras controversas são isoladas para decisão de árbitros e Senseis.
- **Reprodução Visual com Corte Real do Golpe**: Exibição do clipe exato em torno do impacto da luta.
- **Ciclo Completo com 4 Opções**:
  - `✅ Ippon (Válido)`: Confirma o ponto e recalibra a sensibilidade.
  - `❌ Golpe Inválido`: Homologa tentativa sem pontuação.
  - `🚫 Não tem Golpe`: Remove detecção espúria.
  - `🗑️ Descartar`: Descarte gracioso de lances ruidosos sem penalizar o modelo adaptativo.
- **Salvaguarda do Golden Benchmark**: Bloqueio de qualquer recalibração que cause regressão no conjunto de ouro.
- **Pacotes ZIP de Governança**: Exportação e importação consolidada de dados de treino e clipes de vídeo.

### 4. 📹 Fontes de Vídeo Flexíveis
- **Upload Local**: Suporte a arquivos `.mp4`, `.mov` e `.avi`.
- **Streaming Web**: Download e análise com cache local de transmissões do YouTube.
- **Tempo Real**: Câmeras USB via OpenCV local ou WebRTC no navegador para servidores web/cloud.

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
pytest tests/
```

---

## 🌟 Novidades da Versão `v 0.4.0.0`

- **Limpeza Automática & Otimização de Armazenamento**: Gerenciamento proativo do diretório de uploads e arquivos temporários (`data/tmp/senpai_uploads`), liberando espaço automaticamente na inicialização e pós-processamento, além de controle manual por botão de 1 clique nas configurações.
- **Compatibilidade Web Universal de Vídeo HTML5**: Transcodificação otimizada para contêiner MP4 (H.264 / AAC / YUV420p com flag `+faststart`) aplicada de forma transparente para vídeos anotados e originais, prevenindo erros de decodificação (`Video source error`) em navegadores.
- **Auditoria Contínua & Execução Dinâmica de Testes**: Contador dinâmico em tempo de execução para a suíte de 317 testes automatizados com taxa de 100% de aprovação.
- **Mensagens Humanizadas de Streaming & Conexão**: Feedback claro e informativo para streams ao vivo e links remotos, distinguindo canais offline de quedas de conexão.

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
