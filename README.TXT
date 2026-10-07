===============================================================================

                        SENPAI (先輩 AI) - GUIA RÁPIDO

          Sistema de Avaliação e Análise Biomecânica de Kendo com IA

                              Versão: v 0.3.6.0

===============================================================================



O SenpAI é um sistema de visão computacional e análise biomecânica projetado

para avaliar golpes em lutas de Kendo de acordo com o conceito de Yuko-Datotsu

e o princípio do Ki-Ken-Tai-Ichi (Espírito, Espada e Corpo em um só).



-------------------------------------------------------------------------------

1. REQUISITOS E INSTALAÇÃO

-------------------------------------------------------------------------------



Requisitos de Sistema:

- Python 3.11 (Recomendado: distribuição oficial da Python Software Foundation)

- Ambiente Virtual (.venv)

- Windows 10/11, Linux ou macOS

- Opcional para Modo GPU: Placa aceleradora NVIDIA (GeForce RTX/GTX) com drivers CUDA.



Dependências principais:

- opencv-python

- ultralytics (YOLOv8-Pose com aceleração GPU CUDA FP16)

- mediapipe

- numpy / scipy

- fastdtw

- streamlit (para a interface web)



Guia de Instalação no Windows (Passo a Passo Oficial):



1. Instalar o Python 3.11 Oficial (assinado digitalmente pela PSF via Winget):

   winget install Python.Python.3.11



   (ou baixe o instalador oficial de 64 bits em https://www.python.org/downloads/)



2. Criar o ambiente virtual (.venv):

   py -3.11 -m venv .venv



3. Ativar o ambiente virtual:

   - No PowerShell:

     .\.venv\Scripts\activate

   - No Prompt de Comando (CMD):

     .\.venv\Scripts\activate.bat



4. Instalar as dependências do projeto:

   pip install -r requirements.txt



5. (Opcional) Para habilitação de aceleração CUDA em GPU NVIDIA:

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

   pip install ultralytics onnxruntime-gpu



Nota de Compatibilidade Windows (Smart App Control / WDAC):

O uso do Python 3.11 oficial assinado pela PSF garante total conformidade

com as políticas de segurança do Windows (evitando bloqueios como os error 4551

/ ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION comuns em executáveis não assinados).



-------------------------------------------------------------------------------

2. INSTRUÇÕES DE USO (COMO EXECUTAR)

-------------------------------------------------------------------------------



--- A. INTERFACE WEB INTERATIVA (STREAMLIT - RECOMENDADO) ---

Para iniciar o dashboard web com o ambiente virtual (.venv) ativo:



   python -m streamlit run app.py

   (ou: .\.venv\Scripts\python.exe -m streamlit run app.py)



Nota: Usar "python -m streamlit" executa o Streamlit diretamente pelo

interpretador oficial assinado do Python, evitando bloqueios do Smart App Control

do Windows sobre o wrapper executável streamlit.exe.



Recursos da Interface Web:

- Página Inicial de Boas-Vindas (Home): Ao iniciar o sistema, uma tela inicial dedicada apresenta o SenpAI, os padrões oficiais FIK/AJKF, o status de hardware ativo e um demonstrativo passo a passo de como iniciar o uso em 4 etapas. A página é visível exclusivamente quando nem a página de Análise de Lutas nem a página de Menu de Configurações estiverem selecionadas.

- Barra de Atalhos Rápidos na Home: Botões diretos para iniciar a Análise de Lutas, acessar as Configurações ou baixar a documentação oficial (manual.md).

- Calibração Realista de Acurácia (< 50% de fábrica): O modelo inicia em uma faixa preliminar honesta (32% a 46%), eliminando falsos positivos e evoluindo progressivamente conforme manuais da FIK são processados e árbitros graduados registram correções.

- Supressão de Falsos Positivos & Múltiplos Disparos: Elevação do cooldown de golpes para 35 frames (~1.2s), Non-Maximum Suppression (NMS) temporal e validação geométrica de Maai (contato entre combatentes), eliminando repetições e placares irreais (ex: 66 a 30 em lutas 1 a 1).

- Persistência Incremental no Auto-Treinamento por IA: Consolidação contínua do conhecimento no início e término de cada treino, retomando sempre do último checkpoint e preservando todo o aprendizado prévio.

- Interface otimizada e dimensionada (escala visual 90%) para visualização integrada e ergonômica.

- Limpeza Automática de Análise na Transição de Modos: Ao alternar entre os Modos de Operação (Tempo Real, Detecção Gravada e Treinamento & Aprendizado), a análise anterior, revisões, arquivos carregados e workers em execução são limpos automaticamente para manter o ambiente pronto para uma nova análise.

- Botão Manual de Limpeza e Nova Análise: Ação rápida "🧹 Nova Análise / Limpar" integrada no topo dos resultados gravados e de treinamento para reinicialização instantânea.

- Calibração de Sensibilidade Inteligente: No Modo de Treinamento & Aprendizado, o perfil é definido por padrão automaticamente como Treino Geral / Keiko (Normal).

- Validação e Consenso Multi-Câmeras (Tempo Real): A definição de haver ou não o golpe é tomada com base no conjunto das imagens das câmeras em background. Quanto mais câmeras ativas, maior a exigência de confirmação síncrona em frames (quórum escalonado de 1 a 4 câmeras), com descarte silencioso de movimentações sem consenso.

- Placar e Contador de Pontos em Tempo Real: No topo da lista de golpes ao vivo, monitora os Ippons e total de tentativas para Kenshi Shiro (Branco) e Kenshi Aka (Vermelho), com total consolidado e atualização instantânea a cada golpe avaliado.

- Feed de Golpes em Ordem Invertida (Tempo Real): Conforme os golpes ocorrem na transmissão ao vivo, os eventos mais recentes são exibidos diretamente no topo do histórico com diagnóstico expansível em HTML nativo de alta performance.

- Placar Oficial Eletrônico (Sanbon-shobu) com contagem de Ippon para Aka (Vermelho) e Shiro (Branco), declaração regulamentar de resultado e marcação em caracteres Katakana oficiais (メ, コ, ド, ツ).

- Destaque Visual na Linha do Tempo: golpes responsáveis pela marcação de Ippon são destacados com a nomenclatura oficial Katakana (メ MEN, コ KOTE, ド DO, ツ TSUKI), banner cromático por lutador e identificação no seletor de navegação rápida.

- Seleção de Origem do Vídeo no Modo de Detecção Gravada: suporte completo tanto para Upload de Arquivo Local (.mp4, .avi, .mov) quanto para inclusão e download direto de Links do YouTube / Streaming Web com extração de metadados, miniatura e cache inteligente.

- Novo Formato de Detecção via Streaming (Modos de Lutas e Treinamento): Suporte direto a transmissões ao vivo ou vídeos da web (YouTube Live, YouTube VOD/Shorts, Twitch, playlists HLS .m3u8, RTMP e HTTP/RTSP). O motor decodifica o stream contínuo com zero buffer, executando a análise na mesma dinâmica da detecção em tempo real, com placar Sanbon-Shobu e alertas de golpe (em Lutas) ou HUD dos 3 Pilares e contagem de repetições (em Treinamento), incluindo teste prévio de conectividade com amostra, resolução, FPS e latência.

- Controle de Velocidade e Navegação do Vídeo no Streaming (Ambos os Modos): Barra de controle interativa acoplada diretamente à reprodução do streaming permitindo selecionar 4 velocidades de reprodução (0.5x câmera lenta, 1.0x velocidade normal, 1.5x acelerada e 2.0x rápida), além de botões para retroceder (⏪ -10s, ⏪ -5s), pausar/retomar (⏸️/▶️) e avançar (⏩ +5s, ⏩ +10s). O stream mantém conexão persistente sem reconectar a cada comando, regulando a cadência de leitura de frames no motor assíncrono e exibindo o tempo decorrido do vídeo (MM:SS.s) e taxa de FPS em tempo real.

- Navegação Interativa no Vídeo com Salto Temporal Calibrado: clique no evento (Sonkyō Inicial, Golpe ou Sonkyō Final) ou selecione no dropdown para posicionar o player 1 segundo antes (-1.0s) do início do evento para revisão detalhada.

- Delimitação Temporal da Luta por Sonkyō: identificação automática da postura ritualística de Sonkyō de início e término do combate, com filtragem estrita de golpes na janela regulamentar.

- Rastreamento dos 2 Kenshi Principais e Filtragem de Planos: descarte inteligente de elementos de fundo (outras lutas, arquibancadas), árbitros (Shinpans) e oclusões de primeiro plano, com aplicação automática de margens da quadra (Shiai-jo) e supressão de ruído visual no vídeo anotado.

- Delimitação de Quadra (Shiai-jo) & Descarte Silencioso de Não-Kendocas: isolamento estrito dos combatentes centrais na área regulamentar de combate, eliminando caixas de anotação sobre árbitros e espectadores externos.

- Menu de Configurações com seletor de Hardware (CPU vs GPU NVIDIA CUDA com Tensor Cores FP16 / Ultralytics YOLOv8-Pose) e painel de Governança de Treinamento.

- Governança de Treinamento e Painel de Revisão por Dan: Distribuição de treinamentos por Dan (1º a 8º Dan) e linha dedicada para Treinamentos Automatizados por IA (Web & Vídeo). Cálculo de nível médio Dan exclusivo para revisões humanas (sem distorção por treinos da IA).

- Painel de Diagnóstico, Alertas e Log de Debug do Sistema com monitoramento de eventos em tempo real (`senpai_debug.log`).

- Edição de Golpes por Dan no Modo de Detecção Gravada (Shodan 1º Dan a Hachidan 8º Dan) com badges CONFIRMADO e EDITADO.

- Botão de Reset Geral e Individual de revisão, regra de auditabilidade (sem exclusão de dados) e retreinamento contínuo.

- Gestão de Pacotes de Treinamento: exportação e importação de arquivos JSON com registro por Dan e data.

- Layout responsivo em 2 colunas: vídeo fixo (sticky) à esquerda e lista de eventos com barra de rolagem à direita.

- Treinamento Automático por IA (Web & Vídeo Knowledge Ingestion): Na aba de Governança de Treinamento, permite disparar o motor autônomo de busca e mineração de conhecimento técnico (FIK, AJKF/ZNKR, biomecânica e corpus de vídeos) com duração determinada (em minutos) para aprender e recalibrar as 14 modalidades pedagógicas ou lutas (Shiai), conforme a necessidade mais latente, com tabela de evolução em HTML de alta performance.

- Conformidade Estrita de Tipagem e Estabilidade: Narrowing de tipos e tratamento defensivo em todos os widgets Streamlit (câmeras, Dan, perfis, qualidades e acelerador), além de blindagem de renderização condicional da Linha do Tempo prevenindo exceções de runtime em resets ou antes da execução da análise.

- Controles Interativos de Reprodução de Vídeo & Mesa VAR: Painel de controle acoplado diretamente abaixo do vídeo na Detecção Gravada, permitindo Play, Pause, Reinício, saltos rápidos (±1s, ±5s, ±10s), 7 taxas de velocidade (0.1x ultra-lenta a 2.0x rápida), volume e tela cheia via manipulação direta do `<video>` no DOM (executado via `st.iframe`). Sincronização automática com a Linha do Tempo sem dependência do parâmetro 'key' de `st.video`.

- Gestão e Retreinamento por Planilha Excel (.xlsx): Exportação da telemetria completa dos golpes detectados para planilha Excel, geração de template vazio para preenchimento manual e importação de planilhas revisadas por árbitros (1º a 8º Dan) com retreinamento adaptativo imediato da IA.

- Decisão dos Shinpans (Árbitros de Shiai) & Linha do Tempo Liberada: Opção regulamentar para registrar exclusivamente os golpes válidos (Ippons) concedidos pelos árbitros em Shiai. A linha do tempo é liberada (sem pré-carregar golpes não assinalados), disponibilizando painel de sugestões rápidas da IA (botão "➕ Ippon dos Shinpans" em 1 clique), inseridores inline "+" e inclusão manual com placar oficial computando estritamente os Ippons arbitrais.

- 5 Regras Estritas de Transição de Estado na Linha do Tempo:

  1. Habilitar edição desmarcada: Apresenta a lista completa de golpes detectados pela IA.

  2. Habilitar edição marcada + Dan selecionado (1º a 8º Dan): Apresenta golpes de IA com ações de revisão do treinador.

  3. Habilitar edição marcada + Decisão dos Shinpans: Linha do tempo limpa apenas para Ippons concedidos pelos árbitros.

  4. Habilitar edição desmarcada após Decisão dos Shinpans: Todos os golpes de IA voltam a ser apresentados imediatamente.

  5. Seleção de Dan após Decisão dos Shinpans: Todos os golpes de IA voltam a ser apresentados com ferramentas do Dan escolhido.

- Governança & Calibração dos Shinpans com Peso Balanceado (4.5): As decisões dos árbitros utilizam peso médio fixo (4.5) para recalibração normalizada dos critérios biomecânicos sem distorcer o cálculo do nível médio Dan puro dos treinadores (1º ao 8º Dan), com estatísticas e contadores dedicados.

- Motor de Aprendizado Ativo & Triagem por Incerteza (Eixo 4): Amostragem seletiva de lances com incerteza na faixa de contorno (45% a 65%) para curadoria humana prioritária, eliminando ruído de casos triviais.

- Salvaguarda Mandatória com Dataset Canônico Golden Benchmark: Base imutável de testes (WKC e exames de alto Dan) que bloqueia automaticamente qualquer retreinamento ou alteração que provoque esquecimento catastrófico ou regressão na acurácia (> 2.0%).

- Consenso Multi-Árbitro Ponderado por Dan (FIK 2 de 3): Julgamento colegiado com detecção e cálculo numérico de Grau de Divergência Arbitral.

- Gestão de Confiança com Decaimento Temporal Exponencial: Modelo de meia-vida (180 dias) para controle de autoridade de revisores por inatividade.

- Assistente LLM Especialista em Kendo: Integração nativa com APIs Gemini/OpenAI e motor de regras offline determinístico da FIK para anotação cinemática estruturada e explicação didática de lances controversos.

- Painel de Curadoria Rápida no Streamlit: Na aba de Calibração, visualização das métricas do Golden Benchmark e ações de aprovação/rejeição rápida com 1 clique na fila de incerteza.



--- B. LINHA DE COMANDO (CLI) ---

Para processar vídeos diretamente no terminal:



- Executar análise com vídeo próprio (perfil normal padrão):

   python main.py --video "caminho/do/seu_video.mp4"



- Executar teste rápido com GPU NVIDIA (se disponível):

   python main.py --video "caminho/do/seu_video.mp4" --device gpu



- Especificar perfil de calibração:

   python main.py --video "video.mp4" --profile rigido



- Especificar caminho do vídeo anotado de saída:

   python main.py --video "video.mp4" --output "meu_resultado.mp4"



- Otimizar perfil com base no dataset de feedback gravado:

   python main.py --profile normal --optimize-profile



- Executar suíte completa de testes automatizados (232 testes) com relatório descritivo salvo em logs/:

   python run_tests.py

   (ou no Windows: .\.venv\Scripts\python.exe run_tests.py)

   * Executa a suíte de 232 testes automatizados distribuídos em 25 módulos com 100% de aprovação

   * Validações especializadas de rastreamento K=2, supressão de Shinpans, governança Dan, Decisão dos Shinpans e Aprendizado Ativo (Eixo 4)

   * Gera o arquivo descritivo logs/senpai_test_report.log com detalhes de cada teste

   * Mantém estritamente apenas o último log de testes executado na pasta logs/



- Ou executar via runner padrão unittest:

   .\.venv\Scripts\python.exe -m unittest discover tests



-------------------------------------------------------------------------------

3. OPÇÕES E FLAGS DA LINHA DE COMANDO (CLI)

-------------------------------------------------------------------------------



Sintaxe: python main.py [OPÇÕES]



Opções disponíveis:

  --video CAMINHO         Caminho para o vídeo de entrada (.mp4, .avi, .mov).

  --output CAMINHO        Caminho para salvar o vídeo anotado.

                          (Padrão: output_annotated.mp4)

  --profile PERFIL        Perfil de calibração a aplicar:

                          - normal     : Treino geral / Keiko (Padrão)

                          - rigido     : Campeonato / Exame de Dan

                          - permissivo : Iniciantes / Educacional

  --device DISPOSITIVO    Dispositivo de processamento:

                          - cpu        : Somente CPU (Padrão)

                          - gpu        : GPU NVIDIA CUDA (quando houver, com fallback para CPU)

  --model MODELO          Modelo neural de visão computacional:

                          - yolov8     : YOLOv8 Pose (Baseline Estável & Consolidado)

                          - yolov11    : YOLOv11 Pose (Otimizado C3k2 & Alta Velocidade)

                          - yolov12    : YOLOv12 Pose (Atenção Centrada & Anti-Oclusão no Tsubazeriai)

                          - yolov26    : YOLOv26 Pose (Next-Gen Ultra AI 2026 - Arbitragem FIK)

  --mode MODO             Modo de operação:

                          - user       : Análise padrão (Padrão)

                          - learning   : Ativa recursos para registro de feedback

  --optimize-profile      Executa o motor de aprendizagem por reforço para

                          recalibrar o perfil selecionado com base nas

                          anotações gravadas em data/feedback_dataset.json.



-------------------------------------------------------------------------------

4. MODOS DE OPERAÇÃO E NAVEGAÇÃO

-------------------------------------------------------------------------------



0. PÁGINA INICIAL / VISÃO GERAL (home):

   Ponto de partida do sistema carregado na inicialização. Apresenta:

   - Visão geral institucional e conformidade com diretrizes da FIK e AJKF/ZNKR.

   - Indicador de status de aceleração de hardware em tempo real (GPU NVIDIA CUDA / CPU).

   - Demonstrativo passo a passo de como iniciar o uso em 4 etapas didáticas.

   - Apresentação visual e comparativa dos 2 Grandes Modos de Operação do SenpAI.

   - Explicabilidade da Calibração Realista (< 50% na fase inicial) e governança por Dan.

   - Atalhos diretos de ação rápida para Análise de Lutas, Treinamento & Aprendizado e Configurações.



1. MODO DE ANÁLISE DE LUTAS (match):

   Dedicado ao combate competitivo oficial (Shiai-geiko). Avalia a validade de cada

   golpe com base nas 5 condições de Yūko-Datotsu da FIK, delimita o tempo de combate

   por Sonkyō e gerencia o placar Sanbon-Shobu (Aka vs Shiro). Possui 3 formatos:



   A) Detecção em Tempo Real (realtime):

      - Suporta de 1 a 4 câmeras simultâneas (Webcams locais ou Streams RTSP / Câmeras IP).

      - Validação de consenso multi-câmeras em background com quórum escalonado e descarte silencioso de falsos positivos.

      - Placar oficial instantâneo (Sanbon-Shobu) com Ippons e golpes ao vivo.

      - Histórico de golpes em ordem invertida (mais recentes no topo) com cards explicativos de Ki-Ken-Tai-Ichi.



   B) Detecção via Streaming (streaming): [NOVO]

      - Conexão e análise direta em tempo real de links de streaming da web (YouTube Live, YouTube VOD/Shorts, Twitch, HLS .m3u8, RTMP ou HTTP/RTSP).

      - Decodificação de fluxo de alta velocidade com buffer zero sem necessidade de download prévio completo do arquivo.

      - Avaliação biomecânica instantânea de Yūko-Datotsu, placar Sanbon-Shobu oficial (Shiro vs Aka) e banner de golpes em tempo real.

      - Diagnóstico e teste de conexão prévio com amostra em miniatura, métricas de resolução, FPS e latência da rede.



   C) Detecção Gravada (recorded):

      - Análise biomecânica completa de vídeos gravados (upload local ou YouTube).

      - Delimitação temporal estrita entre Sonkyō inicial e final.

      - Placar eletrônico oficial com caracteres Katakana (メ, コ, ド, ツ).

      - Linha do tempo interativa com controles VAR (saltos ±1s/±5s/±10s e velocidades de 0.1x a 2.0x).

      - Gestão de golpes via Excel (.xlsx) com importação e retreinamento adaptativo por Dan.

      - Modo "Decisão dos Shinpans" e edição de golpes na interface (1º Dan Shodan a 8º Dan Hachidan).



2. MODO DE TREINAMENTO & APRENDIZADO (training):

   Projetado para instrutores e atletas para refinamento técnico e preparação para exames.

   Reconhece as 14 modalidades oficiais de Kendo e avalia os 3 Pilares Fundamentais:

   • Movimentação (35%): Postura (Shisei), nivelamento de ombros e base de pés.

   • Precisão (35%): Trajetória no alvo, Ki-Ken-Tai-Ichi e linha central.

   • Constância (30%): Regularidade rítmica, cadência (CPM) e resistência à fadiga.

   Possui 3 formatos:



   A) Análise em Tempo Real (realtime):

      - Executada via Webcams locais USB ou câmeras de rede RTSP (1 a 4 câmeras simultâneas).

      - Configuração e teste de câmeras com detecção automática de dispositivos e diagramas de layout no Dojo.

      - HUD dinâmico dos 3 Pilares atualizado continuamente com scores e barras visuais.

      - Contagem automática de repetições / cortes com cálculo de cadência em tempo real (CPM).

      - Biofeedback postural instantâneo (alertas de Shisei ereto, calcanhar suspenso, alinhamento de ombros).

      - Geração e download de Relatório Consolidado de Treinamento da Sessão Ao Vivo (.md e .json).



   B) Detecção via Streaming (streaming): [NOVO]

      - Avaliação biomecânica do treinamento ao vivo em transmissões da web (YouTube Live, YouTube VOD, Twitch, HLS .m3u8, RTMP ou RTSP) na mesma dinâmica da análise em tempo real.

      - Identificação contínua da modalidade praticada dentre as 14 oficiais de Kendo com ideogramas Kanji.

      - Projeção do HUD dos 3 Pilares Fundamentais com pontuação e barras de progresso ao vivo.

      - Contagem automática de repetições e biofeedback biomecânico instantâneo durante a exibição do stream.

      - Resumo consolidado pós-treino com emissão de relatórios para download (.md e .json).



   C) Análise de Vídeos Gravados (recorded):

      - Rastreamento individual e nomeação de cada Kendoca presente.

      - Diagnósticos pedagógicos: Pontos Fortes, Pontos de Atenção e Prescrições de Exercícios do Dojo.

      - Emissão de relatórios individuais em Markdown (.md) e dados da sessão em JSON (.json).

      - Layout ergonômico lado a lado e botão de inversão de lados dos atletas.



-------------------------------------------------------------------------------

5. PERFIS DE CALIBRAÇÃO & MOTOR MATEMÁTICO (EIXO 1)

-------------------------------------------------------------------------------



O SenpAI incorpora um Motor Matemático de Calibração formal (mathematical_calibrator.py):

- Otimização Bayesiana / Numérica:

  Busca automática de parâmetros ótimos via Optuna (TPE) ou SciPy SLSQP,

  com restrições estritas (soma dos pesos = 1.0 e cada peso >= 0.10).

- Penalização Assimétrica de Competição (Shiai/FIK):

  Custo 3x maior para Falso Positivo (3.0x FP / 1.0x FN), pois as normas

  arbitrais da FIK exigem certeza absoluta para a validação de Ippon.

- Classificador Probabilístico Calibrado (Platt Scaling):

  Calcula a probabilidade real contínua de Ippon: P(Ippon) = sigmoid(A*s + B),

  gerando percentuais precisos de confiança arbitral (ex: 88.4%).

- Monitoramento de Deriva Temporal (Concept Drift):

  Aplica o teste estatístico bilateral Kolmogorov-Smirnov (scipy.stats.ks_2samp)

  comparando scores recentes com a base histórica para detecção de desvios.

- Pesos Especializados por Tipo de Golpe (weights_by_strike_type):

  • Men: ênfase máxima em sincronismo mão-pé Ki-Ken-Tai-Ichi (30% Fumikomi).

  • Kote: ênfase em extensão de cotovelo e contato firme no Bogu (45% Alvo).

  • Do: ênfase em Hasuji e trajetória de corte lateral (45% Alvo / 20% Postura).

  • Tsuki: ênfase em colinearidade e estocada penetrante (50% Alvo).



Perfis Pré-configurados:

- Iniciantes / Educacional (Permissivo):

  Limiar mínimo: 50%. Foco formativo, tolerando pequenas falhas posturais.

- Treino Geral / Keiko (Normal):

  Limiar mínimo: 74%. Equilíbrio padrão para treinos diários e exames de Kyu.

- Campeonato / Audit de Dan (Rígido):

  Limiar mínimo: 78%. Alta exigência em Ki-Ken-Tai-Ichi e Zanshin.

- Oficial de Competição (Shiai):

  Limiar mínimo: 65%. Calibrado com custo assimétrico estrito de 3x contra Falsos Positivos.

- Personalizado (Custom):

  Ajuste livre via sliders em tempo real no dashboard Web.



-------------------------------------------------------------------------------

6. PARÂMETROS FÍSICOS ACIONÁVEIS & HIERARQUIA DE FONTES (EIXO 2)

-------------------------------------------------------------------------------



O SenpAI converte pesquisas regulamentares e vídeos oficiais em restrições

físicas acionáveis e priors bayesianos rígidos (actionable_research.py):



- Pipeline de Extração Estruturada (Knowledge -> JSON Schema):

  Extrai e valida limites biomecânicos a partir de manuais da FIK/AJKF e literatura:

  • Extensão de cotovelos no Men: [150.0°, 175.0°] (ideal: 165.0°).

  • Inclinação máxima de coluna (Shisei): <= 8.5°.

  • Janela de simultaneidade do Fumikomi: [-45.0, 30.0] ms.

  • Duração mínima de Zanshin regulamentar: >= 0.80 s.

  • Desvio angular máximo de corte (Hasuji): <= 12.0°.

  • Zona de contato regulamentar: Datotsu-bu / Monouchi do Shinai.



- Hierarquia Estrita de Fontes & Resolução de Conflitos:

  Classificação das 5 camadas de autoridade marcial:

  1. FIK Official Rulebook (Autoridade Máxima - 1.00): prevalece sempre.

  2. AJKF Referee Handbook (Autoridade Alta - 0.85): manuais práticos de arbitragem.

  3. Literatura Especializada (Autoridade Média - 0.65): tratados técnicos de mestres.

  4. Artigos Científicos (Autoridade Baixa - 0.45): estudos laboratoriais de biomecânica.

  5. Blogs e Fóruns (Descartado - 0.00): desconsiderados como prior de calibração.

  • Regra do Conservadorismo: em empates de autoridade, o critério que exige maior

    rigor técnico e menor tolerância a erro prevalece incondicionalmente.



- Mineração de Vídeos de Referência Oficial:

  Extrai e calcula distribuições empíricas de referência (média, desvio padrão,

  mínimo, máximo e percentis p25, p50 [mediana], p75, p90) a partir de lutas

  oficiais com confirmação arbitral (2 ou 3 bandeiras), salvas em:

  data/empirical_reference_distributions.json.



- Injeção de Priors Bayesianos no Calibrador:

  As restrições físicas atuam como fronteiras intransponíveis (boundary conditions)

  durante a otimização de pesos e sub-limiares pelo BayesianCalibrationOptimizer.



- Reconhecimento Multimodal de Yuko-Datotsu (Eixo 3):

  • Interação Atacante <-> Defensor: colisão Shinai-Bogu e eliminação de Ku-totsu (golpe no vazio).

  • Hasuji (5° Pilar): ângulo da lâmina (Men vertical +-15°, Do diagonal 30°-45°, Tsuki horizontal +-10°).

  • Detecção de Seme: pressão pré-golpe no Chushin-sen em janela de 20 a 30 frames.

  • Detecção de Oji-waza / Debana: identificação retroativa de contrataques em 10 a 15 frames.

  • Fusão Multimodal de Áudio: sincronismo de Datotsu-on (estalo 1.5k-4kHz) e Kiai vocal com delta_t <= 40ms.

  • Action Spotting TCN: classificação temporal de 10 classes de Kendo com supressão de falso disparo em Tsubazeriai.



- Invariância de Câmera & Normalização Espacial 3D (Eixo 5):

  • Compensação de Perspectiva: estimativa do vetor de combate (Aka-Shiro), classificação

    angular (Frontal 0°-30°, Oblíquo 30°-60°, Lateral 60°-90°) e normalização 1/sin(theta)

    da inclinação da coluna (Shisei).

  • Estimativa Monocular 3D: reconstrução de pseudo-keypoints (x, y, z) e distância

    tridimensional (Maai 3D), eliminando definitivamente Ku-totsu (golpe no vazio).

  • Diagnóstico de Qualidade: notas de confiabilidade por critério e modificadores de peso.



-------------------------------------------------------------------------------

7. CRITÉRIOS AVALIADOS (KI-KEN-TAI-ICHI, HASUJI & MAAI)

-------------------------------------------------------------------------------



- Alvo / Impacto (Ken): Precisão do impacto do Kensen na zona de ataque (Men, Kote, Do, Tsuki).

- Contato e Distância (Maai): Proximidade física e alcance entre os combatentes no impacto (rejeição de golpes fora de alcance).

- Fumikomi (Tai): Sincronismo entre o impacto do pé no solo e a aceleração dos pulsos.

- Postura Corporal (Tai): Alinhamento da coluna e estabilidade no momento do golpe.

Para documentação técnica completa, detalhamento dos algoritmos e histórico de

mudanças, consulte o arquivo manual.md.



-------------------------------------------------------------------------------

8. DIREITOS DE USO, CÓPIA E PROPRIEDADE INTELECTUAL

-------------------------------------------------------------------------------



(C) 2026 SenpAI (先輩 AI) - Plataforma Inteligente de Kendo.

TODOS OS DIREITOS DE USO E CÓPIA RESERVADOS.



Este software, seus algoritmos de visão computacional, modelos heurísticos e

neurais, perfis de calibração biomecânica, interfaces visuais e documentações

associadas constituem propriedade intelectual protegida pelas leis de direitos

autorais e tratados internacionais.



É expressamente proibida a reprodução, cópia, modificação, redistribuição,

comercialização ou engenharia reversa, no todo ou em parte, por qualquer meio

ou processo, sem a prévia e expressa autorização por escrito dos detentores

dos direitos. Desenvolvido em estrita observância às diretrizes regulamentares

da International Kendo Federation (FIK) e All Japan Kendo Federation (AJKF/ZNKR).



