===============================================================================
                        SENPAI (先輩 AI) - GUIA RÁPIDO
          Sistema de Avaliação e Análise Biomecânica de Kendo com IA
===============================================================================

O SenpAI é um sistema de visão computacional e análise biomecânica projetado
para avaliar golpes em lutas de Kendo de acordo com o conceito de Yuko-Datotsu
e o princípio do Ki-Ken-Tai-Ichi (Espírito, Espada e Corpo em um só).

-------------------------------------------------------------------------------
1. REQUISITOS E INSTALAÇÃO
-------------------------------------------------------------------------------

Requisitos de Sistema:
- Python 3.10 ou 3.11
- Ambiente Virtual (.venv recomendado)
- Opcional para Modo GPU: Placa aceleradora NVIDIA (GeForce RTX/GTX) com drivers CUDA.

Dependências principais:
- opencv-python
- ultralytics (YOLOv8-Pose com aceleração GPU CUDA FP16)
- mediapipe
- numpy / scipy
- fastdtw
- streamlit (para a interface web)

Instalação das dependências:
   pip install -r requirements.txt
   (ou usando uv: uv pip install -r requirements.txt)

Para habilitação opcional de aceleração CUDA em GPU NVIDIA:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

-------------------------------------------------------------------------------
2. INSTRUÇÕES DE USO (COMO EXECUTAR)
-------------------------------------------------------------------------------

--- A. INTERFACE WEB INTERATIVA (STREAMLIT - RECOMENDADO) ---
Para iniciar o dashboard web com reprodutor de vídeo, gráficos e sliders:

   streamlit run app.py
   (ou no Windows: .\.venv\Scripts\streamlit.exe run app.py)

Recursos da Interface Web:
- Interface otimizada e dimensionada (escala visual 80%) para visualização integrada e ergonômica.
- Validação e Consenso Multi-Câmeras (Tempo Real): A definição de haver ou não o golpe é tomada com base no conjunto das imagens das câmeras em background. Quanto mais câmeras ativas, maior a exigência de confirmação síncrona em frames (quórum escalonado de 1 a 4 câmeras), com descarte silencioso de movimentações sem consenso.
- Placar Oficial Eletrônico (Sanbon-shobu) com contagem de Ippon para Aka (Vermelho) e Shiro (Branco), declaração regulamentar de resultado e marcação em caracteres Katakana oficiais (メ, コ, ド, ツ).
- Destaque Visual na Linha do Tempo: golpes responsáveis pela marcação de Ippon são destacados com a nomenclatura oficial Katakana (メ MEN, コ KOTE, ド DO, ツ TSUKI), banner cromático por lutador e identificação no seletor de navegação rápida.
- Detecção da Flag Dorsal (Tasukuki) e botão de ação rápida de inversão de identidade (Aka ⇄ Shiro).
- Navegação Interativa no Vídeo com Salto Temporal Calibrado: clique no evento (Sonkyō Inicial, Golpe ou Sonkyō Final) ou selecione no dropdown para posicionar o player 1 segundo antes (-1.0s) do início do evento para revisão detalhada.
- Delimitação Temporal da Luta por Sonkyō: identificação automática da postura ritualística de Sonkyō de início e término do combate, com filtragem estrita de golpes na janela regulamentar.
- Rastreamento dos 2 Kenshi Principais e Filtragem de Planos: descarte inteligente de elementos de fundo (outras lutas, arquibancadas) e oclusões de primeiro plano.
- Menu de Configurações com seletor de Hardware (CPU vs GPU NVIDIA CUDA com Tensor Cores FP16) e painel de Governança de Treinamento.
- Painel de Diagnóstico, Alertas e Log de Debug do Sistema com monitoramento de eventos em tempo real (`senpai_debug.log`).
- Edição de Golpes por Dan no Modo de Detecção Gravada (Shodan 1º Dan a Hachidan 8º Dan) com badges CONFIRMADO e EDITADO.
- Botão de Reset Geral e Individual de revisão, regra de auditabilidade (sem exclusão de dados) e retreinamento contínuo.
- Gestão de Pacotes de Treinamento: exportação e importação de arquivos JSON com registro por Dan e data.
- Layout responsivo em 2 colunas: vídeo fixo (sticky) à esquerda e lista de eventos com barra de rolagem à direita.
- Alternador de exibição no player: Vídeo Anotado (Visão AI com HUD superior) vs Vídeo Original.

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

- Executar suíte completa de testes automatizados (52 testes) com relatório descritivo salvo em logs/:
   python run_tests.py
   (ou no Windows: .\.venv\Scripts\python.exe run_tests.py)
   * Executa a suíte de 52 testes automatizados com 100% de aprovação
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
  --mode MODO             Modo de operação:
                          - user       : Análise padrão (Padrão)
                          - learning   : Ativa recursos para registro de feedback
  --optimize-profile      Executa o motor de aprendizagem por reforço para
                          recalibrar o perfil selecionado com base nas
                          anotações gravadas em data/feedback_dataset.json.

-------------------------------------------------------------------------------
4. MODOS DE OPERAÇÃO
-------------------------------------------------------------------------------

1. MODO DE DETECÇÃO GRAVADA (recorded):
   Análise biomecânica completa de combates gravados. Inclui detecção e delimitação
   temporal da luta por Sonkyō, pontuação ponderada de Yuko-Datotsu (Ki-Ken-Tai-Ichi),
   placar eletrônico oficial (Aka vs Shiro com caracteres Katakana: メ, コ, ド, ツ),
   destaque de golpes pontuados na Linha do Tempo (メ MEN, コ KOTE, ド DO, ツ TSUKI),
   navegação interativa com salto calibrado de 1 segundo antes do evento, edição de
   golpes por Dan (Shodan a Hachidan), corte automático de clipes e relatórios
   diagnósticos de combate.

2. MODO DE TREINAMENTO & APRENDIZADO (training):
   Projetado para instrutores e atletas. Permite rotular resultados (TP, FP, FN),
   calibrar a exigência por nível Dan e recalibrar adaptativamente os limiares.

3. MODO DE DETECÇÃO EM TEMPO REAL (realtime):
   Processamento instantâneo via Webcam ou streams RTSP/RTCP de 1 a 4 câmeras IP simultâneas.
   Conta com validação de consenso multi-câmeras em background baseada no conjunto
   de imagens/frames (com quórum escalonado e descarte silencioso de falsos positivos).

-------------------------------------------------------------------------------
5. PERFIS DE CALIBRAÇÃO (SENSIBILIDADE)
-------------------------------------------------------------------------------

- Treino Geral / Keiko (Normal):
  Limiar mínimo: 65%. Equilíbrio ideal entre precisão técnica e tolerância.
- Campeonato / Audit de Dan (Rígido):
  Limiar mínimo: 82%. Exige execução perfeita de Ki-Ken-Tai-Ichi e Zanshin.
- Iniciantes / Educacional (Permissivo):
  Limiar mínimo: 45%. Foco na precisão ao alvo, tolerando pequenas falhas
  de postura e sincronismo.
- Personalizado (Custom):
  Disponível na interface Web para ajuste manual de pesos e limiares via sliders.

-------------------------------------------------------------------------------
6. CRITÉRIOS AVALIADOS (KI-KEN-TAI-ICHI)
-------------------------------------------------------------------------------

- Alvo / Impacto (Ken): Precisão do impacto do Kensen na zona de ataque (Men, Kote, Do, Tsuki).
- Fumikomi (Tai): Sincronismo entre o impacto do pé no solo e a aceleração dos pulsos.
- Postura Corporal (Tai): Alinhamento da coluna e estabilidade no momento do golpe.
- Zanshin (Ki): Manutenção da prontidão e postura nos frames após o impacto.

Para documentação técnica completa, detalhamento dos algoritmos e histórico de
mudanças, consulte o arquivo manual.md.
