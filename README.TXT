===============================================================================
                       SHINPANAI (審判 AI) - GUIA RÁPIDO
        Sistema de Arbitragem e Análise Biomecânica de Kendo com IA
===============================================================================

O Shinpanai é um sistema de visão computacional e análise biomecânica projetado
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
- Menu de Configurações com seletor de Hardware (CPU vs GPU NVIDIA) e painel de Governança de Treinamento.
- Painel de Diagnóstico, Alertas e Log de Debug do Sistema com monitoramento de eventos em tempo real (registra resets, uploads e retreinamentos).
- Botão para baixar o arquivo completo de log de debug (`shinpanai_debug.log`) e teste de diagnóstico automatizado de integridade.
- Edição de Golpes por Dan no Modo de Arbitragem Gravada (Shodan 1º Dan a Hachidan 8º Dan).
- Badges visuais de status em tempo real: CONFIRMADO (verde) e EDITADO (azul) em cada card de golpe.
- Botão de Reset Geral da Revisão para a sessão e botão de Reset Individual por golpe.
- Permite confirmar marcações, editar parâmetros do golpe e incluir marcações adicionais (golpes perdidos).
- Regra de Auditabilidade: Exclusão de marcações estritamente desabilitada para preservação de dados.
- Botão de salvamento final com retreinamento adaptativo do modelo de detecção.
- Painel de Governança nas Configurações: contador de treinamentos, nível médio (Dan) dos treinamentos e tabela por Dan.
- Recursos de gestão nas Configurações: apagar treinamento (reset), baixar treinamento (JSON com Dan e data) e carregar pacotes baixados (suporte a múltiplos formatos JSON).
- Upload de vídeos de lutas (.mp4, .avi, .mov).
- Layout responsivo em 2 colunas: vídeo fixo (sticky) à esquerda e lista de golpes com barra de rolagem à direita.
- Alternador de exibição no player: Vídeo Anotado (Visão AI) vs Vídeo Original.
- 3 modos de uso: Tempo real, Luta gravada e Treinamento & Aprendizado.
- Seleção e personalização fina de perfis de arbitragem em tempo real.
- Visualização de vídeos anotados com suporte a esqueleto 3D e zonas de impacto.

--- B. LINHA DE COMANDO (CLI) ---
Para processar vídeos diretamente no terminal:

- Executar análise com vídeo próprio (perfil normal padrão):
   python main.py --video "caminho/do/seu_video.mp4"

- Executar teste rápido de demonstração com GPU NVIDIA (se disponível):
   python main.py --demo --device gpu

- Especificar perfil de arbitragem:
   python main.py --video "video.mp4" --profile rigido

- Especificar caminho do vídeo anotado de saída:
   python main.py --video "video.mp4" --output "meu_resultado.mp4"

- Otimizar perfil com base no dataset de feedback gravado:
   python main.py --profile normal --optimize-profile

- Executar suíte completa de testes automatizados:
   .\.venv\Scripts\python.exe -m unittest discover tests

-------------------------------------------------------------------------------
3. OPÇÕES E FLAGS DA LINHA DE COMANDO (CLI)
-------------------------------------------------------------------------------

Sintaxe: python main.py [OPÇÕES]

Opções disponíveis:
  --video CAMINHO         Caminho para o vídeo de entrada (.mp4, .avi).
  --output CAMINHO        Caminho para salvar o vídeo anotado.
                          (Padrão: output_annotated.mp4)
  --profile PERFIL        Perfil de arbitragem a aplicar:
                          - normal     : Treino geral / Keiko (Padrão)
                          - rigido     : Campeonato / Exame de Dan
                          - permissivo : Iniciantes / Educacional
  --device DISPOSITIVO    Dispositivo de processamento:
                          - cpu        : Somente CPU (Padrão)
                          - gpu        : GPU NVIDIA (quando houver, com fallback para CPU)
  --mode MODO             Modo de operação:
                          - user       : Análise padrão (Padrão)
                          - learning   : Ativa recursos para registro de feedback
  --optimize-profile      Executa o motor de aprendizagem por reforço para
                          recalibrar o perfil selecionado com base nas
                          anotações gravadas em data/feedback_dataset.json.
  --demo                  Gera um vídeo sintético de demonstração e executa a
                          arbitragem sem necessidade de vídeo externo.

-------------------------------------------------------------------------------
4. MODOS DE OPERAÇÃO
-------------------------------------------------------------------------------

1. MODO DE ARBITRAGEM GRAVADA (recorded):
   Projetado para análise de combates gravados. Permite habilitar o botão de edição
   dos golpes detectados com seleção do DAN do revisor (Shodan a Hachidan).
   Permite confirmar, editar ou incluir golpes perdidos (sem permissão de exclusão).
   Ao final, salva e retreina o modelo de detecção.

2. MODO DE TREINAMENTO & APRENDIZADO (training):
   Projetado para instrutores e atletas. Permite rotular resultados (TP, FP, FN),
   calibrar a exigência por nível Dan e recalibrar adaptativamente os limiares.

3. MODO DE DETECÇÃO EM TEMPO REAL (realtime):
   Processamento instantâneo via Webcam ou streams RTSP/RTCP de câmeras IP.

-------------------------------------------------------------------------------
5. PERFIS DE ARBITRAGEM (SENSIBILIDADE)
-------------------------------------------------------------------------------

- Treino Geral / Keiko (Normal):
  Limiar mínimo: 65%. Equilíbrio ideal entre precisão técnica e tolerância.
- Campeonato / Audit de Dan (Rígido):
  Limiar mínimo: 82%. Exige execução perfeita de Ki-Ken-Tai-Ichi e Zanshin.
- Iniciantes / Educacional (Permissivo):
  Limiar mínimo: 45%. Foco na atração ao alvo correto, tolerando pequenas falhas
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



