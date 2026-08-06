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

Requisitos:
- Python 3.10 ou 3.11
- Ambiente Virtual (.venv recomendado)

Dependências principais:
- opencv-python
- mediapipe
- numpy / scipy
- fastdtw
- streamlit (para a interface web)

Instalação das dependências:
   pip install -r requirements.txt
   (ou usando uv: uv pip install -r requirements.txt)

-------------------------------------------------------------------------------
2. INSTRUÇÕES DE USO (COMO EXECUTAR)
-------------------------------------------------------------------------------

--- A. INTERFACE WEB INTERATIVA (STREAMLIT - RECOMENDADO) ---
Para iniciar o dashboard web com reprodutor de vídeo, gráficos e sliders:

   streamlit run app.py
   (ou no Windows: .\.venv\Scripts\streamlit.exe run app.py)

Recursos da Interface Web:
- Upload de vídeos de lutas (.mp4, .avi, .mov) ou geração de vídeo demo sintético.
- Alternância entre Modo Usuário (análise limpa) e Modo Aprendizagem.
- Seleção e personalização fina de perfis de arbitragem em tempo real.
- Visualização de vídeos anotados com suporte a esqueleto 3D e zonas de impacto.
- Painel de anotação e treinamento adaptativo por aprendizagem por reforço.

--- B. LINHA DE COMANDO (CLI) ---
Para processar vídeos diretamente no terminal:

- Executar análise com vídeo próprio (perfil normal padrão):
   python main.py --video "caminho/do/seu_video.mp4"

- Executar teste rápido de demonstração (vídeo sintético gerado automaticamente):
   python main.py --demo

- Especificar perfil de arbitragem:
   python main.py --video "video.mp4" --profile rigido

- Especificar caminho do vídeo anotado de saída:
   python main.py --video "video.mp4" --output "meu_resultado.mp4"

- Otimizar perfil com base no dataset de feedback gravado:
   python main.py --profile normal --optimize-profile

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

1. MODO USUÁRIO (user):
   Projetado para atletas, alunos e treinadores. Exibe diretamente o status dos
   golpes (✅ IPPON / ❌ INVÁLIDO), diagnósticos biomecânicos e orientações de
   correção técnica sem poluição visual.

2. MODO APRENDIZAGEM (learning):
   Projetado para árbitros e especialistas. Permite rotular os resultados da IA:
   - TP (True Positive): Golpe correto / Validação confirmada.
   - FP (False Positive): Golpe validado indevidamente pela IA.
   - FN (False Negative): Golpe válido que a IA deixou de identificar.
   As anotações alimentam o otimizador por reforço para recalibrar
   automaticamente os limiares e pesos do perfil.

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
===============================================================================
