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
- Navegação Interativa no Vídeo com Salto Temporal Calibrado: clique no evento (Sonkyō Inicial, Golpe ou Sonkyō Final) ou selecione no dropdown para posicionar o player 1 segundo antes (-1.0s) do início do evento para revisão detalhada.
- Delimitação Temporal da Luta por Sonkyō: identificação automática da postura ritualística de Sonkyō de início e término do combate, com filtragem estrita de golpes na janela regulamentar.
- Rastreamento dos 2 Kenshi Principais e Filtragem de Planos: descarte inteligente de elementos de fundo (outras lutas, arquibancadas) e oclusões de primeiro plano.
- Menu de Configurações com seletor de Hardware (CPU vs GPU NVIDIA CUDA com Tensor Cores FP16 / Ultralytics YOLOv8-Pose) e painel de Governança de Treinamento.
- Governança de Treinamento e Painel de Revisão por Dan: Distribuição de treinamentos por Dan (1º a 8º Dan) e linha dedicada para Treinamentos Automatizados por IA (Web & Vídeo). Cálculo de nível médio Dan exclusivo para revisões humanas (sem distorção por treinos da IA).
- Painel de Diagnóstico, Alertas e Log de Debug do Sistema com monitoramento de eventos em tempo real (`senpai_debug.log`).
- Edição de Golpes por Dan no Modo de Detecção Gravada (Shodan 1º Dan a Hachidan 8º Dan) com badges CONFIRMADO e EDITADO.
- Botão de Reset Geral e Individual de revisão, regra de auditabilidade (sem exclusão de dados) e retreinamento contínuo.
- Gestão de Pacotes de Treinamento: exportação e importação de arquivos JSON com registro por Dan e data.
- Layout responsivo em 2 colunas: vídeo fixo (sticky) à esquerda e lista de eventos com barra de rolagem à direita.
- Treinamento Automático por IA (Web & Vídeo Knowledge Ingestion): Na aba de Governança de Treinamento, permite disparar o motor autônomo de busca e mineração de conhecimento técnico (FIK, AJKF/ZNKR, biomecânica e corpus de vídeos) com duração determinada (em minutos) para aprender e recalibrar as 14 modalidades pedagógicas ou lutas (Shiai), conforme a necessidade mais latente, com tabela de evolução em HTML de alta performance.
- Conformidade Estrita de Tipagem e Estabilidade: Narrowing de tipos e tratamento defensivo em todos os widgets Streamlit (câmeras, Dan, perfis, qualidades e acelerador), além de blindagem de renderização condicional da Linha do Tempo prevenindo exceções de runtime em resets ou antes da execução da análise.

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

- Executar suíte completa de testes automatizados (81 testes) com relatório descritivo salvo em logs/:
   python run_tests.py
   (ou no Windows: .\.venv\Scripts\python.exe run_tests.py)
   * Executa a suíte de 81 testes automatizados com 100% de aprovação
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
   Projetado para instrutores e atletas para refinamento técnico e preparação para exames.
   - Layout Ergonômico Lado a Lado: O painel de avaliação pedagógica fica perfeitamente posicionado
     ao lado do vídeo, sem a Linha do Tempo de arbitragem competitiva, oferecendo visualização direta e focada.
   - Suporte a Inversão de Lados dos Kendocas: Botão de alternância instantânea da ordem de exibição dos atletas.
   - 14 Modalidades Oficiais de Treinamento com Kanjis:
     1. Ashi-sabaki (足捌き) — Deslocamentos fundamentais (okuri-ashi, ayumi-ashi, hiraki-ashi, tsugi-ashi).
     2. Suburi (素振り) — Golpes repetidos no ar (jōge-buri, naname-buri, shōmen-uchi, sayū-men).
     3. Kihon (基本) — Fundamentos de postura (shisei), distância (maai), guarda (kamae), golpe e zanshin.
     4. Kirikaeshi (切り返し) — Sequência de golpes para ritmo, precisão, respiração e resistência.
     5. Uchikomi-geiko (打込稽古) — Execução de golpes em oportunidades oferecidas pelo motodachi.
     6. Kakari-geiko (掛稽古) — Ataques contínuos e intensos durante períodos curtos.
     7. Yakusoku-geiko (約束稽古) — Exercícios combinados com ações previamente definidas.
     8. Waza-geiko (技稽古) — Técnicas ofensivas e contra-ataques (debana, nuki, kaeshi, suriage, hiki-waza).
     9. Oji-waza (応じ技) — Técnicas de resposta direta ao ataque do oponente.
     10. Ji-geiko (地稽古) — Combate livre aplicando fundamentos e técnicas.
     11. Shiai-geiko (試合稽古) — Simulação de luta com regras, arbitragem e pontuação.
     12. Nihon Kendō Kata (日本剣道形) — Formas tradicionais praticadas com bokutō.
     13. Bokutō ni yoru Kendō Kihon Waza Keiko Hō (木刀による剣道基本技稽古法) — Fundamentos técnicos com espada de madeira.
     14. Shinsa (審査) — Avaliação para exame de graduação (postura, técnica, etiqueta, kiai e zanshin).
   - Avaliação dos 3 Pilares:
     • Movimentação (35%): Postura (Shisei), nivelamento de ombros, base de pés (Ashi) e amplitude.
     • Precisão (35%): Trajetória no alvo, sincronismo Ki-Ken-Tai-Ichi e controle de linha central.
     • Constância (30%): Regularidade do ritmo, resistência à fadiga e adequação à cadência.
   - Rastreamento Individual e Nomeação: Nomeação interativa por Kenshi com emissão de Relatório Individual em Markdown (.md) e dados da sessão em JSON.
   - Diagnósticos Pedagógicos: Pontos Fortes, Pontos de Atenção e Prescrições Práticas de Exercícios do Dojo.

3. MODO DE DETECÇÃO EM TEMPO REAL (realtime):
   Processamento instantâneo via Webcam ou streams RTSP/RTCP de 1 a 4 câmeras IP simultâneas.
   - Conta com validação de consenso multi-câmeras em background baseada no conjunto
     de imagens/frames (com quórum escalonado e descarte silencioso de falsos positivos).
   - Painel do Contador de Pontos no topo da lista: placar dinâmico com contagem de Ippons
     e total de golpes para Shiro (⚪) e Aka (🔴), além do total consolidado da sessão.
   - Lista de golpes em ordem invertida: os golpes mais recentes aparecem sempre no topo,
     com cards detalhados dos 4 pilares de Ki-Ken-Tai-Ichi e relatório técnico expansível em HTML.

-------------------------------------------------------------------------------
5. PERFIS DE CALIBRAÇÃO (SENSIBILIDADE)
-------------------------------------------------------------------------------

- Iniciantes / Educacional (Permissivo):
  Limiar mínimo: 45%. Foco na precisão ao alvo, tolerando pequenas falhas
  de postura e sincronismo.
- Treino Geral / Keiko (Normal):
  Limiar mínimo: 65%. Equilíbrio ideal entre precisão técnica e tolerância.
- Campeonato / Audit de Dan (Rígido):
  Limiar mínimo: 82%. Exige execução perfeita de Ki-Ken-Tai-Ichi e Zanshin.
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
