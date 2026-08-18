# Melhorias & Issues

## 🎯 Visão da Versão Final: 3 Modos de Operação (Nodos)

A versão final do **Shinpanai** será organizada em **3 Nodos / Modos Principais de Operação**:

1. **Modo de Detecção em Tempo Real**
   - **Processamento Ao Vivo**: Processamento e detecção instantânea de golpes via transmissão ao vivo (Webcam / Câmeras de transmissão).
   - **Suporte Multi-Câmera (RTCP)**: Incluir suporte ao protocolo RTCP/RTSP para integração de múltiplos ângulos de câmera.
   - **Sinalização Instantânea**: Exibição em tempo real dos pontos válidos durante a luta.

3. **Modo de Treinamento & Aprendizado**
   - **Análise Técnica e Exercícios**: Incluir modo de treino de Kendo onde a IA fornecerá dicas de melhorias nas técnicas e recomendará exercícios de desenvolvimento específicos para o Kenshi.
   - **Avaliação de Exame de Graduação**: Módulo específico para simulação e avaliação diagnóstica de exames de graduação (Kyu / Dan), testando a conformidade com os exigentes critérios de cada nível.
   - **Seleção de Graduação (Dan) na Revisão por Reforço**: Seleção da graduação (Dan) do revisor no painel de feedback/aprendizagem por reforço para calibrar o grau de rigor e exigência dos treinamentos e diagnósticos.
   - **Acompanhamento Biomecânico**: Avaliação detalhada de postura, sincronismo (*Fumikomi*) e *Zanshin*.

---

## 🚀 Melhorias & Funcionalidades por Módulo

### 1. Confiança, Explicabilidade e Revisão Humana
- **Estado “Inconclusivo”**: Não forçar uma decisão quando o vídeo estiver obstruído, desfocado ou sem ângulo suficiente.
- **Comparação Lado a Lado**: Mostrar o golpe analisado junto de um exemplo técnico de referência.
- **Revisão Manual Quadro a Quadro**: Permitir que o revisor ajuste o instante exato do impacto e confirme ou altere a decisão da IA.

### 2. Qualidade e Preparação do Vídeo
- **Diagnóstico Automático Antes da Análise**: Verificar resolução, FPS, iluminação, estabilidade, oclusões e visibilidade dos Kenshi antes de iniciar o processamento.
- **Assistente de Posicionamento de Câmera**: Orientar altura, distância e ângulo recomendados antes da gravação ou transmissão.
- **Estabilização e Correção de Imagem**: Corrigir tremores, distorção de lente, contraste e baixa iluminação.
- **Sincronização Multicâmera**: Alinhar automaticamente transmissões por timestamp, áudio ou evento visual.
- **Detecção de Câmera Inadequada**: Alertar quando o ângulo não permitir avaliar corretamente determinado critério técnico.

### 3. Arbitragem e Regras (Modo Live & Geral)
- **Detecção de Área e Limites do Shiai-jo**: Identificar saídas de quadra, posição relativa dos atletas e eventos próximos às bordas.
- **Detecção de Infrações**: Evoluir o sistema para identificar possíveis *Hansoku*, empurrões irregulares, quedas e outras ocorrências.
- **Controle do Estado da Luta**: Manter placar, tempo decorrido, prorrogação (*Encho*), penalidades e linha do tempo de eventos em ordem cronológica.
- **Análise Contextual**: Diferenciar um golpe isoladamente correto de uma ação ocorrida após interrupção, fora da área ou em condição inválida.

### 4. Identificação e Rastreamento dos Kenshi (Aka / Shiro)
- **Cadastro Opcional de Perfil Técnico**: Manter histórico, graduação, lateralidade e evolução do Kenshi (mediante consentimento).

### 5. Modo de Treinamento & Evolução do Praticante
- **Plano de Treino Personalizado**: Gerar exercícios direcionados com metas, frequência, dificuldade e critérios mensuráveis.
- **Evolução Longitudinal**: Comparar sessões ao longo do tempo e exibir tendências de postura, velocidade, precisão e sincronismo.
- **Metas por Graduação**: Adaptar exercícios e nível de exigência ao Kyu/Dan pretendido.
- **Biblioteca Técnica**: Organizar exemplos de golpes, erros comuns e exercícios categorizados por fundamento.
- **Feedback com Prioridade**: Limitar os apontamentos de cada sessão aos erros mais relevantes para não sobrecarregar o praticante.
- **Comparação Antes/Depois**: Exibir clipes equivalentes de diferentes sessões lado a lado.
- **Modo Instrutor**: Permitir que o Sensei revise, comente e aprove as recomendações e diagnósticos gerados pela IA.

### 6. Aprendizagem por Reforço & Governança dos Modelos
- **Separação entre Feedback e Treinamento**: Garantir que uma correção individual de usuário não altere imediatamente o comportamento global do modelo.
- **Validação do Revisor**: Ponderar o feedback considerando graduação, experiência, consistência e quantidade de avaliações (não apenas o Dan).
- **Consenso entre Revisores**: Encaminhar casos controversos para validação de múltiplos avaliadores.
- **Versionamento de Modelos e Calibrações**: Registrar a versão exata do modelo e perfil utilizado para produzir cada decisão.
- **Rollback de Versão**: Permitir reverter para um modelo ou perfil de calibração anterior caso uma atualização piore os resultados.
- **Conjunto de Validação Fixo**: Avaliar cada nova versão do modelo com um dataset de vídeos representativos antes de publicar.
- **Métricas por Cenário**: Medir a precisão desagregada por tipo de golpe, câmera, iluminação, graduação e nível de oclusão.
- **Detecção de Viés**: Investigar diferenças de desempenho relacionadas a equipamento, biotipo, velocidade ou ambiente.

### 7. Relatórios, Exportação e Interoperabilidade
- **Relatórios Comparativos**: Comparar desempenho entre Kenshi, sessões de treino, exames e versões do modelo.
- **Exportação Estruturada**: Oferecer formatos CSV/JSON além do PDF para pesquisas e integrações externas.
- **Compartilhamento Controlado**: Gerar links temporários com permissões granulares de visualização, comentário ou edição.
- **API Documentada**: Facilitar integração com sistemas de campeonatos e plataformas de treinamento de terceiros.
- **Pacote de Evidências**: Exportar pacote contendo clipe de vídeo, frames relevantes, métricas, decisão tomada e versão do modelo utilizada.

### 8. Operação, Segurança e Privacidade
- **Consentimento e Retenção de Dados**: Definir regras claras de quem pode enviar vídeos, tempo de armazenamento e exclusão.
- **Criptografia e Controle de Acesso**: Proteger vídeos, perfis dos praticantes e avaliações.
- **Logs de Auditoria**: Registrar alterações de configuração, intervenções humanas e mudanças de resultado.
- **Recuperação de Processamento**: Retomar análises interrompidas sem necessitar reprocessar o vídeo inteiro.
- **Fila e Estimativa de Processamento**: Exibir progresso, tempo estimado restante e consumo de recursos (CPU/GPU).
- **Monitoramento Operacional**: Acompanhar erros de runtime, consumo de memória, latência e falhas em streams de câmera.

### 9. Acessibilidade e Experiência de Uso (UX)
- **Internacionalização (i18n)**: Suporte a Português, Japonês e Inglês, com padronização da terminologia técnica do Kendo.
- **Atalhos de Teclado**: Navegação rápida entre golpes, frames e decisões de arbitragem.
- **Modo de Alto Contraste e Daltonismo**: Garantir que a interface não dependa exclusivamente das cores Aka/Shiro.
- **Tutorial Interativo**: Onboarding guiado apresentando os 3 modos e orientando a primeira análise.
- **Perfis de Usuário**: Níveis de acesso diferenciados para Atleta, Instrutor, Árbitro, Pesquisador e Administrador.
- **Salvamento Automático**: Preservar análises e revisões em tempo real contra perdas acidentais.

### 10. Configurações Gerais do Sistema
- **Calibração & Limiares**: Escolha e ajuste fino dos perfis de arbitragem e critérios técnicos.
- **Câmeras & Rede**: Parâmetros de suporte ao protocolo RTCP/RTSP para múltiplas câmeras.
- **Interface & Preferências**: Opções visuais e de exibição do dashboard.
- **Testes Automatizados**: Expansão contínua da cobertura de testes unitários, de integração e e2e da aplicação.

---

## 🐛 Issues & Bugs Conhecidos

- **Processamento & Hardware**:
  - Vazamento de memória (memory leak) durante o processamento de vídeos longos ou transmissões ao vivo.
- **Vídeo, Marcações e Sincronização**:
  - O vídeo com as marcações sobrepostas não está funcionando corretamente.
  - Dessincronização entre o vídeo original, as marcações e os clipes gerados.
  - Divergência entre os timestamps no frontend e os números de frames analisados no backend.
  - Tratamento insuficiente de vídeos com FPS variável, rotações de orientação ou codecs diversos.
  - Erros na captura e aquisição de imagens em tempo real via webcam e transmissões RTSP.
  - Perda de conexão e dessincronização em transmissões de múltiplas câmeras via RTSP.
- **Rastreamento de Atletas & Plano de Fundo**:
  - Falha na persistência ou troca acidental de identidade entre os Kenshi Aka e Shiro durante a luta.
- **Interface & Feedback de Usuário**:
  - Interface sem indicação clara de status de processamento, mensagens de erro ou estado de análise incompleta.

---

## ✅ Concluídas


Versão 1.6.0
   -Melhorias Aplicadas:
      - **Processamento Hardware (CPU vs GPU - v1.6.0)**: Seleção dinâmica e otimização entre modo CPU (exclusivo) e GPU NVIDIA CUDA (quando presente) para aceleração de detecção de pose e inferência de IA com fallback automático no Menu de Configurações.
      - **Delimitação da Luta por Sonkyō (v1.6.0)**: Garantir que golpes válidos sejam considerados estritamente entre o *sonkyou* inicial e final da luta.
      - **Correção e Inversão Manual de Identidade (Aka ⇄ Shiro - v1.6.0)**: Botão de ação rápida na interface e controle no pipeline para inversão e reatribuição da pontuação, eventos e relatórios entre Kenshi Aka (Vermelho) e Kenshi Shiro (Branco).
      - **Detecção e Scoring (Modo de Arbitragem Gravada - v1.6.0)**: Validação de *Yuko-Datotsu* com score ponderado (*Ki-Ken-Tai-Ichi*: impacto no alvo, sincronismo de *Fumikomi*, postura e *Zanshin*), corte automático de clipes, relatórios diagnósticos de combate e navegação integrada com salto temporal calibrado para 1 segundo antes do evento.
      - **Detecção de Sonkyō & Delimitação Temporal da Luta (v1.6.0)**: Identificação e verificação automática da postura ritualística de *Sonkyō* (agachamento profundo sobre os calcanhares, flexão de joelhos e coluna ereta) para marcação do Início Oficial (`match_start_frame`) e Encerramento Oficial (`match_end_frame`) da luta no Modo de Arbitragem Gravada.
      - **Filtragem Estrita de Golpes por Sonkyō (v1.6.0)**: Consideração e avaliação técnica de Yuko-Datotsu (Ippon) realizada **estritamente entre os momentos de Sonkyō de início e término**, descartando movimentações e cortes fora da janela regulamentar de combate.
      - **Rastreamento Focado nos 2 Kenshi Principais (v1.6.0)**: Associação e acompanhamento contínuo dos 2 atletas que executaram o Sonkyō inicial de abertura no Shiaijo (`Kenshi Aka - Vermelho` e `Kenshi Shiro - Branco`).
      - **Filtragem de Planos Diferentes & Descarte de Ruídos Visuais (v1.6.0)**: Calibração automática da escala geométrica do plano principal de combate, descartando elementos de segundo plano (outras lutas ao fundo, árbitros distantes, arquibancadas) e oclusões de primeiro plano (pessoas passando na frente da câmera).
      - **Interface & HUD de Sonkyō e Planos (v1.6.0)**: Exibição no painel de resumo de combate dos timestamps de início e fim da luta por Sonkyō, tempo líquido efetivo de combate, contador de planos descartados, identificação do atacante em cada golpe e renderização no vídeo anotado com HUD superior e bounding boxes diferenciados para Aka e Shiro.
      - **Suíte de Testes Automatizados de Sonkyō e Planos (v1.6.0)**: Criação de `tests/test_sonkyo_and_plane_filtering.py` com cobertura completa para classificação da postura de Sonkyō, cálculo de limites da luta, filtragem de golpes fora da janela regulamentar, classificação de planos (fundo e transeuntes) e integração de ponta a ponta no pipeline.
   - Issues Solucondas
      - Issue de performance quando está no modo GPU: Ganho de velocidade/FPS insuficiente no modo GPU necessitando otimização fina de pipeline e carregamento de tensores em VRAM.
      - Falta de fallback automático quando a GPU não suporta o modelo ou fica sem memória (OOM).
      - Resultados não reprodutíveis entre execuções em CPU e GPU.

Versão 1.5.0
   -Melhorias Aplicadas:
    - **Processamento de Vídeo com Múltiplas Câmeras (RTSP)**: Suporte nativo a múltiplas fontes de vídeo simultâneas em tempo real via protocolo RTSP (Real Time Streaming Protocol).
    - **Aquisição de Vídeo em Tempo Real (Webcam)**: Captura e processamento em tempo real via webcam (`/dev/video0`) além do upload de arquivos locais.
    - **Interface Gráfica Unificada (SHINPANAI Hub)**: Dashboard centralizado para gerenciamento de câmeras, transmissões ao vivo, uploads locais e painel de arbitragem.
    - **Sincronização de Múltiplas Câmeras (Sync)**: Ferramentas e lógica de sincronização para combates com múltiplas perspectivas (ex: câmera frontal + câmera lateral).
    - **Arbitragem Assistida em Vídeo**: Processamento e análise biomecânica de vídeos pré-gravados de combates com pontuação ponderada (Ki-Ken-Tai-Ichi), cortes de clipes e relatórios diagnósticos no Modo de Arbitragem Gravada.
   - **Aprendizagem por Reforço & Seleção de Graduação (Dan)**: Otimização contínua de perfis de calibração via feedback do usuário (TP, FP, FN) incorporando a seleção de graduação (Dan) do revisor para ajustar dinamicamente a sensibilidade dos limiares.
   - **Sistema de Diagnóstico, Alertas & Log de Debug**: Módulo central de logging (`logger_manager.py`) retendo logs no disco (`logs/shinpanai_debug.log`) e em memória, Seção 4 no menu de configurações com monitoramento em tempo real de alertas (com filtro por `ERROR`, `WARNING`, `INFO`, `DEBUG`), registro automático de resets de treinamento, importações/exportações de pacotes JSON e retreinamentos por Dan no log, botão de download do log de debug, ferramenta de diagnóstico automatizado, badges de status de golpe (CONFIRMADO / EDITADO), botões de reset de revisão e suíte de testes unitários (`test_logger_manager.py`).

Versão 1.4.0
   -Melhorias Aplicadas:
      - **Licença & Disclaimer (Open Code)**: Licença oficial de código aberto (GNU General Public License v3.0 em `LICENSE.txt`), definindo uso livre para estudo, modificação e contribuição comunitária com isenção de responsabilidade (*AS IS*).
      - **Modo de Arbitragem Gravada**: Definição e consolidação do modo de análise de vídeos pré-gravados do sistema.
      - **Edição de Golpes por Dan (Modo Gravado)**: Adição do botão para habilitar a edição dos golpes detectados no Modo de Arbitragem Gravada, combo box de seleção da graduação DAN do revisor (Shodan 1º Dan a Hachidan 8º Dan), suporte a confirmação, edição e inclusão de marcações, regra estrita de não exclusão e salvamento com retreinamento automático do modelo.
- **Governança de Treinamento no Menu de Configurações**: Seção de governança no Menu de Configurações contendo contador de treinamentos realizados, nível médio (Dan) dos treinamentos, tabela de quantidade de treinamentos por Dan, opção de apagar treinamento (reset ao estágio inicial), opção de baixar treinamento atual (pacote JSON contendo Dan e data de cada treinamento) e opção de carregar treinamento baixado anteriormente com recalibração imediata.
- **Suíte de Testes Automatizados de Governança por Dan**: Implementação do arquivo `tests/test_dan_training_governance.py` garantindo cobertura de código para todo o fluxo de revisão por Dan, retreinamento e gestão de pacotes de dados.

Versão 1.3.0
   -Melhorias Aplicadas:
      - **Remoção de Vídeo de Demonstração**: Remoção da opção de seleção/geração de vídeo sintético de demonstração da interface, mantendo apenas a análise de vídeos reais via upload ou transmissão ao vivo
      - **Menu de Configurações**: Criação do Menu de Configurações centralizado para gerenciar os parâmetros globais da aplicação.
      - **Seleção de Hardware (CPU vs GPU)**: Implementação do seletor entre modo CPU (exclusivo) e GPU (quando presente) com detecção dinâmica e fallback automático.
- **Mover Seleção de CPU e GPU para Página de Configurações**: Estruturação da navegação multi-páginas, movendo a gestão de hardware para uma página dedicada de configurações.
- **Processamento por GPU (PyTorch CUDA)**: Utilização da GPU NVIDIA quando disponível para aceleração da detecção de pose e inferência dos modelos.
- **Modo de Arbitragem Gravada**: Renomeado modo Usuário para "Modo de Arbitragem Gravada" (análise de vídeos gravados de combates).
- **Modo de Treinamento & Aprendizado**: Renomeado modo Aprendizagem para "Modo de Treinamento & Aprendizado" (reforço, anotação TP/FP/FN e otimização por Dan).
- **Modo de Detecção em Tempo Real**: Implementação da detecção ao vivo via Webcam local ou streams de câmeras IP (RTSP/RTCP) com métricas de FPS e ticker de alertas instantâneos.

Versão 1.2.1
   -Melhorias Aplicadas:
      - **Layout Responsivo Web**: Manter o vídeo em uma coluna fixa (sticky) e a lista de golpes/relatório com barra de rolagem ao lado.













