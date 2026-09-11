# Melhorias & Issues

## 🎯 Visão da Versão Final: 3 Modos de Operação (Nodos)

A versão final do **SenpAI** será organizada em **3 Nodos / Modos Principais de Operação**:

1. **Modo de Detecção em Tempo Real**

2. **Modo de Detecção Gravada**

3. **Modo de Treinamento & Aprendizado**
  - **Treinamento em tempo real**: Permitir analise do treinamento em tempo real
  - **Upload Sem Restrição de Tamanho**: Suporte a uploads de vídeos locais longos de keiko e exames sem limite de tamanho.

---

## 🚀 Melhorias & Funcionalidades por Módulo

### 1. Confiança, Explicabilidade e Revisão Humana
- **Estado “Inconclusivo”**: Não forçar uma decisão quando o vídeo estiver obstruído, desfocado ou sem ângulo suficiente.
- **Comparação Lado a Lado**: Mostrar o golpe analisado junto de um exemplo técnico de referência.

### 2. Qualidade e Preparação do Vídeo
- **Diagnóstico Automático Antes da Análise**: Verificar resolução, FPS, iluminação, estabilidade, oclusões e visibilidade dos Kenshi antes de iniciar o processamento.
- **Assistente de Posicionamento de Câmera**: Orientar altura, distância e ângulo recomendados antes da gravação ou transmissão.
- **Estabilização e Correção de Imagem**: Corrigir tremores, distorção de lente, contraste e baixa iluminação.
- **Sincronização Multicâmera**: Alinhar automaticamente transmissões por timestamp, áudio ou evento visual.
- **Detecção de Câmera Inadequada**: Alertar quando o ângulo não permitir avaliar corretamente determinado critério técnico.
- **Utilização de DLSS para processamento e renderização de vídeo em tempo real**: Utilizar a geração de frames via inteligência artificial para melhorar a qualidade do vídeo e de captação dos golpes.

### 3. Avaliação Técnica e Regras (Modo Live & Geral)
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
- **Atalhos de Teclado**: Navegação rápida entre golpes, frames e decisões de avaliação.
- **Modo de Alto Contraste e Daltonismo**: Garantir que a interface não dependa exclusivamente das cores Aka/Shiro.
- **Tutorial Interativo**: Onboarding guiado apresentando os 3 modos e orientando a primeira análise.
- **Perfis de Usuário**: Níveis de acesso diferenciados para Atleta, Instrutor, Árbitro, Pesquisador e Administrador.
- **Salvamento Automático & Tolerância a Falhas**: Preservar análises, revisões e treinamentos automáticos em tempo real contra perdas acidentais *(Implementado: Checkpoints periódicos atômicos, consolidação preventiva e salvamento de emergência no Treinamento Automático por IA)*.

### 10. Configurações Gerais do Sistema
- **Calibração & Limiares**: Escolha e ajuste fino dos perfis de calibração e critérios técnicos.
- **Armazenamento de Treinamento**: Monitoramento e diagnóstico em tempo real do espaço em disco ocupado pelos datasets de treinamento, histórico de sessões, pesos neurais de IA e bases de conhecimento.
- **Varredura por Câmeras & Rede**: Procurar e conectar a cameras na rede local com protocolo RTCP/RTSP para múltiplas câmeras.
- **Interface & Preferências**: Opções visuais e de exibição do dashboard.
- **Testes Automatizados**: Expansão contínua da cobertura de testes unitários, de integração e e2e da aplicação.

#### 11. Licenciamento e proteção do Ambiente
- **Licenciamento**: Implementação de sistema de licenciamento.
- **Marca d'água no código**: Inclusão de marca d'água no código fonte.
- **Anti-Violação** Proteção contra cópia do código fonte.
- **Anti-Tampering**: Proteção contra adulteração do código fonte.
- **Marca d'agua nos vídeos**: Proteção de vídeos gerados com dados de identificação de movimentação.

### 12. Arquitetura
- **Modularização da Aplicação**: Transformar o código atual em módulos (Monolito -> Microserviços).
- **Padronização de Protocolos de Comunicação**: Padronizar os protocolos de comunicação entre os módulos.
- **Observabilidade**: Implementar sistema de monitoramento.
- **Otimização de Performance**: Otimizar o código para melhor performance.
- **Otimização de Memória**: Otimizar o código para melhor gerenciamento de memória.
- **Otimização de Bateria**: Otimizar o código para melhor gerenciamento de bateria.

### 13. Integrações
- **Integração com Sistemas de Federações**: Integração com sistemas de federações para obtenção de dados de atletas e eventos.
- **Integração com Sistemas de Arbitragem**: Integração com sistemas de arbitragem para obtenção de dados de árbitros e eventos.
- **Integração com Sistemas de Treinamento**: Integração com sistemas de treinamento para obtenção de dados de atletas e eventos.
- **Integração com Sistemas de Arbitragem**: Integração com sistemas de arbitragem para obtenção de dados de árbitros e eventos.

### 14. Controle de usuários 
- **Cadastro de usuários**: Implementação de sistema de cadastro de usuários.
- **Cadastro de Academias**: Implementação de sistema de cadastro de Confederações, Federações, Ligas e academias.
- **Cadastro de Atletas**: Implementação de sistema de cadastro de atletas.
- **Cadastro de Árbitros**: Implementação de sistema de cadastro de árbitros.
- **Cadastro de Instrutores**: Implementação de sistema de cadastro de instrutores.
- **Cadastro de novos Kendocas**: Filiação de novos Kendocas para treinamentos remotamente ou na acabemia mais próxima por dados de GPS.
- **Edição de usuários**: Implementação de sistema de edição de usuários.
- **Exclusão de usuários**: Implementação de sistema de exclusão de usuários.
- **Controle de acessos**: Implementação de sistema de controle de acessos.
- **Controle de permissões**: Implementação de sistema de controle de permissões.
- 

---

## 🐛 Issues & Bugs Conhecidos

- **Processamento & Hardware**:
  - Vazamento de memória (memory leak) durante o processamento de vídeos longos ou transmissões ao vivo.
  - Configuração de limpeza de arquivos temporários
- **Processamento em tempo real**:
  - ~~Delay na recepção de vídeo via RTSP causando falha de sincronização com cameras locais (webcam)~~ *(Resolvido via `ThreadedVideoStream` com buffer size 1 e descarte de frames defasados)*
- **Vídeo, Marcações e Sincronização**:
  - Dessincronização entre o vídeo original, as marcações e os clipes gerados.
  - Divergência entre os timestamps no frontend e os números de frames analisados no backend.
  - Tratamento insuficiente de vídeos com FPS variável, rotações de orientação ou codecs diversos.
  - ~~Erros na captura e aquisição de imagens em tempo real via webcam e transmissões RTSP.~~ *(Resolvido com otimização FFmpeg TCP, `probe_stream_connection` e leitor assíncrono)*
  - ~~Perda de conexão e dessincronização em transmissões de múltiplas câmeras via RTSP.~~ *(Resolvido com reconexão resiliente e threads assíncronas dedicadas)*
- **Rastreamento de Atletas & Plano de Fundo**:
  - Falha na persistência ou troca acidental de identidade entre os Kenshi Aka e Shiro durante a luta.
  - Falha na detecção de elementos de distorção da detecção (Shinpan, Expectadores estáticos ou transitando na frente das cameras, movimentação da camera, etc.)

---

## 📱 Módulo Mobile (SenpAI Companion App)

O **SenpAI Mobile** foi concebido como uma extensão portátil e interativa do ecossistema SenpAI, conectando praticantes (*Kenshi*), professores (*Sensei*) e árbitros (*Shinpan*) ao poder de processamento de visão computacional e IA do sistema.

### 1. Transformação do Smartphone em Câmera Inteligente ("SenpAI Cam")
- **Transmissão RTSP / WebRTC em Alta Performance**: Utilização da câmera do smartphone como nó de captura sem fio de alta taxa de quadros (60/120 FPS), integrando-se automaticamente ao servidor/hub do SenpAI.
- **Assistente de Enquadramento com Realidade Aumentada (AR)**: Guia visual na tela sobrepondo as linhas recomendadas do *Shiai-jo*, distância e ângulo ideais do tripé antes de iniciar a gravação.
- **Modo de Gravação Offline & Sincronização Automática**: Capacidade de gravar treinos locais no dojo sem conexão de internet ativa e realizar upload em segundo plano quando reconectar ao Wi-Fi.
- **Modo Econômico & Gestão Térmica**: Escurecimento de tela e baixo consumo de energia durante gravações e transmissões longas em campeonatos.

### 2. Treinamento Individual & Assistente Pessoal ("Pocket Sensei")
- **Feedback em Tempo Real por Áudio (Bluetooth)**: Instruções e correções instantâneas por voz diretamente no fone de ouvido durante treinos solo de *Suburi* e *Uchikomi* (ex: *"Aumente o Zanshin"*, *"Sincronize o Fumikomi"*, *"Men detectado com 92% de precisão"*).
- **Feedback do Sensei**: Envio do treinamento gravado para o Sensei responsável e obtenção de feedback por voz ou texto.
- **Tabata System**: Sistema de treino intervalado de alta intensidade (HIIT) com suporte a protocolos Tabata (ver aplicativo Tabata Timer Cronometro Tabata).
- **Metrônomo Biomecânico & Contador de Suburi**: Monitoramento de cadência rítmica, altura de elevação do *Shinai* e consistência de postura com contagem automática de repetições.
- **Acesso Completo às Rotinas Propostas de Treino**: Planos de treino personalizados gerados pelo motor de IA com metas diárias, séries recomendadas e vídeos tutoriais de referência técnica.
- **Simulador Interativo de Exames de Graduação (Kyu/Dan)**: Checklist diagnóstico pré-exame com avaliação de postura, rituais de cortesia (*Reiho*), *Kihon* e conformidade biomecânica para cada nível.

### 3. Gamificação, Comunidade & Social Dojo
- **Scoreboard & Rankings do Dojo**: Placares semanais e mensais de volume de treino, regularidade (*streaks*), precisão de golpes e evolução técnica entre os membros do dojo.
- **Cartões de Golpe & Exportação para Redes Sociais**: Geração de cards em vídeo/GIF estilizados (formato Shorts/Reels) com traçado do esqueleto biomecânico, métricas de *Ki-Ken-Tai-Ichi* e selo de validação da IA para compartilhamento direto no Instagram/WhatsApp.
- **Compartilhamento de Sessões com Amigos e Sensei**: Envio de clipes e relatórios com marcações e anotações para análise remota pelo professor do dojo.
- **Painel do Sensei (Gestão de Alunos)**: Interface dedicada para instrutores acompanharem o progresso da turma, atribuírem tarefas de treino personalizadas e deixarem notas por áudio/texto em lances específicos.

### 4. Modo Shinpan & Árbitro de Bolso (Suporte ao Shiaijo)
- **Controle Remoto de Placar e Cronômetro**: Uso do celular como mesa de controle wireless de pontuação (*Ippon*, *Hansoku*, *Encho*, contagem de tempo), sincronizando os eventos diretamente com a gravação de vídeo do SenpAI.
- **VAR de Bolso (Revisão Rápida com Gestos Touch)**: Player otimizado para replay instantâneo com avanço/retrocesso quadro a quadro tátil (*scrubbing* fluido), permitindo revisão rápida de lances duvidosos na lateral da quadra.
- **Notificações Hápticas de Consenso de Câmeras**: Alerta por vibração no celular ou smartwatch quando o quórum de múltiplas câmeras validar um ponto com alta confiança estatística.

### 5. Integração com Wearables & Sensores
- **Suporte a Smartwatch (Apple Watch / Wear OS)**: Leitura de dados de frequência cardíaca, impacto e aceleração do punho integrados à análise visual de *Ki-Ken-Tai-Ichi*.
- **Diário de Bordo & Biometria do Kenshi**: Histórico consolidado de tempo de treino, calorias, fadiga e tempo de reação ao longo de semanas e meses.

---

## 🔬 Calibração de Precisão & Mitigação de Falsos Positivos (Concluído)
- **Reformulação da Acurácia Básica (< 50%)**: Eliminação de valores iniciais inflados (75%-93%). O modelo cru sem treino agora assume baselines honestas de fábrica (32% a 46.5%), refletindo o estado preliminar com alto índice de falsos positivos até calibração pelo auto-treinamento e feedbacks.
- **Priorização de Feedback Real**: Priorização da taxa real de acerto anotada pelos Shinpans (`precision_pct` do `FeedbackManager`) sobre estimativas teóricas quando houver dados reais.
- **Supressão de Disparos Múltiplos (Debounce Temporal de Golpes)**: Elevação do `min_event_gap_frames` do `EventSpotter` de 15 para 35 frames (~1.2s a 30 FPS) e implementação de Non-Maximum Suppression (NMS) temporal, impedindo que oscilações na subida e descida da Shinai no mesmo ataque sejam fatiadas e detectadas como 2 ou 3 golpes.
- **Discriminação de Contato Físico (Maai)**: Verificação de distância relativa entre os combatentes no instante do impacto. Golpes desferidos no ar a distâncias excessivas (> 0.48 de afastamento horizontal) são classificados como fora do Maai (sem contato) e descartados como Ippon, eliminando a inflação desmedida do placar em vídeos de Shiai.
