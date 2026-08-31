# Melhorias & Issues

## 🎯 Visão da Versão Final: 3 Modos de Operação (Nodos)

A versão final do **SenpAI** será organizada em **3 Nodos / Modos Principais de Operação**:

1. **Modo de Detecção em Tempo Real**

2. **Modo de Detecção Gravada**
   
3. **Modo de Treinamento & Aprendizado**
.

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
- **Salvamento Automático**: Preservar análises e revisões em tempo real contra perdas acidentais.

### 10. Configurações Gerais do Sistema
- **Calibração & Limiares**: Escolha e ajuste fino dos perfis de calibração e critérios técnicos.
- **Câmeras & Rede**: Parâmetros de suporte ao protocolo RTCP/RTSP para múltiplas câmeras.
- **Interface & Preferências**: Opções visuais e de exibição do dashboard.
- **Testes Automatizados**: Expansão contínua da cobertura de testes unitários, de integração e e2e da aplicação.

---

## 🐛 Issues & Bugs Conhecidos

- **Processamento & Hardware**:
  - Vazamento de memória (memory leak) durante o processamento de vídeos longos ou transmissões ao vivo.
  - Configuração de limpeza de arquivos temporários
- **Processamento em tempo real**:
  - Delay na recepção de vídeo via RTSP causando falha de sincronização com cameras locais (webcam)
- **Vídeo, Marcações e Sincronização**:
  - Dessincronização entre o vídeo original, as marcações e os clipes gerados.
  - Divergência entre os timestamps no frontend e os números de frames analisados no backend.
  - Tratamento insuficiente de vídeos com FPS variável, rotações de orientação ou codecs diversos.
  - Erros na captura e aquisição de imagens em tempo real via webcam e transmissões RTSP.
  - Perda de conexão e dessincronização em transmissões de múltiplas câmeras via RTSP.
- **Rastreamento de Atletas & Plano de Fundo**:
  - Falha na persistência ou troca acidental de identidade entre os Kenshi Aka e Shiro durante a luta.

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











