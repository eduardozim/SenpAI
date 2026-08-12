# Melhorias & Issues

## 🎯 Visão da Versão Final: 3 Modos de Operação (Nodos)

A versão final do **Shinpanai** será organizada em **3 Nodos / Modos Principais de Operação**:

1. **Modo de Treinamento (Treino de Kendo & Aprendizado)**
   - **Análise Técnica e Exercícios**: Incluir modo de treino de Kendo onde a IA fornecerá dicas de melhorias nas técnicas e recomendará exercícios de desenvolvimento específicos para o Kenshi.
   - **Avaliação de Exame de Graduação**: Módulo específico para simulação e avaliação diagnóstica de exames de graduação (Kyu / Dan), testando a conformidade com os exigentes critérios de cada nível.
   - **Seleção de Graduação (Dan) na Revisão por Reforço**: Seleção da graduação (Dan) do revisor no painel de feedback/aprendizagem por reforço para calibrar o grau de rigor e exigência dos treinamentos e diagnósticos.
   - **Acompanhamento Biomecânico**: Avaliação detalhada de postura, sincronismo (*Fumikomi*) e *Zanshin*.

2. **Modo de Arbitragem Gravada (Modo Atual)**
   - **Arbitragem Assistida em Vídeo**: Processamento e análise de vídeos pré-gravados de combates de Kendo.
   - **Detecção e Scoring**: Validação de *Yuko-Datotsu* com score ponderado (*Ki-Ken-Tai-Ichi*), corte automático de clipes e geração de relatórios diagnósticos.
   - **Aprendizagem por Reforço & Seleção de Graduação (Dan)**: Otimização contínua de perfis de calibração via feedback do usuário (TP, FP, FN) incorporando a seleção de graduação (Dan) do revisor para ajustar dinamicamente a sensibilidade dos limiares.

3. **Modo de Detecção em Tempo Real**
   - **Processamento Ao Vivo**: Processamento e detecção instantânea de golpes via transmissão ao vivo (Webcam / Câmeras de transmissão).
   - **Suporte Multi-Câmera (RTCP)**: Incluir suporte ao protocolo RTCP/RTSP para integração de múltiplos ângulos de câmera.
   - **Sinalização Instantânea**: Exibição em tempo real dos pontos válidos durante a luta.

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
- **Remoção de Vídeo de Demonstração**: Remover a opção de seleção/geração de vídeo sintético de demonstração da interface, mantendo apenas a análise de vídeos reais via upload ou transmissão ao vivo.

### 3. Arbitragem e Regras (Modo Live & Geral)
- **Detecção de Área e Limites do Shiai-jo**: Identificar saídas de quadra, posição relativa dos atletas e eventos próximos às bordas.
- **Detecção de Infrações**: Evoluir o sistema para identificar possíveis *Hansoku*, empurrões irregulares, quedas e outras ocorrências.
- **Controle do Estado da Luta**: Manter placar, tempo decorrido, prorrogação (*Encho*), penalidades e linha do tempo de eventos em ordem cronológica.
- **Análise Contextual**: Diferenciar um golpe isoladamente correto de uma ação ocorrida após interrupção, fora da área ou em condição inválida.

### 4. Identificação e Rastreamento dos Kenshi (Aka / Shiro)
- **Calibração Inicial Aka/Shiro**: Confirmar visualmente a identificação dos competidores antes de iniciar a análise.
- **Correção Manual de Identidade**: Permitir reajustar a atribuição Aka/Shiro em um trecho específico da gravação.
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
- **Menu de Configurações Centralizado**: Painel para gerenciar os parâmetros globais da aplicação:
  - **Processamento Hardware**: Seleção entre modo **CPU (exclusivo)** ou **GPU (quando presente)** para aceleração de detecção de pose e inferência de IA.
  - **Calibração & Limiares**: Escolha e ajuste fino dos perfis de arbitragem e critérios técnicos.
  - **Gestão & Exportação de Treinamentos**: Permitir exportar os treinamentos realizados com filtro pelo Dan (graduação do aplicador/revisor), além de funcionalidade para resetar o treinamento ao padrão inicial do sistema.
  - **Câmeras & Rede**: Parâmetros de suporte ao protocolo RTCP/RTSP para múltiplas câmeras.
  - **Interface & Preferências**: Opções visuais e de exibição do dashboard.
- **Licença & Disclaimer (Open Code)**: Incluir disclaimer legal oficial de Licença de Código Aberto (*Open Code / Open Source License*), definindo o uso livre para estudo, modificação e contribuição comunitária, com isenção de responsabilidade (*AS IS*).
- **Testes Automatizados**: Expansão contínua da cobertura de testes unitários, de integração e e2e da aplicação.

---

## 🐛 Issues & Bugs Conhecidos

- **Processamento & Hardware**:
  - Issue de performance quando está no modo GPU: Ganho de velocidade/FPS insuficiente no modo GPU necessitando otimização fina de pipeline e carregamento de tensores em VRAM.
  - Falta de fallback automático quando a GPU não suporta o modelo ou fica sem memória (OOM).
  - Resultados não reprodutíveis entre execuções em CPU e GPU.
  - Vazamento de memória (memory leak) durante o processamento de vídeos longos ou transmissões ao vivo.
- **Vídeo, Marcações e Sincronização**:
  - O vídeo com as marcações sobrepostas não está funcionando corretamente.
  - Dessincronização entre o vídeo original, as marcações e os clipes gerados.
  - Divergência entre os timestamps no frontend e os números de frames analisados no backend.
  - Tratamento insuficiente de vídeos com FPS variável, rotações de orientação ou codecs diversos.
  - Perda de conexão e dessincronização em transmissões de múltiplas câmeras via RTSP.
- **Rastreamento de Atletas**:
  - Falha na persistência ou troca acidental de identidade entre os Kenshi Aka e Shiro durante a luta.
- **Interface & Feedback de Usuário**:
  - Interface sem indicação clara de status de processamento, mensagens de erro ou estado de análise incompleta.

---

## 🗺️ Priorização Recomendada (Roadmap de Desenvolvimento)

Para garantir a estabilidade e a qualidade do Shinpanai, o desenvolvimento deve seguir a seguinte ordem de prioridades:

1. **Fase 1: Correção de Pipeline e Estabilização Básica**
   - Corrigir a geração de vídeo com marcações sobrepostas.
   - Estabilizar o pipeline de processamento em CPU e GPU com fallbacks adequados.
2. **Fase 2: Diagnóstico de Qualidade e Confiabilidade**
   - Implementar diagnóstico automático de qualidade de vídeo antes da análise.
   - Introduzir o cálculo de confiança e a classificação de estado "Inconclusivo".
3. **Fase 3: Rastreamento e Revisão Fina**
   - Garantir a identificação persistente e sem trocas entre Aka e Shiro.
   - Implementar a ferramenta de revisão manual quadro a quadro com ajuste do ponto de impacto.
4. **Fase 4: Validação com Especialistas (Dataset)**
   - Implementar bateria de testes com um conjunto fixo de vídeos previamente validados por árbitros experientes.
5. **Fase 5: Governança de IA e Auditabilidade**
   - Introduzir versionamento formal de modelos, logs de auditoria e funcionalidade de rollback de calibrações.
6. **Fase 6: Consolidação do Modo de Treinamento**
   - Consolidar os módulos de treino de Kendo, planos personalizados e simulação de exames de graduação.
7. **Fase 7: Operação em Tempo Real e Multicâmera**
   - Implementar a detecção ao vivo e o suporte a múltiplas câmeras RTSP (dada a alta complexidade de latência, sincronia e resiliência).

---

## ✅ Concluídas

- **Modo de Arbitragem Gravada**: Definição e consolidação do modo de análise de vídeos pré-gravados do sistema.
- **Layout Responsivo Web**: Manter o vídeo em uma coluna fixa (sticky) e a lista de golpes/relatório com barra de rolagem ao lado (v1.2.1).
- **Menu de Configurações**: Criação do Menu de Configurações centralizado para gerenciar os parâmetros globais da aplicação (v1.3.0).
- **Seleção de Hardware (CPU vs GPU)**: Implementação do seletor entre modo CPU (exclusivo) e GPU (quando presente) com detecção dinâmica e fallback automático (v1.3.0).
- **Mover Seleção de CPU e GPU para Página de Configurações**: Estruturação da navegação multi-páginas, movendo a gestão de hardware para uma página dedicada de configurações (v1.3.0).
- **Remoção de Vídeo de Demonstração**: Remoção da opção de geração/seleção de vídeo sintético de demonstração da interface, mantendo apenas a análise de vídeos reais de lutas via upload (v1.3.0).
- **Processamento por GPU (PyTorch CUDA)**: Utilização da GPU NVIDIA quando disponível para aceleração da detecção de pose e inferência dos modelos (v1.3.0).
