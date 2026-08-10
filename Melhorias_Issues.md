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

## 🚀 Outras Melhorias & Funcionalidades

- **Menu de Configurações Centralizado**: Criar um menu/painel de configurações para centralizar todas as configurações gerais do sistema:
  - **Processamento Hardware**: Seleção entre modo **CPU (exclusivo)** ou **GPU (quando presente)** para aceleração de detecção de pose e inferência de IA.
  - **Calibração & Limiares**: Escolha e ajuste fino dos perfis de arbitragem e critérios técnicos.
  - **Câmeras & Rede**: Parâmetros de suporte ao protocolo RTCP/RTSP para múltiplas câmeras.
  - **Interface & Preferências**: Opções visuais e de exibição do dashboard.
- **Identificação do Kenshi**: Identificar e diferenciar individualmente o Kenshi (ex: Aka / Shiro) que teve o golpe validado.
- **Testes Automatizados**: Expansão da criação de testes automatizados da aplicação.
- **Licença & Disclaimer (Open Code)**: Incluir disclaimer legal oficial de Licença de Código Aberto (*Open Code / Open Source License*), definindo o uso livre para estudo, modificação e contribuição comunitária, com isenção de responsabilidade (*AS IS*).

---

### ✅ Concluídas

- **Modo de Arbitragem Gravada**: Definição e consolidação do modo atual do sistema.
- **Layout Responsivo Web**: Manter o vídeo em uma coluna fixa (sticky) e a lista de golpes/relatório com barra de rolagem ao lado (v1.2.1).

---

## 🐛 Issues

- Usar a GPU quando disponível para a detecção.
- O vídeo com as marcações não está funcionando.


