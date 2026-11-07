# 1.2.0

- Cadastro de participantes P01–P10 na apresentação antes do combate.
- OCR assíncrono via RapidOCR opcional, duas leituras concordantes para
  nomes confirmados; nomes ausentes permanecem pendentes.
- Associação conservadora de nomes ao jogador local, separando
  participant_id da trilha transitória track_id.
- Campos match_roster, participant_id e status de identificação em JSONL.
- Interface mostra quantidade de nomes confirmados e ID local quando verificado.
- Sem automação de partidas; fallback de visão existente preservado.

# 1.1.0

- Novo botão "Abrir gravação" no painel, apontando para a sessão ativa,
  a última sessão local após reiniciar, ou o diretório geral vazio.
- Abertura direta da pasta com Explorer (Windows) ou gerenciador do sistema,
  sem executar shell e sem carregar caminhos externos ao BrawlIA.
- Melhorias no pipeline de captura e instrumentação de latência; novos testes
  de regressão e manutenção do modo exclusivamente observador.

# 1.0.0

- Nova interface de captura de um botão; seleção automática da janela do jogo.
- FPS adaptativos de 5–60 com base no custo real da análise e captura.
- Registro de diagnóstico adaptativo por sessão, vídeo e PNGs por padrão.
- Núcleo de percepção heurística preservado até comparação com modelo treinado.
- Modo somente leitura, sem automação de partidas.

# 0.12.0

- Identidade do jogador agora ancorada por assinatura visual estável do
  nome acima da barra e evidência de HP atual, em vez de mudanças de HP.
- Reidentificação conservadora, limitada por tempo, distância e confiança.
- Quando existem nomes visuais ambíguos no mesmo quadro, o rastreador
  não atribui o mesmo ID a múltiplos objetos.
- Diagnóstico e testes regressivos; ainda sem OCR para nomes literais.

# 0.11.0

- Novo fallback de nome do jogador em vegetação por verde luminoso, com
  limiar configurável e validação obrigatória da barra abaixo do nome.
- Menos trabalho duplicado: reutilização da máscara verde já calculada.
- Diagnósticos de fallback e tempos separados no avaliador ONNX offline.
- Experimentos baseados nos vídeos de 06/11/2026; resultados visuais
  são hipóteses de melhoria, não precisão comprovada.
- Testes de regressão e isolamento do modo somente leitura.

# 0.10.0

Primeira publicação deste projeto no GitHub, baseada na versão local 0.9.

- Configuração efetiva centralizada para cores/recipiente/glifos; migração explícita de opções antes sem efeito.
- Máscaras por quadro, consultas de glifos reutilizadas e busca local do jogador com retomada global.
- Preparação de câmera antes da associação; movimento do corpo vinculado aos mesmos IDs.
- Tempos de câmera/associação/corpo separados, preservando alias legado de tracking.
- Histórico independente da fila substituível do vídeo, inclusive com vídeo desligado; limites e falhas explícitos.
- Mapa de vídeo por frame_id separado do histórico completo.
- Configuração completa, hash e ambiente gravados na auditoria e resumo.
- Relações barra/dono na revisão e exportação; conjuntos vazios e barras sem dono reportados.
- Avaliação de barras sobre PNGs humanos revisados e comparação de configurações.
- Dependências de runtime fixadas, testes Windows/Linux e documentação.

Sem modelo treinado, OCR numérico de HP ou controle do jogo. Precisão em partidas novas exige rótulos e validação reais.
