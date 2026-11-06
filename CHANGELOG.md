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
