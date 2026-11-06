# Validação da primeira publicação 0.10.0

## Ambiente local

173 testes unitários e de integração passaram no Linux/Python 3.12, com as dependências fixadas em `requirements.txt`. Ruff, Black e `pip check` aprovados. São os 154 casos anteriores e 19 regressões novas de configuração, cache, câmera, histórico e dados.

Casos novos incluem: faixas HSV com efeito real, aviso de opções antigas, hash efetivo estável, cache com semântica local preservada, busca local com retomada global, pan maior que o gate mantendo ID, câmera acumulada durante ausência sem emitir detecção, matriz inválida rejeitada, tempos separados, fila de histórico em ordem, sobrecarga explícita, limite de bytes, vídeo lento sem perder histórico, histórico com vídeo desligado e relações de dono na exportação.

O teste de vídeo lento bloqueia a abertura do codificador enquanto a análise continua. Ele exige que o histórico contenha todas as análises e registre zero rejeições, mesmo com substituições na fila de vídeo. O teste de limite não declara completude quando uma escrita não foi concluída.

O fluxo Tk em tela virtual Linux usa uma fonte artificial: calibração, detecção de preenchimento baixo, vídeo reaberto, PNGs, revisão humana, vínculo barra/dono e histórico completo. Não é captura real do BlueStacks.

## Diagnóstico offline

As duas versões receberam os mesmos 359 quadros registrados de um vídeo já anotado, reconstruídos segundo seu mapeamento. Em ambas houve 288 quadros em combate, 270 com candidato a jogador e um único ID de jogador. Tempo médio sem desenho neste ensaio: aproximadamente 35,3 ms na 0.10 e 36,8 ms na 0.9.

É um único ensaio diagnóstico, não uma demonstração estatística de ganho de desempenho. Overlays, compressão e o ambiente offline mudam os resultados; esses números não são diretamente comparáveis ao relatório da captura original nem medem precisão. Não se publicam o vídeo, os PNGs de partidas ou seus registros.

## CI e limites

O workflow executa os testes em Windows/Linux, com Python 3.11/3.12 e dependências fixadas. O estado efetivo das execuções aparece em Actions; a existência do workflow não equivale a dizer que uma execução passou.

Sem rede treinada, pesos de modelo, OCR numérico de HP ou benchmark GPU. A avaliação nova requer PNGs revisados e mede somente localização de barras. Não se treinou ou avaliou modelo sem dados rotulados. Fontes Win32/MSS/ADB reais e desempenho ao vivo ainda exigem confirmação no PC de uso.
