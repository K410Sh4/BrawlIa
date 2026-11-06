# Arquitetura e contratos

- `capture.py`: WindowCapture/MSS no próprio worker, ou screenshot ADB. Não controla o jogo.
- `stream.py`: pacote com sequência/tempo, último frame disponível, redimensionamento e callbacks. Quadros substituídos antes de analisar são contabilizados.
- `config.py`: dataclasses imutáveis, validação, migração de opções removidas e valores padrão. `configuracoes.json` deve coincidir com os padrões.
- `health.py`: faixas HSV e suporte de glifos, cache por quadro, recipiente e preenchimento. Não faz OCR. A largura é estimada; preenchimento não equivale a HP numérico.
- `player.py`: nome verde, assinatura visual, continuidade e busca local com retomada global imediata. Ausência não é preenchida por previsão.
- `state.py`: HUD e telas terminais por templates/política temporal. Ausência temporária de HUD pode manter combate incerto por prazo limitado.
- `motion.py`: prepara fluxo óptico/câmera uma vez por quadro, antes da associação. Depois mede consistência corpo/barra usando os IDs do tracker. Sem ID (API isolada), mantém associação geométrica de compatibilidade.
- `tracking.py`: transforma posições anteriores pela câmera válida antes de associar; retorna somente observações atuais. Velocidade de associação é relativa à câmera quando disponível. `screen_motion_px_per_s` permanece velocidade bruta de tela. Ao mudar entre referência de câmera/tela, a velocidade de associação é reiniciada.
- `engine.py`: orquestra estado, câmera, busca visual, classificação, associação e movimento do corpo. `timings_ms.camera_motion`, `association` e `body_motion` são independentes. `tracking` é alias legado de `association`, não soma com ele. `core`/`total` já incluem as etapas e não devem ser somados a elas.
- `observations.py`: fila de eventos sem imagens, ordem preservada, limites explícitos, métricas sobre o histórico completo.
- `recording.py`: vídeo em fila substituível, encaixe sem deformar, mapa frame/vídeo. A API isolada conserva modo legado de vídeo+histórico; o controlador usa modo vídeo+mapa e histórico separado.
- `samples.py`: fila de três PNGs nativos, amostragem periódica/rajadas, previsão correspondente. Deduplicação por ID, sem criar repetições do vídeo.
- `session.py`: inicia pastas/gravadores antes da captura; encerra pipeline antes de drenar gravadores; consolida erros e contagens.
- `provenance.py`: configuração efetiva com hash SHA-256 e versões de Python/bibliotecas.
- `dataset.py` / `review_ui.py`: revisão humana, geometria válida, relações barra/dono e exportação por sessão.
- `evaluation.py`: avaliação de barras em PNGs revisados, sem confundir presença de candidato com precisão.
- `app.py`: Tk, seleção de fontes, controles e eventos por mailbox. Nenhum worker acessa diretamente variáveis Tk.

## Tempo e câmera

Timestamps são segundos monotônicos da captura ou do vídeo, nunca tempo de calendário para velocidade/TTL. Um pacote e sua análise têm o mesmo ID. Uma transformação de câmera só é usada se válida, finita e compatível com a resolução. Falta de consenso usa associação de tela; não comprova imobilidade.

`affine_original` é matriz 2×3 em pixels da análise. `affine_thumbnail` permanece em pixels da miniatura. Não aplique deslocamento de miniatura diretamente à imagem original. A mesma transformação pode projetar trilhas não observadas entre quadros, mas elas não são devolvidas como detecções.

## Histórico e vídeo

`analise.jsonl` é autoridade para contagens de análises; `video_frames.jsonl` é a autoridade para a correspondência exata com o vídeo. Relacione por `frame_id` e valide o `session_id`. Quadros de vídeo podem se repetir; registros do histórico não são repetidos para preencher FPS.

Fila cheia/espaço insuficiente no histórico gera erro explícito, em vez de descartar o evento mais antigo. Quando a escrita falha, `accepted`, `records`, `rejected`, `error` e `complete` tornam a incompletude visível. O callback seguinte interrompe a leitura; se não houver callback seguinte, a finalização ainda consolida o erro.

Os relatórios são gerados pelo histórico, não pela contagem de frames codificados. As métricas são inferências do detector, não rótulos humanos. O vídeo desligado não desliga o histórico.
