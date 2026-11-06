# BrawlIA V1 — Captura automática e controle de FPS

## Uma ação

1. Abra o BlueStacks e mantenha o jogo visível, sem janelas cobrindo o emulador.
2. Inicie o BrawlIA pelo `iniciar_windows.bat`.
3. Clique em **Iniciar captura automática**. O programa localiza a janela
   compatível (priorizando Brawl Stars), grava vídeo e histórico, mantém PNGs
   para auditoria e inicia a análise. O mesmo botão para a captura.
4. O caminho das gravações aparece na própria tela. A finalização conclui
   `analise.mp4`, `analise.jsonl`, `sessao.json` e o resumo da sessão.

Não há interação automática com o jogo; é um observador de pixels.
A captura do desktop via MSS requer que a janela esteja visível, não
minimizada e completamente dentro da área dos monitores. Um erro de
captura interrompe a sessão em vez de gravar silenciosamente conteúdo errado.

## FPS inteligente

O controle `brawlla/adaptive.py` mede p90 da duração real da captura, análise
e idade do quadro. A cada 2 segundos ajusta as taxas de captura e análise,
reduzindo rapidamente em carga alta e aumentando apenas com headroom
sustentado. Faixa suportada: **5–60 FPS**. O gravador de vídeo conserva
um FPS nominal fixo (15) para compatibilidade com MP4; os FPS adaptativos
reais e os quadros analisados permanecem registrados no histórico.

Importante: atualmente o detector principal usa **CPU/OpenCV**. A
placa gráfica renderiza o BlueStacks e pode executar modelos via
DirectML/ONNX se um modelo for integrado posteriormente. O controle mede
**capacidade efetiva do processamento**, não percentual de utilização da GPU;
não prometemos 60 FPS nem aumento por GPU sem medições no PC.

## Refatoração

A nova página elimina os seletores de fonte, FPS e gravação da interface
principal. O pipeline mantém a lógica de buffers de frame recente e o
controlador de sessões, gravação e evidências já auditados.

A visão heurística existente permanece como fallback até termos modelos
treinados, dados rotulados e comparação A/B. Esta V1 é o novo fluxo de
captura, não a substituição por uma rede neural treinada.

## Testes

```powershell
python -m unittest discover -s tests -v
```

Verifique `sessao.json` → `pipeline.adaptive_fps`, além de
`capture_fps`, `analysis_fps`, `frame_age_ms` e
`replaced_before_analysis`. Faça dois testes comparáveis, com e sem
gravação de vídeo, antes de otimizar a resolução de análise.
