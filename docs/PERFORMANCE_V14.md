# BrawlIA V1.4 — visão e latência, OCR auditável

A V1.3 recebeu 559 análises em cerca de 44 s, com média de 12,61 análises/s
e 261 quadros acima do orçamento de 33,3 ms. Durante combate, a etapa de
cores/cenário foi o gargalo medido. Além disso, 0/10 nomes foram confirmados
na apresentação; isso **não** comprova que o motor OCR estava ausente.

## Implementado

1. `TerrainObserver`: leitura do terreno separada das barras. A V1.4
   atualiza contornos a cada três quadros analisados ou no máximo 180 ms,
   aplicando transformações da câmera quando há fluxo válido. Contornos
   reutilizados são **aproximados** e não podem confirmar identidades ou HP.
2. Divisão dos tempos do detector entre `vision_masks`,
   `vision_other_bars`, `vision_terrain` e `vision_player` em JSONL.
   Permite descobrir qual fração dos ~69 ms deve receber a próxima otimização.
3. `MatchRoster`: diagnóstico por cartão do OCR, quantidade de chamadas
   vazias, baixa confiança, confirmações e tempo das leituras. A triagem
   da apresentação é limitada a uma avaliação a cada 150 ms.
4. `DxcamWindowCapture` opcional: para testar captura por Desktop Duplication
   no Windows. Se falhar, retorna ao MSS, registrando o backend. DXcam não
   substitui MSS na configuração padrão sem benchmark no PC real.
5. Benchmarks offline e regressões para terreno, OCR e fallback de captura.

## Testes no PC

No perfil padrão, a captura permanece MSS, sem alteração da interface.
Para experimentar o backend alternativo, instale:

```powershell
.\.venv-v1\Scripts\python.exe -m pip install dxcam
$env:BRAWLIA_CAPTURE_BACKEND = "dxcam"
.\.venv-v1\Scripts\python.exe app.py
```

Na próxima sessão, `analise.jsonl` registra `capture.source_backend` como
`mss` ou `dxcam`. Para voltar à captura anterior:

```powershell
Remove-Item Env:BRAWLIA_CAPTURE_BACKEND -ErrorAction SilentlyContinue
```

Compare as sessões no mesmo local/mapa, com condições semelhantes,
antes de afirmar ganho de FPS ou estabilidade.

Para avaliar somente o custo da percepção, use:

```powershell
.\.venv-v1\Scripts\python.exe qa\benchmark_v14.py "CAMINHO\analise.mp4" --limit 150
```

O benchmark gera `benchmark_v14.json`. Não envolve nova gravação,
não coleta PNGs e não envia comandos ao jogo.

**Limites:** o detector neural ainda não foi integrado à análise principal;
OCR de nomes parcialmente cortados continua sujeito a falhas;
30 FPS é meta configurada, não desempenho garantido. A calibração da
amostragem temporal deve ser revista se os contornos se deslocarem
visivelmente com a câmera.
