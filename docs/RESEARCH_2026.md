# Engenharia de desempenho: BrawlIA 0.11

Este plano combina técnicas públicas e código original; não há acesso a
pesos privados, raciocínios internos de outros modelos ou engenharia reversa de
componentes proprietários. Implementações externas exigem verificação de
licença e proveniência. A implementação atual permanece somente leitura.

## Evidência anterior: a prioridade é percepção

Sessão V0.10 de 06/11/2026, 1510 análises, 28,61% de cobertura do
**candidato** a jogador em combate, 361 trilhas temporárias. O detector de
cores/cenário consumiu em média 36,88 ms; associação de IDs, 0,32 ms.
Essas métricas não são taxa de acerto. Não há dataset rotulado suficiente
para afirmar ganho de precisão.

## Implementado

1. Verde luminoso de nome: quando falha a busca normal, tenta hue >=64 e
   value >=200, respeitando limites do verde configurado. Isola os glifos
   das regiões grandes de vegetação. Ainda exige uma barra e seus glifos
   observados no **mesmo quadro**, nunca cria identidade fictícia.
2. Fallback editável em `configuracoes.json` e diagnóstico explícito.
   Pode ser desligado para comparação A/B sobre o mesmo vídeo.
3. Reutiliza o índice de máscara verde já existente em `FrameEvidence`
   para não repetir a conversão de limiar na busca de barras verdes.
4. Latência ONNX offline separada em preprocess / model_runtime /
   postprocess; o campo legado `inference_ms` continua sendo o total.
5. `qa/benchmark_player.py` executa detector de nome com e sem fallback
   nos mesmos frames. Não mede acurácia sem rótulos humanos.

## Pesquisa, sem copiar código externo

- ONNX Runtime DirectML: GPU AMD compatível com DX12, mas DirectML
  continua em manutenção; **Windows ML** é a escolha recomendada para
  projetos nativos novos. No Python atual manteremos DirectML como opção
  e mediremos CPU vs GPU em hardware real:
  https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
  https://onnxruntime.ai/docs/get-started/with-windows.html
- Ultralytics ByteTrack associa detecções de alta confiança e depois
  usa detecções de menor confiança para **recuperar trilhas existentes**;
  BoT-SORT adiciona compensação da câmera e ReID opcional:
  https://docs.ultralytics.com/modes/track
- PylaAI usa arquitetura modular e ONNX; código sob CC BY-NC.
  Jooi025/BrawlStarsBot usa MIT. Nenhum código/peso desses projetos foi
  incorporado nesta melhoria.

## Ordem de experimentos

1. Reproduzir vídeos existentes com e sem fallback. Exigir análise de
   100+ quadros em vegetação, 100+ fora dela e rótulos de barra/nome.
2. Identificar falsos positivos no HUD antes de alterar limites de área.
3. Avaliar associação em duas etapas (estilo ByteTrack), usando
   somente observações presentes, sem baixar limiares indiscriminadamente.
4. Medir 15 FPS análise no PC-alvo com p95 de cada etapa, sem vídeo e
   depois com vídeo/PNG para quantificar a sobrecarga.
5. Opcional: avaliar detector próprio em ONNX no mesmo dataset para
   medir precisão, recall e latência separadamente.

Critérios de avanço: nenhuma regressão na segurança do modo somente
leitura, maior recall sem piora inaceitável de falsos positivos e p95
total dentro de 66,7 ms para 15 FPS. O último critério é um alvo de
latência, não um resultado já atingido.
