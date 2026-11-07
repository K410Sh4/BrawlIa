# BrawlIA — núcleo experimental de reconhecimento V1

## Estado da implementação

A V1.4 continua preservada e executável com iniciar_windows.bat.
O novo núcleo V1 reside em brawlla/recognition_v1/ e roda
independentemente; ainda NÃO substitui a GUI existente e não possui
modelo neural treinado nem controle do jogo.

## Iniciar no Windows

1. Abra o BlueStacks e deixe toda a janela visível.
2. Execute iniciar_reconhecimento_v1.bat.
3. Na janela de prévia, pressione **Q** para encerrar e finalizar o dataset.

Execução direta:

    python run_recognition_v1.py --preview --fps 30 --duration 120

Os dados são gravados localmente em
%LOCALAPPDATA%\Brawlla\datasets\recognition_v1\<sessão>\.
Não são enviados para a internet.

## Componentes

| Arquivo | Função |
|---|---|
| geometry.py | Retângulos e coordenadas globais |
| hud_context.py | Âncoras geométricas de controles, contexto temporal |
| silhouette.py | Assinaturas leves de silhueta (heurísticas) |
| context_engine.py | Tracks, velocidade e diferença entre ausência e oclusão |
| roi_manager.py | Regiões dinâmicas, varredura global periódica |
| recognizer.py | Detecta candidatos visuais e produz JSON de contexto |
| dataset_store.py | PNGs crus + observações e previsões, em thread separada |
| runtime.py | Orquestra captura regional real e análise no mesmo thread MSS |

## Limitações atuais (importantes)

- **Identificação V1 é experimental.** Círculos do HUD e cores
  de barras geram candidatos, não confirmação infalível.
- O botão de acessório é opcional; o estado da partida também usa
  os templates do StateDetector existente quando há quadro completo.
- **HP numérico não é lido pela nova via regional**, e preenchimento
  sem contorno não é convertido em porcentagem inventada.
- Um recorte não é tratado como tela completa nem reusa pixels não
  observados. A prévia regional deixa o restante da imagem preto
  e indica essa ausência explicitamente.
- Um quadro completo é adquirido periodicamente (padrão: a cada
  0,90 segundo), e sempre que necessário.
- O detector de barras regional é uma nova linha de pesquisa: **não
  promete reproduzir a precisão do detector completo da V1.4**.
- Silhuetas ajudam na associação, mas não distinguem de forma
  confiável personagens visualmente semelhantes.
- O runtime inicial é sequencial: **30 FPS é a meta, não a taxa garantida**.
- A coleta usa um limite de espaço e amostras, com erro explícito se
  faltar disco ou a fila ficar cheia.

## Dataset local

Cada sessão produz:

- observations.jsonl: resultado de cada ciclo de reconhecimento;
- manifest.jsonl: imagem, hash SHA256, ROI global, estado e previsões;
- images/: recortes crus em PNG, sem anotações desenhadas;
- dataset_summary.json: quantidade de quadros e erros de armazenamento.

ground_truth=false marca que predições não são rótulos verdadeiros.
Para treinar, revise e rotule os exemplos separadamente. O formato
novo ainda requer um adaptador antes da exportação COCO do dataset legado.

## Benchmark proposto

Compare a V1.4 padrão com a V1 regional usando as mesmas condições.
Avalie FPS efetivo, latência P95, pixels realmente adquiridos,
CPU/RAM, falsos positivos, recuperação de alvos e preservação de
eventos críticos. Não considere redução de pixels sinônimo de ganho de
CPU nem de qualidade de detecção.

## Segurança e histórico

O projeto continua em modo somente leitura. Não executa comandos
de movimento, tiro, entrada em partidas ou interação com contas.
A branch preserva o código V1.4 completo para rollback.


## Benchmark local da nova versão

Execute:

    python qa/benchmark_recognition_v1.py "CAMINHO_DA_SESSAO"

O comando lê observations.jsonl e gera benchmark_recognition_v1.json
com FPS efetivo, latência P95 do ciclo, pixels capturados, eventos e
proporção de capturas regionais/globais. Este tempo inclui a captura e
a análise, não sendo diretamente equivalente à latência da V1.4.


## Refatoração da auditoria de captura seletiva

Esta refatoração preserva a origem das regiões quando HUD e alvo
se sobrepõem. O planejamento mantém sources com nomes e retângulos
globais. A detecção de barras é efetuada nas subáreas de alvo,
não conforme o nome arbitrário da região unida.

A política agora limita alvos confirmados por ranking e orçamento
de pixels. Uma região central ajuda na recuperação enquanto ainda
não há alvo confirmado. Capturas completas permanecem periódicas.

ContextMemory mantém candidatos em estado candidate até a segunda
observação compatível; depois disso são confirmed e podem receber
ROI dedicada. Candidatos expiram mais cedo e rastros confirmados
possuem TTL próprio, mesmo sem a região voltar a ser observada.

O módulo de avaliação de barras combina evidências de formato,
contêiner, caracteres claros próximos e corpo, sem afirmar HP numérico
ou identidade de personagem. Verificações do HUD são limitadas a
quatro por segundo, em regiões menores e com Hough escalonado.

### Instrumentação

O arquivo observations.jsonl registra capture_ms, analysis_ms,
hud_check_ms, roi_suppressed_targets, capture_reason,
dataset_queue_depth e pixels adquiridos, além da duração total.
Cada PNG no manifest.jsonl inclui captured_at_s por recorte.
A escrita de PNGs não dispara para cada novo candidato provisório.

O benchmark local separa custos de captura, análise, HUD, enfileiramento
e razões para varreduras globais. Não mede acurácia sem rótulos humanos.
Capturas anteriores com overlays não substituem validação no BlueStacks.
