# BrawlIA / Brawlla 0.10.0

[![Tests](https://github.com/K410Sh4/BrawlIa/actions/workflows/ci.yml/badge.svg)](https://github.com/K410Sh4/BrawlIa/actions/workflows/ci.yml)

Observador local de Brawl Stars no BlueStacks. Captura imagens, propõe barras/jogador/caixas, acompanha IDs e produz evidências para revisão. **Somente leitura: não envia comandos de jogo.** A detecção atual é heurística, sem modelo treinado ou OCR numérico de HP.

## Executar no Windows

1. Instale Python 3.11 ou 3.12 e extraia [o código](https://github.com/K410Sh4/BrawlIa/archive/refs/heads/main.zip) em uma pasta.
2. Execute `iniciar_windows.bat`. O inicializador cria `.venv` e instala as versões de `requirements.txt`.
3. Selecione a janela do BlueStacks ou ADB; mantenha a janela descoberta no modo MSS.
4. Confira captura 30 FPS, análise 15 FPS e largura 1280. Vídeo 15 FPS evita repetir trabalho de codificação; preferências anteriores são respeitadas.
5. Para coletar dados, marque **Coletar frames brutos → 2 FPS**. Rajadas selecionam até 10 FPS, limitadas pelo FPS real de análise.
6. Opcionalmente calibre o recorte, preservando o HUD. Pare e inicie outra sessão para cada partida. Espere a finalização antes de fechar.

No Linux: `python -m venv .venv`, ative o ambiente, instale `requirements.txt` e execute `python app.py`. A fonte Janela/Win32 é específica do Windows. ADB precisa estar disponível/configurado; não é necessário para os testes artificiais.

`gerar_exe_windows.bat` compila um EXE no seu Windows com PyInstaller. O repositório contém código e inicializadores; não há EXE compilado nesta publicação.

## Mudanças da 0.10.0

- Faixas de verde, amarelo, vermelho nas duas pontas do HSV, branco, outras cores e recipiente azul/escuro vêm da configuração efetiva.
- Máscaras compartilhadas por quadro e cache das consultas de glifos. A leitura do jogador usa região local entre buscas completas, com busca completa imediata quando falha; não extrapola observações ausentes.
- Compensação de câmera antes da associação principal de IDs; a evidência de corpo/barra usa esses mesmos IDs.
- Tempos separados de câmera, busca visual, classificação, associação, movimento do corpo, desenho e pipeline.
- Histórico completo em fila própria. O vídeo continua podendo substituir quadros pendentes, sem substituir registros aceitos do histórico.
- Cada sessão guarda configuração efetiva, hash, versão do Python e bibliotecas.
- Revisão permite vincular barras aos seus donos. Exportação COCO preserva relações em arquivo separado.
- Avaliação de localização de barras em PNGs humanos revisados, com TP/FP/FN, precisão/recall e comparação de configurações.
- Testes automáticos em Windows/Linux e Python 3.11/3.12; dependências de execução fixadas.

## Arquivos e histórico

No Windows, a raiz padrão é `%LOCALAPPDATA%\Brawlla`; no Linux, `$XDG_DATA_HOME/Brawlla` ou `~/.local/share/Brawlla`. Os dados locais não são enviados ao GitHub.

```text
Brawlla/
  preferencias.json
  gravacoes/<session_id>/
    sessao.json          # configuração efetiva, ambiente, contagens, erros
    analise.jsonl        # todas as análises aceitas, inclusive com vídeo desligado
    analise.mp4          # ou AVI; somente quando vídeo está ativado
    video_frames.jsonl   # mapeamento dos eventos que chegaram ao vídeo
    resumo.json
    relatorio.json
    relatorio.html
  capturas/<session_id>/
    dataset_session.json
    manifest.jsonl
    frame_000000001.png
    frame_000000001.prediction.json
    labels/frame_000000001.json
```

O esquema acima descreve arquivos, não uma sequência de processamento. `session_id`, `frame_id` e timestamp monotônico associam imagem/análise/vídeo. PNG é o recorte nativo antes das marcações. Predições são separadas dos rótulos humanos.

Na 0.10, `analise.jsonl` não depende da fila do vídeo. Seus campos legados `video_frame`/`video_repeats` ficam nulos; para a correspondência exata, associe `frame_id` com `video_frames.jsonl`. Esse segundo arquivo contém índices, repetições, resoluções e encaixe no vídeo. Eventos que não chegaram ao vídeo continuam no histórico.

Histórico: fila de 256 eventos sem imagens, limite de 256 MiB e reserva de 512 MiB livres. Se faltar capacidade/espaço, a falha é explícita; o pipeline interrompe na tentativa de envio seguinte e a auditoria registra o histórico incompleto. Nenhum registro aceito é substituído silenciosamente. O vídeo tem limites próprios (30 min / 2 GiB aproximados); amostras têm 1500 imagens / 1 GiB aproximado, além da reserva de espaço. Ao atingir limites de vídeo/amostras, o histórico pode continuar dentro da sua cota.

## Configurar e reproduzir

Edite `configuracoes.json` antes de iniciar uma nova leitura. Os valores padrão vêm de `brawlla/config.py`; um teste mantém o JSON sincronizado com eles. `motion_enabled=false` desliga fluxo óptico sem desligar a detecção visual de barras. Limiares são heurísticos, não probabilidades calibradas.

As antigas opções `container_dark_fraction` e `container_dark_support_min` já não tinham efeito na 0.9. Ao carregar arquivos antigos, são removidas com aviso: use `min_container_fraction`, `min_short_fill_fraction` e `single_glyph_container_fraction`. As faixas de cor do JSON agora têm efeito real; perfis antigos que as alteraram devem ser reavaliados.

`sessao.json.analyzer_config` contém os valores usados e `provenance.config_sha256` identifica o perfil. A configuração da interface fica em `options`; versões do ambiente ficam em `provenance`.

## Revisar dados e ligar uma barra ao dono

- Abra **Revisar amostras** e selecione uma pasta com `manifest.jsonl`.
- Desenhe corpos/caixas com as categorias `jogador`, `outro_personagem`, `caixa`; recebem `entity_id` local à imagem.
- Selecione o corpo em **Dono da próxima barra**, escolha `barra_vida` e desenhe a barra. O rótulo recebe `owner_id`.
- Complete todas as categorias presentes antes de marcar Aprovada/Corrigida. Descarte ou marque Incerta quando necessário. Um negativo exige nenhuma caixa e a opção Negativo explícita.
- Salve a revisão. Predições nunca são promovidas a verdade automaticamente.

**Exportar dataset** gera COCO `train.json`, `val.json`, `test.json`, `relationships.json` e `dataset.json`. As imagens não são duplicadas: usar a raiz `image_root` do resumo. `relationships.json` liga IDs de anotações de barra/dono e indica o conjunto. Barras antigas sem vínculo são permitidas para detecção, mas contadas como `unlinked_bars`.

A divisão é estável por sessão (hash, proporções esperadas 80/10/10 em muitos grupos), sem misturar frames da mesma sessão. Poucas partidas podem deixar conjuntos vazios; isso é indicado por `empty_splits`. Uma mesma partida não deve ser repartida entre sessões de conjuntos diferentes. JPEGs antigos são preservados e ignorados na exportação nova.

## Avaliar antes de treinar

```bash
python evaluate_dataset.py CAMINHO_DAS_CAPTURAS --config configuracoes.json --output evaluation.json
python evaluate_dataset.py CAMINHO_DAS_CAPTURAS --config candidato.json --compare-config baseline.json --output evaluation.json
```

Somente PNGs aprovados/corrigidos por humano entram. O avaliador mede localização de barras em imagens independentes; não mede dono, pontos de vida exatos, estado da partida ou rastreamento temporal. Os rótulos precisam estar completos. Nenhuma rede é treinada automaticamente, nem há pesos de modelo incluídos. Consulte [o plano de dados](docs/DATASET.md).

## Avaliação opcional com ONNX (experimental)

A leitura ao vivo permanece heurística e não foi substituída. Há agora um
adaptador **somente offline** para avaliar modelos ONNX externos em vídeos,
com escalonamento de inferência, comparação de latência e seleção de
CUDA/DirectML (AMD/Intel no Windows) com fallback para CPU. Nenhum modelo
ou peso é incluído, e não há controle de jogo.

Instruções: [ONNX OFFLINE](docs/ONNX_OFFLINE.md). A precisão somente poderá
ser comparada com quadros revisados por pessoas.

## Desenvolvimento e testes

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip check
python -m unittest discover -s tests -v
python -m ruff check .
python -m black --check .
```

Os testes usam dados artificiais e diretórios temporários. Passar nos testes não mede precisão no jogo. [Arquitetura e contratos](docs/ARCHITECTURE.md); [limitações e validação](docs/VALIDATION.md).
