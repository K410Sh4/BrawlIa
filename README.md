# BrawlIA V1.1 — Captura automática

Ferramenta local de **observação e análise visual**, sem comandos no jogo.

A nova interface do BrawlIA é um painel de captura com **um único botão**:
o usuário abre o BlueStacks, clica em "Iniciar captura automática" e
o sistema localiza a janela visível, grava vídeo + histórico + amostras PNG,
além de adaptar FPS de captura e análise à capacidade observada.

## Iniciar no Windows

1. Python 3.11 ou 3.12 recomendado.
2. Extraia o ZIP e abra o BlueStacks numa janela visível.
3. Execute `iniciar_windows.bat`.
4. Clique em **Iniciar captura automática**. O mesmo botão encerra a sessão,
   finaliza os arquivos e libera outra captura.
5. Clique em **Abrir gravação** para abrir no Explorador a pasta da sessão
   atual ou da última sessão salva, inclusive após reiniciar o programa.

O programa não acessa janelas ocultas ou minimizadas. MSS captura os
pixels visíveis do desktop, portanto evite janelas sobre o emulador.

A pasta de destino aparece na interface. Arquivos de sessão ficam em:
`%LOCALAPPDATA%\\Brawlla\\gravacoes\\<sessao>`.

## FPS dinâmicos

Taxas de 5 até 60 FPS, limitadas por latência p90 de captura e análise,
com margem de segurança. Ajustes gradativos a cada janela de amostras de
2 segundos. **Isso mede a capacidade efetiva da máquina, não a ocupação
direta da GPU.** Na V1.0, a visão heurística ainda é executada em CPU/OpenCV.
Suporte de GPU via ONNX/DirectML permanece experimental e somente offline.

O vídeo mantém FPS nominal fixo de 15 para preservar o formato MP4; o
ritmo de captura/análise é adaptativo e consta em `sessao.json`.

## Verificação

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Arquivos para auditar: `analise.jsonl`, `video_frames.jsonl`,
`analise.mp4`, `resumo.json`, `sessao.json` e, quando elegíveis,
amostras PNG com metadados na pasta `capturas`.

**Guia detalhado:** [captura automática e desempenho](docs/AUTO_CAPTURE_V1.md).

## Segurança e limites

- Não controla teclado, mouse, joystick, partidas ou contas.
- Sem OCR de nomes literais; o tracker usa assinaturas visuais do nome.
- Heurísticas HSV atuais não são equivalentes a um modelo treinado.
- Cobertura de candidatos não é precisão. Requer dataset anotado para
  comparar acertos e falsos positivos.
- O desempenho real deve ser medido no seu PC, com gravação ativada.

## Recursos de pesquisa e avaliação

- [Visão ONNX offline](docs/ONNX_OFFLINE.md)
- [Identidade por nome visual](docs/NAME_IDENTITY.md)
- [Dataset e anotações](docs/DATASET.md)
- [Arquitetura e métricas anteriores](docs/ARCHITECTURE.md)

Ferramentas de avaliação e relatórios continuam disponíveis no repositório,
mas os controles antigos não fazem parte do painel principal.
