# BrawlIA V1.4 — Visão com perfil de desempenho

Ferramenta local de **observação e análise visual**, sem comandos no jogo.

A nova interface do BrawlIA é um painel de captura com **um único botão**:
o usuário abre o BlueStacks, clica em "Iniciar captura automática" e
o sistema localiza a janela visível, grava vídeo + histórico + amostras PNG,
com metas fixas de captura, análise e vídeo a 30 FPS.

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

## Visão computacional a 30 FPS

No painel principal, 30 FPS são metas fixas de captura e análise,
com a política de usar sempre o quadro mais recente. Não há promessa de
30 análises reais por segundo em qualquer cena. **Isso mede a capacidade efetiva da máquina, não a ocupação
direta da GPU.** Na V1.0, a visão heurística ainda é executada em CPU/OpenCV.
Suporte de GPU via ONNX/DirectML permanece experimental e somente offline.

O vídeo mantém FPS nominal fixo de 30 para preservar o formato MP4; o
ritmo de captura/análise é adaptativo e consta em `sessao.json`.
A V1.1 evita uma cópia desnecessária do quadro completo nas fontes MSS
e considera também o tempo de entrega à gravação no feedback de FPS.

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

## Cadastro da apresentação da partida — V1.2

O analisador reconhece a apresentação com 10 cartões (duas fileiras) e
atribui `P01` a `P10` às **posições**, não às trilhas de barras.
Um worker OCR separado lê os nomes em até três imagens da apresentação,
confirmando nomes repetidos. Nomes truncados, incertos ou duplicados não
identificam automaticamente o jogador. O OCR durante combate é acionado
somente em recortes de nome do jogador com intervalo de 2,5 segundos, e
só associa `participant_id` após correspondência única e repetida.
Também há coleta limitada de nomes sobre barras de outros personagens
com trilhas estáveis, alternando as amostras para evitar consumo excessivo.

O campo `match_roster` nos registros inclui os 10 slots, status e
`local_participant_id`. Um `track_id` pode mudar sem alterar o
`participant_id`, mas **nem toda trilha terá participante identificado**.
A maioria dos outros personagens continua sem nome associado até termos um
detector confiável de nomes inimigos.

**Instalação automática de OCR no Windows:** `iniciar_windows.bat` tenta
instalar `requirements-ocr.txt` opcionalmente. Se indisponível, a captura
e os rastreadores continuam funcionando, sem inventar nomes.

[Detalhes e limites do cadastro](docs/ROSTER_V12.md).

## V1.3: foco em visão computacional

- Captura alvo **30 FPS**, análise alvo **30 FPS**, vídeo **30 FPS nominal**.
- Controle adaptativo desligado no painel principal, sem reduzir o alvo para
  5 FPS durante cenas complexas; se cada análise demorar mais de 33,3 ms, a
  taxa real cairá e os quadros antigos serão substituídos para evitar backlog.
- Coleta de **PNGs desligada**. `record_samples=False`, sem inicializar
  `SampleRecorder`. Vídeo MP4, JSONL e telemetria continuam ativos.
- `autoavaliacao.json`: relatório leve de detecções incertas, perda do jogador,
  HP potencialmente ambíguo e latência. Usa apenas metadados já disponíveis.
  Ajuda a escolher exemplos para revisão no vídeo gravado, não aprende pesos
  nem altera limites automaticamente.
- OCR da apresentação permanece opcional; quando executado pode consumir
  CPU e reduzir a taxa efetiva. Um MP4 com 30 FPS pode repetir quadros se a
  análise não alcançar a taxa.

## V1.4 — otimização auditável

O cenário agora tem cache temporal limitado e compensação pela câmera quando
há evidência suficiente. As barras e os nomes continuam sendo lidos no
quadro atual. O OCR passou a registrar diagnósticos por cartão, sem salvar
imagens adicionais. A captura DXcam é **opcional** para comparação com MSS,
sem garantir FPS mais altos antes de testes no PC-alvo.

[Guia e benchmarks V1.4](docs/PERFORMANCE_V14.md).
