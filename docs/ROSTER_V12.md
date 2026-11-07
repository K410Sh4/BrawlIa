# BrawlIA V1.2 — Cadastro de participantes antes do combate

## Como funciona

1. A análise reconhece as duas fileiras de cinco cartões pelo layout escuro da apresentação.
2. As posições recebem os IDs de partida `P01` a `P10` imediatamente, **ainda sem nomes**.
3. Um worker separado processa no máximo três quadros da apresentação,
   com intervalo de 550 ms; o analisador em tempo real não aguarda o OCR.
4. O RapidOCR (modelos ONNX) tenta ler cada nome nos recortes ampliados.
   Um nome é **tentativo** até duas leituras concordantes. A confiança
   mínima por leitura é 0,60.
5. No combate, um OCR de recorte do nome do jogador é agendado, no máximo,
   a cada 2,5 s quando há observação. São exigidas correspondência única
   >=0,85 e duas leituras concordantes para vincular um ID de trilha ao ID
   `Pxx` do cadastro.
6. O sistema registra `match_roster`, `participant_id` e
   `participant_identity_status` no `analise.jsonl`. A UI mostra o
   progresso das leituras e qual ID está **confirmado** como local.

## Limites deliberados

- O layout foi medido em uma gravação de 1280x720 do usuário.
  Outras interfaces, escalas ou modos de jogo podem exigir outro adaptador.
- Os nomes dos cartões podem estar cortados. O texto OCR pode ser parcial
  ou incorreto; as correspondências só são feitas quando há evidência
  suficiente e singular.
- **Não atribuímos todos os jogadores observados a um Pxx automaticamente.**
  A V1.2 também amostra, no máximo a cada 2,5 s, o espaço acima de barras
  de personagens não locais com pelo menos cinco observações. Só identifica
  quando o OCR coincide, de modo único e repetido, com um nome confirmado
  na apresentação; caso contrário, permanece `unresolved`.
- Personagens e nomes não são equivalentes: podem existir brawlers iguais.
  A ordem do cartão não é usada como prova de qual jogador é o usuário.
- A inferência OCR ainda é realizada em CPU. O worker pode competir com
  análise OpenCV, então compare FPS e p95 com a V1.1 antes de aumentar
  frequências. Modelos treinados e DirectML para inferência integral não
  fazem parte desta etapa.
- Se o OCR não estiver disponível, o registro marca `ocr_unavailable` e
  nunca inventa nomes; captura, vídeo e análise heurística continuam.
- O cadastro é resetado ao detectar uma nova apresentação após combate,
  sem reutilizar identidade de uma partida em outra.
- IDs P01–P10 são **somente daquela partida**, e não identificadores de
  contas de usuário.

## Instalação e teste local

O inicializador Windows tenta instalar `requirements-ocr.txt` no
ambiente `.venv-v1`. Caso haja problema de pacote, pode executar:

```powershell
.\.venv-v1\Scripts\python.exe -m pip install -r requirements-ocr.txt
.\.venv-v1\Scripts\python.exe -m unittest discover -s tests -v
```

Fonte: https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/install/
e https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/usage/.

Para validar, comece a gravação **antes** da apresentação, espere a contagem
regressiva e compare `confirmed_names` com os nomes visíveis no vídeo.
Se a apresentação não for encontrada, o status será `not_seen`, sem
tentativa de inferir nomes a partir de objetos quaisquer.
