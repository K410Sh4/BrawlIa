# Identidade visual ancorada no nome — BrawlIA 0.12

## O que muda

A barra de HP é um *estado variável*, e não uma identidade. O leitor do
jogador já extraía uma assinatura normalizada dos glifos verdes do nome,
mas o rastreador atribuía IDs pelo movimento/posição. Agora:

1. Quando **nome e barra** estão presentes no mesmo quadro, o observador
   emite uma chave `visual:...` derivada da assinatura inicial do nome.
2. Se um candidato novo se parece suficientemente com a referência e
   apresenta a mesma chave, ele pode recuperar a trilha original mesmo
   quando sua posição mudou mais do que o rastreamento geométrico permite.
3. Se não houver observação, a saída continua vazia; durante até 6 s a
   última trilha com nome pode ficar inativa e ser recuperada posteriormente,
   desde que a posição permaneça dentro de um limite de 480 px.
4. Se dois candidatos exibirem a mesma chave no mesmo quadro, o rastreador
   não aplica esse vínculo de identidade e volta a usar a geometria.
5. Ao confirmar a tela de resultados, a resolução ou a próxima partida,
   as referências antigas são descartadas.

O número `track_id` continua numérico e compatível com relatórios
anteriores; `track_name_identity_key` é a chave adicional de identidade.
`identity_association` indica `nome_visual` ou `geometria` e o evento
`reidentified` diferencia recuperação por nome de nova criação.

**Limites importantes**

- A chave é um fingerprint do *aspecto* do nome do jogador e **não OCR**.
- Não informa quais caracteres aparecem no apelido, nem identidade real.
- A assinatura continua vinculada ao primeiro nome aceito; uma primeira
  detecção incorreta pode prejudicar as demais. Por isso a correspondência
  visual exige similaridade >= 0.55 e uma barra observada.
- O BrawlIA ainda não tem segmentação confiável de nomes de adversários.
  Outros objetos continuam usando a geometria original.
- Nomes iguais não são identificadores globalmente únicos; até um OCR
  verdadeiro exigiria desambiguação espacial e temporal.
- O limiar e a janela de reidentificação podem ser configurados no arquivo
  `configuracoes.json`, seção `tracking`.

## Como validar

Na mesma gravação, compare número de `track_id` para
`kind=jogador_candidato` antes/depois. Uma redução é útil, mas não basta:
revise quadros, especialmente após sumir/reaparecer, para ver se o ID foi
atribuído à pessoa correta. Anotar nomes e caixas manualmente em PNGs
permite futuramente medir falsos vínculos e mudanças indevidas.

O vídeo da V0.11 de 06/11 mostrou 915 observações de jogador divididas
em IDs 1, 21 e 29. A proposta busca reduzir essas trocas sem elevar
substantivamente o custo de processamento. O benchmark real da V0.12
ainda depende da próxima execução e não é uma taxa de acerto confirmada.
