# Dados necessários para a próxima etapa

A 0.10 prepara coleta, revisão e comparação. Não inclui modelo treinado, pesos baixados, integração automática com nuvem ou resultados de treinamento inexistentes.

## Rótulos

Categorias: jogador, outro_personagem, caixa e barra_vida. O corpo/caixa inteiro usa `entity_id`; a barra usa `owner_id` apontando para ele na mesma imagem. IDs são locais à imagem, sem promessa de reidentificação entre frames. Vínculo órfão ou corpo com ID duplicado é rejeitado. Caixas ainda precisam de evidência visual; estar parado sozinho não define caixa.

Rotular também efeitos/oclusão/incerteza como estado de revisão. Aprovar sem objetos exige negativo explícito. Imagens com objetos não rotulados não são negativos válidos. Somente revisão humana aprovada/corrigida é exportada.

COCO contém caixas convencionais; relações ficam em `relationships.json`, com split, image_id e IDs numéricos das anotações de barra e dono. Dados antigos sem vínculo continuam úteis para localizar barras, mas não para avaliar associação ao dono.

## Diversidade e grupos

Coletar partidas diferentes, mapas/skins/modos/resoluções variados, vida crítica e cheia, flashes, sobreposição, caixas, HUD e telas negativas. Não usar apenas muitos frames quase iguais da mesma partida. Reservar partidas completas para validação/teste. O hash de sessão evita vazamento dentro de uma sessão; o operador precisa evitar dividir a mesma partida entre sessões diferentes.

Poucas sessões podem produzir conjuntos vazios. Confirme `empty_splits` e distribuição de classes antes de treinar. O exportador não inventa diversidade nem replica dados para ocupar conjuntos.

## Comparação

`evaluate_dataset.py` compara configurações nas mesmas imagens humanas revisadas. Relata TP, FP, FN, precisão/recall e tempo visual médio com IoU configurável. Deduplica propostas quase idênticas e usa pareamento um a um por IoU decrescente. É um diagnóstico de localização de barras; não é mAP completo, avaliação de identidade, HP exato ou replay temporal.

Uma futura rede deverá ser comparada com a heurística em partidas reservadas, incluindo tamanho pequeno de barra e dono. A escolha de backend GPU depende de ensaio no Windows/hardware real. O projeto não assume CUDA nem aceleração AMD sem validação.
