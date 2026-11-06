# Prova de conceito: modelos ONNX offline (V0.11 experimental)

Esta implementação mantém a V0.10 e sua captura original intactas. É um
**avaliador passivo**, sem joystick, cliques, reentrada em partidas ou ação sobre
contas. Não inclui modelo treinado, dados privados ou pesos de terceiros.

## Referências técnicas

- [PylaAI](https://github.com/PylaAI/PylaAI): divisão de componentes e
  inferência ONNX/DirectML observadas como referências de arquitetura. Licença
  CC BY-NC 4.0: **não foi copiado código, modelo ou outro material do PylaAI**.
- [Jooi025/BrawlStarsBot](https://github.com/Jooi025/BrawlStarsBot):
  detectores YOLO e processamento independente; repositório MIT. Implementação
  presente escrita do zero, sem incorporar código/arquivos de terceiros.
- [ivanyordanovgt/brawl-stars-bot](https://github.com/ivanyordanovgt/brawl-stars-bot):
  documentação sobre detectores e frequência de inferência.

## Instalação separada no Windows (Python 3.11/3.12)

Execute `iniciar_windows.bat` para preparar as dependências existentes.
No ambiente ativado, para GPU AMD/Intel em Windows:

```powershell
python -m pip install onnxruntime-directml
```

Para inferência em CPU, instale `onnxruntime` no lugar de DirectML. Não
instale ambos no mesmo ambiente virtual sem avaliar os conflitos de pacotes.

## Avaliação com vídeo existente

Use exclusivamente um arquivo `.onnx` cujas classes e direitos de utilização
você conheça. O BrawlIA **não baixa pesos nem cria classes automaticamente**.
O arquivo opcional `classes.json` deve conter `["player","enemy",...]` na
mesma ordem das classes do modelo.

```powershell
python compare_model.py "analise.mp4" --model "meu_modelo.onnx" --classes classes.json --provider directml --every-n 3 --output modelo_analise.jsonl
```

O padrão é YOLOv8 em formato `[cx,cy,w,h,score_classe...]` (sem objectness).
Para um modelo já com NMS e saída `[x1,y1,x2,y2,score,classe]`, use
`--format xyxy`. Outros formatos (YOLOv5, segmentação, pose, saída int8 ou
batch variável) não são implementados: primeiro valide a exportação.

Saídas: `modelo_analise.jsonl` (apenas frames inferidos) e
`modelo_analise.summary.json` (tempo, provider e número de quadros).
Quadros pulados por escalonamento não viram observações sintéticas.
A seleção do DirectML só é confirmada se o provider efetivamente aparecer
no resultado. Se não estiver disponível, há fallback para CPU.

## Critérios para habilitar no pipeline principal

1. Rótulos humanos de pelo menos dois vídeos representativos com arbustos,
   HUD, HP baixo, várias cores e transições entre telas.
2. Comparação com o detector heurístico atual no **mesmo conjunto de quadros**:
   precisão, recall, falso positivo no HUD e associação ao dono.
3. Medida de tempo p50/p95, FPS e uso de CPU/GPU (separados da gravação).
4. Revisão dos limites de uso do modelo e da licença do exportador.
5. Somente então integrar como fonte opcional de observações reais; sem
   substituir a análise atual nem inferir vida exata de largura de barra.

A sessão V0.10 de 06/11 teve cobertura de candidato de jogador de 33,33% e
custo médio de cores/cenário de 41,99 ms. Esses valores são **linha de base
diagnóstica**, não métricas de acurácia. Sem modelo ONNX compatível e anotações
humanas, esta prova de conceito não aumenta por si só a precisão no jogo.
