"""Associação global determinística de observações com rastros.

Algoritmo húngaro em Python puro para matrizes pequenas, com opção explícita
de não associar. Não exige scipy nem interfere no capturador MSS.
"""

from __future__ import annotations

import math


def optimal_associations(
    scores: list[list[float]], minimum_score: float = 0.18
) -> list[tuple[int, int]]:
    """Retorna pares (observação, rastro) com soma de afinidades máxima.

    Pontuações não finitas/abaixo do limiar não podem forçar uma associação.
    Há uma coluna fictícia por observação para permitir casos sem par.
    """
    if not scores:
        return []
    n = len(scores)
    count = len(scores[0])
    if any(len(row) != count for row in scores):
        raise ValueError("Matriz de afinidades irregular")
    if not count:
        return []
    m = count + n
    # Minimização: custo=1-score, dummies têm custo da não associação.
    costs = [
        [
            1.0 - value if math.isfinite(value) and value >= minimum_score else 1000.0
            for value in row
        ]
        + [1.0 - minimum_score] * n
        for row in scores
    ]
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    value = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if value < minimum[j]:
                        minimum[j] = value
                        way[j] = j0
                    if minimum[j] < delta:
                        delta = minimum[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for col in range(1, count + 1):
        row = p[col] - 1
        if (
            row >= 0
            and math.isfinite(scores[row][col - 1])
            and (scores[row][col - 1] >= minimum_score)
        ):
            pairs.append((row, col - 1))
    return sorted(pairs)
