"""
ベルマン-フォードによる負閉路検出ユーティリティ
  * bellman_ford_negative_cycles
      - 最大 k 本まで検出してそれぞれ [v0,…,v0] を返す
"""

from typing import List, Tuple

def bellman_ford_negative_cycles(
    n: int,
    edges: List[Tuple[int, int, float]],
    max_cycles: int = 1,
    rel_eps: float = 1e-12,
) -> List[List[int]]:
    """
    Parameters
    ----------
    n : int
        頂点数
    edges : [(u,v,w), ...]
        有向辺集合
    max_cycles : int
        検出本数上限
    rel_eps : float
        相対許容誤差

    Returns
    -------
    List of cycles (各サイクルは頂点 ID のリスト [v0,...,v0])
    """
    # ---------- ① 全頂点に到達させるためダミー源点を張る ----------
    dist   = [0.0] * (n + 1)          # 最後の頂点 n がダミー
    parent = [-1]  * (n + 1)

    ext_edges = edges + [(n, v, 0.0) for v in range(n)]

    # ---------- ② |V|-1 回緩和 ----------
    for _ in range(n):
        updated = False
        for u, v, w in ext_edges:
            if dist[u] + w < dist[v] - rel_eps * max(1.0, abs(dist[u])):
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            break

    # ---------- ③ 追加 1 回で更新される頂点は負閉路へ到達 ----------
    cycles = []
    found = set()                     # 閉路に含まれる頂点の集合（重複排除）
    for u, v, w in ext_edges:
        if dist[u] + w < dist[v] - rel_eps * max(1.0, abs(dist[u])):
            # v から n 回親をたどると必ず閉路内
            x = v
            for _ in range(n):
                x = parent[x]

            if x in found:            # 既に回収済みの閉路
                continue

            cycle = [x]
            y = parent[x]
            while y != x and y != -1:
                cycle.append(y)
                y = parent[y]
            cycle.append(x)
            cycle.reverse()

            # 長さ 2 は自己ループまたは双方向エッジなのでスキップ
            if len(cycle) > 3:
                cycles.append(cycle)
                found.update(cycle)
                if len(cycles) >= max_cycles:
                    break
    return cycles
