from typing import List, Tuple


def bellman_ford_negative_cycle(n: int, edges: List[Tuple[int, int, float]]):
    # ①距離と親を初期化
    dist   = [0.0] * n          # すべて 0 にするとダミー源点を張るのと同じ
    parent = [-1]  * n

    # ②|V|-1 回緩和
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            break

    # ③もう 1 回緩和して更新が走れば負閉路
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            # v から親を |V| 回たどると必ず閉路内に入る
            x = v
            for _ in range(n):
                x = parent[x]

            # 閉路を取り出す
            cycle = [x]
            y = parent[x]
            while y != x:
                cycle.append(y)
                y = parent[y]
            cycle.reverse()
            return cycle        # 頂点 ID のリスト

    return None                 # 負閉路なし
