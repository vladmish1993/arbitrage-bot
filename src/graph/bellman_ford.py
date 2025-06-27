"""
Negative cycle detection utility using Bellman-Ford
  * bellman_ford_negative_cycles
      - Detect up to k cycles and return each as [v0,...,v0]
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
        number of vertices
    edges : [(u,v,w), ...]
        directed edge set
    max_cycles : int
        maximum number of cycles
    rel_eps : float
        relative tolerance

    Returns
    -------
    List of cycles (each cycle is a list of vertex IDs [v0,...,v0])
    """
    # ---------- 1. Add dummy source to reach all vertices ----------
    dist   = [0.0] * (n + 1)          # The last vertex n is a dummy
    parent = [-1]  * (n + 1)

    ext_edges = edges + [(n, v, 0.0) for v in range(n)]

    # ---------- 2. Relax |V|-1 times ----------
    for _ in range(n):
        updated = False
        for u, v, w in ext_edges:
            if dist[u] + w < dist[v] - rel_eps * max(1.0, abs(dist[u])):
                dist[v] = dist[u] + w
                parent[v] = u
                updated = True
        if not updated:
            break

    # ---------- 3. Vertices updated in one more pass are in a negative cycle ----------
    cycles = []
    found = set()                     # Set of vertices in cycles (to avoid duplicates)
    for u, v, w in ext_edges:
        if dist[u] + w < dist[v] - rel_eps * max(1.0, abs(dist[u])):
            # following parent n times from v guarantees a vertex in a cycle
            x = v
            for _ in range(n):
                x = parent[x]

            if x in found:            # Cycle already collected
                continue

            cycle = [x]
            y = parent[x]
            while y != x and y != -1:
                cycle.append(y)
                y = parent[y]
            cycle.append(x)
            cycle.reverse()

            # Skip length 2 cycles since they are self-loops or two-way edges
            if len(cycle) > 3:
                cycles.append(cycle)
                found.update(cycle)
                if len(cycles) >= max_cycles:
                    break
    return cycles
