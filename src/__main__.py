# src/__main__.py

import asyncio
import logging
import math
import numpy as np
import igraph as ig

from clients.raydium_client import RaydiumClient
from clients.orca_client import OrcaClient
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from executor import execute_cycle
from config import NEG_CYCLE_THRESHOLD, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("bot")

ray  = RaydiumClient()
orca = OrcaClient()

def find_negative_cycle(edges):
    """
    Bellman–Ford で負閉路を検出し、その経路（mint 配列）を返す。
    edges: List[(mint_u, mint_v, rate_uv)]
    """
    # 1) 頂点集合
    verts = {m for e in edges for m in e[:2]}
    idx = {m: i for i, m in enumerate(verts)}
    inv = {i: m for m, i in idx.items()}
    n = len(verts)

    # 2) 重み = -log(rate)
    w = [(-math.log(r), idx[u], idx[v]) for u, v, r in edges]

    # 3) Bellman–Ford
    dist  = [0.0] * n
    parent = [-1] * n
    for _ in range(n - 1):
        updated = False
        for cost, u, v in w:
            if dist[u] + cost < dist[v]:
                dist[v] = dist[u] + cost
                parent[v] = u
                updated = True
        if not updated:
            break

    # 4) もう一度緩和して改善が出た頂点→負閉路
    for cost, u, v in w:
        if dist[u] + cost < dist[v]:
            # v から parent を n 回たどると必ず閉路内に入る
            x = v
            for _ in range(n):
                x = parent[x]
            cycle_idx = []
            cur = x
            while True:
                cycle_idx.append(cur)
                cur = parent[cur]
                if cur == x or cur == -1:
                    break
            cycle_idx.reverse()
            return [inv[i] for i in cycle_idx]
    return None


async def arbitrage_job():
    edges_ray, edges_orc = await asyncio.gather(
        ray.edge_list(), orca.edge_list()
    )
    edges = edges_ray + edges_orc

    cycle = find_negative_cycle(edges)
    if not cycle:
        log.info("No neg-cycle found")
        return

    amount_in = 0.001
    rate_prod = np.prod(
        [ next(r for (u,v,r) in edges if u==cycle[i] and v==cycle[(i+1)%len(cycle)])
          for i in range(len(cycle)) ]
    )
    est_gain = amount_in * (rate_prod - 1)
    if rate_prod <= 1 or est_gain < 0:
        log.info("Cycle found but not profitable")
        return

    log.info("Cycle %s  amount %.6f → gain %.6f",
             "→".join(cycle), amount_in, est_gain)
    await execute_cycle(cycle, amount_in, est_gain)

async def main():
    # イベントループ上でスケジューラを起動
    scheduler = AsyncIOScheduler()
    scheduler.add_job(arbitrage_job, "interval", seconds=30, id="arbitrage_job")
    scheduler.start()
    log.info("Scheduler started, entering event loop…")
    # 永久待機
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
