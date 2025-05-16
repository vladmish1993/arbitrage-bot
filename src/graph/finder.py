from typing import List, Dict
import numpy as np
from igraph import Graph

class ArbPathFinder:
    def __init__(self, routes: List[Dict]):
        # Jupiter routes → エッジリスト生成 (inputMint, outputMint, rate)
        edges = []
        for r in routes:
            in_mint  = r['inToken']['address']
            out_mint = r['outToken']['address']
            rate     = r['outAmount'] / r['inAmount'] if r['inAmount'] else 0
            if rate > 0:
                edges.append((in_mint, out_mint, -np.log(rate)))

        verts = list({v for u,v,_ in edges} | {u for u,_,_ in edges})
        idx = {v:i for i,v in enumerate(verts)}
        g = Graph(directed=True)
        g.add_vertices(len(verts))
        for u,v,w in edges:
            g.add_edge(idx[u], idx[v], weight=w)
        self.graph, self.idx2mint = g, verts

    def find_negative_cycle(self):
        cycle = self.graph.get_negative_cycle(weights='weight')
        return [self.idx2mint[i] for i in cycle] if cycle else None