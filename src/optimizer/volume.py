import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, Dict, Tuple


def optimize_volume(path: List[str], pool_infos: Dict[Tuple[str,str], Dict]):
    """Simple optimization using one-side liquidity of each pool for now"""
    def revenue(x):
        prod = 1.0
        for i in range(len(path)-1):
            A, B = path[i], path[i+1]
            info = pool_infos.get((A,B), {'rate':1, 'liquidityA':1})
            rate = info['rate']
            liqA = info['liquidityA']
            prod *= rate * x / (x + liqA)
        return -prod

    res = minimize_scalar(revenue, bounds=(1e-6, 1e6), method='bounded')
    return res.x if res.success else 0.0