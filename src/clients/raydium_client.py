# src/clients/raydium_client.py
import httpx, math, asyncio, logging
from typing import List, Tuple

RAY_URL = "https://api.raydium.io/v2/sdk/pairs/solana.mainnet.json"
log = logging.getLogger(__name__)

class RaydiumClient:
    async def edge_list(self) -> List[Tuple[str,str,float]]:
        async with httpx.AsyncClient(timeout=10) as c:
            res = await c.get(RAY_URL)
        res.raise_for_status()
        pairs = res.json().values()      # dict keyed by lpMint
        edges = []
        for p in pairs:
            try:
                a, b  = p["baseMint"], p["quoteMint"]
                ra, rb = float(p["baseReserve"]), float(p["quoteReserve"])
                if ra > 0 and rb > 0:
                    price_ab = rb / ra
                    edges.append((a, b, price_ab))
                    edges.append((b, a, 1/price_ab))
            except (KeyError, ValueError, TypeError):
                continue
        log.info("Raydium: %d pairs (%d edges)", len(pairs), len(edges))
        return edges
