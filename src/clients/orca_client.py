# src/clients/orca_client.py
import httpx, logging, asyncio
from typing import List, Tuple

log = logging.getLogger(__name__)

ORCA_AMM_URL  = "https://api.orca.so/v1/pools"
ORCA_WP_URL   = "https://api.mainnet.orca.so/v1/whirlpool/list"

class OrcaClient:
    async def _fetch(self, url: str):
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(url, follow_redirects=True)
        r.raise_for_status()
        return r.json()

    async def edge_list(self) -> List[Tuple[str, str, float]]:
        edges: List[Tuple[str, str, float]] = []

        # ---------- 1) legacy constant-product pools ----------
        amm_json = await self._fetch(ORCA_AMM_URL)
        pools_iter = amm_json if isinstance(amm_json, list) else amm_json.get("pools", [])
        for p in pools_iter:
            try:
                a, b  = p["tokenAMint"], p["tokenBMint"]
                ra, rb = float(p["tokenAAmount"]), float(p["tokenBAmount"])
                if ra > 0 and rb > 0:
                    price = rb / ra
                    edges += [(a, b, price), (b, a, 1/price)]
            except (KeyError, ValueError, TypeError):
                continue

        # ---------- 2) Whirlpool (CLMM) ----------
        wp_json = await self._fetch(ORCA_WP_URL)
        for p in wp_json.get("whirlpools", []):
            try:
                a, b   = p["tokenMintA"], p["tokenMintB"]
                dec_a, dec_b = p["tokenDecimalA"], p["tokenDecimalB"]
                sqrt_x64 = int(p["sqrtPrice"])
                price = (sqrt_x64 / 2**64) ** 2 * 10 ** (dec_a - dec_b)
                edges += [(a, b, price), (b, a, 1/price)]
            except (KeyError, ValueError, TypeError):
                continue

        log.info("Orca edges: %d", len(edges))
        return edges
