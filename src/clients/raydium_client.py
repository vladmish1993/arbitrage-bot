# src/clients/raydium_client.py
"""
Raydium API client

-----------------------------
Purpose:
    - Fetch constant-product AMM trade info managed by Raydium via API
    - Example: https://api-v3.raydium.io/pools/info/list?poolType=standard&poolSortField=liquidity&sortType=desc&pageSize=500&page=1
    - Generate graph data for the Bellman-Ford algorithm
"""

import copy
import json
import logging
import math
from typing import Dict, List, Tuple

import httpx

from .base_dex_client import BaseDexClient, DEFAULT_TIMEOUT

# Raydium specific constants
POOL_URL = "https://api-v3.raydium.io/pools/info/list"

# Logger setup
log = logging.getLogger(__name__)


class RaydiumClient(BaseDexClient):
    """
    Client that generates graph data for arbitrage based on information obtained from the Raydium API
    
    Inherits BaseDexClient and implements Raydium specific features
    """

    @property
    def dex_name(self) -> str:
        """Return DEX name"""
        return "Raydium"

    def _validate_pool_data(self, pool: Dict) -> bool:
        """
        Validate Raydium specific pool data
        
        Parameters
        ----------
        pool : Dict
            Pool data
            
        Returns
        -------
        bool
            Validation result
        """
        # Raydium specific required fields
        required_fields = ["id", "mintA", "mintB", "price", "tvl", "feeRate", "mintAmountA", "mintAmountB"]
        
        # Run base class validation
        if not super()._validate_pool_data(pool, required_fields):
            return False
        
        # Raydium specific liquidity check
        liquidity_a = float(pool.get("mintAmountA", 0))
        liquidity_b = float(pool.get("mintAmountB", 0))
        if not self._validate_liquidity(liquidity_a, liquidity_b, pool.get("id")):
            return False
            
        return True

    async def get_graph(
        self,
        poolType: str = "standard",
        poolSortField: str = "liquidity",
        sortType: str = "desc",
        pageSize: int = 500,
        max_pages: int = 1,
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Fetch pool information from all pages of the Raydium API and
        return an edge list for Bellman-Ford.

        Parameters
        ----------
        poolType : str, default "standard"
            Pool type
        poolSortField : str, default "liquidity"
            Sort field
        sortType : str, default "desc"
            Sort order
        pageSize : int, default 500
            Number of pools per page
        max_pages : int, default 5
            Maximum number of pages to fetch
            
        Returns
        -------
        List[Tuple[str, str, float, Dict]]
            Edge list: (from_token, to_token, weight, pool_info)
        """
        edges: List[Tuple[str, str, float, Dict]] = []
        valid_pools = 0

        for page in range(1, max_pages + 1):
            params = {
                "poolType": poolType,
                "poolSortField": poolSortField,
                "sortType": sortType,
                "pageSize": pageSize,
                "page": page,
            }

            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    resp = await client.get(POOL_URL, params=params)
                resp.raise_for_status()

                data = resp.json()
                pools_data = data.get("data", {}).get("data", [])
                log.info("%s: page %d fetched %d pools", self.dex_name, page, len(pools_data))

                # Break if no pools retrieved
                if not pools_data:
                    break

                for pool in pools_data:
                    if not self._validate_pool_data(pool):
                        continue
                    valid_pools += 1

                    # Common information
                    token_a = pool["mintA"]
                    token_b = pool["mintB"]
                    a_addr = token_a["address"]
                    b_addr = token_b["address"]

                    base = {
                        "dex": self.dex_name,
                        "pool_id": pool["id"],
                        "price": pool["price"],
                        "tvl": pool["tvl"],
                        "fee_rate": pool["feeRate"],
                        "liquidity_a": float(pool["mintAmountA"]),
                        "liquidity_b": float(pool["mintAmountB"]),
                    }

                    # A→B
                    info_ab = copy.deepcopy(base)
                    info_ab["token_a"] = {
                        "address": a_addr,
                        "symbol": token_a.get("symbol"),
                        "decimals": token_a.get("decimals"),
                    }
                    info_ab["token_b"] = {
                        "address": b_addr,
                        "symbol": token_b.get("symbol"),
                        "decimals": token_b.get("decimals"),
                    }
                    w_ab = self._calculate_weight(info_ab, "A_TO_B")
                    if math.isfinite(w_ab):
                        edges.append((a_addr, b_addr, w_ab, info_ab))

                    # B→A
                    info_ba = copy.deepcopy(base)
                    info_ba["token_a"] = {
                        "address": b_addr,
                        "symbol": token_b.get("symbol"),
                        "decimals": token_b.get("decimals"),
                    }
                    info_ba["token_b"] = {
                        "address": a_addr,
                        "symbol": token_a.get("symbol"),
                        "decimals": token_a.get("decimals"),
                    }
                    # Swap reserves as well
                    info_ba["liquidity_a"], info_ba["liquidity_b"] = (
                        info_ba["liquidity_b"],
                        info_ba["liquidity_a"],
                    )
                    w_ba = self._calculate_weight(info_ba, "B_TO_A")
                    if math.isfinite(w_ba):
                        edges.append((b_addr, a_addr, w_ba, info_ba))

            except httpx.HTTPStatusError as e:
                log.error("%s API HTTP error on page %d: %s", self.dex_name, page, e)
                break
            except Exception as e:
                log.error("Unexpected error on page %d: %s", page, e)
                break

        log.info("Graph created: %d edges from %d valid pools (pages: 1–%d)", 
                 len(edges), valid_pools, page)
        return edges

    async def print_pools(
        self,
        poolType: str = "standard",
        poolSortField: str = "liquidity",
        sortType: str = "desc",
        pageSize: int = 500,
        max_pages: int = 1,
    ) -> None:
        """
        Fetch pool information from the Raydium API and
        fetch all pages to log arbitrage metrics.

        Parameters
        ----------
        poolType : str, default "standard"
            Pool type
        poolSortField : str, default "liquidity"
            Sort field
        sortType : str, default "desc"
            Sort order
        pageSize : int, default 500
            Number of pools per page
        max_pages : int, default 1
            Maximum number of pages to fetch
        """

        for page in range(1, max_pages + 1):
            params = {
                "poolType": poolType,
                "poolSortField": poolSortField,
                "sortType": sortType,
                "pageSize": pageSize,
                "page": page,
            }

            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    resp = await client.get(POOL_URL, params=params)
                resp.raise_for_status()

                data = resp.json()
                pools_data = data.get("data", {}).get("data", [])
                log.info("%s: page %d fetched %d pools", self.dex_name, page, len(pools_data))

                # Stop looping if there are no pools
                if not pools_data:
                    break

                arbitrage_data: List[Dict] = []
                for pool in pools_data:
                    if not self._validate_pool_data(pool):
                        continue

                    arbitrage_data.append({
                        "pool_id": pool.get("id"),
                        "token_a": {
                            "address": pool.get("mintA", {}).get("address"),
                            "symbol":  pool.get("mintA", {}).get("symbol"),
                            "decimals": pool.get("mintA", {}).get("decimals"),
                        },
                        "token_b": {
                            "address": pool.get("mintB", {}).get("address"),
                            "symbol":  pool.get("mintB", {}).get("symbol"),
                            "decimals": pool.get("mintB", {}).get("decimals"),
                        },
                        "price":             pool.get("price"),
                        "tvl":               pool.get("tvl"),
                        "fee_rate":          pool.get("feeRate"),
                        "liquidity_a":       pool.get("mintAmountA"),
                        "liquidity_b":       pool.get("mintAmountB"),
                        "daily_volume":      pool.get("day", {}).get("volume"),
                        "daily_volume_quote":pool.get("day", {}).get("volumeQuote"),
                        "daily_apr":         pool.get("day", {}).get("apr"),
                    })

                # Log output
                log.info(
                    "%s arbitrage data (page %d/%d):\n%s",
                    self.dex_name,
                    page,
                    max_pages,
                    json.dumps(arbitrage_data, ensure_ascii=False, indent=2, sort_keys=True),
                )

            except httpx.HTTPStatusError as e:
                log.error("%s API HTTP error on page %d: %s", self.dex_name, page, e)
                break
            except httpx.TimeoutException as e:
                log.error("%s API timeout on page %d: %s", self.dex_name, page, e)
                break
            except Exception as e:
                log.error("Unexpected error on page %d: %s", page, e)
                break


# Alias for backward compatibility
RaydiumClient.get_raydium_graph = RaydiumClient.get_graph
RaydiumClient.print_raydium_pools = RaydiumClient.print_pools

# Test command (data output)
# python -c "import asyncio; import logging; logging.basicConfig(level=logging.INFO); from src.clients.raydium_client import RaydiumClient; client = RaydiumClient(); asyncio.run(client.print_pools(pageSize=3))"

# Test command (graph creation)  
# python -c "import asyncio; from src.clients.raydium_client import RaydiumClient; client = RaydiumClient(); edges = asyncio.run(client.get_graph(pageSize=5)); print(f'Graph built: {len(edges)} edges'); print('First edge example:', edges[0] if edges else 'None')"
