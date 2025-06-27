# src/clients/meteora_client.py
"""
Meteora DLMM API client

-----------------------------
Purpose:
    - Fetch pool information from Meteora's DLMM API
    - Build graph edges compatible with :class:`ArbitragePathFinder`
    - Provide utility to print pool information
"""

from __future__ import annotations

import copy
import json
import logging
import math
from typing import Dict, List, Tuple

import httpx

from .base_dex_client import BaseDexClient, DEFAULT_TIMEOUT

# Meteora specific constants
# NOTE: The Meteora API exposes multiple endpoints. We use the paginated
# pair listing endpoint so we can iterate through pools easily.
POOL_URL = "https://dlmm-api.meteora.ag/pair/all_with_pagination"

# Logger setup
log = logging.getLogger(__name__)


class MeteoraClient(BaseDexClient):
    """Client for the Meteora DLMM API"""

    @property
    def dex_name(self) -> str:
        """Return DEX name"""
        return "Meteora"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_token_info(self, token: Dict) -> Dict:
        """Return token info dict with standardised keys"""
        return {
            "address": token.get("address") or token.get("mint"),
            "symbol": token.get("symbol"),
            "decimals": token.get("decimals"),
        }

    def _validate_pool_data(self, pool: Dict) -> bool:
        """Validate Meteora specific pool data"""
        required_fields = [
            "id",
            "tokenA",
            "tokenB",
            "price",
            "tvl",
            "feeRate",
            "reserveA",
            "reserveB",
        ]

        # Translate new API field names to the ones expected by the base class
        alt_pool = {
            "id": pool.get("address") or pool.get("id"),
            "tokenA": pool.get("token_x")
            or pool.get("tokenA")
            or pool.get("mint_x")
            or pool.get("mintA"),
            "tokenB": pool.get("token_y")
            or pool.get("tokenB")
            or pool.get("mint_y")
            or pool.get("mintB"),
            "price": pool.get("current_price") or pool.get("price"),
            "tvl": pool.get("liquidity")
            or pool.get("tvl")
            or pool.get("tvlUsd")
            or pool.get("tvl_usd"),
            "feeRate": pool.get("base_fee_percentage")
            or pool.get("feeRate")
            or pool.get("fee_rate"),
            "reserveA": pool.get("reserve_x_amount")
            or pool.get("reserveA")
            or pool.get("liquidityA")
            or pool.get("liquidity_a"),
            "reserveB": pool.get("reserve_y_amount")
            or pool.get("reserveB")
            or pool.get("liquidityB")
            or pool.get("liquidity_b"),
        }

        if not super()._validate_pool_data(alt_pool, required_fields):
            return False

        liquidity_a = float(alt_pool.get("reserveA", 0))
        liquidity_b = float(alt_pool.get("reserveB", 0))
        return self._validate_liquidity(liquidity_a, liquidity_b, alt_pool.get("id"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def get_graph(
        self,
        limit: int = 50,
        max_pages: int = 1,
        sort_key: str = "tvl",
        order_by: str = "desc",
    ) -> List[Tuple[str, str, float, Dict]]:
        """Fetch pool information and build an edge list"""
        edges: List[Tuple[str, str, float, Dict]] = []
        valid_pools = 0

        for page in range(1, max_pages + 1):
            params = {
                "page": page,
                "limit": limit,
                "sort_key": sort_key,
                "order_by": order_by,
            }

            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    resp = await client.get(POOL_URL, params=params)
                resp.raise_for_status()

                data = resp.json()
                pools_data = data.get("data") or data.get("pools") or []
                log.info("%s: page %d fetched %d pools", self.dex_name, page, len(pools_data))

                if not pools_data:
                    break

                for pool in pools_data:
                    if not self._validate_pool_data(pool):
                        continue
                    valid_pools += 1

                    token_a_raw = (
                        pool.get("token_x")
                        or pool.get("tokenA")
                        or pool.get("token_a")
                        or pool.get("mint_x")
                        or pool.get("mintA")
                    )
                    token_b_raw = (
                        pool.get("token_y")
                        or pool.get("tokenB")
                        or pool.get("token_b")
                        or pool.get("mint_y")
                        or pool.get("mintB")
                    )
                    token_a = self._extract_token_info(token_a_raw or {})
                    token_b = self._extract_token_info(token_b_raw or {})

                    a_addr = token_a.get("address")
                    b_addr = token_b.get("address")
                    if not a_addr or not b_addr:
                        continue

                    base = {
                        "dex": self.dex_name,
                        "pool_id": pool.get("address") or pool.get("id"),
                        "price": pool.get("current_price") or pool.get("price"),
                        "tvl": pool.get("liquidity")
                        or pool.get("tvl")
                        or pool.get("tvlUsd")
                        or pool.get("tvl_usd"),
                        "fee_rate": pool.get("base_fee_percentage")
                        or pool.get("feeRate")
                        or pool.get("fee_rate"),
                        "liquidity_a": float(
                            pool.get("reserve_x_amount")
                            or pool.get("reserveA")
                            or pool.get("liquidityA")
                            or pool.get("liquidity_a")
                            or 0
                        ),
                        "liquidity_b": float(
                            pool.get("reserve_y_amount")
                            or pool.get("reserveB")
                            or pool.get("liquidityB")
                            or pool.get("liquidity_b")
                            or 0
                        ),
                    }

                    # A→B
                    info_ab = copy.deepcopy(base)
                    info_ab["token_a"] = token_a
                    info_ab["token_b"] = token_b
                    w_ab = self._calculate_weight(info_ab, "A_TO_B")
                    if math.isfinite(w_ab):
                        edges.append((a_addr, b_addr, w_ab, info_ab))

                    # B→A
                    info_ba = copy.deepcopy(base)
                    info_ba["token_a"] = token_b
                    info_ba["token_b"] = token_a
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

        log.info(
            "Graph created: %d edges from %d valid pools (pages: 1–%d)",
            len(edges),
            valid_pools,
            page,
        )
        return edges

    async def print_pools(
        self,
        limit: int = 50,
        max_pages: int = 1,
        sort_key: str = "tvl",
        order_by: str = "desc",
    ) -> None:
        """Fetch pools and output structured information for debugging"""
        for page in range(1, max_pages + 1):
            params = {
                "page": page,
                "limit": limit,
                "sort_key": sort_key,
                "order_by": order_by,
            }

            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    resp = await client.get(POOL_URL, params=params)
                resp.raise_for_status()

                data = resp.json()
                pools_data = data.get("data") or data.get("pools") or []
                log.info("%s: page %d fetched %d pools", self.dex_name, page, len(pools_data))

                if not pools_data:
                    break

                arbitrage_data: List[Dict] = []
                for pool in pools_data:
                    if not self._validate_pool_data(pool):
                        continue

                    token_a_raw = (
                        pool.get("token_x")
                        or pool.get("tokenA")
                        or pool.get("token_a")
                        or pool.get("mint_x")
                        or pool.get("mintA")
                    )
                    token_b_raw = (
                        pool.get("token_y")
                        or pool.get("tokenB")
                        or pool.get("token_b")
                        or pool.get("mint_y")
                        or pool.get("mintB")
                    )
                    token_a = self._extract_token_info(token_a_raw or {})
                    token_b = self._extract_token_info(token_b_raw or {})

                    arbitrage_data.append(
                        {
                            "pool_id": pool.get("address") or pool.get("id"),
                            "token_a": token_a,
                            "token_b": token_b,
                            "price": pool.get("current_price") or pool.get("price"),
                            "tvl": pool.get("liquidity")
                            or pool.get("tvl")
                            or pool.get("tvlUsd")
                            or pool.get("tvl_usd"),
                            "fee_rate": pool.get("base_fee_percentage")
                            or pool.get("feeRate")
                            or pool.get("fee_rate"),
                            "liquidity_a": pool.get("reserve_x_amount")
                            or pool.get("reserveA")
                            or pool.get("liquidityA")
                            or pool.get("liquidity_a"),
                            "liquidity_b": pool.get("reserve_y_amount")
                            or pool.get("reserveB")
                            or pool.get("liquidityB")
                            or pool.get("liquidity_b"),
                        }
                    )

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
            except Exception as e:
                log.error("Unexpected error on page %d: %s", page, e)
                break

# Alias for backward compatibility
MeteoraClient.get_meteora_graph = MeteoraClient.get_graph
MeteoraClient.print_meteora_pools = MeteoraClient.print_pools

