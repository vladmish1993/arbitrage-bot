# src/clients/raydium_client.py
"""
Raydium API クライアント

-----------------------------
目的:
    - Raydium が管理する定数積型AMMの取引情報をAPIをもとに取得
    - 例: https://api-v3.raydium.io/pools/info/list?poolType=standard&poolSortField=liquidity&sortType=desc&pageSize=500&page=1
    - ベルマン・フォードアルゴリズム用のグラフデータ生成
"""

import copy
import json
import logging
import math
from typing import Dict, List, Tuple

import httpx

from .base_dex_client import BaseDexClient, DEFAULT_TIMEOUT

# Raydium固有の定数
POOL_URL = "https://api-v3.raydium.io/pools/info/list"

# ロガー設定
log = logging.getLogger(__name__)


class RaydiumClient(BaseDexClient):
    """
    Raydium APIから取得した情報を基に、アービトラージ用のグラフデータを生成するクライアント
    
    BaseDexClientを継承し、Raydium固有の機能を実装
    """

    @property
    def dex_name(self) -> str:
        """DEX名を返す"""
        return "Raydium"

    def _validate_pool_data(self, pool: Dict) -> bool:
        """
        Raydium固有のプールデータ妥当性検証
        
        Parameters
        ----------
        pool : Dict
            プールデータ
            
        Returns
        -------
        bool
            妥当性検証結果
        """
        # Raydium固有の必須フィールド
        required_fields = ["id", "mintA", "mintB", "price", "tvl", "feeRate", "mintAmountA", "mintAmountB"]
        
        # 基底クラスの基本検証を実行
        if not super()._validate_pool_data(pool, required_fields):
            return False
        
        # Raydium固有の流動性チェック
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
        Raydium API から取得したプール情報をすべてのページからフェッチし、
        ベルマン・フォード用のエッジリストを返す。

        Parameters
        ----------
        poolType : str, default "standard"
            プールタイプ
        poolSortField : str, default "liquidity"
            ソートフィールド
        sortType : str, default "desc"
            ソート順
        pageSize : int, default 500
            1ページあたりのプール数
        max_pages : int, default 5
            最大何ページまで取得するか
            
        Returns
        -------
        List[Tuple[str, str, float, Dict]]
            エッジリスト: (from_token, to_token, weight, pool_info)
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

                # 取得プールがなければループ終了
                if not pools_data:
                    break

                for pool in pools_data:
                    if not self._validate_pool_data(pool):
                        continue
                    valid_pools += 1

                    # 共通情報
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
                    # リザーブも入れ替え
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
        Raydium API から取得したプール情報を
        全ページ分フェッチし、アービトラージ指標をログに出力。

        Parameters
        ----------
        poolType : str, default "standard"
            プールタイプ
        poolSortField : str, default "liquidity"
            ソートフィールド
        sortType : str, default "desc"
            ソート順
        pageSize : int, default 500
            1ページあたりのプール数
        max_pages : int, default 1
            最大何ページまで取得するか
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

                # プールがなければ以降ループ不要
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

                # ログ出力
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


# 後方互換性のためのエイリアス
RaydiumClient.get_raydium_graph = RaydiumClient.get_graph
RaydiumClient.print_raydium_pools = RaydiumClient.print_pools

# TEST用コマンド（データ出力）
# python -c "import asyncio; import logging; logging.basicConfig(level=logging.INFO); from src.clients.raydium_client import RaydiumClient; client = RaydiumClient(); asyncio.run(client.print_pools(pageSize=3))"

# TEST用コマンド（グラフ作成）  
# python -c "import asyncio; from src.clients.raydium_client import RaydiumClient; client = RaydiumClient(); edges = asyncio.run(client.get_graph(pageSize=5)); print(f'グラフ作成完了: {len(edges)}エッジ'); print('最初のエッジ例:', edges[0] if edges else 'None')"