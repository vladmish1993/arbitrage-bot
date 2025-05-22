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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

# 定数
POOL_URL = "https://api-v3.raydium.io/pools/info/list"
SOL_ADDRESS = "So11111111111111111111111111111111111111112"  # Wrapped SOL address
VERIFIED_TOKENS_PATH = "src/data/verified_token_prices.json"
MIN_TVL_THRESHOLD = 5_000.0  # 最小TVL閾値（USD）
DEFAULT_TIMEOUT = 10.0  # APIタイムアウト（秒）

# ロガー設定
log = logging.getLogger(__name__)


class RaydiumClient:
    """
    Raydium APIから取得した情報を基に、アービトラージ用のグラフデータを生成するクライアント
    
    Attributes
    ----------
    _token_prices : Optional[Dict[str, float]]
        SOL建てトークン価格データのキャッシュ
    """

    def __init__(self, token_prices_path: Optional[str] = None) -> None:
        """
        RaydiumClientを初期化
        
        Parameters
        ----------
        token_prices_path : Optional[str]
            トークン価格JSONファイルのパス（Noneの場合はデフォルトパスを使用）
        """
        self._token_prices: Optional[Dict[str, float]] = None
        self._token_prices_path = token_prices_path or VERIFIED_TOKENS_PATH
        
    def _load_token_prices(self) -> Dict[str, float]:
        """
        トークン価格データを読み込み（キャッシュ機能付き）
        
        Returns
        -------
        Dict[str, float]
            トークンアドレス -> SOL建て価格のマッピング
            
        Raises
        ------
        FileNotFoundError
            価格ファイルが見つからない場合
        json.JSONDecodeError
            JSONファイルの形式が不正な場合
        """
        if self._token_prices is None:
            try:
                prices_path = Path(self._token_prices_path)
                if not prices_path.exists():
                    log.error("Token prices file not found: %s", prices_path)
                    raise FileNotFoundError(f"Token prices file not found: {prices_path}")
                
                with open(prices_path, 'r', encoding='utf-8') as f:
                    self._token_prices = json.load(f)
                    
                log.info("Token prices loaded: %d tokens", len(self._token_prices))
                
            except json.JSONDecodeError as e:
                log.error("Invalid JSON in token prices file: %s", e)
                raise
            except Exception as e:
                log.error("Failed to load token prices: %s", e)
                raise
                
        return self._token_prices


    async def get_raydium_graph(
        self,
        poolType: str = "standard",
        poolSortField: str = "liquidity",
        sortType: str = "desc",
        pageSize: int = 500,
        page: int = 1
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Raydium APIから取得したプールペアから、ベルマン・フォード用のグラフを作成
        
        Parameters
        ----------
        poolType : str, default "standard"
            プールの種類
        poolSortField : str, default "liquidity"
            ソートフィールド
        sortType : str, default "desc"
            ソート順序（desc: 降順, asc: 昇順）
        pageSize : int, default 500
            1ページの最大取得プール数（APIの上限は1000）
        page : int, default 1
            取得するページ番号
            
        Returns
        -------
        List[Tuple[str, str, float, Dict]]
            ベルマン・フォード用のエッジリスト
            各要素: (from_token_address, to_token_address, weight, pool_data)
            where:
                - from_token_address: 交換元トークンアドレス
                - to_token_address: 交換先トークンアドレス  
                - weight: 交換重み（対数変換済み）
                - pool_data: プール詳細情報（デバッグ用）
                
        Raises
        ------
        httpx.HTTPStatusError
            APIリクエストが失敗した場合
        httpx.TimeoutException
            リクエストがタイムアウトした場合
        """
            
        params = {
            "poolType": poolType,
            "poolSortField": poolSortField,
            "sortType": sortType,
            "pageSize": pageSize,
            "page": page
        }

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.get(POOL_URL, params=params)
            
            response.raise_for_status()
            data = response.json()
            
            # APIレスポンスの構造: data.data.data がプール配列
            pools_data = data.get("data", {}).get("data", [])
            log.info("Raydium: %d pools fetched for graph creation", len(pools_data))
            
            edges: List[Tuple[str, str, float, Dict]] = []
            valid_pools = 0
            
            for pool in pools_data:
                # プールデータの抽出と検証
                if not self._validate_pool_data(pool):
                    continue
                
                valid_pools += 1
                    
                # トークン情報の抽出
                token_a = pool["mintA"]
                token_b = pool["mintB"]
                
                token_a_addr, token_b_addr = token_a["address"], token_b["address"]

                base_info = {
                    "pool_id": pool["id"],
                    "price": pool["price"],
                    "tvl": pool["tvl"],
                    "fee_rate": pool["feeRate"],
                    "liquidity_a": float(pool["mintAmountA"]),
                    "liquidity_b": float(pool["mintAmountB"]),
                }
                
                # ---------- A → B 方向 ----------
                info_ab = copy.deepcopy(base_info)
                info_ab["token_a"] = {
                    "address": token_a_addr,
                    "symbol": token_a.get("symbol"),
                    "decimals": token_a.get("decimals"),
                }
                info_ab["token_b"] = {
                    "address": token_b_addr,
                    "symbol": token_b.get("symbol"),
                    "decimals": token_b.get("decimals"),
                }

                w_ab = self._calculate_weight(info_ab, "A_TO_B")
                if math.isfinite(w_ab):
                    edges.append((token_a_addr, token_b_addr, w_ab, info_ab))

                # ---------- B → A 方向 ----------
                info_ba = copy.deepcopy(base_info)
                # token と流動性をスワップ
                info_ba["token_a"] = {
                    "address": token_b_addr,
                    "symbol": token_b.get("symbol"),
                    "decimals": token_b.get("decimals"),
                }
                info_ba["token_b"] = {
                    "address": token_a_addr,
                    "symbol": token_a.get("symbol"),
                    "decimals": token_a.get("decimals"),
                }
                info_ba["liquidity_a"], info_ba["liquidity_b"] = \
                    info_ba["liquidity_b"], info_ba["liquidity_a"]

                w_ba = self._calculate_weight(info_ba, "B_TO_A")
                if math.isfinite(w_ba):
                    edges.append((token_b_addr, token_a_addr, w_ba, info_ba))
            
            log.info("Graph created: %d edges from %d valid pools (%d total pools)", 
                    len(edges), valid_pools, len(pools_data))
            return edges
        
        except httpx.HTTPStatusError as e:
            log.error("Raydium API HTTP error: %s", e)
            raise
        except httpx.TimeoutException as e:
            log.error("Raydium API timeout: %s", e)
            raise
        except Exception as e:
            log.error("Raydium API unexpected error: %s", e)
            raise

    def _validate_pool_data(self, pool: Dict) -> bool:
        """
        プールデータの妥当性を検証
        
        Parameters
        ----------
        pool : Dict
            プールデータ
            
        Returns
        -------
        bool
            妥当性検証結果
        """
        required_fields = ["id", "mintA", "mintB", "price", "tvl", "feeRate", "mintAmountA", "mintAmountB"]
        
        for field in required_fields:
            if field not in pool or pool[field] is None:
                log.debug("Pool %s missing required field: %s", pool.get("id"), field)
                return False
        
        # TVLとpriceの最小値チェック（アービトラージに不適なプールを除外）
        tvl = pool.get("tvl", 0)
        if tvl < MIN_TVL_THRESHOLD:
            log.debug("Pool %s TVL too low: $%.2f", pool.get("id"), tvl)
            return False
            
        price = pool.get("price", 0)
        if price <= 0:
            log.debug("Pool %s invalid price: %s", pool.get("id"), price)
            return False
        
        # 流動性チェック
        liquidity_a = float(pool.get("mintAmountA", 0))
        liquidity_b = float(pool.get("mintAmountB", 0))
        if liquidity_a <= 0 or liquidity_b <= 0:
            log.debug("Pool %s invalid liquidity: A=%s, B=%s", pool.get("id"), liquidity_a, liquidity_b)
            return False
            
        return True

    def _calculate_weight(self, pool_info: Dict, direction: str) -> float:
        """
        定数積モデルを使用したエッジ重み計算
        
        Parameters
        ----------
        pool_info : Dict
            プール情報
        direction : str
            交換方向（"A_TO_B" or "B_TO_A"）
            
        Returns
        -------
        float
            ベルマン・フォード用の重み（対数変換済み）
            
        Notes
        -----
        spot price (dx→0)を使用して実効レートを計算:
        1. 手数料を差し引き
        2. 実効レート = 出力トークンリザーブ / 入力トークンリザーブ
        3. 重み = -log(実効レート)
        """
        ra_raw = pool_info["liquidity_a"]
        rb_raw = pool_info["liquidity_b"]

        dec_a = pool_info["token_a"]["decimals"]
        dec_b = pool_info["token_b"]["decimals"]

        try:
            dec_a = int(dec_a)
            dec_b = int(dec_b)
        except (TypeError, ValueError):
            return float("inf")

        # decimals 補正済みリザーブ
        ra = ra_raw / (10 ** dec_a)
        rb = rb_raw / (10 ** dec_b)

        raw_fee = pool_info["fee_rate"]
        if raw_fee is None:
            return float("inf")
        fee = raw_fee / 100 if raw_fee > 1 else raw_fee
        
        try:
            if direction == "A_TO_B":
                # A -> B: トークンAを投入してトークンBを受取
                # 実効レート
                effective_rate = (1.0 - fee) * rb / ra
                
            else:
                # B -> A: トークンBを投入してトークンAを受取
                # 実効レート
                effective_rate = (1.0 - fee) * ra / rb
            
            # 異常値チェック
            if effective_rate <= 0 or not math.isfinite(effective_rate):
                return float('inf')
            
            # ベルマン・フォード用の対数重み
            weight = -math.log(effective_rate)
            
            # 数値的安定性チェック
            if not math.isfinite(weight):
                return float('inf')
                
            return weight
            
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            log.debug("Weight calculation error for pool %s direction %s: %s", 
                     pool_info.get("pool_id"), direction, e)
            return float('inf')

    async def print_raydium_pools(
        self,
        poolType: str = "standard",
        poolSortField: str = "liquidity",
        sortType: str = "desc",
        pageSize: int = 500,
        page: int = 1
    ) -> None:
        """
        Raydium APIから取得したプール情報からアービトラージに重要な指標のみを抜き出してログに出力
        
        Parameters
        ----------
        poolType : str, default "standard"
            プールの種類
        poolSortField : str, default "liquidity"
            ソートフィールド
        sortType : str, default "desc"
            ソート順序（desc: 降順, asc: 昇順）
        pageSize : int, default 500
            1ページの最大取得プール数（APIの上限は1000）
        page : int, default 1
            取得するページ番号
            
        Returns
        -------
        None
            アービトラージ指標をログに出力（戻り値なし）
            
        Raises
        ------
        httpx.HTTPStatusError
            APIリクエストが失敗した場合
        httpx.TimeoutException
            リクエストがタイムアウトした場合
        """
        params = {
            "poolType": poolType,
            "poolSortField": poolSortField,
            "sortType": sortType,
            "pageSize": pageSize,
            "page": page
        }

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.get(POOL_URL, params=params)
            
            response.raise_for_status()
            data = response.json()
            
            # APIレスポンスの構造: data.data.data がプール配列
            pools_data = data.get("data", {}).get("data", [])
            
            # アービトラージに重要な指標のみ抜き出し
            arbitrage_data = []
            for pool in pools_data:
                pool_info = {
                    "pool_id": pool.get("id"),
                    "token_a": {
                        "address": pool.get("mintA", {}).get("address"),
                        "symbol": pool.get("mintA", {}).get("symbol"),
                        "decimals": pool.get("mintA", {}).get("decimals")
                    },
                    "token_b": {
                        "address": pool.get("mintB", {}).get("address"),
                        "symbol": pool.get("mintB", {}).get("symbol"),
                        "decimals": pool.get("mintB", {}).get("decimals")
                    },
                    "price": pool.get("price"),
                    "tvl": pool.get("tvl"),
                    "fee_rate": pool.get("feeRate"),
                    "liquidity_a": pool.get("mintAmountA"),
                    "liquidity_b": pool.get("mintAmountB"),
                    "daily_volume": pool.get("day", {}).get("volume"),
                    "daily_volume_quote": pool.get("day", {}).get("volumeQuote"),
                    "daily_apr": pool.get("day", {}).get("apr")
                }
                arbitrage_data.append(pool_info)
            
            # JSONをログに出力
            log.info("Raydium arbitrage data (%d pools):\n%s", 
                    len(arbitrage_data), 
                    json.dumps(arbitrage_data, indent=2, ensure_ascii=False))
        
        except httpx.HTTPStatusError as e:
            log.error("Raydium API HTTP error: %s", e)
            raise
        except httpx.TimeoutException as e:
            log.error("Raydium API timeout: %s", e)
            raise
        except Exception as e:
            log.error("Raydium API unexpected error: %s", e)
            raise


# TEST用コマンド（データ出力）
# python -c "import asyncio; import logging; logging.basicConfig(level=logging.INFO); from src.clients.raydium_client import RaydiumClient; client = RaydiumClient(); asyncio.run(client.print_raydium_pools(pageSize=3))"

# TEST用コマンド（グラフ作成）  
# python -c "import asyncio; from src.clients.raydium_client import RaydiumClient; client = RaydiumClient(); edges = asyncio.run(client.get_raydium_graph(1.0, pageSize=5)); print(f'グラフ作成完了: {len(edges)}エッジ'); print('最初のエッジ例:', edges[0] if edges else 'None')"