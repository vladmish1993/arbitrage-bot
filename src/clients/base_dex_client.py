# src/clients/base_dex_client.py
"""
DEX API クライアントの基底クラス

-----------------------------
目的:
    - 各DEXの共通機能を抽象化
    - ベルマン・フォードアルゴリズム用のグラフデータ生成の共通ロジック
    - トークン価格管理、プール検証、重み計算等の共通処理
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 共通定数
SOL_ADDRESS = "So11111111111111111111111111111111111111112"  # Wrapped SOL address
VERIFIED_TOKENS_PATH = "src/data/verified_token_prices.json"
MIN_TVL_THRESHOLD = 5_000.0  # 最小TVL閾値（USD）
DEFAULT_TIMEOUT = 10.0  # APIタイムアウト（秒）

# ロガー設定
log = logging.getLogger(__name__)


class BaseDexClient(ABC):
    """
    DEX APIクライアントの基底クラス
    
    各DEXの共通機能を提供し、固有の実装は派生クラスで行う
    
    Attributes
    ----------
    _token_prices : Optional[Dict[str, float]]
        SOL建てトークン価格データのキャッシュ
    _token_prices_path : str
        トークン価格JSONファイルのパス
    """

    def __init__(self, token_prices_path: Optional[str] = None) -> None:
        """
        BaseDexClientを初期化
        
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

    def _validate_pool_data(self, pool: Dict, required_fields: List[str]) -> bool:
        """
        プールデータの妥当性を検証（基本検証）
        
        Parameters
        ----------
        pool : Dict
            プールデータ
        required_fields : List[str]
            必須フィールドのリスト
            
        Returns
        -------
        bool
            妥当性検証結果
        """
        # 必須フィールドの存在チェック
        for field in required_fields:
            if field not in pool or pool[field] is None:
                log.debug("Pool %s missing required field: %s", pool.get("id"), field)
                return False
        
        # TVLの最小値チェック（アービトラージに不適なプールを除外）
        tvl = pool.get("tvl", 0)
        if tvl < MIN_TVL_THRESHOLD:
            log.debug("Pool %s TVL too low: $%.2f", pool.get("id"), tvl)
            return False
            
        # 価格の妥当性チェック
        price = pool.get("price", 0)
        if price <= 0:
            log.debug("Pool %s invalid price: %s", pool.get("id"), price)
            return False
        
        return True

    def _calculate_weight(self, pool_info: Dict, direction: str) -> float:
        """
        定数積モデルを使用したエッジ重み計算
        
        Parameters
        ----------
        pool_info : Dict
            プール情報（以下のキーを含む必要がある）:
            - liquidity_a: トークンAの流動性
            - liquidity_b: トークンBの流動性
            - token_a: トークンA情報（decimals含む）
            - token_b: トークンB情報（decimals含む）
            - fee_rate: 手数料率
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

    def _validate_liquidity(self, liquidity_a: float, liquidity_b: float, pool_id: str = None) -> bool:
        """
        流動性の妥当性をチェック
        
        Parameters
        ----------
        liquidity_a : float
            トークンAの流動性
        liquidity_b : float
            トークンBの流動性
        pool_id : str, optional
            プールID（ログ用）
            
        Returns
        -------
        bool
            流動性が妥当かどうか
        """
        if liquidity_a <= 0 or liquidity_b <= 0:
            log.debug("Pool %s invalid liquidity: A=%s, B=%s", pool_id, liquidity_a, liquidity_b)
            return False
        return True

    @abstractmethod
    async def get_graph(self, **kwargs) -> List[Tuple[str, str, float, Dict]]:
        """
        DEXからプール情報を取得し、ベルマン・フォード用のエッジリストを返す
        
        Parameters
        ----------
        **kwargs
            DEX固有のパラメータ
            
        Returns
        -------
        List[Tuple[str, str, float, Dict]]
            エッジリスト: (from_token, to_token, weight, pool_info)
        """
        pass

    @abstractmethod
    async def print_pools(self, **kwargs) -> None:
        """
        DEXのプール情報をログに出力
        
        Parameters
        ----------
        **kwargs
            DEX固有のパラメータ
        """
        pass

    @property
    @abstractmethod
    def dex_name(self) -> str:
        """
        DEX名を返す
        
        Returns
        -------
        str
            DEX名
        """
        pass