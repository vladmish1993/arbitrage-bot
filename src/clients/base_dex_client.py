# src/clients/base_dex_client.py
"""
Base class for DEX API clients

-----------------------------
Purpose:
    - Abstract common functionality of each DEX
    - Shared logic to build graph data for the Bellman-Ford algorithm
    - Common routines for token price management, pool validation and weight calculation
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Common constants
SOL_ADDRESS = "So11111111111111111111111111111111111111112"  # Wrapped SOL address
VERIFIED_TOKENS_PATH = "src/data/verified_token_prices.json"
MIN_TVL_THRESHOLD = 5_000.0  # Minimum TVL threshold (USD)
DEFAULT_TIMEOUT = 10.0  # API timeout (seconds)

# Logger setup
log = logging.getLogger(__name__)


class BaseDexClient(ABC):
    """
    Base class for DEX API clients
    
    Provides common functionality for each DEX; subclass handles specific implementations
    
    Attributes
    ----------
    _token_prices : Optional[Dict[str, float]]
        Cache of token prices in SOL
    _token_prices_path : str
        Path to the token price JSON file
    """

    def __init__(self, token_prices_path: Optional[str] = None) -> None:
        """
        Initialize BaseDexClient
        
        Parameters
        ----------
        token_prices_path : Optional[str]
            Path to the token price JSON file(use default if None)
        """
        self._token_prices: Optional[Dict[str, float]] = None
        self._token_prices_path = token_prices_path or VERIFIED_TOKENS_PATH
        
    def _load_token_prices(self) -> Dict[str, float]:
        """
        Load token price data with caching
        
        Returns
        -------
        Dict[str, float]
            Mapping from token address to price in SOL
            
        Raises
        ------
        FileNotFoundError
            When the price file is not found
        json.JSONDecodeError
            When the JSON file format is invalid
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
        Validate pool data (basic checks)
        
        Parameters
        ----------
        pool : Dict
            プールデータ
        required_fields : List[str]
            List of required fields
            
        Returns
        -------
        bool
            Validation result
        """
        # Check for required fields
        for field in required_fields:
            if field not in pool or pool[field] is None:
                log.debug("Pool %s missing required field: %s", pool.get("id"), field)
                return False
        
        # Minimum TVL check (exclude pools unsuitable for arbitrage)
        tvl = pool.get("tvl", 0)
        if tvl < MIN_TVL_THRESHOLD:
            log.debug("Pool %s TVL too low: $%.2f", pool.get("id"), tvl)
            return False
            
        # Validate price
        price = pool.get("price", 0)
        if price <= 0:
            log.debug("Pool %s invalid price: %s", pool.get("id"), price)
            return False
        
        return True

    def _calculate_weight(self, pool_info: Dict, direction: str) -> float:
        """
        Edge weight calculation using the constant product model
        
        Parameters
        ----------
        pool_info : Dict
            Pool information (must include the following keys):
            - liquidity_a: liquidity of token A
            - liquidity_b: liquidity of token B
            - token_a: token A info (including decimals)
            - token_b: token B info (including decimals)
            - fee_rate: fee rate
        direction : str
            swap direction ("A_TO_B" or "B_TO_A")
            
        Returns
        -------
        float
            Weight for Bellman-Ford (log transformed)
            
        Notes
        -----
        Calculate effective rate using spot price (dx→0):
        1. Subtract fees
        2. Effective rate = output token reserve / input token reserve
        3. Weight = -log(effective rate)
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

        # Reserves adjusted for decimals
        ra = ra_raw / (10 ** dec_a)
        rb = rb_raw / (10 ** dec_b)

        raw_fee = pool_info["fee_rate"]
        if raw_fee is None:
            return float("inf")
        fee = raw_fee / 100 if raw_fee > 1 else raw_fee
        
        try:
            if direction == "A_TO_B":
                # A -> B: Swap token A for token B
                # Effective rate
                effective_rate = (1.0 - fee) * rb / ra
                
            else:
                # B -> A: Swap token B for token A
                # Effective rate
                effective_rate = (1.0 - fee) * ra / rb
            
            # Check for abnormal values
            if effective_rate <= 0 or not math.isfinite(effective_rate):
                return float('inf')
            
            # Log weight for Bellman-Ford
            weight = -math.log(effective_rate)
            
            # Numerical stability check
            if not math.isfinite(weight):
                return float('inf')
                
            return weight
            
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            log.debug("Weight calculation error for pool %s direction %s: %s", 
                     pool_info.get("pool_id"), direction, e)
            return float('inf')

    def _validate_liquidity(self, liquidity_a: float, liquidity_b: float, pool_id: str = None) -> bool:
        """
        Check validity of liquidity
        
        Parameters
        ----------
        liquidity_a : float
            liquidity of token A
        liquidity_b : float
            liquidity of token B
        pool_id : str, optional
            Pool ID (for logging)
            
        Returns
        -------
        bool
            Whether the liquidity values are valid
        """
        if liquidity_a <= 0 or liquidity_b <= 0:
            log.debug("Pool %s invalid liquidity: A=%s, B=%s", pool_id, liquidity_a, liquidity_b)
            return False
        return True

    @abstractmethod
    async def get_graph(self, **kwargs) -> List[Tuple[str, str, float, Dict]]:
        """
        Retrieve pool information from the DEX and return edges for Bellman-Ford
        
        Parameters
        ----------
        **kwargs
            DEX specific parameters
            
        Returns
        -------
        List[Tuple[str, str, float, Dict]]
            Edge list: (from_token, to_token, weight, pool_info)
        """
        pass

    @abstractmethod
    async def print_pools(self, **kwargs) -> None:
        """
        Print DEX pool information to the log
        
        Parameters
        ----------
        **kwargs
            DEX specific parameters
        """
        pass

    @property
    @abstractmethod
    def dex_name(self) -> str:
        """
        Return the DEX name
        
        Returns
        -------
        str
            DEX name
        """
        pass