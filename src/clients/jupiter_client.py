# src/clients/jupiter_client.py
"""
Jupiter API クライアント

-----------------------------
目的:
    - Jupiter の TokenAPI / PriceAPI を使用してトークン情報と価格データを取得
    - verified トークンのメタ情報取得
    - SOL建て価格データの取得と保存
    - 最適スワップルートの取得（Quote API）
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# 定数
TOKEN_URL = "https://lite-api.jup.ag/tokens/v1/tagged/verified"
PRICE_URL = "https://lite-api.jup.ag/price/v2"
SOL_MINT = "So11111111111111111111111111111111111111112"
MAX_IDS_PER_BATCH = 100  # Price API に一度に渡せる最大 mint 数
DEFAULT_TIMEOUT = 10.0  # APIタイムアウト（秒）
RATE_LIMIT_DELAY = 0.1  # レート制限対応の待機時間（秒）
DEFAULT_OUTPUT_PATH = "src/data/verified_token_prices.json"

# ロガー設定
log = logging.getLogger(__name__)


class JupiterClient:
    """
    Jupiter API を使用してトークン情報、価格データ、スワップルートを取得するクライアント
    
    主な機能:
    - verified トークン一覧の取得
    - SOL建て価格データの取得と保存
    - 最適スワップルートの取得
    
    Attributes
    ----------
    _client : httpx.AsyncClient
        HTTP クライアント（接続プール再利用）
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        """
        JupiterClientを初期化
        
        Parameters
        ----------
        timeout : float, default 10.0
            APIリクエストのタイムアウト（秒）
        """
        self._client = httpx.AsyncClient(timeout=timeout)
        log.debug("JupiterClient initialized with timeout=%.1fs", timeout)

    async def __aenter__(self) -> "JupiterClient":
        """非同期コンテキストマネージャー（入場）"""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """非同期コンテキストマネージャー（退場）"""
        await self.aclose()

    async def aclose(self) -> None:
        """内部HTTPクライアントをクリーンアップ"""
        await self._client.aclose()
        log.debug("JupiterClient closed")

    async def verified_tokens(self, limit: int = 250) -> List[Dict[str, Any]]:
        """
        Jupiter の verified トークン一覧を取得
        
        Parameters
        ----------
        limit : int, default 250
            取得するトークンの最大数
            
        Returns
        -------
        List[Dict[str, Any]]
            各トークンのメタ情報オブジェクト
            各要素には address, symbol, name, decimals などが含まれる
            
        Raises
        ------
        httpx.HTTPStatusError
            APIリクエストが失敗した場合
        httpx.TimeoutException
            リクエストがタイムアウトした場合
        """
        params = {"limit": limit}
        
        try:
            log.debug("Fetching verified tokens: limit=%d", limit)
            response = await self._client.get(TOKEN_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                log.error("Unexpected API response format: expected list, got %s", type(data))
                raise ValueError("Unexpected API response format")
            
            log.info("Jupiter: %d verified tokens fetched", len(data))
            return data
            
        except httpx.HTTPStatusError as e:
            log.error("Jupiter Token API HTTP error: %s", e)
            raise
        except httpx.TimeoutException as e:
            log.error("Jupiter Token API timeout: %s", e)
            raise
        except Exception as e:
            log.error("Jupiter Token API unexpected error: %s", e)
            raise

    async def verified_mints(self, limit: int = 250) -> List[str]:
        """
        verified トークンのmintアドレス一覧を取得
        
        Parameters
        ----------
        limit : int, default 250
            取得するトークンの最大数
            
        Returns
        -------
        List[str]
            mintアドレスのリスト
            無効なアドレスは除外される
        """
        tokens = await self.verified_tokens(limit)
        
        # API のキー名が "mint" または "address" の可能性に対応
        mints = []
        for token in tokens:
            mint_address = token.get("mint") or token.get("address")
            if mint_address:
                mints.append(mint_address)
            else:
                log.warning("Token missing mint address: %s", token.get("symbol", "unknown"))
        
        log.info("Extracted %d valid mint addresses from %d tokens", len(mints), len(tokens))
        return mints

    async def update_token_prices(
        self,
        mints: List[str],
        output_path: Optional[Path] = None,
        vs_token: str = SOL_MINT
    ) -> None:
        """
        トークンのSOL建て価格を取得してJSONファイルに保存
        
        Parameters
        ----------
        mints : List[str]
            価格を取得するmintアドレスのリスト
        output_path : Optional[Path], default None
            出力ファイルパス（Noneの場合はデフォルトパスを使用）
        vs_token : str, default SOL_MINT
            基準通貨のmintアドレス（通常はSOL）
            
        Returns
        -------
        None
            価格データをJSONファイルに保存（戻り値なし）
            
        Raises
        ------
        httpx.HTTPStatusError
            APIリクエストが失敗した場合
        httpx.TimeoutException
            リクエストがタイムアウトした場合
        OSError
            ファイル書き込みに失敗した場合
            
        Notes
        -----
        大量のmintアドレスを100件ずつのバッチに分けて処理し、
        レート制限を考慮して適切な間隔で API を呼び出します。
        
        出力ファイル形式:
        {
          "token_address_1": 0.123456,
          "token_address_2": 1.000000,
          ...
        }
        """
        if not mints:
            log.warning("No mints provided for price update")
            return
        
        if output_path is None:
            output_path = Path(DEFAULT_OUTPUT_PATH)
        
        log.info("Starting price update for %d tokens", len(mints))
        
        results: Dict[str, float] = {}
        successful_batches = 0
        failed_batches = 0
        
        # バッチ処理（100件ずつ）
        batches = [mints[i:i + MAX_IDS_PER_BATCH] for i in range(0, len(mints), MAX_IDS_PER_BATCH)]
        
        for batch_idx, batch in enumerate(batches):
            try:
                params = {
                    "ids": ",".join(batch),
                    "vsToken": vs_token,
                }
                
                log.debug("Processing batch %d/%d (%d tokens)", 
                         batch_idx + 1, len(batches), len(batch))
                
                response = await self._client.get(PRICE_URL, params=params)
                response.raise_for_status()
                
                response_data = response.json()
                log.debug("Price API response keys: %s", list(response_data.keys()))
                
                # APIレスポンス形式: {"data": {"token_id": {"id": "...", "price": "..."}}}
                data_dict = response_data.get("data", {})
                
                batch_results = 0
                for token_id, token_data in data_dict.items():
                    if isinstance(token_data, dict) and "price" in token_data:
                        try:
                            price = float(token_data["price"])
                            results[token_id] = price
                            batch_results += 1
                            log.debug("Price for %s: %.8f", token_id[:8], price)
                        except (ValueError, TypeError) as e:
                            log.warning("Invalid price data for %s: %s", token_id, e)
                    else:
                        log.warning("Invalid token data format for %s: %s", token_id, token_data)
                
                successful_batches += 1
                log.debug("Batch %d completed: %d prices collected", batch_idx + 1, batch_results)
                
                # レート制限対応
                if batch_idx < len(batches) - 1:  # 最後のバッチでは待機不要
                    await asyncio.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                failed_batches += 1
                log.error("Batch %d failed: %s", batch_idx + 1, e)
                continue
        
        if not results:
            raise ValueError("No price data retrieved")
        
        # ファイル書き込み（アトミック操作）
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 一時ファイルに書き込み後、原子的にリネーム
            temp_path = output_path.with_suffix(".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=True)
            
            temp_path.replace(output_path)
            
            log.info("Price update completed: %d prices saved to %s (success: %d/%d batches)", 
                    len(results), output_path, successful_batches, len(batches))
            
            if failed_batches > 0:
                log.warning("Some batches failed: %d/%d", failed_batches, len(batches))
                
        except OSError as e:
            log.error("Failed to write price file: %s", e)
            raise

    async def get_token_price(self, mint_address: str, vs_token: str = SOL_MINT) -> Optional[float]:
        """
        単一トークンの価格を取得
        
        Parameters
        ----------
        mint_address : str
            価格を取得するトークンのmintアドレス
        vs_token : str, default SOL_MINT
            基準通貨のmintアドレス
            
        Returns
        -------
        Optional[float]
            トークン価格（取得できない場合はNone）
        """
        try:
            params = {
                "ids": mint_address,
                "vsToken": vs_token,
            }
            
            response = await self._client.get(PRICE_URL, params=params)
            response.raise_for_status()
            
            response_data = response.json()
            data_dict = response_data.get("data", {})
            
            if mint_address in data_dict:
                token_data = data_dict[mint_address]
                if isinstance(token_data, dict) and "price" in token_data:
                    price = float(token_data["price"])
                    log.debug("Price for %s: %.8f %s", mint_address[:8], price, vs_token[:8])
                    return price
            
            log.warning("No price data found for token: %s", mint_address)
            return None
            
        except Exception as e:
            log.error("Failed to get price for %s: %s", mint_address, e)
            return None


# TEST用コマンド（トークン一覧取得）
# python -c "import asyncio,logging; logging.basicConfig(level=logging.INFO); from src.clients.jupiter_client import JupiterClient; exec('async def test():\n async with JupiterClient() as client:\n  tokens = await client.verified_tokens(limit=5)\n  print(f\"取得: {len(tokens)}トークン\")\n  print(\"例:\", tokens[0] if tokens else \"None\")'); asyncio.run(test())"

# TEST用コマンド（価格更新）
# python -c "import asyncio,logging; logging.basicConfig(level=logging.INFO); from src.clients.jupiter_client import JupiterClient; exec('async def test():\n async with JupiterClient() as client:\n  mints = await client.verified_mints(limit=10)\n  await client.update_token_prices(mints)'); asyncio.run(test())"