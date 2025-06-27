# src/clients/jupiter_client.py
"""
Jupiter API クライアント

-----------------------------
Purpose:
    - Retrieve token info and prices using Jupiter's TokenAPI and PriceAPI
    - Obtain metadata for verified tokens
    - Retrieve and store SOL based price data
    - Obtain optimal swap route (Quote API)
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Constants
TOKEN_URL = "https://lite-api.jup.ag/tokens/v1/tagged/verified"
PRICE_URL = "https://lite-api.jup.ag/price/v2"
SOL_MINT = "So11111111111111111111111111111111111111112"
MAX_IDS_PER_BATCH = 100  # Maximum mints per request for the Price API
DEFAULT_TIMEOUT = 10.0  # API timeout (seconds)
RATE_LIMIT_DELAY = 0.1  # Delay for rate limiting (seconds)
DEFAULT_OUTPUT_PATH = "src/data/tokens_info.json"

# Logger setup
log = logging.getLogger(__name__)


class JupiterClient:
    """
    Client that uses the Jupiter API to obtain token info, price data and swap routes
    
    Key features:
    - Retrieve list of verified tokens
    - Retrieve and store SOL based price data
    - Obtain optimal swap route
    
    Attributes
    ----------
    _client : httpx.AsyncClient
        HTTP client (connection pool reuse)
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        """
        Initialize JupiterClient
        
        Parameters
        ----------
        timeout : float, default 10.0
            API request timeout (seconds)
        """
        self._client = httpx.AsyncClient(timeout=timeout)
        log.debug("JupiterClient initialized with timeout=%.1fs", timeout)

    async def __aenter__(self) -> "JupiterClient":
        """Enter async context manager"""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager"""
        await self.aclose()

    async def aclose(self) -> None:
        """Clean up internal HTTP client"""
        await self._client.aclose()
        log.debug("JupiterClient closed")

    async def verified_tokens(self, limit: int = 250) -> List[Dict[str, Any]]:
        """
        Get a list of Jupiter verified tokens
        
        Parameters
        ----------
        limit : int, default 250
            Maximum number of tokens to fetch
            
        Returns
        -------
        List[Dict[str, Any]]
            Metadata objects for each token
            Each item includes address, symbol, name, decimals and so on
            
        Raises
        ------
        httpx.HTTPStatusError
            When the API request fails
        httpx.TimeoutException
            When the request times out
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
        Get list of mint addresses for verified tokens
        
        Parameters
        ----------
        limit : int, default 250
            Maximum number of tokens to fetch
            
        Returns
        -------
        List[str]
            List of mint addresses
            Invalid addresses are excluded
        """
        tokens = await self.verified_tokens(limit)
        
        # Handle API key names being either "mint" or "address"
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
        Get SOL-denominated prices and metadata and save to JSON
        
        Parameters
        ----------
        mints : List[str]
            List of mint addresses to fetch prices for
        output_path : Optional[Path], default None
            Output file path (default used if None)
        vs_token : str, default SOL_MINT
            Mint address of the base currency (usually SOL)
            
        Returns
        -------
        None
            Save price data and metadata to a JSON file (no return value)
            
        Raises
        ------
        httpx.HTTPStatusError
            When the API request fails
        httpx.TimeoutException
            When the request times out
        OSError
            When file writing fails
            
        Notes
        -----
        Process large numbers of mint addresses in batches of 100
        Call the API at appropriate intervals considering rate limits.
        
        Output file format:
        {
          "token_address_1": {
            "price": 0.123456,
            "decimals": 9,
            "symbol": "TOKEN",
            "name": "Token Name",
            ...
          },
          ...
        }
        """
        if not mints:
            log.warning("No mints provided for price update")
            return
        
        if output_path is None:
            output_path = Path(DEFAULT_OUTPUT_PATH)
        
        log.info("Starting price update for %d tokens", len(mints))
        
        # Fetch token metadata first
        log.info("Fetching token metadata...")
        all_tokens = await self.verified_tokens(limit=max(250, len(mints)))
        
        # Build a metadata dictionary keyed by mint address
        token_metadata = {}
        for token in all_tokens:
            mint_address = token.get("mint") or token.get("address")
            if mint_address:
                token_metadata[mint_address] = {
                    "symbol": token.get("symbol", ""),
                    "name": token.get("name", ""),
                    "decimals": token.get("decimals", 0),
                    "logoURI": token.get("logoURI", ""),
                    "tags": token.get("tags", []),
                    "extensions": token.get("extensions", {}),
                }
        
        results: Dict[str, Dict[str, Any]] = {}
        successful_batches = 0
        failed_batches = 0
        
        # Batch processing (100 at a time)
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
                
                # API response format: {"data": {"token_id": {"id": "...", "price": "..."}}}
                data_dict = response_data.get("data", {})
                
                batch_results = 0
                for token_id, token_data in data_dict.items():
                    if isinstance(token_data, dict) and "price" in token_data:
                        try:
                            price = float(token_data["price"])
                            
                            # Combine metadata and price
                            token_info = token_metadata.get(token_id, {}).copy()
                            token_info["price"] = price
                            token_info["address"] = token_id
                            
                            # Set default values when metadata is missing
                            if not token_info.get("symbol"):
                                token_info["symbol"] = token_id[:8]
                            if "decimals" not in token_info:
                                token_info["decimals"] = 9  # Solanaのデフォルト
                            
                            results[token_id] = token_info
                            batch_results += 1
                            log.debug("Price for %s (%s): %.8f", 
                                     token_info["symbol"], token_id[:8], price)
                        except (ValueError, TypeError) as e:
                            log.warning("Invalid price data for %s: %s", token_id, e)
                    else:
                        log.warning("Invalid token data format for %s: %s", token_id, token_data)
                
                successful_batches += 1
                log.debug("Batch %d completed: %d prices collected", batch_idx + 1, batch_results)
                
                # Rate limiting
                if batch_idx < len(batches) - 1:  # no wait needed for the last batch
                    await asyncio.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                failed_batches += 1
                log.error("Batch %d failed: %s", batch_idx + 1, e)
                continue
        
        if not results:
            raise ValueError("No price data retrieved")
        
        # Write file atomically
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to a temp file then rename atomically
            temp_path = output_path.with_suffix(".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=True)
            
            temp_path.replace(output_path)
            
            log.info("Price update completed: %d tokens saved to %s (success: %d/%d batches)", 
                    len(results), output_path, successful_batches, len(batches))
            
            if failed_batches > 0:
                log.warning("Some batches failed: %d/%d", failed_batches, len(batches))
                
        except OSError as e:
            log.error("Failed to write price file: %s", e)
            raise

    async def get_token_price(self, mint_address: str, vs_token: str = SOL_MINT) -> Optional[float]:
        """
        Get the price of a single token
        
        Parameters
        ----------
        mint_address : str
            Mint address of the token to price
        vs_token : str, default SOL_MINT
            Mint address of the base currency
            
        Returns
        -------
        Optional[float]
            Token price (None if unavailable)
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


# Test command (fetch token list)
# python -c "import asyncio,logging; logging.basicConfig(level=logging.INFO); from src.clients.jupiter_client import JupiterClient; exec('async def test():\n async with JupiterClient() as client:\n  tokens = await client.verified_tokens(limit=5)\n  print(f\"Found: {len(tokens)} tokens\")\n  print(\"Example:\", tokens[0] if tokens else \"None\")'); asyncio.run(test())"

# Test command (price update)
# python -c "import asyncio,logging; logging.basicConfig(level=logging.INFO); from src.clients.jupiter_client import JupiterClient; exec('async def test():\n async with JupiterClient() as client:\n  mints = await client.verified_mints(limit=10)\n  await client.update_token_prices(mints)'); asyncio.run(test())"