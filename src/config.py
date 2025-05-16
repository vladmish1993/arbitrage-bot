# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SOLANA_RPC_URL      = os.getenv("SOLANA_RPC_URL")
SQLITE_PATH         = Path(os.getenv("SQLITE_PATH", "/app/data/cache.db"))

# --- DexScreener ----------------------------------------------------------
DEXSCREENER_PAGE_URL = "https://api.dexscreener.com/latest/dex/pairs/solana"
DEX_PAGE_LIMIT  = int(os.getenv("DEX_PAGE_LIMIT", 10))    # 1ページ＝~30ペア
DEX_CONCURRENCY = int(os.getenv("DEX_CONCURRENCY", 5))    # 同時コネクション


# ---- ボット動作 ------------------------------------------------------------
NEG_CYCLE_THRESHOLD = float(os.getenv("NEG_CYCLE_THRESHOLD", "-1e-6"))
EXECUTE_TRADES      = os.getenv("EXECUTE_TRADES", "False").lower() == "true"
LOG_LEVEL           = os.getenv("LOG_LEVEL", "INFO")
