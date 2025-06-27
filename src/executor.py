# src/executor.py
import logging, asyncio
from config import EXECUTE_TRADES

log = logging.getLogger(__name__)

async def execute_cycle(cycle, amount_in, est_gain):
    """
    DEV/Step-0:
      If EXECUTE_TRADES is False no transaction is sent; only logs are produced.
      Even if True, modify the TODO section before enabling.
    """
    if not EXECUTE_TRADES:
        log.info("[DRY-RUN] Would execute %s  amount %.6f  gain %.6f",
                 "→".join(cycle), amount_in, est_gain)
        return "dry-run"

    # ---- Actual send logic (build Raydium/Orca/Jupiter v4 Tx etc.) ----
    log.warning("EXECUTE_TRADES=True but the send code is not implemented")
    await asyncio.sleep(0)
