# src/executor.py
import logging, asyncio
from config import EXECUTE_TRADES

log = logging.getLogger(__name__)

async def execute_cycle(cycle, amount_in, est_gain):
    """
    DEV/Step-0:
      EXECUTE_TRADES=False なら送信せずログのみ。
      True の場合でも TODO 部分を書き換えてから有効化してください。
    """
    if not EXECUTE_TRADES:
        log.info("[DRY-RUN] Would execute %s  amount %.6f  gain %.6f",
                 "→".join(cycle), amount_in, est_gain)
        return "dry-run"

    # ---- 実送信ロジック (Raydium/Orca/Jupiter v4 Tx 構築 etc.) ----
    log.warning("EXECUTE_TRADES=True ですが送信コードは未実装です")
    await asyncio.sleep(0)
