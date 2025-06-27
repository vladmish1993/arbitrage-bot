import binascii
from typing import Optional

from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solana.transaction import Transaction
from solana.rpc.types import TxOpts

from config import SOLANA_RPC_URL, SECRET_KEY_HEX

class SolanaClient:
    """Thin wrapper class (send_transaction / airdrop)"""

    def __init__(self):
        self.client = AsyncClient(SOLANA_RPC_URL)
        if SECRET_KEY_HEX:
            secret = binascii.unhexlify(SECRET_KEY_HEX)
            self.keypair = Keypair.from_secret_key(secret)
        else:
            # Generate ephemeral wallet for devnet and airdrop
            self.keypair = Keypair()

    async def airdrop_if_needed(self, min_balance: int = 2_000_000_000):
        bal = await self.client.get_balance(self.keypair.public_key)
        if bal['result']['value'] < min_balance:
            print('Airdropping 2 SOL ...')
            await self.client.request_airdrop(self.keypair.public_key, 2_000_000_000)

    async def send_transaction(self, txn: Transaction) -> str:
        resp = await self.client.send_transaction(
            txn,
            self.keypair,
            opts=TxOpts(skip_preflight=True, preflight_commitment='confirmed')
        )
        return resp['result']