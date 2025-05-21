# src/clients/jupiter_client.py
"""
Jupiter TokenAPI クライアント

Todo 
1(テスト用). TokenAPIから認証済みトークンからN個を取得 (https://lite-api.jup.ag/tokens/v1/tagged/verified)
1. マイ取引対象フィルタにあるトークンを取得 (まだトークン選定の最適化を行っていないので、これは一旦保留とする)
2. 1で取得したトークンと有名SPL(wrappedSOL,USDC)のペアプールの情報を全てのDEXの場合で取得
3. さらにDEXごとの手数料やスワップにかかるガス代等の情報を取得し、スワップにかかる費用を双方向グラフとして表すための情報を返す(edges: List[(mint_u, mint_v, rate_uv)]のような形でReturnする。この時rate計算に用いる仮のスワップ元手金額は、INPUTのような定数とする)
3(補足). 各DEXごとに、トークンアドレスのリストを送るとそれらの情報を取得してくれる(DEX名)_client.pyのようなファイルを作成しておく 

-----------------------------
目的:
    - Jupiter が管理する「verified」SPL トークン一覧を取得するだけ
    - プール情報・レート取得は DEX 別クライアントへ委譲

"""

# src/clients/jupiter_client.py
import httpx
import logging
from typing import List, Dict

TOKEN_URL = "https://lite-api.jup.ag/tokens/v1/tagged/verified"
log = logging.getLogger(__name__)


class JupiterClient:
    """Jupiter TokenAPI から verified トークンを取得するだけのクライアント"""

    async def verified_tokens(self, limit: int = 250) -> List[Dict]:
        """
        Jupiter の verified トークン一覧を取得して JSON 配列で返す。

        Parameters
        ----------
        limit : int
            取得件数（API の上限は 250）

        Returns
        -------
        List[dict]
            トークンメタ情報のリスト
        """
        params = {"limit": limit}
        async with httpx.AsyncClient(timeout=10) as c:
            res = await c.get(TOKEN_URL, params=params)
        res.raise_for_status()

        data = res.json()
        log.info("Jupiter: %d verified tokens fetched", len(data))
        return data

    async def verified_mints(self, limit: int = 250) -> List[str]:
        """
        mint アドレスだけを抽出して返すラッパー。
        """
        tokens = await self.verified_tokens(limit)
        return [t["mint"] for t in tokens]
