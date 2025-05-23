# src/__main__.py
"""
アービトラージボット メイン実行モジュール

-----------------------------
機能:
    1. トークン価格取得・更新
    2. Raydiumプールからグラフ作成  
    3. 負のサイクル（アービトラージ機会）探索
    4. 最適投入額決定（未実装）
    5. テスト実行機能
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# クライアントとコア機能のインポート
from src.clients.jupiter_client import JupiterClient
from src.clients.raydium_client import RaydiumClient
from src.graph.arbitrage_path_finder import ArbitragePathFinder
# from executor import execute_cycle  # 未実装のためコメントアウト

# 設定
DEFAULT_POOL_SIZE = 100   # 取得するプール数
MIN_PROFIT_THRESHOLD = 0.001  # 最小利益率 0.1%
MAX_CYCLES_TO_FIND = 5    # 最大検出サイクル数
LOG_LEVEL = logging.INFO

# ロガー設定
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("ArbitrageBot")


class ArbitrageBot:
    """
    アービトラージボットのメインクラス
    
    トークン価格取得からアービトラージ機会検出までの
    一連のプロセスを管理します。
    """
    
    def __init__(self):
        """ボットを初期化"""
        self.jupiter_client: Optional[JupiterClient] = None
        self.raydium_client: Optional[RaydiumClient] = None
        self.path_finder: Optional[ArbitragePathFinder] = None
        
    async def __aenter__(self):
        """非同期コンテキストマネージャー開始"""
        self.jupiter_client = JupiterClient()
        self.raydium_client = RaydiumClient()
        self.path_finder = ArbitragePathFinder()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャー終了"""
        if self.jupiter_client:
            await self.jupiter_client.aclose()

    async def update_token_prices(self, limit: int = 250) -> bool:
        """
        Jupiter APIからトークン価格を取得・更新
        
        Parameters
        ----------
        limit : int, default 250
            取得するトークン数の上限
            
        Returns
        -------
        bool
            価格取得の成功可否
        """
        log.info("🔄 Starting token price update (limit: %d)", limit)
        
        try:
            # verified トークンリストを取得
            verified_mints = await self.jupiter_client.verified_mints(limit=limit)
            log.info("📋 Retrieved %d verified token mints", len(verified_mints))
            
            # SOL建て価格を取得・保存
            await self.jupiter_client.update_token_prices(
                mints=verified_mints,
                output_path=Path("src/data/verified_token_prices.json")
            )
            
            log.info("✅ Token prices updated successfully")
            return True
            
        except Exception as e:
            log.error("❌ Failed to update token prices: %s", e)
            return False

    async def create_arbitrage_graph(
        self, 
        pool_size: int = DEFAULT_POOL_SIZE
    ) -> bool:
        """
        Raydiumプールデータからアービトラージ用グラフを作成
        
        Parameters
        ----------
        pool_size : int, default 100
            取得するプール数
            
        Returns
        -------
        bool
            グラフ作成の成功可否
        """
        log.info("🏗️ Creating arbitrage graph (pools: %d)", 
                 pool_size)
        
        try:
            # Raydiumプールからエッジデータを取得
            edges = await self.raydium_client.get_raydium_graph(
                pageSize=pool_size,
                max_pages=3
            )
            
            if not edges:
                log.warning("⚠️ No edges retrieved from Raydium")
                return False
                
            # グラフを構築
            self.path_finder.build_graph_from_raydium_edges(edges)
            
            # グラフ統計を表示
            stats = self.path_finder.get_graph_stats()
            log.info("📊 Graph created: %d vertices, %d edges, density: %.3f", 
                    stats['vertices'], stats['edges'], stats['density'])
            
            print("=== Graph Statistics ===")
            for key, value in stats.items():
                print(f"{key:25s}: {value}")
            print("=== ================ ===")
            
            return True
            
        except Exception as e:
            log.error("❌ Failed to create arbitrage graph: %s", e)
            return False

    async def find_arbitrage_opportunities(
        self,
        min_profit: float = MIN_PROFIT_THRESHOLD,
        max_cycles: int = MAX_CYCLES_TO_FIND
    ) -> List[Dict]:
        """
        負のサイクル（アービトラージ機会）を探索
        
        Parameters
        ----------
        min_profit : float, default 0.005
            最小利益率（0.5%）
        max_cycles : int, default 5
            最大検出サイクル数
            
        Returns
        -------
        List[Dict]
            検出されたアービトラージ機会のリスト
        """
        log.info("🔍 Searching for arbitrage opportunities (min profit: %.2f%%)", 
                min_profit * 100)
        
        try:
            # 負のサイクルを検出
            cycles = self.path_finder.find_negative_cycles(
                max_cycles=max_cycles,
                min_profit_threshold=min_profit
            )
            
            if cycles:
                log.info("🎯 Found %d profitable arbitrage opportunities:", len(cycles))
                
                for i, cycle in enumerate(cycles, 1):
                    profit_pct = cycle['profit_rate'] * 100
                    cycle_length = cycle['cycle_length']
                    
                    # サイクルのトークンシンボルを取得
                    symbols = []
                    for detail in cycle['path_details']:
                        symbols.append(detail.get('from_symbol', 'Unknown'))
                    symbols.append(cycle['path_details'][-1].get('to_symbol', 'Unknown'))
                    
                    log.info("  %d. %s (%.2f%% profit, %d hops)", 
                            i, " → ".join(symbols), profit_pct, cycle_length)
                    
                    # 詳細情報をデバッグログで出力
                    log.debug("     Path details:")
                    for j, detail in enumerate(cycle['path_details']):
                        log.debug("       %d. %s → %s (TVL: $%.0f, Fee: %.2f%%)",
                                j+1, detail['from_symbol'], detail['to_symbol'],
                                detail.get('tvl', 0), detail.get('fee_rate', 0) * 100)
            else:
                log.info("😕 No profitable arbitrage opportunities found")
                
            return cycles
            
        except Exception as e:
            log.error("❌ Failed to find arbitrage opportunities: %s", e)
            return []

    # async def determine_optimal_amount(
    #     self, 
    #     cycle: Dict,
    #     max_amount: float = 10.0
    # ) -> float:
    #     """
    #     最適投入額を決定（未実装）
    #     
    #     Parameters
    #     ----------
    #     cycle : Dict
    #         アービトラージサイクル情報
    #     max_amount : float, default 10.0
    #         最大投入額（SOL）
    #         
    #     Returns
    #     -------
    #     float
    #         最適投入額
    #         
    #     Notes
    #     -----
    #     この関数は以下を考慮して最適額を計算する予定：
    #     - 価格インパクト
    #     - 流動性制約
    #     - ガス費用
    #     - リスク管理
    #     """
    #     # TODO: 実装予定
    #     # 1. 各プールの流動性と価格インパクトを分析
    #     # 2. 投入額による利益の変化をシミュレーション  
    #     # 3. リスク調整後の最適額を決定
    #     log.info("🧮 Determining optimal amount for cycle...")
    #     
    #     # プレースホルダー実装
    #     return min(max_amount, 1.0)  # 現在は1SOL固定

    async def run_arbitrage_cycle(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        update_prices: bool = True
    ) -> List[Dict]:
        """
        完全なアービトラージサイクルを実行
        
        Parameters
        ----------
        pool_size : int, default 100
            取得するプール数  
        update_prices : bool, default True
            価格更新を実行するか
            
        Returns
        -------
        List[Dict]
            検出されたアービトラージ機会
        """
        log.info("🚀 Starting complete arbitrage cycle")
        
        try:
            # 1. トークン価格取得・更新
            if update_prices:
                price_success = await self.update_token_prices(limit=10000)
                if not price_success:
                    log.warning("⚠️ Price update failed, continuing with existing prices")
            
            # 2. グラフ作成
            graph_success = await self.create_arbitrage_graph(pool_size)
            if not graph_success:
                log.error("❌ Graph creation failed, aborting cycle")
                return []
            
            # 3. 負サイクル探索
            opportunities = await self.find_arbitrage_opportunities()
            
            # 4. 最適投入額決定（未実装）
            # for opportunity in opportunities:
            #     optimal_amount = await self.determine_optimal_amount(opportunity)
            #     opportunity['optimal_amount'] = optimal_amount
            #     log.info("💰 Optimal amount for cycle: %.4f SOL", optimal_amount)
            
            log.info("✅ Arbitrage cycle completed successfully")
            return opportunities
            
        except Exception as e:
            log.error("❌ Arbitrage cycle failed: %s", e)
            return []


async def test_arbitrage_bot():
    """
    アービトラージボットのテスト実行
    """
    log.info("🧪 Starting arbitrage bot test")
    
    try:
        async with ArbitrageBot() as bot:
            # テスト実行（小規模）
            opportunities = await bot.run_arbitrage_cycle(
                pool_size=1000,        # 1000プール（テスト用）
                update_prices=False   # テスト時は価格更新スキップ
            )
            
            if opportunities:
                log.info("🎉 Test completed successfully - found %d opportunities", 
                        len(opportunities))
                
                # 最も利益率の高い機会を表示
                best_opportunity = max(opportunities, key=lambda x: x['profit_rate'])
                log.info("🏆 Best opportunity: %.2f%% profit", 
                        best_opportunity['profit_rate'] * 100)
            else:
                log.info("📝 Test completed - no opportunities found")
                
    except Exception as e:
        log.error("❌ Test failed: %s", e)


async def main():
    """
    メイン実行関数
    """
    log.info("🤖 Arbitrage Bot Starting...")
    
    # コマンドライン引数をチェック
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # テスト実行
        await test_arbitrage_bot()
    else:
        # 通常実行
        async with ArbitrageBot() as bot:
            opportunities = await bot.run_arbitrage_cycle()
            
            if opportunities:
                log.info("🎯 Found %d arbitrage opportunities", len(opportunities))
                # TODO: 実際の取引実行
                # await execute_cycle(best_cycle, optimal_amount)
            else:
                log.info("😕 No arbitrage opportunities found")
    
    log.info("🏁 Arbitrage Bot Finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("👋 Bot stopped by user")
    except Exception as e:
        log.error("💥 Bot crashed: %s", e)
        sys.exit(1)



# test_arbitrage_bot関数のワンライナーテスト:
# python -c "import asyncio; import logging; logging.basicConfig(level=logging.DEBUG); from src.__main__ import test_arbitrage_bot; asyncio.run(test_arbitrage_bot())"

# 全体テスト
# python -m src test