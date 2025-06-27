# src/__main__.py
"""
Main execution module for the arbitrage bot

-----------------------------
Features:
    1. Fetch and update token prices
    2. Build graph from Raydium pools  
    3. Search for negative cycles (arbitrage opportunities)
    4. Determine optimal volume (not implemented)
    5. Test run functionality
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Import clients and core functionality
from src.clients.jupiter_client import JupiterClient
from src.clients.raydium_client import RaydiumClient
from src.graph.arbitrage_path_finder import ArbitragePathFinder
# from executor import execute_cycle  # Commented out as not implemented yet

# Configuration
DEFAULT_POOL_SIZE = 100   # Number of pools to fetch
MIN_PROFIT_THRESHOLD = 0.001  # Minimum profit rate 0.1%
MAX_CYCLES_TO_FIND = 5    # Maximum cycles to detect
LOG_LEVEL = logging.INFO

# Logger configuration
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger("ArbitrageBot")


class ArbitrageBot:
    """
    Main class of the arbitrage bot
    
    Manages the process from fetching token prices to detecting arbitrage opportunities.
    
    """
    
    def __init__(self):
        """Initialize the bot"""
        self.jupiter_client: Optional[JupiterClient] = None
        self.raydium_client: Optional[RaydiumClient] = None
        self.path_finder: Optional[ArbitragePathFinder] = None
        
    async def __aenter__(self):
        """Enter async context manager"""
        self.jupiter_client = JupiterClient()
        self.raydium_client = RaydiumClient()
        self.path_finder = ArbitragePathFinder()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager"""
        if self.jupiter_client:
            await self.jupiter_client.aclose()

    async def update_token_prices(self, limit: int = 250) -> bool:
        """
        Fetch and update token prices from the Jupiter API
        
        Parameters
        ----------
        limit : int, default 250
            Maximum number of tokens to fetch
            
        Returns
        -------
        bool
            Whether price retrieval succeeded
        """
        log.info("🔄 Starting token price update (limit: %d)", limit)
        
        try:
            # Retrieve verified token list
            verified_mints = await self.jupiter_client.verified_mints(limit=limit)
            log.info("📋 Retrieved %d verified token mints", len(verified_mints))
            
            # Get and save SOL-denominated prices
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
        Create an arbitrage graph from Raydium pool data
        
        Parameters
        ----------
        pool_size : int, default 100
            Number of pools to fetch
            
        Returns
        -------
        bool
            Whether graph creation succeeded
        """
        log.info("🏗️ Creating arbitrage graph (pools: %d)", 
                 pool_size)
        
        try:
            # Retrieve edge data from Raydium pools
            edges = await self.raydium_client.get_raydium_graph(
                pageSize=pool_size,
                max_pages=3
            )
            
            if not edges:
                log.warning("⚠️ No edges retrieved from Raydium")
                return False
                
            # Build the graph
            self.path_finder.build_graph_from_raydium_edges(edges)
            
            # Display graph statistics
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
        Search for negative cycles (arbitrage opportunities)
        
        Parameters
        ----------
        min_profit : float, default 0.005
            Minimum profit rate (0.5%)
        max_cycles : int, default 5
            Maximum cycles to detect
            
        Returns
        -------
        List[Dict]
            List of detected arbitrage opportunities
        """
        log.info("🔍 Searching for arbitrage opportunities (min profit: %.2f%%)", 
                min_profit * 100)
        
        try:
            # Detect negative cycles
            cycles = self.path_finder.find_negative_cycles(
                max_cycles=max_cycles,
                min_profit_threshold=min_profit
            )
            
            if cycles:
                log.info("🎯 Found %d profitable arbitrage opportunities:", len(cycles))
                
                for i, cycle in enumerate(cycles, 1):
                    profit_pct = cycle['profit_rate'] * 100
                    cycle_length = cycle['cycle_length']
                    
                    # Get token symbols for the cycle
                    symbols = []
                    for detail in cycle['path_details']:
                        symbols.append(detail.get('from_symbol', 'Unknown'))
                    symbols.append(cycle['path_details'][-1].get('to_symbol', 'Unknown'))
                    
                    log.info("  %d. %s (%.2f%% profit, %d hops)", 
                            i, " → ".join(symbols), profit_pct, cycle_length)
                    
                    # Output detailed information to debug log
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
    #     Determine optimal amount (not implemented)
    #     
    #     Parameters
    #     ----------
    #     cycle : Dict
    #         Arbitrage cycle information
    #     max_amount : float, default 10.0
    #         Maximum amount (SOL)
    #         
    #     Returns
    #     -------
    #     float
    #         Optimal amount
    #         
    #     Notes
    #     -----
    #     This function will calculate the optimal amount considering:
    #     - Price impact
    #     - Liquidity constraints
    #     - Gas costs
    #     - Risk management
    #     """
    #     # TODO: Implementation planned
    #     # 1. Analyze each pool's liquidity and price impact
    #     # 2. Simulate profit change by input amount  
    #     # 3. Determine the optimal amount after risk adjustment
    #     log.info("🧮 Determining optimal amount for cycle...")
    #     
    #     # Placeholder implementation
    #     return min(max_amount, 1.0)  # currently fixed at 1 SOL

    async def run_arbitrage_cycle(
        self,
        pool_size: int = DEFAULT_POOL_SIZE,
        update_prices: bool = True
    ) -> List[Dict]:
        """
        Run the complete arbitrage cycle
        
        Parameters
        ----------
        pool_size : int, default 100
            Number of pools to fetch  
        update_prices : bool, default True
            Whether to update prices
            
        Returns
        -------
        List[Dict]
            Detected arbitrage opportunities
        """
        log.info("🚀 Starting complete arbitrage cycle")
        
        try:
            # 1. Fetch and update token prices
            if update_prices:
                price_success = await self.update_token_prices(limit=10000)
                if not price_success:
                    log.warning("⚠️ Price update failed, continuing with existing prices")
            
            # 2. Build the graph
            graph_success = await self.create_arbitrage_graph(pool_size)
            if not graph_success:
                log.error("❌ Graph creation failed, aborting cycle")
                return []
            
            # 3. Search for negative cycles
            opportunities = await self.find_arbitrage_opportunities()
            
            # 4. Determine optimal volume (not implemented)
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
    Test run of the arbitrage bot
    """
    log.info("🧪 Starting arbitrage bot test")
    
    try:
        async with ArbitrageBot() as bot:
            # Test run (small scale)
            opportunities = await bot.run_arbitrage_cycle(
                pool_size=1000,        # 1000 pools (for testing)
                update_prices=False   # Skip price update during test
            )
            
            if opportunities:
                log.info("🎉 Test completed successfully - found %d opportunities", 
                        len(opportunities))
                
                # Display the most profitable opportunity
                best_opportunity = max(opportunities, key=lambda x: x['profit_rate'])
                log.info("🏆 Best opportunity: %.2f%% profit", 
                        best_opportunity['profit_rate'] * 100)
            else:
                log.info("📝 Test completed - no opportunities found")
                
    except Exception as e:
        log.error("❌ Test failed: %s", e)


async def main():
    """
    Main execution function
    """
    log.info("🤖 Arbitrage Bot Starting...")
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test
        await test_arbitrage_bot()
    else:
        # Normal execution
        async with ArbitrageBot() as bot:
            opportunities = await bot.run_arbitrage_cycle()
            
            if opportunities:
                log.info("🎯 Found %d arbitrage opportunities", len(opportunities))
                # TODO: Execute real trades
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



# One-line test for test_arbitrage_bot function:
# python -c "import asyncio; import logging; logging.basicConfig(level=logging.DEBUG); from src.__main__ import test_arbitrage_bot; asyncio.run(test_arbitrage_bot())"

# Full test
# python -m src test