# src/graph/arbitrage_path_finder.py
"""
Arbitrage path detection module

-----------------------------
Purpose:
    - Build a graph from Raydium/Jupiter edge data
    - Detect negative cycles (arbitrage opportunities) with Bellman-Ford
    - Provide detailed information on detected paths
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from igraph import Graph

from src.graph.bellman_ford import bellman_ford_negative_cycles

# Constants
SOL_ADDRESS = "So11111111111111111111111111111111111111112"
MIN_CYCLE_LENGTH = 2  # Minimum cycle length
MAX_CYCLE_LENGTH = 5  # Maximum cycle length (for efficiency)

# Logger setup
log = logging.getLogger(__name__)


class ArbitragePathFinder:
    """
    A graph-based analysis class for detecting arbitrage paths
    
    Uses the Bellman-Ford algorithm to detect negative cycles and
    identify arbitrage opportunities.
    
    Attributes
    ----------
    graph : igraph.Graph
        Built graph object
    idx_to_token : List[str]
        インデックス -> トークンアドレスのマッピング
    token_to_idx : Dict[str, int]
        トークンアドレス -> インデックスのマッピング
    edge_data : Dict[Tuple[int, int], Dict]
        Storage for edge information
    """

    def __init__(self) -> None:
        """Initialize ArbitragePathFinder"""
        self.graph: Optional[Graph] = None
        self.idx_to_token: List[str] = []
        self.token_to_idx: Dict[str, int] = {}
        self.edge_data: Dict[Tuple[int, int], Dict] = {}

    def build_graph_from_raydium_edges(
        self, 
        edges: List[Tuple[str, str, float, Dict]]
    ) -> None:
        """
        Build graph from edge data obtained from the Raydium client
        
        Parameters
        ----------
        edges : List[Tuple[str, str, float, Dict]]
            Raydiumエッジデータ
            Each element: (from_token, to_token, weight, pool_data)
            
        Raises
        ------
        ValueError
            If edge data is empty or invalid
        """
        if not edges:
            raise ValueError("Edges list cannot be empty")
            
        log.info("Building graph from %d Raydium edges", len(edges))
        
        # Extract all unique token addresses
        unique_tokens = set()
        valid_edges = []
        
        for from_token, to_token, weight, pool_data in edges:
            # Basic validation
            if not from_token or not to_token:
                log.warning("Skipping edge with empty token addresses")
                continue
                
            if not isinstance(weight, (int, float)) or not np.isfinite(weight):
                log.warning("Skipping edge with invalid weight: %s", weight)
                continue
                
            if from_token == to_token:
                log.debug("Skipping self-loop edge: %s", from_token)
                continue
                
            unique_tokens.add(from_token)
            unique_tokens.add(to_token)
            valid_edges.append((from_token, to_token, weight, pool_data))
        
        if not valid_edges:
            raise ValueError("No valid edges found after filtering")
            
        # Create token index mapping
        self.idx_to_token = sorted(list(unique_tokens))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        # Build igraph graph
        self.graph = Graph(directed=True)
        self.graph.add_vertices(len(self.idx_to_token))
        
        # Set vertex attributes
        self.graph.vs["token_address"] = self.idx_to_token
        self.graph.vs["is_sol"] = [token == SOL_ADDRESS for token in self.idx_to_token]
        
        # Add edges and edge data
        self.edge_data.clear()
        edge_weights = []
        
        for from_token, to_token, weight, pool_data in valid_edges:
            from_idx = self.token_to_idx[from_token]
            to_idx = self.token_to_idx[to_token]
            
            # Add edge
            self.graph.add_edge(from_idx, to_idx)
            edge_weights.append(weight)
            
            # Store edge data
            self.edge_data[(from_idx, to_idx)] = {
                "weight": weight,
                "pool_data": pool_data,
                "from_token": from_token,
                "to_token": to_token
            }
        
        # Set weight attribute on edges
        self.graph.es["weight"] = edge_weights
        
        log.info("Graph built successfully: %d vertices, %d edges", 
                len(self.idx_to_token), len(valid_edges))

    def build_graph_from_jupiter_routes(
        self, 
        routes: List[Dict]
    ) -> None:
        """
        Build graph from Jupiter routes (for backward compatibility)
        
        Parameters
        ----------
        routes : List[Dict]
            Jupiter routeデータ
            Each element: {inToken: {address: str}, outToken: {address: str}, 
                    inAmount: int, outAmount: int, ...}
        """
        log.info("Building graph from %d Jupiter routes", len(routes))
        
        # Convert Jupiter routes to Raydium edge format
        edges = []
        for route in routes:
            try:
                in_token = route['inToken']['address']
                out_token = route['outToken']['address']
                in_amount = route.get('inAmount', 0)
                out_amount = route.get('outAmount', 0)
                
                if in_amount > 0 and out_amount > 0:
                    rate = out_amount / in_amount
                    weight = -np.log(rate)  # log weight for Bellman-Ford
                    
                    # Mock pool_data
                    pool_data = {
                        "pool_id": f"jupiter_{hash((in_token, out_token))}",
                        "rate": rate,
                        "route_data": route
                    }
                    
                    edges.append((in_token, out_token, weight, pool_data))
                    
            except (KeyError, ZeroDivisionError, ValueError) as e:
                log.warning("Skipping invalid Jupiter route: %s", e)
                continue
        
        # Process as Raydium-style edges
        self.build_graph_from_raydium_edges(edges)

    def find_negative_cycles(
        self, 
        max_cycles: int = 10,
        min_profit_threshold: float = 0.01
    ) -> List[Dict]:
        """
        Detect negative cycles (arbitrage opportunities)
        
        Parameters
        ----------
        max_cycles : int, default 10
            Maximum number of cycles to detect
        min_profit_threshold : float, default 0.01
            Minimum profit threshold (1% = 0.01)
            
        Returns
        -------
        List[Dict]
            Detailed info about detected arbitrage paths
            Each element: {
                "cycle": List[str],           # トークンアドレスのサイクル
                "profit_rate": float,         # profit rate
                "total_weight": float,        # total weight
                "path_details": List[Dict]    # details of each edge
            }
            
        Raises
        ------
        RuntimeError
            If the graph has not been built
        """
        if self.graph is None:
            raise RuntimeError("Graph not built. Call build_graph_from_*() first.")
            
        log.info("Searching for negative cycles (max: %d, min profit: %.2f%%)", 
                max_cycles, min_profit_threshold * 100)
        
        edges = [(e.source, e.target, e['weight']) for e in self.graph.es]
        cycles_idx = bellman_ford_negative_cycles(
            self.graph.vcount(),
            edges,
            max_cycles=max_cycles
        )

        
        try:
            cycles_found = []
            if len(cycles_idx) != 0:
                for idx_list in cycles_idx:
                    ci = self._analyze_cycle(idx_list, min_profit_threshold)
                    if ci:
                        cycles_found.append(ci)
                        log.info("Found profitable cycle with %.2f%% profit",
                                ci["profit_rate"] * 100)
            else:
                log.info("No negative cycles detected")

            return cycles_found
                
        except Exception as e:
            log.error("Error during negative cycle detection: %s", e)
            
        return cycles_found

    def find_cycles_from_token(
        self, 
        start_token: str,
        max_depth: int = MAX_CYCLE_LENGTH
    ) -> List[Dict]:
        """
        Search cycles starting from a specific token
        
        Parameters
        ----------
        start_token : str
            Start token address
        max_depth : int, default 5
            Maximum search depth
            
        Returns
        -------
        List[Dict]
            Details of detected cycles
        """
        if self.graph is None:
            raise RuntimeError("Graph not built. Call build_graph_from_*() first.")
            
        if start_token not in self.token_to_idx:
            log.warning("Start token not found in graph: %s", start_token)
            return []
            
        start_idx = self.token_to_idx[start_token]
        log.info("Searching cycles from token: %s (max depth: %d)", 
                start_token[:8], max_depth)
        
        cycles = []
        
        try:
            # DFS-based cycle search (simple implementation)
            visited = set()
            path = []
            
            def dfs(current_idx: int, depth: int) -> None:
                if depth > max_depth:
                    return
                    
                if current_idx in path:
                    # Cycle found
                    cycle_start = path.index(current_idx)
                    cycle_indices = path[cycle_start:] + [current_idx]
                    cycle_info = self._analyze_cycle(cycle_indices)
                    if cycle_info and cycle_info["profit_rate"] > 0:
                        cycles.append(cycle_info)
                    return
                
                path.append(current_idx)
                
                # Explore adjacent nodes
                for neighbor in self.graph.neighbors(current_idx, mode="out"):
                    if neighbor not in visited or neighbor == start_idx:
                        dfs(neighbor, depth + 1)
                
                path.pop()
            
            dfs(start_idx, 0)
            
        except Exception as e:
            log.error("Error during cycle search from token %s: %s", start_token, e)
            
        log.info("Found %d cycles from token %s", len(cycles), start_token[:8])
        return cycles

    def _analyze_cycle(
        self, 
        cycle_indices: List[int], 
        min_profit_threshold: float = 0.0
    ) -> Optional[Dict]:
        """
        Perform detailed analysis of the cycle
        
        Parameters
        ----------
        cycle_indices : List[int]
            サイクルのインデックスリスト
        min_profit_threshold : float
            Minimum profit threshold
            
        Returns
        -------
        Optional[Dict]
            Cycle analysis result (None if below threshold)
        """
        if len(cycle_indices) < MIN_CYCLE_LENGTH:
            return None
            
        total_weight = 0.0
        path_details = []
        
        # Analyze each edge in the cycle
        for i in range(len(cycle_indices) - 1):
            from_idx = cycle_indices[i]
            to_idx = cycle_indices[i + 1]
            
            edge_key = (from_idx, to_idx)
            if edge_key not in self.edge_data:
                log.warning("Edge data not found for %d -> %d", from_idx, to_idx)
                return None
                
            edge_info = self.edge_data[edge_key]
            total_weight += edge_info["weight"]
            
            path_details.append({
                "from_token": self.idx_to_token[from_idx],
                "to_token": self.idx_to_token[to_idx],
                "from_symbol": edge_info["pool_data"].get("token_a", {}).get("symbol", "Unknown"),
                "to_symbol": edge_info["pool_data"].get("token_b", {}).get("symbol", "Unknown"),
                "weight": edge_info["weight"],
                "pool_id": edge_info["pool_data"].get("pool_id"),
                "tvl": edge_info["pool_data"].get("tvl"),
                "fee_rate": edge_info["pool_data"].get("fee_rate")
            })
        
        # Calculate profit rate (negative weight = positive profit)
        profit_rate = np.exp(-total_weight) - 1.0
        
        if profit_rate < min_profit_threshold:
            log.debug("Cycle profit %.4f%% below threshold %.4f%%", 
                     profit_rate * 100, min_profit_threshold * 100)
            return None
        
        cycle_tokens = [self.idx_to_token[idx] for idx in cycle_indices]
        
        return {
            "cycle": cycle_tokens,
            "profit_rate": profit_rate,
            "total_weight": total_weight,
            "path_details": path_details,
            "cycle_length": len(cycle_indices) - 1
        }

    def get_graph_stats(self) -> Dict[str, any]:
        """
        Get graph statistics

        Returns
        -------
        Dict[str, Any]
            Graph statistics
        """
        if self.graph is None:
            return {"error": "Graph not built"}

        # Basic information
        num_vertices = self.graph.vcount()
        num_edges    = self.graph.ecount()
        sol_nodes    = sum(1 for is_sol in self.graph.vs["is_sol"] if is_sol)
        other_tokens = num_vertices - sol_nodes

        # Connectivity
        is_weakly_conn = self.graph.is_connected(mode="weak")
        comps          = self.graph.components(mode="weak")
        comp_sizes     = comps.sizes()
        num_components = len(comp_sizes)
        largest_comp   = max(comp_sizes) if comp_sizes else 0
        smallest_comp  = min(comp_sizes) if comp_sizes else 0

        # Density
        density = self.graph.density()

        # Degree distribution
        indegrees = self.graph.indegree()
        outdegrees = self.graph.outdegree()
        degrees = [i + o for i, o in zip(indegrees, outdegrees)]
        max_deg = max(degrees) if degrees else 0
        min_deg = min(degrees) if degrees else 0
        avg_deg = sum(degrees) / len(degrees) if degrees else 0
        # Median
        deg_sorted = sorted(degrees)
        mid = len(deg_sorted) // 2
        if len(deg_sorted) % 2 == 0:
            median_deg = (deg_sorted[mid - 1] + deg_sorted[mid]) / 2
        else:
            median_deg = deg_sorted[mid]

        # Clustering coefficient (global transitivity)
        clustering = self.graph.transitivity_undirected()

        return {
            "vertices": num_vertices,
            "edges": num_edges,
            "sol_nodes": sol_nodes,
            "other_tokens": other_tokens,
            "is_weakly_connected": is_weakly_conn,
            "component_count": num_components,
            "largest_component_size": largest_comp,
            "smallest_component_size": smallest_comp,
            "density": density,
            "max_degree": max_deg,
            "min_degree": min_deg,
            "avg_degree": avg_deg,
            "median_degree": median_deg,
            "max_indegree": max(indegrees) if indegrees else 0,
            "max_outdegree": max(outdegrees) if outdegrees else 0,
            "clustering_coefficient": clustering,
        }



# Test helper function
def test_arbitrage_finder():
    """Test run for ArbitragePathFinder"""
    import asyncio
    from src.clients.raydium_client import RaydiumClient
    
    async def run_test():
        # Raydiumデータでテスト
        client = RaydiumClient()
        edges = await client.get_raydium_graph(1.0, pageSize=10)
        
        finder = ArbitragePathFinder()
        finder.build_graph_from_raydium_edges(edges)
        
        print("Graph Stats:", finder.get_graph_stats())
        
        cycles = finder.find_negative_cycles(max_cycles=5)
        print(f"Found {len(cycles)} profitable cycles")
        
        for i, cycle in enumerate(cycles):
            print(f"Cycle {i+1}: {cycle['profit_rate']*100:.2f}% profit")
            
    return asyncio.run(run_test())


# Test command
# python -c "import asyncio; from src.core.arbitrage_path_finder import test_arbitrage_finder; test_arbitrage_finder()"