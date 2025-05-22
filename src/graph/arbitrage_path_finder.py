# src/graph/arbitrage_path_finder.py
"""
アービトラージパス検出モジュール

-----------------------------
目的:
    - Raydium/Jupiterのエッジデータからグラフを構築
    - ベルマン・フォードアルゴリズムで負のサイクル（アービトラージ機会）を検出
    - 検出されたパスの詳細情報を提供
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from igraph import Graph

from src.graph.bellman_ford import bellman_ford_negative_cycles

# 定数
SOL_ADDRESS = "So11111111111111111111111111111111111111112"
MIN_CYCLE_LENGTH = 2  # 最小サイクル長
MAX_CYCLE_LENGTH = 5  # 最大サイクル長（計算効率のため）

# ロガー設定
log = logging.getLogger(__name__)


class ArbitragePathFinder:
    """
    アービトラージパスを検出するためのグラフベース分析クラス
    
    ベルマン・フォードアルゴリズムを使用して負のサイクルを検出し、
    アービトラージ機会を特定します。
    
    Attributes
    ----------
    graph : igraph.Graph
        構築されたグラフオブジェクト
    idx_to_token : List[str]
        インデックス -> トークンアドレスのマッピング
    token_to_idx : Dict[str, int]
        トークンアドレス -> インデックスのマッピング
    edge_data : Dict[Tuple[int, int], Dict]
        エッジ情報のストレージ
    """

    def __init__(self) -> None:
        """ArbitragePathFinderを初期化"""
        self.graph: Optional[Graph] = None
        self.idx_to_token: List[str] = []
        self.token_to_idx: Dict[str, int] = {}
        self.edge_data: Dict[Tuple[int, int], Dict] = {}

    def build_graph_from_raydium_edges(
        self, 
        edges: List[Tuple[str, str, float, Dict]]
    ) -> None:
        """
        Raydiumクライアントから取得したエッジデータからグラフを構築
        
        Parameters
        ----------
        edges : List[Tuple[str, str, float, Dict]]
            Raydiumエッジデータ
            各要素: (from_token, to_token, weight, pool_data)
            
        Raises
        ------
        ValueError
            エッジデータが空または無効な場合
        """
        if not edges:
            raise ValueError("Edges list cannot be empty")
            
        log.info("Building graph from %d Raydium edges", len(edges))
        
        # 全ての一意なトークンアドレスを抽出
        unique_tokens = set()
        valid_edges = []
        
        for from_token, to_token, weight, pool_data in edges:
            # 基本検証
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
            
        # トークンインデックスマッピングを作成
        self.idx_to_token = sorted(list(unique_tokens))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        # igraphグラフを構築
        self.graph = Graph(directed=True)
        self.graph.add_vertices(len(self.idx_to_token))
        
        # 頂点に属性を設定
        self.graph.vs["token_address"] = self.idx_to_token
        self.graph.vs["is_sol"] = [token == SOL_ADDRESS for token in self.idx_to_token]
        
        # エッジとエッジデータを追加
        self.edge_data.clear()
        edge_weights = []
        
        for from_token, to_token, weight, pool_data in valid_edges:
            from_idx = self.token_to_idx[from_token]
            to_idx = self.token_to_idx[to_token]
            
            # エッジを追加
            self.graph.add_edge(from_idx, to_idx)
            edge_weights.append(weight)
            
            # エッジデータを保存
            self.edge_data[(from_idx, to_idx)] = {
                "weight": weight,
                "pool_data": pool_data,
                "from_token": from_token,
                "to_token": to_token
            }
        
        # エッジに重み属性を設定
        self.graph.es["weight"] = edge_weights
        
        log.info("Graph built successfully: %d vertices, %d edges", 
                len(self.idx_to_token), len(valid_edges))

    def build_graph_from_jupiter_routes(
        self, 
        routes: List[Dict]
    ) -> None:
        """
        Jupiter routesからグラフを構築（後方互換性のため）
        
        Parameters
        ----------
        routes : List[Dict]
            Jupiter routeデータ
            各要素: {inToken: {address: str}, outToken: {address: str}, 
                    inAmount: int, outAmount: int, ...}
        """
        log.info("Building graph from %d Jupiter routes", len(routes))
        
        # Jupiter routes → Raydium edge形式に変換
        edges = []
        for route in routes:
            try:
                in_token = route['inToken']['address']
                out_token = route['outToken']['address']
                in_amount = route.get('inAmount', 0)
                out_amount = route.get('outAmount', 0)
                
                if in_amount > 0 and out_amount > 0:
                    rate = out_amount / in_amount
                    weight = -np.log(rate)  # ベルマン・フォード用の対数重み
                    
                    # pool_dataの模擬データ
                    pool_data = {
                        "pool_id": f"jupiter_{hash((in_token, out_token))}",
                        "rate": rate,
                        "route_data": route
                    }
                    
                    edges.append((in_token, out_token, weight, pool_data))
                    
            except (KeyError, ZeroDivisionError, ValueError) as e:
                log.warning("Skipping invalid Jupiter route: %s", e)
                continue
        
        # Raydium形式のエッジとして処理
        self.build_graph_from_raydium_edges(edges)

    def find_negative_cycles(
        self, 
        max_cycles: int = 10,
        min_profit_threshold: float = 0.01
    ) -> List[Dict]:
        """
        負のサイクル（アービトラージ機会）を検出
        
        Parameters
        ----------
        max_cycles : int, default 10
            検出する最大サイクル数
        min_profit_threshold : float, default 0.01
            最小利益閾値（1% = 0.01）
            
        Returns
        -------
        List[Dict]
            検出されたアービトラージパスの詳細情報
            各要素: {
                "cycle": List[str],           # トークンアドレスのサイクル
                "profit_rate": float,         # 利益率
                "total_weight": float,        # 総重み
                "path_details": List[Dict]    # 各エッジの詳細
            }
            
        Raises
        ------
        RuntimeError
            グラフが構築されていない場合
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
        特定のトークンから始まるサイクルを検索
        
        Parameters
        ----------
        start_token : str
            開始トークンアドレス
        max_depth : int, default 5
            最大探索深度
            
        Returns
        -------
        List[Dict]
            検出されたサイクルの詳細情報
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
            # DFSベースのサイクル検索（簡易実装）
            visited = set()
            path = []
            
            def dfs(current_idx: int, depth: int) -> None:
                if depth > max_depth:
                    return
                    
                if current_idx in path:
                    # サイクル発見
                    cycle_start = path.index(current_idx)
                    cycle_indices = path[cycle_start:] + [current_idx]
                    cycle_info = self._analyze_cycle(cycle_indices)
                    if cycle_info and cycle_info["profit_rate"] > 0:
                        cycles.append(cycle_info)
                    return
                
                path.append(current_idx)
                
                # 隣接ノードを探索
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
        サイクルの詳細分析を実行
        
        Parameters
        ----------
        cycle_indices : List[int]
            サイクルのインデックスリスト
        min_profit_threshold : float
            最小利益閾値
            
        Returns
        -------
        Optional[Dict]
            サイクル分析結果（利益が閾値以下の場合はNone）
        """
        if len(cycle_indices) < MIN_CYCLE_LENGTH:
            return None
            
        total_weight = 0.0
        path_details = []
        
        # サイクルの各エッジを分析
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
        
        # 利益率計算（負の重み = 正の利益）
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

    def get_graph_stats(self) -> Dict:
        """
        グラフの統計情報を取得
        
        Returns
        -------
        Dict
            グラフの統計情報
        """
        if self.graph is None:
            return {"error": "Graph not built"}
            
        sol_nodes = sum(1 for is_sol in self.graph.vs["is_sol"] if is_sol)
        
        return {
            "vertices": self.graph.vcount(),
            "edges": self.graph.ecount(),
            "sol_nodes": sol_nodes,
            "other_tokens": self.graph.vcount() - sol_nodes,
            "is_connected": self.graph.is_connected(mode="weak"),
            "density": self.graph.density(),
            "max_degree": max(self.graph.degree()) if self.graph.vcount() > 0 else 0
        }


# TEST用関数
def test_arbitrage_finder():
    """ArbitragePathFinderの動作テスト"""
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


# TEST用コマンド
# python -c "import asyncio; from src.core.arbitrage_path_finder import test_arbitrage_finder; test_arbitrage_finder()"