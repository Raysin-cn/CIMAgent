import numpy as np
import scipy.sparse as sp
import networkx as nx
import random
import statistics
from typing import Set, Dict, List, Tuple, Union, Optional



class DiffusionModel:
    def linear_threshold_model_v2(self, G: nx.DiGraph, initial_active: Set[int], max_iterations: int = 200) -> int:
        """
        线性阈值模型的改进版本，包含迭代次数限制
        Enhanced version of Linear Threshold Model with iteration limit
        
        Args:
            G: 有向图网络 / Directed graph network
            initial_active: 初始激活节点集合 / Set of initially active nodes
            max_iterations: 最大迭代次数 / Maximum number of iterations
            
        Returns:
            激活节点的数量 / Number of activated nodes
        """
        # Randomly initialize thresholds between 0.3 and 0.6
        thresholds: Dict[int, float] = {node: random.uniform(0.3, 0.6) for node in G.nodes()}
        
        # Normalize edge weights so that the sum of weights for incoming edges to each node is at most 1
        for node in G.nodes:
            in_edges = list(G.in_edges(node, data=True))
            weight_sum = sum(data['weight'] for _, _, data in in_edges)
            if weight_sum > 0:
                for u, v, data in in_edges:
                    data['weight'] /= weight_sum
        
        active_nodes: Set[int] = set(initial_active)
        newly_active_nodes: Set[int] = set(initial_active)
        iterations: int = 0
        
        while newly_active_nodes and iterations < max_iterations:
            next_active_nodes: Set[int] = set()
            for node in G.nodes():
                if node not in active_nodes:
                    neighbors = list(G.neighbors(node))
                    influence_sum = sum(G[u][node]['weight'] for u in neighbors if u in active_nodes)
                    if influence_sum >= thresholds[node]:
                        next_active_nodes.add(node)
            
            newly_active_nodes = next_active_nodes
            active_nodes.update(newly_active_nodes)
            iterations += 1
        
        print(f'Number of active nodes: {len(active_nodes)}')
        return len(active_nodes)

    def linear_threshold_model(self, G: nx.DiGraph, initial_active: Set[int]) -> int:
        """
        基础线性阈值模型
        Basic Linear Threshold Model
        
        Args:
            G: 有向图网络 / Directed graph network
            initial_active: 初始激活节点集合 / Set of initially active nodes
            
        Returns:
            激活节点的数量 / Number of activated nodes
        """
        # Randomly initialize thresholds between 0.3 and 0.6
        thresholds: Dict[int, float] = {node: random.uniform(0.3, 0.6) for node in G.nodes()}
        
        # Normalize edge weights so that the sum of weights for incoming edges to each node is at most 1
        for node in G.nodes:
            in_edges = list(G.in_edges(node, data=True))
            weight_sum = sum(data['weight'] for _, _, data in in_edges)
            if weight_sum > 0:
                for u, v, data in in_edges:
                    data['weight'] /= weight_sum
        
        active_nodes: Set[int] = set(initial_active)
        newly_active_nodes: Set[int] = set(initial_active)
        
        while newly_active_nodes:
            next_active_nodes: Set[int] = set()
            for node in G.nodes():
                if node not in active_nodes:
                    neighbors = list(G.neighbors(node))
                    influence_sum = sum(G[u][node]['weight'] for u in neighbors if u in active_nodes)
                    if influence_sum >= thresholds[node]:
                        next_active_nodes.add(node)
            
            newly_active_nodes = next_active_nodes
            active_nodes.update(newly_active_nodes)
        
        print(len(active_nodes))
        return len(active_nodes)

    def independent_cascade_model_v2(self, G: nx.DiGraph, initial_active: Set[int], max_iterations: int = 200) -> int:
        """
        独立级联模型的改进版本，包含迭代次数限制
        Enhanced version of Independent Cascade Model with iteration limit
        
        Args:
            G: 有向图网络 / Directed graph network
            initial_active: 初始激活节点集合 / Set of initially active nodes
            max_iterations: 最大迭代次数 / Maximum number of iterations
            
        Returns:
            激活节点的数量 / Number of activated nodes
        """
        active_nodes: Set[int] = set(initial_active)
        newly_active_nodes: Set[int] = set(initial_active)
        iterations: int = 0
        
        while newly_active_nodes and iterations < max_iterations:
            next_active_nodes: Set[int] = set()
            for node in newly_active_nodes:
                for neighbor in G.neighbors(node):
                    if neighbor not in active_nodes:
                        # Calculate activation probability
                        edge_data = G[node][neighbor]
                        probability = 1.0 / G.in_degree(neighbor)
                        
                        # Activate with probability
                        if random.random() <= probability:
                            next_active_nodes.add(neighbor)
            
            newly_active_nodes = next_active_nodes
            active_nodes.update(newly_active_nodes)
            iterations += 1
        
        print(f'Number of active nodes: {len(active_nodes)}')
        return len(active_nodes)




class Diffusion(DiffusionModel):
    def __init__(self, adj_matrix: sp.spmatrix = None, seed: Set[int] = None) -> None:
        """
        初始化扩散模型
        Initialize the diffusion model
        
        Args:
            adj_matrix: 邻接矩阵 / Adjacency matrix
            seed: 种子节点集合 / Set of seed nodes
        """
        self.adj_matrix: sp.spmatrix = adj_matrix
        self.seed: Set[int] = seed

    
    def diffusion_evaluation(self, diffusion: str = 'LT') -> Tuple[float, float]:
        """
        评估不同扩散模型的效果
        Evaluate the performance of different diffusion models
        
        Args:
            diffusion: 扩散模型类型('LT'/'IC'/'SIS') / Type of diffusion model('LT'/'IC'/'SIS')
            
        Returns:
            (平均感染率, 标准差) / (Average infection rate, Standard deviation)
        """
        total_infect: float = 0
        G: nx.DiGraph = nx.from_scipy_sparse_matrix(self.adj_matrix).to_directed()
        values: List[float] = []
        r: int = 100
        for i in range(r):
            if diffusion == 'LT':
                count = self.linear_threshold_model(G, self.seed)
                value = count * 100/G.number_of_nodes()
                values.append(value)
                total_infect += value
            elif diffusion == 'IC':
                count = self.independent_cascade_model_v2(G, self.seed)
                value = count * 100/G.number_of_nodes()
                values.append(value)
                total_infect += value 
            elif diffusion == 'SIS':
                count = self.sis_model(G, self.seed)
                value = count * 100/G.number_of_nodes()
                values.append(value)
                total_infect += value
        return total_infect/r, statistics.stdev(values)
    
    def update_seed(self, seed: Set[int]) -> None:
        """
        更新种子节点集合
        Update the set of seed nodes
        
        Args:
            seed: 新的种子节点集合 / New set of seed nodes
        """
        self.seed = seed

    def update_adj_matrix(self, adj_matrix: sp.spmatrix) -> None:
        """
        更新邻接矩阵
        Update the adjacency matrix
        
        Args:
            adj_matrix: 新的邻接矩阵 / New adjacency matrix
        """
        self.adj_matrix = adj_matrix

