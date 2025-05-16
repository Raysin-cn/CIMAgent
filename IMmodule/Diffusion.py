import numpy as np
import scipy.sparse as sp
import networkx as nx
import random
import statistics
from typing import Set, Dict, List, Tuple, Union, Optional
from IMmodule.DeepIM import DeepIM
import torch
from torch.utils.data import DataLoader



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
        self.model = None  # 添加模型属性

    
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

    def select_seeds_deepim(self, adj_matrix: sp.spmatrix, seed_nums: int) -> List[int]:
        """
        使用DeepIM算法选择种子节点
        Select seed nodes using DeepIM algorithm
        
        Args:
            adj_matrix: 邻接矩阵 / Adjacency matrix
            seed_nums: 需要选择的种子节点数量 / Number of seed nodes to select
            
        Returns:
            List[int]: 选中的种子节点索引列表 / List of selected seed node indices
        """
        if not hasattr(self, 'model'):
            # 如果没有训练好的模型，使用默认参数初始化
            input_dim = adj_matrix.shape[0]
            self.model = DeepIM(
                input_dim=input_dim,
                hidden_dim=1024,
                latent_dim=512,
                nheads=4,
                dropout=0.2,
                alpha=0.2,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        # 将邻接矩阵转换为PyTorch张量
        adj_tensor = torch.from_numpy(adj_matrix.toarray()).float().to(self.model.device)
        
        # 选择种子节点
        selected_seeds = self.model.select_seeds(
            adj=adj_tensor,
            seed_num=seed_nums,
            topk_ratio=0.1,
            optimization_steps=300
        )
        
        return selected_seeds

    def select_seeds(self, adj_matrix: sp.spmatrix, seed_nums: int, method: str = "DeepIM") -> List[int]:
        """
        选择种子节点的统一接口
        Unified interface for seed selection
        
        Args:
            adj_matrix: 邻接矩阵 / Adjacency matrix
            seed_nums: 需要选择的种子节点数量 / Number of seed nodes to select
            method: 选择算法 ("DeepIM"/"DegreeDiscount"/"Random") / Selection algorithm
            
        Returns:
            List[int]: 选中的种子节点索引列表 / List of selected seed node indices
        """
        if method == "DeepIM":
            return self.select_seeds_deepim(adj_matrix, seed_nums)
        elif method == "DegreeDiscount":
            return self.select_seeds_degree_discount(adj_matrix, seed_nums)
        elif method == "Random":
            return np.random.choice(adj_matrix.shape[0], size=seed_nums, replace=False).tolist()
        else:
            raise ValueError(f"Unsupported method: {method}")

    def train_deepim(self, adj_matrix: sp.spmatrix, num_samples: int = 1000):
        """
        训练DeepIM模型
        """
        # 生成训练数据
        inverse_pairs = self.generate_training_data(adj_matrix, num_samples)
        
        # 初始化模型
        model = DeepIM(
            input_dim=adj_matrix.shape[0],
            hidden_dim=1024,
            latent_dim=512,
            nheads=4,
            dropout=0.2,
            alpha=0.2
        )
        
        # 训练模型
        model.train_model(
            train_loader=DataLoader(inverse_pairs, batch_size=32),
            adj=torch.from_numpy(adj_matrix.toarray()).float(),
            epochs=300
        )
        
        return model

    def generate_training_data(self, adj_matrix: sp.spmatrix, num_samples: int):
        """
        生成训练数据
        """
        num_nodes = adj_matrix.shape[0]
        seed_num = max(1, int(num_nodes * 0.1))
        
        inverse_pairs = torch.zeros((num_samples, num_nodes, 2))
        for i in range(num_samples):
            # 随机选择种子节点
            seeds = np.random.choice(num_nodes, size=seed_num, replace=False)
            inverse_pairs[i, seeds, 0] = 1
            
            # 模拟扩散过程
            infected = self.simulate_diffusion(adj_matrix, seeds)
            inverse_pairs[i, infected, 1] = 1
        
        return inverse_pairs

    def simulate_diffusion(self, adj_matrix: sp.spmatrix, seeds: List[int], model: str = 'LT') -> List[int]:
        """
        模拟扩散过程
        Simulate diffusion process
        
        Args:
            adj_matrix: 邻接矩阵 / Adjacency matrix
            seeds: 种子节点列表 / List of seed nodes
            model: 扩散模型类型 ('LT'/'IC'/'SIS') / Type of diffusion model
            
        Returns:
            List[int]: 被感染的节点列表 / List of infected nodes
        """
        G = nx.from_scipy_sparse_array(adj_matrix).to_directed()
        
        if model == 'LT':
            return self._simulate_lt(G, set(seeds))
        elif model == 'IC':
            return self._simulate_ic(G, set(seeds))
        elif model == 'SIS':
            return self._simulate_sis(G, set(seeds))
        else:
            raise ValueError(f"Unsupported model: {model}")

    def _simulate_lt(self, G: nx.DiGraph, seeds: Set[int]) -> List[int]:
        """
        模拟线性阈值模型
        Simulate Linear Threshold model
        """
        # 随机初始化阈值
        thresholds = {node: random.uniform(0.3, 0.6) for node in G.nodes()}
        
        # 标准化边权重
        for node in G.nodes():
            in_edges = list(G.in_edges(node, data=True))
            weight_sum = sum(data.get('weight', 1.0) for _, _, data in in_edges)
            if weight_sum > 0:
                for u, v, data in in_edges:
                    data['weight'] = data.get('weight', 1.0) / weight_sum
        
        active_nodes = set(seeds)
        newly_active = set(seeds)
        
        while newly_active:
            next_active = set()
            for node in G.nodes():
                if node not in active_nodes:
                    neighbors = list(G.neighbors(node))
                    influence_sum = sum(G[u][node].get('weight', 1.0) 
                                      for u in neighbors if u in active_nodes)
                    if influence_sum >= thresholds[node]:
                        next_active.add(node)
            
            newly_active = next_active
            active_nodes.update(newly_active)
        
        return list(active_nodes)

    def _simulate_ic(self, G: nx.DiGraph, seeds: Set[int]) -> List[int]:
        """
        模拟独立级联模型
        Simulate Independent Cascade model
        """
        active_nodes = set(seeds)
        newly_active = set(seeds)
        
        while newly_active:
            next_active = set()
            for node in newly_active:
                for neighbor in G.neighbors(node):
                    if neighbor not in active_nodes:
                        # 计算激活概率
                        prob = 1.0 / G.in_degree(neighbor)
                        if random.random() <= prob:
                            next_active.add(neighbor)
            
            newly_active = next_active
            active_nodes.update(newly_active)
        
        return list(active_nodes)

    def _simulate_sis(self, G: nx.DiGraph, seeds: Set[int]) -> List[int]:
        """
        模拟SIS模型
        Simulate SIS model
        """
        active_nodes = set(seeds)
        newly_active = set(seeds)
        iterations = 0
        max_iterations = 100
        
        while newly_active and iterations < max_iterations:
            next_active = set()
            for node in G.nodes():
                if node in active_nodes:
                    # 恢复概率
                    if random.random() < 0.1:  # 10%的恢复概率
                        continue
                else:
                    # 感染概率
                    neighbors = list(G.neighbors(node))
                    infected_neighbors = sum(1 for n in neighbors if n in active_nodes)
                    if infected_neighbors > 0 and random.random() < 0.3:  # 30%的感染概率
                        next_active.add(node)
            
            newly_active = next_active
            active_nodes.update(newly_active)
            iterations += 1
        
        return list(active_nodes)
        