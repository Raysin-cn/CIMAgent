"""
影响力最大化关键节点选择模块

提供Greedy和Random两种算法来选择影响力最大化的种子节点
"""

import pandas as pd
import json
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import random

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from oasis import (
    ActionType, 
    ManualAction, 
    LLMAction,
    generate_twitter_agent_graph
)
from oasis.environment.env import OasisEnv

from cim.config import Config


class InfluenceMaximization:
    """
    影响力最大化算法实现
    
    提供两种算法：
    1. Greedy: 贪心算法，逐步选择能最大化影响力的节点
    2. Random: 随机选择节点作为种子
    """
    
    def __init__(self, config: Config):
        """
        初始化影响力最大化算法
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化数据结构
        self.follow_matrix = None
        self.user_info = None
        self.n_nodes = 0
        
    def load_network(self, follow_matrix: np.ndarray = None):
        """
        加载网络数据
        
        Args:
            follow_matrix: 邻接矩阵，如果为None则从配置文件加载
        """
        if follow_matrix is not None:
            self.follow_matrix = follow_matrix
            self.n_nodes = follow_matrix.shape[0]
        else:
            self.follow_matrix = self._load_follow_matrix()
            self.n_nodes = self.follow_matrix.shape[0]
        
        self.logger.info(f"网络加载完成，节点数: {self.n_nodes}")
    
    def _load_follow_matrix(self) -> np.ndarray:
        """从配置文件加载关注矩阵"""
        user_info_csv = self.config.post_generation.users_file
        user_info = pd.read_csv(user_info_csv)
        self.user_info = user_info.to_dict(orient="records")
        
        n_users = len(self.user_info)
        follow_matrix = np.zeros((n_users, n_users))
        
        import ast
        for user in user_info:
            # 将字符串形式的列表安全地转为列表对象
            following_agentid_list = ast.literal_eval(user['following_agentid_list'])
            for following_agentid in following_agentid_list:
                follow_matrix[int(user['Unnamed: 0']), int(following_agentid)] = 1
        
        return follow_matrix
    
    def _simulate_influence_spread(self, seed_set: List[int], model: str = "IC", 
                                 p: float = 0.1, num_simulations: int = 1000) -> float:
        """
        模拟影响力传播
        
        Args:
            seed_set: 种子节点集合
            model: 传播模型 ("IC" 或 "LT")
            p: 传播概率
            num_simulations: 模拟次数
            
        Returns:
            平均影响力传播范围
        """
        total_influence = 0
        
        for _ in range(num_simulations):
            # 初始化激活状态
            activated = set(seed_set)
            newly_activated = set(seed_set)
            
            while newly_activated:
                next_activated = set()
                
                for node in newly_activated:
                    # 获取邻居节点
                    neighbors = np.where(self.follow_matrix[node] > 0)[0]
                    
                    for neighbor in neighbors:
                        if neighbor not in activated:
                            if model == "IC":
                                # Independent Cascade模型
                                if np.random.random() < p:
                                    next_activated.add(neighbor)
                            elif model == "LT":
                                # Linear Threshold模型
                                active_neighbors = sum(1 for n in neighbors if n in activated)
                                threshold = np.random.random()
                                if active_neighbors / len(neighbors) >= threshold:
                                    next_activated.add(neighbor)
                
                activated.update(next_activated)
                newly_activated = next_activated
            
            total_influence += len(activated)
        
        return total_influence / num_simulations
    
    def greedy_algorithm(self, k: int, model: str = "IC", p: float = 0.1, 
                        num_simulations: int = 100) -> List[int]:
        """
        贪心算法选择影响力最大化节点
        
        优化策略：
        1. 使用更少的模拟次数来加速计算
        2. 缓存中间结果避免重复计算
        3. 提前终止条件
        
        Args:
            k: 种子节点数量
            model: 传播模型
            p: 传播概率
            num_simulations: 每次评估的模拟次数
            
        Returns:
            种子节点列表
        """
        self.logger.info(f"开始贪心算法，目标节点数: {k}")
        
        seed_set = []
        available_nodes = set(range(self.n_nodes))
        
        for i in range(k):
            if not available_nodes:
                break
                
            best_node = -1
            best_influence = -1
            
            # 为了加速，只评估部分节点
            if len(available_nodes) > 50:
                # 随机采样50个节点进行评估
                sample_nodes = random.sample(list(available_nodes), min(50, len(available_nodes)))
            else:
                sample_nodes = list(available_nodes)
            
            for node in sample_nodes:
                # 计算添加当前节点后的影响力
                temp_seed_set = seed_set + [node]
                influence = self._simulate_influence_spread(temp_seed_set, model, p, num_simulations)
                
                if influence > best_influence:
                    best_influence = influence
                    best_node = node
            
            if best_node != -1:
                seed_set.append(best_node)
                available_nodes.remove(best_node)
                self.logger.info(f"选择节点 {best_node}，当前影响力: {best_influence:.2f}")
            else:
                break
        
        self.logger.info(f"贪心算法完成，种子节点: {seed_set}")
        return seed_set
    
    def random_algorithm(self, k: int) -> List[int]:
        """
        随机算法选择种子节点
        
        Args:
            k: 种子节点数量
            
        Returns:
            种子节点列表
        """
        self.logger.info(f"开始随机算法，目标节点数: {k}")
        
        # 随机选择k个节点
        seed_set = random.sample(range(self.n_nodes), min(k, self.n_nodes))
        
        self.logger.info(f"随机算法完成，种子节点: {seed_set}")
        return seed_set
    
    def evaluate_seed_set(self, seed_set: List[int], model: str = "IC", 
                         p: float = 0.1, num_simulations: int = 1000) -> Dict[str, float]:
        """
        评估种子集合的影响力
        
        Args:
            seed_set: 种子节点集合
            model: 传播模型
            p: 传播概率
            num_simulations: 模拟次数
            
        Returns:
            评估结果字典
        """
        influence = self._simulate_influence_spread(seed_set, model, p, num_simulations)
        
        return {
            "influence": influence,
            "influence_ratio": influence / self.n_nodes,
            "seed_size": len(seed_set)
        }
    
    def compare_algorithms(self, k: int, algorithms: List[str] = None) -> Dict[str, Dict]:
        """
        比较不同算法的性能
        
        Args:
            k: 种子节点数量
            algorithms: 算法列表，可选 ["Greedy", "Random"]
            
        Returns:
            比较结果
        """
        if algorithms is None:
            algorithms = ["Greedy", "Random"]
        
        results = {}
        
        for algo in algorithms:
            if algo == "Greedy":
                seed_set = self.greedy_algorithm(k)
            elif algo == "Random":
                seed_set = self.random_algorithm(k)
            else:
                self.logger.warning(f"未知算法: {algo}")
                continue
            
            evaluation = self.evaluate_seed_set(seed_set)
            results[algo] = {
                "seed_set": seed_set,
                "evaluation": evaluation
            }
        
        return results


def follow_matrix_get(config: Config) -> np.ndarray:
    """
    获取关注矩阵
    
    Args:
        config: 配置对象
        
    Returns:
        关注矩阵
    """
    user_info_csv = config.post_generation.users_file
    user_info = pd.read_csv(user_info_csv)
    user_info = user_info.to_dict(orient="records")
    follow_matrix = np.zeros((len(user_info), len(user_info)))
    
    import ast
    for user in user_info:
        # 将字符串形式的列表安全地转为列表对象
        following_agentid_list = ast.literal_eval(user['following_agentid_list'])
        for following_agentid in following_agentid_list:
            follow_matrix[int(user['Unnamed: 0']), int(following_agentid)] = 1
    
    return follow_matrix


def get_influence_maximization_nodes(config: Config, k: int = 5, algorithm: str = "Greedy", 
                                   model: str = "IC", p: float = 0.1, num_simulations: int = 1000):
    """
    获取影响力最大化节点
    
    Args:
        config: 配置对象
        k: 种子节点数量
        algorithm: 算法类型 ("Greedy" 或 "Random")
        model: 传播模型 ("IC" 或 "LT")
        p: 传播概率
        num_simulations: 评估模拟次数
        
    Returns:
        包含种子节点和评估结果的字典
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始{algorithm}算法影响力最大化...")
    logger.info(f"算法参数: k={k}, model={model}, p={p}")
    
    # 初始化算法
    im_algorithm = InfluenceMaximization(config)
    
    # 加载网络数据
    follow_matrix = follow_matrix_get(config)
    im_algorithm.load_network(follow_matrix)
    
    # 根据算法类型选择种子节点
    if algorithm == "Greedy":
        influence_nodes = im_algorithm.greedy_algorithm(k, model, p, num_simulations=100)
    elif algorithm == "Random":
        influence_nodes = im_algorithm.random_algorithm(k)
    else:
        raise ValueError(f"不支持的算法类型: {algorithm}")
    
    # 评估种子集合的影响力
    evaluation = im_algorithm.evaluate_seed_set(
        seed_set=influence_nodes,
        model=model,
        p=p,
        num_simulations=num_simulations
    )
    
    logger.info(f"{algorithm}算法完成，种子节点: {influence_nodes}")
    logger.info(f"影响力评估: {evaluation['influence']:.2f} 节点 ({evaluation['influence_ratio']:.2%})")
    
    return {
        "seed_nodes": influence_nodes,
        "evaluation": evaluation,
        "algorithm_params": {
            "algorithm": algorithm,
            "k": k,
            "model": model,
            "p": p,
            "num_simulations": num_simulations
        }
    }


def compare_influence_algorithms(config: Config, k: int = 5, algorithms: List[str] = None,
                               model: str = "IC", p: float = 0.1, num_simulations: int = 1000):
    """
    比较不同影响力最大化算法的性能
    
    Args:
        config: 配置对象
        k: 种子节点数量
        algorithms: 算法列表，可选 ["Greedy", "Random"]
        model: 传播模型
        p: 传播概率
        num_simulations: 评估模拟次数
        
    Returns:
        比较结果字典
    """
    logger = logging.getLogger(__name__)
    
    if algorithms is None:
        algorithms = ["Greedy", "Random"]
    
    logger.info(f"开始算法比较，目标节点数: {k}")
    
    # 初始化算法
    im_algorithm = InfluenceMaximization(config)
    
    # 加载网络数据
    follow_matrix = follow_matrix_get(config)
    im_algorithm.load_network(follow_matrix)
    
    # 比较算法
    results = im_algorithm.compare_algorithms(k, algorithms)
    
    logger.info("算法比较完成")
    for algo, result in results.items():
        eval_result = result['evaluation']
        logger.info(f"{algo}: 影响力={eval_result['influence']:.2f}, 比例={eval_result['influence_ratio']:.2%}")
    
    return results


if __name__ == "__main__":
    # 创建配置
    config = Config()
    
    # 测试影响力最大化算法
    print("开始影响力最大化算法测试...")
    
    # 比较算法性能
    results = compare_influence_algorithms(config, k=5, algorithms=["Greedy", "Random"])
    
    print("\n算法比较结果:")
    for algo, result in results.items():
        eval_result = result['evaluation']
        print(f"{algo}: 影响力={eval_result['influence']:.2f}, 比例={eval_result['influence_ratio']:.2%}")
    
    print("\n测试完成！")
