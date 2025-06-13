import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from IMmodule.Diffusion import Diffusion
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.preprocessing import normalize

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def generate_synthetic_graph(n_nodes=1000, p=0.01, directed=True):
    """生成合成图数据
    
    Args:
        n_nodes: 节点数量
        p: 边生成概率
        directed: 是否生成有向图
    """
    if directed:
        # 生成有向图
        adj_matrix = torch.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and np.random.random() < p:  # 避免自环
                    adj_matrix[i,j] = 1
    else:
        # 生成无向图
        adj_matrix = torch.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.random() < p:
                    adj_matrix[i,j] = 1
                    adj_matrix[j,i] = 1
    return adj_matrix

def evaluate_seeds(adj_matrix: sp.spmatrix, seeds: List[int], diffusion: str = 'LT') -> Tuple[float, float]:
    """
    评估种子节点的传播效果
    
    Args:
        adj_matrix: 邻接矩阵
        seeds: 种子节点列表
        diffusion: 扩散模型类型 ('LT'/'IC'/'SIS')
        
    Returns:
        (平均感染率, 标准差)
    """
    diffusion_model = Diffusion(adj_matrix)
    diffusion_model.update_seed(set(seeds))
    return diffusion_model.diffusion_evaluation(diffusion)

def select_seeds_by_degree(adj_matrix: sp.spmatrix, seed_num: int) -> List[int]:
    """基于节点度中心性选择种子节点"""
    # 对于有向图，我们同时考虑入度和出度
    in_degrees = np.array(adj_matrix.sum(axis=0)).flatten()  # 入度
    out_degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # 出度
    total_degrees = in_degrees + out_degrees  # 总度
    return np.argsort(total_degrees)[-seed_num:].tolist()

def select_seeds_by_pagerank(adj_matrix: sp.spmatrix, seed_num: int, alpha: float = 0.85) -> List[int]:
    """基于PageRank算法选择种子节点"""
    # 构建有向NetworkX图
    G = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph())
    # 计算PageRank
    pr = nx.pagerank(G, alpha=alpha)
    # 按PageRank值排序并选择top-k
    return [node for node, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:seed_num]]

def select_seeds_by_kshell(adj_matrix: sp.spmatrix, seed_num: int) -> List[int]:
    """基于k-核分解选择种子节点"""
    # 对于有向图，我们使用有向图的k-shell分解
    G = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph())
    # 计算有向图的k-shell值
    kshell = nx.core_number(G)
    # 按k-shell值排序并选择top-k
    return [node for node, _ in sorted(kshell.items(), key=lambda x: x[1], reverse=True)[:seed_num]]

def select_seeds_random(n_nodes: int, seed_num: int) -> List[int]:
    """随机选择种子节点"""
    return np.random.choice(n_nodes, size=seed_num, replace=False).tolist()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='影响力最大化评估脚本')
    parser.add_argument('--seed_ratio', type=float, default=0.1, help='种子节点比例')
    parser.add_argument('--diffusion', type=str, default='LT', choices=['LT', 'IC', 'SIS'], help='传播模型类型')
    parser.add_argument('--n_nodes', type=int, default=1000, help='图节点数量')
    parser.add_argument('--num_pairs', type=int, default=100, help='训练数据对数量')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--save_dir', type=str, default='models/deepim', help='模型保存目录')
    parser.add_argument('--edge_prob', type=float, default=0.01, help='边生成概率')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成有向合成图
    adj_matrix = generate_synthetic_graph(args.n_nodes, args.edge_prob, directed=True)
    
    # 转换为scipy稀疏矩阵
    adj_sparse = sp.csr_matrix(adj_matrix.numpy())
    
    # 初始化Diffusion模型
    diffusion_model = Diffusion(adj_sparse, 'DeepIM')
    
    # 设置实验参数
    seed_num = int(args.n_nodes * args.seed_ratio)
    
    # 训练模型
    train_config = {
        'seed_nums_rate': args.seed_ratio,
        'inverse_pairs_nums': args.num_pairs,
        'model': args.diffusion,
        'batch_size': 32,
        'epoch_nums': args.epochs,
        'save_interval': 20,
        'model_save_dir': args.save_dir,
        'hidden_dim': 512,
        'latent_dim': 128,
        'learning_rate': 1e-2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    try:
        # 训练模型
        diffusion_model.train_model(**train_config)
        
        # 使用不同方法选择种子节点
        methods = {
            'DeepIM': lambda: diffusion_model.select_seeds(adj_sparse, seed_num),
            'Degree': lambda: select_seeds_by_degree(adj_sparse, seed_num),
            'PageRank': lambda: select_seeds_by_pagerank(adj_sparse, seed_num),
            'K-Shell': lambda: select_seeds_by_kshell(adj_sparse, seed_num),
            'Random': lambda: select_seeds_random(args.n_nodes, seed_num)
        }
        
        results = {}
        
        # 评估每种方法
        print(f"\n=== 评估结果 (种子比例: {args.seed_ratio}, 传播模型: {args.diffusion}) ===")
        for method_name, select_func in methods.items():
            seeds = select_func()
            avg_influence, std = evaluate_seeds(adj_sparse, seeds, args.diffusion)
            results[method_name] = (avg_influence, std)
            
            print(f"\n{method_name}方法选择的种子节点传播效果:")
            print(f"- 平均感染率: {avg_influence:.2f}%")
            print(f"- 标准差: {std:.2f}")
        
        # 保存结果到文件
        results_dir = 'IMmodule/results'
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f'results_{args.diffusion}_{args.seed_ratio}.txt'), 'w') as f:
            f.write(f"种子比例: {args.seed_ratio}\n")
            f.write(f"传播模型: {args.diffusion}\n")
            f.write(f"图类型: 有向图\n")
            f.write(f"边生成概率: {args.edge_prob}\n\n")
            
            for method_name, (avg_influence, std) in results.items():
                f.write(f"{method_name}方法:\n")
                f.write(f"- 平均感染率: {avg_influence:.2f}%\n")
                f.write(f"- 标准差: {std:.2f}\n\n")
            
    except Exception as e:
        print(f"实验失败 (种子比例: {args.seed_ratio}, 传播模型: {args.diffusion}): {str(e)}")
        return

if __name__ == "__main__":
    main() 