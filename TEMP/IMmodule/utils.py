import numpy as np
import networkx as nx
from scipy import sparse
import torch
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

def prepare_data(graph, seed_num, num_samples, diffusion_model='LT'):
    """
    准备训练数据,生成用于训练的种子节点-扩散结果对
    
    参数:
        graph: networkx图对象,表示网络结构
        seed_num: int,每次选择的种子节点数量
        num_samples: int,生成的训练样本数量
        diffusion_model: str,使用的扩散模型,默认为'LT'
        
    返回:
        inverse_pairs: torch.Tensor,形状为(num_samples, num_nodes, 2)的张量
                      第一维表示种子节点,第二维表示扩散结果
        adj: scipy.sparse.csr_matrix,图的邻接矩阵
    """
    adj = nx.adjacency_matrix(graph)
    num_nodes = adj.shape[0]
    
    # 创建训练数据对
    inverse_pairs = torch.zeros((num_samples, num_nodes, 2), dtype=torch.float32)
    
    for i in range(num_samples):
        # 随机选择种子节点
        seeds = np.random.choice(num_nodes, size=seed_num, replace=False)
        inverse_pairs[i, seeds, 0] = 1
        
        # 模拟扩散过程
        infected = simulate_diffusion(graph, seeds, diffusion_model)
        inverse_pairs[i, infected, 1] = 1
    
    return inverse_pairs, adj

def simulate_diffusion(graph, seeds, model='LT'):
    """
    模拟网络中的信息扩散过程
    
    参数:
        graph: networkx图对象,表示网络结构
        seeds: array-like,初始种子节点列表
        model: str,使用的扩散模型,可选'LT'或'IC',默认为'LT'
        
    返回:
        infected: numpy.ndarray,被感染节点的索引数组
    """
    if model == 'LT':
        model = ep.ThresholdModel(graph)
        config = mc.Configuration()
        for n in graph.nodes():
            config.add_node_configuration("threshold", n, 0.5)
    elif model == 'IC':
        model = ep.IndependentCascadesModel(graph)
        config = mc.Configuration()
        for e in graph.edges():
            config.add_edge_configuration("threshold", e, 1/graph.degree(e[1]))
    else:
        raise ValueError('Only IC and LT models are supported')
    
    config.add_model_initial_configuration("Infected", seeds)
    model.set_initial_status(config)
    
    iterations = model.iteration_bunch(100)
    node_status = iterations[0]['status']
    
    for j in range(1, len(iterations)):
        node_status.update(iterations[j]['status'])
    
    infected = np.where(np.array(list(node_status.values())) == 1)[0]
    return infected

def normalize_adj(adj):
    """
    对邻接矩阵进行标准化处理
    
    参数:
        adj: scipy.sparse.csr_matrix,输入的邻接矩阵
        
    返回:
        normalized_adj: scipy.sparse.csr_matrix,标准化后的邻接矩阵
    """
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sparse.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_adj 
