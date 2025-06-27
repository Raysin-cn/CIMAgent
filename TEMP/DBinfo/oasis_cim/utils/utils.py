import numpy as np
import scipy.sparse as sp
import torch
from typing import Union, Tuple
from scipy.sparse import csr_matrix

def csr_to_sparse_tensor(csr: csr_matrix) -> torch.Tensor:
    """
    将CSR格式的稀疏矩阵转换为PyTorch稀疏张量
    Convert CSR format sparse matrix to PyTorch sparse tensor
    
    Args:
        csr: CSR格式的稀疏矩阵 / Sparse matrix in CSR format
        
    Returns:
        PyTorch稀疏张量 / PyTorch sparse tensor
    """
    coo = csr.tocoo()
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float)
    size = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, size)


def normalize_adj(mx: Union[csr_matrix, np.ndarray]) -> Union[csr_matrix, np.ndarray]:
    """
    对邻接矩阵进行行归一化
    Row-normalize adjacency matrix
    
    Args:
        mx: 邻接矩阵（稀疏矩阵或numpy数组） / Adjacency matrix (sparse matrix or numpy array)
        
    Returns:
        归一化后的邻接矩阵 / Normalized adjacency matrix
    """
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def sampling(inverse_pairs: torch.Tensor) -> torch.Tensor:
    """
    对反向传播对中的扩散计数进行采样，选择扩散影响力最大的前10%节点对
    Sample from inverse propagation pairs by selecting top 10% node pairs with highest diffusion counts
    
    Args:
        inverse_pairs: 反向传播节点对张量 / Tensor of inverse propagation node pairs
        
    Returns:
        前10%扩散影响力最大的节点对的索引 / Indices of top 10% node pairs with highest diffusion influence
    """
    diffusion_count = []
    for i, pair in enumerate(inverse_pairs):
        diffusion_count.append(pair[:, 1].sum())
    diffusion_count = torch.Tensor(diffusion_count)
    top_k = diffusion_count.topk(int(0.1*inverse_pairs.shape[0])).indices
    return top_k