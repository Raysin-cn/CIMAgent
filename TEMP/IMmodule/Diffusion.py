import numpy as np
import scipy.sparse as sp
import networkx as nx
import random
import statistics
import torch
from torch.utils.data import DataLoader
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import torch.nn.functional as F
import time
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
from typing import Set, Dict, List, Tuple, Union, Optional
import glob

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from IMmodule.DeepIM import DeepIM
from IMmodule.utils import normalize_adj




def diffusion_simulation(adj, seed, diffusion='LT') -> List[int]:
    infected_nodes = []
    G = nx.from_scipy_sparse_array(adj)
    if diffusion == 'LT':
        model = ep.ThresholdModel(G)
        config = mc.Configuration()
        # 为每个节点设置传播阈值
        for n in G.nodes():
            config.add_node_configuration("threshold", n, 0.5)
    elif diffusion == 'IC':
        model = ep.IndependentCascadesModel(G)
        config = mc.Configuration()
        # 为每条边设置传播概率
        for e in G.edges():
            config.add_edge_configuration("threshold", e, 1 / nx.degree(G)[e[1]])
    elif diffusion == 'SIS':
        model = ep.SISModel(G)
        config = mc.Configuration()
        # 设置模型参数
        config.add_model_parameter('beta', 0.001)
        config.add_model_parameter('lambda', 0.001)
    else:
        raise ValueError('Only IC, LT and SIS are supported.')

        # 设置模型的初始状态为感染的种子节点
    config.add_model_initial_configuration("Infected", seed)

    # 将配置应用于模型
    model.set_initial_status(config)

    # 进行模拟迭代（这里是 100 次迭代）
    iterations = model.iteration_bunch(100)

    # 提取每次迭代的节点状态
    node_status = iterations[0]['status']

    # 在每次迭代中更新节点状态
    for j in range(1, len(iterations)):
        node_status.update(iterations[j]['status'])

    # 将节点状态转换为 0（未感染）和 1（感染）
    inf_vec = np.array(list(node_status.values()))
    inf_vec[inf_vec == 2] = 1

    # 将被感染的节点加入列表
    infected_nodes.extend(np.where(inf_vec == 1)[0])

    return np.unique(infected_nodes)




class Diffusion():
    def __init__(self, adj_matrix: sp.spmatrix = None, IM_algos: str = None) -> None:
        """
        初始化扩散模型
        Initialize the diffusion model
        
        Args:
            adj_matrix: 邻接矩阵 / Adjacency matrix
            IM_algos: 影响力最大化算法类型 ('DeepIM'/'Degree'/'PageRank'/'K-Shell'/'Random')
        """
        self.adj_matrix: sp.spmatrix = adj_matrix
        self.seed: Set[int] = None
        self.model = IM_algos  # 影响力最大化算法模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def update_adj(self, adj):
        self.adj_matrix = adj

    def load_model(self, algos):
        if algos.lower() == 'deepim':
            self.model = DeepIM(self.adj_matrix.shape[0])
        else:
            self.model = 'Random'

    def load_model_params(self, model_type='deepim', model_save_dir=None):
        """
        加载模型参数
        Args:
            model_type: 模型类型，'deepim'或其他
            model_save_dir: 模型参数存储路径
        Returns:
            z_hat: 模型参数
        """
        if model_type == 'deepim':
            self.load_model('deepim')
            # 加载最新的模型参数
            model_files = glob.glob(os.path.join(model_save_dir, 'deepim', 'epoch_*.pth'))
            if not model_files:
                print("未找到保存的模型参数，将使用随机初始化")
                return None
                
            # 获取最新的模型文件
            latest_model = max(model_files, key=os.path.getctime)
            
            try:
                # 加载模型参数
                checkpoint = torch.load(latest_model, map_location=self.device)
                z_hat = checkpoint['z_hat'].to(self.device)
                print(f"已加载模型参数: {latest_model}")
                return z_hat
            except Exception as e:
                print(f"加载模型参数时出错: {str(e)}")
                return None
        else:
            print("使用随机选择模型")
            return None

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
        values: List[float] = []
        r: int = 100
        
        for i in range(r):
            # 使用diffusion_simulation函数进行扩散模拟
            infected_nodes = diffusion_simulation(self.adj_matrix, self.seed, diffusion)
            value = len(infected_nodes) * 100 / self.adj_matrix.shape[0]
            values.append(value)
            total_infect += value
            
        return total_infect/r, statistics.stdev(values)
    


    def update_seed(self, seed: Set[int]) -> None:
        self.seed = seed

    def update_adj_matrix(self, adj_matrix: sp.spmatrix) -> None:
        self.adj_matrix = adj_matrix

    def select_seeds_by_degree(self, adj_matrix: sp.spmatrix, seed_num: int) -> List[int]:
        """基于节点度中心性选择种子节点"""
        # 对于有向图，同时考虑入度和出度
        in_degrees = np.array(adj_matrix.sum(axis=0)).flatten()  # 入度
        out_degrees = np.array(adj_matrix.sum(axis=1)).flatten()  # 出度
        total_degrees = in_degrees + out_degrees  # 总度
        return np.argsort(total_degrees)[-seed_num:].tolist()

    def select_seeds_by_pagerank(self, adj_matrix: sp.spmatrix, seed_num: int, alpha: float = 0.85) -> List[int]:
        """基于PageRank算法选择种子节点"""
        # 构建有向NetworkX图
        G = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph())
        # 计算PageRank
        pr = nx.pagerank(G, alpha=alpha)
        # 按PageRank值排序并选择top-k
        return [node for node, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:seed_num]]

    def select_seeds_by_kshell(self, adj_matrix: sp.spmatrix, seed_num: int) -> List[int]:
        """基于k-核分解选择种子节点"""
        # 对于有向图，使用有向图的k-shell分解
        G = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph())
        # 计算有向图的k-shell值
        kshell = nx.core_number(G)
        # 按k-shell值排序并选择top-k
        return [node for node, _ in sorted(kshell.items(), key=lambda x: x[1], reverse=True)[:seed_num]]

    def select_seeds_random(self, seed_num: int) -> List[int]:
        """随机选择种子节点"""
        return np.random.choice(self.adj_matrix.shape[0], size=seed_num, replace=False).tolist()

    def select_seeds(self, adj_matrix: sp.spmatrix, seed_nums: int) -> List[int]:
        """
        根据指定的算法选择种子节点
        
        Args:
            adj_matrix: 邻接矩阵
            seed_nums: 需要选择的种子节点数量
            
        Returns:
            选择的种子节点列表
        """
        if not isinstance(self.model, str):
            # 原有的DeepIM方法
            adj_matrix = adj_matrix + adj_matrix.T.multiply(adj_matrix.T > adj_matrix) - adj_matrix.multiply(adj_matrix.T > adj_matrix)
            adj_matrix = normalize_adj(adj_matrix + sp.eye(adj_matrix.shape[0]))
            adj_matrix = torch.Tensor(adj_matrix.toarray()).to_sparse().to(self.device)
            return self.model.select_seeds(adj_matrix, seed_nums)
        if self.model.lower() == 'degree':
            return self.select_seeds_by_degree(adj_matrix, seed_nums)
        elif self.model.lower() == 'pagerank':
            return self.select_seeds_by_pagerank(adj_matrix, seed_nums)
        elif self.model.lower() == 'k-shell':
            return self.select_seeds_by_kshell(adj_matrix, seed_nums)
        elif self.model.lower() == 'random':
            return self.select_seeds_random(seed_nums)
        else:
            raise ValueError(f"不支持的影响力最大化算法: {self.model}")

    def generate_training_data(self, adj_matrix: sp.spmatrix, seed_nums:int, inverse_pairs_nums: int, diffusion: str = 'LT'):
        """
        生成训练数据
        """
        num_nodes = adj_matrix.shape[0]
        inverse_pairs = torch.zeros((inverse_pairs_nums, num_nodes, 2), dtype=torch.float32)
        for i in range(inverse_pairs_nums):
            # 随机选择种子节点
            seeds = np.random.choice(np.arange(0, num_nodes), size=seed_nums, replace=False)
            inverse_pairs[i, seeds, 0] = 1
            # 模拟扩散过程
            infected_nodes = diffusion_simulation(adj_matrix, seeds, diffusion)
            inverse_pairs[i, infected_nodes, 1] = 1
        
        return inverse_pairs
    def train_model(self, **kwargs) -> None:
        train_config = {'seed_nums_rate':kwargs['seed_nums_rate'],
                        'inverse_pairs_nums':kwargs['inverse_pairs_nums'],
                        'model':kwargs['model'],
                        'batch_size':kwargs['batch_size'],
                        'epoch_nums':kwargs['epoch_nums'],
                        'device':kwargs['device'],
                        'save_interval':kwargs['save_interval'],
                        'model_save_dir':kwargs['model_save_dir']}
        node_nums = self.adj_matrix.shape[0]
        seed_nums = int(node_nums*train_config['seed_nums_rate'])
        inverse_pairs = self.generate_training_data(self.adj_matrix, seed_nums, train_config['inverse_pairs_nums'], train_config['model'])

        adj = self.adj_matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        adj = torch.Tensor(adj.toarray()).to_sparse()

        train_set, test_set = torch.utils.data.random_split(inverse_pairs,
                                                        [len(inverse_pairs) - train_config['batch_size'],
                                                         train_config['batch_size']])
        
        train_loader = DataLoader(dataset=train_set, batch_size=train_config['batch_size'], shuffle=True, drop_last=False)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

        input_dim = inverse_pairs.shape[1]
        output_dim = input_dim
        device = train_config['device']
        if isinstance(self.model, str):
            self.load_model(self.model)
        if isinstance(self.model, str):
            return None
        optimizer = Adam(self.model.parameters(), lr=1e-4)
        def loss_all(x, x_hat, y, y_hat):
            # 重构损失
            recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
            # 前向传播损失
            forward_loss = F.mse_loss(y_hat, y, reduction='sum')
            
            # 总损失
            total_loss = recon_loss + forward_loss
            return total_loss, recon_loss, forward_loss
        
        for epoch in range(train_config['epoch_nums']):
            begin_time = time.time()
            total_loss = 0  # 标量
            precision_re = 0  # 标量
            recall_re = 0  # 标量
            
            for batch_idx, data_pair in enumerate(train_loader):
                x = data_pair[:, :, 0].float().to(device)  # [batch_size, num_nodes]
                y = data_pair[:, :, 1].float().to(device)  # [batch_size, num_nodes]
                optimizer.zero_grad()
                
                # 前向传播
                x_hat, y_hat = self.model(x, adj)  # x_hat: [batch_size, num_nodes], y_hat: [batch_size, num_nodes], mu: [batch_size, latent_dim], logvar: [batch_size, latent_dim]

                # 计算损失
                total, recon, forward = loss_all(x, x_hat, y, y_hat)  # total: scalar, recon: scalar, forward: scalar
                
                # 对x_hat进行阈值采样
                x_pred = x_hat.detach().cpu().numpy()  # [batch_size, num_nodes]
                x_pred[x_pred > 0.01] = 1
                x_pred[x_pred != 1] = 0
                
                # 计算精确率和召回率
                x_true = x.detach().cpu().numpy()  # [batch_size, num_nodes]
                for i in range(x_pred.shape[0]):
                    precision_re += precision_score(x_true[i], x_pred[i], zero_division=0)  # scalar
                    recall_re += recall_score(x_true[i], x_pred[i], zero_division=0)  # scalar
                
                # 反向传播和优化
                total.backward()
                optimizer.step()
                
                total_loss += total.item()  # scalar
            
            # 打印训练信息
            end_time = time.time()
            print(f"Epoch: {epoch + 1}",
                  f"\tTotal Loss: {total_loss/len(train_loader):.4f}",
                  f"\tReconstruction Precision: {precision_re/len(train_loader):.4f}",
                  f"\tReconstruction Recall: {recall_re/len(train_loader):.4f}",
                  f"\tTime: {end_time - begin_time:.4f}")
            # 保存模型参数
            os.makedirs(os.path.join(train_config['model_save_dir'], 'autoencoder'), exist_ok=True)
            if (epoch + 1) % train_config['save_interval'] == 0:
                save_path = os.path.join(train_config['model_save_dir'], f'autoencoder/epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_loss': total_loss,
                }, save_path)
                print(f"模型已保存到: {save_path}")

        def loss_inverse(y_true, y_hat, x_hat):
            y_true = y_true.transpose(0, 1)
            forward_loss = F.mse_loss(y_hat, y_true)
            L0_loss = torch.sum(torch.abs(x_hat)) / x_hat.shape[1]
            return forward_loss + L0_loss, L0_loss

        def sampling(inverse_pairs):
            # 计算每个样本的传播影响范围
            diffusion_count = []
            for i, pair in enumerate(inverse_pairs):
                # 对每个样本的第二列(传播结果)求和
                diffusion_count.append(pair[:, 1].sum())
            # 转换为张量并移动到正确的设备
            diffusion_count = torch.Tensor(diffusion_count).to(device)
            # 选择前10%的样本
            top_k = diffusion_count.topk(int(0.1 * inverse_pairs.shape[0])).indices
            return top_k
        
        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        topk_seed_idx = sampling(inverse_pairs)
        # 对top_k中的样本进行编码
        z_hat = 0
        for i in topk_seed_idx:
            # 对每个样本的第一列(种子节点)进行编码
            z_hat += self.model.encoder(inverse_pairs[i, :, 0].unsqueeze(0).to(device))
        
        # 计算平均编码
        z_hat = z_hat / len(topk_seed_idx)

        z_hat = z_hat.detach()
        z_hat.requires_grad = True
        z_optimizer = Adam([z_hat], lr=1e-4)

        for epoch in range(train_config['epoch_nums']):
            begin_time = time.time()
            total_loss = 0
            
            # 前向传播
            x_hat = self.model.decoder(z_hat)
            # 确保adj在正确的设备上
            adj = adj.to(device)
            y_hat = self.model.gat(x_hat.squeeze(0).unsqueeze(-1), adj)
            
            # 计算损失
            y_true = torch.ones(x_hat.shape).to(device)
            loss, L0 = loss_inverse(y_true, y_hat, x_hat)
            
            # 反向传播和优化
            loss.backward()
            z_optimizer.step()
            z_optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # 打印训练信息
            end_time = time.time()
            print(f'Iteration: {epoch + 1}',
                  f'\tTotal Loss: {loss.item():.5f}',
                  f'\tTime: {end_time - begin_time:.4f}')
            
            # 保存模型参数
            os.makedirs(os.path.join(train_config['model_save_dir'], 'deepim'), exist_ok=True)
            if (epoch + 1) % train_config['save_interval'] == 0:
                save_path = os.path.join(train_config['model_save_dir'], f'deepim/epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'z_hat': z_hat,
                    'optimizer_state_dict': z_optimizer.state_dict(),
                    'total_loss': total_loss,
                }, save_path)
                print(f"z_hat已保存到: {save_path}")



if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 配置训练参数
    train_config = {
        'seed_nums_rate': 0.1,
        'inverse_pairs_nums': 100,
        'model': 'LT',
        'batch_size': 32,
        'epoch_nums': 500,
        'save_interval': 20,
        'model_save_dir': 'models/deepim',
        'hidden_dim': 512,
        'latent_dim': 128,
        'learning_rate': 1e-2
    }
    
    # 创建保存模型的目录
    os.makedirs(train_config['model_save_dir'], exist_ok=True)
    
    # 读取用户关注关系数据
    import pandas as pd
    import numpy as np
    import scipy.sparse as sp
    
    # 读取CSV文件
    df = pd.read_csv("data/CIM_experiments/users_info.csv")
    
    # 获取用户ID列表
    user_ids = df['user_id'].unique()
    num_users = len(user_ids)
    
    # 创建用户ID到索引的映射
    id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    
    # 初始化邻接矩阵
    adj = sp.lil_matrix((num_users, num_users))
    
    # 填充邻接矩阵
    for _, row in df.iterrows():
        follower_idx = id_to_idx[row['user_id']]
        # 从following_list中获取关注列表
        following_ids = eval(row['following_list']) if isinstance(row['following_list'], str) else row['following_list']
        for following_id in following_ids:
            if following_id in id_to_idx:  # 确保following_id在映射中
                following_idx = id_to_idx[following_id]
                adj[follower_idx, following_idx] = 1
    
    # 转换为CSR格式以提高效率
    adj = adj.tocsr()
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Diffusion(adj, 'DeepIM')
    train_config['device'] = device
    
    # 开始训练
    model.train_model(**train_config)

