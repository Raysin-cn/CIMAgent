import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from typing import List
import numpy as np
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep



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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h_ = F.relu(self.FC_input(x))
        h_ = F.relu(self.FC_input2(h_))
        h_ = F.relu(self.FC_input2(h_))
        output = self.FC_output(h_)
        return output

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        #self.prelu = nn.PReLU()
        
    def forward(self, x):
        h = F.relu(self.FC_input(x))
        h = F.relu(self.FC_hidden_1(h))
        h = F.relu(self.FC_hidden_2(h))
        # x_hat = self.FC_output(h)
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        std = torch.exp(0.5*var) # standard deviation
        epsilon = torch.randn_like(var)
        return mean + std*epsilon

    def forward(self, x):
        z = self.Encoder(x)
        x_hat = self.Decoder(z)
        
        return x_hat

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        if adj.layout == torch.sparse_coo:
            edge = adj.indices()
        else:
            edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SpGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(SpGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, x, adj):
        # x shape: [batch_size, in_features]
        batch_size = x.size(0)
        num_nodes = x.size(1)
        device = x.device
        
        # 重塑输入以进行线性变换
        Wh = torch.mm(x, self.W)  # [batch_size, out_features]
        
        # 计算注意力分数
        e = self._prepare_attentional_mechanism_input(Wh)  # [batch_size, 1]
        zero_vec = -9e15*torch.ones_like(e, device=device)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # 在最后一个维度上softmax
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 应用注意力
        h_prime = torch.bmm(attention, Wh)  # [batch_size, out_features]
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh shape: [batch_size, out_features]
        device = Wh.device
        
        # 计算注意力分数
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :].to(device))  # [batch_size, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :].to(device))  # [batch_size, 1]
        
        # 广播以计算所有节点对之间的注意力
        e = Wh1 + Wh2  # [batch_size, 1]
        return F.leaky_relu(e, negative_slope=self.alpha)

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(torch.cat([att(x, adj) for att in self.attentions], dim=1))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

class DeepIM(nn.Module):
    def __init__(self, 
                 input_dim,
                 **kwargs):
        hidden_dim = kwargs.get('hidden_dim', 1024)
        latent_dim = kwargs.get('latent_dim', 512)
        nheads = kwargs.get('nheads', 4)
        dropout = kwargs.get('dropout', 0.2)
        alpha = kwargs.get('alpha', 0.2)
        device = kwargs.get('device', 'cuda')
        super(DeepIM, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # 初始化模型组件
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, latent_dim, hidden_dim, input_dim)
        self.vae = VAEModel(self.encoder, self.decoder)
        self.gat = SpGAT(nfeat=1, nhid=64, nclass=1, dropout=dropout, nheads=nheads, alpha=alpha)
        
        # 将模型移动到指定设备
        self.vae = self.vae.to(device)
        self.gat = self.gat.to(device)
        
    def forward(self, x, adj):
        # 确保输入在正确的设备上
        x = x.to(self.device)
        adj = adj.to(self.device)
        
        # VAE前向传播
        x_hat = self.vae(x)  # [batch_size, nodes_nums]
        # 修改GAT的输入处理
        batch_size = x_hat.size(0)
        y_hat_list = []
        
        # 遍历每个batch中的样本
        for i in range(batch_size):
            # 将单个样本转换为[nodes_nums, 1]形状
            x_i = x_hat[i].unsqueeze(-1)  # [nodes_nums, 1]
            # GAT前向传播
            y_i = self.gat(x_i, adj)  # [nodes_nums, 1]
            y_hat_list.append(y_i)
            
        # 合并所有batch的结果
        y_hat = torch.stack(y_hat_list, dim=0)  # [batch_size, nodes_nums, 1]
        y_hat = y_hat.squeeze(-1)  # [batch_size, nodes_nums]
        
        return x_hat, y_hat
    
    def train_model(self, 
                   train_loader,
                   adj,
                   epochs=300,
                   lr=1e-3):
        """训练模型"""
        optimizer = Adam([{'params': self.vae.parameters()}, 
                         {'params': self.gat.parameters()}],
                        lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, data_pair in enumerate(train_loader):
                x = data_pair[:, :, 0].float().to(self.device)
                y = data_pair[:, :, 1].float().to(self.device)
                
                optimizer.zero_grad()
                
                x_hat, y_hat, mu, logvar = self(x, adj)
                
                # 计算损失
                recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
                forward_loss = F.mse_loss(y_hat, y, reduction='sum')
                
                loss = recon_loss + forward_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    def select_seeds(self, 
                    adj,
                    seed_num) -> List[int]:
        """
        选择种子节点
        
        Args:
            adj: 邻接矩阵
            seed_num: 需要选择的种子节点数量
            
        Returns:
            选择的种子节点列表
        """
        self.eval()
        
        def loss_inverse(y_true, y_hat, x_hat):
            # 修改y_true的维度处理
            y_true = y_true.unsqueeze(0)  # 添加batch维度 [1, nodes_nums]
            forward_loss = F.mse_loss(y_hat, y_true)
            L0_loss = torch.sum(torch.abs(x_hat)) / x_hat.shape[1]
            return forward_loss + L0_loss, L0_loss

        # 随机初始化z_hat
        z_hat = torch.randn(1, self.latent_dim).to(self.device)
        
        # 优化过程
        z_hat = z_hat.detach()
        z_hat.requires_grad = True
        z_optimizer = Adam([z_hat], lr=1e-2)
        y_true = torch.ones(self.input_dim).to(self.device)
        
        # 固定优化步数为30
        for i in range(500):
            # 前向传播
            x_hat = self.decoder(z_hat)
            y_hat = self.gat(x_hat.squeeze(0).unsqueeze(-1), adj)
            
            # 计算损失
            loss, L0 = loss_inverse(y_true, y_hat, x_hat)
            
            # 反向传播和优化
            z_optimizer.zero_grad()
            loss.backward()
            z_optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f'Iteration: {i + 1}',
                      f'\tTotal Loss: {loss.item():.5f}')
        
        # 选择种子节点
        top_k = x_hat.topk(seed_num)
        seeds = top_k.indices[0].cpu().numpy()
        
        return seeds

