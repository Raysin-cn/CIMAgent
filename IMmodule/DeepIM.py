import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from typing import List


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.fc4 = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h), self.fc4(h)

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = F.relu(self.FC_input(z))
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
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.Encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.Decoder(z), mu, logvar

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
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, alpha):
        super(SpGAT, self).__init__()
        self.nfeat = nfeat
        self.dropout = dropout
        self.attentions = [SpGATLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        # x shape: [batch_size, nfeat]
        device = x.device  # 获取输入张量的设备
        adj = adj.to(device)  # 确保邻接矩阵在正确的设备上
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # 在特征维度上拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x

class DeepIM(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=1024,
                 latent_dim=512,
                 nheads=4,
                 dropout=0.2,
                 alpha=0.2,
                 device='cuda'):
        super(DeepIM, self).__init__()
        self.device = device
        self.input_dim = input_dim
        
        # 初始化模型组件
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, latent_dim, hidden_dim, input_dim)
        self.vae = VAEModel(self.encoder, self.decoder)
        self.gat = SpGAT(nfeat=input_dim, nhid=64, nclass=1, dropout=dropout, nheads=nheads, alpha=alpha)
        
        # 将模型移动到指定设备
        self.vae = self.vae.to(device)
        self.gat = self.gat.to(device)
        
    def forward(self, x, adj):
        # 确保输入在正确的设备上
        x = x.to(self.device)
        adj = adj.to(self.device)
        
        # VAE前向传播
        x_hat, mu, logvar = self.vae(x)
        # 修改GAT的输入处理
        # 将x_hat转换为正确的形状 [batch_size, input_dim]
        x_hat = x_hat.view(x_hat.size(0), self.input_dim)
        # GAT前向传播
        y_hat = self.gat(x_hat, adj)
        return x_hat, y_hat, mu, logvar
    
    def train_model(self, 
                   train_loader,
                   adj,
                   epochs=300,
                   lr=1e-4):
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
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                forward_loss = F.mse_loss(y_hat, y, reduction='sum')
                
                loss = recon_loss + kl_loss + forward_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    def select_seeds(self, 
                    adj,
                    seed_num,
                    topk_ratio=0.1,
                    optimization_steps=300) -> List[int]:
        """选择种子节点"""
        self.eval()
        with torch.no_grad():
            # 生成初始潜在表示
            z_hat = torch.zeros(self.encoder.fc3.out_features).to(self.device)
            
            # 优化潜在表示
            z_hat.requires_grad = True
            z_optimizer = Adam([z_hat], lr=1e-4)
            
            for i in range(optimization_steps):
                x_hat = self.decoder(z_hat)
                # 修改GAT的输入处理
                x_hat = x_hat.view(1, -1, self.input_dim)
                y_hat = self.gat(x_hat, adj)
                
                # 计算损失
                y_true = torch.ones_like(y_hat)
                forward_loss = F.mse_loss(y_hat, y_true)
                l0_loss = torch.sum(torch.abs(x_hat)) / x_hat.shape[0]
                
                loss = forward_loss + l0_loss
                
                z_optimizer.zero_grad()
                loss.backward()
                z_optimizer.step()
            
            # 选择top-k节点作为种子
            top_k = x_hat.topk(seed_num)
            seeds = top_k.indices.cpu().numpy()
            
        return seeds

