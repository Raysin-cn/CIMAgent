import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

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

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, alpha):
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [SpGATLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x

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
        Wh = torch.mm(x, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(1, 0)
        return F.leaky_relu(e, negative_slope=self.alpha)

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
        
        # 初始化模型组件
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim, input_dim)
        self.vae = VAEModel(self.encoder, self.decoder)
        self.gat = SpGAT(nfeat=1, nhid=64, nclass=1, dropout=dropout, nheads=nheads, alpha=alpha)
        
        # 将模型移动到指定设备
        self.vae = self.vae.to(device)
        self.gat = self.gat.to(device)
        
    def forward(self, x, adj):
        # VAE前向传播
        x_hat, mu, logvar = self.vae(x)
        # GAT前向传播
        y_hat = self.gat(x_hat.unsqueeze(-1), adj)
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
                    optimization_steps=300):
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
                y_hat = self.gat(x_hat.unsqueeze(-1), adj)
                
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

