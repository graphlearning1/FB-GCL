import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv

import sys


SIGMA = 1e-10


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class Neg_Identity(nn.Module):
    def __init__(self):
        super(Neg_Identity, self).__init__()
        
    def forward(self, data):
        return -data


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_ln=False):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))
        self.use_ln = use_ln
        self.lns = nn.ModuleList()

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            for i in range(n_layers - 1):
                self.lns.append(nn.BatchNorm1d(hid_dim))
        
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            if not self.use_ln:
                x = F.relu(self.convs[i](graph, x))
            else:
                x = F.relu(self.lns[i](self.convs[i](graph, x)))

        x = self.convs[-1](graph, x)

        return x

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class FB_GCN(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, temp, use_mlp=False, lam=0.5, alpha=0.1):
        super(FB_GCN, self).__init__()
        if use_mlp:
            self.encoder_adj = MLP(in_dim, hid_dim, out_dim, use_bn=True)
        else:
            self.encoder_adj = GCN(in_dim, hid_dim, out_dim, num_layers)
            self.encoder_x = GCN(in_dim, hid_dim, out_dim, num_layers)


        self.attention = Attention(out_dim)

        self.temp = temp
        self.lam = lam
        self.alpha = alpha

    def get_embedding(self, graph,graph_x, feat):

        h_adj = self.encoder_adj(graph, feat)
        h_x = self.encoder_x(graph_x, feat)

        emb = torch.stack([h_x, h_adj], dim=1)
        h_fuse,_ = self.attention(emb)

        return h_fuse.detach()


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def contrastive_adj_loss(self, X, adj):
        f = lambda x: torch.exp(x)
        refl_sim = f(self.sim(X, X))
        between_sim = adj * refl_sim
        pos_val = between_sim.sum(1)
        neg_val = refl_sim.sum(1) - pos_val
        return -torch.log((pos_val + SIGMA) / (neg_val + SIGMA)).mean()

    def dim_lable_loss(self, Z, X, is_neg=False):
        # dim_center = torch.mm(X.T, Z) /(X.T).sum(1).unsqueeze(dim=1)
        exp = torch.tensor([1e-5]).to(X.device)
        dim_center = torch.mm(X.T, Z) / ((X.T).sum(1).unsqueeze(dim=1) + exp)
        f = lambda x: torch.exp(x)
        refl_sim = f(self.sim(Z, dim_center))
        X_hot = torch.where(X > 0, 1., 0.).to(X.device)
        pos_val = (refl_sim * X_hot).sum(1) + SIGMA
        neg_val = refl_sim.sum(1) - pos_val + SIGMA

        node_cal = (pos_val / neg_val)
        loss = -torch.log(node_cal + exp)
        return loss.mean()



    def forward(self, graph, graph_x, feat, adj_label, adj_X, adj_rec, batch_size=None):
        h_adj = self.encoder_adj(graph, feat)
        h_x = self.encoder_x(graph_x, feat)

        emb = torch.stack([h_x, h_adj], dim=1)
        h_fuse,_ = self.attention(emb)

        if batch_size==None:
            loss_a = self.contrastive_adj_loss(h_adj, adj_label)
            loss_x = self.contrastive_adj_loss(h_x, adj_X)

            loss_feat = self.dim_lable_loss(h_fuse, feat)
            loss_adj = self.contrastive_adj_loss(h_fuse, adj_rec)

        else:
            loss_a = self.batched_semi_loss(h_adj, adj_label, batch_size)
            loss_x = self.batched_semi_loss(h_x, adj_X, batch_size)

            loss_adj = self.contrastive_adj_loss(h_fuse, adj_rec)
            loss_feat = self.dim_lable_loss(h_fuse, feat)

        return self.lam*(loss_a + loss_x) + self.alpha*loss_feat + loss_adj



