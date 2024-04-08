import torch
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
from dgl import DGLGraph
import torch.nn.functional as F

import scipy.sparse as sp
import numpy as np
import networkx as nx
import sys
import json
import os
from networkx.readwrite import json_graph
import random

from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler
from dgl.data import load_data
from collections import namedtuple, Counter
from sklearn.model_selection import train_test_split


def to_bidirected(graph):
    num_nodes = graph.num_nodes()
    
    graph = graph.remove_self_loop()
    src, dst = graph.edges()
    
    new_src = torch.cat([src, dst])
    new_dst = torch.cat([dst, src])
    
    new_graph = dgl.graph((new_src, new_dst), num_nodes = num_nodes)
    
    return new_graph


def load_dataset(name, split_idx = 0):

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = load_planetoid_dataset(name)
    elif name in ['computer', 'photo']:
        dataset = load_amazon_dataset(name)
    elif name in ['cs', 'physics']:
        dataset = load_coauthor_dataset(name)
    elif name in ['arxiv', 'products']:
        dataset = load_ogb_dataset(name)
    elif name in ['ppi', 'flickr', 'reddit', 'yelp', 'amazon']:
        dataset = load_graphsaint_dataset(name)
    elif name in ['chameleon', 'squirrel', 'film']:
        dataset = load_hete_dataset(name, split_idx)
    elif name in ['wikics']:
        dataset = loadwikics()
    return dataset

def load_dataset_split(name, train_ratio, test_ratio, val_ratio=0, split_idx = 0):

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = load_planetoid_dataset_random(name, train_ratio)
    elif name in ['computer', 'photo']:
        dataset = load_amazon_dataset(name, train_ratio, val_ratio, test_ratio)
    elif name in ['cs', 'physics']:
        dataset = load_coauthor_dataset(name, train_ratio, val_ratio, test_ratio)
    elif name in ['arxiv', 'products']:
        dataset = load_ogb_dataset(name)
    elif name in ['ppi', 'flickr', 'reddit', 'yelp', 'amazon']:
        dataset = load_graphsaint_dataset(name)
    elif name in ['chameleon', 'squirrel', 'film']:
        dataset = load_hete_dataset(name, split_idx)
    elif name in ['wikics']:
        dataset = loadwikics()
    return dataset


def loadwikics(train_ratio = 0.1, val_ratio = 0.1, test_ratio = 0.8):
    path = r'E:\code\AFGRL-master\data\pyg\WikiCS\processed\byg.data.pt'
    dataset = torch.load(path)[0]
    x = dataset.x
    labels = dataset.y
    edge_index = dataset.edge_index
    edge_weight = dataset.edge_attr

    graph = dgl.graph((edge_index[0], edge_index[1]))

    graph.ndata['feat'] = x
    graph.ndata['label'] = labels

    N = graph.number_of_nodes()
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    num_class = labels.max().item() + 1
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    return (graph, feat, label, num_class, train_idx, val_idx, test_idx)



def load_planetoid_dataset(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
        
    graph = dataset[0]
    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    return (graph, feat, label, num_class, train_idx, val_idx, test_idx)


def load_planetoid_dataset_random(name, train_ratio, val_ratio=0.):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()

    graph = dataset[0]

    N = graph.number_of_nodes()
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    return (graph, feat, label, num_class, train_idx, val_idx, test_idx)

def load_amazon_dataset(name, train_ratio = 0.1, val_ratio = 0.1, test_ratio = 0.8):
    if name == 'computer':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
        
    graph = dataset[0]
 
    N = graph.number_of_nodes()
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    return (graph, feat, label, num_class, train_idx, val_idx, test_idx)

def load_coauthor_dataset(name, train_ratio = 0.1, val_ratio = 0.1, test_ratio = 0.8):
    if name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()
        
    graph = dataset[0]

    N = graph.number_of_nodes()
    train_num = int(N * train_ratio)
    val_num = int(N * (train_ratio + val_ratio))

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:train_num]
    val_idx = idx[train_num:val_num]
    test_idx = idx[val_num:]

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')

    return (graph, feat, label, num_class, train_idx, val_idx, test_idx)



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True




def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]

    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def cal_adj_by_feat(x, k=10):
    sim_mat = sim(x, x)
    neighb_node = torch.topk(sim_mat, k=k, dim=1)[1]
    adj_x = torch.zeros_like(sim_mat)
    adj_x = adj_x.scatter_(1, neighb_node, 1)
    return adj_x

def cal_adj_by_feat_batch(x, k=10):
    sim_mat = sim(x, x)
    neighb_node = torch.topk(sim_mat, k=k, dim=1)[1]
    col = neighb_node.flatten()
    index = torch.arange(x.shape[0])
    raw = index.repeat((k, 1)).T.flatten().to(col.device)
    i = torch.stack((raw, col), dim=0)
    v = torch.ones(raw.shape[0]).to(x.device)
    adj_x = torch.sparse.FloatTensor(i, v, torch.Size([x.shape[0], x.shape[0]]))
    return adj_x

    
    