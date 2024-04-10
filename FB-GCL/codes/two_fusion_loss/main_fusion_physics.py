
import os
import  sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import argparse

from model_two_fusion_adpabl import LogReg, FB_GCN
import torch
import torch.nn as nn
import numpy as np
import dgl
import yaml
import warnings
warnings.filterwarnings('ignore')
import pickle as pkl
import os
# from visual import print_plot

from utils_data import load_dataset, set_random_seed, cal_adj_by_feat, cal_adj_by_feat_batch

parser = argparse.ArgumentParser(description='GSLDiff')

parser.add_argument('--dataname', type=str, default="physics", help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')
parser.add_argument('--wd1', type=float, default=5e-4, help='Use MLP instead of GNN')
parser.add_argument('--lr1',  type=float, default=1e-5, help='Use MLP instead of GNN')
parser.add_argument('--lam',  type=float, default=1, help='Use MLP instead of GNN')
parser.add_argument('--alpha',  type=float, default=0.5, help='Use MLP instead of GNN')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

def Evaluate(model, graph, g_x, feat, train_idx, val_idx, test_idx, max_epoch):

    with torch.no_grad():
        graph = graph.remove_self_loop().add_self_loop()
        embeds = model.get_embedding(graph, g_x, feat)

    print_plot(embeds, test_idx, labels)
    train_embs = embeds[train_idx]
    val_embs = embeds[val_idx]
    test_embs = embeds[test_idx]

    label = labels.to(args.device)

    train_labels = label[train_idx]
    val_labels = label[val_idx]
    test_labels = label[test_idx]

    train_feat = feat[train_idx]
    val_feat = feat[val_idx]
    test_feat = feat[test_idx]

    ''' Linear Evaluation '''
    logreg = LogReg(train_embs.shape[1], num_class)
    opt = torch.optim.Adam(logreg.parameters(), lr=lr2, weight_decay=wd2)

    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(max_epoch):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = torch.argmax(logits, dim=1)
        train_acc = torch.sum(preds == train_labels).float() / train_labels.shape[0]
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with torch.no_grad():
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = torch.argmax(val_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            val_acc = torch.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                eval_acc = test_acc
            elif val_acc == best_val_acc and test_acc > eval_acc:
                eval_acc = test_acc

            # print('Epoch:{}, train_acc:{:.4f}, val_acc:{:4f}, test_acc:{:4f}'.format(epoch, train_acc, val_acc, test_acc))

    print('Best val acc:{:.4f}, test acc:{:.4f}'.format(best_val_acc, eval_acc))
    return eval_acc

if __name__ == '__main__':

    acc_list1 = []
    acc_list2 = []
    acc_list3 = []
    acc_list4 = []
    for seed in range(0,20):
        set_random_seed(seed)

        dataname = args.dataname

        config_path = f'../configs/iso_two_fuse/{dataname}.yml'
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        print(args)
        print(config)

        # load hyperparameters
        hid_dim = int(config['hid_dim'])
        out_dim = hid_dim
        n_layers = int(config['n_layers'])
        temp = float(config['temp'])

        epochs = int(config['epochs'])
        lr1 = float(config['lr1'])
        wd1 = float(config['wd1'])
        lr2 = float(config['lr2'])
        wd2 = float(config['wd2'])
        lam = float(config['lam'])
        alpha = float(config['alpha'])
        max_epoch = int(config['epoch_f'])
        k = int(config['k'])

        # wd1 = args.wd1
        # lr1 = args.lr1
        # alpha = args.alpha
        # lam = args.lam
        # alpha = 0
        # lam = 0
        print('epochs_{}, lr1_{}, wd1_{}, lr2_{}, wd2_{}, lam_{}, alpha_{}'.format(epochs, lr1, wd1, lr2, wd2, lam, alpha))

        device = args.device

        graph, feat, labels, num_class, train_idx, val_idx, test_idx = load_dataset(dataname)
        in_dim = feat.shape[1]


        model = FB_GCN(in_dim, hid_dim, out_dim, n_layers, temp, args.use_mlp, lam, alpha)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)
        print(lam)

        graph = graph.to(device)
        feat = feat.to(device)

        data_path = "../data/{}_k_{}.txt".format(dataname, k)

        if not os.path.exists(data_path):
            with open(data_path, 'wb') as f:
                adj_X = cal_adj_by_feat(feat, k=k)
                edges = torch.where(adj_X > 0)
                pkl.dump(edges, f)
        else:
            with open(data_path, 'rb') as f:
                edges = pkl.load(f)

        g_x = dgl.graph(edges)
        graph = graph.remove_self_loop().add_self_loop()
        g_x = g_x.remove_self_loop().add_self_loop()

        for epoch in range(epochs + 1):
            # for i in range(5):
            batch = 20000
            sub_node = torch.randint(0, graph.num_nodes(), [batch]).to(device)
            graph_adj = graph.subgraph(sub_node)
            graph_x = g_x.subgraph(sub_node)
            feat_sub = feat[sub_node]
            graph_adj = graph_adj.remove_self_loop().add_self_loop()
            graph_x = graph_x.remove_self_loop().add_self_loop()

            adj_label = torch.zeros((graph_adj.num_nodes(), graph_adj.num_nodes())).to(device)
            src, tar = graph_adj.edges()[0], graph_adj.edges()[1]
            adj_label[src, tar] = 1
            adj_label[tar, src] = 1
            index = torch.arange(adj_label.shape[0]).to(device)
            adj_label[index, index] = 1

            adj_X = torch.zeros((graph_x.num_nodes(), graph_x.num_nodes())).to(device)
            src, tar = graph_x.edges()[0], graph_x.edges()[1]
            adj_X[src, tar] = 1
            adj_X[tar, src] = 1


            adj_rec = adj_label.clone()
            for h in range(2):
                adj_rec = torch.mm(adj_rec, adj_label)
            adj_rec = adj_rec * adj_label
            adj_rec = torch.where(adj_rec > 0, 1, 0).to(adj_rec.device)

            model.train()
            optimizer.zero_grad()

            if dataname == 'physics':
                loss = model(graph_adj, graph_x, feat_sub, adj_label, adj_X, adj_rec)
            else:
                loss = model(graph_adj, graph_x, feat_sub, adj_label, adj_X,  adj_rec)

            loss.backward()
            optimizer.step()

            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))
            test_acc = Evaluate(model, graph, g_x, feat, train_idx, val_idx, test_idx, max_epoch)
            acc_list4.append(test_acc.cpu().numpy())

    final_acc, final_acc_std = np.mean(acc_list4), np.std(acc_list4)
    print(f"#final_f1: {final_acc:.4f}±{final_acc_std:.4f}")




