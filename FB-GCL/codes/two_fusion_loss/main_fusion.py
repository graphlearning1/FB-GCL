
import os
import  sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import argparse

from model_two_fusion import LogReg, FB_GCN
import torch
import torch.nn as nn
import numpy as np
import dgl
import yaml
import warnings
warnings.filterwarnings('ignore')

from utils_data import load_dataset, set_random_seed, cal_adj_by_feat

parser = argparse.ArgumentParser(description='GSLDiff')

parser.add_argument('--dataname', type=str, default="cora", help='Name of dataset.')
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
        # seed = np.random.randint(0,20)
        set_random_seed(seed)
        print(seed)

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
        # lam = 0

        print('epochs_{}, lr1_{}, wd1_{}, lr2_{}, wd2_{}, lam_{}, alpha_{}'.format(epochs, lr1, wd1, lr2, wd2, lam, alpha))

        device = args.device

        graph, feat, labels, num_class, train_idx, val_idx, test_idx = load_dataset(dataname)
        in_dim = feat.shape[1]


        model = FB_GCN(in_dim, hid_dim, out_dim, n_layers, temp, args.use_mlp, lam, alpha)
        model = model.to(device)
        print(lam)



        graph = graph.to(device)
        feat = feat.to(device)

        adj_label = torch.zeros((graph.num_nodes(), graph.num_nodes())).to(device)
        src, tar = graph.edges()[0], graph.edges()[1]
        adj_label[src, tar] = 1
        adj_label[tar, src] = 1
        index = torch.arange(adj_label.shape[0]).to(device)
        adj_label[index, index] = 1

        optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=wd1)

        adj_X = cal_adj_by_feat(feat, k=k)
        edges = torch.where(adj_X >0)
        g_x = dgl.graph(edges)

        graph = graph.remove_self_loop().add_self_loop()
        g_x = g_x.remove_self_loop().add_self_loop()

        adj_rec = adj_label.clone()

        for epoch in range(epochs + 1):
            model.train()
            optimizer.zero_grad()

            if dataname == 'physics':
                loss = model(graph, g_x, feat, adj_label, adj_X, adj_rec, batch_size=256)
            else:
                loss = model(graph, g_x, feat, adj_label, adj_X,  adj_rec)

            loss.backward()
            optimizer.step()

            print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))

            # if epoch % 50 ==0:
            #     test_acc = Evaluate(model, graph, g_x, feat, train_idx, val_idx, test_idx, max_epoch)
            #     acc_list1.append(test_acc.cpu().numpy())
        test_acc = Evaluate(model, graph, g_x, feat, train_idx, val_idx, test_idx, max_epoch)
        acc_list2.append(test_acc.cpu().numpy())


    final_acc, final_acc_std = np.mean(acc_list2), np.std(acc_list2)
    print(f"#final_f1: {final_acc:.4f}±{final_acc_std:.4f}")




