from __future__ import print_function
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(solver='liblinear', multi_class='ovr')
    log.fit(train_embeds, train_labels)
    predict = (log.predict(test_embeds)).tolist()
    accuracy = accuracy_score(test_labels, predict)
    print("Test Accuracy:", accuracy)
    return accuracy


def clustering(embeds, labels):
    labels = np.array(labels)
    rep = 1
    # u, s, v = sp.linalg.svds(embeds, k=32, which='LM')
    # u = normalize(embeds.dot(v.T))
    u = embeds.cpu()
    k = len(np.unique(labels))
    ac = np.zeros(rep)
    nm = np.zeros(rep)
    f1 = np.zeros(rep)
    for i in range(rep):
        kmeans = KMeans(n_clusters=k).fit(u)
        predict_labels = kmeans.predict(u)
        #intraD[i] = square_dist(predict_labels, u)
        # intraD[i] = dist(predict_labels, feature)
        cm = clustering_metrics(labels, predict_labels)
        ac[i], nm[i], f1[i] = cm.evaluationClusterModelFromLabel()

    print(np.mean(ac))
    print(np.mean(nm))
    print(np.mean(f1))
    return np.mean(ac), np.mean(nm), np.mean(f1)