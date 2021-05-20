import json
from typing import OrderedDict
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from operator import itemgetter
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances


with open('data1.json', mode='r') as fp:
    data1 = json.load(fp)

with open('data2.json', mode='r') as fp:
    data2 = json.load(fp)

with open('data3.json', mode='r') as fp:
    data3 = json.load(fp)

with open('labels_true.json', mode='r') as fp:
    labels_true = json.load(fp)


class Centers:

    def __init__(self, dtype):
        self.dtype = dtype
        self.vocab = defaultdict(int)
        self.__C = sp.lil_matrix((0, 0), dtype=self.dtype)
    
    def centers(self):
        return self.__C
    
    def centers_normalized(self, n):
        return sp.diags(1 / n).dot(self.__C)

    def resize_horizontaly(self, article):
        for term in article.keys():
            if term not in self.vocab:
                self.vocab[term] = len(self.vocab)
                self.__C.resize(self.__C.shape[0], self.__C.shape[1] + 1)

    def resize_verticaly(self):
        self.__C.resize(self.__C.shape[0] + 1, self.__C.shape[1])

    def add_article(self, cluster_id, article_repr):
        self.__C[cluster_id] += article_repr
    
    def to_sparse(self, article):
        A = sp.lil_matrix((1, len(self.vocab)), dtype=self.dtype)
        if len(article) == 0:
            return A
        keys = list(article.keys())
        values = list(article.values())
        A[:,itemgetter(*keys)(self.vocab)] = values
        return A


class Clusters:

    def __init__(self, config):
        self.config = config

        self.C1 = Centers(np.bool)
        self.C2 = Centers(np.float)
        self.C3 = Centers(np.float)

        self.n = np.empty(0, dtype=np.uint)
        self.cluster_distr = defaultdict(list)
        self.article_distr = OrderedDict()
    
    def get_labels(self):
        return list(map(int, self.article_distr.values()))

    def jaccard_similarity(self, A, B):
        if A.shape[1] == 0 or B.shape[1] == 0:
            return 0
        return (1 - pairwise_distances(A.A, B.A, metric='jaccard'))[0]

    def cosine_similarity(self, A, B):
        if A.shape[1] == 0 or B.shape[1] == 0:
            return 0
        return (1 - pairwise_distances(A, B, metric='cosine'))[0]

    def num_clusters(self):
        return len(self.n)

    def new_article(self, aid, a):
        self.C1.resize_horizontaly(a[0])
        self.C2.resize_horizontaly(a[1])
        self.C3.resize_horizontaly(a[2])

        A1 = self.C1.to_sparse(a[0])
        A2 = self.C2.to_sparse(a[1])
        A3 = self.C3.to_sparse(a[2])

        if self.num_clusters() == 0:
            self.n.resize(self.n.shape[0] + 1)
            cid = self.num_clusters() - 1
            self.n[cid] += 1
            self.cluster_distr[cid].append(aid)
            self.article_distr[aid] = cid

            self.C1.resize_verticaly()
            self.C2.resize_verticaly()
            self.C3.resize_verticaly()

            self.C1.add_article(cid, A1[0])
            self.C2.add_article(cid, A2[0])
            self.C3.add_article(cid, A3[0])

            return

        d = (
            self.config['alpha'] * self.jaccard_similarity(A1, self.C1.centers()) +
            self.config['beta'] * self.cosine_similarity(A2, self.C2.centers_normalized(self.n)) +
            self.config['gamma'] * self.cosine_similarity(A3, self.C3.centers_normalized(self.n))
        )

        d[np.where(d < self.config['thr'])] = np.inf

        if np.all(d == np.inf):
            self.n.resize(self.n.shape[0] + 1)
            cid = self.num_clusters() - 1
            self.n[cid] += 1
            self.cluster_distr[cid].append(aid)
            self.article_distr[aid] = cid

            self.C1.resize_verticaly()
            self.C2.resize_verticaly()
            self.C3.resize_verticaly()

            self.C1.add_article(cid, A1[0])
            self.C2.add_article(cid, A2[0])
            self.C3.add_article(cid, A3[0])

        else:
            cid = np.argmin(d)
            self.n[cid] += 1
            self.cluster_distr[cid].append(aid)
            self.article_distr[aid] = cid

            self.C1.add_article(cid, A1[0])
            self.C2.add_article(cid, A2[0])
            self.C3.add_article(cid, A3[0])


data = list(zip(data1, data2, data3))

mapping = { v: k for k, v in enumerate(set(labels_true)) }
labels_true = itemgetter(*labels_true)(mapping)

s = 0.1

for gamma in np.arange(0, 1 + s, s):
    for beta in np.arange(0, 1 + s, s):
        for alpha in np.arange(0, 1 + s, s):
            if (alpha + beta + gamma) != 1:
                continue

            for thr in np.arange(0, 1 + s, s):
                config = dict(alpha=alpha, beta=beta, gamma=gamma, thr=thr)

                C = Clusters(config)

                for aid, a in enumerate(data):
                    print(f'{aid + 1} / {len(data)} c={C.num_clusters()}', end='\r')
                    C.new_article(aid, a)
                
                print(config, adjusted_rand_score(labels_true, C.get_labels()), C.num_clusters())

                with open(f'res/alpha={alpha:.4f} beta={beta:.4f} gamma={gamma:.4f} thr={thr:.4f}.json', mode='w') as fp:
                    json.dump(C.get_labels(), fp, indent=4)
