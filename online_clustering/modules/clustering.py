import numpy as np
import scipy.sparse as sp
from collections import defaultdict, OrderedDict

from modules.similarity_metrics import cosine_similarity, jaccard_similarity


class ArticleRepr:

    def __make_data(self):
        if self.__binarized:
            return [1] * len(self)
        return self.__weights

    def __make_row(self, vocab):
        return [0] * len(self)

    def __make_col(self, vocab):
        return [ vocab[term] for term in self.__terms ]
    
    def __init__(self, article_dict, binarized):
        self.__terms = list(article_dict.keys())
        self.__weights = list(article_dict.values())
        self.__binarized = binarized
        assert(len(self.__terms) == len(self.__weights))

    def __len__(self):
        return len(self.__terms)
    
    def to_sparse(self, vocab):
        shape = (1, len(vocab))
        if len(self) == 0:
            return sp.csr_matrix(shape, dtype=np.float)
        data = self.__make_data()
        row  = self.__make_row(vocab)
        col  = self.__make_col(vocab)
        return sp.csr_matrix((data, (row, col)), shape=shape, dtype=np.float)


class Article:

    def __init__(self, article_dict):
        self.__article_repr_1 = ArticleRepr(article_dict[0], True)
        self.__article_repr_2 = ArticleRepr(article_dict[1], False)
        self.__article_repr_3 = ArticleRepr(article_dict[2], False)
    
    def get_article_repr(self, vocab):
        return (
            self.__article_repr_1.to_sparse(vocab[0]),
            self.__article_repr_2.to_sparse(vocab[1]),
            self.__article_repr_3.to_sparse(vocab[2])
        )


class ClusterCenters:

    def __resize_vocab(self, article_dict):
        for term in article_dict.keys():
            if term not in self.__vocab:
                self.__vocab[term] = len(self.__vocab)

    def __init__(self, binarized):
        self.__vocab = dict()
        self.__C = sp.csr_matrix((0, 0), dtype=np.float)
        self.__binarized = binarized

    def get_vocab(self):
        return self.__vocab
    
    def get_centers(self, n):
        if self.__binarized:
            C = self.__C.copy()
            C.data[:] = 1
            return C
        return sp.diags(1 / n).dot(self.__C)
    
    def get_shape(self):
        return self.__C.shape

    def resize_horizontaly(self, article_dict):
        self.__resize_vocab(article_dict)
        self.__C.resize(self.__C.shape[0], len(self.__vocab))

    def resize_verticaly(self):
        self.__C.resize(self.__C.shape[0] + 1, self.__C.shape[1])

    def add_article(self, cluster_id, article_repr):
        self.__C[cluster_id] += article_repr[0]


class Clusters:

    def __init__(self, config):
        self.__config = config
        self.__cluster_centers_1 = ClusterCenters(True)
        self.__cluster_centers_2 = ClusterCenters(False)
        self.__cluster_centers_3 = ClusterCenters(False)
        self.__n = np.empty(0, dtype=int)
        self.__cluster_distr = defaultdict(list)
        self.__article_distr = OrderedDict()

    def get_vocab(self):
        return (
            self.__cluster_centers_1.get_vocab(),
            self.__cluster_centers_2.get_vocab(),
            self.__cluster_centers_3.get_vocab()
        )
    
    def get_centers(self):
        return (
            self.__cluster_centers_1.get_centers(self.__n),
            self.__cluster_centers_2.get_centers(self.__n),
            self.__cluster_centers_3.get_centers(self.__n)
        )
    
    def get_shape(self):
        return (
            self.__cluster_centers_1.get_shape(),
            self.__cluster_centers_2.get_shape(),
            self.__cluster_centers_3.get_shape()
        )
    
    def get_labels(self):
        return list(map(int, self.__article_distr.values()))

    def get_n(self):
        return self.__n

    def get_cluster_distr(self):
        return self.__cluster_distr
    
    def get_parameters(self):
        map_int = lambda x: list(map(int, x))
        map_float = lambda x: list(map(float, x))

        vocab = self.__cluster_centers_3.get_vocab()
        centers = self.__cluster_centers_3.get_centers(self.__n)
        return {
            'n': map_int(self.__n),
            'cluster_distr': list(map(lambda x: (int(x[0]), map_int(x[1])), self.__cluster_distr.items())),
            'article_distr': list(map(lambda x: (int(x[0]), int(x[1])), self.__article_distr.items())),
            'cluster_centers_3': {
                'vocab': list(map(lambda x: (x[0], int(x[1])), vocab.items())),
                'data': map_float(centers.data),
                'indices': map_int(centers.indices),
                'indptr': map_int(centers.indptr),
            }
        }

    def num_clusters(self):
        return len(self.__n)
    
    def resize_horizontaly(self, article_dict):
        self.__cluster_centers_1.resize_horizontaly(article_dict[0])
        self.__cluster_centers_2.resize_horizontaly(article_dict[1])
        self.__cluster_centers_3.resize_horizontaly(article_dict[2])

    def resize_verticaly(self):
        self.__n.resize(self.__n.shape[0] + 1)
        self.__cluster_centers_1.resize_verticaly()
        self.__cluster_centers_2.resize_verticaly()
        self.__cluster_centers_3.resize_verticaly()

    def add_article(self, cluster_id, article_id, article):
        self.__n[cluster_id] += 1
        self.__cluster_distr[cluster_id].append(article_id)
        self.__article_distr[article_id] = cluster_id
        article_repr = article.get_article_repr(self.get_vocab())
        self.__cluster_centers_1.add_article(cluster_id, article_repr[0])
        self.__cluster_centers_2.add_article(cluster_id, article_repr[1])
        self.__cluster_centers_3.add_article(cluster_id, article_repr[2])
    
    def similarity(self, article):
        article_repr = article.get_article_repr(self.get_vocab())
        centers = self.get_centers()
        return (
            self.__config['a'] * jaccard_similarity(article_repr[0], centers[0]) +
            self.__config['b'] * cosine_similarity(article_repr[1], centers[1]) +
            self.__config['c'] * cosine_similarity(article_repr[2], centers[2])
        )


class Executor:

    def __init__(self, config):
        self.config = config
        self.clusters = Clusters(config)
    
    def new_article(self, article_id, article_dict):
        self.clusters.resize_horizontaly(article_dict)

        A = Article(article_dict)

        if self.clusters.num_clusters() == 0:
            self.clusters.resize_verticaly()
            cluster_id = self.clusters.num_clusters() - 1
            self.clusters.add_article(cluster_id, article_id, A)
            return

        d = self.clusters.similarity(A)

        d[np.where(d < self.config['thr'])] = -np.inf

        if np.all(d == -np.inf):
            self.clusters.resize_verticaly()
            cluster_id = self.clusters.num_clusters() - 1
            self.clusters.add_article(cluster_id, article_id, A)
        else:
            cluster_id = np.argmax(d)
            self.clusters.add_article(cluster_id, article_id, A)
