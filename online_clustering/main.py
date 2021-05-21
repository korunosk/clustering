import json
import numpy as np
from operator import itemgetter
from sklearn.metrics import adjusted_rand_score

from modules.clustering import Article, Clusters


with open('data/data1.json', mode='r') as fp:
    data1 = json.load(fp)

with open('data/data2.json', mode='r') as fp:
    data2 = json.load(fp)

with open('data/data3.json', mode='r') as fp:
    data3 = json.load(fp)

with open('data/labels_true.json', mode='r') as fp:
    labels_true = json.load(fp)


class Executor:

    def __init__(self, config):
        self.config = config
        self.clusters = Clusters(config)
    
    def new_article(self, article_id, article_dict):
        self.clusters.resize_horizontaly(article_dict)

        A = Article(article_dict)

        if self.clusters.num_clusters() == 0:
            cluster_id = self.clusters.num_clusters() - 1
            self.clusters.resize_verticaly()
            self.clusters.add_article(cluster_id, article_id, A)
            return

        d = self.clusters.similarity(A)

        d[np.where(d < self.config['thr'])] = np.inf

        if np.all(d == np.inf):
            cluster_id = self.clusters.num_clusters() - 1
            self.clusters.resize_verticaly()
            self.clusters.add_article(cluster_id, article_id, A)

        else:
            cluster_id = np.argmin(d)
            self.clusters.add_article(cluster_id, article_id, A)


data = list(zip(data1, data2, data3))

mapping = { v: k for k, v in enumerate(set(labels_true)) }
labels_true = itemgetter(*labels_true)(mapping)

step = 0.1

import time

for c in np.arange(0, 1 + step, step):
    for b in np.arange(0, 1 + step, step):
        for a in np.arange(0, 1 + step, step):
            if (a + b + c) != 1:
                continue

            for thr in np.arange(0, 1 + step, step):
                st = time.time()

                config = dict(a=a, b=b, c=c, thr=thr)

                e = Executor(config)

                for article_id, article_dict in enumerate(data):
                    print(f'{article_id + 1} / {len(data)} c={e.clusters.num_clusters()}', end='\r')
                    e.new_article(article_id, article_dict)
                
                score = adjusted_rand_score(labels_true, e.clusters.get_labels())
                et = time.time() - st

                print(f'config=(a={a:.2f} b={b:.2f} c={c:.2f} thr={thr:.2f}) score={score:.4f} #clusters={e.clusters.num_clusters()} et={et:.4f}')

                with open(f'data/res/a={a:.2f} b={b:.2f} c={c:.2f} thr={thr:.2f}.json', mode='w') as fp:
                    json.dump(e.clusters.get_labels(), fp, indent=4)
