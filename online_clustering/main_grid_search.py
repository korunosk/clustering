import numpy as np
import pandas as pd
import time

from modules.clustering import Executor
from config import load_data, dump_res


if __name__ == '__main__':

    data = list(zip(load_data('entities_title.json'), load_data('keyphrases_title.json'), load_data('keyphrases_text.json')))

    try:
        labels_true = pd.factorize(load_data('labels_true.json'))
    except:
        print('No labels')

    step = 0.1

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
                    
                    et = time.time() - st

                    fname = f'a={a:.2f} b={b:.2f} c={c:.2f} thr={thr:.2f}'
                    print(f'config=({fname}) #clusters={e.clusters.num_clusters()} et={et:.4f}')

                    dump_res(e.clusters.get_labels(), fname + '.json')
