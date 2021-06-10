import json
import numpy as np
import pandas as pd
from operator import itemgetter

from modules.preprocessing import Vectorizer
from config import DATASET, MAX_NUM_ARTICLES
from loaders import make_data_path, load_data, dump_data


if __name__ == '__main__':

    with open(make_data_path('processed_articles'), mode='r') as fp:
        processed_articles = [ json.loads(line) for line in fp.readlines() ]

    X1, feature_names1 = Vectorizer.count(map(itemgetter('entities_title'), processed_articles))
    X2, feature_names2 = Vectorizer.count(map(itemgetter('keyphrases_title'), processed_articles))
    X3, feature_names3 = Vectorizer.count(map(itemgetter('keyphrases_text'), processed_articles))

    data1 = Vectorizer.make_data(X1, feature_names1)
    data2 = Vectorizer.make_data(X2, feature_names2)
    data3 = Vectorizer.make_data(X3, feature_names3)

    dump_data(data1, 'entities_title.json')
    dump_data(data2, 'keyphrases_title.json')
    dump_data(data3, 'keyphrases_text.json')

    try:
        articles = load_data('articles.json')
        
        if DATASET == 'lsir':   
            np.random.seed(42)
            articles = np.random.choice(articles, MAX_NUM_ARTICLES, replace=False).tolist()
            np.random.seed()

        labels = list(map(itemgetter('label'), articles))
        codes, uniques = pd.factorize(labels)
        codes = codes.tolist()
        dump_data(codes, 'labels_true.json')
    except Exception as e:
        print(e)
