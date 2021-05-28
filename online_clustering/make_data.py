import json
import os
from tqdm import tqdm
from operator import itemgetter

from modules.preprocessing import Preprocessor, Vectorizer


DATA_PATH = 'data'
DATASETS  = [ 'elasticsearch', 'news-please-repo' ]


with open(os.path.join(DATA_PATH, DATASETS[0], 'articles.json'), mode='r') as fp:
    articles = json.load(fp)

articles = articles[:10000]

processed_articles = []

for article in tqdm(articles):
    processed_articles.append({
        'entities_title': Preprocessor.get_entities_spacy(article['title']),
        'keyphrases_title': Preprocessor.get_keyphrases_pke(article['title']),
        'keyphrases_text': Preprocessor.get_keyphrases_pke(article['text']),
    })

with open(os.path.join(DATA_PATH, DATASETS[0], 'processed_articles.json'), mode='w') as fp:
    json.dump(processed_articles, fp, indent=4)

X1, feature_names1 = Vectorizer.tfidf(map(itemgetter('entities_title'), processed_articles))
X2, feature_names2 = Vectorizer.tfidf(map(itemgetter('keyphrases_title'), processed_articles))
X3, feature_names3 = Vectorizer.tfidf(map(itemgetter('keyphrases_text'), processed_articles))

entities_title   = Vectorizer.make_data(X1, feature_names1)
keyphrases_title = Vectorizer.make_data(X2, feature_names2)
keyphrases_text  = Vectorizer.make_data(X3, feature_names3)

with open(os.path.join(DATA_PATH, DATASETS[0], 'entities_title.json'), mode='w') as fp:
    json.dump(entities_title, fp, indent=4)

with open(os.path.join(DATA_PATH, DATASETS[0], 'keyphrases_title.json'), mode='w') as fp:
    json.dump(keyphrases_title, fp, indent=4)

with open(os.path.join(DATA_PATH, DATASETS[0], 'keyphrases_text.json'), mode='w') as fp:
    json.dump(keyphrases_text, fp, indent=4)


try:
    labels_true = map(itemgetter('label'), articles)
    mapping = { label: i for i, label in enumerate(set(labels_true)) }
    labels_true = itemgetter(*labels_true)(mapping)

    with open(os.path.join(DATA_PATH, DATASETS[0], 'labels_true.json'), mode='w') as fp:
        json.dump(labels_true, fp, indent=4)
except:
    print('No labels')