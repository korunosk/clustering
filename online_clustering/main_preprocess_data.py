import json
import numpy as np
from tqdm import tqdm

from modules.preprocessing import Preprocessor
from config import MAX_NUM_ARTICLES
from loaders import make_data_path, load_data


if __name__ == '__main__':

    articles = load_data('articles.json')
    # # Only for LSIR data
    # np.random.seed(42)
    # articles = np.random.choice(articles, MAX_NUM_ARTICLES, replace=False).tolist()
    # np.random.seed()

    with open(make_data_path('processed_articles'), mode='w') as fp:
        for article in tqdm(articles):
            processed_article = {
                'entities_title': Preprocessor.get_entities_spacy(article['title']),
                'keyphrases_title': Preprocessor.get_keyphrases_pke(article['title']),
                'keyphrases_text': Preprocessor.get_keyphrases_pke(article['text']),
            }
            line = json.dump(processed_article, fp)
            fp.write('\n')
