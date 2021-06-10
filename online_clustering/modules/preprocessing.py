import numpy as np
import pke
import spacy
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter


BLACKLIST = set([ 'CARDINAL', 'DATE', 'ORDINAL', 'TIME', 'PERCENT', 'QUANTITY', 'MONEY' ])

nlp = spacy.load('en_core_web_lg')


class Preprocessor:
    @staticmethod
    def get_entities_spacy(text: str) -> Counter:
        """Compute Spacy's entities with their respective counts.

        :param text: Text to analize
        :type text: str
        :return: Dictionary with entities as keys and counts as values
        :rtype: Counter
        """
        if not text:
            return dict()

        in_blacklist = lambda ent: ent.label_ in BLACKLIST
        entites = [ent.text.lower() for ent in nlp(text).ents if not in_blacklist(ent)]

        return Counter(entites)
    
    @staticmethod
    def get_keyphrases_pke(text: str) -> Counter:
        """Compute pke's keyphrases with their respective counts.

        :param text: Text to analize
        :type text: str
        :return: Dictionary with keyphrases as keys and counts as values
        :rtype: Counter
        """
        if not text:
            return dict()

        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(text, normalization=None, spacy_model=nlp)
        extractor.candidate_selection()
        extractor.candidate_weighting()

        keyphrases = [kp.lower() for kp, _ in extractor.get_n_best(n=1000)]
        
        return Counter(keyphrases)

# class Vectorizer:

#     @staticmethod
#     def tfidf(corpus):
#         vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, preprocessor=lambda doc: doc)
#         X = vectorizer.fit_transform(corpus)
#         feature_names = np.array(vectorizer.get_feature_names())
#         return X, feature_names
    
#     @staticmethod
#     def count(corpus):
#         vectorizer = CountVectorizer(tokenizer=lambda doc: doc, preprocessor=lambda doc: doc)
#         X = vectorizer.fit_transform(corpus).astype(float)
#         feature_names = np.array(vectorizer.get_feature_names())
#         return X, feature_names

#     @staticmethod
#     def make_data(X, feature_names):
#         data = []
#         for i in range(X.shape[0]):
#             v = X[i].todense().A1
#             idx = np.where(v > 0)[0]
#             data.append(dict(zip(feature_names[idx], v[idx])))
#         return data
