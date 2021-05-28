import numpy as np
import pke
import spacy
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load('en_core_web_lg')


class Preprocessor:

    @staticmethod
    def get_entities_spacy(text):
        return [ ent.text for ent in nlp(text).ents ]
    
    @staticmethod
    def get_keyphrases_pke(text):
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(text, normalization=None)
        extractor.candidate_selection()
        extractor.candidate_weighting()
        return list(chain.from_iterable([kp] * text.count(kp) for kp, _ in extractor.get_n_best(n=1000)))


class Vectorizer:

    @staticmethod
    def tfidf(corpus):
        vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, preprocessor=lambda doc: doc)
        X = vectorizer.fit_transform(corpus)
        feature_names = np.array(vectorizer.get_feature_names())
        return X, feature_names

    @staticmethod
    def make_data(X, feature_names):
        data = []
        for i in range(X.shape[0]):
            v = X[i].todense().A1
            idx = np.where(v > 0)[0]
            data.append(dict(zip(feature_names[idx], v[idx])))
        return data
