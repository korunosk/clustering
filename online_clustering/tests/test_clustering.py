import unittest

import numpy as np
import scipy.sparse as sp

from online_clustering.modules.clustering import Article, Clusters


class ClustersTestCase(unittest.TestCase):

    def test_resize_horizontaly(self):
        article_dict = [
            { 'a': 1 },
            { },
            { 'b': 2, 'c': 3 },
        ]
        article_repr_1 = sp.csr_matrix([[1]])
        article_repr_2 = sp.csr_matrix([[]])
        article_repr_3 = sp.csr_matrix([[2, 3]])
        n = np.array([1])
        article = Article(article_dict)
        c = Clusters(dict(a=1, b=0, c=0))

        c.resize_horizontaly(article_dict)
        shape = c.get_shape()
        self.assertEqual(shape[0], (0, 1))
        self.assertEqual(shape[1], (0, 0))
        self.assertEqual(shape[2], (0, 2))

    
    def test_resize_verticaly(self):
        article_dict = [
            { 'a': 1 },
            { },
            { 'b': 2, 'c': 3 },
        ]
        article_repr_1 = sp.csr_matrix([[1]])
        article_repr_2 = sp.csr_matrix([[]])
        article_repr_3 = sp.csr_matrix([[2, 3]])
        n = np.array([1])
        article = Article(article_dict)
        c = Clusters(dict(a=1, b=0, c=0))

        c.resize_verticaly()
        shape = c.get_shape()
        self.assertEqual(shape[0], (1, 0))
        self.assertEqual(shape[1], (1, 0))
        self.assertEqual(shape[2], (1, 0))

    def test_vocab(self):
        article_dict = [
            { 'a': 1 },
            { },
            { 'b': 2, 'c': 3 },
        ]
        article_repr_1 = sp.csr_matrix([[1]])
        article_repr_2 = sp.csr_matrix([[]])
        article_repr_3 = sp.csr_matrix([[2, 3]])
        n = np.array([1])
        article = Article(article_dict)
        c = Clusters(dict(a=1, b=0, c=0))

        c.resize_horizontaly(article_dict)
        vocab = c.get_vocab()
        self.assertEqual(vocab[0], { 'a': 0 })
        self.assertEqual(vocab[1], { })
        self.assertEqual(vocab[2], { 'b': 0, 'c': 1 })
    
    def test_add_article(self):
        article_dict = [
            { 'a': 1 },
            { },
            { 'b': 2, 'c': 3 },
        ]
        article_repr_1 = sp.csr_matrix([[1]])
        article_repr_2 = sp.csr_matrix([[]])
        article_repr_3 = sp.csr_matrix([[2, 3]])
        n = np.array([1])
        article = Article(article_dict)
        c = Clusters(dict(a=1, b=0, c=0))

        c.resize_horizontaly(article_dict)
        c.resize_verticaly()
        c.add_article(0, 0, article)
        centers = c.get_centers()
        self.assertEqual(np.abs(article_repr_1 - centers[0]).sum(), 0)
        self.assertEqual(np.abs(article_repr_2 - centers[1]).sum(), 0)
        self.assertEqual(np.abs(article_repr_3 - centers[2]).sum(), 0)


class ArticleTestCase(unittest.TestCase):
    
    def test_get_article_repr(self):
        article_dict = [
            { 'a': 1 },
            { },
            { 'b': 2, 'c': 3 },
        ]
        article_repr_1 = sp.csr_matrix([[1]])
        article_repr_2 = sp.csr_matrix([[]])
        article_repr_3 = sp.csr_matrix([[2, 3]])
        n = np.array([1])
        article = Article(article_dict)
        c = Clusters(dict(a=1, b=0, c=0))

        c.resize_horizontaly(article_dict)
        vocab = c.get_vocab()
        article_repr = article.get_article_repr(vocab)
        self.assertEqual(np.abs(article_repr_1 - article_repr[0]).sum(), 0)
        self.assertEqual(np.abs(article_repr_2 - article_repr[1]).sum(), 0)
        self.assertEqual(np.abs(article_repr_3 - article_repr[2]).sum(), 0)
