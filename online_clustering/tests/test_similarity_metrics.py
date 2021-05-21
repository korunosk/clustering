import unittest

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import pairwise_distances

from online_clustering.modules.similarity_metrics import cosine_similarity, jaccard_similarity


class SimilarityMetricsTestCase(unittest.TestCase):

    def test_cosine_similarity(self):
        a = np.array([[1,0,0],[0,0,1]], dtype=float)
        b = np.array([[1,1,0],[0,1,0],[0,1,1]], dtype=float)
        A = sp.csr_matrix(a, dtype=float)
        B = sp.csr_matrix(b, dtype=float)
        d1 = cosine_similarity(A, B)
        d2 = 1 - pairwise_distances(a, b, metric='cosine')
        self.assertEqual(np.abs(d1 - d2).sum(), 0)
    
    def test_jaccard_similarity(self):
        a = np.array([[1,0,0],[0,0,1]], dtype=bool)
        b = np.array([[1,1,0],[0,1,0],[0,1,1]], dtype=bool)
        A = sp.csr_matrix(a, dtype=float)
        B = sp.csr_matrix(b, dtype=float)
        d1 = jaccard_similarity(A, B)
        d2 = 1 - pairwise_distances(a, b, metric='jaccard')
        self.assertEqual(np.abs(d1 - d2).sum(), 0)
