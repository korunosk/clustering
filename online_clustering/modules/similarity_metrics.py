import numpy as np
import sklearn.preprocessing as pp


def cosine_similarity(A, B):
    if A.shape[1] == 0 or B.shape[1] == 1:
        return 0

    row_normed_A = pp.normalize(A, axis=1)
    row_normed_B = pp.normalize(B, axis=1)

    similarity = row_normed_A * row_normed_B.T
    
    return similarity.A

def jaccard_similarity(A, B):
    if A.shape[1] == 0 or B.shape[1] == 1:
        return 0

    rows_sum_A = A.getnnz(axis=1)
    rows_sum_B = B.getnnz(axis=1)
    ab = A * B.T
    
    aa = np.repeat(rows_sum_A, ab.getnnz(axis=1))
    bb = rows_sum_B[ab.indices]
    
    similarity = ab.copy()
    similarity.data /= (aa + bb - ab.data)

    return similarity.A
