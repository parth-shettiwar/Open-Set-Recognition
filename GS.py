import numpy as np
from numpy import genfromtxt
from ksvd import ApproximateKSVD
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.feature_extraction import image
import sympy as sp
import numpy as np
from numpy.linalg import svd

def orthonormalize(n, basis):
    if (not n == len(basis)) or not (n == len(b) for b in basis):
        raise ValueError("dimension does not match basis!")

    if not linearly_independent(basis):
        raise ValueError("original basis not linearly independent!")
    v = [[0] * n] * n
    output = [[0] * n] * n
    for i in range(0, n):
        j = i - 1
        v[i] = basis[i]
        while j >= 0:
            v[i] = sub(v[i], mult(inner_product(basis[i], output[j]), output[j]))
            j -= 1
        output[i] = mult(1 / norm(v[i]), basis[i])
    return output


def mult(c, v2):
    output = [0] * len(v2)
    for i in range(0, len(v2)):
        output[i] = c * v2[i]
    return output


def linearly_independent(vectors):
    eigenvalues, eigenvectors = numpy.linalg.eig(vectors)
    if 0.0 in numpy.absolute(eigenvalues):
        return False
    return True


def sub(v1, v2):
    output = [0] * len(v1)
    for i in range(0, len(v1)):
        output[i] = v1[i] - v2[i]
    return output


def inner_product(v1, v2):
    output = 0
    for i in range(0, len(v1)):
        output += v1[i] * v2[i]
    return output


def norm(v):
    return numpy.sqrt(inner_product(v, v))


def rank(A, atol=1e-13, rtol=0):

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-13, rtol=0):
   
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def orthogonal_complement(x, normalize=True, threshold=1e-15):

    x = np.asarray(x)
    r, c = x.shape
    if r < c:
        import warnings
        warnings.warn('fewer rows than columns', UserWarning)
    s, v, d = np.linalg.svd(x)
    rank = (v > threshold).sum()

    oc = s[:, rank:]

    if normalize:
        k_oc = oc.shape[1]
        oc = oc.dot(np.linalg.inv(oc[:k_oc, :]))
    return oc