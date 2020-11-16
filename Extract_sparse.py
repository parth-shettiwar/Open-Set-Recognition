import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.linear_model import orthogonal_mp_gram

#Constanst
N = 1000	  #No of samples
m = 10        #No of classes
K = 900		  #No of Dictionary atoms
n = 784		  #No of features
lam1 = 1e-4   #Regularisation term for W
lam2 = 1e-4   #Regularisation term for A

class ApproximateKSVD(object):

    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, X, D, gamma):
        for j in range(self.n_components):
            I = gamma[:, j] > 0
            if np.sum(I) == 0:
                continue
            D[j, :] = 0
            g = gamma[I, j].T
            r = X[I, :] - gamma[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            gamma[I, j] = g.T
        return D, gamma

    def _initialize(self, X):
        if min(X.shape) < self.n_components:
            D = np.random.randn(self.n_components, X.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(X, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, X):
        gram = D.dot(D.T)
        Xy = D.dot(X.T)

        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * X.shape[1])

        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

    def fit(self, X):
        D = self._initialize(X)
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = np.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def fitwithdict(self, X, D):
        for i in range(self.max_iter):
            gamma = self._transform(D, X)
            e = np.linalg.norm(X - gamma.dot(D))
            if e < self.tol:
                break
            D, gamma = self._update_dict(X, D, gamma)

        self.components_ = D
        return self

    def transform(self, X):
        return self._transform(self.components_, X)


def block_diag(*arrs):
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args) 

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

X = genfromtxt('test.csv', delimiter = ',')
y = X[:,0]
X = X[:,1:]
print(X.shape)
dictionary = genfromtxt('DiscDictionary.csv', delimiter = ',')
dictionary = np.array(dictionary)
dictionary = (dictionary.T[:784,:]).T
dictionary = normalize(dictionary, axis=1, norm='l2')
print(dictionary.shape)
aksvd = ApproximateKSVD(n_components=int(K))
gamma = aksvd.fitwithdict(X, dictionary).components_
print(gamma.shape)

#tsne
X_embedded = TSNE(n_components=2).fit_transform(gamma)
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
