import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.preprocessing import normalize

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

#Initializations
Y = genfromtxt('lol.csv', delimiter = ',')
FinalDict = genfromtxt('DiscDictionary.csv', delimiter = ',')
Y = np.array(Y)
FinalDict = np.array(FinalDict)
D = np.ones(n)
D = D.reshape(n,1)
aksvd = ApproximateKSVD(n_components=int(K/m))
for i in range(0,m):
	currinp = Y[i*(int(N/m)):i*(int(N/m))+int(N/m),:]
	print(currinp.shape)
	dictionary = aksvd.fit(currinp).components_
	dictionary = np.transpose(dictionary)
	print(dictionary.shape)
	D = np.column_stack((D,dictionary))
D = D[:,1:]
X = aksvd._transform(D.T,Y)
ans = X.dot(D.T)
ex = ans[301]
ex = ex.reshape(28,28)
plt.imshow(ex)
plt.show()
Y = Y.T
Q = np.ones(int((K*N)/(m*m)))
Q = Q.reshape(int((K/m)),int((N/m)))
Q = block_diag(Q,Q,Q,Q,Q,Q,Q,Q,Q,Q) #m times
print(Q.shape)
H = np.ones(int(int((m*N))/int((m*m))))
H = block_diag(H,H,H,H,H,H,H,H,H,H)  #m times
print(H.shape)

#initialize A
ones = np.eye(K,K)
A = (X.T.dot(X) + lam2*ones)
A = np.linalg.inv(A)
A = A.dot(X.T)
A = np.transpose(A.dot(Q.T))
print(A.shape)

#initialize W
W = (X.T.dot(X) + lam1*ones)
W = np.linalg.inv(W)
W = W.dot(X.T)
W = np.transpose(W.dot(H.T))
print(W.shape)

# #Final Training
# FinalDict = np.row_stack((D,A,W))
FinalInp = np.row_stack((Y,Q,H))
# print(FinalDict.shape)
# print(FinalInp.shape)
aksvd = ApproximateKSVD(n_components=int(K))
# FinalDict = aksvd.fitwithdict(FinalInp.T,FinalDict.T).components_
# print(FinalDict.shape)
# with open('DiscDictionary.csv', 'w') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(FinalDict)
# csvFile.close()
FinalDict = normalize(FinalDict, axis=1, norm='l2')
print(FinalDict.shape)
X = aksvd._transform((FinalDict.T[:784,:]).T,Y.T)
print(X.shape)
with open('Discsparse.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(X)
csvFile.close()
# print(mylist.shape)
# ex = mylist[23]
# ex = ex.reshape(28,28)
# plt.imshow(ex)
# plt.show()
# aksvd = ApproximateKSVD(n_components=500)
# dictionary = aksvd.fit(mylist).components_
# gamma = aksvd.transform(mylist)
# print(gamma)
# ans = gamma.dot(dictionary)
# ex = ans[23]
# ex = ex.reshape(28,28)

# #Save the details
# # with open('Dictionary.csv', 'w') as csvFile:
# #     writer = csv.writer(csvFile)
# #     writer.writerows(dictionary)
# # csvFile.close()

# # with open('Sparsecode.csv', 'w') as csvFile:
# #     writer = csv.writer(csvFile)
# #     writer.writerows(gamma)
# # csvFile.close()

# plt.imshow(ex)
# plt.show()
