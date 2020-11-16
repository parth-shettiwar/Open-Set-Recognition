import numpy as np
import scipy as sp
import pandas as pd
import urllib.request
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import KernelCenterer
from numpy import genfromtxt
from sklearn.manifold import TSNE
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

def _hik(x, y):

    return np.minimum(x, y).sum()

X_train = genfromtxt('Discsparse.csv', delimiter = ',')
X_train = np.array(X_train)
labels = np.ones(1000)
labels[0:100] = 0
labels[100:200] = 1
labels[200:300] = 2
labels[300:400] = 3
labels[400:500] = 4
labels[500:600] = 5
labels[600:700] = 6
labels[700:800] = 7
labels[800:900] = 8
labels[900:1000] = 9


from scipy.linalg import svd

def nullspace(A, eps=1e-12):
    u, s, vh = svd(A)
    null_mask = (s <= eps)
    null_space = sp.compress(null_mask, vh, axis=0)
    return sp.transpose(null_space)


def learn(K, labels):
    classes = np.unique(labels)
    if len(classes) < 2:
        raise Exception("KNFST requires 2 or more classes")
    n, m = K.shape
    if n != m:
        raise Exception("Kernel matrix must be quadratic")
    
    centered_k = KernelCenterer().fit_transform(K)
    
    basis_values, basis_vecs = np.linalg.eigh(centered_k)
    
    basis_vecs = basis_vecs[:,basis_values > 1e-12]
    basis_values = basis_values[basis_values > 1e-12]
 
    basis_values = np.diag(1.0/np.sqrt(basis_values))

    basis_vecs  = basis_vecs.dot(basis_values)

    L = np.zeros([n,n])
    for cl in classes:
        for idx1, x in enumerate(labels == cl):
            for idx2, y in enumerate(labels == cl):
                if x and y:
                    L[idx1, idx2] = 1.0/np.sum(labels==cl)
    M = np.ones([m,m])/m
    H = (((np.eye(m,m)-M).dot(basis_vecs)).T).dot(K).dot(np.eye(n,m)-L)
    
    t_sw = H.dot(H.T)
    print(t_sw.shape)
    eigenvecs = nullspace(t_sw)
    print("Eigenvec shape",eigenvecs.shape)
    if eigenvecs.shape[1] < 1:
        eigenvals, eigenvecs = np.linalg.eigh(t_sw)
        
        eigenvals = np.diag(eigenvals)
        min_idx = eigenvals.argsort()[0]
        eigenvecs = eigenvecs[:, min_idx]
    proj = ((np.eye(m,m)-M).dot(basis_vecs)).dot(eigenvecs)
    labels_points = []
    for cl in classes:
        k_cl = K[labels==cl, :]        
        pt = np.mean(k_cl.dot(proj), axis=0)
        labels_points.append(pt)
        
    return proj, np.array(labels_points)

kernel_mat = metrics.pairwise_kernels(X_train, metric = 'rbf')
proj, labels_points = learn(kernel_mat, labels)
ks = metrics.pairwise_kernels(X_train,X_train, metric = 'rbf')
print("Proj_shape",proj.shape)
print(labels_points.shape)
new = np.row_stack((proj,labels_points))
X_embedded = TSNE(n_components=2).fit_transform(proj)
print(X_embedded.shape)

#Write TSNE points to a CSV
with open('Tsnedisc.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(X_embedded)
csvFile.close()
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()


#Discriminator
N = 100 # number of points per class
D = 9 # dimensionality
K = 10 # number of classes
X = proj
X = np.array(X)
X = normalize(X, axis=1, norm='l1')
print(X.shape) 
llabels = np.ones(1000, dtype='uint8')
llabels[0:100] = 0
llabels[100:200] = 1
llabels[200:300] = 2
llabels[300:400] = 3
llabels[400:500] = 4
llabels[500:600] = 5
llabels[600:700] = 6
llabels[700:800] = 7
llabels[800:900] = 8
llabels[900:1000] = 9
# labels = labels.reshape(1,1000)
y = llabels
print(y.shape)
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()

# initialize parameters randomly
h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
  
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(correct_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print ("iteration %d: loss %f" % (i, loss))

  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  dhidden = np.dot(dscores, W2.T)
  dhidden[hidden_layer <= 0] = 0
  dW = np.dot(X.T, dhidden)
  db = np.sum(dhidden, axis=0, keepdims=True)
  
  # add regularization gradient contribution
  dW2 += reg * W2
  dW += reg * W
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2

# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ((np.mean(predicted_class == y)))