import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from keras.datasets import cifar10, cifar100

N = 100 # number of points per class
D = 3072 # dimensionality
K = 100 # number of classes
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
X = x_train
X = X.reshape(50000,3072)
U = np.ones(3072)
U = U.reshape(1,3072)
for i in range(100):
  Xn = X[500*i:500*i+100,]
  print(Xn.shape)
  U = np.row_stack((U,Xn))
X = U[1:,]
print(X.shape)
y = y_train
X = np.array(X)
# X = normalize(X, axis=1, norm='l2')
# X = X[10:,:]
print(X.shape) 
# labels = np.ones(1000, dtype='uint8')
# labels[0:100] = 0
# labels[100:200] = 1
# labels[200:300] = 2
# labels[300:400] = 3
# labels[400:500] = 4
# labels[500:600] = 5
# labels[600:700] = 6
# labels[700:800] = 7
# labels[800:900] = 8
# labels[900:1000] = 9
# # labels = labels.reshape(1,1000)
# y = labels
print(y.shape)
# lets visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
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
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters
  # first backprop into parameters W2 and b2
  dW2 = np.dot(hidden_layer.T, dscores)
  db2 = np.sum(dscores, axis=0, keepdims=True)
  # next backprop into hidden layer
  dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
  dhidden[hidden_layer <= 0] = 0
  # finally into W,b
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