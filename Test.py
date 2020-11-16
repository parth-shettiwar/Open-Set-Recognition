import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

dictionary = genfromtxt('DiscDictionary.csv', delimiter = ',')
gamma = genfromtxt('Discsparse.csv', delimiter = ',')
mylist = genfromtxt('lol.csv', delimiter = ',')

#test
dictionary = np.array(dictionary)
dictionary = (dictionary.T[:784,:]).T
dictionary = normalize(dictionary, axis=1, norm='l2')
gamma = np.array(gamma)
print(dictionary.shape)
print(gamma.shape)
ans = gamma.dot(dictionary)
ex = dictionary[21]
ex = ex.reshape(28,28)
plt.imshow(ex)
plt.show()

#tsne
X_embedded = TSNE(n_components=2).fit_transform(gamma)
y=np.zeros(1000)
y[:99]=0
y[100:199]=1
y[200:299]=2
y[300:399]=3
y[400:499]=4
y[500:599]=5
y[600:699]=6
y[700:799]=7
y[800:899]=8
y[900:999]=9
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]
plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 10), marker='.')
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()