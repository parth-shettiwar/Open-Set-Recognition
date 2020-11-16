from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers
import numpy as np
from numpy import genfromtxt

model = Sequential()
model.add(Dense(100, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(784))
sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

X_train = genfromtxt('zeros.csv', delimiter = ',')
X_train = np.array(X_train)
X_train = X_train[:,1:]
X_train = X_train.T
y_train = X_train[:,500:]
X_train = X_train[:,0:500]
X_train = X_train.T
y_train = y_train.T
# y_train = X_train[:,0]
# X_train = X_train[:,1:]
X_valid = X_train
hist = model.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.2)

y_pred = model.predict_classes(X_valid)
print(y_train)
print(y_pred)