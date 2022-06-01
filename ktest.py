from keras.layers import Layer
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


import pandas as pd
import numpy as np

data = pd.read_csv('data/classification.csv')

X = data.iloc[:, :-1]
y = data['success']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# y = (y <= 25).astype(int)

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.losses import binary_crossentropy

model = Sequential()
# model.add(Flatten(input_shape=(28, 28)))
model.add(RBFLayer(10, 0.5, input_shape=(2,)))
model.add(Dense(1, activation='sigmoid', name='foo'))

model.compile(optimizer='rmsprop', loss=binary_crossentropy)

model.fit(x_train, y_train, batch_size=256, epochs=3)
y_pred = model.predict(x_test)
y_pred = np.round_(y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
