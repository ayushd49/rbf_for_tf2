import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MSE
import tensorflow as tf


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv('data/classification.csv')

X = data.iloc[:, :-1]
y = data['success']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()

hidden_layer = Dense(5,
                     activation='relu',
                     input_shape=(2,))

output_layer = Dense(1,
                     activation='sigmoid')

model.add(hidden_layer)
model.add(output_layer)

model.summary()

print('---------Model training starts-------')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(x_train, y_train, batch_size=30, epochs=20)

print('------------model training finishes--------')


y_pred = model.predict(x_test)
y_pred = np.round_(y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)


