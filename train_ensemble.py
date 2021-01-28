from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

from sklearn.model_selection import train_test_split

print(tf.__version__)

#Checking GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
import time

concrete_data = pd.read_csv("./cement.csv")

t = time.time()
min_d = concrete_data.min()
max_d = concrete_data.max()
normalized_df=(concrete_data - min_d)/(max_d - min_d)

normal_train = normalized_df.iloc[:,:8]
normal_label = normalized_df.iloc[:,-1:]


xtrain, xtest, ytrain, ytest = train_test_split(normal_train, normal_label, test_size = 0.15, random_state = 1)

print(xtrain.shape)
print(ytrain.shape)


print(xtest.shape)
print(ytest.shape)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 1000 == 0:
      print('epoch: ', epoch,' loss: ', logs["loss"], "val_loss", logs["val_loss"])
      
      
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    plt.legend()
    plt.show()

def fit_model(xtrain,ytrain,A):
    model = keras.Sequential([
        layers.Dense(A[0], activation=tf.nn.relu, input_shape=[len(xtrain.keys())]),
        layers.Dense(A[1], activation=tf.nn.relu),
        layers.Dense(A[2], activation=tf.nn.relu),
        layers.Dense(A[3], activation=tf.nn.sigmoid)
        ])
    optimizer = tf.keras.optimizers.SGD(lr=A[4])
    model.compile(loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])
  
    history = model.fit(xtrain, ytrain,epochs=A[5], verbose=0,validation_split = 0.2,callbacks=[PrintDot()])
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.head())
    print(hist.tail())
    plot_history(history)
    test_predictions = model.predict(xtest).flatten()
    plot_history(history)
    test_predictions = model.predict(xtest).flatten()

    plt.scatter(ytest, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0,plt.xlim()[1]])
    plt.ylim([0,plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    print(time.time()-t)
    return model

# create directory for models
os.makedirs('./models')

n_members = 5
for i in range(0,n_members):
    trn_mtrx = [[30,25,15,1,0.01,10000],[25,20,10,1,0.02,8000],[27,23,12,1,0.04,9000],[26,24,12,1,0.01,7000],[25,21,12,1,0.01,9000]]
    model = fit_model(xtrain, ytrain, trn_mtrx[i])
    filename = './models/model_' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)



































