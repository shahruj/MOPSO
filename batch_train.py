import numpy as np
import pandas as pd   
import tensorflow as tf   
from tensorflow import keras
from time import gmtime, strftime 
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import boto3, re, sys, math, json, os, urllib.request,time
import subprocess
import sys
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
args, _ = parser.parse_known_args()
data_dir = args.data

x_train  = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

x_test = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  df = pd.DataFrame({"Epoch":hist['epoch'],"Mean Abs Error":hist['mean_absolute_error'],"val_mean_absolute_error":hist['val_mean_absolute_error'],"MSE":hist['mean_squared_error'],"val_MSE":hist['val_mean_squared_error']})
  
  return df


def build_model():
    model = keras.Sequential([
      layers.Dense(30, activation=tf.nn.relu, input_shape=[len(x_train.keys())]),
      layers.Dense(25, activation=tf.nn.relu),
      layers.Dense(15, activation=tf.nn.relu),
      layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    optimizer = tf.keras.optimizers.SGD(lr=0.01)
    
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

if __name__ =='__main__':
    model = build_model()
    data_location = "/opt/ml/model/"
    print("yes")
    history = model.fit(x_train,y_train,epochs=5,verbose=0,validation_split = 0.2)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch    
    
    error = plot_history(history)
    test_predictions = model.predict(x_test).flatten()
    
    pred = pd.DataFrame({"predictions":test_predictions,"ytest":y_test.iloc[:,0]})
    print(error.head())
    print(pred.head())

    pred.to_csv("pred.csv")
    error.to_csv("error.csv")
    
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file("pred.csv", 'fypcementbucket92485661',"model/pred.csv")
    s3.meta.client.upload_file("error.csv", 'fypcementbucket92485661',"model/error.csv")
    
    tf.contrib.saved_model.save_keras_model(model, data_location)
    
    
    
    
    
    
    
    
