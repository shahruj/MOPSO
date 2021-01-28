# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:59:10 2020

@author: Shahruj Rashid
"""
from __future__ import absolute_import, division, print_function
import random
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.model_selection import train_test_split
import time
import sklearn.metrics as sm 

os.makedirs('./PSO_HPO2')
concrete_data = pd.read_csv("./cement.csv")

t = time.time()
min_d = concrete_data.min()
max_d = concrete_data.max()
normalized_df=(concrete_data - min_d)/(max_d - min_d)

normal_train = normalized_df.iloc[:,:8]
normal_label = normalized_df.iloc[:,-1:]


xtrain, xtest, ytrain, ytest = train_test_split(normal_train, normal_label, test_size = 0.15, random_state = 1)


    
    
def predens(temp):
    model = keras.Sequential([
        layers.Dense(temp[0], activation=tf.nn.relu, input_shape=[len(xtrain.keys())]),
        layers.Dense(temp[1], activation=tf.nn.relu),
        layers.Dense(temp[2], activation=tf.nn.relu),
        layers.Dense(1, activation=tf.nn.sigmoid)
        ])
    optimizer = tf.keras.optimizers.SGD(lr=temp[3])
    model.compile(loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])
    model.fit(xtrain, ytrain,epochs=100)
    filename = './PSO_HPO2/model_' + str(temp[0])+"_"+str(temp[1])+"_"+str(temp[2])+"_"+str(temp[3])+'.h5'
    model.save(filename)
    a = model.predict(xtest).flatten()
    b = ytest.to_numpy().flatten()
    score = sm.r2_score(a, b)
    print(score)
    return score

class Particle:
    def __init__(self,x0,dim,bounds):
        self.position_i=[]
        self.velocity_i=[]
        self.pos_best_i=[]
        self.bestvalue_i=-1
        self.value_i=-1
        self.dimension = dim
        
        for i in range(0,self.dimension):
            if i in exclude:
                self.velocity_i.append(0)
                self.position_i.append(x0[i])
            else:
                self.velocity_i.append(random.uniform(-1,1))
                #assigns the particle random values spaced out in the search space
                if i!=3:
                    self.position_i.append(random.randint(bounds[i][0], bounds[i][1]))
                else:
                    self.position_i.append(random.uniform(bounds[i][0], bounds[i][1]))

    def evaluate(self,bounds):
        temp=self.position_i
        self.value_i=predens(temp)
        self.value_i= self.value_i
        if self.bestvalue_i<self.value_i or self.bestvalue_i == -1:
            self.pos_best_i = self.position_i.copy()
            self.bestvalue_i=self.value_i.copy()

    
    def update_velocity(self,pos_best_g):
        inertia = 0.4
        c1 = 3
        c2 = 1.5
        for i in range(0,self.dimension):
            if i in exclude:
                None
            else:
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                v2per_best=c1*r1*(self.pos_best_i[i]-self.position_i[i])
                v2glo_best=c2*r2*(pos_best_g[i]-self.position_i[i])
                self.velocity_i[i]=inertia*self.velocity_i[i]+v2per_best+v2glo_best
            
    def update_postion(self,bounds):
        for i in range(0,self.dimension):          
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            if i!=3:
                self.position_i[i]=round(self.position_i[i])
        
            if self.position_i[i]>=bounds[i][1]:
                self.position_i[i]=bounds[i][1]
            
            if self.position_i[i]<=bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
        
def pso(x0,bounds,n_particles,max_iter,dimension):
    t = time.time()
    bestvalue_g = -1
    pos_best_g = []
    particle_arr = []
    for i in range(0,n_particles):
        particle_arr.append(Particle(x0,dimension,bounds))
        particle_arr[i].evaluate(bounds)
    pos_best_g=list(particle_arr[0].position_i)
    bestvalue_g=particle_arr[0].value_i
    iterate = 0
    while(iterate<max_iter):
        print("iteration:"+str(iterate))
        for k in range(0,n_particles):
            particle_arr[k].evaluate(bounds)
#            if(k==0):
#                pos_best_g=list(particle_arr[k].position_i)
#                bestvalue_g=list(particle_arr[k].value_i)
            if(particle_arr[k].value_i>bestvalue_g):
                pos_best_g=list(particle_arr[k].position_i)
                bestvalue_g=particle_arr[k].value_i
        for k in range(0,n_particles):
            particle_arr[k].update_velocity(pos_best_g)
            particle_arr[k].update_postion(bounds)
        iterate=iterate+1
    
    print('\nFinal Solution:')
    print(pos_best_g)

   
        
        
initial = [30,25,15,0.01]
#initial = [506,230,100,171,16,950,770,28]
#initial =[527.4816090279297, 243.27777791045514, 100.728059000431, 166.47780784848885, 29.471793970240903, 948.0611656380358, 870.0998933400892, 28]
#excludes indexes from the optimisation, 7 indicates time to being fixed at 28. 
exclude =[]

bounds = [[10,60],[10,60],[10,60],[0.001,1]]
#pso(model,start value,bounds,n_particles,max_iter,dimension):
pso(initial,bounds,4,10,4)   
