# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:58:03 2020

@author: Shahruj Rashid
"""

from __future__ import absolute_import, division, print_function
from random import random
from random import uniform
import math
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import numpy as np
import pickle
import random
from matplotlib  import cm
from mpl_toolkits.mplot3d import Axes3D 
 
model1 = pickle.load(open("model.dat", "rb"))
model2 = keras.models.load_model('my_model_2')
#model = pickle.load(open("model.dat", "rb"))


#function returns the output of the model when passed in a particle's parameters
#add models here  
def evaluate_position(temp):
    temp = np.asarray(temp)
    CNNout=model2.predict(temp.reshape(1,8))[0]
#    CNNout=model1.predict(temp.reshape(1,8))[0]
    CNNout = (CNNout*(82.599225-2.331808))+2.331808
    return CNNout

def evaluate_fitness(arr):
    ideal = [100,0,180]
    fitness = 0
    for i in range(0,len(arr)):
        fitness = fitness + (ideal[i]-arr[i])**2
    return math.sqrt(fitness)    


class Particle:
    def __init__(self,dim):
        #parameters of the particl    
        self.position_i=[] 
        self.velocity_i=[]      
        self.param_1=-1
        self.param_2=-1 
        self.param_3=-1
        
        self.pos_best_i=[] #pos_best_i contains the individual best position of the particle       
        self.bestparam_1=-1
        self.bestparam_2=-1
        self.bestparam_3=-1
        
        self.fitness = math.inf
        self.dimension = dim 
        #defines the dimension of the particle where dim = dimension of the search space
        
        for i in range(0,self.dimension): 
            #instantiates the particle according to the search space dimension
            if i in exclude: #does not instantiates particle which we want to exclude from the search 
                self.velocity_i.append(0) #makes velocity in that direction 0
                self.position_i.append(x0[i]) #assigns the input value to that variable 
            else:
                self.velocity_i.append(uniform(-1,1)) #assigns a random velocity
                self.position_i.append(random.randint(bounds[i][0], bounds[i][1])) #assigns a random position according to the bounds

        
    def evaluate(self,bounds,price,meow): # this function evaluates the particle 
        temp=[]
        self.param_2 = 0
        self.param_3 = 0
        for i in range(0,self.dimension): #fills up temp with the position parameters of the particle to be evaluated
            temp.append((self.position_i[i]- bounds[i][0])/(bounds[i][1] - bounds[i][0]))
            self.param_2 = self.param_2 + (self.position_i[i]*price[i]) #finds the cost of the position being evaluated and updates individual cost of the position
            self.param_3 = self.param_3 + (self.position_i[i]*meow[i])
        self.param_1=evaluate_position(temp) #updates the value(UCS) of the position returned by the model according to input position      
        if ((self.param_1>self.bestparam_1 and self.param_2<self.bestparam_2 and self.param_3>self.bestparam_3)): #checks if the current cost and value dominates the personal best (cost and value)
            self.pos_best_i = self.position_i #updates personal best if it dominates 
            self.bestparam_1= self.param_1
            self.bestparam_2 = self.param_2
            self.bestparam_3 = self.param_3
        elif (self.param_1==self.bestparam_1 and self.param_2==self.bestparam_2 and self.param_3==self.bestparam_3): #if the current values don't dominate the personal best, doesn't do anything 
            toss = random.uniform(0, 1) 
            if(toss>=0.5):
                self.pos_best_i = self.position_i
                self.bestparam_1= self.param_1
                self.bestparam_2 = self.param_2
                self.bestparam_3 = self.param_3
        self.fitness = evaluate_fitness([self.param_1,self.param_2,self.param_3])
        
        
    def instantiate(self,bounds,price,meow):
        temp=[]
        self.param_2=0
        self.param_3=0
        for i in range(0,self.dimension):
            temp.append((self.position_i[i]- bounds[i][0])/(bounds[i][1] - bounds[i][0]))
            self.param_2 = self.param_2 + (self.position_i[i]*price[i])
            self.param_3 = self.param_3 + (self.position_i[i]*meow[i])
        self.param_1=evaluate_position(temp)
        self.bestparam_1 = self.param_1
        self.bestparam_2 = self.param_2
        self.bestparam_3 = self.param_3
        self.pos_best_i = self.position_i
        self.fitness = evaluate_fitness([self.param_1,self.param_2])
        
        
        
    def update_velocity(self,paretofront,pareto_index,best_fitness, k): #this function decides the velocity for the next iteration
        inertia = 0.4
        c1 = 3
        c2 = 1.5
        h = 0
        toss=random.uniform(0,1) #does a toss
        if(toss<0.8): #with a 0.8 probability, it chooses from the fittest values in the repository 
            h=random.randint(0, len(best_fitness)-1)
            h=best_fitness[h]
        else: #else it chooses a random value from the repository
            h=random.randint(0, len(pareto_index)-1)
            h=pareto_index[h]
        for i in range(0,self.dimension): #updates the velocity for each dimension
            if i not in exclude: #doesn't update for paramters we exclude. 
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                v2per_best=c1*r1*(self.pos_best_i[i]-self.position_i[i]) #finds vector to the personal best
                v2glo_best=c2*r2*(paretofront[h,i]-self.position_i[i]) #finds vector to one of the points on the repository decided by the toss                
                self.velocity_i[i]=inertia*self.velocity_i[i]+v2per_best+v2glo_best #finds the velocity for the next iteration
        
            
    def update_postion(self,bounds, k): #updates position according to calculated velocity 
        for i in range(0,self.dimension): 
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            #print("position[i]"+str(self.position_i[i])+">="+str(bounds[i][1])+"upp bounds")
            if self.position_i[i]>=bounds[i][1]: #prevents search space from exceeding the bound (upper limit)
                self.position_i[i]=bounds[i][1]
            #print("position[i]"+str(self.position_i[i])+">="+str(bounds[i][0])+"low bounds")           
            if self.position_i[i]<=bounds[i][0]: #prevents search space from exeeding the bound (lower limit)
                self.position_i[i]=bounds[i][0]
        
np.set_printoptions(suppress=True)
        
def mopso(x0,bounds,n_particles,max_iter,dimension):
    #last 3 index is param1, param2, and fitness accordingly
    allindex = list(range(100))
    paretofront=np.zeros((100,dimension+4))
    paretofront[:,dimension+2]=math.inf
    best_fitness=[]
    no_of_fitness = 5
    pareto_index=[]
    empty_index =[]
    particle_arr = [] #instantiates the array of particles
    for i in range(0,n_particles): #appends particles accroding to n_particles
        particle_arr.append(Particle(dimension))
        particle_arr[i].instantiate(bounds,cost,meow)
        paretofront[i,0:dimension]=(list(particle_arr[i].position_i))
        paretofront[i,dimension]=particle_arr[i].param_1
        paretofront[i,dimension+1]=particle_arr[i].param_2
        paretofront[i,dimension+2]=particle_arr[i].param_3
        paretofront[i,dimension+3]=particle_arr[i].fitness
        pareto_index.append(i)
    empty_index=list(set(allindex)-set(pareto_index))
    
    iterate = 0
    plt.scatter(paretofront[:,8], paretofront[:,9])

    while(iterate<=max_iter):
        #plt.scatter(paretofront[pareto_index,8], paretofront[pareto_index,9],paretofront[pareto_index,10])
        print("iteration: "+str(iterate))
        if(len(pareto_index)>no_of_fitness):
            best_fitness = np.argpartition(paretofront[pareto_index,dimension+3], no_of_fitness)[0:no_of_fitness].tolist()            
        else:
            best_fitness =  pareto_index
        for k in range(0,n_particles): #for loop for each particle 
            particle_arr[k].evaluate(bounds,cost,meow) #evaluates the particle
            pop=[]
            for i in pareto_index: #checks if the particle dominates any of the point currently on the pareto front 
                if((particle_arr[k].param_1>paretofront[i,8] and particle_arr[k].param_2<paretofront[i,9] and particle_arr[k].param_3>paretofront[i,10])):
                    pop.append(i) #keeps track of the dominated indexes on the paretofront
            if(len(empty_index)!=0):
                paretofront[empty_index[0],0:dimension]=(list(particle_arr[k].position_i))
                paretofront[empty_index[0],dimension]=particle_arr[k].param_1
                paretofront[empty_index[0],dimension+1]=particle_arr[k].param_2
                paretofront[empty_index[0],dimension+2]=particle_arr[k].param_3
                paretofront[empty_index[0],dimension+3]=particle_arr[k].fitness  
                pareto_index.append(empty_index[0])
                empty_index.pop(0)              
            empty_index = empty_index + pop        
            particle_arr[k].update_velocity(paretofront,pareto_index,best_fitness, k)
            particle_arr[k].update_postion(bounds, k)                                
            pareto_index = [x for x in pareto_index if x not in pop]
        plt.pause(0.05)
        iterate=iterate+1

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("compressive strength vs cost",fontsize=14)
    ax.set_xlabel("UCS",fontsize=12)
    ax.set_ylabel("cost",fontsize=12)
    ax.set_zlabel('3rd Param',fontsize=12)
    ax.grid(True,linestyle='-',color='0.75')
    ax.scatter(paretofront[pareto_index,dimension], paretofront[pareto_index,dimension+1],paretofront[pareto_index,dimension+2])

    
    plt.show    
    
    
    
#x0 is the initial input value
x0 =[527, 243, 101, 166, 29, 948, 870, 28]
#exclude contains the index not to optimise eg 7, means don't optimise index 7 in x0 which stays as 28 throughout 
exclude =[7]
#cost of each of the materials 
cost = [0.110,0.060,0.055,0.00024,2.940,0.010,0.006,0]
meow = [0.110,0.060,0.055,0.00024,2.940,0.010,0.006,0]
#the counds of each of the parameters 
bounds = [(102,540),(0,359),(0,200),(121,247),(0,32),(801,1145),(594,992),(1,365)]

#perform mopso wirh model, x0 as input, bounds, 20 paricles, 400 iteration and 8 dimensions
mopso(x0,bounds,20,50,8)   
