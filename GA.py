# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:58:03 2020
@author: Shahruj Rashid
"""
from random import random
from random import uniform
import math
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import numpy as np
#import pickle
import random
from matplotlib  import cm 
#model1 = pickle.load(open("model.dat", "rb"))
model2 = keras.models.load_model('my_model_2')
#model = pickle.load(open("model.dat", "rb"))


#function returns the output of the model when passed in a particle's parameters
#add models here  

bounds = np.array([[102,540],[0,359],[0,200],[121,247],[0,32],[801,1145],[594,992],[1,365]])

def evaluate_position(temp):
    temp = np.asarray(temp)
    CNNout=model2.predict(temp)
    CNNout = (CNNout*(82.599225-2.331808))+2.331808
    return CNNout

def fitness(arr):
    ideal = 100
    temp = np.asarray(arr)
    temp = np.absolute(arr-ideal)
    return temp    

       
class Population:
    def __init__(self,n,dim):
        self.pop=np.zeros((n,dim+2))
        self.bestpop = np.zeros((4,dim+1))
        self.dimension = dim
        self.pop_size = n
        for i in range(0,n):
            for j in range(0,self.dimension):
                self.pop[i,j] = random.randint(bounds[j,0], bounds[j,1])
    
    def evaluate_gene(self):
        temp=np.divide(np.subtract(self.pop[:,0:self.dimension],bounds[:,0]),np.subtract(bounds[:,1],bounds[:,0]))
        self.pop[:,self.dimension]=evaluate_position(temp).transpose()
        
    def evaluate_fitness(self):
        self.pop[:,self.dimension+1]=fitness(self.pop[:,self.dimension])
        
    def selection(self):
        arg = self.pop[:,self.dimension+1].argsort()[0:4]
        self.bestpop = self.pop[arg,:]
    
    def crossover(self):
        self.pop[0:4,:]=self.bestpop
        for i in range(4,int(4+(self.pop_size-4)*0.5)):
            sel = [0,1,2,3]
            a = sel.pop(random.randint(0,len(sel)-1))
            b = sel.pop(random.randint(0,len(sel)-1))
            self.pop[i,0:self.dimension//2] = self.bestpop[a,0:self.dimension//2]
            self.pop[i,self.dimension//2:self.dimension] = self.bestpop[b,self.dimension//2:self.dimension]
               
    def mutate(self):
        for i in range(int(4+(self.pop_size-4)*0.5),self.dimension):
            sampl = np.random.uniform(low=0, high=1, size=(1,self.dimension))
            a = random.randint(0,3)
            self.pop[i,0:self.dimension]=np.add(self.bestpop[a,0:self.dimension],np.multiply(self.bestpop[a,0:self.dimension],sampl))
            for j in range(0,self.dimension):
                if self.pop[i,j]>=bounds[j,1]: #prevents search space from exceeding the bound (upper limit)
                    self.pop[i,j]=bounds[j,1] 
                if self.pop[i,j]<=bounds[j,0]: #prevents search space from exeeding the bound (lower limit)
                    self.pop[i,j]=bounds[j,0]
    def print_generation(self):
        print(self.pop)
            
        
np.set_printoptions(suppress=True)
        
def GA(bounds,n_particles,max_gen,dimension):
    pop = Population(n_particles,dimension)
    generation = 0
    while(generation <= max_gen):
        print("generation: "+str(generation))
        pop.evaluate_gene()
        pop.evaluate_fitness()
        pop.selection()
        pop.crossover()
        pop.mutate()
        pop.print_generation()
        generation +=1
    
#perform GA wirh model,bounds, 13 paricles, 100 iteration and 8 dimensions
GA(bounds,20,100,8)





   