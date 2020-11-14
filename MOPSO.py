# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:06:18 2020

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
#model1 = pickle.load(open("model.dat", "rb"))
model2 = keras.models.load_model('my_model_2')
#model = pickle.load(open("model.dat", "rb"))

#function returns the output of the model when passed in a particle's parameters 
def evaluate_position(temp):
    CNN=model2.predict(np.array(temp).reshape(1,8))[0]
    CNN = (CNN*(82.599225-2.331808))+2.331808
    return CNN

#Class which defines a particle
class Particle:
    def __init__(self,dim):
        #parameters of the particl    
        self.position_i=[] #position_i is the an array which contains the individual position of the particle in the search space
        self.velocity_i=[] #velocity_i is the an array which contains the individual velocity of the particle in the search space      
        self.value_i=-1 #value is current value of the individual position (for this implementation value refers to UCS)
        self.cost=-1 #cost is current cost of the individual position
        
        
        self.pos_best_i=[] #pos_best_i an array which contains the individual best position of the particle       
        self.bestvalue_i=-1 #bestvalue_i is the best evaluated value of the individual best position
        self.bestcost=-1 #bestcost is the best evaluated cost of the individual best position

        self.dimension = dim #defines the dimension of the particle where dim = dimension of the search space
        
        for i in range(0,self.dimension): #instantiates the particle according to the search space dimension
            if i in exclude: #does not instantiates particle which we want to exclude from the search 
                self.velocity_i.append(0) #makes velocity in that direction 0
                self.position_i.append(x0[i]) #assigns the input value to that variable 
            else:
                self.velocity_i.append(uniform(-1,1)) #assigns a random velocity
                self.position_i.append(random.randint(bounds[i][0], bounds[i][1])) #assigns a position according to the bounds

        
    def evaluate(self,bounds,price): # this function evaluates the particle 
        temp=[]
        self.cost = 0
        for i in range(0,self.dimension): #fills up temp with the position parameters of the particle to be evaluated
            temp.append((self.position_i[i]- bounds[i][0])/(bounds[i][1] - bounds[i][0]))#normalizes the variable as model takes in normalised values 
            self.cost = self.cost + (self.position_i[i]*price[i]) #finds the cost of the position being evaluated and updates individual cost of the position
        self.value_i=evaluate_position(temp) #updates the value(UCS) of the position returned by the model according to input position      
        if (self.value_i>self.bestvalue_i and self.cost<self.bestcost or self.bestvalue_i == -1): #checks if the current cost and value dominates the personal best (cost and value)
            self.pos_best_i = self.position_i.copy() #updates personal best if it dominates 
            self.bestvalue_i= self.value_i.copy() 
            self.bestcost = self.cost
        elif (self.value_i<self.bestvalue_i and self.cost>self.bestcost): #if the current values don't dominate the personal best, doesn't do anything 
            None
        else: #if it neither dominates nor is it dominated, it does a toss 
            toss = random.uniform(0, 1) 
            if(toss>0.5):
                self.pos_best_i = self.position_i.copy()
                self.bestvalue_i= self.value_i.copy()
                self.bestcost = self.cost
            else:
                None

    
    def update_velocity(self,best_position,size,best_fitness): #this function decides the velocity for the next iteration
        inertia = 0.4
        c1 = 3
        c2 = 1.5
        toss=random.uniform(0,1) #does a toss
        if(toss<0.8): #with a 0.8 probability, it chooses from the fittest values in the repository 
            h=random.choice(best_fitness)
        else: #else it chooses a random value from the repository
            h=random.randint(0,size)
        for i in range(0,self.dimension): #updates the velocity for each dimension
            if i in exclude: #doesn't update for paramters we exclude. 
                None
            else: 
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                v2per_best=c1*r1*(self.pos_best_i[i]-self.position_i[i]) #finds vector to the personal best
                v2glo_best=c2*r2*(best_position[h,i]-self.position_i[i]) #finds vector to one of the points on the repository decided by the toss
                self.velocity_i[i]=inertia*self.velocity_i[i]+v2per_best+v2glo_best #finds the velocity for the next iteration
            
    def update_postion(self,bounds): #updates position according to calculated velocity 
        for i in range(0,self.dimension): 
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            if self.position_i[i]>=bounds[i][1]: #prevents search space from exceeding the bound (upper limit)
                self.position_i[i]=bounds[i][1]
            
            if self.position_i[i]<=bounds[i][0]: #prevents search spave from exeeding the bound (lower limit)
                self.position_i[i]=bounds[i][0]              
       
       
archive=[]
def mopso(x0,bounds,n_particles,max_iter,dimension):
    archive_values=np.zeros((10000,2)) #instantiates the UCS and COST of the points on the pareto front 
    best_position=np.zeros((10000,8)) #instantiates the postion of the points on the pareto front
    fitness=np.zeros(10000) #instantiates the fitness of each of the points on the pareto front 
    best_fitness=[0] #cotains the index of the points with best fitness in the archive
    size=0 #size of the paretofront 
    
    particle_arr = [] #instantiates the array of particles
    for i in range(0,n_particles): #appends particles accroding to n_particles
        particle_arr.append(Particle(dimension))
        particle_arr[i].evaluate(bounds,cost)
    iterate = 0
    best_position[0,:]=(list(particle_arr[0].position_i)) 
    archive_values[0,0],archive_values[0,1]=particle_arr[0].value_i,particle_arr[0].cost
    size=1
    while(iterate<=max_iter):
        print("iteration:"+str(iterate))
        for k in range(0,n_particles): #for loop for each particle 
            particle_arr[k].evaluate(bounds,cost) #evaluates the particle
            pop=[] 
            DOMINATED =  False
            for i in range(0,size): #checks if the particle dominates any of the point currently on the pareto front 
                if(particle_arr[k].value_i>archive_values[i,0] and particle_arr[k].cost<archive_values[i,1]):
                    DOMINATED = True
                    pop.append(i) #keeps track of the dominated indexes on the paretofront  
                elif(particle_arr[k].value_i<archive_values[i,0] and particle_arr[k].cost>archive_values[i,1]):
                    None
                else:
                    DOMINATED = True
            if(DOMINATED==True): #if the particle dominates any of the points on the paretor from 
                #removes those particles from the pareto front 
                size=size-len(pop) 
                archive_values= np.delete(archive_values,pop,axis=0) 
                best_position=np.delete(best_position,pop,axis=0)
                fitness=np.delete(fitness,pop) 
                best_position[size-1,:]=(list(particle_arr[k].position_i))
                archive_values[size-1,0]=particle_arr[k].value_i
                archive_values[size-1,1]=particle_arr[k].cost
                fitness[size-1]=math.sqrt(((archive_values[size-1,0]-80)**2)+((archive_values[size-1,1]-0)**2))
                size=size+1
        for k in range(0,n_particles): #after each iteration, recalculates the fitness values of the positions 
            particle_arr[k].update_velocity(best_position,size,best_fitness)
            particle_arr[k].update_postion(bounds)
        #plots the current pareto front 
        plt.scatter(archive_values[:,0], archive_values[:,1])
        plt.pause(0.05)
        fitness = np.ma.masked_equal(fitness,0)

        test=fitness.compressed()
        best_fitness=np.argpartition(test, 8)[-8:].tolist()
        #increments for the next iteration 
        iterate=iterate+1

    print(np.ma.masked_equal(best_position,0).compressed())
    np.savetxt("GBR_position.csv", best_position[0:size-1,:], delimiter=",",fmt='%f')
    np.savetxt("GBR_values.csv", archive_values[0:size-1,:], delimiter=",",fmt='%f')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_title("compressive strength vs cost",fontsize=14)
    ax.set_xlabel("UCS",fontsize=12)
    ax.set_ylabel("cost",fontsize=12)
    ax.grid(True,linestyle='-',color='0.75')
    ax.scatter(archive_values[:,0], archive_values[:,1])
    plt.show


    
        
        
#x0 is the initial input value
x0 =[527, 243, 101, 166, 29, 948, 870, 28]
#exclude contains the index not to optimise eg 7, means don't optimise index 7 in x0 which stays as 28 throughout 
exclude =[7]
#cost of each of the materials 
cost = [0.110,0.060,0.055,0.00024,2.940,0.010,0.006,0]

#the counds of each of the parameters 
bounds = [(102,540),(0,359),(0,200),(121,247),(0,32),(801,1145),(594,992),(1,365)]

#perform mopso wirh model, x0 as input, bounds, 20 paricles, 400 iteration and 8 dimensions
mopso(x0,bounds,20,400,8)   

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
