# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:56:21 2017

@author: Matt
"""
import numpy as np
#import sympy as sp
#import scipy as sc
#from inspect import signature

import datetime
import time
import os

from Lattice import Lattice
from Model import Model
from MonteCarloUpdate import MonteCarloUpdate
from ModelFunctions import caps,flatten

        
        
class system(object):
    # Define system class for general model of lattice sites with
    #  Lattice Parameters: Lattice Length L, Lattice Dimension d, Temperature T
    #  Lattice Model Type: Model Name and Max Spin Value q
    #  Order Parameters: Hamiltonian Coupling Constants, 
    #                    Order Parameter Function
    #  Monte Carlo Update Parameters: Perform Monte Carlo Update
    #                                 Number of initial updates Neqb,
    #                                 Number of Measurement Updates Nmeas
    #                                 Number of Clusters Ncluster
    #                                 Monte Carlo Update Algorithm
    # DataSave: Boolean
    # Animate: True
    
    def __init__(self,L=6,d=2,T=3, model=['potts',4],
                 orderparam=[0,1],
                 update = [True,None,None,1,'wolff'],
                 datasave = True,
                 animate = [True,True,True]):
        
        # Initialize model class, lattice class
        self.m = Model(model,orderparam,d)
        self.l = Lattice(L,d)
        
        # Define system parameters of:
        # Temperature, Size, Dimension, Number of Spins, 
        # Maximum spin q value and specific spin model
        self.T = np.atleast_1d(T).tolist()
        self.t = T[0]
        self.L = L
        self.d = d
        self.Nspins = self.l.Nspins
        self.q = self.m.q

       
        
        # Initialize numpy array with random spin at each site
        self.sites = self.m.state_sites(self.Nspins)

#        # Initialize nearest neighbour sites array
#        self.nn = self.l.neighbour_sites[0]
        


        
        # Save Observables data
        self.observables = []
        self.save = datasave
        self.datasave(save=False)
        




        
        # Perform Monte Carlo Updates for various Temperatures
        # Update, Transition, Animate Properties
        self.update = update
        p_trans = {'metropolis': self.m.energy,'wolff':self.m.model_params[
                                self.m.model.__name__]['bond_prob']}
        self.transition = [p_trans[self.update[-1]],self.m.state_sites]
        self.animate = animate
        self.animate.extend((self.m.state_range, self.dataName)) 
        
        self.MC = MonteCarloUpdate(self.sites,self.l.neighbour_sites,
                         self.m.observables_data,self.T,self.d,
                         self.update,self.transition,self.animate)
        


    def MonteCarlo(self):
 
        t0 = time.clock()

        observables = self.MC.MonteCarloAlgorithm()
                    
        # Save Observables Data 
        self.datasave(self.save,observables,
                      lambda : 'runtime '+str(time.clock()-t0))     
      

    

    
   
    def datasave(self,save,observables=None,*comments):
        dataDir = '%s_Data' %(caps(self.m.model.__name__))
        self.dataName         = '%s/%s_d%d_L%d__%s' %(
                        dataDir,caps(self.m.model.__name__),
                        self.d,self.L,
                        datetime.datetime.now().strftime(
                                                '%Y-%m-%d-%H-%M'))

        
        if save == True:
            # Open and write observables to a file in a specific directory
            if not observables :
                if not(os.path.isdir(dataDir)):
                        os.mkdir(dataDir)
                return
            

            # Make observables headers for file
            file = open(self.dataName+'.txt', 'w')
            headers = []
            
            # Write Observables to File
            obs_sizes = self.m.observables_sizes(
                    self.sites,self.l.neighbour_sites,self.t)
            for i,f in enumerate(self.m.observables_functions):
                for j in range(obs_sizes[i]):
                    headers.append(f.__name__+'_'+str(j+1) 
                                    if obs_sizes[i] > 1
                                    else f.__name__)
            file.write('\t'.join(headers) + '\n')
            
            # Convert lists of lists of observables to array
            for data in [list(flatten(x)) for x in observables]:
                dataline = ''
                for d in data:
                        dataline += '%0.8f \t' %(float(d))
                dataline += '\n'
                file.write(dataline)
            if comments:
                for c in comments:
                    file.write(str(c())+'\n')
            
            file.close()
        return
    

if __name__ == "__main__":
    T = [5,2.5,2,1.5,1,0.5]
    T0 = 0.5
    s = system(T=T)
    s.MonteCarlo()
