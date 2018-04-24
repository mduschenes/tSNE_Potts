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
from ModelFunctions import caps, display

import warnings
warnings.filterwarnings("ignore")




        
class system(object):
    # Define system class for general model of lattice sites with
    #  Lattice Parameters: Lattice Length L, Lattice Dimension d, Temperature T
    #  Lattice Model Type: Model Name and Max Spin Value q
    #  Order Parameters: Hamiltonian Coupling Constants, 
    #                    Order Parameter Function
    # Observables: List of Observed quantities
    #  Monte Carlo Update Parameters: Perform Monte Carlo Update
    #                                 Number of initial updates Neqb,
    #                                 Number of Measurement Updates Nmeas
    #                                 Measurement Update Frequency Nmeas_f
    #                                 Number of Clusters Ncluster
    # 
    # Observe: Booleans for Plot Types (Sites, Clusters, Edges, Observables)
    # DataSave: Boolean

    
    def __init__(self,L=6,d=2,T=3, model=['potts',4,[0,1]],
                 update = [True,20,10,1,1],
                 observe = [[True,'Spin Configurations','Cluster'],
                            [True,'temperature','energy','order']],
                 datasave = True):



        # Initialize model class, lattice class
        m = Model(model,d,observe[1][1:])
        l = Lattice(L,d)

        
        # Define system parameters of:
        # Size, Dimension, q states, Temperature,
        # State Range, State Generator, 
        # Transition Probability
        # Model Name
        # Monte Carlo Algorithm
        # Observables
        # Data File and Directory        
#
# 'state_range':self.m.state_range,
#                            'state_gen':self.m.state_sites,
#                            'prob_trans': {'metropolis': self.m.energy,
#                                          'wolff':self.m.model_params[
#                                          self.m.model.__name__]['bond_prob']}

        self.model_props = {'L': L, 'd': d, 'q': m.q, 'T': T,
                            'state_range': m.state_range,
                            'state_gen': m.state_gen,
                            'prob_trans': {'metropolis': m.energy,
                                           'wolff':m.model_params[
                                                   m.model.__name__][
                                                   'bond_prob']},
                            'model': m.model.__name__,
                            'algorithm': '',
                            'observables': m.observables_functions,
                            'observables_props': m.observables_props,
                            'data_dir': '%s_Data'%(caps(m.model.__name__)),
                            'data_file': '/%s_d%d_L%d__%s' %(
                                          caps(m.model.__name__),d,L,
                                          datetime.datetime.now().strftime(
                                                           '%Y-%m-%d-%H-%M'))}
        
        
        # Perform Monte Carlo Updates for various Temperatures
        # Define Update, Animate Properties
        
        # Initialize Monte Carlo Updating Class
        self.MC = MonteCarloUpdate(sites = m.state_gen(l.Nspins),
                                   neighbour_sites = l.neighbour_sites,
                                   model_props = self.model_props,
                                   update = update)
        
        
        
        # Save Observables data
        self.observables = []
        self.observe = observe
        self.save = datasave        
        self.data_path = lambda a='algorithm',s='',b='': [
                                         self.MC.model_props.get(p,b) for p in
                                         ['data_dir','data_file',a]] + [s]




        # Initialize Model and Lattice variables
        self.m = m
        self.l = l
        
        return




    # Main Monte Carlo Algorithm
    def MonteCarlo(self,algorithm = ['wolff','metropolis'], n_iter=1):
 
        
        algorithm = np.atleast_1d(algorithm)
        
        n_alg = np.size(algorithm)
        if n_iter == 1 and n_alg > 1:
            n_iter = n_alg
            
            
        # Initialize Plotting
            
        self.data_keys {'sites': [[(t,p) for p in self.observe[0][1:]] 
                                         for t in self.T],
                         'obs' : [[(a,p) for p in self.observe[1][1:]]
                                         for a in algorithm]} 
        
        self.plot_obs = Data_Plot(self.keys_obs,plot=self.observe[1][0])
  



        #print(self.model_props['data_file'][1:]])
        # Perform Monte Carlo Algorithm for n_iter configurations
        for i in range(n_iter):
            
            
            # Initialize sites with random spin at each site each Iteration
            self.MC.sites =self.MC.model_props['state_gen'](self.MC.Nspins)
        

        
            # Perform Monte Carlo
            self.MC.MCAlg(algorithm[i%n_alg],self.data_keys)     
        
        
            # Print Performance for each Iteration
            display(True,True,'%s runtime: '%(algorithm[i%n_alg]))        
            
            # Save Observables Data per Iteration
            self.observables.append(self.MC.observables)
        
            # Plot Observables Data
                    self.plot_obs = Data_Plot(self.MC.keys_obs,plot=self.observe[1][0])

            self.plot_obs.plotter(data = self.observables,
                  plot_props = self.MCPlot_obs(keys_obs))
        
            # Save Observables Data     
            self.data_save(self.save,self.MC.observables,self.data_path())
            
        
        return
    
    
    
   
    def data_save(self,save=True,data=None,data_path=None,nfiles=1):
    
        save = np.atleast_1d(save)
        
        # Write Data to File Directory
        if data_path is None:
            data_dir = os.getcwd()
            data_file = data_dir+'/DataSet'
        else:
            data_dir = data_path[0]
            data_file = ''.join(data_path)
        
        if save[0]:
            # Open and write observables to a file in a specific directory
            if not data :
                if not(os.path.isdir(data_dir)):
                        os.mkdir(data_dir)
                return
            
            for n in range(nfiles):
                
                if save[-1]:
                    np.savez_compressed(
                            data_file+('_%d'%n if nfiles>1 else ''),a=data[n])      
        return
    
    
    
    
    
    
    

if __name__ == "__main__":
    T = [5,2.5,2,1.5,1,0.5]
    T0 = 0.5
    s = system(T=T0)
    s.MonteCarlo(algorithm=['wolff','metropolis'],n_iter=1)