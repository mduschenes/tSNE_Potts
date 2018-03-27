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
    # 
    # DataSave: Boolean
    # Animate: True
    
    def __init__(self,L=10,d=2,T=3, model=['potts',4,[0,1]],
                 update = [True,None,None,1],
                 datasave = True,
                 animate = [False,True,True]):
        #print(L,d,T)
        # Initialize model class, lattice class
        self.m = Model(model,d)
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

        self.modelprops = {'L':self.L,'d':self.d,'q':self.q,
                           'model':self.m.model.__name__,'algorithm':'',
                           'data_dir': '%s_Data'%(caps(self.m.model.__name__)),
                           'data_file': '/%s_d%d_L%d__%s' %(
                                          caps(self.m.model.__name__),
                                          self.d,self.L,
                                          datetime.datetime.now().strftime(
                                                           '%Y-%m-%d-%H-%M')) }
        
        # Initialize numpy array with random spin at each site
        self.sites = self.m.state_sites(self.Nspins)

#        # Initialize nearest neighbour sites array
#        self.nn = self.l.neighbour_sites[0]
        


        
        # Save Observables data
        self.observables = []
        self.save = datasave        




        
        # Perform Monte Carlo Updates for various Temperatures
        # Update, Transition, Animate Properties
        animate.append(self.m.state_range)
        prob_trans = {'metropolis': self.m.energy,'wolff':self.m.model_params[
                                self.m.model.__name__]['bond_prob']}
        transition = [prob_trans,self.m.state_sites]
        
        
        # Initialize Monte Carlo Updating
        self.MC = MonteCarloUpdate(self.sites,self.l.neighbour_sites,
                         self.m.observables_data,self.T,self.modelprops,
                         update,transition,animate)
        


    def MonteCarlo(self,algorithm = ['metropolis','wolff'],n_iter=1):
 
        t0 = time.clock()
        
        algorithm = np.atleast_1d(algorithm)
        n_alg = np.size(algorithm)
        
        if n_iter == 1 and n_alg > 1:
            n_iter = n_alg
        
        for i in range(n_iter):
            
            self.modelprops['algorithm']= algorithm[i%n_alg]
            self.MC.MCalg(algorithm[i%n_alg])     
    
            # Save Observables Data 
        
        self.observables = self.MC.observables
        
        self.data_save(self.save,self.observables,
                       self.m.observables_prop(
                        self.sites,self.l.neighbour_sites,self.t,'__name__'),
                       self.m.observables_prop(
                            self.sites,self.l.neighbour_sites,self.t,'size'),
                       n_iter, 
                       [self.modelprops['data_dir'],
                            self.modelprops['data_file']],
                       lambda i: '%s runtime: %0.5f'%(
                                       algorithm[i%n_alg],time.clock()-t0))
    

    
   
    def data_save(self,save,data=None,headers='data',cols=1,nfiles=1,
                                      data_path=None,*comments):
    
        # Write Data to File Directory
        if data_path == None:
            data_dir = os.getcwd()
            data_file = data_dir+'/DataSet'
        else:
            data_dir = data_path[0]
            data_file = ''.join(data_path)
    
        # Data Structure of Data Headers, and Number of Collumns per Header
        headers = np.atleast_1d(headers)
        cols = np.atleast_1d(cols)
        
        
        if save == True:
            # Open and write observables to a file in a specific directory
            if not data :
                if not(os.path.isdir(data_dir)):
                        os.mkdir(data_dir)
                return
            
            
            for n in range(nfiles):
                # Make observables headers for file
                file = open(data_file+('_%d'%n if nfiles>1 else '')+'.txt','w')
                header = []
                
                # Write Observables to File
                
                
                for i,h in enumerate(headers):
                    for j in range(cols[i]):
                        header.append(h+'_'+str(j+1) 
                                        if cols[i] > 1
                                        else h)
                file.write('\t'.join(header) + '\n')
                
                # Convert lists of lists of observables to array
                for data_n in [list(flatten(x)) for x in data[n]]:
                    dataline = ''
                    for d in data_n:
                            dataline += '%0.8f \t' %(float(d))
                    dataline += '\n'
                    file.write(dataline)
                if comments:
                    for c in comments:
                        file.write(str(c(n))+'\n')
                
                file.close()
        return
    

if __name__ == "__main__":
    T = [5,2.5,2,1.5,1,0.5]
    T0 = 0.5
    s = system(T=T)
    s.MonteCarlo(n_iter=4)
