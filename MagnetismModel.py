# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:56:21 2017
@author: Matt
"""
import datetime

from Lattice import Lattice
from Model import Model
from MonteCarloUpdate import MonteCarloUpdate
from ModelFunctions import caps



        
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
    #                                 Number of Clusters Ncluster (Not used)
    # 
    # Observe: Booleans for Plot Types (Sites, Clusters, Edges, Observables)
    # DataSave: Boolean

    
    # Define system parameters of:
    # Size, Dimension, q states, Temperature
    # State Range, State Generator, State Interactions, Transition Probability
    # Model Name, Monte Carlo Algorithm 
    # Observables functions, Observables properties 
    # Data File and Directory        
    
    
    def __init__(self,L=15,d=2,T=3, model=['potts',2,[0,1]],
                update = [True,10,10,1,1],
                observe = {'configurations': [False,'sites','cluster'],
                           'observables': [True,'temperature','energy',
                                                'order','specific_heat',
                                                'susceptibility'],
                           'observables_mean': [True]
                           },
                datasave = True):


        # Initialize model class, lattice class
        m = Model(model,d,T,observe['observables'][1:])
        l = Lattice(L,d)

        self.model_props = {'L': L, 'd': d, 'q': m.q, 'T': T,
                            'state_range': m.state_range,
                            'state_gen': m.state_gen,
                            'state_int': m.model_params['int'],
                            'prob_update': m.model_params['prob_update'],
                            'model': m.model_name,
                            'algorithm': 'wolff',
                            'algorithms': ['metropolis','wolff'],
                            'observables': m.observables_functions,
                            'observables_props': m.observables_props,
                            'data_dir': '%s_Data/'%(caps(m.model_name)),
                            'data_file': '%s_d%d_L%d__%s' %(
                                          caps(m.model_name),d,L,
                                          datetime.datetime.now().strftime(
                                                           '%Y-%m-%d-%H-%M'))}
        
        
        # Perform Monte Carlo Updates for various Temperatures
        
        self.MonteCarlo = MonteCarloUpdate(sites = m.state_gen(l.Nspins),
                                           neighbour_sites = l.neighbour_sites,
                                           model_props = self.model_props,
                                           update = update,
                                           observe = observe)
        # Initialize Model and Lattice variables
        self.m = m
        self.l = l
        
        return
    
    
# Run System for Temperatures and Iteration Configurations  
if __name__ == "__main__":
    L=15
    d=2
    T = [3.0,2.5,1.75,1.2,0.8,0.5,0.2]
    T0 = 0.25
    model=['potts',2,[0,1]]
    update = [True,10,10,1,1]
    observe = {'configurations': [False,'sites','cluster'],
                           'observables': [True,'temperature','energy',
                                                'order','specific_heat',
                                                'susceptibility'],
                           'observables_mean': [True]
                           }
    datasave = True
    
    
    props_iter = {'algorithm':['wolff','metropolis']}
    
    s = system(L,d,T,model,update,observe,datasave)
    s.MonteCarlo.MCUpdate(props_iter)