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
    #                                 Number of Clusters Ncluster
    # 
    # Observe: Booleans for Plot Types (Sites, Clusters, Edges, Observables)
    # DataSave: Boolean

    
    # Define system parameters of:
    # Size, Dimension, q states, Temperature
    # State Range, State Generator, Transition Probability
    # Model Name, Monte Carlo Algorithm, Observables
    # Data File and Directory        
    # Initialize model class, lattice class
    
    
    def __init__(self,L=10,d=2,T=3, model=['ising',1,[0,1]],
                update = [True,10,10,1,1],
                observe = {'configurations': [False,'sites','cluster'],
                           'observables': [True,'energy','order']
                           },
                datasave = True):


        m = Model(model,d,observe['observables'][1:])
        l = Lattice(L,d)

        self.model_props = {'L': L, 'd': d, 'q': m.q, 'T': T,
                            'state_range': m.state_range,
                            'state_gen': m.state_gen,
                            'prob_trans': {'metropolis': m.energy,
                                           'wolff':m.model_params[
                                                   m.model.__name__][
                                                   'bond_prob']},
                            'model': m.model.__name__,
                            'algorithm': '',
                            'algorithms': ['metropolis','wolff'],
                            'observables': m.observables_functions,
                            'observables_props': m.observables_props,
                            'data_dir': '%s_Data/'%(caps(m.model.__name__)),
                            'data_file': '%s_d%d_L%d__%s' %(
                                          caps(m.model.__name__),d,L,
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
    
    
    
if __name__ == "__main__":
    T = [3.5,2.5,2.2,1.2]
    T0 = 0.5
    s = system(T=T)
    #s.MonteCarlo.MCAlg(algorithm=['wolff','metropolis'],n_iter=1)