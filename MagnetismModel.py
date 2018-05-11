# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:56:21 2017
@author: Matt
"""
import datetime
import numpy as np
import argparse


from Lattice import Lattice
from Model import Model
from MonteCarloUpdate import MonteCarloUpdate
from misc_functions import caps



        
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
    
    
    def __init__(self,L=15,d=2,T=3,q=2,m='potts',
				model=[[0,1],np.int_],
                update = [True,10,10,1,1],
                observe = {'configurations': [False,'sites','cluster'],
                           'observables': [True,'energy',
                                                'order','specific_heat',
                                                'susceptibility'],
                           'observables_mean': [True]
                           },
                datasave = True):


        # Initialize model class, lattice class
        m = Model(model = {'model_name':m,'q':q, 'd':d, 'T': T,
						   'order_param':model[0],'data_type':model[1]},
				 observe = observe['observables'][1:])
        l = Lattice(L,d)

        self.model_props = {'L': L, 'd': d, 'q': m.q, 'T': T,
							'neighbour_sites': l.neighbour_sites,
							'model': m.model_name,
                            'state_range': m.state_range,
                            'state_gen': m.state_gen,
                            'state_int': m.model_params['int'],
                            'prob_update': m.model_params['prob_update'],
                            'algorithm': 'wolff',
                            'algorithms': ['metropolis','wolff'],
							'update': update,
							'observe': observe,
                            'observables': m.observables_functions,
                            'observables_props': m.observables_props,
							'data_type':model[1],
							'data_save':datasave,
                            'data_dir': '%s_Data/'%(caps(m.model_name)),
                            'data_file': '%s_d%d_L%d__%s' %(
                                          caps(m.model_name),d,L,
                                          datetime.datetime.now().strftime(
                                                           '%Y-%m-%d-%H-%M'))}
        
        
        # Perform Monte Carlo Updates for various Temperatures
        
        self.MonteCarlo = MonteCarloUpdate(model_props = self.model_props)
        # Initialize Model and Lattice variables
        self.m = m
        self.l = l
        
        return
    
    
# Run System for Temperatures and Iteration Configurations with arg_parser

# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")
# group = parser.add_mutually_exclusive_group()

# Add Args
parser.add_argument('-L','--L',help = 'System Length Scale',
					type=int,default=15)# choices=[0,1,2])#action="store_true")

parser.add_argument('-d','--dimension',help = 'System Dimension',
					type=int,default=2)

parser.add_argument('-q','--q_parameter',help = 'Model spin range',
					type=int,default=2)

parser.add_argument('-m','--model',help = 'Model Type',
					type=str,default='potts')

parser.add_argument('-T','--temperature',help = 'System Temperature',
					nargs = '+')


# Parse Args Command
args = parser.parse_args()

if __name__ == "__main__":
    
    # System Parameters
	L=args.L
	d=2
	q = 2
	T = [3.0,2.5,1.75,1.2]
	Tlow = [0.5,0.25,0.15,0.1,0.05,0.02]
	T0 = 0.25
	m = 'potts'
	model=[[0,1],np.int_]
	update = [True,5,5,1,1]
	observe = {'configurations': [False,'sites','cluster'],
						   'observables': [False,'energy',
												'order','specific_heat',
												'susceptibility'],
						   'observables_mean': [False]
			  }
	datasave = True


	# Monte Carlo Simulation Parameters (with Update, Observe Parameters)
	props_iter = {'algorithm':['wolff','metropolis']}
	disp_updates = True

	s = system(L,d,T,q,m,model,update,observe,datasave)
	s.MonteCarlo.MC_update(props_iter,disp_updates=disp_updates) 