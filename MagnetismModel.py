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
from MonteCarloPlot import MonteCarloPlot
from misc_functions import caps,display



        
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
    
    
	def __init__(self,
				model_props =   {'model_name':'potts','q':2, 'd':2, 'L': 15, 
							     'T': 1.13,'coupling_param':[0,1],
								 'data_type':np.int_},
                update_props =  {'update_bool':True,'Neqb':5, 'Nmeas':5, 
								 'Nmeas_f':1, 'Ncluster':1},
                observe_props = {'configurations':   [False,'sites','cluster'],
                                 'observables':      [True,'energy', 'order',
													  'specific_heat',
													  'susceptibility'],
                                 'observables_mean': [True]},
				process_props = {'disp_updates': 1, 'data_save': 1}):

		# Initialize model class, lattice class
		m = Model(model=model_props, observe=observe_props['observables'][1:])
		l = Lattice(L=model_props['L'], d=model_props['d'])

		# Initialize System Attributes
		self.m = m
		self.l = l

		# Update model_props
		model_props.update({
							'neighbour_sites': l.neighbour_sites,
							'state_range': m.state_range,
							'state_gen': m.state_gen,
							'state_int': m.model_params['int'],
							'state_update': m.model_params['state_update'],
							'algorithm': 'wolff',
							'algorithms': ['metropolis','wolff'],
							'update_props': update_props,
							'observe_props': observe_props,
							'observables': m.observables_functions,
							'observables_props': m.observables_props,
							'data_dir': '%s_Data/'%(caps(m.model_name)),
							'data_file': '%s_d%d_L%d__%s' %(
										  caps(m.model_name),model_props['d'], 
										  model_props['L'],
										  datetime.datetime.now().strftime(
														   '%Y-%m-%d-%H-%M'))
						   })
		model_props.update(process_props)
		
		
		# Perform Monte Carlo Updates for various Temperatures
		self.MonteCarlo = MonteCarloUpdate(model_props = model_props)
		self.model_props = model_props
		return
    
    
# Run System for Temperatures and Iteration Configurations with arg_parser

# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")
# group = parser.add_mutually_exclusive_group()
# group.add_argument("-v","--verbose",help = "verbosity measure",
                     # action='count')# choices=[0,1,2])#action="store_true")
# Add Args
parser.add_argument('-L','--L',help = 'System Length Scale',
					type=int,default=15)# choices=[0,1,2])#action="store_true")

parser.add_argument('-d','--d',help = 'System Dimension',
					type=int,default=2)

parser.add_argument('-q','--q',help = 'Model spin range',
					type=int,default=2)

parser.add_argument('-m','--model_name',help = 'Model Name',
					type=str,default='potts')

parser.add_argument('-T','--T',help = 'System Temperature',
					nargs = '+',type=float)

parser.add_argument('-v','--version',help = 'Version: Python or Cython',
						type=str,choices=['py','cy'],default='py')

# Parse Args Command
args = parser.parse_args()

if __name__ == "__main__":
    
    # System Parameters
	# L=15
	# d=2
	# q = 2
	# T = [3.0,2.5,1.75,1.2,1.0,0.8,0.5]
	# m = 'potts'
	model_props={'coupling_param':[0,1],'data_type':np.int_}
	
	# Check for Version
	if args.version == 'py':
		from MonteCarloUpdate_python import MonteCarloUpdate
	elif args.version == 'cy':
		from MonteCarloUpdate_cython import MonteCarloUpdate
	del args.version
	
	model_props.update(vars(args))

	update_props = {'update_bool':True,'Neqb':5, 'Nmeas':5, 
					                'Nmeas_f':1, 'Ncluster':1}

	observe_props = {'configurations': [False,'sites','cluster'],
			         'observables': [True,'energy','order','specific_heat',
									                          'susceptibility'],
                     'observables_mean': [True]}

	process_props = {'disp_updates': True, 'data_save': True}

	s = system(model_props=model_props,     update_props=update_props,
			   observe_props=observe_props, process_props=process_props)

	
	# Monte Carlo Simulation Parameters
	iter_props = {'algorithm':['wolff','metropolis']}	
	s.data,s.model_props = s.MonteCarlo.MC_update(iter_props) 
		
	
	
	
	# Plot Observables
	plot_obj = MonteCarloPlot(s.model_props['observe_props'],
							  s.model_props, s.model_props['T'])
	
	
	
	plot_obj.MC_plotter({'observables': s.data['observables'],
						 'observables_mean': s.data['observables']},
					  *[s.model_props['T'],
					    [p['algorithm'] for p in s.model_props['iter_props']]])

	display(print_it=True,time_it=False,m='Observables Figures Plotted')

		
	if s.model_props['data_save']:
		plot_obj.plot_save(s.model_props)