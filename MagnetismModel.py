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
from Model_Analysis import Model_Analysis
from data_functions import Data_Process
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
    
    
	def __init__(self):
		return

	def process(self,data_props):
		self.process = Model_Analysis(data_props)		
		return
		
	def update(self,
				model_props =   {'model_name':'potts','q':2, 'd':2, 'L': 15, 
							     'T': 1.13,'coupling_param':[0,1],
								 'data_type':np.int_},
                update_props =  {'Nsweep': 0.95,'Neqb':5, 'Nmeas':5, 
								 'Nmeas_f':1, 'Ncluster':1},
                observe_props = {'configurations':   [False,'sites','cluster'],
                                 'observables':      [True,'energy', 'order',
													  'specific_heat',
													  'susceptibility'],
                                 'observables_mean': [True]},
				data_props = {'data_files': '*.npy',
							'data_types': ['sites','observables','model_props'],
							'data_typed': 'dict_split',
							'data_format': 'npy',
							'data_dir': 'dataset/',
							},
				iter_props={'algorithm':['wolff','metropolis']}):
		
		# Initialize model class, lattice class
		
		m = Model(model=model_props, 
		          observe=observe_props['observables_mean'][1:])
		l = Lattice(L=model_props['L'], d=model_props['d'])

		# Initialize System Attributes
		self.m = m
		self.l = l

		# Update model_props
		model_props.update({
							'N_sites': np.shape(l.neighbour_sites)[1],
							'neighbour_sites': l.neighbour_sites,
							'state_range': m.state_range,
							'state_gen': m.state_gen,
							'state_int': m.model_params['int'],
							'state_update': m.model_params['state_update'],
							'update_props': update_props,
							'observe_props': observe_props,
							'observables': m.observables_functions,
						   })
		
		# Perform Monte Carlo Updates for various Temperatures
		MonteCarloUpdate(model_props).MC_update(iter_props)
		
		return
    
	
    
# Run System for Temperatures and Iteration Configurations with arg_parser

# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")

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
					nargs = '+',type=float,default=1.1)

parser.add_argument('-alg','--algorithm',help='Update Algorithm',
					default='wolff',type=str,
					choices=['wolff','metropolis','metropolis_wolff'])
					
parser.add_argument('-Ne','--Neqb',help = 'Equilibrium Sweeps',
					type=int,default=10)
					
parser.add_argument('-Nm','--Nmeas',help = 'Measurement Sweeps',
					type=int,default=10)
					
parser.add_argument('-Nf','--Nmeas_f',help = 'Measurement Sweeps Frequency',
					type=int,default=1)
					
parser.add_argument('-Nr','--Nratio',help = 'Measurement Sweeps Ratio',
					type=float,default=1)
					
parser.add_argument('-u','--update',help = 'Perform Monte Carlo Updates',
					action='store_false')
					
parser.add_argument('-an','--analysis',help = 'Perform Analysis',
					action='store_true')
					
parser.add_argument('-j','--job',help = 'Job Number',
					type=int,default=0)
					
parser.add_argument('-v','--version',help = 'Version: Python or Cython',
						type=str,choices=['py','cy'],default='py')
						
_, unparsed = parser.parse_known_args()

# Unknown Arguments
for arg in unparsed:
    if arg.startswith(("-", "--")):
        parser.add_argument(arg, type=str,help = arg+' Argument')
						
# Parse Args Command
args = parser.parse_args()

if __name__ == "__main__":
    
    # System Parameters

	# Check for Version
	if args.version == 'py':
		from MonteCarloUpdate_python import MonteCarloUpdate
	elif args.version == 'cy':
		from MonteCarloUpdate_cython import MonteCarloUpdate

	# Update, Observe, Process, Simulate Parameters
	model_props = {'coupling_param':[0,1],'data_type':np.int_,
				   'algorithm':'wolff',
				   'algorithms':['wolff','metropolis','metropolis_wolff'],
				   'disp_updates':True, 'data_save':True, 'return_data':False,
				   'data_date':datetime.datetime.now().strftime(
										    			   '%Y-%m-%d-%H-%M-%S')}
	
	update_props = {'Neqb':args.Neqb, 'Nmeas':args.Nmeas, 'Nratio': 0.9,
					'Nmeas_f':args.Nmeas_f, 'Ncluster':1}
	
	observe_props = {'configurations':   [False,'sites','cluster'],
			         'observables':      [True,'energy','order'],
                     'observables_mean': [True,'energy','order','specific_heat',
									                          'susceptibility']}
	data_props = {'data_types': ['sites','observables','model_props'],
				  'data_formats': {'sites':np.ndarray,'observables':np.ndarray,
								  'model_props':dict},
				  'data_format': 'npy',
				  'data_typed': 'dict_split',
				  'data_dir': 'dataset/',
				  'data_file_format':['model_name','L','d','q','T','data_date']
				 }
	data_props['data_files'] = '*.' + data_props['data_format'] 
		
	iter_props = {'algorithm':['wolff','metropolis']}
	
	if any(k in vars(args) for k in iter_props.keys()):
		for k in iter_props.keys():
			iter_props[k] = np.atleast_1d(getattr(args,k,iter_props[k]))[0]

	# Delete non keyword arg attributes
	for k in ['version','Neqb','Nmeas','Nmeas_f']:
		delattr(args,k)
	model_props.update(vars(args))
	model_props['T'] = np.atleast_1d(model_props['T']).astype(np.float32)
	model_props.update(data_props)
	Data_Process().format(model_props)
	
	
	display(time_it=False,
			m='Job: '+str(model_props['job'])+'\n'+
						  model_props['data_file']+'\n')
	
	# Define System
	s = system()

	# Perform Monte Carlo
	if args.update:
		s.update(model_props,update_props,observe_props,data_props,iter_props)
	
	# Analyse Results
	# if args.analysis:
		# s.process(data_props)
		# s.process.process() 
	
	
	

	

		
	