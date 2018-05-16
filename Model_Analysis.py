"""
Created on Sun May 13 6:55:25 2018
@author: Matt
"""
import numpy as np

from MonteCarloPlot import MonteCarloPlot
from data_functions import Data_Process
from misc_functions import display


class Model_Analysis(object):
	def __init__(self,data_props = 
					{'data_files': '*.npz',
					 'data_types': ['sites','observables','model_props'],
					 'data_typed': 'dict_split',
					 'data_format': 'npz',
					 'data_dir': 'dataset/',
					}):
		
		display(print_it=True,time_it=False,m='Model Analysis...')
		
		# Import Data
		_,_,self.data,_ = Data_Process().importer(data_props)
		self.data_props = data_props
				
		return
				
	
	def process(self):
	
		# Process Each Data Set
		for k,sites in self.data['sites'].items():
			
			# Check if Data exists
			if self.data['observables'].get(k) is not None:
				continue

			model_props = self.data['model_props'][k]
			model_props.update(self.data_props)
			# Measure Data
			observables = self.measure(sites, model_props['neighbour_sites'],
									model_props['T'],model_props['observables'])
			self.data['observables'][k] = observables
	
			# Save Data
			if model_props.get('data_save',True):
				Data_Process().exporter({'observables':observables},model_props)
			
			# Plot Data
			self.plot(observables,model_props,'iter_props','algorithm')
		
		display(print_it=True,time_it=False,m='Observables Processed')
	
	def measure(self,sites,neighbours,T,observables):

		if sites.ndim < 4:
			sites = sites[np.newaxis,:]

		n_iter = np.shape(sites)[0]

		data = [{} for _ in range(n_iter)]

		for i_iter in range(n_iter):        
			for k,obs in observables.items():
				data[i_iter][k] = obs(sites[i_iter],neighbours,T)

		return data
		
	
	def plot(self,data,model_props,parameters,parameter=None):
	
		# Plot Instance
		plot_obj = MonteCarloPlot(model_props['observe_props'],
							      model_props, model_props['T'])
		plot_args = [model_props['T'],[]]
		for p in model_props[parameters]:
			if isinstance(p,dict):
				plot_args[1].append(p.get(parameter,parameter))
			else:
				plot_args[1].append(p)
	
		# Plot Data
		plot_obj.MC_plotter({'observables':      data,
						     'observables_mean': data},
						     *plot_args)
		
		# Save Figures
		if model_props.get('data_save',True):
			plot_obj.plot_save(model_props,read_write='w')
			
		return
		
		
		
	def sort(self,params):
	
		# Sort by params as {param0_i: {param1_j: [param2_k values]}}
		# i.e) q, L, T
		
		self.data['parameters'] = {}
		
		for p in params:
			for file in self.data['model_props']:
				pass
		
		return
		
		
		
		
		