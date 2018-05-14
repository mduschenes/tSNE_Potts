"""
Created on Sun May 13 6:55:25 2018
@author: Matt
"""
import numpy as np

from MonteCarloPlot import MonteCarloPlot
from data_functions import Data_Process
from misc_functions import display


class Model_Analysis(object):
	def __init__(self,data_params = 
					{'data_files': '*.npz',
					 'data_types': ['sites','observables','model_props'],
					 'data_typed': 'dict_split',
					 'data_format': 'npz',
					 'data_dir': 'dataset/',
					}):
		
		display(print_it=True,time_it=False,m='Model Analysis...')
		
		# Import Data
		self.data = importer(data_params)
		
		# Process Each Data Set
		for k,sites in self.data['sites'].items():

			model_props = self.data['model_props'][k]
		
			# Measure Data
			observables = self.measure(
					sites,model_props['neighbour_sites'],model_props['T'])
			self.data['observables'][k] = observables
		
			# Save Data
			if model_props.get('data_save',True):
				Data_Process().exporter({'observables':observables},model_props)
			
			# Plot Data
			self.plot(observables,model_props)		
			
		display(print_it=True,time_it=False,m='Observables Processed')
		return
				
				
	def importer(self,data_params):
		_,_,data,_ = Data_Process().importer(data_params)
		return data		
	
	
	def measure(self,sites,neighbours,T,observables):

		if sites.ndim < 4:
			sites = sites[np.newaxis,:]

		n_iter = np.shape(sites)[0]

		data = [{} for _ in range(n_iter)]

		for i_iter in range(n_iter):        
			for k,obs in observables.items():
				data[i_iter][k] = obs(sites[i_iter],neighbours,T)

		return data
		
	
	def plot(self,data,model_props):
	
		# Plot Instance
		plot_obj = MonteCarloPlot(model_props['observe_props'],
							      model_props, model_props['T'])
	
	
		# Plot Data
		plot_obj.MC_plotter({'observables':      data,
						     'observables_mean': data},
					  *[model_props['T'],
					    [p.get('algorithm',model_props['algorithm'])
							for p in model_props['iter_props']]])
		
		# Save Data
		if model_props.get('data_save',True):
			plot_obj.plot_save(model_props)