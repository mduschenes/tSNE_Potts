"""
Created on Sun May 13 6:55:25 2018
@author: Matt
"""
import numpy as np

from MonteCarloPlot import MonteCarloPlot
from data_functions import Data_Process
from misc_functions import display, nested_dict


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
		_,sizes,self.data,_ = Data_Process().importer(data_props,upconvert=False)
		self.data_props = data_props
		return
				
	
	def process(self,data=None):
	
		if data is None:
			data = self.data
	
		# Process Each Data Set
		for k,sites in data['sites'].items():
			# Check if Data exists
			if data['observables'].get(k) is not None:
				# Plot Data
				if self.data_props.get('plot'):
					self.plot(data['observables'][k],data['model_props'][k],'iter_props','algorithm')
				continue

			model_props = data['model_props'][k]
			model_props.update(self.data_props)
			
			
			# # Update model_props files
			# for var_type in model_props['data_types']:
				# if 'model_props' in var_type: break
			# for f in ['txt',model_props.get('props_format','npy')]:
				# Data_Process().exporter({var_type: model_props},
							            # model_props,format=f,read_write='ow')
			
			# Measure Data
			observables = self.measure(sites, model_props['neighbour_sites'],
									model_props['T'],model_props['observables'])
									
			
			
			# Save Data
			if model_props.get('data_save',True):
				Data_Process().exporter({'observables':observables},model_props,
									     format=model_props['data_format'])
			
			# Plot Data
			if self.data_props.get('plot'):
				self.plot(observables,model_props,'iter_props','algorithm')
				
			data['observables'][k] = observables
		# Sort Data
		if self.data_props.get('sort'):
			self.sort(data,self.data_props['sort_parameters'])
		
		display(print_it=True,time_it=False,m='Observables Processed')
		return
		
	def measure(self,sites,neighbours,T,observables):
	
		print(np.shape(sites))
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
		
		
		
	def sort(self,data,parameters):
	
		# Sort by params as {param0_i: {param1_j: [param2_k values]}}
		# i.e) q, L, T
		
		
		# for p in model_props[parameters]:
			# if isinstance(p,dict):
				# plot_args[1].append(p.get(parameter,parameter))
			# else:
				# plot_args[1].append(p)
		
		# self.data['parameters'] = {}
		
		data_sorted = data.copy()
		
		
		data_sorted['parameters'] = {}
		for p in parameters:
			print(data['model_props'])
			data['parameters'][p] = np.array(sorted(set(
						  [m[p] for m in data['model_props'].values()])))
						  
		
		data['configurations'] = nested_dict(list(data['parameters'].values()))
		print(data['configurations'])
		
		
		# for p in params[0]:
			# for k,sites in data['sites']:
				# if data['model_props'][k][p] 
		
			# data['configurations'][p0] = 1
		
		# def data_sort(data,params):
			# d = d
		
		
		
				
		
		# data_parameterized = {}
		# for t in data.keys():
			# data_parameterized[t] = sorted()
		
		# data_parameterized = sorted()
		
		
		# for p in parameters:
			# for k in data['sites'].keys():
				
		
		return
		
		
		
		
		