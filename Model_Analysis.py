"""
Created on Sun May 13 6:55:25 2018
@author: Matt
"""
import numpy as np
from data_functions import Data_Process
from misc_functions import flatten,array_dict, caps, display


class Model_Analysis(object):
	def __init__(self,data_params = 
					{'data_files': '*.npz',
					 'data_types': ['sites','observables','model_props'],
					 'data_typed': 'dict',
					 'data_format': 'npz',
					 'data_dir': 'dataset/',
					}):
				
				
	def importer(self,data_params):
		_,data_size,data_typed,data_sets=Data_Process().importer(data_params)
		
		
	
	
	def measure(self,sites,neighbours,T,observables):

		if sites.ndim < 4:
			sites = sites[np.newaxis,:]

		n_iter = np.shape(sites)[0]

		data = [{} for _ in range(n_iter)]

		for i_iter in range(n_iter):        
			for k,obs in observables.items():
				data[i_iter][k] = obs(sites[i_iter],neighbours,T)

		return data