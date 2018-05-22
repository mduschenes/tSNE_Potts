"""
Created on Sun May 13 6:55:25 2018
@author: Matt
"""
import numpy as np
import copy

from MonteCarloPlot import MonteCarloPlot
from data_functions import Data_Process
from misc_functions import display


class ModelAnalysis(object):
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
		
		self.data_props_temp = data_props.copy()
		
		return
		
	
	def process(self,data=None):
	
		if data is None:
			data = self.data
	
		# Process Each Data Set
		display(print_it=True,time_it=False,m='Processing Observables')
		for k,sites in data['sites'].items():
			
			# Update data properties
			data['model_props'][k].update(self.data_props)
						
			# Check if Data exists
			if data['observables'].get(k) is not None:
				
				# Plot Data
				if self.data_props.get('plot') and False:
					self.plot(data['observables'][k],data['model_props'][k],
							 {'arr_0': ['T',model_props['T']],
							 'arr_1':['algorithm',np.atleast_1d(
										data['model_props'][k][
										'iter_props'][0]['algorithm'])]})
				continue

			model_props = data['model_props'][k]
			
			
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
							   format=model_props['data_format']['observables'])
						
			# Plot Data
			if self.data_props.get('plot'):
				self.plot({'observables':observables,
						   'observables_mean': observables},
						   model_props,
							{'arr_0': ['T',model_props['T']],
							 'arr_1':['algorithm', np.atleast_1d(model_props[
											   'iter_props'][0]['algorithm'])]})
				
			data['observables'][k] = observables
		
		
		# Sort Data
		if self.data_props.get('sort'):
			display(m='Sorting Data',time_it=False)
			data = self.sort(data,self.data_props['sort_parameters'])
		
		
		# Plot Sorted Observables
		p = self.data_props['sort_parameters']

		if self.data_props.get('plot') and self.data_props.get('sort'):
			
			display(m='Plotting Sorted Data',time_it=False)
			for props0 in data['model_sorted'].keys():
				
				
				self.plot({'sorted': data['obs_sorted'][p]},data['model_sorted'][p],
						{'arr_0':p[3],'arr_1': [2],
						 'sup_title':'%s = %s'%(p[0],str(props0))})
		
		
		display(print_it=True,time_it=False,m='Model Analysis Complete...')
		return
		
	def measure(self,sites,neighbours,T,observables):
	
		if sites.ndim < 4:
			sites = sites[np.newaxis,:]

		n_iter = np.shape(sites)[0]

		data = [{} for _ in range(n_iter)]

		for i_iter in range(n_iter):        
			for k,obs in observables.items():
				data[i_iter][k] = obs(sites[i_iter],neighbours,T)

		return data
		
	
	def plot(self,data,model_props,plot_args,new_plot=True):
	
		# Check if Plot already exists:
		if not new_plot:
			plot = Data_Process().importer(self.data_props,
										data_files = model_props.get(
														'data_file','plot'),
										data_format = 'pdf',
										data_obj_format = None)
			if plot is not None:
				return
							  
		plot_obj = MonteCarloPlot(model_props['observe_props'],
							      model_props, **plot_args)
								  
		# Plot Data
		plot_obj.MC_plotter(data,**plot_args)
		
		# Save Figures
		if model_props.get('data_save',True):
			plot_obj.plot_save(model_props,read_write='w')
			
		return
		

		
	def sort_coord(self,data,parameters):
	
		# # Sort by params as {param0_i: {param1_j: [param2_k values]}}
		# # i.e) q, L, T
		
		def dim_reduce(arr,axis=0):
			try:
				return np.squeeze(arr,axis)
			except:
				return arr
			
		def return_item(d,item=0,index=0):
			return list(d.items())[index][item]

		def nested_tree(branches,root={}):
			if max([len(b) for b in branches]) > 0:
				tree = {}
				for b0 in set([b[0] for b in branches]):
					tree[b0] = nested_tree([b[1:] for b in branches if b[0] == b0],root)
			else:
				if isinstance(root,np.ndarray):
					tree = np.copy(root)
				else:
					tree = root.copy()
			return tree

		def ind_tree(tree,branch,func,*args):
			for b in branch[:-1]:
				tree = tree[b]
			tree[branch[-1]] = func(tree,branch,*args)
			
		def tree(branches,root,properties,index=None,*args):
			branches_i = [b[:index] for b in branches]
			tree = nested_tree(branches_i,root)
			for b,p in zip(branches_i,properties):
				ind_tree(tree,b,p,*args)
			return tree
			
			
		 # Check if sorted data exists
		data_props = self.data_props.copy()
		data_name = 'sorted_data'
		data_sorted_format = 'npy'
		data_obj_format = 'dict'
		data_props['data_files'] = data_name
		data_props['data_format'] = data_sorted_format
		data_temp = Data_Process().importer(data_props)
		#print(data_temp)
		if data_temp is not None:
			print('data_temp',type(data_temp[0][data_name].item()))
			exit()
			return data_temp[0]
		
		# Define Parameter Branches
		root = []
		branches = [None for _ in range(len(data['model_props']))]

		# Define Parameter Functions to be inserted at roots
		
		# 'data_properties':['model_name','d','algorithm','observe_props',
								   # 'data_files','data_types','data_format',
								   # 'data_obj_format','data_name_format'],
		
		def model_props_sorted(tree,branch,properties):
			
			# Update data properties
			props = properties.copy()
			props['data_file_format'] = [props['data_file_format'][0]]+[parameters[0]]+['']
			
			for p in self.data_props['data_properties']:
				props[p] = copy.deepcopy(self.data_props.get(p,None))
				
			
				
			
			
			
			data['model_sorted'][props0]['data_file_format'] = (
				   [data['model_sorted'][props0]['data_file_format'][0]]+p+[''])
			Data_Process().format(data['model_sorted'][props0],
									  file_update=True)
		
		for i,m in enumerate(data['model_props'].values()):
			branches[i] = [m[p] for p in parameters]
		
		for s in ['sites','observables','model_props']:
		
			data[s+'_sorted'] = tree(parameter_sets,root,list(data[s].items()))
		data['obs_sorted'] = tree(parameter_sets,root,list(data['observables'].items()))
		data['model_sorted'] = tree(parameter_sets,root,list(data['model_props'].items()))

		# Save Sorted Data
		Data_Process().exporter({data_name:data},self.data_props,
									 format=data_sorted_format)
			
		return data
		
		
		