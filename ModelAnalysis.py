"""
Created on Sun May 13 6:55:25 2018
@author: Matt
"""
import numpy as np
import copy,os

from MonteCarloPlot import MonteCarloPlot
from data_functions import Data_Process
from misc_functions import display,hashable,dim_reduce


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
		data = Data_Process().importer(data_props,upconvert=False)
		
		
		self.data = data[2] if data is not None else data
		self.data_props = data_props
		
		return
		
	
	def process(self,data=None):
	
		if data is None:
			data = self.data
		
		if self.data is None:
			display(time_it=False,m='No Data to Process')
			return
		
		# Process Each Data Set
		display(print_it=True,time_it=False,m='Processing Observables')
		if not data.get('observables'):
			data['observables'] = {}
		for k,sites in data['sites'].items():
			
			# Update data properties
			data['model_props'][k].update(self.data_props)
						
			# Check if Data exists
			if data.get('observables',{}).get(k) is not None:
				
				# Plot Data
				if self.data_props.get('plot') and False:
					self.plot({'observables':data['observables'][k],
							   'observables_mean': data['observables'][k]},
							  data['model_props'][k],
							  {'arr_0': ['T',data['model_props'][k]['T']],
							  'arr_1':['algorithm',np.atleast_1d([p['algorithm'] for p in 
												data['model_props'][k][
											'iter_props']])]})
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
							 'arr_1':['algorithm', np.atleast_1d([p['algorithm'] for p in 
												data['model_props'][k][
											'iter_props']])]})
				
			data['observables'][k] = observables
		
		
		# Sort Data
		if self.data_props.get('sort'):
			display(m='Sorting Data',time_it=False)
			data = self.sort(data,self.data_props['sort_parameters'],
							 self.data_props)
		
		
		# Plot Sorted Observables
		p = self.data_props['sort_parameters']

		if self.data_props.get('plot') and self.data_props.get('sort'):
			
			display(m='Plotting Sorted Data',time_it=False)
			for p0 in data['model_props_sorted'].keys():
				self.plot({
					'observables_sorted': data['observables_sorted'][p0],
					'observables_mean_sorted': data['observables_sorted'][p0]},
					data['model_props_sorted'][p0],
					{'arr_0':[p[2],{a1: sorted(list(
												 data['observables_sorted'][p0][
													a1].keys())) for a1 in data[
											 'observables_sorted'][p0].keys()}] ,
					'arr_1': [p[1],sorted(list(data['observables_sorted'][p0
																].keys()))],
			        'sup_title':'%s = %s'%(p[0],str(p0))})
		
		
		# Reduce Dimensions of Data
		if self.data_props.get('plot') and self.data_props.get('sort') and (
						self.data_props.get('reduce')):
			display(m='Reducing Dimensions of Sorted Data',time_it=False)
			for p0 in data['model_props_sorted'].keys():
				for t in data['model_props_sorted'][p0]['data_types']:
					if t not in data.keys():
						data['model_props_sorted'][p0]['observe_props'][t] = [True,]
				self.plot({
					'observables_sorted': data['observables_sorted'][p0],
					'observables_mean_sorted': data['observables_sorted'][p0]},
					data['model_props_sorted'][p0],
					{'arr_0':[p[2],{a1: sorted(list(
												 data['observables_sorted'][p0][
													a1].keys())) for a1 in data[
											 'observables_sorted'][p0].keys()}] ,
					'arr_1': [p[1],sorted(list(data['observables_sorted'][p0
																].keys()))],
			        'sup_title':'%s = %s'%(p[0],str(p0))})
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
										format = 'pdf',
										data_obj_format = None)
			if plot is not None:
				return
							  
		plot_obj = MonteCarloPlot({k: model_props['observe_props'][k] 
										for k in data.keys()},
							      model_props, **plot_args)
								  
		# Plot Data
		plot_obj.MC_plotter(data,**plot_args)
		
		# Save Figures
		if model_props.get('data_save',True):
			plot_obj.plot_save(model_props,read_write='w')
			
		return
		

		
	def sort(self,data,parameters,data_props):
	
		# # Sort by params as {param0_i: {param1_j: [param2_k values]}}
		# # i.e) q, L, T
		
		def tree_sort(parameters,data_props,data,
					  branch_func,root=[],depth=None,
					  *args):
				
			# Create Branches
			branches = {k: b[:depth] 
						   for k,b in set_branch(data_props,parameters).items()}

			# Create Tree
			tree = get_tree(branches.values(),root)

			# Set Tree
			for k in branches.keys():
				set_tree(tree,branches[k],data[k],branch_func,*args)
			
			return tree, branches

		def get_branch(tree,branch,depth=-1):
			for b in branch[:depth-1]:
				tree = tree[b]
			return list(tree[branch[depth]].keys())

		def set_branch(data_props,parameters):
			branch = {}
			for k,d in data_props.items():
				branch[k] = [hashable(d[p]) for p in parameters]
			return branch


		def get_tree(branches,root=[]):
			if max([len(b) for b in branches]) > 0:
				tree = {}
				for b0 in set([b[0] for b in branches]):
					tree[b0]=get_tree([b[1:] for b in branches if b[0] == b0],
										root)
			else:
				if isinstance(root,np.ndarray):
					tree = np.copy(root)
				else:
					tree = root.copy()
			return tree

		def set_tree(tree, branch, data, branch_func= lambda t,b,d,*a:d, *args):
			for b in branch[:-1]:
				tree = tree[b]
			tree[branch[-1]] = branch_func(tree,branch,data,*args)
		
		def sites_sorted(tree,branch,data,root):
			if tree[branch[-1]] == root:
				return dim_reduce(data)
			else:
				return np.append(tree[branch[-1]],dim_reduce(data),axis=0)
			
		def observables_sorted(tree,branch,data,depth):
			if tree[branch[-1]] == []:
				return data[0]
			else:
				for k in data[0].keys():
					tree[branch[-1]][k] = np.append(np.atleast_1d(
														tree[branch[-1]][k]),
												    np.atleast_1d(data[0][k]),
													axis=0)
				return tree[branch[-1]] 

		def model_props_sorted(tree,branch,data,depth):
			
			if tree[branch[-1]] != {}:
				return tree[branch[-1]]
			
			# Update data properties
			props = {} #copy.deepcopy(data)
			
			for p in data_props['data_properties']:
				props[p] = copy.deepcopy(data_props.get(p,None))
			
			for i,p in enumerate(data_props['sort_parameters'][:depth]):
				props[p] = branch[i]
			
			props['data_name_format'] = [props['data_name_format'][0]] + (
											[props['sort_parameters'][0]]+[''])
			
			for s in props['observe_props'].copy().keys():
				props['observe_props'][s+'_sorted'] = props['observe_props'][s]
				
			Data_Process().format(props, file_update=True)
			
			return props
			

		
		
		# Sort by params as {param0_i: {param1_j: [param2_k values]}}
		# i.e) q, L, T


		# Check if sorted data exists
		data_props = copy.deepcopy(data_props)
		data_types = [s for s in data_props['data_types'] 
						if s not in ['sorted','tsne','pca']]
		file_header = data_props['data_dir'].split('/')[-2]
		file_format = lambda s: file_header + s + '_sorted'
		if data.get('sorted'):
			for k,v in (data['sorted'].copy()).items():
				data[k.split(file_header)[1]+ 'sorted'] = v.copy()
				data['sorted'].pop(k)
				
				# Save Sorted Data
				# Data_Process().exporter({file_format(k.split(file_header)[
				# 									1].replace('_',''):data[
										# k.split(file_header)[1]+ 'sorted']},
								     # data_props,read_write='ow')	
			data.pop('sorted')			
			return data

		root = {}
		depth = {}
		args = {'sites':root, 'observables': depth, 'model_props':depth}
		data_sorted = {}
		for s in data_types:
			
			root[s] = {} if s == 'model_props' else []
			depth[s] = 1 if s == 'model_props' else None
			args[s] = args[s][s]
			(data_sorted[s+'_sorted'],
			   data_props['branches']) = tree_sort(
													parameters, 
													data['model_props'],
													data[s], 
													locals().get(s+'_sorted'),
													root[s],depth[s],args[s]
												  )
			# Save Sorted Data
			Data_Process().exporter({file_format(s):data_sorted[s+'_sorted']},
								     data_props,export=False)

		data.update(data_sorted)
		
		return data
		
		
		