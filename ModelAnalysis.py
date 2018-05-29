"""
Created on Sun May 13 6:55:25 2018
@author: Matt
"""
import numpy as np
import copy,os,random

from MonteCarloPlot import MonteCarloPlot
from Model import Model
from data_functions import Data_Process
from misc_functions import display,hashable,dim_reduct
from dim_reduce_functions import dim_reduce

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
		data = Data_Process().importer(data_props,upconvert=False,disp=False)
		
		# Used Typed Data Format
		self.data = data[2] if data is not None else data
		self.data_props = data_props
		
		return
		
	
	def process(self,data=None):
	
		def data_set(keys,string=''):
			return list((k,k.replace(string,'')) for k in keys)
				
		if data is None:
			data = self.data
		
		if self.data is None:
			display(time_it=False,m='No Data to Process')
			return
		
		
		# Process Each Data Set
		display(print_it=True,time_it=False,m='Processing Observables')
		self.observables(data,self.data_props)
		
		
		# Sort and Reduce Dimensions of Data
		
		p = self.data_props['analysis_params']['sort_params']	
		keys = {'sort': ['observables_mean_sorted'],
				'dim_reduc':['tsne','pca']}
		string = '_mean'
		if self.data_props.get('sort') and self.data_props.get('dim_reduc'):
			display(m='Sorting and Reducing Data',time_it=False)
			t = self.data_props['data_types']
			data_keys = data_set(keys['sort']+keys['dim_reduc'],string)
		
		elif self.data_props.get('sort'):
			display(m='Sorting Data',time_it=False)
			t = ['sites','observables','model_props']
			data_keys = data_set(keys['sort'],string)
		
		elif self.data_props.get('dim_reduc'):
			display(m='Reducing Data',time_it=False)
			t = ['sites','model_props','tsne','pca']
			data_keys = data_set(keys['dim_reduc'],string)
		
		else:
			display(print_it=True,time_it=False,m='Model Analysis Complete...')
			return
			
			
			
			
		data = self.sort(data,p,self.data_props,t)
	
		# Plot Sorted Data
		if self.data_props.get('plot'):
			display(m='Plotting Sorted Data',time_it=False)
			for p0 in data['model_props_sorted'].keys():
				self.plot({k: data[d][p0] for (k,d) in data_keys},
				data['model_props_sorted'][p0],
				{'arr_0':[p[2],{a1: sorted(list(data['sites_sorted'][p0][
												a1].keys())) for a1 in data[
										 'sites_sorted'][p0].keys()}],
				'arr_1': [p[1],sorted(list(data['sites_sorted'][p0
																].keys()))],
				'sup_title':'%s = %s'%(p[0],str(p0))})
			
		display(print_it=True,time_it=False,m='Model Analysis Complete...')
		return
	
	def observables(self,data,data_props):
		
		def plot_observables(data,key):
			self.plot({#'observables':data['observables'][key],
					   'observables_mean': data['observables'][key]},
					   data['model_props'][key],
					   {'arr_0': ['T',data['model_props'][key]['T']],
						'arr_1':['algorithm',np.atleast_1d([p['algorithm'] 
							for p in data['model_props'][key]['iter_props']])]})
			return
		
		
		if not data.get('observables'):
			data['observables'] = {}
		
		for k,sites in data['sites'].items():
		
			# Update data properties
			data['model_props'][k].update(data_props)
						
			# Check if Data exists
			if data.get('observables',{}).get(k) is not None:
				# Plot Data
				if data_props.get('plot') and False:
					plot_observables(data,k)
				continue

			model_props = data['model_props'][k]
			
			# Measure Data
			
			m = Model(model=model_props,
					 observe= self.data_props['observe_props'][
											  'observables_mean'][1:])
			
			data['observables'][k] = self.measure(sites, 
												 model_props['neighbour_sites'],
												 model_props['T'],
												 m.observables_functions)
			# Save Data
			if model_props.get('data_save',True):
				Data_Process().exporter({'observables':data['observables'][k]},
							   model_props,
							   format=model_props['data_format']['observables'])
						
			# Plot Data
			if data_props.get('plot') and False:
				plot_observables(data,k)
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
			if Data_Process().importer(data_props,
									   data_files = model_props.get('data_file',
																	'None'),
										format = 'pdf') is not None:
				return

		plot_obj = MonteCarloPlot({k: model_props['observe_props'].get(k,
									[True,k]) for k in data.keys()},
							      model_props, **plot_args)
								  
		# Plot Data 
		plot_obj.MC_plotter(data,**plot_args)
		
		# Save Figures
		if model_props.get('data_save',True):
			plot_obj.plot_save(model_props,read_write='ow',fig_size=(12,9))
			
		return
		
	  
	def sort(self,data,parameters,
				  data_props,data_types=None):
	
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
				set_tree(tree,branches[k],data.get(k,data),branch_func,*args)
			
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
				return dim_reduct(data)
			else:
				return np.append(tree[branch[-1]],dim_reduct(data),axis=0)
			
		def observables_sorted(tree,branch,data,root,depth):
			if tree[branch[-1]] == root:
				return data[0]
			else:
				for k in data[0].keys():
					tree[branch[-1]][k] = np.append(np.atleast_1d(
														tree[branch[-1]][k]),
												    np.atleast_1d(data[0][k]),
													axis=-1)
				return tree[branch[-1]] 

		def model_props_sorted(tree,branch,data,depth):
			
			if tree[branch[-1]] != {}:
				return tree[branch[-1]]
			
			# Update data properties
			props = {} #copy.deepcopy(data)
			
			for p in data_props['data_properties']:
				props[p] = copy.deepcopy(data_props.get(p,None))
			
			for i,p in enumerate(data_props['analysis_params'][
											'sort_params'][:depth]):
				props[p] = branch[i]
			
			props['data_name_format'] = [props['data_name_format'][0]] + (
										[props['analysis_params'][
											   'sort_params'][0]]+[''])
			
			for s in set(list(props['observe_props'].keys())+ data_types):
				if s in ['sites', 'sorted']:
					continue
				elif s in ['tsne','pca']:
					props['observe_props'][s] = [True,s]
				else:
					props['observe_props'][
						   s+'_sorted'] = props['observe_props'].get(s,[True,s])
				
			Data_Process().format(props, file_update=True)
			
			return props
			
		def reduce_sorted(tree,branch,data,root,model_props,rep):
			
			keys_func = lambda k,d: np.repeat(k,np.shape(d)[0]//len(k))
			
			for b in branch:
				data = data[b]
				model_props = model_props.get(b,model_props)					
			
			keys = sorted(data.keys())
			data = [data[k] for k in keys]
			
			# Check if Data Exists
			if tree[b] != root:
				return tree[b]
			
			dim_reduc_params = data_props['analysis_params'][
										  'dim_reduc_params'].copy()
			temp_name = '_'.join(['rep',rep[0],
							'_'.join(p for p in parameters[1:len(branch)]) + (
							'_'.join(str(b) for b in branch[1:]))])
			data_temp = Data_Process().importer(model_props,
												data_files = model_props[
													'data_file']+temp_name,
												disp=True)
			input('Prepare for data concatenation')
			if data_temp is None:
				print('New %s Data at '%rep,parameters[:len(branch)],
					branch)
				Ns = int(np.shape(data)[1]*dim_reduc_params.pop('Ns'))
				#i = int(random.random()*np.shape(data)[1]*(1-Ns))
				i = 0
				# Ns = np.array(random.sample(,
										# int(np.shape(data)[1]*Ns)))
										
				input('Prepare for data concatenation after random choice')
				data = np.concatenate(tuple([d[i:i+Ns]
											 for d in data]),axis=0)
				if True:
					data =  dim_reduce(data,rep=rep, **dim_reduc_params)
				
				Data_Process().exporter({temp_name:data},model_props,
										 read_write = 'a')
				
				
			else:
				print('Previous %s Data at '%rep,parameters[:len(branch)],
					branch,
					np.shape(data_temp[0][model_props['data_file']+temp_name]))
				
				data = data_temp[0][model_props['data_file']+temp_name]
			
			keys = keys_func(keys,data)
				
			return (keys,data)
			
			
		
		
		# Sort by params as {param0_i: {param1_j: [param2_k values]}}
		# i.e) q, L, T


		# Check if sorted data exists
		data_props = copy.deepcopy(data_props)
		
		if data_types is None:
			data_types = [s for s in data_props['data_types'] 
						if s not in ['sorted']]
		
		if ('tsne' in data_types or 'pca' in data_types) and (
					'sites_sorted' not in data.keys()):
			data_types = ['sites'] + [s for s in data_types
										if s not in ['sites','sorted']]
							
		file_header = data_props['data_dir'].split('/')[-2]
		file_format = lambda s: file_header + s
		if data.get('sorted'):
			for k,v in (data['sorted'].copy()).items():
				data[k.split(file_header)[1]+ 'sorted'] = v.copy()
				data['sorted'].pop(k)
			data.pop('sorted')			
			return data

		root = {}
		depth = {}
		args = {}
		data_sorted = {}
		branch_func = {}
		for s in data_types:
				
			s_key = s+'_sorted'
			s_data = data.get(s,{})
			root[s] = []
			depth[s] = None
			args[s] = (root[s],)
			branch_func[s] = locals().get(s_key)
			export = False
			if s == 'sites':
				root[s] = []
				depth[s] = None
			elif s == 'observables':
				root[s] = []
				depth[s] = None
				args[s] = (root[s],depth[s])
			elif s == 'model_props':
				root[s] = {}
				depth[s] = -2
				args[s] = (depth[s],)
			elif s in ['tsne','pca']:
				root[s] = []
				depth[s] = -1
				args[s] += (data['model_props_sorted'],s)
				s_key = s
				s_data =  data['sites_sorted']
				branch_func[s] = locals().get('reduce_sorted')
			else:
				root[s] = []	
				depth[s] = None
				args[s] = (depth[s],)
			
			print('Sorting ',s)
			if data.get(s_key) is not None:
				if file_header in data[s_key].keys():
					data[s_key] = data[s_key][file_header]
				continue
			
			(data[s_key],
			   data_props['branches']) = tree_sort(
													parameters, 
													data['model_props'],
													s_data, 
													branch_func[s],
													root[s],depth[s],*args[s]
												  )
			# Save Sorted Data
			Data_Process().exporter({file_format(s_key):data[s_key]},
								     data_props,read_write='ow',export=export)

		
		return data
		
		
		
