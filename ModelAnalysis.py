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
	
		def data_set(keys,string='',exceptions={None:None}):
			keys_tuple = []
			for k in keys:
				if k in exceptions.keys():
					for p0 in data['model_props_sorted'].keys():
						for s in data[d][p0].keys():
							keys_tuple.append(((exceptions[k],data[d][p0][s]), 
												k.replace(string,'')))
				else:
					keys_tuple.append((k,k.replace(string,'')))
			return keys_tuple
			
		if data is None:
			data = self.data
		
		if self.data is None:
			display(time_it=False,m='No Data to Process')
			return
		
		
		# Process Each Data Set
		display(print_it=True,time_it=False,m='Processing Observables')
		observables_functions = self.data_props.get('observables_functions',
										   self.data_props['observe_props'][
														'observables_mean'][1:])
		self.observables(data,observables_functions,self.data_props)
		
		
		# Sort and Reduce Dimensions of Data
		
		p_sort = self.data_props['analysis_params']['sort_params']	
		keys = {'sort': ['observables_mean_sorted'],
				'dim_reduc':['tsne','pca']}
		string = '_mean' 
		exceps = {'sites_sorted':'configurations'}
		if self.data_props.get('sort') and self.data_props.get('dim_reduc'):
			display(m='Sorting and Reducing Data',time_it=False)
			data_types = self.data_props['data_types']
			data_keys = data_set(keys['sort']+keys['dim_reduc'],string,exceps)
		
		elif self.data_props.get('sort'):
			display(m='Sorting Data',time_it=False)
			data_types = ['sites','observables','model_props']
			data_keys = data_set(keys['sort'],string,exceps)
		
		elif self.data_props.get('dim_reduc'):
			display(m='Reducing Data',time_it=False)
			data_types = ['sites','model_props','tsne','pca']
			data_keys = data_set(keys['dim_reduc'],string,exceps)
		
		else:
			display(print_it=True,time_it=False,m='Model Analysis Complete...')
			return
			
			
		data = self.sort(data,p_sort,self.data_props,data_types)
	
		# Plot Sorted Data
		if self.data_props.get('plot'):
			display(m='Plotting Sorted Data',time_it=False)
			for p0 in data['model_props_sorted'].keys():
				self.plot({k: data[d][p0] for (k,d) in data_keys},
				data['model_props_sorted'][p0],
				{'arr_0':[p_sort[2],{a1: sorted(list(data['sites_sorted'][p0][
												a1].keys())) for a1 in data[
										 'sites_sorted'][p0].keys()}],
				'arr_1': [p_sort[1],sorted(list(data['sites_sorted'][p0
																].keys()))],
				'sup_title':'%s = %s'%(p_sort[0],str(p0)),
				 'orientation': False})
			
		display(print_it=True,time_it=False,m='Model Analysis Complete...')
		return
	
	def observables(self,data,observables_functions,data_props):
		
		def plot_observables(data,key):
			self.plot({#'observables':data['observables'][key],
					   'observables_mean': data['observables'][key]},
					   data['model_props'][key],
					   {'arr_0': ['T',data['model_props'][key]['T']],
						'arr_1':['algorithm',np.atleast_1d([p['algorithm'] 
							for p in data['model_props'][key]['iter_props']])]})
			return
		
		def plot_sites(data,key):
			data_sites = {('sites',t): d[-1,:] 
							for d,t in zip(np.atleast_3d(data['sites'][key]),
											data['model_props'][key]['T'])}
			self.plot({#'observables':data['observables'][key],
					   'configurations': data_sites},
					   data['model_props'][key],
					   {'arr_0': ['T',data['model_props'][key]['T']],
						'arr_1':['algorithm',np.atleast_1d([p['algorithm'] 
							for p in data['model_props'][key]['iter_props']])]})
			return
		
		
		if not data.get('observables'):
			data['observables'] = {}
			
		# Update Plotting Properties
		for s in set(list(data_props['observe_props'].keys()) + 
						  data_props.get('data_types',[])):
			if s in ['sorted']:
				continue
			elif s in ['sites']:
				t = 'configurations'
				data_props['observe_props']['configurations'] = [True,s]
			elif s in ['tsne','pca']:
				t = s
				data_props['observe_props'][s] = [True,s]
			else:
				t = s+'_sorted'
			data_props['observe_props'][t] = data_props['observe_props'].get(t,
											 data_props['observe_props'].get(s,
											 [True,s]))
		
		for k,sites in data['sites'].items():
		
			# Update data properties
			data['model_props'][k].update(data_props)
						
			# Check if Data exists
			if data.get('observables',{}).get(k) is not None:
				# Plot Data
				if data_props.get('plot') and False:
					plot_observables(data,k)
				#if data_props.get('plot'):
					plot_sites(data,k)
				continue

			model_props = data['model_props'][k]
			
			# Measure Data
			
			m = Model(model=model_props,observe=observables_functions)
			
			data['observables'][k] = self.measure(sites, 
												 model_props['neighbour_sites'],
												 model_props['T'],
												 m.observables_functions)
			# Save Data
			if model_props.get('data_save',True):
				Data_Process().exporter({'observables':data['observables'][k]},
							   model_props,
							   file=k,
							   format=model_props['data_format']['observables'])
							   
						
			# Plot Data
			if data_props.get('plot') and False:
				plot_observables(data,k)
			#if data_props.get('plot'):
				plot_sites(data,k)
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

		plot_obj = MonteCarloPlot({k: self.data_props['observe_props'].get(k,
									[True,k]) for k in data.keys()},
							      model_props, **plot_args)
								  
		# Plot Data 
		plot_obj.MC_plotter(data,**plot_args)
		
		# Save Figures
		if model_props.get('data_save',True):
			plot_obj.plot_save(model_props,read_write='ow',fig_size=(18,12))
			
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
				b0_list = []
				for b in branches:
					b0_list.extend(np.atleast_1d(b[0]))
				for b0 in set(b0_list):
					tree[b0]=get_tree([b[1:] for b in branches 
									         if b0 in np.atleast_1d(b[0])],
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
			tree = branch_func(tree,branch,data,*args)
			
		
		def sites_sorted(tree,branch,data,root):
			for i,b in enumerate(np.atleast_1d(branch[-1])):
				if tree[b] == root:
					tree[b] = data[i]
				else:
					tree[b] = np.append(tree[b],data[i],axis=0)
			return tree
			
		def observables_sorted(tree,branch,data,root,depth):
			for i,b in enumerate(np.atleast_1d(branch[-1])):
				for j in range(np.shape(data)[0]):
					for k in data[j].keys():
						if k not in tree[b].keys():
							tree[b][k] = data[j][k][i]
						else:
							tree[b][k] = np.append(np.atleast_1d(tree[b][k]),
												   np.atleast_1d(
												   dim_reduct(data[j][k][i])),
												   axis=-1)
			return tree
		

		def model_props_sorted(tree,branch,data,depth):
			
			if tree[branch[-1]] != {}:
				return tree
			
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
				
			Data_Process().format(props, file_update=True)
			tree[branch[-1]] = props
			return tree
			
		def reduce_sorted(tree,branch,data,root,model_props,rep):
			
			keys_func = lambda k,d: np.repeat(k,np.shape(d)[0]//len(k))
			data_func = lambda data,N0,N: np.concatenate(tuple([d[N0:N0+N]
														for d in data]),axis=0)
			
			for b in branch:
				data = data[b]
				model_props = model_props.get(b,model_props)					
			
			keys = sorted(data.keys())
			data_concat = [data[k] for k in keys]
			
			# Check if Data Exists
			if tree[b] != root:
				return tree
			
			dim_reduc_params = data_props['analysis_params'][
										  'dim_reduc_params'].copy()
			temp_name = '_'.join(['rep',rep[0],
							'_'.join(p for p in parameters[1:len(branch)]) + (
							'_'.join(str(b) for b in branch[1:]))]) + (
							'.'+model_props['data_format'][rep])
			data_temp = Data_Process().importer(model_props,
												data_files = model_props[
													'data_file']+temp_name,
												disp=True)
			
			Ns = int(np.shape(data_concat)[1]*dim_reduc_params.pop('Ns'))
			N0 = 0
			data_concat = data_func(data_concat,N0,Ns)
			if data_temp is None:
				print('New %s Data at '%rep,parameters[:len(branch)],
					branch)
				#N0 = int(random.random()*np.shape(data)[1]*(1-Ns))
				# Ns = np.array(random.sample(,
										# int(np.shape(data)[1]*Ns)))
										
				#input('Prepare for data concatenation after random choice')
				if True:
					data_reduc =  dim_reduce(data_concat,rep=rep, 
												**dim_reduc_params)
				
				Data_Process().exporter({temp_name.split('.')[0]:data},
									     model_props,
										 read_write = 'a')
				
				
			else:
				print('Previous %s Data at '%rep,parameters[:len(branch)],
					branch,
					np.shape(data_temp[0][model_props['data_file']+
								temp_name.split('.')[0]]))
				
				data_reduc = data_temp[0][model_props['data_file']+
									temp_name.split('.')[0]]
			
			keys = keys_func(keys,data_reduc)
			tree[b] = (keys,data_reduc,[int(np.mean(d)) for d in data_concat])			
			return tree
			
			
		
		
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
							
		file_header = os.path.split(data_props['data_dir'])[0]
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
				root[s] = {}
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
		
		
		
