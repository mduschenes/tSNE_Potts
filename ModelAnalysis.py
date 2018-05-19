"""
Created on Sun May 13 6:55:25 2018
@author: Matt
"""
import numpy as np

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
				if self.data_props.get('plot'):
					self.plot(data['observables'][k],data['model_props'][k],
							 {'arr_0': 'T',
							 'arr_1':['algorithm',
								    data['model_props'][k]['iter_props'][0]['algorithm']]})
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
									     format=model_props['data_format'])
						
			# Plot Data
			if self.data_props.get('plot'):
				self.plot(observables,model_props,
							{'arr_0': 'T',
							 'arr_1':['algorithm', np.atleast_1d(model_props[
											   'iter_props'][0]['algorithm'])]})
				
			data['observables'][k] = observables
		
		
		# Sort Data
		# if self.data_props.get('sort'):
			# display(m='Sorting Data',time_it=False)
			# data = self.sort(data,self.data_props['sort_parameters'])
		
		# # Plot Sorted Observables
		# p = self.data_props['sort_parameters']
		# print(data['model_sorted'])
		# if self.data_props.get('plot'):
			
			# display(m='Plotting Sorted Data',time_it=False)
			
			# for props0 in data['parameters']['values'][p[0]]:
				
				# # Update data file name
				# data['model_sorted'][props0]['data_file_format'] = (
				   # [data['model_sorted'][props0]['data_file_format'][0]]+p+[''])
				# Data_Process().format(data['model_sorted'][props0],
									  # file_update=True)
				
				# print({'arr_0':[key_prop0,val_prop0],
					   # 'arr_1': key_prop1},data['model_sorted'][p]['data_file'])
					# #continue
				
				# self.plot(data['obs_sorted'][p],data['model_sorted'][p],
						# {'arr_0':p[3],'arr_1': [2],
						 # 'sup_title':'%s = %s'%(p[0],str(props0))})
		
		
		# display(print_it=True,time_it=False,m='Model Analysis Complete...')
		# return
		
	def measure(self,sites,neighbours,T,observables):
	
		if sites.ndim < 4:
			sites = sites[np.newaxis,:]

		n_iter = np.shape(sites)[0]

		data = [{} for _ in range(n_iter)]

		for i_iter in range(n_iter):        
			for k,obs in observables.items():
				data[i_iter][k] = obs(sites[i_iter],neighbours,T)

		return data
		
	
	def plot(self,data,model_props,parameters={}):
	
		# Plot Instance
		plot_args = {}
		if parameters == {}:
			plot_args = {'arr_0': ['T',model_props['T']],
						 'arr_1':['algorithm',[model_props['algorithm']]]}
		else:
			for key_arg,param in parameters.items():
				try:
					plot_args[key_arg] = [param,model_props[param]]
				except:
					plot_args[key_arg] = param
							  
		print(plot_args)
		plot_obj = MonteCarloPlot(model_props['observe_props'],
							      model_props, **plot_args)
								  
		# Plot Data
		plot_obj.MC_plotter({'observables':      data,
						     'observables_mean': data},
						     **plot_args)
		
		# Save Figures
		if model_props.get('data_save',True):
			plot_obj.plot_save(model_props,read_write='w')
			
		return
		
		
		
	# def sort(self,data,parameters,parameters0=lambda x: {},coupled=False):
	
		# # Sort by params as {param0_i: {param1_j: [param2_k values]}}
		# # i.e) q, L, T
		
		# def dim_reduce(arr,axis=0):
			# try:
				# return np.squeeze(arr,axis)
			# except:
				# return arr
		
		# def return_item(d,item=0,index=0):
			# return list(d.items())[index][item]
		
		# def nested_dict(keys,base_type={}):
			# if len(keys) > 0:
				# d = {}
				# for k in keys[0]:
					# d[k] = nested_dict(keys[1:],base_type)
			# else:
				# d = np.copy(base_type)
			# return d
			
		# def build_dict(data,container,parameters,model,bar=False):
			# # if bar: 
				# # for p in parameters[0].values():
					# # print(container[p][0][list(parameters[1].keys())[0]])
			# if isinstance(container,dict) and len(parameters)>1:
				# for ic,cparam in enumerate(container.keys()):
					# for k in data.keys():
						# if model[k][return_item(parameters[0],0)] == cparam:
							# build_dict({k:data[k]},container[cparam],
							            # parameters[1:],model)
							# continue
			# elif isinstance(container,np.ndarray):
				# k,v =  return_item(data,slice(0,2))
				# if isinstance(container[0],object) and len(parameters) == 1:
					# #container = dim_reduce(container,0)
					# container[int(np.where(return_item(parameters[0],1)== 
								# model[k][return_item(parameters[0],0)])[0])] = (
																# dim_reduce(v,0))
					# if not None in container:
						# try:
							# container = np.concatenate(tuple(container))
						# except:
							# pass
				# elif isinstance(container[0],dict):
						# i = np.array(return_item(parameters[0],1)) == (
									    # model[k][return_item(parameters[0],0)])
						# c = container[i].item()
						# for kc in c.keys():
							# c[kc][return_item(parameters[1],1) == (
									# model[k][return_item(parameters[1],0)]
									# )] = dim_reduce(v[0][kc],0)
				# elif isinstance(container[0],np.ndarray):
					# for p in parameters:
						# print(k,p)
						# container[0] = data[k].copy()
						# for kp,vp in p.items():
							# temp = container[0][0][kp]
							# if isinstance(vp,str):
								# container[0][0][kp] = vp
							# else:
								# container[0][0][kp] = np.array(vp,
															  # dtype=type(vp[0]))
							# print(kp,temp,container[0][0][kp])
					# container = dim_reduce(container)
			
			# # Process Container
			# if bar: 
				# print(parameters)
				# for i in [1,2]:
					# for p in list(parameters[0].values())[0]:
						# print(list(parameters[i].keys())[0])
						# print([m[list(parameters[i].keys())[0]]for m in model_props])
						# print(container[p][0][0][list(parameters[i].keys())[0]])
				# exit()
			# try:
				# if isinstance(return_item(container,1)[0][0],dict):
					# for k in container.keys():
						# container[k] = container[k][0][0]
			# except:
				# pass
		
		# # Data Characteristics
		# data = data.copy()
		# key0 = return_item(data['sites'])
		
		
		# # Find parameter Coordinates for each dataset
		# model_props = data['model_props']
		
		# parameter_sets = {}
		# for k,m in data['model_props'].items():
			# parameter_sets[k] = [m[p] for p in parameters]
		
		# # Find Unique Parameter Coordinates
		# paramets_unique = sorted(map(list,set(map(tuple,d))),
				# key=lambda x: tuple([x[i] for i in range(len(x))]))
		
		
		
		
		
		
		
		# sites_size = list(set([np.shape(dim_reduce(v,0)) 
								# for v in data['sites'].values()]))
		# obs_size = {k: np.shape(dim_reduce(v))for k,v 
						# in data['observables'][key0][0].items()}
		
		
		# # Order of N Parameters given indicates how data will be sorted
		# # Let the last sorting be by the last parameter pN
		# pN = parameters[-1]
		# pN1 = parameters[-2]
		
		
		
		# # Create new Parameters type in Data, with properties values, types etc.
		# data['parameters'] = {}
		# data['parameters']['values'] = [None for p in parameters]
		# data['parameters']['types'] = {}
		# data['parameters']['sizes'] = {}

		# for i,p in enumerate(parameters):
			# types = type(data['model_props'][key0][p])
			# vals = sorted(set(np.append([],[m[p]for m in model_props]
															   # ).astype(types)))
			# data['parameters']['values'][i] = vals
			# data['parameters']['types'][p] = types
			# data['parameters']['sizes'][p] = np.size(vals)
		
		
		
		
		# # Create nested dictionary to sort data by parameters p1...pN-1
		# if len(sites_size)>1:
			# data['sites_sorted']=nested_dict(data['parameters']['values'][:-1],
					   # np.array([None for _ in 
								# range(data['parameters']['sizes'][pN])],
								# dtype=object))
		# else:
			# data['sites_sorted']=nested_dict(data['parameters']['values'][:-1],
									# np.zeros((data['parameters']['sizes'][pN],)+
														# tuple(sites_size[0])))
		
		# data['obs_sorted'] = nested_dict(data['parameters']['values'][:-2],
							# np.array([
							# {k: np.zeros((data['parameters']['sizes'][pN],)+
								# tuple(s)) for k,s in obs_size.items()}.copy() 
						   # for _ in range(data['parameters']['sizes'][pN1])]))
		# data['model_sorted'] = nested_dict(data['parameters']['values'][:-2],
																		 # [[{}]])
						 
						 
						 
						 
		# # Sort Data into nested dictionary structure
		# # Change data structyre of parameter values for easier sorting
		# # Depending on whether the last two parameters are coupled,
		# # build dictionaries differently.
		# # param_vals = data['parameters']['values'].copy()
		# # data['parameters']['values'] = []
		# # for i,p,d in enumerate(zip(parameters,param_vals)):
			# # if i < len(paramval)-1:
				# # data['parameters']['values'].append({p:d}) 
			# # else:
				# # data['parameters']['values'].append(
								# # {(pN1,PN): [param_vals[-2], param_vals[-1]})
		# # # Potentially Add other Parameters to Final Sorting Dictionary
		# # build_dict(data['sites'],data['sites_sorted'],
				   # # data['parameters']['values'],data['model_props'])

		# # build_dict(data['observables'],data['obs_sorted'],
				   # # data['parameters']['values'],data['model_props'])
		
		# # #print(data['model_sorted'])
		# # data['parameters']['values'][-1].update(parameters0)
		# # build_dict(data['model_props'],data['model_sorted'],
				   # # data['parameters']['values'],data['model_props'],True)
		
		
		# # # Change back data structure of parameter values
		# # data['parameters']['values'] =  {p: data['parameters']['values'][i]
											# # for i,p in enumerate(parameters)}
		# #print(data['model_sorted'])
		
		# return data
		
		
		
		
		