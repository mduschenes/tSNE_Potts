# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 19:54:38 2018
@author: Matt
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os,glob,copy

from misc_functions import (flatten,dict_check, one_hot,caps,display,
							list_sort,str_check,delim_check,dict_reorder)
import plot_functions

NP_FILE = 'npz'
IMG_FILE = 'pdf'
DIRECTORY = 'dataset/'
DELIM = [' ','.']
BACKEND = 'Agg'
DISP = False
ORIENTATION = False

class Data_Process(object):
    
    # Create figure and axes dictionaries for dataset keys
	def __init__(self,keys=[None],plot=[False],
				 np_file = None,img_file=None,backend=None,
				 directory=None,delim=None,disp=False,orientation=None):
		 
		# Standardized attributes
		for f,F in [(v,V) for v,V in locals().items() 
					if v not in ['keys','plot']]:
				F = globals().get(f.upper()) if  F is None else F
				setattr(self,f.upper(),F)
		
		# Set plotting backend
		#matplotlib.use(self.BACKEND)

		
		# Initialize figures and axes dictionaries
		if not None in keys:
			self.figs = {}
			self.axes = {}
			
			self.keys = keys
			
			self.plot_set(plot)
			
			self.data_key = 'data_key'
			
			# Create Figures and Axes with keys
			self.figures_axes(keys,self.ORIENTATION)

		return  
    


     # Plot Data by keyword
	def plotter(self,data,domain=None,plot_props={},data_keys=None,keys=None,
				disp=None):
		if data_keys is None and self.axes.keys() is not {}:
			data_keys = list(self.axes.keys())
		elif data_keys is None:
			data_keys = self.data_key
		data_keys = np.atleast_1d(data_keys)
		keys0 = keys
		for i,data_key in enumerate(data_keys):
			keys = keys0

			if not self.plot.get(data_key,True):
				return

			# Display
			if disp is None:
				disp = self.DISP
				
			# Ensure Data, Domain and Keys are Dictionaries
					
			plot_key = ''
			if not isinstance(data,dict):
				data = {plot_key: data}

			if data_key is None:
				data_key = self.data_key
				if self.data_key in data_keys[:i]:
					return
				else:
					data_keys[i] = data_key


			self.plot[data_key] = True

			if keys is None:
				keys = [k for k in data.keys() if k in self.axes.get(data_key,
															   data.keys())
										   or k == plot_key]

			keys = [k for k in keys if data.get(k,[]) != []]

			dom = {}
			
			def shape_data(x,y):
				x = np.atleast_1d(x)
				if x is not None and np.size(x) == np.size(y):
					return np.reshape(x,np.shape(y))
				else:
					return x

			for k in keys:    
				if not isinstance(domain,dict) and isinstance(data[k],dict):
					dom[k]={ki: shape_data(domain,data[k][ki]) 
								for ki in data[k].keys()}
				elif not isinstance(domain,dict):
					dom[k] = shape_data(domain,data[k])
				elif isinstance(data[k],dict):
					if isinstance(domain[k],dict):
						dom[k] = {ki: shape_data(domain[k][ki],data[k][ki])
									for ki in data[k].keys()}
					else:
						dom[k] = {ki: shape_data(domain[k],data[k][ki]) 
									for ki in data[k].keys()}
				else:
					dom = domain

			domain = dom

			# Create Figures and Axes
			self.figures_axes({data_key:keys})            
			
			
			# Plot for each data key
			for key in sorted(keys):
			
				display(print_it=disp,time_it=False,
							m = 'Plotting %s'%(str_check(key)))
				props = plot_props.get(key,plot_props)
				# props['other']['backend'] = self.BACKEND
				props['other']['plot_key'] = key
				# try:
				ax = self.axes[data_key][key]
				fig = self.figs[data_key][key]
				plt.figure(fig.number)
				fig.sca(ax)
				ax.cla()
			# except:
				# 	self.figures_axes({data_key:keys})
					
				# 	ax = self.axes[data_key][key]
				# 	fig = self.figs[data_key][key]
				# 	plt.figure(fig.number)
				# 	fig.sca(ax)
				# 	ax.cla()
				
				# Plot Data
				#try:
				getattr(plot_functions,'plot_' + props.get('data',{}).get(
						'plot_type','plot'))(data[key],domain[key],fig,ax,props)
				# except AttributeError:
					# props.get('data',{}).get('plot_type')(
								   # data[key],domain[key],fig,ax,props)

					# display(m='Figure %s Created'%(
										# plot_props[key]['data']['plot_type']))
											
				
				# Figure Title and Legend
				plt.suptitle(**props.get('other',{}).get('sup_title',{}))
				if props.get('other',{}).get('sup_legend') and (
					len(ax.get_legend_handles_labels()) > 1):
					fig.legend(*(list_sort(ax.get_legend_handles_labels(),1)),
								**props.get('other',{}).get('legend',{}))

		return
            

	# Set to Plot
	def plot_set(self,plot,keys=None,update=True):
		if keys is None:
			keys = self.keys
		if np.size(plot) == 1 and not isinstance(plot,dict) and update:
			self.plot = {k: np.atleast_1d(plot)[0] for k in keys}
		elif (not isinstance(plot,dict)) and update:
			self.plot = {k: plot for k in keys}
		else:
			self.plot = plot
		return

	# Clost all current figures and reset figures and axes dictionaries
	def plot_close(self):
		plt.close('all')   
		self.axes = {}
		self.figs ={}
		return
    
    
	# Save all current figures
	def plot_save(self,data_params={'data_dir':'dataset/',
									'figure_format':None},
					   fig_keys = None,directory=None,format=None,
					   label = '',read_write='w',fig_size=(8.5,11)):#(8.5,11)):
        
        # Save Figures for current Data_Process Instance
		if format is None:
			format = data_params.get('figure_format',self.IMG_FILE)
			
		# Data Directory
		if directory is None:
			directory = data_params.get('data_dir','dataset/')

		if not os.path.isdir(directory):
			os.makedirs(directory)
		
		
		# Check for specific fig_keys to save
		if fig_keys is None:
			fig_keys = self.figs.keys()
		
		for fig_k,fig_i in [(k,f) for k,f in self.figs.items()
							if k in fig_keys and self.plot[k]]:
			
			for ifig in set([f.number for f in fig_i.values()]):
			
				# Set Current Figure
				plt.figure(ifig)        
				fig = plt.gcf()
				
				# Change Plot Size for Saving                
				plot_size = fig.get_size_inches()
				if fig_size is not None:
					fig.set_size_inches(fig_size)

				# Set File Name and ensure no Overwriting
				if not label in [None,'']:
					label = '_' + str_check(label)
				else:
					label = ''
				
				if data_params.get('data_file'):
					file = directory + data_params.get('data_file') + '_' + (
								fig_k + label)
				else:
					file = directory + fig_k + label
			
			
				i = 0
				file_end = ''
				while os.path.isfile(file + file_end + '.'+format):
					file_end = '_%d'%i
					i+=1

				# Save Figure as File_Format
				if read_write == 'w': 
					if i > 0: pass 
					else: file_end = ''
				elif read_write == 'ow':
					file_end = ''
				plt.savefig(file + file_end+'.'+format,
							dpi=500,bbox_inches='tight')
				fig.set_size_inches(plot_size) 
			
		return
    
      
    
    
	 # Create figures and axes for each passed set of keys for datasets
	def figures_axes(self,Keys,orientation=False):     
		if not isinstance(Keys,dict):
			Keys = {self.data_key: Keys}
			   
		for keys_label,keys in Keys.items():

			keys_new = [k if k not in self.axes.get(keys_label,{}).keys() 
						else None for k in flatten(keys,False)]
			if not None in keys_new and self.plot[keys_label] and keys_new != []:
				
				if len(self.axes.get(keys_label,{})) > 1: 
					print('Figure Keys Updated...')

				self.axes[keys_label] = {}
				self.figs[keys_label] = {}

				key_shape = np.shape(keys)[:2]
				
				if orientation and np.size(key_shape) == 1:
					key_shape = (1,)+key_shape
				elif orientation:
					key_shape = (key_shape[1],key_shape[0])
					
				fig, ax = plt.subplots(*(key_shape))
				
				fig.canvas.set_window_title('Figure %d:  %s Datasets'%(
												  fig.number,caps(keys_label)))
				for k,a in zip(keys_new,flatten(np.atleast_1d(ax).tolist(),
																		False)):
					if k is not None:
						a.axis('off')
						self.axes[keys_label][k] = a
						self.figs[keys_label][k] = fig      
				
		return 






	# Import Data
	def importer(self,data_params = 
					{'data_files': ['x_train','y_train','x_test','y_test'],
					 'data_sets': ['x_train','y_train','x_test','y_test'],
					 'data_typed':'dict_split',
					 'data_format': None,
					 'data_dir': 'dataset/',
					 'data_lists': False,
					 'one_hot': [False,'y_'],
					 'upconvert':False,
					 'data_seed':{'train':{'seed_type':'test',
					 					   'seed_proportion':[0,0,1/20]},
					 			'test':{'seed_type':'test',
					 					   'seed_proportion':[0,1,1/20]}}
					},directory=None,data_files = None,
					format=None,data_typing=None,
					data_obj_format=None,disp=None,
					upconvert=False,delim=None,data_lists=False):

		# Display
		if disp is None:
			disp = self.DISP
			
		# Bad delimeters
		if delim is None:
			delim = data_params.get('delim',self.DELIM)
					
		# Data Files Dictionary
		# Check of importing batch of files
		data_params = dict_check(data_params,'data_files')

		if not data_params.get('data_sets'):
			data_params['data_sets'] = data_params['data_files']            

		if data_files is None:
			data_files = data_params.get('data_files','*.'+self.NP_FILE)
		
		if directory is None:
			directory = data_params.get('data_dir',self.DIRECTORY)


		if data_lists is None:
			data_lists = data_params.get('data_lists',False)

		if upconvert is None:
			upconvert = data_params.get('upconvert',False)

		if not os.path.isdir(directory):
			os.makedirs(directory)
			
		# Import Data				
		
		def import_func(file,directory,format,key,dtype=None,**kwargs):		
			formatter = lambda s,f: s.split('.')[0] + '.'+f.split('.')[-1]			
			if not os.path.getsize(directory + formatter(file,format)) > 0:
				return None			
			
			if format == 'values':
				data = file
			
			elif format == 'npy':
				data = np.load(directory + formatter(file,format))
				
			elif format == 'npz':
				data = np.load(directory + formatter(file,format))[
								   kwargs.get(key,{}).get('array_name','arr_0')]
			elif format == 'txt':
				data = np.loadtxt(directory + formatter(file,format))
			
			elif format == 'pdf':
				data = file
			
			else:
				data = None
			
			dtype = key if dtype is None else dtype
			
			return process(key,data,dtype)
		
		def process(key,data,dtype):

			if data_obj_format is None:
				data_obj = data_params.get('data_obj_format',{}
													).get(dtype,'array')
			else:
				data_obj = data_obj_format
			
			if data is None:
				return data
			
			elif 'np' in format.get(key,[]) and data_obj == 'dict':
				try:
					return data.item()
				except AttributeError:
					return data
			elif not isinstance(data,str):
				return data		
			else:
				return data		
		
		# Import Data				
		if isinstance(data_files,(tuple,str)):
			files = []
			format_files = {}
			for f in np.atleast_1d(data_files):
				if format is not None:
					f = f.split('.')[0] + '.' + format.split('.')[-1]
				if '*' in f:
					ftemp = [os.path.basename(x) 
								for x in glob.glob(directory+f)]
				else:
					ftemp = [x for x in os.listdir(
									     directory) 
										 if os.path.isfile(os.path.join(
													directory, x)) 
										 and f in x ]
				files.extend(ftemp)
			
			
			for fi in files:
				format_files[fi] = fi.split('.')[-1]
				
			if files == []:
				return None
				
			files = dict_check(files,data_params.get('data_sets',files.copy()))
		else: 
			files = data_files.copy()
			files = dict_check(files,data_params.get('data_sets',files.copy()))
			format_files = {k: data_params.get('data_format',self.NP_FILE)
							for k in files.keys()}
						
		
		if format is None:
			format = {k: format_files.get(f,self.NP_FILE)
								for k,f in files.items()}
		else:
			format = {k:format for k in files.keys()}
						
		data = {}
		data_types = {k: t if t in k else None 
						for t in data_params['data_types']
						for k in files.keys()}
						
		for i,(k,v) in enumerate(files.items()):
			if v is not None:
				display(print_it=disp,time_it=False,
					m = 'Importing %d/%d %s'%(i+1,len(files),k))
				data[k] = import_func(v,directory,format[k],k,
											data_types[k])
		if data == {}:
			return None
		
		
		
		# Processs Data
		
		
		# Ensure Files have no spaces	
		
		delim_check(data,delim)
		delim_check(format,delim)
		
		# Convert Labels to one-hot Labels
		if data_params.get('one_hot',[None])[0]:
			for k in data.keys(): 
				if data_params.get('one_hot',[None])[1] in k:
					data[k] = one_hot(data[k])
		
		

		
		
		
		# Type of Data Sets and Data Set Keys
		if data_typing is None:
			data_typing = data_params.get('data_typing','dict')
		
		if not data_params.get('data_types'):
			data_typed = data.copy()
			
		else:
			def data_typer(data,dtype,data_typed):				
				k_list = [k for k in data.keys() if (dtype in k) and (
							 k not in dt.keys() for dt in data_typed.values())]
				if data_typing == 'dict':
					return {k: process(k,data[k],dtype) for k in k_list},k_list
			
				elif data_typing == 'dict_split':
					return {k.split(dtype)[0]: process(k,data[k],dtype) 
							for k in k_list},k_list
				
				else:
					return [process(k,data[k],dtype) for k in k_list],k_list
		
			data_typed = {}
			data_keys = {}
			for t in data_params['data_types']:
				# Define Sets
				data_typed[t],data_keys[t] = data_typer(data,t,data_typed)
				if data_typed[t] == {}:
					data_typed.pop(t);
					continue
				
				
				# Check Delimeters
				delim_check(data_typed[t],delim)
				delim_check(data_keys[t],delim)

			# If missing data, seed data into different types
			if data_params.get('data_seed'):

				index_types = (list,np.ndarray,slice)
				new_types = [(t,d) for t,d in data_params['data_seed'].items() 
							if (t not in data_typed.keys()) and (
								data_typed.get(d['seed_type']))]
				other_types = [t for t in data_typed.keys() if  (
										t not in data_params['data_seed']) and (
										t not in dict_reorder(
											data_params['data_seed'],
											'seed_type',True).values())]
				for t,d in new_types:
					t_seed = d['seed_type']

					dicttype_seed = isinstance(data_typed[t_seed],dict)
					if dicttype_seed:
						data_typed[t] = {}
					else:
						data_typed[t] = [None]*len(data_typed[t_seed])

					for i,k in enumerate(data_typed[t_seed]):
						if dicttype_seed:
							v = data_typed[t_seed][k]
						else:
							v = k
						
						shape_seed = list(np.shape(v))
						axis_seed = d['seed_dimensions'][0]
						start_seed = d['seed_dimensions'][1]

						if len(d['seed_dimensions']) == 3:
							indices_seed = d['seed_dimensions'][2]  
						else:
							indices_seed = None
						n_seed = shape_seed[axis_seed]

						if isinstance(start_seed,int):
							if isinstance(indices_seed,float):
								indices_seed = np.arange(i_start,
												i_start+int(indices_seed*
												n_seed))
							elif isinstance(indices_seed,int):
								indices_seed = np.arange(i_start,
												i_start+indices_seed)
						elif isinstance(start_seed,index_types):
							indices_seed = np.array(start_seed)
						elif start_seed == 'end':
							start_seed = n_seed
							if isinstance(indices_seed,float):
								indices_seed = np.arange(
												start_seed-int(indices_seed*
												n_seed),
												start_seed)
							elif isinstance(indices_seed,int):
								indices_seed = np.arange(
													start_seed-indices_seed,
													start_seed)
						else:
							indices_seed = np.arange(n_seed)

						indices_seed[-1] = min(indices_seed[-1],
												n_seed-1)
						shape_seed[axis_seed] = np.size(indices_seed)

						data_seed = np.reshape(
											np.take(v,indices_seed,axis_seed),
											shape_seed)
						
						k_t = '%s_%d'%(str(t),i)
						data_params['data_sets'].append(k_t)
						data_keys[t].append(k_t)
						data[k_t] = copy.deepcopy(data_seed)

						if dicttype_seed:
							data_typed[t][k_t] = copy.deepcopy(data_seed)
							if d.get('remove_seed'):
								data_typed[t_seed][k] = np.delete(v,
														indices_seed,axis_seed)
								data[k] = np.delete(v,indices_seed,axis_seed)
						else:
							data_typed[t][i] = copy.deepcopy(data_seed)
							if d.get('remove_seed'):
								data_typed[t_seed][i] = np.delete(v,
														indices_seed,axis_seed)
								data[k] = np.delete(v,indices_seed,axis_seed)

					for t_other in other_types:
						indices_other = copy.copy(indices_seed)
						for j,k_other in enumerate(data_typed[t_other]):
							if dicttype_seed:
								v_other = data_typed[t_other][k_other]
							else:
								v_other = k_other

							if np.shape(v_other)[axis_seed] == n_seed:
								if dicttype_seed:
									data_typed[t_other][k_other] = (
												np.delete(v_other,
												indices_other,axis_seed))
									data[k_other] = np.delete(v_other,
												indices_other,axis_seed)
								else:
									data_typed[t_other][j] = (
												np.delete(v_other,
												indices_other,axis_seed))
									data[k_other] = np.delete(v_other,
												indices_other,axis_seed)
							else:
								try:
									indices_other[-1] = min(
											np.shape(v_other)[axis_seed]-1,
											n_seed-1)
									if dicttype_seed:
										data_typed[t_other][k_other] = (
												np.take(v_other,
												indices_other,axis_seed))
										data[k_other] = np.take(v_other,
												indices_other,axis_seed)
									else:
										data_typed[t_other][j] = np.take(
											v_other,indices_other,axis_seed)
										data[k_other] = np.take(
											v_other,indices_other,axis_seed)
								except IndexError as e:
									indices_other = np.arange(
												np.size(indices_other))
									if dicttype_seed:
										data_typed[t_other][k_other] = (
												np.take(v_other,
												indices_other,axis_seed))
										data[k_other] = np.take(v_other,
											indices_other,axis_seed)
									else:
										data_typed[t_other][j] = np.take(
											v_other,indices_other,axis_seed)
										data[k_other] = np.take(v_other,
												indices_other,axis_seed)


						
							

		# Size of Data Sets
		data_sizes = {}
		for k in data.keys():
			v_shape = np.shape(data[k])
			if data_lists and (np.size(v_shape)==1):
				data[k] = np.reshape(data[k],(v_shape[0],1))
			data_sizes[k] = np.shape(data[k])
		delim_check(data_sizes,delim)
		
		# If not NP_FILE format, Export as NP_FILE for easier importing
		for k,f in format.items():
			for t,v in data_keys.items():
				if upconvert and f in ['npy','txt'] and k in v and (
					   data_params.get('data_obj_format',{}).get(t,'dict') == 
					   									'array'):
					self.exporter({k:data[k]},data_params,format=self.NP_FILE)
		
		return data, data_sizes, data_typed, data_keys




	# Export Data
	def exporter(self,data,
				 data_params={'data_dir':'dataset/','data_file':None},
				 export=True, label = '', directory=None, file=None,
				 disp=None,format = None,read_write='w',delim=None):
	   
	   
		if not export:
			return	
	   
	   # Display
		if disp is None:
			disp = self.DISP
	   
		# Bad delimimeters
		if delim is None:
			delim = data_params.get('delim',self.DELIM)
	   
		# Label
		if not label in [None,'']:
			label = '_' + str_check(label)
		else:
			label = ''
	   
	   # Check if Data is dict type
		data = dict_check(data.copy(),'') 
		delim_check(data,delim)
		
		# Data Directory
		if directory is None:
			directory = data_params.get('data_dir',self.DIRECTORY)
		
		if not os.path.isdir(directory):
			os.makedirs(directory)
			
		# Data Names
		if file is None:
			if not data_params.get('data_file'):
				file = lambda value: label+value
			else:
				file = data_params.get('data_file')
				
		if not callable(file):
			g = file
			file  = lambda k: g + label + k
		
		# Data Format
		
		if not isinstance(format,dict):
			format_files = {}
			for k in data.keys():
				if format is None:
					if isinstance(data_params.get('data_format',None),dict):
						
						dtype = [t for t in data_params['data_format'].keys() 
									if k in t or t in k]
						dtype = dtype[0] if dtype != [] else None
						format_files[k] = data_params['data_format'].get(
											 dtype,self.NP_FILE).split('.')[-1]
					else:
						format_files[k] = data_params.get('data_format',
												  self.NP_FILE).split('.')[-1]
				else:
					format_files[k] = format.split('.')[-1]
			format = format_files
			
		   

		# Write Data to File, Ensure no Overwriting
		for k,v in data.items():
			
			i = 0
			file_end = ''
			file_k = file(k).split('.')[0].replace(' ','')
			while os.path.isfile(directory+file_k+
								 file_end+ '.' +format[k]):
				file_end = '_%d'%i
				i+=1
			
			file_path = directory+file_k
			
			display(print_it=disp,time_it=False,
					m = 'Exporting %s'%(k))
			
			if not isinstance(v,np.ndarray):
				if v == {} or v == []:
					return
				else:
					v = np.asarray(v)
						
			
						
			if format[k] == 'npy':
				try:
					if read_write == 'a' and i>0:
						v0 = np.load(file_path+ '.' + 'npy')
						file_end = ''
						np.save(file_path+file_end+'.'+'npy',np.array([v0,v]))
					elif read_write == 'ow':
						read_write = 'w'
						file_end = ''
						np.save(file_path+file_end+'.'+'npy',v)
					else: 
						np.save(file_path+file_end+'.'+'npy',v)
				except MemoryError:
					v = np.array(v)
					format[k] = 'npz'
					
			if format[k] == 'npz':
				try:
					if read_write == 'a' and i>0:
						v0 = np.load(file_path+ '.' + 'npz')['arr_0']
						file_end = ''
						np.savez_compressed(file_path+file_end+'.'+'npz',
											np.array([v0,v]))
					elif read_write == 'ow':
						read_write = 'w'
						file_end = ''
						np.savez_compressed(file_path+file_end+'.'+'npz',v)
					else: 
						np.savez_compressed(file_path+file_end+'.'+'npz',v)
				except MemoryError:
					print('Memory Error - File %s Not Saved'%file_path)
					return
			if format[k] == 'txt':
				try:
					if read_write == 'a':
						file_end = ''
					elif read_write == 'ow':
						read_write = 'w'
						file_end = ''
					with open(file_path+file_end+'.'+'txt',read_write) as f:
						if isinstance(v,dict):
							for key in sorted(
								list(v.keys()),
								key=lambda x: (len(x),len(str(v[x])),
																str.lower(x))):
								
								f.write('%s:  %s \n \n \n'%(str(key),
																   str(v[key])))
						else:
							f.write(str(v))
				except MemoryError:
					print('Memory Error - File %s Not Saved'%file_path)	
					return
			
		return
	
	def format(self,data_params,file_format=None,directory=None,
					file_update=False,initials=True):
				
		
		# Data File Format
		if file_format is None:
			file_format = data_params.get('data_name_format',[None])
		else:
			file_update = True
		file_format = np.atleast_1d(file_format)
		
		file_header = caps(str_check(data_params.get(file_format[0],
													 file_format[0])))
		
		file_footer = caps(str_check(data_params.get(file_format[-1],
													 file_format[-1])))
		
		if initials:
			inds = 0
		else:
			inds = slice(0,None)
		
		# Format Data Directory
		if directory is None:
			directory = data_params.get('data_dir',self.DIRECTORY)
		
		# Format Data File
		if not data_params.get('data_file') or file_update:
			data_params['data_file'] = file_header
			
			for i,w in enumerate(file_format[1:-1]):
				if not(i==0 and file_header == ''):
					data_params['data_file'] += '_'
				data_params['data_file'] += str_check(w)[inds] + (
										str_check(data_params.get(w,w)))
			
			if file_footer not in ['',None]:
				data_params['data_file'] += '_'+file_footer 
			
			for c,d in [('.','f'),(', ',''),('f ','f0')]:
				data_params['data_file'] = data_params['data_file'].replace(c,d)
		
			data_params['data_file'] = data_params['data_file'].replace(' ','')
		
		return
			
			
			
		
        
        
                
        
        

			
		
        
        
                
        
        
