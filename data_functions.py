# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 19:54:38 2018
@author: Matt
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os,glob,copy

from misc_functions import (flatten,dict_check, one_hot,caps,
							list_sort,str_check,delim_check)
import plot_functions

NP_FILE = 'npz'
IMG_FILE = 'pdf'
DIRECTORY = 'dataset/'
DELIM = [' ','.']

class Data_Process(object):
    
    # Create figure and axes dictionaries for dataset keys
	def __init__(self,keys=[None],plot=[False],
				 np_file = None,img_file=None,directory=None,delim=None):
		 
		# Standardized numpy file format
		for f,F in [(v,V) for v,V in locals().items() 
					if v not in ['keys','plot']]:
			F = globals().get(f.upper()) if  F is None else F
			setattr(self,f.upper(),F)
			
		# Initialize figures and axes dictionaries
		if not None in keys:
			self.figs = {}
			self.axes = {}
			
			self.keys = keys
			self.plot = plot
			self.data_key = 'data_key'
			
			# Create Figures and Axes with keys
			self.figures_axes(keys)

		return  
    


     # Plot Data by keyword
	def plotter(self,data,domain=None,plot_props={},data_key=None,keys=None):

		if not self.plot.get(data_key,True):
			return

		# Ensure Data, Domain and Keys are Dictionaries

		self.plot[data_key] = True

		plot_key = ''
		if not isinstance(data,dict):
			data = {plot_key: data}

		if data_key is None:
			data_key = self.data_key



		if keys is None:
			keys = [k for k in data.keys() if k in self.axes.get(data_key,
														   data.keys())
									   or k == plot_key]

		keys = [k for k in keys if data.get(k,[]) != []]


		dom = {}
		for k in keys:    
			if not isinstance(domain,dict) and isinstance(data[k],dict):
				dom[k] =  {ki: domain for ki in data[k].keys()}
			elif not isinstance(domain,dict):
				dom[k] = domain
			elif isinstance(data[k],dict):
				if isinstance(domain[k],dict):
					dom[k] = {ki: domain[k][ki] for ki in data[k].keys()}
				else:
					dom[k] = {ki: domain[k] for ki in data[k].keys()}
			else:
				dom = domain

		domain = dom
			


		# Create Figures and Axes
		self.figures_axes({data_key:keys})            

		#            display(m='Figures Created')
		# Plot for each data key
		for key in keys:
			props = plot_props.get(key,plot_props)

			try:
				ax = self.axes[data_key][key]
				fig = self.figs[data_key][key]
				plt.figure(fig.number)
				fig.sca(ax)
			except:
				self.figures_axes({data_key:keys})
				
				ax = self.axes[data_key][key]
				fig = self.figs[data_key][key]
				plt.figure(fig.number)
				fig.sca(ax)

			# Plot Data
			try:
				getattr(plot_functions,'plot_' + props.get('data',{}).get(
						'plot_type','plot'))(data[key],domain[key],fig,ax,props)

			except AttributeError:
				props.get('data',{}).get('plot_type')(
							   data[key],domain[key],fig,ax,props)

				display(m='Figure %s Created'%(
										plot_props[key]['data']['plot_type']))

										
										
			plt.suptitle(**plot_props[key].get(
										'other',{}).get('sup_title',{}))
			if plot_props[key].get('other',{}).get('sup_legend'):
				fig.legend(*(list_sort(ax.get_legend_handles_labels(),1)))

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
					   fig_keys = None,directory=None,
					   label = '',read_write='w'):
        
        # Save Figures for current Data_Process Instance
		
		format = data_params.get('figure_format',self.IMG_FILE)
		# Data Directory
		if directory is None:
			directory = data_params.get('data_dir','dataset/')

		if not os.path.isdir(directory):
			os.mkdirs(directory)
		
		
		# Check for specific fig_keys to save
		if fig_keys is None:
			fig_keys = self.figs.keys()
		
		for fig_k,fig_i in [(k,f) for k,f in self.figs.items()if k in fig_keys]:
			
			for ifig in set([f.number for f in fig_i.values()]):
			
				# Set Current Figure
				plt.figure(ifig)        
				fig = plt.gcf()
				
				# Change Plot Size for Saving                
				plot_size = fig.get_size_inches()
				fig.set_size_inches((8.5, 11))

				# Set File Name and ensure no Overwriting
				file = ''.join([directory,
								data_params.get('data_file',''),
								label,'_',fig_k])
								# ,'_',
								# '_'.join(set([k if isinstance(k,str) 
										  # else '' 
										  # for k in sorted(fig_i.keys()) 
										  # if fig_i[k].number == ifig]))])
				
		
				i = 0
				file_end = ''
				while os.path.isfile(file + file_end + '.'+format):
					file_end = '_%d'%i
					i+=1

				# Save Figure as File_Format
				if read_write == 'w': 
					if i > 0: return 
					else: file_end = ''
				plt.savefig(file + file_end+'.'+format,
							bbox_inches='tight',dpi=500)
				fig.set_size_inches(plot_size) 
			
		return
    
      
    
    
	 # Create figures and axes for each passed set of keys for datasets
	def figures_axes(self,Keys):     
		
		if not isinstance(Keys,dict):
			Keys = {self.data_key: Keys}
			   
		for keys_label,keys in Keys.items():

			keys_new = [k if k not in self.axes.get(keys_label,{}).keys() 
						else None for k in flatten(keys,False)]
			
			if not None in keys_new and self.plot[keys_label]:
				
				if len(self.axes.get(keys_label,{})) > 1: 
					print('Figure Keys Updated...')

				self.axes[keys_label] = {}
				self.figs[keys_label] = {}

				fig, ax = plt.subplots(*(np.shape(keys)[:2]))
				
				fig.canvas.set_window_title('Figure: %d  %s Datasets'%(
												  fig.number,caps(keys_label)))
				for k,a in zip(keys_new,flatten(np.atleast_1d(ax).tolist())):
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
					 'one_hot': False
					},directory=None,format=None,upconvert=False,delim=None):

		# Bad delimeters
		if delim is None:
			delim = data_params.get('delim',self.DELIM)
					
		# Data Files Dictionary
		# Check of importing batch of files
		data_params = dict_check(data_params,'data_files')            

		if directory is None:
			directory = data_params.get('data_dir',self.DIRECTORY)
		
		if isinstance(data_params['data_files'],(tuple,str)):
			files = []
			format_files = {}
			for f in np.atleast_1d(data_params['data_files']):
				if '*' in f:
					ftemp = [os.path.basename(x) 
								for x in glob.glob(directory+f)]
				else:
					ftemp = [x for x in os.listdir(
									     directory) 
										 if os.isfile(os.join(
													directory, x)) 
										 and data_params['data_files'] in d ]
				
				files.extend(ftemp)
				
				for fi in ftemp:
					format_files[fi] = fi.split('.')[-1].replace('*','')
			
			data_params['data_files'] = files
		else: 
			format_files = {}
						
				
		if not data_params.get('data_sets'):
			data_params['data_sets'] = data_params['data_files']
		
			
		data_params['data_files'] = dict_check(
					data_params['data_files'],data_params['data_sets'])
		
		
		if format is None:
			format = {k: format_files.get(f,data_params.get('data_format',
															self.NP_FILE))
								for k,f in data_params['data_files'].items()}
		else:
			format = {k:format for k in data_params['data_files'].keys()}
		
		# Import Data				
		import_func = {}
		formatter = lambda s,f: s.split('.')[0] + '.'+f.split('.')[-1]
		import_func['values'] = lambda v:v
		import_func['npy'] = lambda v: np.load(directory+ formatter(v,'npy'))
		import_func['npz'] = lambda v: np.load(
										  directory+formatter(v,'npz'))['arr_0']
		import_func['txt'] = lambda v: np.loadtxt(directory+formatter(v,'txt'))
									  
		
		#print(data_params['data_files'])
		
		data = {k: import_func[format[k]](v)
					for k,v in data_params['data_files'].items() 
					if v is not None}
						
		# Ensure Files have no spaces	
		
		delim_check(data,delim)
		delim_check(format,delim)
		delim_check(data_params['data_sets'],delim)
		
		# Convert Labels to one-hot Labels
		if data_params.get('one_hot'):
			for k in data.keys(): 
				if 'y_' in k:
					data[k] = one_hot(data[k])
		
		
		# Size of Data Sets
		data_sizes = {}
		for k in data.keys():
			data[k] = np.atleast_2d(data[k])
			v_shape = np.shape(data[k])
			if data_params.get('data_lists') and (v_shape[0] == 1) and (
			   v_shape[1:] != 1) and (np.size(v_shape)<=2):
				data[k] = np.transpose(data[k])
			data_sizes[k] = np.shape(data[k])
		delim_check(data_sizes,delim)
		
		# Type of Data Sets
		if not data_params.get('data_types'):
			data_params['data_types'] = list(data.keys())
			data_typed = data.copy()
			
		elif data_params.get('data_typed','dict_split') == 'dict':
			data_typed = {t: {k: data[k].copy() 
                          for k in data.keys() if t in k}
                          for t in data_params['data_types']}
		
		
		elif data_params.get('data_typed','dict_split') == 'dict_split':
			def process(key,data,dtype):
				if data_params.get('data_formats',{}).get(dtype) is dict and (
					'np' in format[key]):
					return key.split(dtype)[0],data[0][0]
				else:
					return key.split(dtype)[0],data
			data_typed = {}
			for t in data_params['data_types']:
				data_typed[t] = {}
				for k in data.keys():
					if t in k:
						k,d = process(k,data[k].copy(),t)
						data_typed[t][k] = d 
		
		else:
			data_typed = {t: [data[k].copy() 
                          for k in data.keys() if t in k]
                          for t in data_params['data_types']}
						  
		data_keys =   {t: [k 
					  for k in data.keys() if t in k]
					  for t in data_params['data_types']}
		
		for t in data_typed.keys():
			delim_check(data_typed[t],delim)
			delim_check(data_keys[t],delim) 
		
		
		# If not NP_FILE format, Export as NP_FILE for easier importing
		for k,f in format.items():
			for t,v in data_keys.items():
				if upconvert and f in ['npy','txt'] and k in v and (
					   data_params.get('data_formats',{}).get(t) is np.ndarray):
					self.exporter({k:data[k]},data_params,format=self.NP_FILE)
		
		return data, data_sizes, data_typed, data_keys




	# Export Data
	def exporter(self,data,
				 data_params={'data_dir':'dataset/','data_file':None},
				 export=True, label = '', directory=None,
				  format = None,read_write='w',delim=None):
	   
	   
		if not export:
			return	
	   
		# Bad delimimeters
		if delim is None:
			delim = data_params.get('delim',self.DELIM)
	   
	   # Check if Data is dict type
		data = dict_check(data,'') 
		delim_check(data,delim)
		
		# Data Directory
		if directory is None:
			directory = data_params.get('data_dir',self.DIRECTORY)
		
		if not os.path.isdir(directory):
			os.mkdirs(directory)
			
		# Data Names
		if not data_params.get('data_file'):
			file = lambda value: value
		
		elif not callable(data_params['data_file']):
			g = data_params['data_file']
			file  = lambda k: g + k
		
		# Data Format
		
		if not isinstance(format,dict):
			format_files = {}
			for k in data.keys():
				if format is None:
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
			while os.path.isfile(directory+file_k+label+
								 file_end+ '.' +format[k]):
				file_end = '_%d'%i
				i+=1
			
			file_path = directory+file_k+label
			
			if format[k] == 'npy':
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
			
			elif format[k] == 'npz':
				if read_write == 'a' and i>0:
					v0 = np.load(file_path+ '.' + 'npz')
					file_end = ''
					np.savez_compressed(file_path+file_end+'.'+'npz',
										np.array([v0,v]))
				elif read_write == 'ow':
					read_write = 'w'
					file_end = ''
					np.savez_compressed(file_path+file_end+'.'+'npz',v)
				else: 
					np.savez_compressed(file_path+file_end+'.'+'npz',v)
			
			elif format[k] == 'txt':
				if read_write == 'a':
					file_end = ''
				elif read_write == 'ow':
					read_write = 'w'
					file_end = ''
				with open(file_path+file_end+'.'+'txt',read_write) as file_txt:
					if isinstance(v,dict):
						for key in sorted(
							list(v.keys()),
							key=lambda x: (len(x),len(str(v[x])),str.lower(x))):
							
							file_txt.write('%s:  %s \n \n \n'%(str(key),
															   str(v[key])))
					else:
						file_txt.write(str(v))
			
			
		return
	
	def format(self,data_params,directory=None):
				
		
		# Data File Format
		if not data_params.get('data_file_format'):
			data_params['data_file_format'] = []
		
		file_format = np.atleast_1d(data_params['data_file_format'])
			
		file_header = caps(str_check(data_params.get(file_format[0],
													 file_format[0])))
		
		file_footer = caps(str_check(data_params.get(file_format[-1],
													 file_format[-1])))
		
		# Format Data Directory
		if directory is None:
			directory = data_params.get('data_dir',self.DIRECTORY)
		
		# Format Data File
		if not data_params.get('data_file'):
			data_params['data_file'] = file_header
			
			for w in data_params['data_file_format'][1:-1]:
				data_params['data_file'] += '_' + str_check(w)[0] + (
										str_check(data_params.get(w,w)))
			
			if file_footer not in ['',None]:
				data_params['data_file'] += '_'+file_footer 
			
			for c,d in [('.','f'),(', ',''),('f ','f0')]:
				data_params['data_file'] = data_params['data_file'].replace(c,d)
		
			data_params['data_file'] = data_params['data_file'].replace(' ','')
		
		return
			
			
			
		
        
        
                
        
        
