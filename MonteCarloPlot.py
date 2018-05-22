"""
Created on Sat May 12 19:14:39 2018
@author: Matt
"""
import numpy as np
from data_functions import Data_Process
from misc_functions import flatten,array_dict, caps, display,str_check

# Plotting for Monte Carlo
class MonteCarloPlot(object):
	
	
	def __init__(self,model_keys,model_props,**kwargs):

		# Define Configurations and Observations Plot Shape Keys
		# Subplots are plotted based on the key passed associated with each
		# data set.
		# keys = {K: [plot_bool, observables_0,..., obserservables_n]}
		
		# Define model properties
		self.model_props = model_props
		
		# Define plotting keys
		plot_keys = {}
		plot_bool = {}
		for K,V in model_keys.items():
			if K == 'configurations':
				plot_keys[K] =[[(k,t) for t in kwargs['arr_0'][1]] 
								for k in V[1:]]
			else:
				plot_keys[K] = [[k for k in V[1:]]]
			plot_bool[K] = V[0]
		
		
		# Define Plotting Instance
		Data_Process().plot_close()
		self.plot_obj = Data_Process(plot_keys,plot_bool,backend = 'Agg')
		self.plot_save = self.plot_obj.plot_save
		self.plot_keys = plot_keys
		
		return


	# Define plotting function
	def MC_plotter(self,data,**kwargs):
		for K in data.keys():
		
			if not self.plot_obj.plot.get(K,False):
				return
				
			if K == 'configurations':
				self.plot_obj.plotter(
							data = {kt: data[K][kt[0]] 
								 for kt in flatten(self.plot_keys[K]) 
								 if kt[1] in kwargs['arr_0'][1]},
							plot_props = self.MC_plot_props(K,self.plot_keys[K],
														**kwargs),
							 data_key = K)
			elif K == 'observables':
				self.plot_obj.plotter(
							data = {k: {(a1,a0): data[K][ia1][k][ia0] 
								for ia0,a0 in enumerate(
											  np.atleast_1d(kwargs['arr_0'][1]))
								for ia1,a1 in enumerate(
											  np.atleast_1d(kwargs['arr_1'][1]))
									   }
								for k in flatten(self.plot_keys[K])},       
							plot_props = self.MC_plot_props(K,self.plot_keys[K],
														**kwargs),
							data_key = K)
														   
			elif K == 'observables_mean': 
				self.plot_obj.plotter(
					data = {k: {a: data['observables'][ia][k] 
							for ia,a in enumerate(kwargs['arr_1'][1])}
							for k in flatten(self.plot_keys[K])},
					domain = {k: {a1: np.atleast_1d(kwargs['arr_0'][1])
							for ia1,a1 in enumerate(kwargs['arr_1'][1])}
							for k in flatten(self.plot_keys[K])},
					plot_props = self.MC_plot_props(K,self.plot_keys[K],
														**kwargs),
							data_key = K)
			
			elif K == 'observables_sorted':
				self.plot_obj.plotter(
							data = {k: {(a1,a0): data[K][a1][a0][k] 
								for ia0,a0 in enumerate(
										 np.atleast_1d(kwargs['arr_0'][1][a1]))
								for ia1,a1 in enumerate(
										np.atleast_1d(kwargs['arr_1'][1]))
									   }
								for k in flatten(self.plot_keys['observables'])},       
							plot_props = self.MC_plot_props(
												  'observables',
												  self.plot_keys['observables'],
												  **kwargs),
							data_key = K)
			
			elif K == 'observables_mean_sorted': 
				self.plot_obj.plotter(
					data = {k: {a: [data['observables_sorted'][a1][a0][k] 
							for a0 in  sorted(np.atleast_1d(
													kwargs['arr_0'][1][a1]))]
							for ia1,a1 in enumerate(kwargs['arr_1'][1])}
							for k in flatten(
									self.plot_keys['observables_mean'])},
					domain = {k: {a1: sorted(np.atleast_1d(
													kwargs['arr_0'][1][a1]))
							for ia1,a1 in enumerate(kwargs['arr_1'][1])}
							for k in flatten(
										self.plot_keys['observables_mean'])},
					plot_props = self.MC_plot_props('observables_mean',
									 self.plot_keys['observables_mean'],
									 **kwargs),
							data_key = K)
			

		return


 # props = {'ax':{'title':'Function of Dataset'},
         # 'ax_attr':{'xaxis': {'label_text':'Arg1','ticks_position': 'none'},
		 # 'get_xticklabels':{'visible':False,'fontsize':12},
                    # 'get_yticklabels':{'fontsize':20}},
         # 'plot':{'label':'DataPoint','marker':'*','linestyle':'--'}
         # }
# plt.setp(getattr(ax,'get_xticklabels')(),**{'visible':False,'fontsize':12});




	# Data type dependent plot properties keys
	def MC_plot_props(self,plot_type,plot_keys,**kwargs):
				
		# Function plot sites or clusters of sites
		def sites_region(sites):
			sites0 = np.asarray(sites,dtype=float)
			sites0[np.in1d(sites,self.model_props['state_range'],
				   invert=True)]=np.nan
			return sites0
			# if  np.array_equiv(sites,sites0):
				# return sites0
			# else:
				# region = np.nan*np.ones(np.shape(sites0))
				# region[sites] = np.copy(sites0[sites])
				# return region

		def sup_title(label,**kwargs):
			return kwargs.get('sup_title','')
			# caps(label,every_word=True,sep_char=' ',split_char='_') + (
				# ' - %s - $q = %s$ \n $T =  %s$ '%(
				# caps(self.model_props['model_name']),
				# str(self.model_props['q'] + (1 if 
					# self.model_props['model_name']=='ising' else 0)),
				# str(self.model_props['T']))) + '\n'\
					# '$N_{eqb} = %d \hspace{1cm} N_{meas} = %d \hspace{1cm}'\
					# 'N_{meas_{freq}} = %d$'%(
					# self.model_props['update_props']['Neqb'], 
					# self.model_props['update_props']['Nmeas'],
					# self.model_props['update_props']['Nmeas_f'])

		if plot_type == 'configurations':
			
			def Plot_Props(keys):
			
				return {
					 k: {
					
					  'ax':   {'title' : '', 
								'xlabel': '', 
								'ylabel': ''},
					  'ax_attr': {'get_xticklabels':{'visible':False,
													 'fontsize':12},
								  'xaxis': {'ticks_position': 'none'},
								  'get_yticklabels':{'visible':False,
													 'fontsize':12},
								  'yaxis': {'ticks_position': 'none'}},
					  'plot':  {'interpolation':'nearest'},
					  
					  'data':  {'plot_type':'image',
								'plot_range': '',
								'data_process':lambda data: np.real(data)},
								
					  'other': {'label': lambda x='':caps(str_check(x),
												   every_word=True,
												   sep_char=' ',split_char='_'),
								'cbar_plot':False,
								'cbar_title':'Spin Values',
								'cbar_color':'bone',
								'cbar_color_bad':'magenta',
								'pause':0.01,
								'sup_legend': False,
								'sup_title': {'t':
											sup_title('Monte Carlo %s Updates'%(
												self.model_props['algorithm']),
												**kwargs)}
								}
					 }
					for k in keys}
					  
					  
			# Set Varying Properties                  
			def set_prop(props,key,func,**kwargs):
				for k in props.keys():
					props[k][key[0]][key[1]] = func(k,**kwargs)
				return
			 
				
			# Properties Specific to Plot
			plot_props_sites = {'title': '', 'xlabel': '', 'ylabel': ''}
				
				
			def plot_title(k,**kwargs):
				if k[0] != plot_keys[0][0][0]:
					return plot_props_sites['title']
				else:
					return 'T = %0.2f'%k[1]
				
			def plot_ylabel(k,**kwargs):
				if k[1] != plot_keys[0][0][1]:
					return plot_props_sites['ylabel']
				else:
					return caps(k[0])
						   
				
			def plot_xlabel(k,**kwargs):
				if k[0] != plot_keys[-1][-1][0]:
					return plot_props_sites['xlabel']
				else:
					return r'$%s$: %d'%(kwargs['i_mc'][0],kwargs['i_mc'][1])
				
			
			def cbar_plot(k,**kwargs):
				if k[1] == plot_keys[0][-1][1]:
					return True
				else:
					return False 

			def data_process(k,**kwargs):
				
				data_plot_shape = [int(np.power(self.model_props['N_sites'],
							1/self.model_props['d']))]*self.model_props['d']
				
				return lambda d: np.reshape(sites_region(d),
											data_plot_shape)
			
			def plot_range(k,**kwargs):
				return np.append(self.model_props['state_range'],
								 self.model_props['state_range'][-1]+1)    
			
			
			

			plot_props = Plot_Props(flatten(plot_keys))
			
			set_prop(plot_props,['ax','title'],plot_title,**kwargs)
			set_prop(plot_props,['ax','xlabel'],plot_xlabel,**kwargs)
			set_prop(plot_props,['ax','ylabel'],plot_ylabel,**kwargs)
			set_prop(plot_props,['data','data_process'],data_process,**kwargs)
			set_prop(plot_props,['data','plot_range'],plot_range,**kwargs)
			set_prop(plot_props,['other','cbar_plot'],cbar_plot,**kwargs)
				
			return plot_props


		elif plot_type == 'observables':


			
			
			def Plot_Props(keys):
			
				return {
					 k: {
					
					  'ax':   {'title' : '', 
								'xlabel': '', 
								'ylabel': ''},
					  
					  'plot':  {'stacked':True, 'fill':True, 'alpha': 0.35, 
								'histtype':'bar'},
					  
					  'data':  {'plot_type':'histogram',
								'plot_range': '',
								'data_process':lambda data: np.real(data)},
								
					  'other': {'label': lambda x='':caps(str_check(x),
												   every_word=True,
												   sep_char=' ',split_char='_'),
								'sup_legend': True,
								'sup_title': {'t':
											sup_title('Observables Histogram')},
								'pause':0.01}
					 }
					for k in keys}
			 
			
			
		   
			

					  
			# Set Varying Properties                  
			def set_prop(props,key,func,**kwargs):
				for k in props.keys():
					props[k][key[0]][key[1]] = func(k,**kwargs)
				return
			
			
			def plot_title(k,**kwargs):
				return ''
				
			def plot_ylabel(k,**kwargs):
				if k != plot_keys[0][0]:
					return ''
				else:
					return 'Counts'
				
			def plot_xlabel(k,**kwargs):
				return caps(str_check(k),every_word=True,
							sep_char=' ',split_char='_')
			
			def plot_label(k,**kwargs):
				return lambda k: '%s = %0.2f,  %s = %s'%(kwargs['arr_0'][0],
														k[1],
														kwargs['arr_1'][0],
														caps(str_check(k[0]),
															every_word=True,
															sep_char=' ',
															split_char='_'))

			
			
			
			plot_props = Plot_Props(flatten(plot_keys))
			
			set_prop(plot_props,['ax','title'],plot_title,**kwargs)
			set_prop(plot_props,['ax','xlabel'],plot_xlabel,**kwargs)
			set_prop(plot_props,['ax','ylabel'],plot_ylabel,**kwargs)
			set_prop(plot_props,['other','label'],plot_label,**kwargs)
			

			return plot_props


		elif plot_type == 'observables_mean':


			
			
			def Plot_Props(keys):
			
				return {
					 k: {
					
					  'ax':   {'title' : '', 
								'xlabel': '', 
								'ylabel': ''},
					  
					  'plot':  {'marker':'*'},
					  
					  'data':  {'plot_type':'plot',
								'plot_range': '',
								'data_process':''
							   },
								
					  'other': {'label': lambda x='':caps(str_check(x),
											       every_word=True,
												   sep_char=' ',split_char='_'),
								'axis_ticks':{'x':{'lim':1,'ticksmax':1/2,
														   'ticksmin':1/4},
											  'y':{'lim':10,'ticksmax':None,
															 'ticksmin':None}},
								'pause':0.01,
								'sup_legend': True,
								'sup_title': {'t':
											sup_title('Observables')}
								}
					 }
					for k in keys}
			 
			
			
		   
			

					  
			# Set Varying Properties                  
			def set_prop(props,key,func,**kwargs):
				for k in props.keys():
					props[k][key[0]][key[1]] = func(k,**kwargs)
				return
			
			
			def plot_title(k,**kwargs):
				return caps(str_check(k),every_word=True,
							sep_char=' ',split_char='_')
				
			def plot_ylabel(k,**kwargs):
				return ''
				
			def plot_xlabel(k,**kwargs):
				return 'Temperature'
			
			def plot_label(k,**kwargs):
				return lambda k: '%s = %s'%(kwargs['arr_1'][0],
										    caps(str_check(k),every_word=True,
												sep_char=' ',split_char='_'))

			
			def data_process(k,**kwargs):
				if k == 'order':
					return lambda x:  np.mean(np.abs(x),axis=-1)
				else:
					return lambda x:  np.mean(x,axis=-1)
			
			
			
			plot_props = Plot_Props(flatten(plot_keys))
			
			set_prop(plot_props,['ax','title'],plot_title,**kwargs)
			set_prop(plot_props,['ax','xlabel'],plot_xlabel,**kwargs)
			set_prop(plot_props,['ax','ylabel'],plot_ylabel,**kwargs)
			set_prop(plot_props,['other','label'],plot_label,**kwargs)
			set_prop(plot_props,['data','data_process'],data_process,**kwargs)
			

			return plot_props

			
			
			
			
			
			
			


							  