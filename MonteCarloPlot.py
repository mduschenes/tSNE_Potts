"""
Created on Sat May 12 19:14:39 2018
@author: Matt
"""
import numpy as np
from data_functions import Data_Process
from misc_functions import flatten,array_dict, caps, display

# Plotting for Monte Carlo
class MonteCarloPlot(object):
	
	
	def __init__(self,model_keys,model_props,*args):

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
				plot_keys[K] = [[(k,t) for t in args[0]] for k in V[1:]]
			elif K == 'observables_mean':			
				plot_keys[K] = [[k for k in model_keys['observables'][1:]]]
			else:
				plot_keys[K] = [[k for k in V[1:]]]
			plot_bool[K] = V[0]
		
		# Define Plotting Instance
		Data_Process().plot_close()
		self.plot_obj = Data_Process(plot_keys,plot_bool)
		self.plot_save = self.plot_obj.plot_save
		self.plot_keys = plot_keys
	
			
		return


	# Define plotting function
	def MC_plotter(self,data,*args):

		for K in data.keys():
			if K == 'configurations':
				self.plot_obj.plotter(
							data = {kt: data[K][kt[0]] 
								 for kt in flatten(self.plot_keys[K]) 
								 if kt[1] in args[0]},
							plot_props = self.MC_plot_props(K,self.plot_keys[K],
														*args[2:]),
							 data_key = K)
			elif K == 'observables':
				self.plot_obj.plotter(
							data = {k: {(a,t): data[K][ia][k][it] 
								for it,t in enumerate(args[0])
								for ia,a in enumerate(args[1])}
								for k in flatten(self.plot_keys[K])},       
							plot_props = self.MC_plot_props(K,self.plot_keys[K],
														*args[2:]),
							data_key = K)
														   
			elif K == 'observables_mean': 
				self.plot_obj.plotter(
					data = {k: {a: data['observables'][ia][k] 
							for ia,a in enumerate(args[1])}
							for k in flatten(self.plot_keys['observables'])},
					domain = {k: {a: args[0] 
							for ia,a in enumerate(args[1])}
							for k in flatten(self.plot_keys['observables'])},
					plot_props = self.MC_plot_props(K,self.plot_keys[K],
														*args[2:]),
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
	def MC_plot_props(self,plot_type,plot_keys,*args):

		# Function plot sites or clusters of sites
		def sites_region(sites):
			sites0 = np.asarray(sites,dtype=float)
			sites0[np.isin(sites,self.model_props['state_range'],
				   invert=True)]=np.nan
			return sites0
			# if  np.array_equiv(sites,sites0):
				# return sites0
			# else:
				# region = np.nan*np.ones(np.shape(sites0))
				# region[sites] = np.copy(sites0[sites])
				# return region

		def sup_title(label):
			return label +  ' - %s - $q = %d$ \n $T =  %s$ '%(
				caps(self.model_props['model_name']),
				self.model_props['q'] + (1 if 
					self.model_props['model_name']=='ising' else 0),
				str(self.model_props['T'])) + '\n'\
					'$N_{eqb} = %d \hspace{1cm} N_{meas} = %d \hspace{1cm}'\
					'N_{meas_{freq}} = %d$'%(
					self.model_props['update_props']['Neqb']/(
												   self.model_props['N_sites']), 
					self.model_props['update_props']['Nmeas']/(
												   self.model_props['N_sites']),
					self.model_props['update_props']['Nmeas_f']/(
												   self.model_props['N_sites']))

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
								
					  'other': {'label': lambda x='':x,
								'cbar_plot':False,
								'cbar_title':'Spin Values',
								'cbar_color':'bone',
								'cbar_color_bad':'magenta',
								'pause':0.01,
								'sup_legend': False,
								'sup_title': {'t':
											sup_title('Monte Carlo %s Updates'%(
												self.model_props['algorithm']))}
								}
					 }
					for k in keys}
					  
					  
			# Set Varying Properties                  
			def set_prop(props,key,func,*args):
				for k in props.keys():
					props[k][key[0]][key[1]] = func(k,*args)
				return
			 
				
			# Properties Specific to Plot
			plot_props_sites = {'title': '', 'xlabel': '', 'ylabel': ''}
				
				
			def plot_title(k,*args):
				if k[0] != plot_keys[0][0][0]:
					return plot_props_sites['title']
				else:
					return 'T = %0.2f'%k[1]
				
			def plot_ylabel(k,*args):
				if k[1] != plot_keys[0][0][1]:
					return plot_props_sites['ylabel']
				else:
					return caps(k[0])
						   
				
			def plot_xlabel(k,*args):
				if k[0] != plot_keys[-1][-1][0]:
					return plot_props_sites['xlabel']
				else:
					return r'$t_{MC}$: %d'%args[0] 
				
			
			def cbar_plot(k,*args):
				if k[1] == plot_keys[0][-1][1]:
					return True
				else:
					return False 

			def data_process(k,*args):
				
				data_plot_shape = [int(np.power(self.model_props['N_sites'],
							1/self.model_props['d']))]*self.model_props['d']
				
				return lambda d: np.reshape(sites_region(d),
											data_plot_shape)
			
			def plot_range(k,*args):
				return np.append(self.model_props['state_range'],
								 self.model_props['state_range'][-1]+1)    
			
			
			

			plot_props = Plot_Props(flatten(plot_keys))
			
			set_prop(plot_props,['ax','title'],plot_title)
			set_prop(plot_props,['ax','xlabel'],plot_xlabel,*args)
			set_prop(plot_props,['ax','ylabel'],plot_ylabel)
			set_prop(plot_props,['data','data_process'],data_process)
			set_prop(plot_props,['data','plot_range'],plot_range)
			set_prop(plot_props,['other','cbar_plot'],cbar_plot)
				
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
								
					  'other': {'label': lambda x='':x,
								'sup_legend': True,
								'sup_title': {'t':
											sup_title('Observables Histogram')},
								'pause':0.01}
					 }
					for k in keys}
			 
			
			
		   
			

					  
			# Set Varying Properties                  
			def set_prop(props,key,func,*args):
				for k in props.keys():
					props[k][key[0]][key[1]] = func(k,*args)
				return
			
			
			def plot_title(k,*args):
				return ''
				
			def plot_ylabel(k,*args):
				if k != plot_keys[0][0]:
					return ''
				else:
					return 'Counts'
				
			def plot_xlabel(k,*args):
				return caps(k,every_word=True,sep_char=' ',split_char='_')
			
			def plot_label(k,*args):
				return lambda k: 'T = %0.2f   %s'%(k[1],caps(k[0]))                                             

			
			
			
			plot_props = Plot_Props(flatten(plot_keys))
			
			set_prop(plot_props,['ax','title'],plot_title)
			set_prop(plot_props,['ax','xlabel'],plot_xlabel,*args)
			set_prop(plot_props,['ax','ylabel'],plot_ylabel)
			set_prop(plot_props,['other','label'],plot_label)
			

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
								
					  'other': {'label': lambda x='':x,'pause':0.01,
								 'sup_legend': True,
								'sup_title': {'t':
											sup_title('Observables')}
								}
					 }
					for k in keys}
			 
			
			
		   
			

					  
			# Set Varying Properties                  
			def set_prop(props,key,func,*args):
				for k in props.keys():
					props[k][key[0]][key[1]] = func(k,*args)
				return
			
			
			def plot_title(k,*args):
				return caps(k,every_word=True,sep_char=' ',split_char='_')
				
			def plot_ylabel(k,*args):
				return ''
				
			def plot_xlabel(k,*args):
				return 'Temperature'
			
			def plot_label(k,*args):
				return lambda k: caps(k)

			
			def data_process(k,*args):
				if k == 'order':
					return lambda x:  np.mean(np.abs(x),axis=-1)
				else:
					return lambda x:  np.mean(x,axis=-1)
			
			
			
			plot_props = Plot_Props(flatten(plot_keys))
			
			set_prop(plot_props,['ax','title'],plot_title)
			set_prop(plot_props,['ax','xlabel'],plot_xlabel,*args)
			set_prop(plot_props,['ax','ylabel'],plot_ylabel)
			set_prop(plot_props,['other','label'],plot_label)
			set_prop(plot_props,['data','data_process'],data_process)
			

			return plot_props

			
			
			
			
			
			
			


							  