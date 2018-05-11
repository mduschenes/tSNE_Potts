# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:18:39 2018
@author: Matt
"""

import numpy as np
import warnings,copy
warnings.filterwarnings("ignore")


from data_functions import Data_Process
from misc_functions import flatten,array_dict, caps, display


class MonteCarloUpdate(object):
    
	def __init__(self,model_props):

        # Perform Monte Carlo Updates for nclusters of sites and Plot Data
        
        #  Monte Carlo Update Parameters: Perform Monte Carlo Update Boolean
        #                                 Number of initial updates Neqb,
        #                                 Number of Measurement Updates Nmeas
        #                                 Measurement Update Frequency Nmeas_f
        #                                 Number of Clusters Ncluster
        # State Transition Parameters:
        #                                 Transition Probability prob_transiton
        #                                 Transition Values site_states
        # Observe: [Boolean to Animate [Sites, Observables]]


#        model_props = {'L': L, 'd': d, 'q': m.q, 'T': T,
#                            'state_range': m.state_range,
#                            'state_gen': m.state_gen,
#                            'prob_trans': {'metropolis': m.energy,
#                                           'wolff':m.model_params[
#                                                   m.model.__name__][
#                                                   'bond_prob']},
#                            'model': m.model.__name__,
#                            'algorithm': 'wolff',
#                            'algorithms': ['metropolis','wolff'],
#                            'update': update,
#                            'observe': observe,
#                            'observables': m.observables_functions,
#                            'observables_props': m.observables_props,
#                            'observables_mean': m.obs_mean,
#                            'data_type': np.int_,
#                            'data_save': data_save,
#                            'data_dir': '%s_Data/'%(caps(m.model.__name__)),
#                            'data_file': '%s_d%d_L%d__%s' %(
#                                          caps(m.model.__name__),d,L,
#                                          datetime.datetime.now().strftime(
#                                                           '%Y-%m-%d-%H-%M'))}





		# Define System Configuration
		self.sites = []
		self.cluster = []
		self.neighbour_sites = model_props['neighbour_sites']
		self.nearest_neighbours = self.neighbour_sites[0]


		self.Nspins = np.shape(self.neighbour_sites)[1]
		self.T = np.atleast_1d(model_props['T'])


		# Define Monte Carlo Update Parameters:
		# Number of updates to reach "equilibrium" before measurement, 
		# Number of measurement updates.
		update = model_props['update']
		self.mcupdate = update[0]
		self.Neqb = int(((1) if update[1]==None else update[1])*self.Nspins)
		self.Nmeas = int(((1) if update[2]==None else update[2])*self.Nspins)
		self.Nmeas_f = int(((1) if update[3]==None else update[3])*self.Nspins)
		self.Ncluster = update[4]

		self.state_int = model_props['state_int']
		self.state_gen = model_props['state_gen']


		# Define Configurations and Observables Data Dictionaries
		self.model_props = model_props       
		Data_Process().plot_close()
		self.plot_obj = None
		self.data = {}

		return



	# Perform Monte Carlo Update Algorithm and Plot Sites and Observables            
	def MC_update(self,props_iter={'algorithm':'wolff'},disp_updates=True):

		# Initialize props_iter as array of dictionaries
		props_iter,n_iter = array_dict(props_iter)

		# Initialize Plotting and Data for n_iterations
		self.data['sites'] = np.empty((n_iter,len(self.T),
		                               self.Nmeas//self.Nmeas_f,self.Nspins),
									   dtype=self.model_props['data_type'])   
		self.data['cluster'] = np.empty((n_iter,len(self.T),
		                                  self.Nmeas//self.Nmeas_f,self.Nspins),  
										 dtype=int)
		self.plot_obj = self.MC_plot(self.model_props['observe'],None,None,
									 self.T)

		display(disp_updates,False,
				'Monte Carlo Simulation... \n%s: q = %d \nT = %s'%(
					  (self.model_props['model'],self.model_props['q'],
					   str(self.T))) + '\nNeqb = %d, Nmeas = %d'%(
								 self.Neqb/self.Nspins,self.Nmeas/self.Nspins),
					   line_break=True)
					  
					  


		# Perform Monte Carlo Algorithm for n_iter configurations        
		for i_iter in range(n_iter):
				
			# Update dictionary and plotting for each Iteration
			self.model_props.update(props_iter[i_iter])

			
			# Initialize sites with random spin at each site for each Iteration
			self.sites = self.state_gen(self.Nspins)
			self.cluster = np.zeros(self.Nspins,dtype=np.intc)
			
			self.prob_update = self.model_props['prob_update'][
									   self.model_props['algorithm']]

			display(disp_updates,0,'Iter %d: %s Algorithm'%(
								   i_iter,caps(self.model_props['algorithm'])))

			MC_alg = getattr(self,self.model_props['algorithm'])

			# Perform Monte Carlo at temperatures t = T
			for i_t,t in enumerate(self.T):
						   
				# Perform Equilibration Monte Carlo steps initially
				for i_mc in range(self.Neqb):
					MC_alg(t)					
					
				# Perform Measurement Monte Carlo Steps
				for i_mc in range(self.Nmeas):

					MC_alg(t)                        
					 
					# Update Configurations and Observables
					if i_mc % self.Nmeas_f == 0:
						self.data['sites'][i_iter,i_t,i_mc//self.Nmeas_f,:] = (
														   np.copy(self.sites))
						
						self.data['cluster'][i_iter,i_t,i_mc//self.Nmeas_f,:]=(
														  np.copy(self.cluster))
						
						self.plot_obj = self.MC_plot({'configurations': 
								self.model_props['observe']['configurations']},
											    self.plot_obj,
												{'configurations':
													{'sites': self.sites,
													 'cluster':self.cluster}},
												*[[t],[],i_mc/self.Nspins])                
			  
				display(printit=disp_updates,m='Updates: T = %0.2f'%t)
				
			# Save Current Data
			if self.model_props['data_save']:
				self.plot_obj.plot_save(self.model_props,
								        label=self.model_props['algorithm'],
										fig_keys='configurations')
				Data_Process().exporter(self.data,self.model_props,
										self.model_props['algorithm'])  
		
			display(printit=disp_updates,
					m='Runtime: ',t0=-(len(self.T)+2),line_break=True)
				
				
		# Compute, Plot and Save Observables Data and Figures
		self.data['observables'] = self.MC_measurements(self.data['sites'],
											self.neighbour_sites,self.T,
											self.model_props['observables'])                                                   
			
			
		display(printit=disp_updates,m='Observables Calculated')

		self.plot_obj = self.MC_plot(self.model_props['observe'], self.plot_obj,
								{'observables': self.data['observables'],
								 'observables_mean': self.data['observables']},
								*[self.T,[p['algorithm']for p in props_iter],
								  i_mc/self.Nspins])

		display(printit=disp_updates,m='Figures Plotted')

		
		if self.model_props['data_save']:
			self.plot_obj.plot_save(self.model_props,
							        fig_keys=['observables','observables_mean'])
			Data_Process().exporter(self.data,self.model_props)  
											   

		return


	# Update Algorithms
	def metropolis(self,T=1):
		# Randomly alter random spin sites and accept spin alterations
		# if energetically favourable or probabilistically likely

		# Generate Random Spin Site and store previous Spin Value

		isite = [np.random.randint(self.Nspins) for j in 
				  range(self.Ncluster)]
		self.cluster = isite
		sites0 = self.sites[isite]

		# Update Spin Value
		self.sites[isite] = self.state_gen(self.Ncluster,sites0)

		# Calculate Change in Energy and decide to Accept/Reject Spin Flip
		nn = self.sites[self.neighbour_sites[0][isite]]
		dE = -np.sum(self.state_int(self.sites[isite],nn)) + (
			  np.sum(self.state_int(sites0,nn)))

		if dE > 0:
			if self.prob_update[dE,T] < np.random.random():
				self.sites[isite] = sites0                
		return


	def wolff(self,T=1):      
		# Create Cluster Array and Choose Random Site

		isite = np.random.randint(self.Nspins)

		self.cluster = np.zeros(self.Nspins,dtype=bool)
		self.cluster_value0 = self.sites[isite]

		# Perform cluster algorithm to find indices in cluster
		self.cluster_update(isite,T)

		# Flip spins in cluster to new value
		self.sites[self.cluster] = self.state_gen(1,self.cluster_value0)
		return

			   

	# Cluster Function
	def cluster_update(self,i,T):

		# Add indices to cluster
		self.cluster[i] = True
		J = (j for j in self.neighbour_sites[0][i] if (not self.cluster[j]) and 
										(self.sites[j] == self.cluster_value0)) 

		for j in J:
			if self.prob_update[T] > np.random.random():
				self.cluster_update(j,T)

		return



	def MC_measurements(self,sites,neighbours,T,observables):

		if sites.ndim < 4:
			sites = sites[np.newaxis,:]

		n_iter = np.shape(sites)[0]

		data = [{} for _ in range(n_iter)]

		for i_iter in range(n_iter):        
			for k,obs in observables.items():
				data[i_iter][k] = obs(sites[i_iter],neighbours,T)

		return data

	# Plotting
	def MC_plot(self,keys,plot_obj,data,*args):

		# Define Configurations and Observations Plot Shape Keys
		# Subplots are plotted based on the key passed associated with each
		# data set.
		# keys = {K: [plot_bool, observables_0,..., obserservables_n]}

		# Define plotting instance
		if plot_obj is None:
			plot_keys = {}
			plot_bool = {}
			for K,V in keys.items():
				if K == 'configurations':
					plot_keys[K] = [[(k,t) for t in args[0]] for k in V[1:]]
				elif K == 'observables_mean':			
					plot_keys[K] = [[k for k in keys['observables'][1:]]]
				else:
					plot_keys[K] = [[k for k in V[1:]]]
				plot_bool[K] = V[0]
			plot_obj = Data_Process(plot_keys,plot_bool)
			
		else:
			plot_keys = plot_obj.keys

		if data is None:
			return plot_obj
					   

		

		# Plot Data
		self.MC_plotter(plot_keys,plot_obj,data,*args)
			
		return plot_obj


	# Define plotting function
	def MC_plotter(self,plot_keys,plot_obj,data,*args):
		for K in data.keys():
			if K == 'configurations':
				plot_obj.plotter(
							 data = {kt: data[K][kt[0]] 
								 for kt in flatten(plot_keys[K]) 
								 if kt[1] in args[0]},
							 plot_props = self.MC_plot_props(K,plot_keys[K],
														*args[2:]),
							 data_key = K)
			elif K == 'observables':
				plot_obj.plotter(
							data = {k: {(a,t): data[K][ia][k][it] 
								for it,t in enumerate(args[0])
								for ia,a in enumerate(args[1])}
								for k in flatten(plot_keys[K])},       
							plot_props = self.MC_plot_props(K,plot_keys[K],
														*args[2:]),
							data_key = K)
														   
			elif K == 'observables_mean': 
				plot_obj.plotter(
							data = {k: {a: data['observables'][ia][k] 
								for ia,a in enumerate(args[1])}
								for k in flatten(plot_keys['observables'])},
							domain = {k: {a: args[0] 
								for ia,a in enumerate(args[1])}
								for k in flatten(plot_keys['observables'])},
							plot_props = self.MC_plot_props(K,plot_keys[K],
														*args[2:]),
							data_key = K)

		return


 # props = {'ax':{'title':'Function of Dataset'},
         # 'ax_attr':{'xaxis': {'label_text':'Arg1','ticks_position': 'none'},'get_xticklabels':{'visible':False,'fontsize':12},
                    # 'get_yticklabels':{'fontsize':20}},
         # 'plot':{'label':'DataPoint','marker':'*','linestyle':'--'}
         # }
# plt.setp(getattr(ax,'get_xticklabels')(),**{'visible':False,'fontsize':12});




	# Data type dependent plot properties keys
	def MC_plot_props(self,plot_type,plot_keys,*args):

		# Function plot sites or clusters of sites
		def sites_region(sites,sites0):
			if  np.array_equiv(sites,sites0):
				return sites0
			else:
				region = np.nan*np.ones(np.shape(sites0))
				region[sites] = np.copy(sites0[sites])
				return region


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
								'sup_title': {'t':'Monte Carlo Updates' 
											   + ' - '+ 
											caps(self.model_props['model'])+
											' - '+
											caps(
											  self.model_props['algorithm'])+
											'\n'\
									        'N_{eqb} = %d\ \ \ \    '\
											'N_{meas} = %d\ \ \ \    '\
											'N_{meas_{freq}} = %d\ \ \ \ '%(
											self.Neqb/self.Nspins,
											self.Nmeas/self.Nspins,
											self.Nmeas_f/self.Nspins)
											}
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
				
				data_plot_shape = [int(np.power(self.Nspins,
							1/self.model_props['d']))]*self.model_props['d']
				
				return lambda d: np.reshape(sites_region(d,self.sites),
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
								'sup_title': {'t':'Observables Histogram -'\
											' %s - q = %d \n T =  %s '%(
											caps(self.model_props['model']),
											self.model_props['q'] + (1 if 
											 self.model_props[
												'model']=='ising' else 0),
											str(self.T)) + '\n'\
											'N_{eqb} = %d\ \ \ \    '\
											'N_{meas} = %d\ \ \ \    '\
											'N_{meas_{freq}} = %d\ \ \ \ '%(
											self.Neqb/self.Nspins,
											self.Nmeas/self.Nspins,
											self.Nmeas_f/self.Nspins)},
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
								 'sup_title': {'t':'Observables Histogram -'\
											' %s - q = %d \n T =  %s '%(
											caps(self.model_props['model']),
											self.model_props['q'] + (1 if 
											 self.model_props[
												'model']=='ising' else 0),
											str(self.T)) + '\n'\
											'N_{eqb} = %d\ \ \ \    '\
											'N_{meas} = %d\ \ \ \    '\
											'N_{meas_{freq}} = %d\ \ \ \ '%(
											self.Neqb/self.Nspins,
											self.Nmeas/self.Nspins,
											self.Nmeas_f/self.Nspins)}}
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
					return lambda x:  np.mean(x,axis=-1)
				else:
					return lambda x:  np.mean(x,axis=-1)
			
			
			
			plot_props = Plot_Props(flatten(plot_keys))
			
			set_prop(plot_props,['ax','title'],plot_title)
			set_prop(plot_props,['ax','xlabel'],plot_xlabel,*args)
			set_prop(plot_props,['ax','ylabel'],plot_ylabel)
			set_prop(plot_props,['other','label'],plot_label)
			set_prop(plot_props,['data','data_process'],data_process)
			

			return plot_props

			
			
			
			
			
			
			


							  