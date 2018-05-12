"""
Created on Tue Feb 20 14:18:39 2018
@author: Matt
"""
#!python
#cython: language_level=3,boundscheck=false,wraparound=false,cdivision=True

import numpy as np

cimport numpy as cnp
from numpy cimport ndarray
cimport cython

import warnings,copy
warnings.filterwarnings("ignore")


from data_functions import Data_Process
from misc_functions import flatten,array_dict, caps, display


ctypedef  long  SITE_TYPE



# cdef extern from "stdlib.h":
	# double drand48()
	# void srand48(long int seedval)

# cdef extern from "time.h":
	# long int time(int)

# srand48(time(0))

# from libc.stdlib cimport rand
# cdef extern from "stdlib.h":
	# long RAND_MAX

# cdef double RAND_DIV = 1/RAND_MAX


# def MonteCarloUpdater(model_props={}):
	# return MonteCarloUpdate(model_props)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class MonteCarloUpdate(object):
    
	# Declare Class Attributes

	# Define System Configuration
	cdef object sites, cluster
	cdef int N_sites

	# Define Configurations and Observables Data Dictionaries
	cdef dict model_props,data

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
        # observe_props: [Boolean to Animate [Sites, Observables]]


#        model_props = {'L': L, 'd': d, 'q': m.q, 'T': T,
#                            'state_range': m.state_range,
#                            'state_gen': m.state_gen,
#                            'prob_trans': {'metropolis': m.energy,
#                                           'wolff':m.model_params[
#                                                   m.model.__name__][
#                                                   'bond_prob']},
#                            'model_name': m.model.__name__,
#                            'algorithm': 'wolff',
#                            'algorithms': ['metropolis','wolff'],
#                            'update': update,
#                            'observe_props': observe_props,
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
		model_props['T'] = np.atleast_1d(model_props['T'])
		model_props['N_sites'] = np.shape(model_props['neighbour_sites'])[1]
		self.N_sites = np.shape(model_props['neighbour_sites'])[1]
		
		self.sites = np.zeros(self.N_sites,dtype=np.int_)
		self.cluster = np.zeros(self.N_sites,dtype=int)

		# Define Monte Carlo Update Parameters:
		# Number of updates to reach "equilibrium" before measurement, 
		# Number of measurement updates.
		# Each sweep consists of Nsites updates
		
		for n in ['Neqb','Nmeas','Nmeas_f']:
			model_props['update_props'][n] *= self.N_sites


		# Define Configurations and Observables Data Dictionaries
		self.model_props = model_props 
		self.data = {}
		Data_Process().plot_close()

		return



	# Perform Monte Carlo Update Algorithm and Plot Sites and Observables 
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.cdivision(True)
	def MC_update(self,iter_props={'algorithm':'wolff'}):

		# Monte Carlo Update Function
		cdef object MC_alg 

		# Initialize iter_props as array of dictionaries
		cdef int n_iter
		iter_props,n_iter = array_dict(iter_props)
		
		# Declare Update Variable Types
		cdef int disp_updates = self.model_props['disp_updates']
		cdef int i_iter=0,i_t=0,i_mc=0,i_mc_meas=0
		cdef int Neqb = self.model_props['update_props']['Neqb']
		cdef int Nmeas = self.model_props['update_props']['Nmeas']
		cdef int Nmeas_f = self.model_props['update_props']['Nmeas_f']
		cdef object state_update
		cdef object state_int = self.model_props['state_int']
		cdef object state_gen = self.model_props['state_gen']

		# Declare Model Variable Types
		cdef int N_sites = self.N_sites
		cdef int N_neighbours = len(self.model_props['neighbour_sites'][0,0,:])
		cdef SITE_TYPE[::1] sites = np.zeros(self.N_sites,
									        dtype=self.model_props['data_type'])
		cdef SITE_TYPE[::1] cluster = np.zeros(self.N_sites,
									        dtype=self.model_props['data_type'])
		cdef int[::1] cluster_bool = np.zeros(N_sites,dtype=np.intc)
		cdef int[:,::1] neighbours = self.model_props['neighbour_sites'][0]


		# Initialize Plotting and Data Types
		cdef int n_T = len(self.model_props['T'])
		cdef int n_meas = Nmeas//Nmeas_f
		cdef SITE_TYPE[:,:,:,::1] data_sites = np.empty((n_iter,n_T,
														 n_meas,N_sites), 
									  dtype=self.model_props['data_type'])
									  
		cdef object plot_obj = self.MC_plot(self.model_props['observe_props'],
										    None,None, self.model_props['T'])


		display(disp_updates,False,
				'Monte Carlo Simulation... \n%s: q = %d \nT = %s'%(
					  (self.model_props['model_name'],self.model_props['q'],
					   str(self.model_props['T'])))+'\nNeqb = %d, Nmeas = %d'%(
								 Neqb/N_sites,Nmeas/N_sites),
					   line_break=True)
					  
					  

		# Declare iteration variables

		# Perform Monte Carlo Algorithm for n_iter configurations        
		for i_iter in range(n_iter):
				
			# Update dictionary and plotting for each Iteration
			self.model_props.update(iter_props[i_iter])

			
			# Initialize sites with random spin at each site for each Iteration
			sites = state_gen(self.N_sites)
			state_update = self.model_props['state_update'][
									   self.model_props['algorithm']]

			display(disp_updates,time_check=True,
					m= 'Iter %d: %s Algorithm'%(
								   i_iter,caps(self.model_props['algorithm'])))

			MC_alg = getattr(self,self.model_props['algorithm'])

			# Perform Monte Carlo at temperatures t = T
			for i_t,t in enumerate(self.model_props['T']):
				# Perform Equilibration Monte Carlo steps initially
				for i_mc in range(Neqb):
					MC_alg(sites,cluster.copy(),cluster_bool.copy(),neighbours,
						   N_sites,N_neighbours,t, 
						   state_update, state_gen, state_int)					
					
				# Perform Measurement Monte Carlo Steps
				for i_mc in range(Nmeas):
					MC_alg(sites,cluster.copy(),cluster_bool.copy(),neighbours,
						   N_sites,N_neighbours,t, 
						   state_update, state_gen, state_int)
					# print('cluster',np.asarray(cluster))
					# print('sites',np.asarray(sites))
					# Update Configurations and Observables
					if i_mc % Nmeas_f == 0:
						i_mc_meas = i_mc//Nmeas_f
						data_sites[i_iter,i_t,i_mc_meas] = sites
						# display(printit=disp_updates,
								# m='Monte Carlo Step: %d'%(i_mc//N_sites))
						
						plot_obj = self.MC_plot(
						{'configurations': 
						  self.model_props['observe_props']['configurations']},
	    				  plot_obj,
						  {'configurations': {'sites': np.asarray(sites),
											  'cluster':np.asarray(cluster)}},
						*[[t],[],i_mc/N_sites])                
			  
				display(printit=disp_updates,m='Updates: T = %0.2f'%t)
				
			# Save Current Data
			if self.model_props['data_save']:
				plot_obj.plot_save(self.model_props,
								        label=self.model_props['algorithm'],
										fig_keys='configurations')
				Data_Process().exporter({'sites': np.asarray(data_sites)},self.model_props,
										self.model_props['algorithm'])  
		
			display(printit=disp_updates,
					m='Runtime: ',t0=-(i_t+2),line_break=True)
				
				
		# Compute, Plot and Save Observables Data and Figures
		self.data['sites'] = data_sites
		self.data['observables'] = self.MC_measurements(self.data['sites'],
											self.model_props['neighbour_sites'],
											self.model_props['T'],
											self.model_props['observables'])                                                   
			
			
		display(printit=disp_updates,m='Observables Calculated')

		plot_obj = self.MC_plot(self.model_props['observe_props'], plot_obj,
								{'observables': self.data['observables'],
								 'observables_mean': self.data['observables']},
								*[self.model_props['T'],
								  [p['algorithm']for p in iter_props],
								  i_mc/N_sites
								 ])

		display(printit=disp_updates,m='Figures Plotted')

		
		if self.model_props['data_save']:
			plot_obj.plot_save(self.model_props,
							        fig_keys=['observables','observables_mean'])
			Data_Process().exporter(self.data,self.model_props)  
											   
		return

	# Update Algorithms
	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.cdivision(True)
	def metropolis(self,SITE_TYPE[::1] sites,
							  SITE_TYPE[::1] cluster,
							  int [::1] cluster_bool,
							  int[:,::1] neighbours,
							  int N_sites,int N_neighbours,
							  double T, 
							  dict state_update, 
							  object state_gen,
							  object state_int):  
	#def metropolis(ndarray[cnp.int8_t ,ndim=1] sites,
	#               ndarray[cnp.int8_t, ndim=1] cluster,
	#               ndarray[cnp.int8_t, ndim=2] neighbours, 
	#               double T, dict state_update, state_gen,state_int):
	# Randomly alter random spin sites and accept spin alterations
	# if energetically favourable or probabilistically likely

		# Generate Random Spin Site and store previous Spin Value

		cdef int isite = int(np.random.randint(N_sites)) #int(drand48()*(N_sites))
		

		cdef SITE_TYPE sites0 = sites[isite]

		# Update Spin Value
		sites[isite] = state_gen(1,sites0)
		cluster[isite] = sites0
		# Calculate Change in Energy and decide to Accept/Reject Spin Flip
		cdef int i=0,dE=0 
		cdef SITE_TYPE nn
		for i in range(N_neighbours):
			nn = sites[neighbours[isite][i]]
			dE += -state_int(sites[isite],nn) + state_int(sites0,nn)
		# -np.sum(state_int(sites[isite],nn)) + np.sum(state_int(sites0,nn))
		if dE > 0:
			if state_update[dE,T] < np.random.random(): #(
				sites[isite] = sites0
				
		return

	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.cdivision(True)
	def wolff(self,SITE_TYPE[::1] sites,
							  SITE_TYPE[::1]  cluster,
							  int [::1] cluster_bool,
							  int[:,::1] neighbours,
							  int N_sites,int N_neighbours,
							  double T, 
							  dict state_update, 
							  object state_gen,
							  object state_int):  
		# Create Cluster Array and Choose Random Site

		@cython.boundscheck(False)
		@cython.wraparound(False)
		@cython.cdivision(True)
		def cluster_update(int i):

			# Add indices to cluster
			cluster_bool[i] = 1
			cluster[i] = cluster_value0
			cdef int[::1] nn = neighbours[i]
			cdef int n=0,j = 0
			for j in range(N_neighbours):
				n = nn[j]
				if (cluster_bool[n]==0) and (sites[n] == cluster_value0):
					if state_update[T] >  np.random.random():
						cluster_update(n)
			return
		cdef int isite = np.random.randint(N_sites) #int(drand48()*(N_sites))
		
		cdef SITE_TYPE cluster_value0 = sites[isite]

		# Perform cluster algorithm to find indices in cluster
		cluster_update(isite)

		# Flip spins in cluster to new value
		cdef int val = int(state_gen(1,cluster_value0))

		cdef int i = 0

		for i in range(N_sites):
			if cluster_bool[i] == 1:
				sites[i] = val

		return
























	# Update Algorithms
	def metropolis_python(self,T=1):
		# Randomly alter random spin sites and accept spin alterations
		# if energetically favourable or probabilistically likely

		# Generate Random Spin Site and store previous Spin Value

		isite = [np.random.randint(self.N_sites) for j in 
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
			if self.state_update[dE,T] < np.random.random():
				self.sites[isite] = sites0                
		return


	def wolff_python(self,T=1):      
		# Create Cluster Array and Choose Random Site

		isite = np.random.randint(self.N_sites)

		self.cluster_bool = np.zeros(self.N_sites,dtype=np.int_)
		self.cluster_value0 = self.sites[isite]

		# Perform cluster algorithm to find indices in cluster
		self.cluster_update_python(isite,T)

		# Flip spins in cluster to new value
		self.sites[self.cluster] = self.state_gen(1,self.cluster_value0)
		return

			   

	# Cluster Function
	def cluster_update_python(self,i,T):

		# Add indices to cluster
		self.cluster_bool[i] = 1
		self.cluster[i] = self.cluster_value
		J = (j for j in self.neighbour_sites[0][i] if (not self.cluster[j]) and 
										(self.sites[j] == self.cluster_value0)) 

		for j in J:
			if self.state_update[T] > np.random.random():
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
	@cython.wraparound(True)
	def MC_plot_props(self,plot_type,plot_keys,model_props,*args):

		# Function plot sites or clusters of sites
		def sites_region(sites):
			sites0 = np.asarray(sites,dtype=float)
			sites0[np.isin(sites,self.model_props['state_range'],invert=True)]=np.nan
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
					self.model_props['update_props']['Neqb']/self.N_sites, 
					self.model_props['update_props']['Nmeas']/self.N_sites,
					self.model_props['update_props']['Nmeas_f']/self.N_sites)

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
											sup_title('Monte Carlo Updates')}
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
				
				data_plot_shape = [int(np.power(self.N_sites,
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

			
			
			
			
			
			
			


							  