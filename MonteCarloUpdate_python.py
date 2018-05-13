"""
Created on Tue Feb 20 14:18:39 2018
@author: Matt
"""

import numpy as np

import warnings,copy
warnings.filterwarnings("ignore")

from MonteCarloPlot import MonteCarloPlot

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
		Data_Process().plot_close()

		return



	# Perform Monte Carlo Update Algorithm and Plot Sites and Observables 
	def MC_update(self,iter_props={'algorithm':'wolff'}):

		# Initialize iter_props as array of dictionaries
		iter_props,n_iter = array_dict(iter_props)
		self.model_props['iter_props'] =  iter_props
		
		# Declare Update Variable Types
		disp_updates = self.model_props['disp_updates']
		Neqb = self.model_props['update_props']['Neqb']
		Nmeas = self.model_props['update_props']['Nmeas']
		Nmeas_f = self.model_props['update_props']['Nmeas_f']
		state_int = self.model_props['state_int']
		state_gen = self.model_props['state_gen']

		# Declare Model Variable Types
		N_sites = self.N_sites
		N_neighbours = len(self.model_props['neighbour_sites'][0,0,:])
		sites = np.zeros(self.N_sites,dtype=self.model_props['data_type'])
		cluster = np.zeros(self.N_sites, dtype=self.model_props['data_type'])
		cluster_bool = np.zeros(N_sites,dtype=bool)
		neighbours = self.model_props['neighbour_sites'][0]


		# Initialize Plotting and Data Types
		data = {}
		n_T = len(self.model_props['T'])
		n_meas = Nmeas//Nmeas_f
		data_sites = np.empty((n_iter,n_T,n_meas,N_sites), 
								dtype=self.model_props['data_type'])
									  
		plot_obj = MonteCarloPlot({'configurations':
						   self.model_props['observe_props']['configurations']},
						   self.model_props, self.model_props['T'])


		display(disp_updates,False,
				'Monte Carlo Simulation... \n%s: q = %d \nT = %s'%(
					  (self.model_props['model_name'],self.model_props['q'],
					   str(self.model_props['T'])))+'\nNeqb = %d, Nmeas = %d'%(
								 Neqb/N_sites,Nmeas/N_sites),
					   line_break=True,time_check=True)
					  
					  

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

					if i_mc % Nmeas_f == 0:
						i_mc_meas = i_mc//Nmeas_f
						data_sites[i_iter,i_t,i_mc_meas] = sites
						# display(print_it=disp_updates,
								# m='Monte Carlo Step: %d'%(i_mc//N_sites))
						
						plot_obj.MC_plotter(
						  {'configurations': {'sites': np.asarray(sites),
											  'cluster':np.asarray(cluster)}},
						*[[t],[],i_mc/N_sites])                
			  
				display(print_it=disp_updates,m='Updates: T = %0.2f'%t)
				
			# Save Current Data
			if self.model_props['data_save']:
				plot_obj.plot_save(self.model_props,
								        label=self.model_props['algorithm'],
										fig_keys='configurations')
				Data_Process().exporter({'sites': np.asarray(data_sites)},
										 self.model_props,
										 self.model_props['algorithm'])  
		
			display(print_it=disp_updates,
					m='Runtime: ',t0=-(i_t+2),line_break=True)
				
			
		# Compute Observables Data
		data['sites'] = data_sites
		data['observables'] = self.MC_measurements(data['sites'],
											self.model_props['neighbour_sites'],
											self.model_props['T'],
											self.model_props['observables'])                                                   
			
		if self.model_props['data_save']:
			Data_Process().exporter(data,self.model_props)
			Data_Process().exporter({'model_props':self.model_props},
							        self.model_props,format='.txt') 
			
		display(print_it=disp_updates,m='Observables Calculated')
		display(print_it=disp_updates,time_it=False,
				m='Monte Carlo Simulation Complete...')
				
		return data, self.model_props

	# Update Algorithms
	def metropolis(self,sites, cluster, cluster_bool, neighbours,
						N_sites,N_neighbours, T, 
						state_update, state_gen, state_int):
		# Randomly alter random spin sites and accept spin alterations
		# if energetically favourable or probabilistically likely

		# Generate Random Spin Site and store previous Spin Value

		isite = np.random.randint(N_sites)
		sites0 = sites[isite]

		# Update Spin Value
		sites[isite] = state_gen(1,sites0)
		cluster[isite] = sites0
		# Calculate Change in Energy and decide to Accept/Reject Spin Flip
		nn = sites[neighbours[0]]
		dE = -np.sum(state_int(sites[isite],nn)) + (
			  np.sum(state_int(sites0,nn)))

		if dE > 0:
			if state_update[dE,T] < np.random.random():
				sites[isite] = sites0                
		return


	def wolff(self,sites, cluster, cluster_bool, neighbours,
						N_sites,N_neighbours, T, 
						state_update, state_gen, state_int):      
		
		# Cluster Function
		def cluster_update(i):

			# Add indices to cluster
			cluster_bool[i] = True
			cluster[i] = cluster_value0
			J = (j for j in neighbours[i] if (not cluster_bool[j]) and 
											(sites[j] == cluster_value0)) 

			for j in J:
				if state_update[T] > np.random.random():
					cluster_update(j)

			return
		
		# Create Cluster Array and Choose Random Site

		isite = np.random.randint(N_sites)

		cluster_value0 = sites[isite]

		# Perform cluster algorithm to find indices in cluster
		cluster_update(isite)

		# Flip spins in cluster to new value
		sites[cluster_bool] = state_gen(1,cluster_value0)
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

	