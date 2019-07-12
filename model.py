# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:23:39 2018
@author: Matt
"""

import numpy as np

from lattice import lattice

class model(lattice):
    
    # Define a Model class for spin model with: 
    # Lattice Model Type: Model Name and Max Spin Value q
    
	def __init__(self,name='potts',q=3,d=2,L=6,T=2.2269,couplings=[0,1],
					dtype=np.int_,**kwargs):
        # Define Models dictionary for various spin models, with
        # [ Individual spin calculation function, 
        #   Spin Upper, Lower, and Excluded Values]
        
        # Define spin value model and q (~max spin value) parameters
        # (i.e: Ising(s)-> s, Potts(s) -> exp(i2pi s/q)  )
		self.name = name.lower()
		self.q = q
		self.d = d
		self.L = L
		if T < 0:
			self.T = self.critical_temperature(model.get('T'),d,q,name)
		else:
			self.T = T

		self.couplings = couplings
		self.dtype = dtype
		self.model_types = ['ising','potts']
		self.observables = ['state','interaction','magnetization','order',
							'twoptcorr','specific_heat','susceptibility',
							'local_energy','state_difference']


		# Set up model Lattice
		lattice.__init__(self,L=L,d=d)

		# Define model specific parameters
		self.models_params = {'ising': {'transition_probability': {
										'metropolis': {dE: 
											np.exp(-self.couplings[1]*dE/self.T) 
												 for dE in range(-2*self.z,
																  2*self.z+1)},												
										'wolff':1 - np.exp(
												   -2*self.couplings[1]/self.T)
												  },
												
									   'state_range': [x for x in range(-self.q,
									   						self.q+1) if x!=0]},
							
							'potts':  {'transition_probability': {
										'metropolis': {dE: 
											np.exp(-self.couplings[1]*dE/self.T) 
												 for dE in range(-self.z,
																  self.z+1)
												  },
												
											'wolff':1 - np.exp(
												   -1*self.couplings[1]/self.T)
													 },
												
									   'state_range': list(range(1,self.q+1))}} 
		
		# Define model specific functions                                
		for t in self.model_types:		
			for k in self.observables:
				self.models_params[t][k] = getattr(self,t+'_'+k,
												   lambda *args:[])


		# Given model name, set model
		self.set_model(self.name)

		return
		
	# Model Name Specific Parameters
	def set_model(self,name):
		for k,v in self.models_params[name].items():
			setattr(self,k,v)

	# Generate N Random q values, exluding s
	def state_generate(self,s=None,N=1):
		# Return array of N random spins, per possible state_range spin values
		# excluding the possible n0 spin
		# Model dependent generator of spin values 

		return self.state(np.random.choice(
										  np.setdiff1d(self.state_range,s),N))
	
	
	
	
	
	### Model Observables (per site) ######

	def obs_mean(self,func=None,wrapper=None,*args):
		if func is None: func = lambda x:x
		if wrapper is None: wrapper = lambda x:x
		
		return np.mean(wrapper(func(*args)),axis=-1)

   


	### Ising Model ###
                              
	def ising_state(self,s):
		return s

	def ising_interaction(self,s1,s2):
		return s1*s2


	def ising_energy(self,sites,neighbours,T):
		# Calculate energy of spins as sum of spins 
		# + sum of r-distance neighbour interactions
		return ((-self.couplings[0]*np.sum(site_int(sites),axis=-1))+(
			   -(1/2)*np.sum(np.array([self.couplings[i+1]*(
									   self.ising_interaction(
										 np.expand_dims(sites,axis=-1),
										 np.take(sites,neighbours[i],axis=-1)))
				for i in range(len(self.couplings)-1)]),
				axis=(0,-2,-1))))/np.shape(sites)[-1]


	def ising_magnetization(self,s,u=None):
		return np.mean(s,axis=-1)

	def ising_order(self,s):
		return self.ising_magnetization(s)
	
	def ising_twoptcorr(self,s):
		shape_l = np.shape(s)[-1]
		return np.sum([self.ising_interaction(np.delete(s,i,axis=-1),
									  np.take(s,i,axis=-1))
					   for i in range(shape_l)],axis=(0,-1))/shape_l


	def ising_specific_heat(self,sites,neighbours,T):  
		wrapper = lambda x: np.power(x,2)
		return np.reshape((self.obs_mean(self.energy,wrapper,sites,neighbours,T) - 
			wrapper(self.obs_mean(self.energy,None,sites,neighbours,T)))*(
			np.shape(sites)[-1]/wrapper(T)),
			(np.size(T),-1))



	def ising_susceptibility(self,sites,neighbours,T):        
		wrapper = lambda x: np.power(x,2)
		return np.reshape(np.mean([(self.obs_mean(self.ising_magnetization, wrapper,
									sites,neighbours,T,u) - 
				wrapper(self.obs_mean(self.ising_magnetization, np.abs,
									sites,neighbours,T,u)))*(
				np.shape(sites)[-1]/T) for u in range(0,self.q)],axis=0),
				(np.size(T),-1))

		
	def ising_local_energy(self,i,sites,neighbours,T): 
		return np.sum(
					np.array([self.couplings[j+1]*
						self.ising_interaction(			
							np.expand_dims(np.take(sites,i,axis=-1),axis=-1),
							np.take(sites,neighbours[j][i],axis=-1)) 
				for j in range(len(self.couplings)-1)]),
				axis=(0,-1))


	def ising_state_difference(self,s1,s2,neighbours):
		return -np.sum(self.ising_interaction(s1,neighbours)) + (
			    np.sum(self.ising_interaction(s2,neighbours)))








	### Potts Model ###
	
	def potts_state(self,s):
		return s

	
	def potts_interaction(self,s1,s2):
		return 1*np.equal(s1,s2)



	def potts_energy(self,sites,neighbours,T):
		# Calculate energy of spins as sum of spins 
		# + sum of r-distance neighbour interactions
		return ((-self.couplings[0]*np.sum(site_int(sites),axis=-1))+(
			   -(1/2)*np.sum(np.array([self.couplings[i+1]*(
									   self.ising_interaction(
										 np.expand_dims(sites,axis=-1),
										 np.take(sites,neighbours[i],axis=-1)))
				for i in range(len(self.couplings)-1)]),
				axis=(0,-2,-1))))/np.shape(sites)[-1]
		

	
	def potts_magnetization(self,s,u=1):
		return np.mean(self.potts_interaction(s,u), axis=-1)
	
	def potts_order(self,s,u=1):
		return np.mean(np.array([(self.q*self.potts_magnetization(s,u)-1)/(
																self.q-1)
						 for u in range(1,2)]),axis=0)

	def potts_twoptcorr(self,s):
		shape_l = np.shape(s)[-1]
		return np.sum([self.potts_interaction(np.delete(s,i,axis=-1),
									  np.take(s,i,axis=-1))
					   for i in range(shape_l)],axis=(0,-1))/shape_l


	def potts_specific_heat(self,sites,neighbours,T):  
		wrapper = lambda x: np.power(x,2)
		return np.reshape((self.obs_mean(self.energy,wrapper,sites,neighbours,T) - 
			wrapper(self.obs_mean(self.energy,None,sites,neighbours,T)))*(
			np.shape(sites)[-1]/wrapper(T)),
			(np.size(T),-1))



	def potts_susceptibility(self,sites,neighbours,T):        
		wrapper = lambda x: np.power(x,2)
		return np.reshape(np.mean([(self.obs_mean(self.potts_magnetization, wrapper,
									sites,neighbours,T,u) - 
				wrapper(self.obs_mean(self.potts_magnetization, np.abs,
									sites,neighbours,T,u)))*(
				np.shape(sites)[-1]/T) for u in range(0,self.q)],axis=0),
				(np.size(T),-1))

		
	def potts_local_energy(self,i,sites,neighbours,T): 
		return np.sum(
					np.array([self.couplings[j+1]*(
						self.potts_interaction)(			
							np.expand_dims(np.take(sites,i,axis=-1),axis=-1),
							np.take(sites,neighbours[j][i],axis=-1)) 
				for j in range(len(self.couplings)-1)]),
				axis=(0,-1))



	def potts_state_difference(self,s1,s2,neighbours):
		return -np.sum(self.potts_interaction(s1,neighbours)) + (
			    np.sum(self.potts_interaction(s2,neighbours)))




	# Critical Temperature
	def critical_temperature(self,T,d,q,model):
		if T < 0 and d == 2:
			T =  np.power(np.log(1+np.sqrt(q)),-1)
			if model == 'ising':
				T *= 2
		return T









	
	
	
	
	