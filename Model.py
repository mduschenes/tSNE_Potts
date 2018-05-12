# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:23:39 2018
@author: Matt
"""

import numpy as np

from misc_functions import get_attr


class Model(object):
    
    # Define a Model class for spin model with: 
    # Lattice Model Type: Model Name and Max Spin Value q
    
	def __init__(self,model={'model_name':'potts', 'q':3, 'd':2, 'T': 2.2269,
							 'order_param':[0,1], 'data_type':np.int_},
                 observe=['temperature','energy','order']):
        # Define Models dictionary for various spin models, with
        # [ Individual spin calculation function, 
        #   Spin Upper, Lower, and Excluded Values]
        
        # Define spin value model and q (~max spin value) parameters
        # (i.e: Ising(s)-> s, Potts(s) -> exp(i2pi s/q)  )
		self.model_name = model['model_name'].lower()
		self.q = model['q']
		self.d = model['d']
		self.T = np.atleast_1d(model['T'])
		self.orderparam = model['order_param']


		self.model_types = ['ising','potts']
        
		# Define model specific parameters
		self.models_params = {'ising': {'state_update': {
				
											'metropolis': {(dE,t): 
											  np.exp(-self.orderparam[1]*dE/t) 
												 for dE in range(-2*2*self.d,
																  2*2*self.d+1)
												 for t in self.T},
												
											'wolff':{t: (1 - np.exp(
													  -2*self.orderparam[1]/t))
													 for t in self.T}},
												
									   'value_range': [-self.q,self.q,0]},
							
							'potts':  {'state_update': {
									
											'metropolis': {(dE,t): 
											  np.exp(-self.orderparam[1]*dE/t) 
												 for dE in range(-2*self.d,
																  2*self.d+1)
												 for t in self.T},
												
											'wolff':{t: (1 - np.exp(
													  -1*self.orderparam[1]/t))
													 for t in self.T}},
												
									   'value_range': [1,self.q,None]}} 
		
		# Define model specific functions                                
		for t in self.model_types:
			for k in ['value','int','order','twoptcorr']:
				self.models_params[t][k] = getattr(self,t+'_'+k,
												   lambda *args:[])
					

		# Define Model Value function and Name for given Input
		self.model_params = self.models_params[self.model_name]
		self.model = self.model_params['value']

		# Define Range of Possible Spin Values and  Energy Function                     
		self.state_range = self.state_ranges(dtype=model['data_type'])

		
		# Define Observables
		self.observables_functions = {k: getattr(self,k,lambda *args:[]) 
										 for k in observe}
		
		self.observables_props = {k: lambda prop,*arg: get_attr(v,prop,v,*arg) 
								 for k,v in self.observables_functions.items()}
		
		return
		
		   
	 
		
	# Generate array of possible q values
	def state_ranges(self,xNone=None,xmin=float('inf'),xmax=-float('inf'),
					 dtype=np.int_):
		
		vals = self.model_params['value_range']
		
		# Exclude xNone Values
		if xNone == None:
			xNone = np.atleast_1d([vals[2]])
			
		# List of range of possible spin values, depending on model
		return np.array([x for x in range(min([xmin,vals[0]]),
										  max([xmax,vals[1]+1])) 
						 if x not in xNone],dtype=dtype)




	# Generate N Random q values, exluding n0
	def state_gen(self,N=1,n0=None):
		# Return array of N random spins, per possible state_range spin values
		# excluding the possible n0 spin
		# Model dependent generator of spin values 

		return self.model(np.random.choice(
										  np.setdiff1d(self.state_range,n0),N))
	
	
	
	
	
##### Model Observables (per site) #########
		 
	def temperature(self,sites,neighbours,T):
		return T
	
	def sites(self,sites,neighbours,T):
		return sites

	def energy(self,sites,neighbours,T):
		# Calculate energy of spins as sum of spins 
		# + sum of r-distance neighbour interactions
		site_int = self.model_params['int']
		return ((-self.orderparam[0]*np.sum(site_int(sites),axis=-1))+(
			   -(1/2)*np.sum(np.array([self.orderparam[i+1]*(
									   site_int(
										 np.expand_dims(sites,axis=-1),
										 np.take(sites,neighbours[i],axis=-1)))
				for i in range(len(self.orderparam)-1)]),
				axis=(0,-2,-1))))/np.shape(sites)[-1]

	
	def order(self,sites,neighbours,T):
		site_order = self.model_params['order']        
		return site_order(sites)
	
	def twoptcorr(self,sites,neighbours,T):
		site_twoptcorr = self.model_params['twoptcorr']
		return site_twoptcorr(sites)
	
	def specific_heat(self,sites,neighbours,T):  
	  wrapper = lambda x: np.power(x,2)
	  return np.reshape((self.obs_mean(sites,neighbours,T,self.energy,wrapper) - 
			 wrapper(self.obs_mean(sites,neighbours,T,self.energy)))*(
			 np.shape(sites)[-1]/wrapper(T)),
			 (np.size(T),-1))
  
 
	def susceptibility(self,sites,neighbours,T):        
	  wrapper = lambda x: np.power(x,2)
	  return np.reshape((self.obs_mean(sites,neighbours,T,self.order,wrapper) - 
			 wrapper(self.obs_mean(sites,neighbours,T,self.order,np.abs)))*(
			 np.shape(sites)[-1]/T),
			 (np.size(T),-1))
	
		
	def local_energy(self,i,sites,neighbours,T):
		site_int = self.model_params['int']
		return -np.sum(np.array([self.orderparam[j+1]*site_int(
				np.expand_dims(np.take(sites,i,axis=-1),axis=-1),
				np.take(sites,neighbours[j][i],axis=-1)) 
				for j in range(len(self.orderparam)-1)]),
				axis=(0,-1))



	def obs_mean(self,sites,neighbours,T,func,wrapper=lambda x:x):
		return np.mean(wrapper(func(sites,neighbours,T)),axis=-1)






		   
	# Model Site Values                                 
	def ising_value(self,s):
		return s
	
	def potts_value(self,s):
		return s
	
	

	# Model Interactions Energy 
	def ising_int(self,*args):
		try:
			return args[0]*args[1]
		except IndexError:
			return args[0]
	
	def potts_int(self,*args):
		try:
			return 1*np.equal(args[0],args[1])
		except IndexError:
			return args[0]
		

	# Model Order Parameter
	def ising_order(self,s):
		return np.sum(s,axis=-1)/np.shape(s)[-1]
	
	def ising_twoptcorr(self,s):
		shape_l = np.shape(s)[-1]
		return np.sum([self.ising_int(np.delete(s,i,axis=-1),
									  np.take(s,i,axis=-1))
					   for i in range(shape_l)],axis=(0,-1))/shape_l
	
	def potts_order(self,s,u=1):
		return np.mean(np.array([(self.q*np.mean(self.potts_int(s,u),
								 axis=-1)- 1)/(self.q-1)
						 for u in range(1,2)]),axis=0)

	def potts_twoptcorr(self,s):
		shape_l = np.shape(s)[-1]
		return np.sum([self.potts_int(np.delete(s,i,axis=-1),
									  np.take(s,i,axis=-1))
					   for i in range(shape_l)],axis=(0,-1))/shape_l














	
	
	
	
	