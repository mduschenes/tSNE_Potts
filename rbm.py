# Restricted Boltzmann Machine (RBM)

class RBM(object):

	def __init__(self,N,q,hyperparams):

		self.vars = ['visible','hidden']
		if isinstance(N,list):
			self.N = dict(zip(self.vars,N))
		elif isinstance(N,dict)
			self.N = N
			self.vars = list(N.keys())

		if isinstance(q,list):
			self.q = dict(zip(self.vars,q))
		elif isinstance(q,dict)
			self.q = q

		self.Nvars = len(self.vars)
		
		self.neurons = {v: np.random.choice(i,n) 
						for v,i,n in zip(self.vars,self.q,self.N)}

		self.weights = np.random.randn(*[self.N[v] for v in self.vars])	
		self.biases = {v: np.random.randn(self.N[v]) for v in self.vars}


		

		return

	def energy(self):
		energy = self.weights
		for v in self.vars:
			energy = (-np.tensordot(self.neurons[n],energy,axes=([0],[0])) + 
					 -np.dot(self.biases[v],self.neurons[v]))
		return energy

	def conditional(self,):

		probability = 

		return probability



	def t
        train_step = network_functions['optimize_functions'][
                                 alg_params['optimize_func']](
                                                      cost=cost(x_,y_,y_est),
                                                      alpha_learn=alpha_learn)