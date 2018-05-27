import numpy as np
import argparse
# import sys
# sys.path.insert(0, "C:/Users/Matt/Google Drive/PSI/"+
				   # "PSI Essay/PSI Essay Python Code/tSNE_Potts/")

from data_functions import Data_Process 
from misc_functions import dict_modify,display


TOL_MIN = 1e-14

# Compute Squared Pairwise distance between all elements in x
def pairwise_distance(x):
	x2 =  np.dot(x,x.T) 
	x2_d = np.diagonal(x2)
	return x2_d + (x2_d - 2*x2).T
#    m = np.shape(x)[0]
#    return  np.tile(x2_d,(m,1)).T + np.tile(x2_d,(m,1)) -2*(x2) 

# Compute Pairwise distance between all elements in x, exluding self-distances
def neighbour_distance(x,n):
	return pairwise_distance(x)[~np.eye(n,dtype=bool)].reshape(n,n-1)

# Compute guassian representation of pairwise distances
def rep_gaussian(d,sigma,normalized=True):
	p = np.atleast_2d(np.exp(-d*sigma))
	np.fill_diagonal(p,0)
	if normalized: return p/np.sum(p,axis=1)
	else: return p,np.maximum(np.sum(p,axis=1),TOL_MIN)

def rep_tdistribution(d,sigma,normalized=True):
	p = np.power(1+d,-1)
	np.fill_diagonal(p,0)
	if normalized: return p/np.sum(p,axis=1)
	else: return p,np.maximum(np.sum(p,axis=1),TOL_MIN)
	

def entropy(p,q,axis=-1):
	return np.sum(q*np.log(q),axis=axis)


def entropy_gaussian(d,sigma):
	p,norm_p = rep_gaussian(d,sigma,normalized=False)
	return np.power(norm_p,-1)*sigma*np.sum(p*d,1),p


def KL_entropy(p,q):
	return entropy(p,p) - entropy(p,q)


def binary_search(x,a,f,f0,tol,n_iter):
		
		fi,y = f(x,a)        
		n = np.shape(y)[0]
		df = fi-f0
		
		i = 0
		a_min = -np.inf*np.ones(n)
		a_max = np.inf*np.ones(n)
		
		
		while np.any(np.abs(df) > tol) and i < n_iter:
			ind_pos = df>0 
			ind_neg = df<=0
			
			ind_min_inf = np.abs(a_min)==np.inf
			ind_max_inf = np.abs(a_max)==np.inf
			
			#print('f',i,fi)
			
			ind_min_ninf = np.logical_not(ind_max_inf)
			ind_max_ninf = np.logical_not(ind_min_inf)
			
			a_min[ind_pos] = a[ind_pos].copy()
			a_min[ind_neg] = a[ind_neg].copy()

			a[ind_pos & ind_max_inf] *=4
			a[ind_pos & ind_max_ninf] += a_max[ind_pos & ind_max_ninf]
			a[ind_neg & ind_min_ninf] += a_min[ind_neg & ind_min_ninf]
			a /= 2
			
			#print(a)

			fi,y = f(x,a)
			df = fi-f0
			i += 1
			
		print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / a)))
		print("Mean value of Perp: %f" % np.mean(fi))

		return y



def process(X,data_params,data_rep,data_type,
			alg_params = {  'N0': None, 'N': 2,'perp': 30.0, 'pca': True,
							'data_dir': 'dataset/tsne/',
							'data_reps':['tsne','pca'],
							'data_name_format':['','pca','perp','N0','']}):

	
	data_params = data_params.copy()


	Data_Process().format(data_params,initials=False)
	if data_params['file_dir'] == '':
		data_params['file_dir'] = data_params['data_dir']
	
	
	# Import Data
	
	data,data_sizes,_,_ = Data_Process().importer(data_params,
							data_typing='dict',data_lists=True,upconvert=True,
							directory=data_params['file_dir'])
	
	display(True,True,'Data Imported... \n'+str(data_sizes)+'\n'+ (
												data_params['data_file']+'\n'))

	# Setup Data
	
	# Change keys structure for appropriate plot labels (i.e. L_XX size labels)
	ind_data = [-3,None]
	ind_type = [0,None]
		
	data_typed = dict_modify(data,data_types_config+data_types_temps,
				              f=lambda k,v: v.copy(),i=ind_data,j=ind_type)
	
	data_sizes = dict_modify(data_sizes,data_types_config+data_types_temps,
				              f=lambda k,v: v,i=ind_data,j=ind_type)
	
	data_keys = {t: sorted(list(d.keys())) for t,d in data_typed.items()}
	
	Y = {r: dict_modify(data,data_types_config,
				              f=lambda k,v: [],i=ind_data,j=ind_type)
			for r in data_reps}
	
	Y2 = Y.copy()
	
	data_types_config = [t[slice(*ind_type)] for t in data_types_config]
	data_types_temps  = [t[slice(*ind_type)] for t in data_types_temps]
	
	
	# Setup Plotting
	plot_keys = {}
	plot_bool = {}
	for r in data_reps:
		for t in Y[r].keys():
			plot_keys[r+'_'+t] = data_keys[t]
			plot_bool[r+'_'+t]= True
	
	Data_Process().plot_close()        
	Data_Proc = Data_Process(plot_keys,plot_bool)
	
	comp = lambda x,i: {k:v[:,i] for k,v in x.items() if np.any(v)}
	
	# tSNE and PCA Analysis

	for t in sorted(data_types_config):
		
		for r in data_reps:
			
			if r == 'pca':
				continue
			
			# Check if Data Exists
			params = data_params.copy()
			file_header = r+'_'+t
			file_name = file_header + data_params['data_file']
			params['data_files'] = file_name+ '.'+data_params['data_format']
			data = Data_Proc.importer(params,data_obj_format='dict',
										format='npz')
			if data is not None:
				print('Previous Data Found for',file_name)
				Y[r][t] = data[0][file_name].item()
				
				Data_Proc.plotter(comp(Y[r][t],1),comp(Y[r][t],0),
					 plot_props(Y[r][t].keys(),r,t[-5:]),
					 data_key=r+'_'+t)
				
			else:
				print('New Data for',file_name)
				for k in data_keys[t]:   
					print(r,t,k)
					Y[r][t][k] = dim_reduce(data = data_typed[t][k],
										 N = data_params['N'],
										 N0 = data_params['N0'], 
										 perp = data_params['perp'], 
										 rep = r, 
										 pca = data_params['pca'])
					
				Data_Proc.exporter({file_header: Y[r][t]},data_params)
				
			
			Data_Proc.plotter(comp(Y[r][t],1),comp(Y[r][t],0),
					 plot_props(Y[r][t].keys(),r,t[-5:]),
					 data_key=file_header)
	Data_Proc.plot_save(data_params,read_write='a')   


def Hbeta0(D=np.array([]), beta=1.0):
	"""
		Compute the perplexity and the P-row for a specific value of the
		precision of a Gaussian distribution.
	"""

	# Compute P-row and corresponding perplexity
	#print('limits',np.max(D),np.min(D),beta)
	P = np.exp(-D.copy() * beta)
	sumP = np.maximum(sum(P),TOL_MIN)
	H = np.log(sumP) + beta * np.sum(D * P) / sumP
	P /= sumP
	return H, P


def x2p0(X=np.array([]), tol=1e-5, perplexity=30.0):
	"""
		Performs a binary search to get P-values in such a way that each
		conditional Gaussian has the same perplexity.
	"""

	# Initialize some variables
	(n, d) = X.shape
	
	sum_X = np.sum(np.square(X), 1)
	D = np.abs(np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X))
	P = np.zeros((n, n))
	beta = np.ones((n, 1))
	logU = np.log(perplexity)

	# Loop over all datapoints
	print("Computing pairwise distances...",(n,d))
	for i in range(n):

		# Print progress
		if i % 1000 == 0:
			print("Computing P-values for point %d of %d..." % (i, n))

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf
		betamax = np.inf
		Di = D[i, np.append(np.r_[0:i], np.r_[i+1:n])]
		(H, thisP) = Hbeta0(Di, beta[i])

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU
		tries = 0
		while np.abs(Hdiff) > tol and tries < 50:
			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy()
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2.
				else:
					beta[i] = (beta[i] + betamax) / 2.
			else:
				betamax = beta[i].copy()
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2.
				else:
					beta[i] = (beta[i] + betamin) / 2.

			# Recompute the values
			#print('B,H,Perp',beta[i],Hdiff,perplexity)
			(H, thisP) = Hbeta0(Di, beta[i])
			Hdiff = H - logU
			tries += 1

		# Set the final row of P
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

	# Return final P-matrix
	#print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)),Hdiff)
	return P


def pca0(X=np.array([]), N0=None):
	"""
		Runs PCA on the NxD array X in order to reduce its dimensionality to
		N dimensions.
	"""

	
	(n, d) = X.shape
	print("Preprocessing the data using PCA...",(n, d))
	#(l, M) = np.linalg.eig(np.dot(X.T, X))
	return np.dot(X - np.tile(np.mean(X, 0), (n, 1)),
				np.linalg.eig(np.dot(X.T, X))[1][:, 0:N0])


def tsne0(X=np.array([]), N=2, N0=50, perplexity=30.0, pca=True):
	"""
		Runs t-SNE on the dataset in the NxD array X to reduce its
		dimensionality to N dimensions. The syntaxis of the function is
		`Y = tsne.tsne(X, N, perplexity), where X is an NxD NumPy array.
	"""

	# Check inputs
	if isinstance(N, float):
		print("Error: array X should have type float.")
		return -1
	if round(N) != N:
		print("Error: number of dimensions should be an integer.")
		return -1

	# Initialize variables
	(n, d) = X.shape
	max_iter = 1000
	initial_momentum = 0.5
	final_momentum = 0.8
	eta = 500
	tol = 1e-5
	min_gain = 0.01
	eps = TOL_MIN
	Y = np.random.randn(n, N)
	dY = np.zeros((n, N))
	iY = np.zeros((n, N))
	gains = np.ones((n, N))

	# Compute P-values
	print('Performing Initial Perplexity Search for ',X.shape)
	if pca:
		P = x2p0(pca0(X.astype(np.float), N0).real, tol, perplexity)
	else:
		P = x2p0(X.astype(np.float),tol,perplexity)
	P += np.transpose(P)
	
	P /= np.maximum(np.sum(P),TOL_MIN)
	P *= 4.									# early exaggeration
	P = np.maximum(P, TOL_MIN)
	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = np.sum(np.square(Y), 1)
		num = -2. * np.dot(Y, Y.T)
		num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
		num[range(n), range(n)] = 0.
		Q = num / np.maximum(np.sum(num),TOL_MIN) # maximum
		# Q = np.maximum(Q,TOL_MIN)
		# print('P',P)
		# print('Q',Q)
		# Compute gradient
		PQ = P - Q
		for i in range(n):
			dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (N, 1)).T *(
																Y[i, :] - Y), 0)

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
				(gains * 0.8) * ((dY > 0.) == (iY > 0.))
		gains[gains < min_gain] = min_gain
		iY *= momentum 
		iY -= eta * (gains * dY)
		Y += iY
		Y -= np.tile(np.mean(Y, 0), (n, 1))

		# Compute current value of cost function
		if (iter + 1) % 100 == 0:
			C = np.sum(P * np.log((P+eps) / (Q+eps)))
			print("Iteration %d: error is %f" % (iter + 1, C))

		# Stop lying about P-values
		if iter == 100:
			P = P / 4.

	# Return solution
	return Y

def dim_reduce(data,N=2,N0=None,perp=30.0,rep='tsne',pca=True):
	#d = pairwise_distance(data_typed[t][k])
	#x2p0(d,entropy_gaussian,tol,perp,n_iter)
	#binary_search(d,np.ones(n),entropy_gaussian,perp*np.ones(n),tol,n_iter)

	# Run Representation
	if rep == 'pca':
		return pca0(data,None)
	elif rep == 'tsne':
		return tsne0(data,N,N0,perp,pca)

		
def plot_props(keys,data_typed,data_rep,data_type):
	return {
		 k: {
		
		  'ax':   {'title' : '', 
					'xlabel': 'x1', 
					'ylabel': 'x2'},
		  
		  'plot':  {'c': np.reshape(data_typed[(
							 'temperatures'+'_'+data_type)][k],(-1,))},
		  
		  'data':  {'plot_type':'scatter',
					'plot_range': np.reshape(data_typed[(
							 'temperatures'+'_'+data_type)][k],(-1,)),
					'data_process':lambda data: np.real(data),
					'domain_process':lambda domain: np.real(domain)},
					
		  'other': {'cbar_plot':True, 'cbar_title':'Temperatures',
				   'cbar_color':'jet','cbar_color_bad':'magenta',
					'label': lambda x='':x,'pause':0.01,
					'sup_legend': False,
					'sup_title': {'t': data_rep + ' Representation - ' +
									   data_type}
					}
		 }
		for k in keys}



if __name__ == "__main__":
#    print("Run Y = tsne.tsne(X, N, perplexity) to perform t-SNE on your dataset.")
#    print("Running example on 2,500 MNIST digits...")
#    X = np.loadtxt("mnist2500_X.txt")
#    labels = np.loadtxt("mnist2500_labels.txt")
#
#    
#    
#    
	# Parse Input Arguments
	parser = argparse.ArgumentParser(description = "Parse Arguments")

	# Add Model Args
	parser.add_argument('-N0','--N0',help = 'Number of Initial Dimensions',
						type=int,default=None)

	parser.add_argument('-N','--N',help = 'Number of Final Dimensions',
						type=int,default=2)#
						
	parser.add_argument('-P','--perp',help = 'Perplexity',
						type=float,default=30.0)#
						
	parser.add_argument('-pca','--pca',help = 'Initially Perform PCA',
						action='store_true')

	parser.add_argument('-repeat','--repeat',help = 'Repeat tSNE',
						action='store_true')
						
	parser.add_argument('--data_dir',help = 'Data Directory',
						type=str,default='dataset/tsne/')
						
	parser.add_argument('--file_dir',help = 'File Output Data Directory',
						type=str,default='dataset/tsne/')
						
	# Parse Args Command
	args = parser.parse_args()


	# Import Data
	temperatures_Ising = ['temperatures_Ising_L20', 'temperatures_Ising_L40',
						  'temperatures_Ising_L80']
	
	temperatures_gauge = ['temperatures_gaugeTheory_L20', 
						  'temperatures_gaugeTheory_L40',
						  'temperatures_gaugeTheory_L80']
	
	ising = ['spinConfigs_Ising_L20','spinConfigs_Ising_L40',
		     'spinConfigs_Ising_L80']
	
	gauge = ['spinConfigs_gaugeTheory_L20', 'spinConfigs_gaugeTheory_L40', 
		     'spinConfigs_gaugeTheory_L80']
	 
	data_files = ising + gauge + temperatures_Ising + temperatures_gauge
	
	data_types_config = ['spinConfigs_Ising','spinConfigs_gauge']
	data_types_temps = ['temperatures_Ising','temperatures_gauge']
	data_obj_format = {k: 'array' for k in data_types_config+data_types_temps}
	data_reps = ['tsne','pca']
	
	data_params =  {'data_files': data_files,
		            'data_types':data_types_config+data_types_temps,
					'data_format': 'npz', 
					'data_obj_format': data_obj_format,
					'data_dir': 'dataset/tsne/',
					'one_hot': [False],
					'data_name_format':['','pca','perp','N0','']}
	 
	data_params.update(vars(args))
	Data_Process().format(data_params,initials=False)
	if data_params['file_dir'] == '':
		data_params['file_dir'] = data_params['data_dir']
	
	
	# Import Data
	
	data,data_sizes,_,_ = Data_Process().importer(data_params,
							data_typing='dict',data_lists=True,upconvert=True,
							directory=data_params['file_dir'])
	
	display(True,True,'Data Imported... \n'+str(data_sizes)+'\n'+ (
												data_params['data_file']+'\n'))

	# Setup Data
	
	# Change keys structure for appropriate plot labels (i.e. L_XX size labels)
	ind_data = [-3,None]
	ind_type = [0,None]
		
	data_typed = dict_modify(data,data_types_config+data_types_temps,
				              f=lambda k,v: v.copy(),i=ind_data,j=ind_type)
	
	data_sizes = dict_modify(data_sizes,data_types_config+data_types_temps,
				              f=lambda k,v: v,i=ind_data,j=ind_type)
	
	data_keys = {t: sorted(list(d.keys())) for t,d in data_typed.items()}
	
	Y = {r: dict_modify(data,data_types_config,
				              f=lambda k,v: [],i=ind_data,j=ind_type)
			for r in data_reps}
	
	Y2 = Y.copy()
	
	data_types_config = [t[slice(*ind_type)] for t in data_types_config]
	data_types_temps  = [t[slice(*ind_type)] for t in data_types_temps]
	
	
	# Setup Plotting
	plot_keys = {}
	plot_bool = {}
	for r in data_reps:
		for t in data_typed.keys():
			plot_keys[r+'_'+t] = data_keys[t]
			plot_bool[r+'_'+t]= True
	
	Data_Process().plot_close()        
	Data_Proc = Data_Process(plot_keys,plot_bool)
	
	comp = lambda x,i: {k:v[:,i] for k,v in x.items() if np.any(v)}
	

	# tSNE and PCA Analysis

	for t in sorted(data_types_config):
		
		for r in data_reps:
			
			if r == 'pca':
				continue
		
			# Check if Data Exists
			params = data_params.copy()
			file_header = r+'_'+t
			file_name = file_header + data_params['data_file']
			params['data_files'] = file_name+ '.'+data_params['data_format']
			data = Data_Proc.importer(params,data_obj_format='dict',
										format='npz')
			if data is not None:
				print('Previous Data Found for',file_name)
				Y[r][t] = data[0][file_name].item()
				
				Data_Proc.plotter(comp(Y[r][t],1),comp(Y[r][t],0),
					 plot_props(Y[r][t].keys(),data_typed,r,t[-5:]),
					 data_key=r+'_'+t)
				
			else:
				print('New Data for',file_name)
				for k in data_keys[t]:   
					print(r,t,k)
					Y[r][t][k] = dim_reduce(data = data_typed[t][k],
										 N = data_params['N'],
										 N0 = data_params['N0'], 
										 perp = data_params['perp'], 
										 rep = r, 
										 pca = data_params['pca'])
					
				Data_Proc.exporter({file_header: Y[r][t]},data_params)
				
			
			Data_Proc.plotter(comp(Y[r][t],1),comp(Y[r][t],0),
					 plot_props(Y[r][t].keys(),data_typed,r,t[-5:]),
					 data_key=r+'_'+t)
	Data_Proc.plot_save(data_params,read_write='a')   
