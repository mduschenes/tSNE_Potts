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
	print("Computing pairwise distances...",n,'points')
	for i in range(n):

		# Print progress
		if i+1 % 500 == 0:
			print("Computing P-values for point %d of %d..." % (i+1, n))

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
	print("Processing the data using PCA...",(n, N0 if N0 is not None else d))
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
		P = x2p0(X[:,:N0].astype(np.float),tol,perplexity)
	P += np.transpose(P)
	
	P /= np.maximum(np.sum(P),TOL_MIN)
	P *= 4.									# early exaggeration
	P = np.maximum(P, TOL_MIN)
	Q = np.empty(np.shape(P))
	C = 0.
	# Run iterations
	print("Processing the data using tSNE...",(n, N0 if N0 is not None else d))
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = np.sum(np.square(Y), 1)
		num = 1. / (1. + np.add(np.add(-2. * np.dot(Y, Y.T), sum_Y).T, sum_Y))
		np.fill_diagonal(num,0) #num[range(n), range(n)] = 0.
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
		if (iter)+1 % 100 == 0:
			C = np.sum(P * np.log((P+eps) / (Q+eps)))
			print("Iteration %d: error is %f" % (iter + 1, C))

		# Stop lying about P-values
		if iter == 100:
			P /= 4.

	# Return solution
	return Y

def dim_reduce(data,N=2,N0=None,perp=30.0,rep='tsne',pca=True,**kwargs):
	#d = pairwise_distance(data_typed[t][k])
	#x2p0(d,entropy_gaussian,tol,perp,n_iter)
	#binary_search(d,np.ones(n),entropy_gaussian,perp*np.ones(n),tol,n_iter)

	# Run Representation
	if rep == 'pca':
		return pca0(data,None)
	elif rep == 'tsne':
		return tsne0(data,N,N0,perp,pca)

		
def plot_props(keys,data_typed,data_rep,data_type,orientation):
	FONT_SIZE = 22

	
	if orientation:
		cbar_plot =lambda i,k: True if i == len(keys)-1 else False
		xlabel = lambda i,k: r'x_1'
		ylabel = lambda i,k: r'x_2' if i == 0 else ''
	else:
		cbar_plot =lambda i,k: True #if i == len(keys)-1 else False
		xlabel = lambda i,k: r'x_1' if i == len(keys)-1 else ''
		ylabel = lambda i,k: r'x_2' 
		
		
	return {
		 k: {
		
		  'ax':   {'title' : 'L = %s'%k[-2:], 
					'xlabel': xlabel(i,k), 
					'ylabel': ylabel(i,k),
					'xlim': lambda lims: tuple([x*1.1 for x in lims]),
					'ylim': lambda lims: tuple([x*1.1 for x in lims])},
		  'ax_attr': {'get_xticklabels':{'visible':False,
													 'fontsize':15},
					  'xaxis': {'ticks_position': 'none'},
					  
					  'get_yticklabels':{'visible':False,
											 'fontsize':15},
					   'yaxis': {'ticks_position': 'none'}},
		  'plot':  {'s':40,
					'c': np.reshape(data_typed[(
							 'temperatures'+'_'+data_type)][k],(-1,))},
		  
		  'data':  {'plot_type':'scatter',
					'data_process':lambda data: np.real(data),
					'domain_process':lambda domain: np.real(domain)},
					
		  'other': {'label': lambda x='':'',#'L = %s'%x[-2:],
								'cbar': {										
										'plot':cbar_plot(i,k),
										'title':{'label':'Temperatures',
												 'size':25},
										'color':'bwr',
										'midpoint': 2.269,
										'labels': {'fontsize': FONT_SIZE}},
								'pause':0.01,
								'sup_legend': False,
								'legend': {'prop':{'size': FONT_SIZE}
										   #'loc':(0.1,0.85)
										  },
								'sup_title': {'t': ''}

								}
		}
		for i,k in enumerate(sorted(keys))}



if __name__ == "__main__":
#    print("Run Y = tsne.tsne(X, N, perplexity) to perform t-SNE on dataset.")
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

	parser.add_argument('-Ns','--Ns',help = 'Step Size of Initial Samples',
						type=int,default=1)#
						
	parser.add_argument('-P','--perp',help = 'Perplexity',
						type=float,default=30.0)#
						
	parser.add_argument('-pca','--pca',help = 'Initially Perform PCA',
						action='store_true')

	parser.add_argument('-import_files','--import_files',help = 'Import Files',
						action='store_true')
						
	parser.add_argument('--models',help = 'Data Set Models',
						type=str, nargs = '+', default = ['Ising','gauge'])

	parser.add_argument('--data_reps',help = 'Data Representations',
						type=str, nargs = '+', default = ['tsne','pca'])
						
	parser.add_argument('--file_name',help = 'File to Import',
						type=str,default=None)
						
	parser.add_argument('-plot','--plot',help = 'Plot Data',
						action='store_true')
						
	parser.add_argument('--data_dir',help = 'Data Output Directory',
						type=str,default='dataset/tsn2/')
						
	parser.add_argument('--file_dir',help = 'File Dataset Directory',
						type=str,default='dataset/tsne/')
						
	# Parse Args Command
	args = parser.parse_args()


	# Import Data
	
	def formatter(file,inds):
		inds = inds.copy()
		if inds[1] is not None and (inds[1] >= 0 or inds[1] < -1):
			while file[inds[1]+1].isdigit():
				inds[1] += 1
		if inds[1] == -1 or inds[1] == len(file): 
			inds[1] = None
		else:
			inds[1] += inds[1]
		return file[inds[0]:inds[1]]
	
	joiner = lambda *strings: '_'.join([s for s in strings])
	
	models = args.models
	delattr(args,'models')
	
	header_configs = 'spinConfigs'
	header_temperatures = 'temperatures'
	configs = {}
	temperatures = {}
	data_params = {}
	inds = {'Ising': [-3,-1],'gauge': [-3,-1],'potts': [6,7]}
	
	temperatures['Ising'] = ['temperatures_Ising_L20', 'temperatures_Ising_L40',
						  'temperatures_Ising_L80']
	
	temperatures['gauge'] = ['temperatures_gaugeTheory_L20', 
						  'temperatures_gaugeTheory_L40',
						  'temperatures_gaugeTheory_L80']
	
	temperatures['potts'] = ['Potts_q4temp_data_t_keys',
							 'Potts_q6temp_data_t_keys',
							 'Potts_q10temp_data_t_keys']
	
	
	configs['Ising'] = ['spinConfigs_Ising_L20','spinConfigs_Ising_L40',
						'spinConfigs_Ising_L80']
	
	configs['gauge'] = ['spinConfigs_gaugeTheory_L20', 
						'spinConfigs_gaugeTheory_L40', 
						'spinConfigs_gaugeTheory_L80']
	 
	configs['potts'] = ['Potts_q4temp_data_t','Potts_q6temp_data_t',
						 'Potts_q10temp_data_t']
		
	
	configs_sets = sum([list(joiner(header_configs,s,formatter(c,inds[s]))
								for c in configs[s]) for s in models],[])
	temperatures_sets = sum([list(joiner(header_temperatures,s,
											   formatter(t,inds[s]))
							for t in temperatures[s])  for s in models],[])
	 
	data_files = sum([c for k,c in configs.items() if k in models] + 
					 [t for k,t in temperatures.items()if k in models],[])
	
	data_types_configs = [joiner(header_configs,s) for s in models]
	data_types_temps = [joiner(header_temperatures,s) for s in models]
	
	# data_files = ising + gauge + temperatures_Ising + temperatures_gauge
	# data_types_config = ['spinConfigs_Ising','spinConfigs_gauge']
	# data_types_temps = ['temperatures_Ising','temperatures_gauge']
	data_obj_format = {k: 'array' for k in data_types_configs+data_types_temps}
	data_reps = args.data_reps
	
	data_params =  {'data_files': data_files,
		            'data_types':data_types_configs+data_types_temps,
					'data_format': 'npz', 
					'data_obj_format': data_obj_format,
					'data_dir': 'dataset/tsne/',
					'one_hot': [False],
					'data_name_format':['','pca','perp','N0','']}
	
	data_params['data_sets'] = configs_sets + temperatures_sets
	data_params.update(vars(args))
	Data_Process().format(data_params,initials=False)
	
	if data_params['file_dir'] == '':
		data_params['file_dir'] = data_params['data_dir']
	
	# Import Data
	
	data,data_sizes,_,_ = Data_Process().importer(data_params,
							data_typing='dict',data_lists=True,upconvert=True,
							directory=data_params['file_dir'],disp=True)
	
	display(True,False,'Data Imported... \n'+str(data_sizes)+ (
												'\nwith parameters: ' )+ (
												data_params['data_file']+'\n'))
	# Setup Data
	
	# Change keys structure for appropriate plot labels (i.e. L_XX size labels)
	ind_data = [-3,None]
	ind_type = [0,None]
		
	data_typed = dict_modify(data,data_types_configs+data_types_temps,
				              f=lambda k,v: v.copy(),i=ind_data,j=ind_type)
	data_sizes = dict_modify(data_sizes,data_types_configs+data_types_temps,
				              f=lambda k,v: v,i=ind_data,j=ind_type)
	
	data_keys = {t: sorted(list(d.keys())) for t,d in data_typed.items()}

	data_types_configs = [t[slice(*ind_type)] for t in data_types_configs]
	data_types_temps  = [t[slice(*ind_type)] for t in data_types_temps]
	
	Y = {r: dict_modify(data,data_types_configs,
				              f=lambda k,v: [],i=ind_data,j=ind_type)
			for r in data_reps}
	
	Y2 = Y.copy()
	

	for t in data_types_temps:
		print({(t,k): np.shape(v) for k,v in data_typed[t].items()})
	
	# Setup Plotting
	orientation = False
	if orientation:
		fig_size = (20,16)
	else:
		fig_size = (14,20)
	imax = 2
	plot_keys = {}
	plot_bool = {}
	for r in data_reps:
		for t in data_typed.keys():
			if t not in data_types_temps:
				plot_keys[r+'_'+t] = data_keys[t]
				plot_bool[r+'_'+t]= True
	
	Data_Process().plot_close()        
	Data_Proc = Data_Process(plot_keys,plot_bool,orientation=orientation)
	
	comp = lambda x,i: {k:v[:,i] for k,v in x.items() if np.any(v)}
	
	# tSNE and PCA Analysis

	for t in sorted(data_types_configs):
		
		for r in data_reps:
			
		
		
			# Check if Data Exists
			data_key = r+'_'+t
			params = data_params.copy()
			params.pop('data_sets');
			file_header = data_key
			if not args.file_name:
				file_name = file_header
			else:
				file_name = args.file_name 
			params['data_files'] = file_name

			if (file_header in file_name) and args.import_files:
				data = Data_Proc.importer(params,data_obj_format='dict',
										format='npz',
										directory = data_params['data_dir'])
			else:
				data = None
				
			if data is not None:
				print('Previous Data Found for',file_name)
				Y[r][t] = list(data[0].values())[0]
				if args.plot:
					Data_Proc.plotter(comp(Y[r][t],1),comp(Y[r][t],0),
								plot_props(Y[r][t].keys(),data_typed,r,t[-5:],
											orientation),
								data_key=data_key)
					break
			else:
				print('New Data for',file_name)
				
				if args.import_files:
					Data_Proc.plot[data_key] = False
					continue
				for k in data_keys[t][:1]: 
					print(r,t,k)
					Y[r][t][k] = dim_reduce(data_typed[t][k],rep=r,**data_params)
										 
					
				Data_Proc.exporter({file_name: Y[r][t]},data_params)
				
				if args.plot:
					Data_Proc.plotter(comp(Y[r][t],1),comp(Y[r][t],0),
					 plot_props(Y[r][t].keys(),data_typed,r,t[-5:],
									orientation),
					 data_key=r+'_'+t)
	if args.plot:
		Data_Proc.plot_save(data_params,read_write='ow',
							fig_size=fig_size,format='pdf')   
