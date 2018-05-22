# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:54:40 2018
@author: Matt
"""

# Functions for Magnetism Model


import numpy as np
import time


##### Model Functions ########

times = [time.clock()]
def display(print_it=True,time_it=True,m='',
			t0=-2,line_break=0,line_tab=0,time_check=False):
	
	line_break *= '\n' 
	line_tab *= '\t'
	m = line_tab+str_check(m)
	
	if time_it or time_check:
		times.append(time.clock())
		if print_it and not time_check:
			print(m,times[-1]-times[t0],line_break)
		elif print_it:
			print(m,line_break)
	elif print_it:
		print(m,line_break)



def flatten(x,flattenint=True):
	# Return a 1d list of all elements in inner lists of x of arbirtrary shape
	if not (isinstance(x,list)):
		return x
	elif len(x) == 1 and flattenint:
		if isinstance(x[0],tuple) or isinstance(x[0],str):
			return x
		else:
			return x[0]
	xlist = []
	for y in x:
		if isinstance(y,type([])):
			xlist.extend(flatten(y))
		else: 
			xlist.append(y)
	return xlist

def caps(word,every_word=False,sep_char=' ',split_char=' '):
	try:
		if not every_word:
			return word[0].upper()+word[1:].lower()
		else:
			return sep_char.join([w[0].upper()+w[1:].lower() 
								 for w in word.split(split_char)])
	except IndexError:
		return word

def attr_wrapper(attribute={}):
	def attr_decorator(func):
		def  attr_func(*args,**kwargs):
			for a,v in attribute.items():
				setattr(attr_func,a,v(*args,**kwargs))
			return func(*args,**kwargs)
		return attr_func	
	return attr_decorator
	
def str_split(word,split_char='_',sep_char=' '):
    return sep_char.join(word.split(split_char))

def str_check(v):
	if not isinstance(v,str): 
		return str(v).replace('[','').replace(']','')
	else: 
		return v.replace('[','').replace(']','')

def delim_check(d,delim=[' ','.']):
	if isinstance(d,dict):
		for k in d.copy().keys():
			d[(k.replace(delim[0],'').split(delim[1]))[0]] = d.pop(k)
	else:
		for i,v in enumerate(d):
			d[i] = (v.replace(delim[0],'').split(delim[1]))[0]
		
def list_sort(a,j):
	return list(zip(*sorted(list(zip(*a)),key=lambda i:i[j])))


def get_attr(f,attr=None,f0=None,*args):

	try:
		if attr == 'size':
			return getattr(f(*args),attr,lambda : 1)()
		else:
			return getattr(f,attr)
		
	except AttributeError:
		return f0 

def dict_modify(D,T=None,f=lambda k,v: v,i=[0,None],j=[0,None]): 
	if T:
	   return  {t[j[0]:j[1]]: {k[i[0]:i[1]]: f(k,v) for k,v in D.items() 
				if t in k} for t in T}
	else:
	   return {k[i[0]:i[1]]: f(k,v) for k,v in D.items()}

def nested_dict(keys,base_type={}):
	if len(keys) > 0:
		d = {}
		for k in keys[0]:
			d[k] = nested_dict(keys[1:],base_type)
	else:
		d = base_type.copy()
	return d
	   
def array_dict(d):
	# Convert dictionary of arrays into array of dictionaries
	if isinstance(d,dict) and d != {}:
		n = min([len(np.atleast_1d(v)) for v in d.values()])
		return [{k:np.atleast_1d(v)[i]for k,v in d.items()}for i in range(n)],n
	elif d != {}:
		return np.atleast_1d(d),np.size(d)
	else:
		return [{}],1

# Check if variable is dictionary
def dict_check(dictionary,key):
				
	# Check if dict is a dictionary
	if not isinstance(dictionary,dict):
		return dict(zip(key,dictionary))
	else:
		return dictionary
    

def dict_make(vals,keys,val_type='constant'):
	if len(keys)>1 and np.size(keys[0])>1:
		return {k: dict_make(vals,keys[1:],val_type) 
				for k in np.atleast_1d(keys[0])}
	else:
		keys = np.atleast_1d(keys[0])
		if val_type=='callable':
			return {k: vals(k) for k in keys}
		elif val_type=='iterable':
			return {k: vals[i] for i,k in enumerate(keys)}
		elif val_type == 'constant':
			return {k: vals for k in keys}

# Sort 2-dimensional a by elements in 1-d array b
def array_sort(a,b,axis=0,dtype='list'):
	b = np.reshape(b,(-1,))

	if dtype == 'dict':
		return {i: np.reshape(np.take(a,np.where(b==i),axis),
							  (-1,)+np.shape(a)[1:]) 
								for i in sorted(set(b))},sorted(set(b))

	elif dtype == 'list':
		return ([np.reshape(np.take(a,np.where(b==i),axis),
					   (-1,)+np.shape(a)[1:]) for i in sorted(set(b))],
						 sorted(set(b)))
	elif dtype == 'ndarray':
		return (np.array([np.reshape(np.take(a,np.where(b==i),axis),
					   (-1,)+np.shape(a)[1:]) for i in sorted(set(b))]),
						 sorted(set(b)))
	elif dtype == 'sorted':
		return np.concatenate(
							[np.reshape(np.take(a,np.where(b==i),axis),
									   (-1,)+np.shape(a)[1:])
							for i in sorted(set(b))],1), sorted(set(b))

	else:
		return a,sorted(set(b))

 # Converts data X to n+1-length one-hot form        
def one_hot(X,n=None):

	n = int(np.amax(X))+1 if n is None else int(n)+1

	sx = np.shape(np.atleast_1d(X))


	y = np.zeros(sx+(n,),dtype=np.int32)

	for i in range(n):
		p = np.zeros(n)
		np.put(p,i,1)
		y[X==i,:] = p


	return np.reshape(y,(-1,n))



def sort(data,parameters,parameters0=lambda x: {},coupled=False):
	
		# # Sort by params as {param0_i: {param1_j: [param2_k values]}}
		# # i.e) q, L, T
		
		def dim_reduce(arr,axis=0):
			try:
				return np.squeeze(arr,axis)
			except:
				return arr
		
		def return_item(d,item=0,index=0):
			return list(d.items())[index][item]
		
		def nested_dict(keys,base_type={}):
			if len(keys) > 0:
				d = {}
				for k in keys[0]:
					d[k] = nested_dict(keys[1:],base_type)
			else:
				d = np.copy(base_type)
			return d
			
		def build_dict(data,container,parameters,model,bar=False):
			# if bar: 
				# for p in parameters[0].values():
					# print(container[p][0][list(parameters[1].keys())[0]])
			if isinstance(container,dict) and len(parameters)>1:
				for ic,cparam in enumerate(container.keys()):
					for k in data.keys():
						if model[k][return_item(parameters[0],0)] == cparam:
							build_dict({k:data[k]},container[cparam],
							            parameters[1:],model)
							continue
			elif isinstance(container,np.ndarray):
				k,v =  return_item(data,slice(0,2))
				if isinstance(container[0],object) and len(parameters) == 1:
					#container = dim_reduce(container,0)
					container[int(np.where(return_item(parameters[0],1)== 
								model[k][return_item(parameters[0],0)])[0])] = (
																dim_reduce(v,0))
					if not None in container:
						try:
							container = np.concatenate(tuple(container))
						except:
							pass
				elif isinstance(container[0],dict):
						i = np.array(return_item(parameters[0],1)) == (
									    model[k][return_item(parameters[0],0)])
						c = container[i].item()
						for kc in c.keys():
							c[kc][return_item(parameters[1],1) == (
									model[k][return_item(parameters[1],0)]
									)] = dim_reduce(v[0][kc],0)
				elif isinstance(container[0],np.ndarray):
					for p in parameters:
						print(k,p)
						container[0] = data[k].copy()
						for kp,vp in p.items():
							temp = container[0][0][kp]
							if isinstance(vp,str):
								container[0][0][kp] = vp
							else:
								container[0][0][kp] = np.array(vp,
															  dtype=type(vp[0]))
							print(kp,temp,container[0][0][kp])
					container = dim_reduce(container)
			
			# Process Container
			# if bar: 
				# print(parameters)
				# for i in [1,2]:
					# for p in list(parameters[0].values())[0]:
						# print(list(parameters[i].keys())[0])
						# print([m[list(parameters[i].keys())[0]]
								# for m in model_props])
						# print(container[p][0][0][list(parameters[i].keys())[0]])
				# exit()
			try:
				if isinstance(return_item(container,1)[0][0],dict):
					for k in container.keys():
						container[k] = container[k][0][0]
			except:
				pass
		
		# # Data Characteristics
		data = data.copy()
		key0 = return_item(data['sites'])
		
		
		# # Find parameter Coordinates for each dataset
		model_props = data['model_props']
		
		# parameter_sets = {}
		# for k,m in data['model_props'].items():
			# parameter_sets[k] = [m[p] for p in parameters]
		
		# # Find Unique Parameter Coordinates
		# paramets_unique = sorted(map(list,set(map(tuple,d))),
				# key=lambda x: tuple([x[i] for i in range(len(x))]))
		
		
		
		
		
		
		
		sites_size = list(set([np.shape(dim_reduce(v,0)) 
								for v in data['sites'].values()]))
		obs_size = {k: np.shape(dim_reduce(v))for k,v 
						in data['observables'][key0][0].items()}
		
		
		# Order of N Parameters given indicates how data will be sorted
		# Let the last sorting be by the last parameter pN
		pN = parameters[-1]
		pN1 = parameters[-2]
		
		
		
		# Create new Parameters type in Data, with properties values, types etc.
		data['parameters'] = {}
		data['parameters']['values'] = [None for p in parameters]
		data['parameters']['types'] = {}
		data['parameters']['sizes'] = {}

		for i,p in enumerate(parameters):
			types = type(model_props[key0][p])
			vals = sorted(set(np.append([],[m[p]for m in model_props.values()]
															   ).astype(types)))
			data['parameters']['values'][i] = vals
			data['parameters']['types'][p] = types
			data['parameters']['sizes'][p] = np.size(vals)
		
		
		
		
		# Create nested dictionary to sort data by parameters p1...pN-1
		if len(sites_size)>1:
			data['sites_sorted']=nested_dict(data['parameters']['values'][:-1],
					   np.array([None for _ in 
								range(data['parameters']['sizes'][pN])],
								dtype=object))
		else:
			data['sites_sorted']=nested_dict(data['parameters']['values'][:-1],
									np.zeros((data['parameters']['sizes'][pN],)+
														tuple(sites_size[0])))
		
		data['obs_sorted'] = nested_dict(data['parameters']['values'][:-2],
							np.array([
							{k: np.zeros((data['parameters']['sizes'][pN],)+
								tuple(s)) for k,s in obs_size.items()}.copy() 
						   for _ in range(data['parameters']['sizes'][pN1])]))
		data['model_sorted'] = nested_dict(data['parameters']['values'][:-2],
																		 [[{}]])
						 
						 
						 
						 
		# Sort Data into nested dictionary structure
		# Change data structyre of parameter values for easier sorting
		# Depending on whether the last two parameters are coupled,
		# build dictionaries differently.
		# param_vals = data['parameters']['values'].copy()
		# data['parameters']['values'] = []
		# for i,p,d in enumerate(zip(parameters,param_vals)):
			# if i < len(paramval)-1:
				# data['parameters']['values'].append({p:d}) 
			# else:
				# data['parameters']['values'].append(
								# {(pN1,PN): [param_vals[-2], param_vals[-1]})
								
								
								
								
		# Potentially Add other Parameters to Final Sorting Dictionary
		
		data['parameters']['values'] = [{parameters[i]: p} 
							for i,p in enumerate(data['parameters']['values'])]
		
		build_dict(data['sites'],data['sites_sorted'],
				   data['parameters']['values'],data['model_props'])

		display(time_it=False,m='Sites Sorted')
		build_dict(data['observables'],data['obs_sorted'],
				   data['parameters']['values'],data['model_props'])
		
		display(time_it=False,m='Observables Sorted')
		#print(data['model_sorted'])
		#data['parameters']['values'][-1].update(parameters0)
		build_dict(data['model_props'],data['model_sorted'],
				   data['parameters']['values'],data['model_props'],True)
		
		
		# Change back data structure of parameter values
		data['parameters']['values'] = {p: 
										data['parameters']['values'][i].values()
										for i,p in enumerate(parameters)}
		# print(data['model_sorted'])
		
		return data





















##### Other Functions #######

#def vector_func(f,args,keys0=[]):
#    
#    if isinstance(args,dict):
#        
#        arg_no_map = {}
#        arg_map = {}       
#
#        map_args = map_over(args,keys0)
#            
#        # Create dictionary of items to be mapped over, and constants      
#        for k,v in args.items():
#            if map_args.get(k):
#                if isinstance(v,dict):
#                    arg_map[k] = v.values()
#                else:
#                    arg_map[k] = v
#            else:
#                arg_no_map[k] = v
#            
#        # Create list of dictionaries for mapped over arrays
#        map_keys = list(zip(*arg_map.keys()))
#        map_values = list(zip(*(arg_map.values())))
#
#        dict_map = list(map(lambda x: dict(zip(*x)),
#                           zip(map_keys*len(map_values),map_values)))
#        
#        # Map over list of dictionaries
#        return np.array(list(map(lambda x: f(**arg_no_map,**x),dict_map)))
#    
#    else:
#        return f(*args)
#
#def map_over(a,keys0=[]):    
#    return {k: False if k in np.atleast_1d(keys0) else True for k in a.keys()}
#
#

#    
#def delta_f(x,y,f=np.multiply):
#    return np.ones(np.size((x*y)[x==y]))
#
#
#def choose_f(f,i=[0,0,0],f0=None,):
#    if not callable(f):
#        g = np.atleast_1d(f)
#        f = lambda k: g[k]
#    return f(i[0]) if i[1] in np.atleast_1d(i[2]) else f0
#
#
#def list_functions(module):
#    return [f for f in module.values() if type(f)==types.FunctionType]
#
#
#def edges(cluster,neighbour_sites):
#    cluster_edges = []
#    for c in np.atleast_1d(cluster):
#         cluster_edges.append([i for i in np.atleast_1d(c) if len(
#                                         [j for j in np.atleast_1d(c) if j
#                                in neighbour_sites[0][i]]) 
#                                      < len(neighbour_sites[0][i])])
#    return cluster_edges



# # Convert Data to Range
#def convert_to_range(X,sort='max',N=None):
#    # Convert discrete domain data X, into values in range 0...N,
#    # where the new values are indices depending on sort method:
#    # max: values are in range of 0... max(x) 
#    # unique: value are indices of ascending order of set of 
#    #         unique elements in x: 0 ... length(set(x)) 
#    #                     sorted: length(x)+1
#    # int N: values are in range: 0...N
#    
#    sort_method = {'max':   range(int(np.amax(X))+1),
#                   'unique':list(set(X.flatten())),
#                   'int':   range(int(max([np.amax(X),N]))+1)}
#    
#    sorter = lambda X,sort: np.array([[
#                          np.where(sort_method[sort]==i)[0][0] 
#                          for i in x]
#                          for x in np.atleast_2d(X)])
#    
#    return sorter(X,sort)

# Sort Dictionary by other Dictionary
#def dict_sort(dict1,dict2,dtype='dict',reorganize=True):  
#    
#    # Create dict0 as sorted version of dict2 using dict1
#    
#    # Sort dict2 into {key_1: {key_2: { val_1: val_2_array(1_sort)} } }
#    dict0 = {k1: {k2: array_sort(v2,v1,0,dtype) 
#                      for k2,v2 in dict2.items()}
#                      for k1,v1 in dict1.items()}
#    
#    # Reorganize to  {key_1: {key_2: { val_1: val_2_array(1_sort)} } }
#    if reorganize and dtype == 'dict':
#    
#        dict0 = {k1: {v1i: {k2: dict0[k1][k2][v1i] 
#                            for k2 in dict2.keys()}
#                            for v1i in sorted(np.reshape(v1,(-1,)))}                                    
#                            for k1,v1 in dict1.items()}
#                
#    return dict0
#
#
#
#class ParamDict(dict):
#    def __getitem__(self, key):
#        val = dict.__getitem__(self, key)
#        return callable(val) and val(self) or val
#
#
#
#def list_f(l,f,a):
#    f = np.atleast_1d(f)
#    a = np.atleast_1d(a)
#    for i in range(np.size(f)):
#        l.f[i](a[i])
#    return l
#
#def listmultichange(x,i,a=lambda b:b):
#    # Return a list x after changing all jth elements by a function a[n](x[j])
#    # for j indicies in list ielements 
#    # CURRENTLY NOT WORKING PROPERLY
#            
#            if isinstance(a,list):
#                for ia,aa in enumerate(a):
#                    if not callable(aa):
#                        a[ia] = lambda b: aa
#            else:
#                if not callable(a):
#                    a = lambda b: a
#                a = [a]
#            
#            if not isinstance(i,list):
#                i = [i]
#            print(a,i)
#             
#            print(type(a))
#            
#            for n,j in enumerate(i):
#                print(n,j)
#                print(a[0])
#                print(type(a[n]))
#                x[j] = a[0](x[j])
#            return x
#
#
#
#def data_save(self,save=[True,True],data=None,data_path=None,nfiles=1,
#              headers=None,cols=1,*comments):
#
##                           ,self.model_props['observables']('__name__'),
##                           self.model_props['observables']('size'),
##                           lambda i: '%s runtime: %0.5f'%(
##                                           algorithm[i%n_alg],(time.clock()-tau0)/n_iter))
##  
#    
#    
#    # Write Data to File Directory
#    if data_path is None:
#        data_dir = os.getcwd()
#        data_file = data_dir+'/DataSet'
#    else:
#        data_dir = data_path[0]
#        data_file = ''.join(data_path)
#        
#    
#    
#    if save[0]:
#        # Open and write observables to a file in a specific directory
#        if not data :
#            if not(os.path.isdir(data_dir)):
#                    os.mkdir(data_dir)
#            return
#        
#        for n in range(nfiles):
#            
#            if save[1]:
#                np.savez_compressed(
#                        data_file+('_%d'%n if nfiles>1 else ''),a=data[n])
#            
#            else:   
#                # Data Structure of Data Headers, and Number of Collumns per Header
#
#                headers = np.atleast_1d(headers)
#                cols = np.atleast_1d(cols)
#                
#                # Make observables headers for file
#                file = open(data_file+('_%d'%n if nfiles>1 else '')+
#                            '.txt','w')
#                header = []
#                
#                # Write Observables to File
#                
#                if not (headers is None): 
#                    for i,h in enumerate(headers):
#                        for j in range(cols[i]):
#                            header.append(h+'_'+str(j+1) 
#                                            if cols[i] > 1
#                                            else h)
#                    file.write('\t'.join(header) + '\n')
#                
#                # Convert lists of lists of observables to array
#                for data_n in [list(flatten(x)) for 
#                               x in flatten(data[n])]:
#                    dataline = ''
#                    for d in data_n:
#                            dataline += '%0.8f \t' %(float(d))
#                    dataline += '\n'
#                    file.write(dataline)
#                if comments:
#                    for c in comments:
#                        file.write(str(c(n))+'\n')
#                
#                file.close()
#
#
#
#
#def listchange(x,i,a=lambda b:b):
#    # Return a list x after changing its ith element by a function a(x[i])
#    # CURRENTLY only accepts function arguments for a 
#    # (desire constant arguments that are converted to functions lamba x: a )
#    
##    if not callable(a):
##        print('a is not a function')
##        a = lambda c: a
##    
##    print(a(1))
###    
###    f = lambda y:a    
###    print(a(0))
###    print(x[i])
###    print(f(x[i]))
#    x[i] = a(x[i])
#    return x
#
#
#    
#def fappend(x,F):
#   for f in F:
#       if not callable(f):
#           f = lambda : f
#       #print(f())
#       ft = f()
#       if not((isinstance(ft,np.array)) or (isinstance(ft,list))):
#           x = np.append(x,ft)
#       else:
#           for j in ft:
#               x = np.append(x,j)
#   return x
#
#def realimag(x,tol=1e-14):
#    # Return entirely real or imaginary components of complex x, 
#    # if the conjugate component is zero, or below a tolerance
#    r = x.real
#    i = x.imag
#    
#    
#    if i == 0:
#        return x
#    
#    if abs(r) <= tol:
#        return i
#    elif abs(i)<=tol:
#        return r
#    else:
#        return x
#
#def signed_val(x):
#    xs = np.shape(x)
#    xf = np.array(x).flatten()
#    ri = np.stack((np.real(xf),np.imag(xf)),1)
#    xms = np.sign(ri[np.arange(np.size(ri,0)), np.argmax(np.abs(ri),1)])*np.abs(xf)
#    return np.reshape(xms,xs)
#    
#def index(x,i):
#    # Return array of elements in list x, specficied in indicices in list i
#    return [x[j] for j in i] if isinstance(i,list) else x[i]
#    
#def listindex(x,i):
#    #print(x)
#    if not isinstance(x,list):
#        return x
#    if not isinstance(i,list):
#        return x[i]
#    elif len(i) == 1:
#        try:
#            return [ y if not isinstance(y,list) 
#                else (y[i[0]]) if len(y)>= i[0] 
#                else y[-1] for y in x]
#        except IndexError: 
#            return [y for y in x]
#        
#    else:    
#        return listindex([y if not isinstance(y,list) 
#                else (y[i[0]]) if len(y)>= i[0]+1 
#                else y[-1]  for y in x]
#                ,i[1:])
#
#
#
#
#
#
## Other Functions
#
#    def dEN(self,J=1):
#        dN = np.array([[0,0],
#              [0,1],
#              [0,2],
#              [0,3],
#              [0,4], 
#              [1,0],       
#              [1,1],       
#              [1,2],       
#              [1,3],       
#              [2,0],
#              [2,1],       
#              [2,2],       
#              [3,0],       
#              [3,1],       
#              [4,0]
#             ])
#        dE = -J*(dN[:,1] - dN[:,0])
#        
#        print(dE)
#        
#        
#        PdE = lambda q: [(5*q-17)/(q-1),7/(q-1),3/(q-1),1/(q-1),1/(q-1),
#               (3*q-3)/(q-1),4/(q-1),1/(q-1),1/(q-1),
#               (2*q-5)/(q-1),2/(q-1),1/(q-1),
#               (q-2)/(q-1),1/(q-1),
#               1]
#        
#        
#    
##        plt.figure()
#        for q  in [2,3,4,5,6]:
#            print('Ptot = %0.4f, %d'%(sum(PdE(q)),q))
##            plt.plot(dE,PdE(q),'*',label=str(q))
##        plt.legend()
#        return
#    
#    
#    def correlation(self,sites,neighbours,T,r = None):
#        # Calculate correlation function c(r) = <s_i*s_j> 
#        # for all spin pairs, where the ri-ir neighbour distance is
#        #  r = {1:L/2}
#        # i.e) [(1/(self.Nspins))*
#        #      sum(sum(n*self.sites[j] for n in self.neighbours(rr)[j])
#        #       - ((sum(self.sites)/self.Nspins)**2) 
#        #       for j in range(self.Nspins)) for rr in r]
#        
#        Nspins = np.size(sites)
#        
#        
#        if r is None:
#            r = np.arange(np.arange(1,np.ceil(
#                                np.power(Nspins,1/self.d)/2),
#                                dtype=np.int64))
#    
#        return list((1/2)*((1/Nspins)**2)*np.sum(sites[:,np.newaxis]
#                *sites[neighbours[r-1]],(1,2)) - (
#                (1/Nspins**2)*np.sum(sites))**2)
#   
#            
#    def Tcrit(self,d):
#        # Declare the critical Ising Model Temperature in d-dimensions
#        if d >= 4:
#            Tc = self.orderparam[1]
#        elif d == 1:
#            Tc = 0
#        elif d == 2:
#            Tc = 1.0/np.log(1.0 + np.sqrt(self.q))*self.orderparam[1]
#        else: # self.d == 3:
#            Tc = None
#        return Tc




#def self_ref(f):
#    f.__defaults__ = f.__defaults__[:-1] + (f,)
#    return f
#
#
#@self_ref
#def foo(self,x):
#    self.y = 0
#    return x + self.y
#
#x = 2
#foo(x)

#def is_mod_function(mod, func):
#    return inspect.isfunction(func) and inspect.getmodule(func) == mod
#
#def list_function(module):
#    return [func.__name__ for func in module.__dict__.itervalues() 
#            if is_mod_function(module, func)]
#
#
#def list_functions(module):
#    def is_local(object):
#        return (isinstance(object, types.FunctionType)) and (
#                object.__module__ == __name__)
#    return [name for name, value in inspect.getmembers(
#            module.modules[__name__], predicate=is_local)]        

    
    
    
