# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:54:40 2018
@author: Matt
"""

# Functions for Magnetism Model


import numpy as np
import time,argparse,re,copy

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
	def capitalize(s):
		return s[0].upper()+s[1:].lower() 

	try:
		if every_word is False:
			return capitalize(word) 
		elif every_word is True:
			word_sep = sep_char.join([capitalize(w)
								 for w in word.split(split_char)])
			return sep_char.join([capitalize(w)
								 for w in word_sep.split(sep_char)])
		else:
			return sep_char.join([w for w in word.split(sep_char)])
	except IndexError:
		return word

		
		
class Recurse(Exception):
	def __init__(self,*args,**kwargs):
		self.args = args
		self.kwargs = kwargs
		
def recurse(*args,**kwargs):
	raise Recurse(*args,**kwargs)
	
def tail_recursive(func):
	def recursive_decorator(*args,**kwargs):	
		while True:
			try:
				func(*args,**kwargs)
				return
			except Recurse as r:
				args = r.args
				kwargs = r.kwargs
				continue
	return recursive_decorator

	
def None_check(type_func):
	def type_wrapper(**kwargs):
		if kwargs['obj'] in ['None','']:
			return None
		elif kwargs['obj'] in ['False','0']:
			return False
		else:
			return type_func(**kwargs)
	return type_wrapper
	
@None_check
def typer(obj,ind,item,types):
	return types[item][ind](obj)

def type_arg(func,**kwargs): 

	class TYPE_ARG(argparse.Action):
		def __call__(self, parser, args, values, 
					 option_string=None):
			setattr(args,self.dest,func(values,**kwargs))
	return TYPE_ARG
	
def dict_arg(args,key_type=str,val_type=float):
	if args == [] or len(args) % 2 != 0:
		raise argparse.ArgumentTypeError('Not key-value pairs args')
		return {}

	types = {}
	for k,t in zip(['keys','vals'],[key_type,val_type]):
		if isinstance(t,type): 
			types[k] = [t for _ in range(len(args))]
		else:
			types[k] = t

	dict_arg = {}
	for i,(k,v) in enumerate(zip(args[0::2], args[1::2])):
		try:
			dict_arg[typer(obj=k,ind=i,item='keys',types=types)] = typer(
										obj=v,ind=i,item='vals',types=types)
		except TypeError as e:
			dict_arg[k] = v
	return dict_arg	
	

def dict_str(d,sep_char=' ',split_char=' '):
	return '_'.join(list(map(lambda x: str_check(x[0])[slice(0,1,1)] + (
								str_check(x[1])), d.items())))
	
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

def hashable(a):
	try: 
		hash(a)
	except TypeError:
		if isinstance(a, (list, tuple, np.ndarray)):
			a = tuple(a) if len(a) > 1 else a[0]
	return a
	
def str_check(v,float_lim=None):
	if not isinstance(v,str): 
		if isinstance(v,float) and float_lim is not None: 
			v = str(v)[:float_lim+2]
		else:
			v = str(v) 
		return v.replace('[','').replace(']','')
	else: 
		return v.replace('[','').replace(']','')

def line_break(string,line_length,delim=' ',line_space=''):
	string_split = string.split(delim)
	line_length = max(line_length,max([len(s) for s in string_split]))
	string = []
	while string_split != []:
		n = 0
		t = []
		while string_split != [] and n+len(string_split[0]) <= line_length:
			s = string_split.pop(0)
			n += len(s)
			t.append(s)
		string.append(delim.join(t))
	return ('\n'+line_space).join(string)

def list_sort(a,j):
	return list(zip(*sorted(list(zip(*a)),key=lambda i:i[j])))

def list_sort_unique(a,j):
	return a

def index_nested(a,i):
	i = np.atleast_1d(i)
	if len(i) > 1:
		return index_nested(a[i[0]],i[1:])
	else:
		return a[i[0]]


def delim_check(d,delim=[' ','.']):
	if isinstance(d,dict):
		for k in d.copy().keys():
			d[(k.replace(delim[0],'').split(delim[1]))[0]] = d.pop(k)
	else:
		for i,v in enumerate(d):
			d[i] = (v.replace(delim[0],'').split(delim[1]))[0]
		
def list_sort(a,j):
	natural_sort = 	lambda s: [int(t) if t.isdigit() else t.lower() 
										for t in re.split('(\d+)', s)]
	return list(zip(*sorted(list(zip(*a)),key=lambda i: natural_sort(i[j]))))


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

def dict_reorder(d,keys=None,return_inner=False):
	# Reorganize dictionary of dictionaries containing keys, into 
	# dictionary of the fields of the dictionaryies associated with these keys.
	if all([isinstance(v,dict) for v in d.values()]):
		if keys is None:
			keys = list(set(flatten([list(k.keys()) for k in d.values()])))
		keys = np.atleast_1d(keys)
		if return_inner and (len(keys) == 1):
			return {k: v.get(keys[0]) for k,v in d.items()}
		else:
			return {key: {k: v.get(key) for k,v in d.items()} for key in keys}
	else:
		if keys is None:
			keys = list(set([k for k in d.values()]))
		keys = np.atleast_1d(keys)
		if return_inner and len(keys) == 1:
			return [k for k,v in d.items() if v == keys[0]]
		else:
			return {key: [k for k,v in d.items() if v == key] for key in keys}	
		return d


def dict_feed(d1,d2,keys=None,return_inner=False,direct_access_d2=False):
	# Reorganize dictionary of dictionaries containing keys, into 
	# dictionary of the fields of the dictionaryies associated with these keys.
	# These dictionaries values are fed as input to d2.
	d1 = copy.deepcopy(d1)
	d2 = copy.deepcopy(d2)
	if all([isinstance(v,dict) for v in d1.values()]):
		if keys is None:
			keys = list(set(flatten([list(k.keys()) for k in d1.values()])))
		keys = np.atleast_1d(keys)

		if (not isinstance(d2,dict)) or (not direct_access_d2):
			d2 = {k: d2 for k in keys}

		key = keys[0]
		v = list(d1.values())[0]

		if return_inner and len(keys) == 1:
			return {k: index_nested(d2[keys[0]],v[keys[0]]) 
							for k,v in d1.items()}
		else:
			return {key: {k: index_nested(d2[key],v[key]) 
							for k,v in d1.items()} for key in keys}
	else:
		if keys is None:
			keys = list(set([k for k in d.values()]))
		keys = np.atleast_1d(keys)
		if return_inner and len(keys) == 1:
			return [index_nested(d2[k],v) 
							for k,v in d1.items() if k == keys[0]]
		else:
			return {key: [index_nested(d2[k],v) 
							for k,v in d1.items() if k == key] for key in keys}	
		return d1

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

	f = lambda a,b,i: np.reshape(np.take(a,np.where(b==i),axis),
									   (-1,)+np.shape(a)[1:])
	b_sorted = sorted(set(b))

	if dtype == 'dict':
		return {i: f(a,b,i) for i in b_sorted},b_sorted
	elif dtype == 'list':
		return [f(a,b,i) for i in b_sorted], b_sorted
	elif dtype == 'ndarray':
		return np.asarray([f(a,b,i) for i in b_sorted]),b_sorted
	elif dtype == 'sorted':
		return np.concatenate([f(a,b,i) for i in b_sorted],1),b_sorted
	else:
		return a,b_sorted

 # Converts data X to n+1-length one-hot form        
def one_hot(X,n=None):

	n = int(np.amax(X))+1 if n is None else int(n)+1

	sx = np.shape(np.atleast_2d(X))

	if np.size(sx) > 2 or sx[1] > 1:
		return X

	y = np.zeros(sx+(0,n),dtype=np.int32)

	for i in range(n):
		p = np.zeros(n)
		np.put(p,i,1)
		y[X==i,:] = p


	return np.reshape(y,(-1,n))


def dim_reduct(arr,axis=0):
	try:
		return np.squeeze(arr,axis)
	except:
		return arr
			
def return_item(d,item=0,index=0):
	return list(d.items())[index][item]
	

	
# def wolff_rec(self,sites, cluster, neighbours,
					# N_sites,N_neighbours, T, update_status,
					# state_update, state_gen, state_int):      


	# @tail_recursive
	# def cluster_update(i):
		# cluster_bool[i] = True
		# cluster[i] = cluster_value0
		# for j in neighbours[i]:
			# if sites[j] == cluster_value0 and not cluster_bool[j] and (
					# state_update[T] > np.random.random()):
				# recurse(j)
		# return
	

	# def node_gen(i):
		# return (j for j in neighbours[i] if (not cluster_bool[j]) and 
										# (sites[j] == cluster_value0))
	# def cluster_add(i):
		# cluster_bool[i] = True
		# cluster[i] = cluster_value0
		# return
	
	# # Cluster Function
	# # def cluster_update(i):
		# # cluster_bool[i] = 1
		# # cluster[i] = cluster_value0
		# # for j in neighbours[i]:
			# # if sites[j] == cluster_value0 and not cluster_bool[j] and (
					# # state_update[T] > np.random.random()):
				# # cluster_update(j)
		# # return
	
	# # Create Cluster Array and Choose Random Site
	# isite = np.random.randint(N_sites)
	# cluster_value0 = sites[isite]
	# cluster[:] = 0 #np.zeros(N_sites)
	# # Perform cluster algorithm to find indices in cluster
	# cluster_update(isite)

	# # Update spins in cluster to new value
	# sites[cluster_bool] = state_gen(1,cluster_value0)
	# return	




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

    
    
    
# def wolff_rec(self,sites, cluster, cluster_bool, neighbours,
						# N_sites,N_neighbours, T, update_status,
						# state_update, state_gen, state_int):      
	

		# @tail_recursive
		# def cluster_update(i):
			# cluster_bool[i] = True
			# cluster[i] = cluster_value0
			# for j in (j for j in neighbours[i] if (not cluster_bool[j]) and 
											# (sites[j] == cluster_value0)):
				# if state_update[T] > np.random.random():
					# recurse(j)
			# return
		

		# def node_gen(i):
			# return (j for j in neighbours[i] if (not cluster_bool[j]) and 
											# (sites[j] == cluster_value0))
		# def cluster_add(i):
			# cluster_bool[i] = True
			# cluster[i] = cluster_value0
			# return
		
		
		
		# # Create Cluster Array and Choose Random Site
		# isite = np.random.randint(N_sites)
		# cluster_value0 = sites[isite]
		# cluster[:] = 0 #np.zeros(N_sites)
		# # Perform cluster algorithm to find indices in cluster
		# cluster_update(isite)

		# # Update spins in cluster to new value
		# sites[cluster_bool] = state_gen(1,cluster_value0)
		# return

	
	# def wolff_while(self,sites, cluster, cluster_bool, neighbours,
						# N_sites,N_neighbours, T, update_status,
						# state_update, state_gen, state_int):      
		

		# def node_gen(i):
			# return list(j for j in neighbours[i] if (not cluster_bool[j]) and 
											# (sites[j] == cluster_value0))
		# def cluster_add(i):
			# cluster_bool[i] = True
			# cluster[i] = cluster_value0
			# return
		
		
		
		# # Create Cluster Array and Choose Random Site
		# isite = np.random.randint(N_sites)
		# cluster_value0 = sites[isite]
		# #cluster[:] = 0 #np.zeros(N_sites)
		
		# # Perform cluster algorithm to find indices in cluster
		# J0 = node_gen(isite)
		# J = J0.copy()
		# cluster_add(isite)
		
		# while J != []:
			# J0 = J.copy()
			# J = []
			# #print('J0',J0)
			# #print('Jinit',J)
			# for j in J0:
				# if not cluster_bool[j] and state_update[T] > np.random.random():
					# #print('Added ',j)
					# cluster_add(j)
					# # try:
						# # J.remove(j)
					# # except ValueError:
						# # pass
					# J.extend(node_gen(j))
					# #print('Jupd',J)

		# # Update spins in cluster to new value
		# sites[cluster_bool] = state_gen(1,cluster_value0)
		# return
		