# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:54:40 2018
@author: Matt
"""

# Functions for Magnetism Model


import numpy as np
import os, time
import types




##### Model Functions ########

times = [time.clock()]
def display(printit=False,timeit=False,m=''):
    if timeit:
        times.append(time.clock())
        if printit:
            print(m,times[-1]-times[-2])
    elif printit:
        print(m)



def flatten(x,flattenint=True):
    # Return a 1d list of all elements in inner lists of x of arbirtrary shape
    if  not isinstance(x,list):
        return x
    elif len(x) == 1 and flattenint:
        return x[0]
    xlist = []
    for y in x:
        if isinstance(y,type([])):
            xlist.extend(flatten(y))
        else: xlist.append(y)
    return xlist

def caps(word):
    try:
        return word[0].upper()+word[1:].lower()
    except IndexError:
        return word

def list_functions(module):
    return [f for f in module.values() if type(f)==types.FunctionType]


def get_attr(f,attr=None,f0=None,*args):

    try:
        if attr == 'size':
            return getattr(f(*args),attr,lambda : 1)()
        else:
            return getattr(f,attr)
        
    except AttributeError:
        return f0 

def delta_f(x,y,f=np.multiply):
    return f(x,y)[x==y]

def choose_f(f,i=[0,0,0],f0=None,):
    if not callable(f):
        g = np.atleast_1d(f)
        f = lambda k: g[k]
    return f(i[0]) if i[1] in np.atleast_1d(i[2]) else f0


# Check if variable is dictionary
def dict_check(dictionary,key):
                
    # Check if dict is a dictionary
    if not isinstance(dictionary,dict):
        return dict(zip(key,dictionary))
    else:
        return dictionary
    
def dict_modify(D,T=None,f=lambda v: v,i=[0,None],j=[0,None]): 
   if T:
       return  {t[j[0]:j[1]]: {k[i[0]:i[1]]: f(v) for k,v in D.items() 
                if t in k} for t in T}
   else:
       return {k[i[0]:i[1]]: f(v) for k,v in D.items()}
    



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

 # Convert Data to Range
def convert_to_range(X,sort='max',N=None):
    # Convert discrete domain data X, into values in range 0...N,
    # where the new values are indices depending on sort method:
    # max: values are in range of 0... max(x) 
    # unique: value are indices of ascending order of set of 
    #         unique elements in x: 0 ... length(set(x)) 
    #                     sorted: length(x)+1
    # int N: values are in range: 0...N
    
    sort_method = {'max':   range(int(np.amax(X))+1),
                   'unique':list(set(X.flatten())),
                   'int':   range(int(max([np.amax(X),N]))+1)}
    
    sorter = lambda X,sort: np.array([[
                          np.where(sort_method[sort]==i)[0][0] 
                          for i in x]
                          for x in np.atleast_2d(X)])
    
    return sorter(X,sort)

















##### Other Functions #######


# Sort Dictionary by other Dictionary
def dict_sort(dict1,dict2,dtype='dict',reorganize=True):  
    
    # Create dict0 as sorted version of dict2 using dict1
    
    # Sort dict2 into {key_1: {key_2: { val_1: val_2_array(1_sort)} } }
    dict0 = {k1: {k2: array_sort(v2,v1,0,dtype) 
                      for k2,v2 in dict2.items()}
                      for k1,v1 in dict1.items()}
    
    # Reorganize to  {key_1: {key_2: { val_1: val_2_array(1_sort)} } }
    if reorganize and dtype == 'dict':
    
        dict0 = {k1: {v1i: {k2: dict0[k1][k2][v1i] 
                            for k2 in dict2.keys()}
                            for v1i in sorted(np.reshape(v1,(-1,)))}                                    
                            for k1,v1 in dict1.items()}
                
    return dict0



class ParamDict(dict):
    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        return callable(val) and val(self) or val



def list_f(l,f,a):
    f = np.atleast_1d(f)
    a = np.atleast_1d(a)
    for i in range(np.size(f)):
        l.f[i](a[i])
    return l

def listmultichange(x,i,a=lambda b:b):
    # Return a list x after changing all jth elements by a function a[n](x[j])
    # for j indicies in list ielements 
    # CURRENTLY NOT WORKING PROPERLY
            
            if isinstance(a,list):
                for ia,aa in enumerate(a):
                    if not callable(aa):
                        a[ia] = lambda b: aa
            else:
                if not callable(a):
                    a = lambda b: a
                a = [a]
            
            if not isinstance(i,list):
                i = [i]
            print(a,i)
             
            print(type(a))
            
            for n,j in enumerate(i):
                print(n,j)
                print(a[0])
                print(type(a[n]))
                x[j] = a[0](x[j])
            return x



def data_save(self,save=[True,True],data=None,data_path=None,nfiles=1,
              headers=None,cols=1,*comments):

#                           ,self.model_props['observables']('__name__'),
#                           self.model_props['observables']('size'),
#                           lambda i: '%s runtime: %0.5f'%(
#                                           algorithm[i%n_alg],(time.clock()-tau0)/n_iter))
#  
    
    
    # Write Data to File Directory
    if data_path is None:
        data_dir = os.getcwd()
        data_file = data_dir+'/DataSet'
    else:
        data_dir = data_path[0]
        data_file = ''.join(data_path)
        
    
    
    if save[0]:
        # Open and write observables to a file in a specific directory
        if not data :
            if not(os.path.isdir(data_dir)):
                    os.mkdir(data_dir)
            return
        
        for n in range(nfiles):
            
            if save[1]:
                np.savez_compressed(
                        data_file+('_%d'%n if nfiles>1 else ''),a=data[n])
            
            else:   
                # Data Structure of Data Headers, and Number of Collumns per Header

                headers = np.atleast_1d(headers)
                cols = np.atleast_1d(cols)
                
                # Make observables headers for file
                file = open(data_file+('_%d'%n if nfiles>1 else '')+
                            '.txt','w')
                header = []
                
                # Write Observables to File
                
                if not (headers is None): 
                    for i,h in enumerate(headers):
                        for j in range(cols[i]):
                            header.append(h+'_'+str(j+1) 
                                            if cols[i] > 1
                                            else h)
                    file.write('\t'.join(header) + '\n')
                
                # Convert lists of lists of observables to array
                for data_n in [list(flatten(x)) for 
                               x in flatten(data[n])]:
                    dataline = ''
                    for d in data_n:
                            dataline += '%0.8f \t' %(float(d))
                    dataline += '\n'
                    file.write(dataline)
                if comments:
                    for c in comments:
                        file.write(str(c(n))+'\n')
                
                file.close()




def listchange(x,i,a=lambda b:b):
    # Return a list x after changing its ith element by a function a(x[i])
    # CURRENTLY only accepts function arguments for a 
    # (desire constant arguments that are converted to functions lamba x: a )
    
#    if not callable(a):
#        print('a is not a function')
#        a = lambda c: a
#    
#    print(a(1))
##    
##    f = lambda y:a    
##    print(a(0))
##    print(x[i])
##    print(f(x[i]))
    x[i] = a(x[i])
    return x


    
def fappend(x,F):
   for f in F:
       if not callable(f):
           f = lambda : f
       #print(f())
       ft = f()
       if not((isinstance(ft,np.array)) or (isinstance(ft,list))):
           x = np.append(x,ft)
       else:
           for j in ft:
               x = np.append(x,j)
   return x

def realimag(x,tol=1e-14):
    # Return entirely real or imaginary components of complex x, 
    # if the conjugate component is zero, or below a tolerance
    r = x.real
    i = x.imag
    
    
    if i == 0:
        return x
    
    if abs(r) <= tol:
        return i
    elif abs(i)<=tol:
        return r
    else:
        return x

def signed_val(x):
    xs = np.shape(x)
    xf = np.array(x).flatten()
    ri = np.stack((np.real(xf),np.imag(xf)),1)
    xms = np.sign(ri[np.arange(np.size(ri,0)), np.argmax(np.abs(ri),1)])*np.abs(xf)
    return np.reshape(xms,xs)
    
def index(x,i):
    # Return array of elements in list x, specficied in indicices in list i
    return [x[j] for j in i] if isinstance(i,list) else x[i]
    
def listindex(x,i):
    #print(x)
    if not isinstance(x,list):
        return x
    if not isinstance(i,list):
        return x[i]
    elif len(i) == 1:
        try:
            return [ y if not isinstance(y,list) 
                else (y[i[0]]) if len(y)>= i[0] 
                else y[-1] for y in x]
        except IndexError: 
            return [y for y in x]
        
    else:    
        return listindex([y if not isinstance(y,list) 
                else (y[i[0]]) if len(y)>= i[0]+1 
                else y[-1]  for y in x]
                ,i[1:])

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

    
    
    
