# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:54:40 2018
@author: Matt
"""

# Functions for Magnetism Model


import numpy as np
import os
import types




##### Model Functions ########


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




















##### Other Functions #######


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

    
    
    
