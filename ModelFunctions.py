# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 15:54:40 2018

@author: Matt
"""

# Functions for Magnetism Model


import numpy as np


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

def get_attr(f,attr,exception,f0,*args):
    try:
        if attr == exception:
            return getattr(f(*args),attr)
        else:
            return getattr(f,attr)
        
    except AttributeError:
        return f0 

def delta_f(x,y,f=np.multiply):
    return f(x,y)[x==y]

def choose_f(f,i=[0,0,0],f0=None,):
    if not callable(f):
        g = f
        f = lambda k: g[k]
    return f(i[0]) if i[1] in np.atleast_1d(i[2]) else f0

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
        
def caps(word):
    return word[0].upper()+word[1:].lower()
