# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 22:57:13 2017

@author: Matt
"""
import numpy as np
import sympy as sp
import scipy
from inspect import signature
import matplotlib.pyplot as plt

def finfo(f):
    return signature(f).parameters

def ide(f):
    return lambda x:x    
    
def flatten(x):
    xlist = []
    for y in x:
        if type(y)==type([]):
            xlist.extend(flatten(y))
        else: xlist.append(y)
    return xlist

def recfunc(a,f=ide,F=ide):
    return F((f(x) for x in flatten(a)))

def listcheck(a):
    return [a] if type(a) != list else a

def nonecheck(a,b):
    return b if a == None else a

def typecheck(a,typec=list,b=None):
    
    if typec != None:
        if (any(b) == None and type(a) != None): b = typec
        elif (any(b) == None):b = lambda x: typec
        else: b = lambda x: b
    else: b = lambda x: None
    
    print(b(sp.symbols('w')))
    return a if type(a) == typec else b(a)
   
def comp(f, N=None):
    
    def compn(f,n):
        return f if n == 1 else lambda x: f(compn(f, n-1)(x))

    def compfg(f,g):
        return lambda x: f(g(x))
    
    f = listcheck(f)
    if N==None:
        N = np.ones(len(f))
    y = lambda x: x
    for g,n in zip(f,N):
        y =  compfg(y,compn(g,n))
    #print(y(sp.symbols('w')))
    return y
    
def dispvars(*args):
    vargs = locals()
    print(vargs)
    [print(t,vargs[t]) for t in vargs]
    #print(len(locals())-1)
    return

def sign_bit(x):
    y = flatten(x)
    return sum((int((-np.sign(y[i])+1)/2) if y[i] != 0 else 1)*(2**(len(y)-i-1)) for i in range(len(y)))
        
        
    
# Define Monte Carlo Integration Sampling
def norm(x,p=2):
    return np.power(sum(x**p),1/p) 
    
def normvarargs(*args,p=2):
    return recfunc(args,lambda x: np.power(x,2),lambda x: np.sqrt(sum(x)))
    
def Vdsphere(d,R=1):
    return (R**d)*(np.pi**(d/2)/ scipy.special.gamma(d/2 + 1))

def VSphereMC(d = 7,R = 1,Nemax = 6,trials = 10):
    Vd = Vdsphere(d,R)
    print(Vd)
    VdN =[]
    for Ne in range(Nemax):
        VdNavg = np.sum(MCarea(norm,10**Ne,d,R) for t in range(trials))/trials
        print(VdNavg)
        VdN.append((Vd - VdNavg)/Vd)
    
    plt.plot(range(Nemax),VdN)

def MCarea(f,N,d,R=1,g=None,xrange=None):
    xrange = nonecheck(xrange, [-R,R])
    g = nonecheck(g,lambda x: R)
    x = xrange[0] + (xrange[1]-xrange[0])*np.random.random_sample((N,d))
    s = np.sum(f(x[i,])<g(x[i,]) for i in range(N))
    return ((2*R)**d)*s/N
    
def IntMC(f,xrange=[0,1],d = 1,R = 1,Ne = 4,trials = 1):
    return np.sum(MCarea(f,10**Ne,d,R,g = lambda x:X, xrange = xrange) for t in range(trials))/trials

def main():
    x = [[1,2],3,[4,5]]
    print(flatten(x))
    print(normvarargs(x))
    
    def fsin(y):
        return np.power(np.sin(1/y),2)
    
    fsinint = []
    X = np.arange(0,1,100)
    fsinint.extend(IntMC(x,fsin) for x in X)
    
    plt.plot(X,fsinint)
    
    
    
###### MAIN #######
#if __name__ == "__main__":
#    main()

#g(1,2,3,4)
#fval = f(*x[i,]) if (((len(f_info(f))>1) and not('args' in f_info(f)))) else f(x[i,])
        #print(np.random.random_sample((N,Nxf)))
        # print [ftype['American'] for f,ftype in myDict.iteritems() if f == 'Apple' and 'American' in ftype]