#!/usr/bin/env python
# coding: utf-8


# Restricted Boltzmann Machine (RBM)
import numpy as np
import os,sys,copy
from data_process import importer,exporter


class RBM(object):

    def __init__(self,N={},q={}):

        self.variables = ['visible','hidden']
        if isinstance(N,list):
            self.N = dict(zip(self.variables,N))
        elif isinstance(N,dict):
            self.N = N
            self.variables = list(N.keys())

        if isinstance(q,list):
            self.q = dict(zip(self.variables,q))
        elif isinstance(q,dict):
            self.q = q

        self.Nvars = len(self.variables)

        self.neurons = {v: np.random.choice(self.q[v],self.N[v])
                        for v in self.variables}

        self.parameters = {'weights': np.random.randn(*[self.N[v] for v in self.variables]),
                           'biases': {v: np.random.randn(self.N[v]) for v in self.variables}
                          }

        self.values = {v: np.arange(self.q[v]) for v in self.variables}

        return

    def energy(self): 
        energy = -self.parameters['weights']
        for v in self.variables:
            energy = np.tensordot(self.neurons[v],energy,axes=([0],[0]))
        for v in self.variables:
            energy += -np.dot(self.parameters['biases'][v],self.neurons[v])
        return energy

    def gradient(self,parameter='weights',variable='visible'):
        if parameter == 'weights':
            return np.dot(self.neurons['visible'][:,None],
                          self.neurons['hidden'][None,:])
        elif parameter == 'biases':
            return self.neurons[variable]

    def conditional(self,i,value,variable='hidden'):
        energy = -self.parameters['weights']
        for j,v in enumerate(self.variables):
            if v != variable:
                energy = np.dot(np.swapaxes(energy,j,-1),self.neurons[v])
        energy = (np.swapaxes(energy,self.variables.index(variable),-1) - self.parameters['biases'][variable])[i]
        return np.exp(-energy*value)/np.sum(np.exp(-energy*self.values[variable]),axis=-1)


    def gibbs(self,data,variable,N):
        self.neurons[variable] = data
        variables = [*[v for v in self.variables if v!=variable],variable]
        values = {v: None for v in variables}
        for i in range(N):
            for v in variables:
                values[v] = self.sample(v)
            for v in variables:
                self.neurons[v] = values[v]
        return

    def sample(self,variable):
        return np.array([np.random.choice(self.values[variable],
                                 size=1,
                                 p=self.conditional(i,self.values[variable],variable))
                         for i in range(self.N[variable])]).reshape((-1))

    def train(self,parameter,variable,data,gradient=None,**hyperparams):

        def descent(m,v,i,parameter,gradient,alpha,**kwargs):
            print(i,self.parameters['params'])
            parameter -= alpha*gradient
            print(self.parameters['params'])
            return [m,v]

        def adam(m,v,i,parameter,gradient,alpha,beta1,beta2,epsilon,**kwargs):
            m = beta1 * m + (1 - beta1)*gradient
            v = beta2 * v + (1 - beta2)*np.power(gradient,2)
            m_hat = m/(1 - np.power(beta1, i))
            v_hat = v/(1-np.power(beta2, i))
            parameter -= alpha*m_hat/(np.sqrt(v_hat)+epsilon)
            return [m,v]

        def adamax(m,v,i,parameter,gradient,alpha,beta1,beta2,epsilon,**kwargs):
            m = beta1 * m + (1 - beta1)*gradient
            m_hat = m/(1 - np.power(beta1, i))
            v = np.maximum(beta2*v,np.abs(gradient))
            parameter -= alpha*m_hat/(v+epsilon)
            return [m,v]

        def nadam(m,v,i,parameter,gradient,alpha,beta1,beta2,epsilon,**kwargs):
            m = beta1 * m + (1 - beta1)*gradient
            v = beta2 * v + (1 - beta2)*np.power(gradient,2)
            m_hat = ((m/(1 - np.power(beta1, i))) + 
                    (1-beta1)*gradient/(1 - np.power(beta1, i)))
            v_hat = v/(1-np.power(beta2, i))
            print(i,self.parameters['params'])
            parameter -= alpha*m_hat/(np.sqrt(v_hat)+epsilon)
            print(self.parameters['params'])
            return [m,v]

        if gradient is None:
            def gradient(data,parameter,variable,gibbs):
                gradient  = 0
                for i,d in enumerate(data):
                    gradient -= self.gradient(parameter,variable)
                    self.gibbs(d,gibbs,variable)
                    gradient += self.gradient(parameter,variable)
                return gradient/(i+1)
            cost = lambda data,*args: gradient(data,parameter,variable,hyperparams['gibbs'])
        else:
            cost = lambda data: gradient(data,self.parameters['params'])


        default = {'alpha':0.1,'epsilon':1e-10,
                   'beta1':0.9,'beta2':0.999,
                   'epochs':100,'batch':0.2,
                   'gibbs':1,
                   'optimizer':'adam'
                  }
        for p in default:
            if hyperparams.get(p) is None:
                hyperparams[p] = default[p]

        optimizer = locals()[hyperparams['optimizer']]

        n = np.shape(data)[0]
        b = int(hyperparams['batch']*n)
        epoch_range = np.arange(1,hyperparams['epochs']+1)
        data_range = np.arange(n)
        batch_range = np.arange(0,n,b)
        args = [0,0]
        print('Initi ',self.parameters[parameter])
        for epoch in epoch_range:
            np.random.shuffle(data_range)
            for i in batch_range:
                batch = (data[data_range])[i:i+b]
                args = optimizer(*args,epoch,self.parameters[parameter],
                                    cost(batch),
                                  **hyperparams)




f = lambda x,a,b: a*(x**2) + b
gab = lambda x,a,b: np.array([x**2,np.ones(len(x))])
N = {'hidden':5,'visible':4}
q = {'hidden':3,'visible':2}

r = RBM()



Ndata = 1000
datax = np.random.rand(Ndata)
datay = f(datax,1,2)
cost = lambda x: np.mean((f(x,*r.params)-f(x,1,1))**2)


r.parameters['params'] = [4,5]
grad = lambda x,parameter: np.mean(np.abs(f(x,*parameter)-f(x,1,2))*2*[x**2,np.ones(np.shape(x)[0])],axis=-1)


r.train('params','visible',datax,grad,alpha=1e-3,batch=1,optimizer='descent',epochs=3)




class




for v in rbm.variables:
    rbm.neurons[v] = np.ones(rbm.N[v])
    rbm.biases[v] = np.zeros(rbm.N[v])
rbm.weights = np.eye(*tuple(rbm.N[v] for v in rbm.variables))
print(rbm.energy())
for i in range(4):
    print(rbm.sample('hidden'))












