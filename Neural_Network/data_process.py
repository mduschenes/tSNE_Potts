# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:35:04 2018

@author: Matt
"""
import numpy as np
import matplotlib.pyplot as plt





data_params = {'data_files': ['x_train','y_train','x_test','y_test'],
              'data_types': None,
              'data_format': 'npz',
              'data_dir': 'dataset/',
              'one_hot': False}







class Data_Process(object):
    
    
        def __init__(self):
            self.fig = {}
            self.ax = {}
            pass 
        
        def process(self,data,domain,data_params,
                    save=False,plot=False,**kwargs):
            
            if save:
                self.exporter(data,data_params)
            
            if plot:
                self.plotter(data,domain,**kwargs)    
            
            return
        
        
        # Import Data
        def importer(self,data_params = {
                        'data_files': None,
                        'data_types': ['x_train','y_train','x_test','y_test'],
                        'data_format': 'npz',
                        'data_dir': 'dataset/',
                        'one_hot': False}):
            
            # Data Dictionary
            data_params = self.dict_check(data_params,'data_files')            
            
            # Import Data
            if data_params.get('data_format','values') == 'values':
                data = {k:v for k,v in 
                        self.dict_check(data_params['data_files'],
                                        data_params['data_types']).items() 
                        if v is not None}
            
            elif data_params.get('data_format',None) == 'npz':
                data = {k: np.load(data_params['data_dir']+v+'.'+
                                   data_params['data_format'])['a'] 
                                  for k,v in data_params['data_files'].items()
                                  if v is not None}
            
            elif data_params.get('data_format',None) == 'txt':
                data = {k: np.loadtxt(data_params['data_dir']+v+'.'+
                                     data_params['data_format']) 
                                  for k,v in data_params['data_files'].items()
                                  if v is not None}
            
            
            # Convert Labels to one-hot Labels
            if data_params.get('one_hot',False):
                for k in data.keys(): 
                    if ('y_' in k or 'label_' in k):
                        data[k] = self.one_hot(data[k])
            
            
            # Size of Data Sets
            data_sizes = {k:np.shape(np.atleast_2d(v)) for k,v in data.items()}
            
            
            
            return data, data_sizes
        
        
        
        # Export Data
        def exporter(self,data,data_params={'data_dir':'dataset/'},
                     file_names = None):
           
            # File Names
            if file_names is None:
                file_names = lambda value: '%depochs_%s'%(np.size(data),value)
            elif not callable(file_names):
                g = np.atleast_1d(file_names)
                file_names = lambda k: g[k]
            
            data = self.dict_check(data,'')

            for k,v in data.items():
                np.savez_compressed(data_params.get('data_dir','dataset/') + 
                                                    file_names(k),a=v) 
            return
       
        
        def plotter(self,data,domain,**kwargs):
        
            # Check Data is Dictionary
            data = self.dict_check(data,'data_files')


            if kwargs.get('plot_f'):   
                keys = [k for k in data.keys() if data.get(k)]+ ['key_plot_f']
            else:
                keys = list([k for k in data.keys() if data.get(k)])
            
            self.figure_axes(keys)
            
            
            for key,val in data.items():
            
                
                if not val:
                    continue
                
                try:
                    plt.figure(self.fig[key].number)
                except:
                    self.figure_axes(list([k for k in data.keys() 
                                   if data.get(k)])+ [kwargs.get('','')])
                    plt.figure(self.fig[key].number)
                
                self.fig[key].sca(self.ax[key])

                plt.plot(domain[key],val,'-*',color='r')
                plt.title('')
                plt.ylabel(key)
                plt.xlabel('Epoch')
                
                plt.pause(0.01)
                

            if kwargs.get('plot_f'):
                kwargs['plot_f'](fig = self.fig['key_plot_f'],
                                 ax = self.ax['key_plot_f'])
                
    
        
        def plot_close(self):
            plt.close('all')   
            self.ax = {}
            self.fig ={}
            return
                    
        def figure_axes(self,keys):
            
            keys = [k for k in np.atleast_1d(keys) if k not in self.ax.keys()]
            
            if keys:
           
                fig, ax = plt.subplots(np.size(keys))
            
                for k,a in zip(keys,ax):
                    self.ax[k] = a
                    self.fig[k] = fig
            
            return
        
        # Check if variable is dictionary
        def dict_check(self,dictionary,key):
                        
            # Check if dict is a dictionary
            if not isinstance(dictionary,dict):
                return dict(zip(key,dictionary))
            else:
                return dictionary
            
        
        
        
        
        # Converts data X to n+1-length one-hot form        
        def one_hot(self,X,n=None):
           
            n = int(np.amax(X))+1 if n is None else int(n)+1
            
            sx = np.shape(np.atleast_1d(X))
            
       
            y = np.zeros(sx+(n,),dtype=np.int32)
            
            for i in range(n):
                p = np.zeros(n)
                np.put(p,i,1)
                y[X==i,:] = p
            
            
            
            
            return np.reshape(y,(-1,n))
        
        
        # Convert Data to Range
        def convert_to_range(self,X,sort='max',N=None):
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

            
        
        
        
        
if __name__ == "__main__":
    d = Data_Process()