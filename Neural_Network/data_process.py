# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:35:04 2018

@author: Matt
"""
import numpy as np
import matplotlib.pyplot as plt
import os.path



class Data_Process(object):
    
    
        def __init__(self):
            self.fig = {}
            self.ax = {}
            pass 
        
        def process(self,data,domain,data_params, keys=None,
                    save=False,plot=False,**kwargs):
            
            if save:
                self.exporter(data,data_params)
            
            if plot:
                self.plotter(data,domain,keys,**kwargs)    
            
            return
        
        
        # Import Data
        def importer(self,data_params = 
                        {'data_files': ['x_train','y_train','x_test','y_test'],
                         'data_sets': ['x_train','y_train','x_test','y_test'],
                         'data_types': ['train','test','other'],
                         'data_format': 'npz',
                         'data_dir': 'dataset/',
                         'one_hot': False
                        }):
        
            # Data Dictionary
            data_params = self.dict_check(data_params,'data_files')            
            
            if not data_params.get('data_sets'):
                data_params['data_sets'] = np.arange(
                        np.size(data_params)).tolist
            
            data_params['data_files'] = self.dict_check(
                        data_params['data_files'],data_params['data_sets'])
            
            
            # Import Data
            if data_params.get('data_format','values') == 'values':
                data = {k:v for k,v in data_params['data_files'].items() 
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
                    if 'y_' in k:
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
       
        
        def plotter(self,data,domain,keys=None,**kwargs):
        
            # Check Data is Dictionary
            data = self.dict_check(data,'data_files')

            if keys is None:
                keys = data.keys()
            
            # Create new Figures/Axes if not existing based on data keys
            if kwargs.get('plot_f'):   
                keys = [k for k in keys if data.get(k)]+ ['plot_f']
            else:
                keys = [k for k in keys if data.get(k)]
            
            self.figure_axes(keys)
            
            
            
            # Plot for each data key
            for i,key in enumerate(keys):
                
                try:
                    plt.figure(self.fig[key].number)
                except:
                    self.figure_axes(keys)
                    plt.figure(self.fig[key].number)
                
                
                self.fig[key].sca(self.ax[key])
                self.ax[key].clear()

                # Plot Special Plot (possibly)
                if key == 'plot_f':
                    
                    kwargs[key](fig = self.fig[key],
                                 ax = self.ax[key])
                else:
                    plt.plot(domain[key],data.get(key),'-o',color='r')
                
                    plt.title('')
                    
                    plt.ylabel(key)
                    plt.xlabel('Epoch')
                    
                    plt.pause(0.01)
                
                self.fig[key].suptitle(kwargs.get('plot_title',''),size = 9,
                                       horizontalalignment='left',x=0.91)
                
            return
                
    
        
        def plot_close(self):
            plt.close('all')   
            self.ax = {}
            self.fig ={}
            return
        
        def plot_save(self,file_dir='',label='',file_format='.pdf'):
            
            for ifig in plt.get_fignums():
                
                # Find Attributes Plotted
                keys = [k for k,v in self.fig.items() if v.number == ifig]
                
                # Set Current Figure
                plt.figure(ifig)                
                fig = plt.gcf()
                
                # Change Plot Size for Saving                
                plot_size = fig.get_size_inches()
                fig.set_size_inches((8.5, 11))

                # Set File Name and ensure no Overwriting
                file = ''.join([file_dir,label,'_'.join(keys)])
                
                i = 0
                file_end = ''
                while os.path.isfile(file+file_end+file_format):
                    file_end = '_%d'%i
                    i+=1

                # Save Figure as File_Format
                plt.savefig(file+file_end+file_format,
                            bbox_inches='tight',dpi=500)
                fig.set_size_inches(plot_size) 
            
            return
        
        
        
        def figure_axes(self,keys):
            
            keys = [k for k in np.atleast_1d(keys) if k not in self.ax.keys()]
            
            if keys:
           
                fig, ax = plt.subplots(np.size(keys),sharex=True)
                
                #fig.canvas.set_window_title('   '.join(keys))
                
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

        def dict_sort(self,dict1,dict2,dtype='dict',reorganize=True):  
            
            # Create dict0 as sorted version of dict2 using dict1
            
            # Sort dict2 into {key_1: {key_2: { val_1: val_2_array(1_sort)} } }
            dict0 = {k1: {k2: self.array_sort(v2,v1,0,dtype) 
                              for k2,v2 in dict2.items()}
                              for k1,v1 in dict1.items()}
            
            # Reorganize to  {key_1: {key_2: { val_1: val_2_array(1_sort)} } }
            if reorganize and dtype == 'dict':
            
                dict0 = {k1: {v1i: {k2: dict0[k1][k2][v1i] 
                                    for k2 in dict2.keys()}
                                    for v1i in sorted(np.reshape(v1,(-1,)))}                                    
                                    for k1,v1 in dict1.items()}
                        
            return dict0

            
        
        def array_sort(self,a,b,axis=0,dtype='list'):
            # Sort 2-dimensional a by elements in 1-d array b
            
            b = np.reshape(b,(-1,))
            
            if dtype == 'dict':
                return {i: np.reshape(np.take(a,np.where(b==i),axis),
                                      (-1,)+np.shape(a)[1:]) 
                                        for i in set(b)}
            elif dtype == 'list':
                return [np.reshape(np.take(a,np.where(b==i),axis),
                                      (-1,)+np.shape(a)[1:]) for i in set(b)]
            elif dtype == 'sorted':
                return np.concatenate(
                                    [np.reshape(np.take(a,np.where(b==i),axis),
                                               (-1,)+np.shape(a)[1:])
                                    for i in set(b)],1)
            else:
                return a
            
        def array_mean(self,a,axis):
            return np.mean(a,axis)
                
        
        
if __name__ == "__main__":
    d = Data_Process()