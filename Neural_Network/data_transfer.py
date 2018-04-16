# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:35:04 2018

@author: Matt
"""
import numpy as np





data_params = {'data_files': ['x_train','y_train','x_test','y_test'],
              'data_types': None,
              'data_format': 'npz',
              'data_dir': 'dataset/',
              'one_hot': False}







class data_transfer(object):
    
    
        def __init__(self):
            pass 
        
        # Import Data
        def importer(self,data_params = {
                        'data_files': None,
                        'data_types': ['x_train','y_train','x_test','y_test'],
                        'data_format': 'npz',
                        'data_dir': 'dataset/',
                        'one_hot': False}):
            
            # Data Dictionary
            data_params = self.dict_check(data_params,
                                         ['data_types','data_files'],1)
            
            
            # Import Data
            if data_params['data_format'] == 'values':
                data = data_params['data_files']
            
            elif data_params['data_format'] == 'npz':
                data = {k: np.load(data_params['data_dir']+v+'.'+
                                   data_params['data_format'])['a'] 
                                  for k,v in data_params['data_files'].items()}
            
            elif data_params['data_format'] == 'txt':
                data = {k: np.loadtxt(data_params['data_dir']+v+'.'+
                                     data_params['data_format']) 
                                  for k,v in data_params['data_files'].items()}
            
            
            # Convert Labels to one-hot Labels
            if data_params['one_hot']:
                for k in data.keys(): 
                    if ('y_' in k or 'label_' in k):
                        data[k] = self.one_hot(data[k])
            
            
            # Size of Data Sets
            data_sizes = {k:np.shape(np.atleast_2d(v)) for k,v in data.items()}
            
            
            
            return data, data_sizes
        
        
        
        # Export Data
        def exporter(self,data,data_params={'data_types': [''],'data_dir':''}):

            # Data Dictionary
            data_params['data_files'] = data
            data_params = self.dict_check(data_params,
                                         ['data_types','data_files'],None)

            for k,v in data.items():
                np.savez_compressed(data_params['data_dir']+k,a=v) 
            return
            
              
        
        
        # Check if variable is dictionary
        def dict_check(self,data_params,
                      keys = ['data_types','data_files'],key_i=None):
            
            # Check if keys[key_i] exists, otherwise replace with other key
            if not isinstance(data_params[keys[key_i]], dict):
                if key_i and data_params[keys[key_i]] is None:
                    data_params[keys[key_i]] = data_params[keys[1-key_i]]
                                        
                
                data_params[keys[1]] = dict(zip(data_params[keys[0]],
                                                    data_params[keys[1]]))
                
            return data_params
        
        
        
        
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
    d = data_transfer