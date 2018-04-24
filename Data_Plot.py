# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 19:54:38 2018
@author: Matt
"""

import numpy as np
import matplotlib.pyplot as plt
import os.path


from ModelFunctions import flatten,dict_check, one_hot
from plot_functions import *



class Data_Plot(object):
    
    # Create figure and axes dictionaries for dataset keys
    def __init__(self, keys=[None],plot=False):
    
    
        # Initialize figures and axes dictionaries
        self.figs = {}
        self.axes = {}
        
        self.plot = plot
        
        if self.plot:
            # Create Figures and Axes with keys
            self.figures_axes(keys)
    
            # Define Possible Plotting Functions
            plot_func = globals()
            self.plot_func = {k: v for k,v in plot_func.items() if 'plot_' in k}
        return  
    


     # Plot Data by keyword
    def plotter(self,data,domain=None,plot_props={},keys=None):

        if self.plot:
            if keys is None:
                keys = data.keys()
            
            keys = [k for k in keys if data.get(k) is not None]
            
            if domain is None:
                domain = {k: None for k in keys}
            
            self.figures_axes(keys)
            
            
            
            # Plot for each data key
            for key in keys:
                                
                try:
                    ax = self.axes[key]
                    fig = self.figs[key]
                    plt.figure(fig.number)
                    fig.sca(ax)
                except:
                    self.figures_axes(keys)
                    
                    ax = self.axes[key]
                    fig = self.figs[key]
                    plt.figure(fig.number)
                    fig.sca(ax)
    
                # Plot Data
                try:
                    self.plot_func['plot_'+plot_props[key]['data']['plot_type']](
                            data[key],domain[key],fig,ax,plot_props[key])
                except AttributeError:
                    self.plot_func = plot_props[key]['data']['plot_type'](
                                data[key],domain[key],fig,ax,plot_props[key])
                
                plt.suptitle(plot_props[key].get('plot_title',''))
                
    #            ,size = 9,horizontalalignment='left',x=0.91)

            
        return
            



    # Clost all current figures and reset figures and axes dictionaries
    def plot_close(self):
        plt.close('all')   
        self.axes = {}
        self.figs ={}
        return
    
    
    # Save all current figures
    def plot_save(self,file_dir='',label='',file_format='.pdf'):
        
        for ifig in plt.get_fignums():
            
            # Find Attributes Plotted
            keys = [k for k,v in self.figs.items() if v.number == ifig]
            
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
    
      
    
    
     # Create figures and axes for each passed set of keys for datasets
    def figures_axes(self,Keys,multi_keys=False):     
        
        
        if not multi_keys:
            Keys = [Keys]
        
        for keys in Keys:
            keys_new = [k if k not in self.axes.keys() else None
                       for k in flatten(keys)]
            
            if not None in keys_new:
                
                if len(self.axes) > 1: print('Figure Keys Updated...')
            
                fig, ax = plt.subplots(*(np.shape(keys)[:2]))
                
                fig.canvas.set_window_title('   '.join(
                                              [str(k) for k in keys_new]))
                
                for k,a in zip(keys_new,flatten(ax.tolist())):
                    if k is not None:
                        self.axes[k] = a
                        self.figs[k] = fig      
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
        data_params = dict_check(data_params,'data_files')            
        
        data_params['data_files'] = dict_check(
                    data_params['data_files'],
                    data_params.get('data_sets',
                                    data_params['data_files']))
        
        
        # Import Data
        import_func = {}
        
        import_func['values'] = lambda v:v
        
        import_func['npz'] = lambda v: np.load(data_params['data_dir']+
                                 v + '.' + data_params['data_format'])['a'] 
        
        import_func['txt'] = lambda v: np.loadtxt(data_params['data_dir']+
                                      v + '.' + data_params['data_format']) 
        
        data = {k: import_func[data_params.get('data_format','values')](v)
                    for k,v in data_params['data_files'].items() 
                    if v is not None}
        
        # Convert Labels to one-hot Labels
        if data_params.get('one_hot'):
            for k in data.keys(): 
                if 'y_' in k:
                    data[k] = one_hot(data[k])
        
        
        # Size of Data Sets
        data_sizes = {}
        for k in data.keys():
            data[k] = np.atleast_2d(data[k])
            v_shape = np.shape(data[k])
            if v_shape[0] == 1 and v_shape[1:] != 1:
                data[k] = np.transpose(data[k],0)
            data_sizes[k] = np.shape(data[k])
                
        
        
        # If not npz format, Export as npz for easier subsequent importing
        if data_params['data_format'] != 'npz':
            
            data_params['data_format'] = 'npz'
            self.exporter(data,data_params)
                
        return data, data_sizes
    
    
    
    
    # Export Data
    def exporter(self,data,data_params={'data_dir':'dataset/'},
                 file_names = None):
       
        # File Names
        if file_names is None:
            file_names = lambda value: value
        elif not callable(file_names):
            g = np.atleast_1d(file_names)
            file_names = lambda k: g[k]
        
        data = dict_check(data,'')
    
        for k,v in data.items():
            np.savez_compressed(data_params.get('data_dir','dataset/') + 
                                                file_names(k),a=v) 
        return     
            
        
        

                
        
        


    
    

            