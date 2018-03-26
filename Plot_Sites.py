# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 19:54:38 2018

@author: Matt
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

bw_cmap = colors.ListedColormap(['black', 'white'])

from ModelFunctions import choose_f, list_f

class Plot_Sites(object):
        def __init__(self,animate=[True,True,True],
                          plot_rows=1,plot_cols=1,
                          plot_titles=None,data_process= lambda data: data,
                          plot_range=None):
           
            self.animate = np.atleast_1d(animate).tolist()
            
            if self.animate[0]:
                
                
                self.plot_cols = plot_cols
                self.plot_rows = plot_rows
                
                self.fig, self.ax = plt.subplots(plot_rows,plot_cols)
                self.ax = np.atleast_1d(self.ax)
                
                            
                if plot_titles == None:
                    self.titles = [lambda x:x,lambda x:x]
                else:
                    self.titles = plot_titles
                
                self.data = [None]*plot_cols
                
                self.data_process = data_process
                self.plot_range = np.append(plot_range,plot_range[-1]+1)
                
                if not self.animate[1] and plot_rows>1:
                    self.ax = self.ax[:,np.newaxis] 
                elif plot_rows == 1:
                    self.ax = self.ax[np.newaxis,:] 
                
            
            return
            
        
        
        
        def plot_sites(self,plot_row,plot_col,plot_iter=''):               
            ax = self.ax[plot_row,plot_col]
            data = self.data_process(self.data[plot_col])
            
            plot_label = []
            i_title = [[plot_col,plot_row,0],
                       [plot_iter,plot_row,self.plot_rows-1],
                       [plot_row,plot_col,0]]
            
            for i_t,t in enumerate(self.titles):
                plot_label.append(choose_f(t,i_title[i_t],''))
            
            if self.plot_range is None:
                self.plot_range = list(range(
                                int(np.min(data[~np.isnan(data)])),
                                int(np.max(data[~np.isnan(data)])+2)))

#            manager = plt.get_current_fig_manager()
#            manager.resize(*manager.window.maxsize())
            
            n_plot_range = len(self.plot_range)
            norm = colors.BoundaryNorm(self.plot_range, ncolors=n_plot_range)
            cmap=plt.cm.get_cmap('bone',n_plot_range)
            cmap.set_bad(color='magenta')
        
#                if ax == None:
#                    _,ax = plt.subplots()
            
            ax.clear()
            plt.sca(ax)
            
            
            
            plot = plt.imshow(np.real(data),cmap=cmap,norm=norm,
                              interpolation='nearest')
            
            plt.sca(ax)
            plt.xticks([])
            plt.yticks([])
            plt.title(plot_label[0])
            plt.xlabel(plot_label[1])
            plt.ylabel(plot_label[2])
            
            
            
            if plot_col in [0,1]:
                self.plot_show()
            
            if plot_col==self.plot_cols-1:
                cax = make_axes_locatable(
                ax).append_axes('right', size='5%', pad=0.05)
                cax.clear()
                cbar = plt.colorbar(plot,cax= cax,label='Spin value')
                cbar.set_ticks(np.array(self.plot_range)+0.5)
                cbar.set_ticklabels(np.array(self.plot_range))
                #cax.clear()
            
            return
        
        
        def plot_show(self,time0 = 0.2):
            plt.pause(time0)
        
        def plot_save(self,file):
            if self.animate[0]:
                self.fig.set_size_inches((8.5, 11), forward=False)
                self.fig.savefig(file+'.pdf',dpi=500) # bbox_inches='tight'
            return