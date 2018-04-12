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
#matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

bw_cmap = colors.ListedColormap(['black', 'white'])

from ModelFunctions import choose_f,caps


class Plot_Data(object):
    def __init__(self,animate=[True,True],
                      plot_rows=1,plot_cols=1,plot_titles=None,
                      data_process= lambda data: np.real(data),
                      plot_range=None,plot_types='sites'):
       
        
        self.plot_types = np.atleast_1d(plot_types)
        self.animate = np.atleast_1d(animate)
        
        # Define Numbers of Subplots
        self.plot_cols = plot_cols
        self.plot_rows = plot_rows
        
        if self.animate[0]:
            
            
            # Create Figures and Axes
            self.figures = []
            self.figures_axes()
            
            
            # Define Titles for each Plot Type
            if plot_titles is None:
                plot_titles = [lambda x:'']*4
                self.plot_titles = dict.fromkeys(
                                        self.plot_types,plot_titles)

            elif not isinstance(plot_titles, dict):
                self.plot_titles = dict(zip(self.plot_types,
                                            np.atleast_2d(plot_titles)))
            else:
                self.plot_titles = plot_titles
        
            # Define Data Processing and Plot Range
            self.data_process = data_process
            if not(plot_range is None):
                self.plot_range = np.append(plot_range,plot_range[-1]+1)
            else:
                self.plot_range = plot_range
                
            # Define Plotting Functions based on Plot Type
            self.plot_functions = dict(map(lambda x: (x,getattr(self,'plot_'+x)),self.plot_types))

                
        return
            
    def plot_data(self,data,plot_index, new_figure, plot_type,plot_props):
        self.plot_functions[plot_type]
        
# Plot Sites
    def plot_sites(self,data,plot_index = [0,0,0,-1],
                   new_figure=False, plot_type='sites', plot_props=None):
        
        
        
        plot_row = plot_index[0]
        plot_col = plot_index[1]
        plot_iter = plot_index[2]
        num_figure = plot_index[3]
        
        if self.animate[0]:
            
            if new_figure:
                self.figures_axes()
                num_figure = -1
            
                
            ax = self.figures[num_figure][plot_type]['ax'][plot_row,plot_col]
            fig = self.figures[num_figure][plot_type]['fig']
        
        
            data = self.data_process(data)
            
            
            if self.plot_range is None:
                self.plot_range = list(range(
                            int(np.min(data[~np.isnan(data)])),
                            int(np.max(data[~np.isnan(data)])+2)))
            
            
            
            plt.figure(fig.number)
            
            # Define Labels for Title, X-axis, Y-axis-left, Y-axis-right 
            # titles object is 4 functions, which will act based on 
            plot_labels = []
            i_title = [[plot_col,plot_row,0],
                   [plot_iter,plot_row,self.plot_rows-1],
                   [plot_row,plot_col,0],[0,plot_col,self.plot_cols-1]]
        
            for i_t,t in enumerate(self.plot_titles[plot_type]):
                plot_labels.append(choose_f(t,i_title[i_t],''))
            
            # Initialize Colourbar Properties
            n_plot_range = len(self.plot_range)
            
            norm = colors.BoundaryNorm(self.plot_range,
                                       ncolors=n_plot_range)
            cmap=plt.cm.get_cmap('bone',n_plot_range)
            cmap.set_bad(color='magenta')
        
            
            ax.clear()
            fig.sca(ax)
            
            
            # Plot Sites Data     (Ensure Data is Rectangular Image)
            try:
                plot = plt.imshow(data,cmap=cmap,norm=norm,
                              interpolation='nearest')
                fig.sca(ax)
    
            except:
                print(data)
                print('Error - Non-rectangular Image Data')
                return
            
            
            plt.sca(ax)
            plt.title(caps(plot_labels[0]))
            plt.xlabel(caps(plot_labels[1]))
            plt.ylabel(caps(plot_labels[2]))
            
            # Plot Colourbar
            if plot_col in [0,1]:
                self.plot_show()
            
            if plot_col==self.plot_cols-1:
                cax = make_axes_locatable(
                ax).append_axes('right', size='5%', pad=0.05)
                fig.sca(cax)
                cax.clear()
                cbar = plt.colorbar(plot,cax= cax,
                                    label=caps(plot_labels[3]))
                cbar.set_ticks(np.array(self.plot_range)+0.5)
                cbar.set_ticklabels(np.array(self.plot_range))
                #cax.clear()
        return 
    
    
    
    
    # Plot Observables
    def plot_observables(self,data,plot_index = [0,0,0],
                 plot_type='observables', new_figure=False,num_figure=-1,plot_props=None):
        
        
        plot_row = plot_index[0]
        plot_col = plot_index[1]
        #plot_iter = plot_index[2]
        
        
        
        if self.animate[0]:
            
            if new_figure:
                self.figures_axes()
                num_figure = -1
                
            ax = self.figures[num_figure][plot_type]['ax'][plot_row,plot_col]
            fig = self.figures[num_figure][plot_type]['fig']
        
        
            data = self.data_process(data)
        
        
        
            # Iterate over Sets of Data
            for i_data,data_set in enumerate(np.atleast_1d(data)):
                
                
                # Number of Observables
                n_data = np.shape(data_set)[1]
                
                # Reshape dataT into 2d array of [x, obs_i]
                data_set = np.reshape(data_set,(-1,n_data)).transpose()                 
                
                # Create New Figures if currently less than n_dataT
                if np.size(self.figures) != n_data-1:
                    #print('%d New Figures Added'% (n_data-np.size(self.figures)-1))
                    self.figures_axes(n_data-np.size(self.figures)-1)
                
                
                # Iterate over Observables
                for i,dat in enumerate(data_set[1:]):
                
                    ax = self.figures[i][plot_type]['ax'][plot_row,plot_col]
                    fig = self.figures[i][plot_type]['fig']
    
                    plt.figure(fig.number)
    
            
                    fig.sca(ax)
            
                    plt.hist(dat,bins=int(1 +3.322*np.log10(np.size(dat))),
                             **plot_props(i_data))
    #                        dat,bins = np.histogram(dat,bins=int(np.size(dat)/1.5))
    #                        plt.bar(bins[0:-1],dat,alpha = 0.5,**plot_props(i_data))
    #                        
    
                    plt.title(self.plot_titles[plot_type][0](''))
                    plt.xlabel(caps(self.plot_titles[plot_type][1]('')[i]))
                    plt.ylabel(caps(self.plot_titles[plot_type][2]('')))
                    plt.legend()
        
                return
    
        
        
    def figures_axes(self,make_fig=1,plot_type):
        # Define Figures and Axis for each Plot Type
        
        for i in range(make_fig):
            figures = {}
           
            for t in self.plot_types:
                fig, ax = plt.subplots(self.plot_rows,self.plot_cols)
                fig.canvas.set_window_title(t) 
                ax = np.atleast_1d(ax)
                
                if self.plot_rows>1:
                    if not self.animate[1]:
                        ax = ax[:,np.newaxis] 
                else:
                    ax = ax[np.newaxis,:] 
                
                figures[t] = {'fig':fig,'ax':ax}

            self.figures[plot_type].append(figures)

        return
    
    
    def plot_show(self,time0 = 0.2,close=False):
        # Pause and Show plot for time0
        plt.pause(time0)
        if close:
            self.plot_close()
        return
    
    def plot_save(self,file,plot_type='observables',plot_num = -1):
        # Save Plot Figure to File
        
        if self.animate[0]:
                
            for i in range(np.size(self.figures)):
                
                fig = self.figures[i][plot_type]['fig']
#                for i in plt.get_fignums():
#                    plt.figure(i)
                plot_size = fig.get_size_inches()
                fig.set_size_inches((8.5, 11))
                
                if plot_type == 'observables':
                    label = '' #self.plot_titles[plot_type][1]('')[i]
                    plot_size = tuple(x+0.5 for x in plot_size)
                else:
                    label = ''
                
                fig.savefig(''.join(file[i])+label+'.pdf',dpi=500)
                fig.set_size_inches(plot_size)       
        return
    
    def plot_close(self):
        plt.close('all')

                
                
                    
               #self.figures[plot_type]['fig'].savefig(file+'.pdf',dpi=500) # bbox_inches='tight'
            