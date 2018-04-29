# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 02:43:27 2018

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



def plot_plot(data,domain=None,fig=None,ax=None,plot_props={}):
    
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.gca()     
    
    # Plot Data
    if isinstance(data,dict):
        for k,d in data.items():
            if (d is not None) and (d != []):
                plt.plot(domain[k],plot_props['data']['data_process'](d),
                            label=plot_props['other'].get('label',k),
                            **plot_props['plot'])
            fig.sca(ax)
    else:
        if (data is not None) and (data != []):
            plt.plot(domain,plot_props['data']['data_process'](data),
                        label=plot_props['other'].get('label',''),
                        **plot_props['plot'])
        fig.sca(ax)

    # Set Figure Properties
    plt.setp(ax,**plot_props.get('set',{}));
    plt.legend()
        
    plt.pause(plot_props['other'].get('pause',0.5))

    return



def plot_histogram(data,domain=None,fig=None,ax=None,plot_props={}):
    
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.gca()
     
    # Plot Data
    plot = {}
    if isinstance(data,dict):
        for k,d in sorted(data.items()):
            if (d is not None) and (d != []):
                plot[k] = plt.hist(plot_props['data']['data_process'](d),
                     bins=int(1 +3.322*np.log10(np.size(d))),
                     label=plot_props['other'].get('label',lambda x:x)(k),
                     **plot_props['plot'])

    else:
        if (data is not None) and (data != []):
            plot[''] = plt.hist(plot_props['data']['data_process'](data),
                 bins=int(1 +3.322*np.log10(np.size(data))),
#                 label=plot_props['other'].get('label',''),
                 **plot_props['plot'])

    # Set Figure Properties
    plt.setp(ax,**plot_props.get('set',{}));
    plt.legend()
    
    #cursor_annotate(plot,leg,fig,ax)
    
    plt.pause(plot_props['other'].get('pause',0.5))
    
    return



def plot_scatter(data,domain=None,fig=None,ax=None,plot_props={}):
     
    
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.gca()     
    
    # Plot Data
    if isinstance(data,dict):
        for k,d in data.items():
            if (d is not None) and (d != []):
                plt.scatter(domain[k],plot_props['data']['data_process'](d),
                               label=plot_props['other'].get('label',k),
                               **plot_props['plot'])
            fig.sca(ax)
    else:
        if (data is not None) and (data != []):
            plt.scatter(domain,plot_props['data']['data_process'](data),
                           label=plot_props['other'].get('label',''),
                           **plot_props['plot'])
        fig.sca(ax)

    # Set Figure Properties
    plt.setp(ax,**plot_props.get('set',{}));
    plt.legend()
    plt.pause(plot_props['other'].get('pause',0.5))

    return





def plot_image(data,domain=None,fig=None,ax=None,plot_props={}):
    
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.gca()   
       
    
    
    # Setup Image Colourbar
    if plot_props['data'].get('plot_range') is None:
        plot_props['data']['plot_range'] = list(range(
                    int(np.min(data[~np.isnan(data)])),
                    int(np.max(data[~np.isnan(data)])+2)))

    n_plot_range = len(plot_props['data']['plot_range'])
        
    norm = colors.BoundaryNorm(plot_props['data']['plot_range'],
                                   ncolors=n_plot_range)
        
    cmap=plt.cm.get_cmap(plot_props['other'].get('cbar_color','bone'),
                             n_plot_range)
    cmap.set_bad(color = plot_props['other'].get('cbar_color_bad','magenta'))
    
        
    #ax.clear()
    #fig.sca(ax)
    
    
    # Plot Data
    if isinstance(data,dict):
        for k,d in data.items():
            if (d is not None) and (d != []):
                plot = plt.imshow(plot_props['data']['data_process'](d),
                              cmap=cmap, norm=norm, interpolation='nearest',
                              label=plot_props['other'].get('label',k),
                              **plot_props['plot'])
            fig.sca(ax)
    else:
        if (data is not None) and (data != []):
            plot = plt.imshow(plot_props['data']['data_process'](data),
                         cmap=cmap, norm=norm, interpolation='nearest',
                         label=plot_props['other'].get('label',''),
                         **plot_props['plot'])
        fig.sca(ax)

    # Set Figure Properties
    plt.setp(ax,**plot_props.get('set',{}));
    plt.legend()
    
    
    # Plot Colourbar
    if plot_props['other']['cbar_plot']:
        cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05)
        fig.sca(cax)
        cax.clear()
        cbar = plt.colorbar(plot,cax= cax,
                            label=plot_props['other']['cbar_title'])
        cbar.set_ticks(np.array(plot_props['data']['plot_range'])+0.5)
        cbar.set_ticklabels(np.array(plot_props['data']['plot_range']))
        #cax.clear()
    
    plt.pause(plot_props['other'].get('pause',0.5))
    
    return



#
#def cursor_annotate(plot,leg,fig,ax):
#    
#    lined = {}
#    for leg_line, plt_line in zip(leg.get_lines(),plot.values()):
#        leg_line.set_picker(5)
#        lined[leg_line] = plt_line
#    
#    def onpick(event):
#        
#        leg_line = event.artist
#        plt_line = lined[leg_line]
#        
#        vis = not plt_line.get_visile()
#        
#        plt_line.set_visible(vis)
#        
#        if vis:
#            leg_line.set_alpha(1.0)
#        else:
#            leg_line.set_alpha(0.2)
#        fig.canvas.draw()
#    
#    def line_hover(event):    
#        for line in ax.get_lines():
#            if line.contains(event)[0]:
#                print(plot.get_label())
#
#    fig.canvas.mpl_connect("pick_event", onpick)
#    plt.show()