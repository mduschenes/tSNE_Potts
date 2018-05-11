# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 02:43:27 2018

@author: Matt
"""
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator
matplotlib.rcParams['text.usetex'] = True



# Define Figure Fonts
#rcParams['axes.labelsize']  = 11
#rcParams['figure.figsize']  = fig_size
#rcParams['font.family']     = 'serif'
#rcParams['font.serif']      = ['Computer Modern']
#rcParams['font.size']       = 9
##rcParams['text.fontsize']   = 10 ##text.fontsize is deprecated
#rcParams['text.usetex']     = True
#rcParams['legend.fontsize'] = 9
#rcParams['xtick.labelsize'] = 9
#rcParams['ytick.labelsize'] = 9
#matplotlib.rcParams['mathtext.fontset'] = 'custom'
#matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
#matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

plot_image_types = ['plot_image']

def plot_decorator(plot_func):
    
	def plot_wrapper(data,domain=None,fig=None,ax=None,plot_props={}):
		
		# Define Figure and Axes
		if fig is None or ax is None:
			fig = plt.figure()
			ax = fig.gca()
		
		ax.axis('on')     
			
		plot_props0 = copy.deepcopy(plot_props)
			
		# Setup Image Colourbar
		if plot_props.get('other',{}).get('cbar_plot')is not None or (
			plot_func.__name__ in plot_image_types):
			
			if plot_props.get('data',{}).get('plot_range') is None:
				plot_props['data']['plot_range'] = list(range(
							int(np.min(data[~np.isnan(data)])),
							int(np.max(data[~np.isnan(data)])+2)))
		
			n_plot_range = len(plot_props['data']['plot_range'])
				
			norm = colors.BoundaryNorm(plot_props['data']['plot_range'],
										   ncolors=n_plot_range)
				
			cmap=plt.cm.get_cmap(plot_props.get('other',{}).get('cbar_color',
																'bone'),
								 n_plot_range)
			cmap.set_bad(
					color = plot_props.get('other',{}).get('cbar_color_bad',
														   'magenta'))
			plot_props.get('plot',{})['cmap'] = cmap
			plot_props.get('plot',{})['norm'] = norm
		
		# Plot Data
		
		if isinstance(data,dict):
			plot = {}
			for k,d in data.items():
				if (d is not None) or (d != []):
					
					plot_props.get('plot',{})['label'] = plot_props.get(
														'other',{}).get(
														'label',lambda x:x)(k)
					d = plot_props.get('data',{}).get(
									   'data_process',lambda x:x)(d)
					
					plot[k] = plot_func(domain[k],d,plot_props.get('plot',{}))
					
					
		else:
			if (data is not None) or (data != []):
				
				plot_props.get('plot',{})['label'] = plot_props.get(
													'other',{}).get(
													'label',lambda x='':x)()
				data = plot_props.get('data',{}).get(
									  'data_process',lambda x:x)(data)
				
				plot = plot_func(domain,data,plot_props.get('plot',{}))


        
		if not plot_props.get('other',{}).get('sup_legend'):
			legend = plt.legend(prop={'size': 7})

		
		
		
		# Plot Colourbar
		if plot_props.get('other',{}).get('cbar_plot') is not None:


			cax = make_axes_locatable(ax).append_axes('right',
													  size='5%',pad=0.05)
			# ax_pos = ax.get_position()
			# cbar_pos = [ax_pos.x0 + 0.175, ax_pos.y0 + 0.3,  
						# ax_pos.width/20, ax_pos.height/4] 
			# cax = fig.add_axes(plot_props.get('other',{}).get('cbar_pos',
														   # cbar_pos))
			
			plt.cla()

			fig.sca(cax)

			cbar = plt.colorbar(plot,cax= cax,
							 label=plot_props.get('other',{}).get(
												  'cbar_title',''))
			if not plot_props.get('other',{}).get('cbar_plot'):
				fig.delaxes(cax)
			else:
				try:
					cbar_vals=np.array(list(set(
											 plot_props['data']['plot_range'])))
					cbar.set_ticks(cbar_vals+0.5)
					cbar.set_ticklabels(cbar_vals)
				except RuntimeError:
					cbar_vals = np.linspace(
					                    min(plot_props['data']['plot_range']),
					                    max(plot_props['data']['plot_range']),5)
					cbar.set_ticks(cbar_vals)
					cbar.set_ticklabels(cbar_vals)



        # Set Figure Properties
		plot_not_set_props = ['other','data','plot']
		if plot_func.__name__ =='plot_histogram':
			ax.yaxis.set_major_locator(MaxNLocator(integer=True))
		for prop in np.setdiff1d(list(plot_props.keys()),plot_not_set_props):
			if '_attr' in prop:
				obj = locals()[prop.replace('_attr','')]
				for k,p in plot_props[prop].items():
					if 'get_' in k:
						plt.setp(getattr(obj,k)(),**p);
					else:
						plt.setp(getattr(obj,k),**p);
			else:
				obj = locals()[prop]
				plt.setp(obj,**plot_props[prop]);




        
		plt.pause(plot_props['other'].get('pause',0.01))
		
		# Reset plot_props
		plot_props = plot_props0
		return     
        
	return plot_wrapper


@plot_decorator
def plot_plot(x,y,props):
	 plot = plt.plot(x,y,**props)
	 return plot


@plot_decorator
def plot_histogram(x,y,props):
	props['bins'] = 20 #int(1+3.322*np.log10(np.size(y))
	plot = plt.hist(y,**props)
	return plot


@plot_decorator
def plot_scatter(x,y,props):
	plot = plt.scatter(x,y,**props)
	return plot


@plot_decorator
def plot_image(x,y,props):  
	plt.cla()
	plot = plt.imshow(y,**props)
	return plot




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