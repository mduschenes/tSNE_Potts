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
from matplotlib.colors import LinearSegmentedColormap
#from colour import Color
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.ticker import MaxNLocator

# Define Figure Font Properties
FONT_SIZE = 22
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['axes.labelsize']  = FONT_SIZE
#rcParams['figure.figsize']  = fig_size
matplotlib.rcParams['font.family']     = 'serif'
matplotlib.rcParams['font.serif']      = ['Computer Modern']
matplotlib.rcParams['font.size']   = FONT_SIZE
#rcParams['legend.fontsize'] = 9
#rcParams['xtick.labelsize'] = 9
#rcParams['ytick.labelsize'] = 9
#matplotlib.rcParams['mathtext.fontset'] = 'custom'
#matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
#matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


plot_image_types = ['image','scatter']
list_sort = lambda a,j: list(zip(*sorted(list(zip(*a)),key=lambda i:i[j])))

def plot_decorator(plot_func):
    
	def plot_wrapper(data,domain=None,fig=None,ax=None,plot_props={}):
		
		# Define Figure and Axes
		if fig is None or ax is None:
			fig = plt.figure()
			ax = fig.gca()
		ax.axis('on')     
			
		plot_props = copy.deepcopy(plot_props)
					
		# Plot Data and Get Plot Properties
		plot = {}
		
		if not isinstance(data,dict):
			data = {plot_props.get('other',{}).get('plot_key',''): data}
		
		if not isinstance(domain,dict):
			domain = {plot_props.get('other',{}).get('plot_key',''): domain}

		for k,d in data.items():
			if (d is not None) or (d != []):
				y,x,props = get_props(d,domain[k],k,plot_props)

				plot[k] = plotter(x,y,plot_func,fig,ax,props)		
		
			# Set Plot Properties
			set_props(plot[k],fig,ax,plot_props)
        
		return     
        
	return plot_wrapper

	
def plotter(x,y,plot_func,fig,ax,props):
	
	props0 = props.copy()
	plot_ind = props0.pop('plot_ind',None)
	
	if not plot_ind:
		plot_obj = plot_func(x,y,props0)
	else:
		
		if not isinstance(plot_ind,list):
			plot_ind = props.keys()
		props0_ind = props0.copy()
		
		for p in [p for p in plot_ind if p not in ['c']]:
			props0_ind.pop(p,None);
		props0_ind['zorder'] = -np.inf
		plot_obj = plot_func(x,y,props0_ind)
		
		for i,(xi,yi) in enumerate(zip(x,y)):
			for p in plot_ind:
				v = props0[p]
				if callable(v):
					props0[p] = v(props0,props)
				elif isinstance(v,(list,np.ndarray)):
					props0[p] = v[i]
			plot_func(xi,yi,props0)
			
	return plot_obj

@plot_decorator
def plot_plot(x,y,props):
	plot = plt.plot(x,y,**props)
	return plot


@plot_decorator
def plot_histogram(x,y,props):
	if np.size(y) <= 1:
		return plt.plot([],label=props['label'])
	props['bins'] = min(10,int(1+3.322*np.log10(np.size(y))))
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


def set_props(plot,fig,ax,plot_props):
	
	# Plot Legend
	if (not plot_props.get('other',{}).get('sup_legend')) and  (
		plot_props.get('other',{}).get('plot_legend',False)) and (
		len(ax.get_legend_handles_labels()) > 1):
		plt.legend(*(list_sort(ax.get_legend_handles_labels(),1)),
					**plot_props.get('other',{}).get('legend',{}))


	# Plot Colourbar
	if plot_props.get('data',{}).get('plot_type','plot') in plot_image_types:
		
		cbar_props = plot_props.get('other',{}).get('cbar',{}).copy()
		cax = make_axes_locatable(ax).append_axes('right',
												  size='5%',pad=0.05)
		
		plt.cla()
		fig.sca(cax)
		cbar = plt.colorbar(plot,cax=cax)
		cbar.set_label(**cbar_props.get('title',{}))
		
		if plot_props.get('data',{}).get('plot_range') is not None:
			try:
				cbar_vals=np.array(list(set(
										 plot_props['data']['plot_range'])))
				vals_slice = cbar_props.get('vals_slice',slice(0,None,1))
				cbar_labels = cbar_vals[vals_slice]
				cbar_vals = cbar_vals[vals_slice]+0.5
			except RuntimeError:
				cbar_vals = np.linspace(
									min(plot_props['data']['plot_range']),
									max(plot_props['data']['plot_range']),5)
				cbar_labels = cbar_vals
			cbar.set_ticks(cbar_vals)
			cbar.set_ticklabels(cbar_labels)
			
		else:
			cbar_vals = None
			cbar_labels = None
		
		if cbar_props.get('plot') == False:
			cbar.ax.set_visible(False)

	
	# Plot Ticks
	for w,wlim in plot_props.get('other',{}).get('axis_ticks',{}).items():
		if wlim.get('lim'):
			wmin,wmax = plt.getp(ax,w+'lim')
			if wmax-wmin > wlim.get('lim') and wlim.get('ticksmax') != None:
				getattr(ax,w+'axis').set_major_locator(
									  plt.MultipleLocator(wlim['ticksmax']))
			elif wlim.get('ticksmin') != None:
				getattr(ax,w+'axis').set_major_locator(
									  plt.MultipleLocator(wlim['ticksmin']))
	
	
	# Set Figure Properties
	plot_not_set_props = ['other','data','plot']
	for prop in np.setdiff1d(list(plot_props.keys()),plot_not_set_props):
		if '_attr' in prop:
			obj = locals()[prop.replace('_attr','')]
			for k,p in plot_props[prop].items():
				if not callable(p):
					p0 = p
					p = lambda a: p0
				if 'get_' in k:
					obj_k = getattr(obj,k)()
				else:
					obj_k = getattr(obj,k)
					
				plt.setp(obj_k,**p(plt.getp(obj,k.replace('get_',''))));
		else:
			obj = locals().get(prop,None)
			prop_obj = plot_props.get(prop,{})
			for p in prop_obj.copy():
				if callable(prop_obj[p]):
					prop_obj[p] = prop_obj.pop(p)(plt.getp(obj,p))
			
			plt.setp(obj,**prop_obj);

	# Pause
	plt.pause(plot_props['other'].get('pause',0.01))
	
	return
	
	
def get_props(data,domain,key,plot_props):

	# Setup Image Colourbar
	if plot_props.get('data',{}).get('plot_type','plot') in plot_image_types or(
				plot_props.get('other',{}).get('cbar_plot')):
		cbar_props = plot_props.get('other',{}).get('cbar')
		# Colorbar Range
		if plot_props.get('data',{}).get('plot_range') is None:
			vmin = np.min(data[~np.isnan(data)])
			vmax = np.max(data[~np.isnan(data)])
			plot_range = np.linspace(vmax,vmin,10)
		else:
			vmin = min(plot_props['data']['plot_range'])
			vmax = max(plot_props['data']['plot_range'])
			plot_range = plot_props['data']['plot_range']
		n_plot_range = len(plot_range)
		
		# Colorbar Normalization
		color_list = cbar_props.get('color','jet')
		if data.dtype == np.integer and (
					plot_props.get('data',{}).get('plot_range') is not None):
						
			cmap=plt.cm.get_cmap(color_list,n_plot_range)
			cmap.set_bad(color = cbar_props.get('color_bad', 'magenta'))
			
			norm = colors.BoundaryNorm(plot_props['data']['plot_range'],
									   ncolors=n_plot_range)
			
			# Update plot_props
			plot_props.get('plot',{})['cmap'] = cmap
			plot_props.get('plot',{})['norm'] = norm
			
		
		else:
			cmap=plt.cm.get_cmap(color_list)
			# if isinstance(color_list,list):
				# ncolors = np.size(data) #color_list[2]
				# color_list =list(Color(color_list[0]).range_to(
								 # Color(color_list[1]),
								 # ncolors))
				# color_list = list(map(lambda x: (x.red,x.green,x.blue),
												# color_list))
				# cmap = colors.ListedColormap(color_list)
			
			#norm = MidpointNormalize(cbar_props.get('midpoint'))
			
			cmap.set_bad(color = cbar_props.get('color_bad', 'magenta'))
			plot_props.get('plot',{})['cmap'] = cmap
			#plot_props.get('plot',{})['norm'] = MidpointNormalize(cbar_props.get('midpoint'))
			

		
		
		#plot_props.get('data',{})['plot_range'] = plot_range
	# Setup Labels
	plot_props.get('plot',{})['label'] = plot_props.get('other',{}).get('label',
													 lambda x:str_check(x))(key)

	# Setup Plot Data
	y = np.squeeze(plot_props.get('data',{}).get('data_process',
								lambda x:np.real(x))(data))
	
	x = np.squeeze(plot_props.get('data',{}).get('domain_process',
								lambda x:np.real(x))(domain))

	return y,x, plot_props.get('plot',{})
	
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
#        
#
#    fig.canvas.mpl_connect("pick_event", onpick)
#    plt.show()