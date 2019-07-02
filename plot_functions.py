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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import axes_size

# Import miscellaneous other functions
from miscellaneous_functions import sort_unhashable

def plot_decorator(plot_func):
    
	def plot_wrapper(data,domain={},fig=None,axes={},key=None,props={}):
		
		# Define Figure and Axes
		ax = axes.get(key)
		ax.axis('on')     
		ax.cla()					
		

		# Pre process data and properties
		pre_process(props)


		if isinstance(data,dict):
			plot = {}
			for k in data.keys():
				
				# Get plot label
				if callable(props['plot'].get('label')):
					props['plot']['label'] = props['plot']['label'](k)

				# Plot data
				plot[k] = plot_func(*set_data(data,domain,k,**props['data']),
										props['plot'])		
				plt.draw()
			
		else:
			plot = {}
			k = None

			# Get plot label
			if callable(props['plot'].get('label')):
				props['plot']['label'] = props['plot']['label'](k)

			# Plot data
			plot[k] = plot_func(*set_data(data,domain,k,**props['data']),
										props['plot'])
			plt.draw()
		# Post process data and properties
		post_process(fig,axes,plot[k],key,props)

		return				

	return plot_wrapper


@plot_decorator
def plot_plot(x,y,props={}):
	plot = plt.plot(x,y,**props)
	return plot


@plot_decorator
def plot_histogram(x,y,props={}):
	if np.size(y) <= 1:
		return plt.plot([],label=props['label'])
	props['bins'] = min(10,int(1+3.322*np.log10(np.size(y))))
	plot = plt.hist(y,**props)
	return plot


@plot_decorator
def plot_scatter(x,y,props={}):
	plot = plt.scatter(x,y,**props)
	return plot


@plot_decorator
def plot_image(x,y,props={}):  
	plt.cla()
	plot = plt.imshow(y,**props)
	return plot



	
	

# Set figure properties
def set_prop(fig,ax,plot,props={},
				props_not_set = ['other','data','plot'],**kwargs):
	
	for prop in np.setdiff1d(list(props.keys()),props_not_set):
		if '_attr' in prop:
			obj = locals()[prop.replace('_attr','')]
			
			for k,p in props[prop].items():
				if not callable(p):
					p = lambda a,p0 = p: p0
				
				if 'set_' in k:
					getattr(obj,k)(**p)
					continue

				if 'get_' in k:
					obj_k = getattr(obj,k)()
				else:
					obj_k = getattr(obj,k)
					
				plt.setp(obj_k,**p(plt.getp(obj,k.replace('get_',''))));
		
		# Not attribute objects must
		else:
			obj = locals().get(prop,None)
			prop_obj = copy.deepcopy(props.get(prop,{}))
			for p in prop_obj.keys():
				if callable(prop_obj[p]):
					prop_obj[p] = prop_obj.pop(p)(plt.getp(obj,p))
			
			plt.setp(obj,**prop_obj);

	return

# Set data for plotting
def set_data(data=[],domain=[],key=None,
		   data_process=None,domain_process=None,
		   **kwargs):

	if not callable(data_process):
		data_process = lambda x:np.real(x)

	if not callable(domain_process):
		domain_process = lambda x:np.real(x)

	if not isinstance(data,dict) or not isinstance(domain,dict):
		return domain_process(domain),data_process(data)
	else:
		return domain_process(domain[key]),data_process(data[key])



# Set legend, with unique sorted labels
def set_legend(obj,props={},inds=-1,**kwargs):	

	legend_labels = sort_unhashable([a.get_legend_handles_labels()
										for a in plt.gcf().axes],
									inds=inds,key=lambda i: i[-1])
	if len(legend_labels[0]) >1:
		obj.get_legend().remove()
		obj.legend(*legend_labels,**props)

	return


# Set axis ticks based on attributes per axis, for example:
# ax.xaxis.set_major_locator(plt.MultipleLocator(base))
def set_ticks(fig=None,ax=None,plot=None, axis_ticks = {},  **kwargs):

	# Iterate over axis
	for axis,axis_props in axis_ticks.items():
		for (set_attr,set_func,set_prop),set_wrapper in axis_props.items():
			getattr(getattr(ax,axis+'axis'),set_attr)(
									set_wrapper(getattr(plt,set_func),ax,axis,
									plt.getp(ax,axis+set_prop)))
	return



# Set colormap
def set_colormap(color=None,boundaries=[],normalization='linear',
					color_bad='magenta',**kwargs):

	if color is None or boundaries == []:
		return {}

	ncolors = len(boundaries)

	cmap = matplotlib.cm.get_cmap(color,ncolors)
	
	if normalization == 'linear':
		norm = colors.Normalize(min(boundaries),max(boundaries))
	elif normalization == 'discrete':
		norm = colors.BoundaryNorm(boundaries,ncolors)
	elif normalization == "log":
		norm = colors.LogNorm(min(boundaries),max(boundaries))
	else:
		norm = colors.NoNorm(None,None)

	cmap.set_bad(color=color_bad)

	return {'cmap':cmap,'norm':norm}



# Set colorbar based on ticks: {tick_value: tick_label}
def set_colorbar(fig=None,axes=None,plot=None,key=None,cmap=None,norm=None,
				 labels={},
				 props={'position':'right','size':'5%','pad':'2%'},
				 new_ax  = False,
				 update_ax = False,
				 display=True,**kwargs):
		
		
		if (not update_ax and axes.get(key+'_cax'))  or (
								cmap is None or not display):
			return

		elif new_ax and not axes.get(key+'_cax'):
			cax = fig.add_axes(new_ax,frame_on=False)
			cax.axis('off')
		elif axes.get(key+'_cax'):
			cax = axes.get(key+'_cax')
		else:
			cax = axes[key]
		cax = make_axes_locatable(cax).append_axes(**props)
		
		
		axes[key+'_cax'] = cax

		# size=axes_size.AxesY(axes[key], aspect=1./props.pop('aspect',100))
		# pad = axes_size.Fraction(props.pop('pad_fraction', 1),size)
		# props.update(dict(size=size,pad=pad))
		fig.sca(cax)
		plt.cla()

		try:
			cbar = plt.colorbar(plot,ax=axes[key],cax=cax)
		except:
			smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
			smap.set_array([]);
			fig.colorbar(smap,cax=cax,**props)		

		if isinstance(labels.get('ticks'),dict):
			cbar.set_ticks(list(labels['ticks'].keys()))
			cbar.ax.set_yticklabels(list(labels['ticks'].values()),**labels.get('ticks_params',{}))
			

		cbar.set_label(**labels.get('label',{}))
		

		cbar.ax.set_visible(display)

		return cax


# Pre processing of plot
def post_process(fig,axes,plot,key,props={}):

	ax = axes[key]
	# Set properties
	set_prop(fig,ax,plot,props)

	# Show figure title
	if props.get('other',{}).get('sup_title'):
		fig.suptitle(**props.get('other',{}).get('sup_title',{}))

	# Plot legend on axes or figure
	if props.get('other',{}).get('legend'):
		set_legend(ax,props.get('other',{}).get('legend',{}))
	
	if props.get('other',{}).get('sup_legend'):
		set_legend(fig,props.get('other',{}).get('legend',{}))


	# Set ticks
	if props.get('other',{}).get('ticks'):
		set_ticks(fig,ax,plot,**props.get('other',{}).get('ticks',{}))

	# Set colorbar
	if props.get('other',{}).get('colorbar',{}):
		set_colorbar(fig,axes,plot,key,**props.get('other',{}).get('colorbar'))

	# Set which axes has labels
	for a in fig.axes:
		try:
			a.label_outer()
		except:
			pass

	# Adjust layout of figure
	try:
		fig.tight_layout(**props.get('other',{}).get('figure_layout',
									{'pad':0.05,'w_pad':0.1, 'h_pad':0.1}))
	except:
		pass
	try:
		fig.subplots_adjust(**props.get('other',{}).get('subplots_adjust',
														{'top':0.85}))
	except:
		pass


	# Other properties
	plt.pause(props.get('other',{}).get('pause',0.01))

	# Clear colorbar for next plot
	# if props.get('other',{}).get('colorbar'):
		# cax.cla()
		# cax.axis('off')
	
	return

	

# Pre processing of plot
def pre_process(props={}):

	# Ensure fields in props exist
	for p in ['plot','data','other']:
		if not props.get(p):
			props[p] = {}

	# Define style
	plt.style.use(props['other'].get('style','matplotlibrc'))

	# Set colormap
	colormap = set_colormap(**props['other'].get('colorbar',{}))
	props['plot'].update(colormap)
	props['other'].get('colorbar',{}).update(colormap)

	return






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