# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 02:43:27 2018

@author: Matt
"""
import numpy as np
import copy
from functools import wraps
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import axes_size
from matplotlib.legend_handler import HandlerPatch

# Import miscellaneous other functions
from misc_functions import sort_unhashable

def plot_decorator(plot_func):
    
	@wraps(plot_func)
	def plot_wrapper(data,domain={},fig=None,axes={},key=None,props={}):
		
		# Define Figure and Axes
		plt.figure(fig.number);
		ax = axes.get(key);
		# fig.sca(ax);
		ax.axis('on')   
		



		if isinstance(data,dict):
			plot = {}
			for k in data.keys():

				# Pre process data and properties
				prop = copy.deepcopy(props.get(k,props))
				pre_process(prop)

				try:
					function = globals()['plot_'+prop['data']['plot_type']].__wrapped__
				except:
					print('errrr')
					function = plot_func
				# Get plot label
				if callable(prop['plot'].get('label')):
					prop['plot']['label'] = prop['plot']['label'](k)

				# Plot data

				plot[k] = function(*set_data(data,domain,k,**prop['data']),
										fig,ax,prop['plot'])		

				# Post process data and properties
				post_process(fig,axes,plot[k],key,prop)
			
		else:
			pre_process(props)
			plot = {}
			k = None

			# Get plot label
			if callable(props['plot'].get('label')):
				props['plot']['label'] = props['plot']['label'](k)

			# Plot data
			plot[k] = plot_func(*set_data(data,domain,k,**props['data']),
										fig,ax,props['plot'])

			# Post process data and properties
			post_process(fig,axes,plot[k],key,props)

		return				

	return plot_wrapper


@plot_decorator
def plot_plot(x,y,fig,ax,props={}):
	plot = plt.plot(x,y,**props)
	return plot


@plot_decorator
def plot_histogram(x,y,fig,ax,props={}):
	if np.size(y) <= 1:
		return plt.plot([],label=props['label'])
	props['bins'] = min(10,int(1+3.322*np.log10(np.size(y))))
	plot = plt.hist(y,**props)
	return plot


@plot_decorator
def plot_scatter(x,y,fig,ax,props={}):
	plot = plt.scatter(x,y,**props)
	return plot


@plot_decorator
def plot_image(x,y,fig,ax,props={}):  
	plt.cla()
	plot = plt.imshow(y,**props)
	return plot

@plot_decorator
def plot_figimage(x,y,fig,ax,props={}):
	plot = plt.figimage(y,**props)
	return plot


@plot_decorator
def plot_graph(x,y,fig,ax,props={}):  
	props.update({'fig':fig,'ax':ax})
	# if 'norm' in props:
	# 	for keys,vals in props.items():
	# 		if isinstance(vals,dict):
	# 			for key,val in vals.items():
	# 				if isinstance(val,dict):
	# 					for k,v in val.items():
	# 						if (isinstance(k,str) and 'color' in k and 
	# 						   isinstance(props[keys][key][k],(list,np.ndarray))
	# 						   and not isinstance(props[keys][key][k][0],str)):
	# 							props[keys][key][k] = props['norm']([
	# 											d if d!=0 else d+1e-14 
	# 											for d in props[keys][key][k]])
	# 				elif (isinstance(key,str) and 'color' in key and 
	# 					  isinstance(props[keys][key],(list,np.ndarray)) and
	# 					  not isinstance(props[keys][key][0],str)):
	# 					props[keys][key] = props['norm']([d if d!=0 else d+1e-14 
	# 											for d in props[keys][key]])
	# 		elif (isinstance(keys,str) and 'color' in keys and 
	# 		     isinstance(props[keys],(list,np.ndarray)) and 
	# 		     not isinstance(props[keys][0],str)):
	# 			props[keys] = props['norm']([d if d!=0 else d+1e-14 
	# 										 for d in props[keys]])
	plot = y.graph_plot(**props)
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
			prop_objs = {}
			for p in prop_obj.keys():
				if callable(prop_obj[p]):
					prop_objs[p] = copy.deepcopy(prop_obj[p])
					prop_obj[p] = prop_obj.pop(p)(plt.getp(obj,p))
				else:
					prop_objs[p] = copy.deepcopy(prop_obj[p])
			
			plt.setp(obj,**prop_obj);
			prop_obj[p] = prop_objs[p]

	return

# Set data for plotting
def set_data(data=[],domain=[],key=None,
		   data_process=None,domain_process=None,
		   **kwargs):

	if not callable(data_process):
		data_process = lambda x:x

	if not callable(domain_process):
		domain_process = lambda x:x

	try:
		if not isinstance(data,dict) or not isinstance(domain,dict):
			return domain_process(domain),data_process(data)
		else:
			return domain_process(domain[key]),data_process(data[key])
	except:
		if not isinstance(data,dict) or not isinstance(domain,dict):
			return domain,data
		else:
			return domain[key],data[key]



# Set legend, with unique sorted labels
def set_legend(obj,handle,label,props={},inds=-1,**kwargs):	

	def make_legend_arrow(legend, orig_handle,
	      xdescent, ydescent,
	      width, height, fontsize):
		return matplotlib.patches.FancyArrow(0, 0.5*height, width, 0, 
	                               length_includes_head=True, 
	                                head_width=0.5*height)
	def make_legend_rectangle(legend, orig_handle,
	      xdescent, ydescent,
	      width, height, fontsize):
		return matplotlib.patches.FancyArrow(0, 0.5*height, width, 0, 
	                               length_includes_head=True, 
	                                head_width=0.25*height)
	
	for ax in plt.gcf().axes:
		h,l=[],[]
		hi,li = ax.get_legend_handles_labels()
		h.extend(hi)
		l.extend(li)
		handles,texts = ax.get_legend().legendHandles,ax.get_legend().texts           
		if not ([] in handles or [] in texts):
			h.extend(handles)
			l.extend([t.get_text() for t in texts])

	if label not in l:
		h.append(handle)
		l.append(label)

	h = [hi[1] for hi in sorted(enumerate(h),
	                            key=lambda x:l[x[0]],reverse=True)]
	l = [li[1] for li in sorted(enumerate(l),
	                            key=lambda x:l[x[0]],reverse=True)]
		

	# handles_labels = sort_unhashable([a.get_legend_handles_labels()
	# 									for a in plt.gcf().axes],
	# 								inds=inds,key=lambda i: i[-1])
	try:
		obj.get_legend().remove()
	except:
		pass	
	
	if h !=[] and l != [] and props_plotting.get('legend') is not None:
		obj.legend(h,l,handler_map={
	            matplotlib.patches.FancyArrowPatch : HandlerPatch(
	                                patch_func=make_legend_arrow),
	            matplotlib.patches.FancyArrow : HandlerPatch(
	                                patch_func=make_legend_arrow),
	            matplotlib.patches.Rectangle : HandlerPatch(
	                                patch_func=make_legend_rectangle)
	            },**props_plotting.get('legend',{}))   

	return


# Set annotations on plot
def set_annotations(fig=None,ax=None,plot=None,annotations={}, **kwargs):
	for pos,options in annotations.items():
		options.update({'xy':pos})
		ax.annotate(**options)
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
					color_bad='w',**kwargs):

	if color is None or boundaries == []:
		return {}


	if isinstance(boundaries,np.ndarray):
		boundaries = np.atleast_2d(boundaries)
		ncolors = len(np.unique(boundaries,axis=0))
		boundaries = np.linalg.norm(boundaries,axis=1)
	else:
		try:
			ncolors = len(set(boundaries))
		except TypeError:
			ncolors = 1
	
	if ncolors == 1 or normalization == 'none':
		cmap = matplotlib.cm.get_cmap(color)
		norm = colors.NoNorm(None,None)
	elif normalization == 'linear':
		cmap = matplotlib.cm.get_cmap(color)
		norm = colors.Normalize(min(boundaries),max(boundaries))
	elif normalization == 'discrete':
		cmap = matplotlib.cm.get_cmap(color,ncolors)
		norm = colors.BoundaryNorm(boundaries,ncolors)
	elif normalization == "log":
		cmap = matplotlib.cm.get_cmap(color)
		norm = colors.LogNorm(max(min(boundaries),1e-14),max(boundaries))
	else:
		cmap = matplotlib.cm.get_cmap(color)
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
		
		
		if (not update_ax and axes.get(str(key)+'_cax'))  or (
								cmap is None or not display):
			return

		elif new_ax and not axes.get(str(key)+'_cax'):
			cax = fig.add_axes(new_ax,frame_on=False)
		elif axes.get(str(key)+'_cax'):
			cax = axes.get(str(key)+'_cax')
		else:
			cax = axes[key]
		if axes.get(str(key)+'_cbar'):
			cbar = axes.get(str(key)+'_cbar')
			cbar.remove()
		axes[str(key)+'_cax'] = cax
		
		fig.sca(cax);
		plt.cla();
		cax.cla();
		cax.axis('off')
		
		cax = make_axes_locatable(cax).append_axes(**props)
		fig.sca(cax)
		plt.cla();
		cax.axis('off');
		plt.cla();

		
		

		# size=axes_size.AxesY(axes[key], aspect=1./props.pop('aspect',100))
		# pad = axes_size.Fraction(props.pop('pad_fraction', 1),size)
		# props.update(dict(size=size,pad=pad))
		fig.sca(cax)
		# cax.axis('off')
		cax.clear();
		plt.cla();

		try:
			if isinstance(p,(list,np.ndarray)) and any(
					[isinstance(p,(list,np.ndarray)) for p in plot]):
				for p in plot:
					try:
						cbar = fig.colorbar(p,ax=axes[key],cax=cax)
					except:
						continue
			else:
				cbar = fig.colorbar(plot,ax=axes[key],cax=cax)
		except:
			smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
			smap.set_array([]);
			cbar = fig.colorbar(smap,cax=cax)	

		if isinstance(labels.get('ticks'),dict):
			cbar.set_ticks(list(labels['ticks'].keys()))
			cbar.ax.set_yticklabels(list(labels['ticks'].values()),
				**labels.get('ticks_params',{}))
			

		cbar.set_label(**labels.get('label',{}))
		

		cbar.ax.set_visible(display)

		axes[str(key)+'_cbar'] = cbar

		return cax


# Pre processing of plot
def post_process(fig,axes,plot,key,props={}):

	ax = axes[key]
	# Set properties
	set_prop(fig,ax,plot,props)

	# Set figure title
	if props.get('other',{}).get('sup_title'):
		fig.suptitle(**props.get('other',{}).get('sup_title',{}))

	# Set legend on axes or figure
	if props.get('other',{}).get('legend'):
		set_legend(ax,plot,props.get('label'),
				   props.get('other',{}).get('legend',{}))
	
	if props.get('other',{}).get('sup_legend'):
		set_legend(fig,plot,props.get('label'),
				   props.get('other',{}).get('legend',{}))

	# Set annotations
	if props.get('other',{}).get('annotations'):
		set_annotations(fig,ax,plot,props['other']['annotations'])


	# Set ticks
	if props.get('other',{}).get('ticks'):
		set_ticks(fig,ax,plot,**props.get('other',{}).get('ticks',{}))

	# Set colorbar
	if props.get('other',{}).get('colorbar',{}):
		set_colorbar(fig,axes,plot,key,**props.get('other',{}).get('colorbar'));


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

	if not props.get('other',{}).get('visible',True):
		for k,ax in axes.items():
			if '_cax' not in str(key) and '_cbar' not in str(key):
				try:
					ax.axis('off')
				except AttributeError:
					ax.ax.set_visible(False)


	# Other properties
	plt.pause(props.get('other',{}).get('pause',0.01))

	
	return

	

# Pre processing of plot
def pre_process(props={}):

	# Ensure fields in props exist
	for p in ['plot','data','other']:
		if not props.get(p):
			props[p] = {}

	# Clear Figure
	if props['other'].get('clear'):
		print('Clear Figure')
		plt.cla()			

	# Define style
	try:
		plt.style.use(props['other'].get('style','matplotlibrc'))
	except:
		pass

	# Set colormap
	if props['other'].get('colorbar'):
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