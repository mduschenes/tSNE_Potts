
# Import standard python modules
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# Important defined functions
import plot_functions
from data_process import exporter
from misc_functions import caps


class plotter(object):
    
    # Create figure and axes dictionaries for keys: 
    # plot_keys={plot_key: [axis_keys]}, plot_bool={plot_key: plot_bool},
    # with given orientation
	def __init__(self,plot_keys={},plot_bool=None,layout_props={}):

		# Initialize figures and axes dictionaries
		self.figs = {}
		self.axes = {}
		
		# Define plot_keys and plot_bool
		self.plot_keys = {}
		self.plot_bool = {}

					
		# Add Figures and Axes with keys
		self.add_plot(plot_keys,plot_bool,layout_props=layout_props)

		return  


	# Plot data: {plot_key:{axis_key:{draw_key:data}}}, as nested dictionaries, 
	# based on plot_keys, with figure_shape: {plot_key:figure_shape},
	# and with plot_props: {plot_key: props}
	def plot(self,data,domain=None,props={},layout_shape=None,layout=None):
		
		for plot_key,plot_data in data.items():

			# Only alter figures that are to be plotted
			if not self.plot_bool[plot_key]:
				continue

			# Create Figures and Axes (if they don't already exist)									

			axes_keys = np.array(self.plot_keys.get(plot_key,[]))
			self.add_plot({plot_key:axes_keys}, layout=layout)
			
			
			# Plot for each data key
			for axes_key in sorted([a for a in axes_keys.flatten() if a is not None
									]):
				if axes_key not in data[plot_key].keys():
					continue
			
				# Define axis properties
				props_ax = props.get(plot_key,{}).get(axes_key,props)
				# props['other']['plot_key'] = plot_key
				# props['other']['axes_key'] = axes_key

				# Plot axis data
				getattr(plot_functions,'plot_' + 
						props[plot_key][axes_key].get('data',{}).get(
						'plot_type','plot'))(
									data[plot_key][axes_key],
									domain[plot_key][axes_key],
									self.figs[plot_key],self.axes[plot_key],
									axes_key,props[plot_key][axes_key])

											

		return
        
    # Add new plots based on plot_keys and plot_bool
	def add_plot(self,plot_keys,plot_bool=None,layout=None,layout_props={}):
		for k in plot_keys:
			self.plot_keys[k] = plot_keys[k]#np.array(plot_keys[k]).flatten().tolist()
			if not self.figs.get(k):
				self.figs[k] = None
			if not self.axes.get(k):
				self.axes[k] = {}
		
		if plot_bool is None:
			self.plot_bool.update({k:True for k in plot_keys.keys()})
		self.figures_axes(plot_keys,layout,**layout_props)
		return

	def scf(self,key):
		plt.figure(self.figs[key].number)
		return

	# Close all current figures and reset figures and axes dictionaries
	def plot_close(self):
		plt.close('all');   
		self.axes = {}
		self.figs ={}
		return


	# Save all current figures based on files: {file: figure_key}
	def plot_export(self,files={},directory='./',
					options={'dpi':500,'bbox_inches':'tight',
							 'figure_size':None}):
			
		
		# Save all current figures if files is empty
		if files == {}:
			files = {k:str(k)+'.pdf' for k in self.figs.keys()}
		
		for key,file in files.items():
					
			# Set current figure
			fig = self.figs[key];  
			
			# Change plot size for saving            
			figure_size = fig.get_size_inches();
			if options.get('figure_size'):
				fig.set_size_inches(options.pop('figure_size'));
			else:
				pass
				# fig.set_size_inches(11,8,5)

			# Export figure to file
			exporter({file:fig},directory,options);

			# Revert to original plot size
			fig.set_size_inches(figure_size); 
			
		return

  


	# Create figures and axes for keys of the form: {plot_key:[axis_keys]}
	# or update layout of existing figures with layout = {plot_key: layout_func}
	def figures_axes(self,plot_keys,layout=None,**layout_props):    

		def layout_grid(keys,fig,**props):
			gs = gridspec.GridSpec(*np.shape(keys)[:2],**props)        	
			axes = {}
			for i,k in enumerate(np.array(keys).flatten()):
				if k is not None:
					axes[k] = fig.add_subplot(list(gs)[i])
			return axes

		def layout_skip(keys,fig,**props):
			gs = gridspec.GridSpec(*np.shape(keys)[:2],**props)        	
			axes = {}
			for i,k in enumerate(
							np.array(keys).flatten()[slice(0,None,2)]):
				if k is not None:
					axes[k] = fig.add_subplot(list(gs)[i:i+2])
			return axes
 
		figure_props = layout_props.pop('figure',{})
		for plot_key,axis_keys in plot_keys.items():

			# Only alter figures that are to be plotted that do not exist
			if not self.plot_bool.get(plot_key):
				continue

			# If layout exists, update existing axes with new layout 
			elif layout is not None and layout.get(plot_key):
				self.axes[plot_key] = layout[plot_key](axis_keys,
												self.figs[plot_key],
												**layout_props)
				continue

			# Only update non-existing figures based on self.axes[plot_key]
			elif any([k not in self.axes.get(plot_key,{}).keys() 
					for k in np.array(axis_keys).flatten() if k is not None]):

				self.figs[plot_key] = plt.figure(**figure_props)
				self.axes[plot_key] = layout_grid(axis_keys,
												self.figs[plot_key],
												**layout_props)
				for ax in self.figs[plot_key].axes:
					ax.cla()
					ax.axis('off')


				# Label figure
				self.figs[plot_key].canvas.set_window_title(
							'Figure %d:  %s Datasets'%(
								self.figs[plot_key].number,caps(plot_key)))
				continue
			else:
				continue

		return

