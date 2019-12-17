# Import standard python modules
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Import miscellaneous other functions
from misc_functions import texify

FONT_SIZE = 12

# Set value in nested dictionary  with branch of keys [key_0,...key_n]                
def set_nested_dict(dictionary,keys,func,**kwargs):
	for key_0 in dictionary.keys():
		dictionary_k = dictionary[key_0]
		for key_i in keys[:-1]:
			if not dictionary_k.get(key_i): dictionary_k.update({key_i:{}})
			dictionary_k = dictionary_k[key_i]
		dictionary_k[keys[-1]] = func(keys,**kwargs)
	return



def set_plot_test(keys,props={},**kwargs):
	
	def ticks_multiple(func,ax,axis,prop,threshold=1,base_max=1/2,base_min=1/4):
		if sum([-k if i==0 else k for i,k in enumerate(prop)]) > threshold:
			return func(base_max)
		else:
			return func(base_min)



	# Default props
	plot_props = {
					 k: {
					
					  'ax':   {'title' : kwargs.get('title',''), 
								'xlabel': texify('$x$',every_word=None),
								'ylabel': texify('$y$',every_word=None),
								'xticks': [-np.pi,0,np.pi], 
								'xticklabels': [r'$-\pi$',r'$0$',r'$\pi$'],
								'zorder':1},
					  'ax_attr': {'get_xticklabels':{'visible':True,
													 'fontsize':FONT_SIZE},
								  'xaxis': {'ticks_position': 'none'},
								  'get_yticklabels':{'visible':True,
													 'fontsize':FONT_SIZE},
								  'yaxis': {'ticks_position': 'none'}},
					  'plot':  {'label': lambda x='':texify(x,
												   every_word=True,
												   sep_char=' ',split_char='_')},
					  'data':  {'plot_type':'scatter',
								'data_process':lambda data: np.real(data)},
								
					  'other': {'style':'simulation.mplstyle',
					  			'label': lambda x='':texify(x,
												   every_word=True,
												   sep_char=' ',split_char='_'),
					  			# 'ticks':{'x':{('set_major_locator',
					  			# 			   'MultipleLocator','lim'): 
					  			# 					ticks_multiple}},
								'colorbar': {										
										'labels':{'label':
													{'label':'Frequencies',
												 	 'fontsize':FONT_SIZE},
												 'ticks_params':
												 	{'labelsize':FONT_SIZE},
												  'ticks':None},
										'color':'viridis',
										'color_bad':'magenta',
										'new_ax':False,
										'ticks':{},
     									'props':{'position':'right',
										 		 'size':'5%','pad':'2%', 
										 		 'shrink':0.7}},										 		    
								'pause':0.1,
								'sup_legend': None,
								'legend': {'title':'Frequencies',
											'prop':{'size': FONT_SIZE},
										   'bbox_to_anchor':(1.02,0.5),
										   'borderaxespad':0, 
										   'loc':"center left",
										   'ncol':2
										  },
								'tight_layout':{'pad':0.05,'w_pad':0.1, 
												 'h_pad':0.1},
								'subplots_adjust':{'top':0.85},
								'sup_title': {'t': texify('Cavity Modes - %s'%(
												kwargs.get('sup_title','')))}
								}
					 }
					for k in keys}
					  
					  



	# Update with defined props
	plot_props.update(props)


	return plot_props




def set_plot_montecarlo(keys,**kwargs):
	def data_process(data,d):
		n = np.power(np.size(data),1/d)
		if int(n) == n and d>1:
			return np.real(data.reshape(n,tuple(n for i in range(d))))
		else:
			return np.real(np.reshape(data,(np.size(data),1)))
	return {
					 k: {
					
					  'ax':   {'title' :'', 
								'xlabel':texify(k,every_word=False),
								'ylabel': ''},
					  'ax_attr': {'get_xticklabels':{'visible':False,
													 'fontsize':FONT_SIZE},
								  'xaxis': {'ticks_position': 'none'},
								  'get_yticklabels':{'visible':False,
													 'fontsize':FONT_SIZE},
								  'yaxis': {'ticks_position': 'none'}},
					  'plot':  {'interpolation':'nearest'},
					  'data':  {'plot_type':'image',
								'data_process': lambda data: data_process(data,
															kwargs.get('d',2))},
								
					  'other': {'style':'simulation.mplstyle',
					  			# 'ticks':{'x':{('set_major_locator',
					  			# 			   'MultipleLocator','lim'): 
					  			# 					ticks_multiple}},
								'colorbar': {										
									'labels':{'label':
												{'label':'Spin Values',
											 	 'fontsize':10},
											  'ticks':{i-1.5:i 
											  		for i in 
											  		range(1,kwargs['q']+3,2)},
											 'ticks_params':
											 	{'fontsize':10},
											},
									'color':'viridis',
									'normalization':'discrete',
									'color_bad':'magenta',
									'new_ax':[0,0.245,0.94,0.47],
									'update_ax':False,
									'display': i==0,
									'boundaries':list(range(1,kwargs['q']+2)),
 									'props':{'position':'right',
									 		 'pad':0.6,'size':0.2,
									 		 }} ,										 		    
								'pause':0.001 if i==1 else 0.001,
								'sup_title':{'t': texify('Iteration: '+
														'%d'%kwargs['i'],
														every_word=False)},
								# 'tight_layout':{'pad':0.05,'w_pad':0.1, 
												 # 'h_pad':0.1},
								# 'subplots_adjust':{'top':0.85},
								}
					 }
					for i,k in enumerate(sorted(keys))}