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
# 								'zorder':1},
# 					  'ax_attr': {'get_xticklabels':{'visible':True,
# 													 'fontsize':FONT_SIZE},
# 								  'xaxis': {'ticks_position': 'none'},
# 								  'get_yticklabels':{'visible':True,
# 													 'fontsize':FONT_SIZE},
# 								  'yaxis': {'ticks_position': 'none'}},
# 					  'plot':  {'label': lambda x='':texify(x,
# 												   every_word=True,
# 												   sep_char=' ',split_char='_')},
# 					  'data':  {'plot_type':'scatter',
# 								'data_process':lambda data: np.real(data)},
								
# 					  'other': {'style':'simulation.mplstyle',

# def PROPS(plot,**kwargs):
#     props = {}
#     if plot == 'hist':
#         props.update({
#                 'plot':{
#                         'color':kwargs.get('color'),
#                         'alpha':kwargs.get('alpha',1),
#                         'orientation': kwargs.get('orientation','vertical'),
#                         'bins':kwargs.get('bins',100),
#                         'range': kwargs.get('range',None),
#                         'density':kwargs.get('density',True),
#                         'zorder':kwargs.get('zorder'),
#                         'label':kwargs.get('label'),
#                         },
#                 'type':plot,
#                 'ax':   {'title' : kwargs.get('title',''), 
#                         'xlabel': kwargs.get('xlabel',r'$X$'),
#                         'ylabel': kwargs.get('ylabel','Counts'), 
#                         'set_aspect':kwargs.get('aspect'),
#                         'set_xlim':{k:v for k,v in zip(['left','right'],kwargs.get('xlim',[None,None]))},
#                         'set_ylim':{k:v for k,v in zip(['bottom','top'],kwargs.get('ylim',[None,None]))},
#                         'set_xscale':{'value':kwargs.get('xscale','linear')},
#                         'set_yscale':{'value':kwargs.get('yscale','linear')}                         
#                         },
#                 'legend': kwargs.get('legend',{}),
#                 'figsize':kwargs.get('figsize',None)            
#                }) 
#     elif plot in ['plot']:
#         props.update({
#                 'plot':{'alpha':kwargs.get('alpha'),
#                         'color':kwargs.get('color'),
#                         'marker':kwargs.get('marker'),
#                         'linestyle':kwargs.get('linestyle','-'),
#                         'linewidth':kwargs.get('linewidth'),
#                         'markersize':kwargs.get('markersize'),
#                         'zorder':kwargs.get('zorder'),
#                         'label':kwargs.get('label')
#                         },
#                 'type':plot,
#                 'ax':   {'title' : kwargs.get('title',''), 
#                         'xlabel': kwargs.get('xlabel',r'$X$'),
#                         'ylabel': kwargs.get('ylabel',''),
#                         'set_aspect':kwargs.get('aspect'),
#                         'set_xlim':{k:v for k,v in zip(['left','right'],kwargs.get('xlim',[None,None]))},
#                         'set_ylim':{k:v for k,v in zip(['bottom','top'],kwargs.get('ylim',[None,None]))},
#                         'set_xscale':{'value':kwargs.get('xscale','linear')},
#                         'set_yscale':{'value':kwargs.get('yscale','linear')}
#                         },
#                   'legend':kwargs.get('legend',{}),
#                   'grid': kwargs.get('grid',False),       
#                })
#     elif plot in ['scatter']:
#         props.update({
#                 'plot':{'alpha':kwargs.get('alpha',1),
#                         'c':kwargs.get('c'),
#                         'marker':kwargs.get('marker'),
#                         's':kwargs.get('s'),
#                         'edgecolors':kwargs.get('edgecolors'),
#                         'linewidths':kwargs.get('linewidths'),
#                         'zorder':kwargs.get('zorder'),
#                         'label':kwargs.get('label'),
#                         'cmap': kwargs.get('cmap'),
#                         'norm': (matplotlib.colors.Normalize(kwargs['norm'].get('vmin'),
#                                         kwargs['norm'].get('vmax')) 
#                                       if (kwargs.get('norm') and 
#                                           kwargs.get('normalize','linear')=='linear') 
#                                       else 
#                                           matplotlib.colors.LogNorm(
#                                               max(kwargs['norm'].get('vmin'),1e-14),
#                                                   kwargs['norm'].get('vmax')) 
#                                       if (kwargs.get('norm') and 
#                                           kwargs.get('normalize')=='log') 
#                                  else None)
#                         },
#                 'type':plot,
#                 'ax':   {'title' : kwargs.get('title',''), 
#                         'xlabel': kwargs.get('xlabel',r'$X$'),
#                         'ylabel': kwargs.get('ylabel',''),
#                         'set_aspect':kwargs.get('aspect'),
#                         'set_xlim':{k:v for k,v in zip(['left','right'],
#                                     kwargs.get('xlim',[None,None]))},
#                         'set_ylim':{k:v for k,v in zip(['bottom','top'],
#                                     kwargs.get('ylim',[None,None]))},
#                         'set_xscale':{'value':kwargs.get('xscale','linear')},
#                         'set_yscale':{'value':kwargs.get('yscale','linear')}
#                         },
#                   'legend':kwargs.get('legend',{}),
#                   'colorbar': kwargs.get('colorbar'),
#                   'grid': kwargs.get('grid',False),       
#                })
#     elif plot == 'fill_between':
#         props.update({
#                 'plot':{'alpha':0.2,
#                         'color':kwargs.get('color'),
#                         'zorder':kwargs.get('zorder',-np.inf),
#                         'label':kwargs.get('label'),
#                         'linewidth':kwargs.get('linewidth')
#                         },
#                 'type':plot,
#                 'ax':   {'title' : kwargs.get('title',''), 
#                         'xlabel': kwargs.get('xlabel',r'$X$'),
#                         'ylabel': kwargs.get('ylabel',''),
#                         'set_aspect':kwargs.get('aspect'),
#                         'set_xlim':{k:v for k,v in zip(['left','right'],
#                                     kwargs.get('xlim',[None,None]))},
#                         'set_ylim':{k:v for k,v in zip(['bottom','top'],
#                                     kwargs.get('ylim',[None,None]))},
#                         'set_xscale':{'value':kwargs.get('xscale','linear')},
#                         'set_yscale':{'value':kwargs.get('yscale','linear')}                         
#                         },
#                   'legend':kwargs.get('legend',{}),
#                   'grid': kwargs.get('grid',False),       
#                }),
#     elif plot == 'Circle':
#         props.update({
#                 'plot':{'radius':kwargs.get('radius'),
#                         'center':kwargs.get('center'),
#                         'alpha':1,
#                         'fill':kwargs.get('fill',False),
#                         'color':kwargs.get('color','r'),
#                         'zorder':kwargs.get('zorder',np.inf),
#                         'label':kwargs.get('label'),
#                         'linewidth':kwargs.get('linewidth')
#                         },
#                 'type':plot,
#                 'ax':   {'title' : kwargs.get('title',''), 
#                         'xlabel': kwargs.get('xlabel',r'$X$'),
#                         'ylabel': kwargs.get('ylabel',''),
#                         'set_aspect':kwargs.get('aspect'),
#                         'set_xlim':{k:v for k,v in zip(['left','right'],
#                                     kwargs.get('xlim',[None,None]))},
#                         'set_ylim':{k:v for k,v in zip(['bottom','top'],
#                                     kwargs.get('ylim',[None,None]))},
#                         'set_xscale':{'value':kwargs.get('xscale','linear')},
#                         'set_yscale':{'value':kwargs.get('yscale','linear')}                         
#                         },
#                   'legend':kwargs.get('legend',{}),
#                   'grid': kwargs.get('grid',False),       
#                }),
#     elif plot in ['contour','contourf','tricontour','tricontourf']:
#         props.update({
#                 'plot':{'alpha':1,
#                         'label':kwargs.get('label'),
#                         'linestyles':kwargs.get('linestyles'),
#                         'cmap': kwargs.get('cmap'),
#                          'norm': (matplotlib.colors.Normalize(kwargs['norm'].get('vmin'),
#                                         kwargs['norm'].get('vmax')) 
#                                       if (kwargs.get('norm') and 
#                                           kwargs.get('normalize','linear')=='linear') 
#                                       else 
#                                           matplotlib.colors.LogNorm(
#                                               max(kwargs['norm'].get('vmin'),1e-14),
#                                                   kwargs['norm'].get('vmax')) 
#                                       if (kwargs.get('norm') and 
#                                           kwargs.get('normalize')=='log') 
#                                  else None)
#                         },
#                 'type':plot,
#                 'ax':   {'title' : kwargs.get('title',''), 
#                         'xlabel': kwargs.get('xlabel',r'$x$'),
#                         'ylabel': kwargs.get('ylabel',r'$y$'),
#                         'set_aspect':kwargs.get('aspect'),
#                         'set_xlim':{k:v for k,v in zip(['left','right'],kwargs.get('xlim',[None,None]))},
#                         'set_ylim':{k:v for k,v in zip(['bottom','top'],kwargs.get('ylim',[None,None]))},
#                         'set_xscale':{'value':kwargs.get('xscale','linear')},
#                         'set_yscale':{'value':kwargs.get('yscale','linear')}                         
#                         },
#                   'legend':kwargs.get('legend',{}),
#                   'colorbar': kwargs.get('colorbar'),
#                   'grid': kwargs.get('grid',False),       
#                }),
#     elif plot == 'loglog':
#         props.update({
#                 'plot':{'alpha':1,
#                         'color':kwargs.get('color'),
#                         'marker':kwargs.get('marker'),
#                         'linestyle':kwargs.get('linestyle','-'),
#                         'markersize':kwargs.get('markersize'),
#                         'zorder':kwargs.get('zorder'),
#                         'label':kwargs.get('label')
#                         },
#                 'type':plot,
#                 'ax':   {'title' : kwargs.get('title',''), 
#                         'xlabel': kwargs.get('xlabel',r'$X$'),
#                         'ylabel': kwargs.get('ylabel',''),
#                         'set_aspect':kwargs.get('aspect'),
#                         'set_xlim':{k:v for k,v in zip(['left','right'],kwargs.get('xlim',[None,None]))},
#                         'set_ylim':{k:v for k,v in zip(['bottom','top'],kwargs.get('ylim',[None,None]))},
#                         'set_xscale':{'value':kwargs.get('xscale','linear')},
#                         'set_yscale':{'value':kwargs.get('yscale','linear')}                         
#                         },
#                   'legend':kwargs.get('legend',{}),
#                   'grid': kwargs.get('grid',False),       
#                }),
#     elif plot == 'plot3D':
#         props.update({
#                 'plot':{'alpha':1,
#                         'color':kwargs.get('color'),
#                         'marker':kwargs.get('marker'),
#                         'linestyle':kwargs.get('linestyle','-'),
#                         'linewidth':kwargs.get('linewidth'),
#                         'markersize':kwargs.get('markersize'),
#                         'zorder':kwargs.get('zorder'),
#                         'label':kwargs.get('label')
#                         },
#                 'type':plot,
#                 'ax':   {'title' : kwargs.get('title',''), 
#                         'xlabel': kwargs.get('xlabel',r'$X$'),
#                         'ylabel': kwargs.get('ylabel',''),
#                         'zlabel': kwargs.get('xlabel',None),
#                         'set_aspect':kwargs.get('aspect'),
#                         'set_xlim':{k:v for k,v in zip(['left','right'],kwargs.get('xlim',[None,None]))},
#                         'set_ylim':{k:v for k,v in zip(['bottom','top'],kwargs.get('ylim',[None,None]))},
#                         'set_xscale':{'value':kwargs.get('xscale','linear')},
#                         'set_yscale':{'value':kwargs.get('yscale','linear')}                         
#                         },
#                   'legend':kwargs.get('legend',{}),
#                   'grid': kwargs.get('grid',False),
#                })
#     return props




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




def set_plot_montecarlo(keys,props={},**kwargs):
	def data_process(data,d):
		n = np.power(np.size(data),1/d)
		if (int(n) == n) and (d>1):
			return np.real(np.reshape(data,tuple(int(n) for i in range(d))))
		else:
			return np.real(np.reshape(data,(np.size(data),1)))

	plot_props = {
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
	props.updates(plot_props)
	return props



def set_plot_analysis(keys,props={},**kwargs):
	
	def ticks_multiple(func,ax,axis,prop,threshold=1,base_max=1/2,base_min=1/4):
		if sum([-k if i==0 else k for i,k in enumerate(prop)]) > threshold:
			return func(base_max)
		else:
			return func(base_min)



	# Default props
	FONT_SIZE = 6
	plot_props = {}
	for k in keys:
		plot_props[k] = {
					
					  'ax':   {'title' : kwargs.get('title',''), 
								'xlabel': kwargs.get('xlabel',texify('$x$',every_word=None)),
								'ylabel': kwargs.get('ylabel',texify('$y$',every_word=None)),
								'xticklabels': kwargs.get('xticklabels',[]),
								'zorder':kwargs.get('zorder',1)},
					  'ax_attr': {'get_xticklabels':{'visible':True,
													 },
								  # 'xaxis': {'ticks_position': 'none'},
								  'get_yticklabels':{'visible':True,
													 # 'fontsize':6
													 },
								  # 'yaxis': {'ticks_position': 'none'}
								  },
					  'plot':  {'label': lambda x='':texify(x,
												   every_word=True,
												   sep_char=' ',split_char='_'),
					  			**({'marker':kwargs.get('marker'),
					  				'linestyle':kwargs.get('linestyle')}
					  				if kwargs.get('plot_type') != 'histogram' else {}),
					  			**({'bins':kwargs.get('bins',20),
					  				'density':True} 
					  				if kwargs.get('plot_type')=='histogram' else {})
					  			},
					  'data':  {'plot_type':kwargs.get('plot_type','plot'),
								'data_process':lambda data: np.real(data)},
								
					  'other': {'style':'analysis.mplstyle',
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
										 		 'shrink':0.7}} if kwargs.get('colorbar') else None,										 		    
								'pause':0.1,
								'sup_legend': None,
								'legend': {'title':'Frequencies',
											'prop':{'size': FONT_SIZE},
										   'bbox_to_anchor':(1.02,0.5),
										   'borderaxespad':0, 
										   'loc':"center left",
										   'ncol':2
										  } if kwargs.get('legend') else None,
								'tight_layout':{
												 # 'pad':0.05,'w_pad':0.1, 
												 # 'h_pad':0.1
												 },
								# 'subplots_adjust':{'top':0.85},
								'sup_title': kwargs.get('sup_title')}
					}
					 
	props.update(plot_props)
	return props				


