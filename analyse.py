# Import standard python modules
import numpy as np
import argparse,os,logging

# Import defined modules
from model import model as Model
from data_process import importer, exporter
from data_plot import plotter
from plot_properties import set_plot_analysis


# Create Plot of Oberservables
def observables(name,data,observables):

	def label(array,q,n,i=0):
		return np.sum((array-i)*np.power(q,np.arange(n)),axis=1)

	def configuration(label,q,n):
		q_n = np.power(q,np.arange(n))
		return np.mod(((np.atleast_1d(label)[:,np.newaxis]/q_n)).
                        astype(int),q)



	for obs in observables:
		
		if obs != 'energy':
			continue

		domain = {}
		samples = {}
		models = {}

		key = 'T'



		
		for file,datum in data.items():
			sites = np.array(datum['sites'])
			model = Model(**datum['model'])
			func = getattr(model,obs)

			attr = str(getattr(model,key))

			samples[attr] = model.obs_mean(getattr(model,obs),
										  None,
										  sites,model.neighbours,model.T)

			samples[attr+'_true'] = model.obs_mean(getattr(model,obs),
										  None,
										  configuration(
										  	np.arange(model.q**model.N),
											model.q,model.N),
										  model.neighbours,model.T)


			models[attr] = model


		attrs = sorted(samples.keys())

		plot_keys = {name+'_'+obs:[[name]]}
		plot = plotter(plot_keys,
						layout_props = {'figure':{'figsize':(20,20)}})
		plot_props = lambda file,**kwargs: set_plot_analysis([file],
										suptitle={'t':'Samples Histogram'},
										xlabel=r'Site $\sigma$',
										ylabel='Counts',
										**kwargs)[file]
		
		plot_data = {x: {k:{i:{} for j in v for i in j }for k,v in plot_keys.items()}
					 for x in ['data','domain','props']}
		
		for attr in attrs:
			for k,v in plot_keys.items():
				for j in v:
					for i in j:
						plot_data['domain'][i][attr] = [attr]
						plot_data['domain'][i][attr+'_true'] = [attr]

						plot_data['data'][i][attr] = samples[attr]
						plot_data['data'][i][attr+'_true'] = samples[attr+'_true']

						plot_data['props'][i][attr] = plot_props(file,
					  							plot_type='plot',
					  							zorder = 1,
					  							xlabel=key,
					  							ylabel=obs)
						plot_data['props'][i][attr+'_true'] = plot_props(file,
					  							plot_type='plot',
					  							zorder = 1,
					  							xlabel=key,
					  							ylabel=obs)


		plot.plot(**plot_data)
		plot.plot_export(directory=DIRECTORY)



# Create Histogram of Samples
def samples(name,data,observables):

	def label(array,q,n,i=0):
		return np.sum((array-i)*np.power(q,np.arange(n)),axis=1)

	def configuration(label,q,n):
		q_n = np.power(q,np.arange(n))
		return np.mod(((np.atleast_1d(label)[:,np.newaxis]/q_n)).
                        astype(int),q)


	def exact(model,n):
		# assert model.d == 1, "Error - No exact solution for d>1"
		# q = model.q
		# transfer = np.zeros((q,q))
		# for i in range(q):
		# 	for j in range(q):
		# 		transfer[i,j] = np.exp(-(1/model.T)*(
		# 					model.couplings[0]*((i==0)+(j==0))/2 + 
		# 					model.couplings[1]*(i==j)))
		# Z = np.trace(np.linalg.matrix_power(transfer,n))
		return lambda sites: np.exp(-(1/model.T)*model.energy(
										sites,model.neighbours,model.T)) 


	domain = {file:None for file in data}
	samples = {file:None for file in data}
	models = {file:None for file in data}



	
	for file,datum in data.items():
		sites = np.array(datum['sites'])
		model = Model(**datum['model'])
		
		domain[file] = np.arange(int(model.q**model.N),dtype=int)
		samples[file] = label(sites,model.q,model.N,1)



		domain[file+'_true'] = np.arange(model.q**model.N)
		probability = exact(model,model.N)
		samples[file+'_true'] = probability(configuration(domain[file+'_true'],
												model.q,model.N))
		samples[file+'_true'] /= np.sum(samples[file+'_true'])


		models[file] = model


	files = sorted(list(data.keys()),key=lambda f: models[f].T)

	plot = plotter({name:[files]},
					layout_props = {'figure':{'figsize':(20,20)}})
	plot_props = lambda file,**kwargs: set_plot_analysis([file],
									suptitle={'t':'Samples Histogram'},
									xlabel=r'Site $\sigma$',
									ylabel='Counts',
									**kwargs)[file]


	plot.plot({name:{file: {
							'Data':samples[file],
							'True':samples[file+'_true']}
				for file in files}},
			  {name:{file: {
			  				'Data':domain[file],
							'True':domain[file+'_true']}
				for file in files}},
			  {name: {file:{
			  				'Data':plot_props(file,
			  							plot_type='histogram',
			  							zorder = -np.inf,
			  							bins=128,
			  							ylim = (0,0.0125),
			  							title = 'T = %0.2f'%(models[file].T)),
			  				'True':plot_props(file+'_true',
			  							plot_type='plot',
			  							ylim = (0,0.0125),
			  							marker='*',
										linestyle='-',
			  							title = 'T = %0.2f'%(models[file].T)),
			  				} 
			  		  for file in files}})

	plot.plot_export(directory=DIRECTORY)





# Parser Object
parser = argparse.ArgumentParser(description = "Parse Arguments")

parser.add_argument('-d','--directory',help = 'Data Directory',
					type=str,default='.')

parser.add_argument('-f','--configs',help = 'Configuration File',
					type=str,default='main.config')


# Parse Args Command
for k,v in dict(**vars(parser.parse_args())).items():
	locals()[k.upper()] = v


# Import config file
config = importer([CONFIGS],DIRECTORY,options=True)[CONFIGS]

CONFIG = 'CONFIG'
PROPERTIES = ['FILE','FILETYPE','ANALYSIS','OBSERVABLES']
TYPE = int

assert CONFIG in config.sections(),'Incorrect main.config file'

for p in PROPERTIES:
	locals()[p] = config.get_typed(CONFIG,p,TYPE,atleast_1d=True)

# Import Files
data = importer([os.path.join('**/',FILE,FILE+'.'+FILETYPE)],DIRECTORY)


# Local Variables
locvars = locals()

# Analysis functions
for a in ANALYSIS:
	if a != 'observables':
		continue
	locvars.get(a,lambda *args:None)(a,data,OBSERVABLES);








# print(data)