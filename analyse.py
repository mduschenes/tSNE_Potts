# Import standard python modules
import numpy as np
import argparse,os,logging

# Import defined modules
from model import model as Model
from data_process import importer, exporter
from data_plot import plotter
from plot_properties import set_plot_analysis

# Create Histogram of Samples
def samples(name,data,observables):

	def label(array,q,n,i=0):
		return np.sum((array-i)*np.power(q,np.arange(n)),axis=1)
	plot = plotter({name:[list(data.keys())]})
	plot_props = {name:set_plot_analysis(data.keys(),
									title='Samples Histogram',
									plot_type='histogram',
									xlabel=r'Site $\sigma$',
									ylabel='Counts')}
	domain = {file:None for file in data}
	samples = {file:None for file in data}
	for file,datum in data.items():
		print(file)
		sites = np.array(datum['sites'])
		model = Model(**datum['model'])
		
		domain[file] = np.arange(int(model.q**model.N),dtype=int)
		samples[file] = label(sites,model.q,model.N,1)

	plot.plot({name:samples},{name:domain},plot_props)

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
PROPERTIES = ['DIRECTORY','FILE','FILETYPE','ANALYSIS','OBSERVABLES']
TYPE = int

assert CONFIG in config.sections(),'Incorrect main.config file'

for p in PROPERTIES:
	locals()[p] = config.get_typed(CONFIG,p,TYPE,atleast_1d=True)


# Import Files
data = importer([os.path.join(FILE,FILE+FILETYPE)],DIRECTORY)


# Local Variables
locvars = locals()

# Analysis functions
print(ANALYSIS,OBSERVABLES)
for a in ANALYSIS:
	locvars.get(a,lambda *args:None)(a,data,OBSERVABLES);








# print(data)