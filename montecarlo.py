# Import standard python modules
import numpy as np
from celluloid import Camera
import matplotlib.animation as animate
import logging

# Import data processing files
from data_process import exporter
from data_plot import plotter
from plot_properties import set_plot_montecarlo


def montecarlo(N,neighbours,props,job=0,directory='.'):

	# Setup logging
	logger = logging.getLogger(__name__)
	if props.get('quiet'):
		log = 'info'
	else:
		log = 'warning'
	


	def Niter(Ni,N,ratios):
		ratios = np.atleast_1d(ratios)
		assert 1>=sum(ratios)
		Nfunc = lambda i,r: (max(1,int(r*Ni*N)) if r>0 else 0) if not i%2 else (
							 max(1,int(r*Ni)) if r > 0 else 0)
		Niters = [Nfunc(i,r) for i,r in enumerate(ratios)]
		Niters.append(Nfunc(len(Niters),1-sum(ratios)))
		return Niters
	def Nperiod(Nf,N,Niters):
		Niters = np.atleast_1d(Niters)
		Nfunc = lambda i,n: max(1,int(N//Nf))if not i%2 else max(1,int(1//(Nf)))
		Nperiods = [Nfunc(i,n) for i,n in enumerate(Niters)]
		return Nperiods


	if props['algorithm'] in ['metropolis','wolff','metropolis_wolff']:
		alg = [(a,globals().get(a)) for a in ['metropolis','wolff']]
	else:
		alg = [(a,globals().get(a)) for a in [props['algorithm']]]
	Neqb  = Niter(props['Neqb'],N,props['Nratio'])
	Nmeas  = Niter(props['Nmeas'],N,props['Nratio'])
	Nperiods = Nperiod(props['Nfreq'],N,Nmeas)
	# Nmeas_total = [Nm//Nf for Nm,Nf in zip(Nmeas,Nperiods)]
	measure_buffer = [(j+1)*Nperiods[i]+sum(Nmeas[:i]) for i in range(len(Nmeas)) 
										for j in range(Nmeas[i]//Nperiods[i])]
	iter_buffer = list(range(len(measure_buffer)))
	stage_eqb_buffer = [(i-1,sum(Neqb[:i])) for i in range(1,len(Neqb)+1)]
	stage_meas_buffer = [(i-1,sum(Nmeas[:i])) for i in range(1,len(Nmeas)+1)]
	getattr(logger,log)(['measure_buffer',Nmeas,sum(Neqb),sum(Nmeas),measure_buffer])
	getattr(logger,log)(['stage_meas_buff',stage_meas_buffer])
	getattr(logger,log)(['stage_eqb_buff',stage_eqb_buffer])
	getattr(logger,log)('''Monte Carlo: %d Eqb MC steps, %d Meas MC Steps, %s Meas sweeps, every %s Steps'''%(
							sum(Neqb),sum(Nmeas),str(Nmeas),str(Nperiods)))

		# Array of sample sites and updated clusters during simulation
	data = {'sites':np.zeros((len(iter_buffer),N),dtype=props['dtype']),
			'cluster': np.zeros((len(iter_buffer),N),dtype=props['dtype'])}
	
	
	# Define simulation parameters for lattice with N sites
	configurations = {'sites':props['state_generate'](N=N),
					'cluster': np.nan*np.ones((N),dtype=props['dtype'])}


	# Setup plotting
	if props.get('plotting'):
		plot = plotter({job:[list(configurations.keys())]})



	# Algorithm Updates
	def update(iteration,updates):
		for k in data.keys():
			data[k][iteration] = updates[k]
		# exporter({'%s.%s'%(job,props.get('filetype','json')):data},directory)
		return

	def simulate(i,measure=True):
		

		if i >= stage_buffer[0][1]:
			stage_buffer.pop(0);

		alg[stage_buffer[0][0]%len(alg)][1](i,N,configurations,neighbours,props)


		if measure and (i+1) == measure_buffer[0]:

			update(iter_buffer.pop(0),configurations)

			getattr(logger,log)('MC Iteration (%s): %d, Cluster Size: %d'%(
					alg[stage_buffer[0][0]%len(alg)][0],i+1,
					np.count_nonzero(~np.isnan(configurations['cluster']))))

			if props.get('plotting'):
				plot.plot({job:configurations},{job:configurations},
					 	  {job:set_plot_montecarlo(keys=configurations.keys(),
					 	   i=measure_buffer.pop(0),**props.get('plotting'))})
		return

	



	# Perform equilibrium iterations
	getattr(logger,log)('Monte Carlo Equilibration')
	stage_buffer = stage_eqb_buffer
	for i in range(sum(Neqb)):
		simulate(i,measure=False)
	getattr(logger,log)('System Equilibrated: %s MC Steps'%(str(Neqb)))

	

	# Perform measurement iterations
	getattr(logger,log)('Monte Carlo Measurements')
	stage_buffer = stage_meas_buffer
	if props.get('plotting'):
		animation = animate.FuncAnimation(plot.figs[job], 
										func=simulate,
										fargs={'measure':True}, 
										frames=sum(Nmeas), 
										interval=300,repeat_delay=10000,blit=0)
		exporter({'%s.gif'%job:animation},directory,
					options={'writer':'imagemagick'})
					
	else:
		for i in range(sum(Nmeas)):
			simulate(i,measure=True)
	getattr(logger,log)('System Measured')

	return data







# Update Algorithms
def metropolis(iteration,N,configurations,neighbours,props):

	# Randomly alter random spin sites and accept spin alterations
	# if energetically favourable or probabilistically likely

	# Generate state and store previous state
	i = np.random.randint(N)
	state0 = configurations['sites'][i]

	configurations['cluster'][:] = np.nan
	
	# Update state
	state = props['state_generate'](state0)
	configurations['sites'][i] = state
	configurations['cluster'][i] = state

	# Calculate Change in Energy and decide to Accept/Reject Spin Flip
	nearest_states = configurations['sites'][neighbours[i]]
	difference = props['state_difference'](state,state0,nearest_states)
	if difference > 0:
		if props['transition_probability']['metropolis'][difference] < (
		   np.random.random()):
			configurations['sites'][i] = state0 
			configurations['cluster'][i] = state0           
	return


def wolff(iteration,N,configurations,neighbours,props):

	# Add to Cluster
	def cluster_add(i):
		cluster_stack[cluster_ind] = i
		configurations['sites'][i] = state 
		configurations['cluster'][i] = state  

	# Create Cluster Array and Choose Random Site
	configurations['cluster'][:] = np.nan
	cluster_stack = np.empty(N,dtype=int)
	i = np.random.randint(N)


	state0 = configurations['sites'][i]
	state = props['state_generate'](state0)

	# Perform cluster algorithm to find indices in cluster
	cluster_ind = 0
	cluster_add(i)
	cluster_ind = 1
	while cluster_ind:
		cluster_ind -= 1
		i = cluster_stack[cluster_ind]
		for j in neighbours[i]:
			if configurations['sites'][j] == state0 and (
			   props['transition_probability']['wolff'] > np.random.random()):
				cluster_add(j)
				cluster_ind += 1
	return

