# Import standard python modules
import numpy as np
from celluloid import Camera
import matplotlib.animation as animate
import logging

# Import data processing files
from data_process import exporter
from data_plot import plotter
from plot_properties import set_plot_montecarlo


def montecarlo(N,d,neighbours,props,job=0,directory='.'):

	# Setup logging
	logger = logging.getLogger(__name__)
	quiet = props.get('quiet',False)
	if quiet or 1:
		log = 'info'
	else:
		log = 'warning'
	
	if props.get('plotting'):
		plotting = props.get('plotting')
	else:
		plotting = False


	def Niter(Ni,N,Nratios,alg):
		Nratios = np.atleast_1d(Nratios)
		Nalg = len(alg)
		assert 1>=sum(Nratios)
		def Nfunc(i,r):
			if r==0:
				return 0
			j = alg[i%len(alg)]
			if j=='metropolis':
				return max(1,int(r*Ni*N))
			elif j=='wolff':
				return max(1,int(r*Ni))

		Niters = [Nfunc(i,r) for i,r in enumerate(Nratios)]
		Niters.append(Nfunc(len(Niters),1-sum(Nratios)))
		return Niters

	def Nperiod(Nf,N,Niters,alg):
		Niters = np.atleast_1d(Niters)
		def Nfunc(i,n):
			j = alg[i%len(alg)]
			if j=='metropolis':
				return max(1,int(1//Nf) if plotting else int(N//Nf))
			elif j=='wolff':
				return max(1,int(1//Nf))

		Nperiods = [Nfunc(i,n) for i,n in enumerate(Niters)]
		return Nperiods


	if props['algorithm'] in ['metropolis','wolff','metropolis_wolff']:
		alg_names = ['metropolis','wolff']
		alg = [(a,globals().get(a)) for a in alg_names]
	else:
		alg_names = [props['algorithm']]
		alg = [(a,globals().get(a)) for a in [props['algorithm']]]

	Nalg = len(alg)
	Neqb  = Niter(props['Neqb'],N,props['Nratio'],alg_names)
	Nmeas  = Niter(props['Nmeas'],N,props['Nratio'],alg_names)
	Nperiods = Nperiod(props['Nfreq'],N,Nmeas,alg_names)
	# Nmeas_total = [Nm//Nf for Nm,Nf in zip(Nmeas,Nperiods)]
	
	measure_buffer = [(j+1)*Nperiods[i]+sum(Nmeas[:i]) for i in range(len(Nmeas)) 
										for j in range(Nmeas[i]//Nperiods[i])]
	iter_buffer = list(range(len(measure_buffer)))
	stage_eqb_buffer = [(i-1,sum(Neqb[:i])) for i in range(1,len(Neqb)+1)]
	stage_meas_buffer = [(i-1,sum(Nmeas[:i])) for i in range(1,len(Nmeas)+1)]
	
	# getattr(logger,log)(['measure_buffer',Nmeas,sum(Neqb),sum(Nmeas),measure_buffer])
	# getattr(logger,log)(['stage_meas_buff',stage_meas_buffer])
	# getattr(logger,log)(['stage_eqb_buff',stage_eqb_buffer])
	getattr(logger,log)('''Monte Carlo: %d Eqb MC steps, %d Meas MC Steps, %s Meas sweeps, every %s Steps'''%(
							sum(Neqb),sum(Nmeas),str(Nmeas),str(Nperiods)))

		# Array of sample sites and updated clusters during simulation
	data = {'sites':np.zeros((len(iter_buffer),N),dtype=props['dtype']),
			'cluster': np.zeros((len(iter_buffer),N),dtype=props['dtype'])}
	

	
	# Define simulation parameters for lattice with N sites
	configurations = {'sites':props['state_generate'](N=N),
					'cluster': np.nan*np.ones((N),dtype=props['dtype'])}

	# Setup plotting
	if plotting:
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

		alg[stage_buffer[0][0]%Nalg][1](i,N,configurations,neighbours,props)


		if measure and (i+1) == measure_buffer[0]:
			m = measure_buffer.pop(0)
			update(iter_buffer.pop(0),configurations)

			if not quiet:
				getattr(logger,log)('MC Iteration (%s): %d, Cluster Size: %d'%(
					alg[stage_buffer[0][0]%Nalg][0],i+1,
					np.count_nonzero(~np.isnan(configurations['cluster']))))

			if plotting:
				plot.plot({job:configurations},{job:configurations},
					 	  {job:set_plot_montecarlo(keys=configurations.keys(),
											 	   i=m,
											 	   **plotting)})
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
	if plotting:
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

