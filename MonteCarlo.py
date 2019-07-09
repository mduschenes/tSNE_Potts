# Import standard python modules
import numpy as np
from celluloid import Camera
import matplotlib.animation as animate
import logging

# Import data processing files
from data_process import exporter
from data_plot import plotter


def montecarlo(N,neighbours,props,job='job_0',directory='.',
					  plot=False,quiet=False):

	
	# Array of sample sites and updated clusters during simulation
	data = {'sites':np.zeros((props['Nmeas_freq'],N),
								dtype=props['dtype']),
			'cluster': np.zeros((props['Nmeas_freq'],N),
								dtype=props['dtype'])}
	
	
	# Define simulation parameters for lattice with N sites
	configurations = {'sites':props['state_generate'](N=N),
					'cluster': np.nan*np.ones((N),dtype=props['dtype'])}


	# Setup logging and plotting
	logger = logging.getLogger(__name__)
	if quiet:
		log = 'info'
	else:
		log = 'warning'
	
	if plot:
		plot_obj = plotter({job:[list(configurations.keys())]})



	# Perform Monte Carlo simulation, based on algorithm
	alg = globals().get(props['algorithm'],globals()['metropolis_wolff'])
	
	if props['algorithm'] == 'metropolis_wolff':
		Neqb  = int(props['Nmeas_ratio']*props['Neqb']*N) + 
				int((1-props['Nmeas_ratio'])*props['Neqb'])
		Nmeas  = int(props['Nmeas']*N)
		Nmeas_freq = 
	else:

	def update(i,updates,data,directory):
		for k in data.keys():
			data[k][i] = updates[k]
		exporter({'%s.json'%job:data},directory)
		return

	def simulate(i,iterations,measure=True):
		alg(i/iterations/props['Nmeas_ratio'],N,
			configurations,neighbours,props)
		
		if measure and (True or (i+1)%(props['Nmeas_freq']*iterations) == 0):
			getattr(logging,log)('MC Iteration: %d, Cluster Size: %d'%(i+1,
					np.count_nonzero(~np.isnan(configurations['cluster']))))

			update(configurations,i//(props['Nmeas_freq']*iterations))

			if plot:
				plot_obj.plot({job:configurations},{job:configurations},
							    {job:plot(configurations.keys(),i)})
		return

	

	# Perform initial equilibrium iterations
	for i in range():
		simulate(i,int(props['Neqb']*N),measure=False)



	# Perform measurement iterations
	if plot:
		# Writer = animate.writers['imagemagick']
		# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
		animation = animate.FuncAnimation(plot_obj.figs[job], 
										func=simulate,
										fargs=(int(props['Nmeas']*N),True), 
										frames=int(props['Nmeas']*N), 
										interval=300,repeat_delay=10000,blit=0)
		exporter({'%s.gif'%job:animation},directory,
					options=dict(writer='imagemagick'))
					
	else:
		for i in range(int(props['Nmeas']*N)):
			simulate(i,int(props['Nmeas']*N),measure=True)


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

def metropolis_wolff(iteration,N,configurations,neighbours,props):      

	if iteration < 1:
		metropolis(iteration,N,configurations,neighbours,props)
	else:
		wolff(iteration,N,configurations,neighbours,props)
	return

