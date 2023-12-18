# MonteCarloUpdates
A class which does iterations of sweeps over Configurations (i.e different update algorithms), and Temperatures. Equilibrium, then Measurement sweeps are performed, where the number of sweeps is a multiple of Nspins and measurements of the configurations occurs every Nmeas_f, also a multiple of Nspins. 
  
# Model
A class that initializes the possible model dependent q values range , as well as observables to be measured based on the input of model name, Hamiltonian coupling constants, and observable names. The observables methods are all completely vectorized to allow for computing everything after the Monte Carlo simulation; they take arrays of samples of spins at different temperatures and compute either the straight values (in the case of energy, order etc.) or the averaged values (in the case of susceptibility, specific heat etc.). 
 
# Lattice
A class that defines the square hyperlattice neighbours for each site in d dimensions, using both the linear position in the 1d array and the d-dimensional position of each site in the hyperlattice. 

# data_functions, plot_functions
General data-processing and plotting modules, which create subplots based on keywords and has a structure of {figure keywork: {axes keyword: {data_curve keyword : data} } } to allow for multiple figures of multiple subplots of multiple curves. Data is saved as .npz files for both the site configurations and the observables, and plot figures are saved  as pdf’s, in folders based on the model name (it will create them). Depending on OS, urls in the 'data_dir' and ‘data_file’ strings in the ‘model_props’ dictionary in *MagnetismModel* may have to be modified. 

# misc_functions
Contains have some miscellaneous functions.

# dim_reduce_functions
Performs dimensional reduction algorithms.
PCA: Performs principal component analysis on dataset components.
t-SNE: Performs t-distributed stochastic neighbour embedding, with optional initial PCA processing. Initially performs binary search for variances in input data that correspond to a user-defined perplexity.
