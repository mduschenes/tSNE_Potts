# -*- coding: utf-8 -*-
"""
Created on Tue May  1 05:13:09 2018

@author: Matt
"""

import tensorflow as tf

def neural_network(opt_params,alg_params):
        
        def neuron_var(shape,name,sigma=0.1):
            initial = tf.truncated_normal(shape,stddev=sigma)
            return tf.Variable(initial,name=name)        
        
        
        # Define Numbers In + Out + Hidden Layers
        opt_params['n_layers'] = np.size(opt_params['n_neuron']) 
        
        # Define Type of Layers (fcc: Fully Connected, cnn: Convolutional)
        if not self.opt_params.get('layers') == 'fcc':
            self.opt_params['layers'] = ['fcc']*(opt_params['n_layers']-1)
        
        # Initialize Weights, Biases and Input/Output Placeholders
        x_ = tf.placeholder(tf.float32, [None,opt_params['n_neuron'][0] ])
        y_ = tf.placeholder(tf.float32, [None,opt_params['n_neuron'][-1]])
        
        W = [None]*(opt_params['n_layers']-1)
        b = [None]*(opt_params['n_layers']-1)
        T = [None]*(opt_params['n_layers'])
        
        # Define Input
        T[0] = x_
        
        # Create Neuron Parameters in Layers
        for i in range(opt_params['n_layers']-1):
            if opt_params.get('layers')[i] == 'fcc':
                Wshape = [opt_params['n_neuron'][i],
                          opt_params['n_neuron'][i+1]]
                bshape = [opt_params['n_neuron'][i+1]]
            
                W[i] = neuron_var(Wshape,'weights_reg_%d'%i,
                                  alg_params['sigma_var'])
                
                b[i] = neuron_var(bshape,'biases_%d'%i,
                                  alg_params['sigma_var'])
            
            # Calculate Activation function for ith layer and Output
            if i != opt_params['n_layers']:
                T[i+1] = opt_params['neuron_func']['layer'](
                                              tf.matmul(T[i],W[i]) + b[i])
            else:
                T[i+1] = opt_params['neuron_func']['output'](
                                              tf.matmul(T[i],W[i]) + b[i])

        # Define Ouput
        y_est = T[-1]
        
        return y_est,x_,y_

def pca(X,n_dims=None):    
    
    if n_dims is None:
        n_dims = np.shape(X)[1]
    
    Xc = X - np.sum((1/np.shape(X)[0])*X,0)
    
    L,P = np.linalg.eig(np.dot(Xc.T,Xc))
    
    return np.dot(Xc,P[:, 0:n_dims]), L/np.sum(L),P    

def tsne():
        
    def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = max(sum(P),10**(-14))
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


    def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """
    
        # Initialize some variables
        print("Computing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)
    
        # Loop over all datapoints
        for i in range(n):
    
    #        # Print progress
    #        if i % 500 == 0:
    #            print("Computing P-values for point %d of %d..." % (i, n))
    
            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            (H, thisP) = Hbeta(Di, beta[i])
    
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 50:
    
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.
    
                # Recompute the values
                (H, thisP) = Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1
    
            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    
        # Return final P-matrix
        print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P





def tsne(X=np.array([]),opt_params no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X,_,_ = pca(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, opt_params['perplexity'])
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    display(True,True,'Iterations for t-SNE')
    for i in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if i < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (i + 1) == max_iter:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (i + 1, C))

        # Stop lying about P-values
        if i == 100:
            P = P / 4.

    # Return solution
    return Y