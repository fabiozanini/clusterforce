# vim: fdm=indent
'''
author:     Fabio Zanini
date:       17/03/15
content:    Clustering functions
'''
# Modules
import numpy as np


# Functions
def energy_function(v, e1, e2):
    # NOTE: v comes in as a flattened array, but it's a 2D vector
    N = v.shape[0] // 2
    v = v.reshape((N, 2))

    # Coefficients
    l1 = 5e-2           # repulsion from consensus
    l2 = 1e-4 / N       # interactions
    l2_rep = 3.0        # -- " -- (repulsive part)
    l2_att = 1e-2       # -- " -- (elastic attraction)

    # Calculate the radius
    r = np.sqrt((v**2).sum(axis=-1))

    # Mutual distances between all points
    a = np.zeros((N, N, 2))
    a[:] = v
    a -= a.swapaxes(0, 1)
    d = np.zeros((N, N))
    d[:] = np.sqrt((a**2).sum(axis=-1))
    
    # Initial level
    e = 0
    
    # Infinity trap
    e += np.cosh(r).sum()

    # Consensus repulsor
    e += -l1 * (e1 * r).sum()

    # Pairwise interactions (constant repulsion + harmonic oscillator)
    e += l2 * (-l2_rep * d + l2_att * e2 * d**2).sum()

    return e


def energy_gradient_function(v, e1, e2):
    # NOTE: v comes in as a flattened array, but it's a 2D vector
    N = v.shape[0] // 2
    v = v.reshape((N, 2))

    # Coefficients
    l1 = 5e-2           # repulsion from consensus
    l2 = 1e-4 / N       # interactions
    l2_rep = 3.0        # -- " -- (repulsive part)
    l2_att = 1e-2       # -- " -- (elastic attraction)

    # Calculate the radius
    r = np.sqrt((v**2).sum(axis=-1))

    # Mutual distances between all points
    a = np.zeros((N, N, 2))
    a[:] = v
    a -= a.swapaxes(0, 1)
    d = np.zeros((N, N))
    d[:] = np.sqrt((a**2).sum(axis=-1))
    
    # Initial level
    J = np.zeros_like(v)

    for ix in xrange(2):
        for i in xrange(N):
            g = 0
    
            # Infinity trap
            #e += np.cosh(r).sum()
            g += np.sinh(r[i]) * (v[i, ix] + 1e-10) / (r[i] + 1e-10)
            
            # Consensus repulsor
            #e += -l1 * (e1 * r).sum()
            g -= l1 * e1[i] * (v[i, ix] + 1e-10) / (r[i] + 1e-10)

            # Pairwise interactions (constant repulsion + harmonic oscillator)
            #e += l2 * (-l2_rep * d + l2_att * e2 * d**2).sum()
            for j in xrange(N):
                g -= 2 * l2 * l2_rep * (v[i, ix] - v[j, ix]) / (d[i, j] + 1e-15)
                g += 4 * l2 * l2_att * e2[i, j] * (v[i, ix] - v[j, ix])

            J[i, ix] = g

    return J.ravel()


def energy_withgradient_function(v, e1, e2):
    # NOTE: v comes in as a flattened array, but it's a 2D vector
    N = v.shape[0] // 2
    v = v.reshape((N, 2))

    # Coefficients
    l1 = 5e-2           # repulsion from consensus
    l2 = 1e-4 / N       # interactions
    l2_rep = 3.0        # -- " -- (repulsive part)
    l2_att = 1e-2       # -- " -- (elastic attraction)

    # Calculate the radius
    r = np.sqrt((v**2).sum(axis=-1))

    # Mutual distances between all points
    a = np.zeros((N, N, 2))
    a[:] = v
    a -= a.swapaxes(0, 1)
    d = np.sqrt((a**2).sum(axis=-1))
    
    # ENERGY
    e = 0
    # Infinity trap
    e += np.cosh(r).sum()
    # Consensus repulsor
    e += -l1 * (e1 * r).sum()
    # Pairwise interactions (constant repulsion + harmonic oscillator)
    e += l2 * (-l2_rep * d + l2_att * e2 * d**2).sum()
    

    # GRADIENT
    J = np.zeros_like(v)
    for ix in xrange(2):
        # Infinity trap and consensus repulsor
        #e += np.cosh(r).sum()
        #e += -l1 * (e1 * r).sum()
        J[:, ix] = (np.sinh(r) - l1 * e1) * (v[:, ix] + 1e-10) / (r + 1e-10)

        ## Pairwise interactions (constant repulsion + harmonic oscillator)
        ##e += l2 * (-l2_rep * d + l2_att * e2 * d**2).sum()
        #for i in xrange(N):
        #    J[i, ix] += 2 * l2 * (-l2_rep * (v[i, ix] - v[:, ix]) / (d[i, :] + 1e-15) +
        #                          2 * l2_att * e2[i, :] * (v[i, ix] - v[:, ix])).sum()

        vd = np.tile(v[:, ix], (N, 1))
        vd -= vd.T
        J[:, ix] += 2 * l2 * (-l2_rep * vd.T / (d + 1e-15) + 2 * l2_att * e2 * vd.T).sum(axis=-1)


    return (e, J.ravel())


def get_distance_parameters(alim):
    from .utils.sequence import get_consensus 
    cons = get_consensus(alim)
    
    # Constant repulsor in the consensus/middle
    e1 = (alim != cons).mean(axis=1)

    # Constant repulsor and elastic attractor between the sequences
    # This fixes a reasonable distance between points because at short distances
    # the attraction is zero (flat energy)
    e2 = np.tile(alim, (alim.shape[0], 1, 1))
    e2 = 1.0 * (e2 == e2.swapaxes(0, 1)).mean(axis=2)

    return e1, e2


def cluster_force(alim, method='CG', plot=False):
    '''Cluster sequences with physical forces
    
    Paramters:
       alim (biopython alignment or numpy matrix of chars): alignment to analyze
       method (str): minimization method (see scipy.optimize.minimize)
       plot (bool): plot clustering
    '''
    import numpy as np
    from scipy.optimize import minimize

    from .utils.sequence import get_consensus 

    alim = np.asarray(alim, 'S1')
    N = alim.shape[0]
    L = alim.shape[-1]

    e1, e2 = get_distance_parameters(alim)

    # Minimize the energy
    v0 = np.random.rand(N, 2)

    if method in ['Powell', 'BFGS']:
        print method, 'not using Jacobian'
        res = minimize(energy_function, v0.ravel(), method=method, args=(e1, e2))
    elif method in ['BFGS-jac']:
        print 'BFGS, using Jacobian'
        res = minimize(energy_withgradient_function, v0.ravel(), method='BFGS', args=(e1, e2),
                       jac=True)
    elif method in ['CG']:
        print method, 'using Jacobian'
        res = minimize(energy_withgradient_function, v0.ravel(), method=method, args=(e1, e2),
                       jac=True)
    else:
        raise ValueError('Method for minimization not found!')

    v = res.x.reshape((N, 2))
    print 'Minimal value of the function:', res.fun

    if plot:
        from matplotlib import cm
        import matplotlib.pyplot as plt

        # Plot the force field and the scatter
        fig, ax = plt.subplots()

        cons = get_consensus(alim)
        dcon = (alim != cons).sum(axis=1)
        colors = cm.jet(1.0 * dcon / dcon.max())
        ax.scatter([0], [0], s=200, edgecolor='k', facecolor='none', lw=2, zorder=-1)
        ax.scatter(v[:, 0], v[:, 1], s=40, c=colors)
        ax.grid(True)
        ax.set_xlim(-1.04*np.abs(v[:, 0]).max(), 1.04*np.abs(v[:, 0]).max())
        ax.set_ylim(-1.04*np.abs(v[:, 1]).max(), 1.04*np.abs(v[:, 1]).max())
        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=dcon.max()))
        sm.set_array(dcon)
        cb = plt.colorbar(sm)
        cb.set_label('Hamming distance from consensus', rotation=270, labelpad=40)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()
        

    return v

