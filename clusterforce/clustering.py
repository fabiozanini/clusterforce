# vim: fdm=indent
'''
author:     Fabio Zanini
date:       17/03/15
content:    Clustering functions
'''
# Modules


# Functions
def cluster_force(alim, method='Powell', plot=False):
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

    cons = get_consensus(alim)
    N = alim.shape[0]
    L = alim.shape[-1]

    # Constant repulsor in the consensus/middle
    l1 = 5e-2
    e1 = (alim != cons).mean(axis=1)
    e1_fun = lambda v, e1: -l1 * (e1 * np.sqrt((v**2).mean(axis=-1)))

    # Exponential repulsor from infinity
    ec_fun = lambda v: np.cosh(np.sqrt((v**2).mean(axis=-1)))

    # The two above create a ring of minimum with a radius that depends on the
    # divergence from consensus

    # Constant repulsor and elastic attractor between the sequences
    # This fixes a reasonable distance between points because at short distances
    # the attraction is zero (flat energy)
    l2 = 1e-4 / N
    l2_rep = 3.0
    l2_att = 1e-2
    e2 = np.tile(alim, (alim.shape[0], 1, 1))
    e2 = 1.0 * (e2 == e2.swapaxes(0, 1)).mean(axis=2)
    def e2_fun(v):
        a = np.tile(v, (v.shape[0], 1, 1))
        d = ((a - a.swapaxes(0, 1))**2).mean(axis=-1)
        e = l2 * (- l2_rep * d + l2_att * e2 * d**2)
        return e


    # v is the matrix with the two dimensional position vector of each sequence
    efun = lambda v: (ec_fun(v.reshape((v.shape[0] // 2, 2))).sum() + 
                      e1_fun(v.reshape((v.shape[0] // 2, 2)), e1).sum() +
                      e2_fun(v.reshape((v.shape[0] // 2, 2))).sum() +
                     0)

    # Minimize the energy
    v0 = np.random.rand(alim.shape[0], 2).ravel()
    res = minimize(efun, v0, method=method)
    v = res.x.reshape((v0.shape[0] // 2, 2))

    if plot:
        from matplotlib import cm
        import matplotlib.pyplot as plt

        # Plot the force field and the scatter
        fig, axs = plt.subplots(1, 2, figsize=(13, 6))

        ax = axs[0]
        xp = np.linspace(0, 10 * np.sqrt((v**2).sum(axis=-1).max()), 200)
        for dtmp in np.linspace(0, 1, 5):
            vp = np.vstack([xp, np.zeros_like(xp)]).T
            yp = e1_fun(vp, dtmp) + ec_fun(vp)
            ax.plot(xp, yp, color=cm.jet(dtmp), lw=2)

            yp = 1.0 + e1_fun(vp, dtmp)
            ax.plot(xp, yp, color=cm.jet(dtmp), lw=2, alpha=0.3, ls='--')

        yp = ec_fun(vp)
        ax.plot(xp, yp, color='k', lw=2, alpha=0.3, ls='--')
        ax.grid(True)


        ax = axs[1]
        dcon = (alim != cons).sum(axis=1)
        colors = cm.jet(1.0 * dcon / dcon.max())
        ax.scatter([0], [0], s=200, edgecolor='k', facecolor='none', lw=2, zorder=-1)
        ax.scatter(v[:, 0], v[:, 1], s=40, c=colors)
        ax.grid(True)
        ax.set_xlim(-1.04*np.abs(v[:, 0]).max(), 1.04*np.abs(v[:, 0]).max())
        ax.set_ylim(-1.04*np.abs(v[:, 1]).max(), 1.04*np.abs(v[:, 1]).max())
        sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=dcon.max()))
        sm.set_array(dcon)
        plt.colorbar(sm)
        

    return v


