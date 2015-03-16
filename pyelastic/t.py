# vim: fdm=indent
'''
author:     Fabio Zanini
date:       16/03/15
content:    Test script for the elastic net clustering.
'''
# Modules
import os
import sys
import numpy as np
from Bio import AlignIO
from matplotlib import cm
import matplotlib.pyplot as plt



# Functions
def load_test_data(region='PR', maxseq=70):
    '''Load test data'''
    fn = '../data/HIV_PR_B.fasta'
    alim = np.array(AlignIO.read(fn, 'fasta')[:maxseq])
    return alim


def get_consensus(alim):
    '''Get alignment consensus'''
    from collections import Counter
    cons = np.array([Counter(col).most_common(1)[0][0] for col in alim.T], 'S1')
    return cons


def cluster_force(alim, method='Powell'):
    '''Cluster sequences with physical forces'''
    cons = get_consensus(alim)

    # Elastic attractor to the consensus/middle
    l1 = 1e0
    e1 = (alim == cons).sum(axis=1)
    e1_fun = lambda v: l1 * (e1 * (v**2).sum(axis=-1)).sum()

    # Exponential repulsor from infinity
    ec_fun = lambda v: np.cosh(np.sqrt((v**2).sum(axis=-1))).sum()

    # Symmetry breaker
    es_fun = lambda v: 1e-5 * v[:, 0].sum()

    # Repulsor between the sequences
    l2 = 1e-8 / alim.shape[-1]
    e2 = np.tile(alim, (alim.shape[0], 1, 1))
    e2 = 1.0 * (e2 != e2.swapaxes(0, 1)).sum(axis=2)
    def e2_fun(v):
        a = np.tile(v, (v.shape[0], 1, 1))
        d = ((a - a.swapaxes(0, 1))**2).sum(axis=-1)
        e = l2 * (e2 / (d + 1e-8)).sum()
        return e


    # v is the matrix with the two dimensional position vector of each sequence
    efun = lambda v: (ec_fun(v.reshape((v.shape[0] // 2, 2))) + 
                      es_fun(v.reshape((v.shape[0] // 2, 2))) +
                      e1_fun(v.reshape((v.shape[0] // 2, 2))) +
                      e2_fun(v.reshape((v.shape[0] // 2, 2))) +
                     0)

    # Minimize the energy
    from scipy.optimize import minimize
    v0 = np.random.rand(alim.shape[0], 2).ravel()
    res = minimize(efun, v0, method=method)
    v = res.x.reshape((v0.shape[0] // 2, 2))

    return v




# Script
if __name__ == '__main__':


    alim = load_test_data()
    cons = get_consensus(alim)
    dcon = (alim != cons).sum(axis=1)

    v = cluster_force(alim)


    # Plot the net
    fig, ax = plt.subplots()
    colors = cm.jet(1.0 * dcon / dcon.max())
    ax.scatter(v[:, 0], v[:, 1], s=40, c=colors)
    sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=dcon.max()))
    sm.set_array(dcon)
    plt.colorbar(sm)

    plt.ion()
    plt.show()
