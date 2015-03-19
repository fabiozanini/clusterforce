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

from clusterforce.clustering import cluster_force, position_sequence
from clusterforce.utils.sequence import get_consensus



# Functions
def load_test_data(region='PR', maxseqs=10):
    '''Load test data'''
    fn = '../data/HIV_PR_B.fasta'
    alim = np.array(AlignIO.read(fn, 'fasta')[:maxseqs])
    return alim



# Script
if __name__ == '__main__':

    if len(sys.argv) > 1:
        maxseqs = int(sys.argv[1])
    else:
        maxseqs = 30

    region = 'p17'
    plot = True

    # Separate last sequence to plot it additionally
    alim = load_test_data(region=region, maxseqs=maxseqs+1)
    seq = alim[-1]
    alim = alim[:-1]

    cons = get_consensus(alim)
    dcon = (alim != cons).sum(axis=1)

    np.random.seed(30)

    # Cluster sequences
    res = cluster_force(alim, plot=plot, method='BFGS-jac')
    v = res['x']

    # Add another one and see where it ends up
    u = position_sequence(seq, v, alim, e1e2=res['e1e2'])
    if plot:
        plt.scatter(u[0], u[1], marker='s', s=50,
                    edgecolor='k', facecolor='none', lw=3)
        plt.text(u[0], u[1], 'new sequence', fontsize=14,
                 va='bottom')


    plt.ion()
    plt.show()
