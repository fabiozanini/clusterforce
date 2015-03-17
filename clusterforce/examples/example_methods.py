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

from clusterforce.clustering import cluster_force3
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
        maxseqs = 50

    region = 'p17'
    plot = True
    methods = ['Powell', 'CG']

    alim = load_test_data(region=region, maxseqs=maxseqs)
    cons = get_consensus(alim)
    dcon = (alim != cons).sum(axis=1)

    vs = []
    for method in methods:
        np.random.seed(30)
        v = cluster_force3(alim, plot=plot, method=method)
        plt.title(method)
        vs.append(v)


    plt.ion()
    plt.show()
