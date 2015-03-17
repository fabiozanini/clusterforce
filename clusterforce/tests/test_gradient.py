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

from clusterforce.clustering import *



# Functions
def load_test_data(region='PR', maxseqs=10):
    '''Load test data'''
    import numpy as np
    from Bio import AlignIO

    fn = '../data/HIV_PR_B.fasta'
    alim = np.array(AlignIO.read(fn, 'fasta')[:maxseqs])
    return alim


def numerical_gradient(v0, e_args, step=1e-5):
    '''Calculate gradient numerically and compare to analytic function'''
    e0 = energy_function(v0, *e_args)
    J = np.zeros(len(v0))
    for i in xrange(len(v0)):
        v = v0.copy()
        v[i] += step
        e = energy_function(v, *e_args)
        J[i] = (e - e0) / step
    return J



# Script
if __name__ == '__main__':

    if len(sys.argv) > 1:
        maxseqs = int(sys.argv[1])
    else:
        maxseqs = 10

    region = 'p17'
    plot = False
    reps = 10

    alim = load_test_data(region=region, maxseqs=maxseqs)
    N = alim.shape[0]
    L = alim.shape[-1]

    e1, e2 = get_distance_parameters(alim)
    v0 = np.random.rand(N, 2)

    e0 = energy_function(v0.ravel(), e1, e2)
    J0 = energy_gradient_function(v0.ravel(), e1, e2)
    Jn = numerical_gradient(v0.ravel(), (e1, e2))
    e00, J00 = energy_withgradient_function(v0.ravel(), e1, e2)

    print np.abs(Jn - J0).mean(), np.abs(Jn - J0).std()
    print np.abs(Jn - J00).mean(), np.abs(Jn - J00).std()
