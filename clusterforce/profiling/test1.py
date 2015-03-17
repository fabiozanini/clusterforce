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



# Script
if __name__ == '__main__':

    if len(sys.argv) > 1:
        maxseqs = int(sys.argv[1])
    else:
        maxseqs = 70

    region = 'p17'
    plot = False
    reps = 10

    alim = load_test_data(region=region, maxseqs=maxseqs)

    import time

    for method in ['Powell', 'BFGS', 'BFGS-jac', 'CG']:
        np.random.seed(30)
        t0 = time.time()
        for i in xrange(reps):
            v = cluster_force(alim, plot=plot, method=method)
        t1 = time.time()
        print reps, 'replicates run. Time per run: {:.2G} secs'.format((t1 - t0) / reps)
