# vim: fdm=indent
'''
author:     Fabio Zanini
date:       17/03/15
content:    Sequence utilities.
'''
# Functions
def get_consensus(alim):
    '''Get alignment consensus'''
    import numpy as np

    cons = np.zeros(alim.shape[-1], 'S1')
    alpha = np.array(['A', 'C', 'G', 'T', '-'])
    counts = np.zeros((len(alpha), len(cons)), int)
    for inuc, nuc in enumerate(alpha):
        counts[inuc] = (alim == nuc).sum(axis=0)

    return alpha[counts.argmax(axis=0)]
