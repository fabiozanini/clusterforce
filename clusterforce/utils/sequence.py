# vim: fdm=indent
'''
author:     Fabio Zanini
date:       17/03/15
content:    Sequence utilities.
'''
# Functions
def get_consensus(ali):
    '''Get alignment consensus'''
    import numpy as np
    from collections import Counter

    cons = np.array([Counter(col).most_common(1)[0][0] for col in ali.T], 'S1')
    return cons

