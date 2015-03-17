# vim: fdm=indent
'''
author:     Fabio Zanini
date:       17/03/15
content:    Setup file for clusterforce.
'''
# Modules
from distutils.core import setup, Extension


# Description
setup (name = 'clusterforce',
       version = '0.1',
       author      = "Fabio Zanini",
       description = """Cluster sequences in an alignment in with force fields""",
       py_modules = ["clusterforce"],

       # metadata for upload to PyPI
       author_email = "fabio.zanini@tuebingen.mpg.de",
       license = "BSD/2-clause",
       keywords = "clustering sequences force",
       url = "https://github.com/iosonofabio/clusterforce",
       )

