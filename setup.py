#region modules
from setuptools import setup, find_packages
# from Cython.Build import cythonize
#endregions

#region variables
#endregions

#region functions
setup(
    name='xctpol',
    version='1.0.0',
    description='Exciton-polaron calculations',
    author='Krishnaa Vadivel',
    author_email='krishnaa.vadivel@yale.edu',
    requires=[
        'numpy',
        'xctph',
        'cython',
        'fp_workflow',
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
#endregions

#region classes
#endregions
