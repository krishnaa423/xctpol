# Exciton-polarons from first principles


Inputs:
- EPW electron-phonon coupling coefficients.
- BSE finite momentum calculations.

Outputs:
- Shifts and lifetimes to phonon and exciton-polaron energies.


The code uses _petsc4py_  and _slepc4py_ for MPI parallelization. GPU paralleliztion is supported, if PETSC and SLEPC is built with GPU support. 
