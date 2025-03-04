#region modules
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI 
import numpy as np 
from functools import wraps
import logging
import time 
#endregion

#region variables
#endregion

#region functions
def logtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"ITERATION: {Ste.iter}, {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

#endregion

#region classes
class Dist:
    def __init__(self):
        pass 

class Sizes:
    def __init__(self):
        pass

class XctEigs:
    def __init__(self):
        pass 

class PhEigs:
    def __init__(self):
        pass

class Xctph:
    def __init__(self):
        pass

class XOcc:
    pass 

class POcc:
    pass

class SeTP:
    pass 

class SeFM:
    pass

class SeBB:
    pass

class XHam:
    pass

class XEigPairs:
    def __init__(self):
        pass

class PHam:
    pass 

class PEigPairs:
    def __init__(self):
        pass

class Ste:
    iter: int = -1 
    
    @logtime
    def __init__(
        self,
        temp: float,
        input_filename: str = 'input.pkl',
        xctph_filename: str = 'xctph.h5',
        num_evecs: int = 10,
        xct_idx: int = 0,
        mu_idx: int = 0,
        delta: float = 1e-3,
        max_iter: int = 10,
        max_error: float = 1e-3,
        zero_out: bool  = False,    # Setting to true does xctpol calculation.
    ):
        self.temp: float = temp
        self.input_filename: str =  input_filename
        self.xctph_filename: str = xctph_filename
        self.num_evecs: int = num_evecs
        self.xct_idx: int = xct_idx
        self.mu_idx: int = mu_idx
        self.delta: float = delta 
        self.max_iter: int = max_iter
        self.max_error: float = max_error 
        self.zero_out: bool = zero_out

    @logtime
    def init_vars(self):
        self.xctph: XctEigs = XctEigs()
        self.pheigs: PhEigs = PhEigs()
        self.sizes: Sizes = Sizes()
        self.xocc: XOcc = XOcc()
        self.xocc: POcc = POcc()
        self.xctph: Xctph = Xctph()
        
        self.se_tp: SeTP = SeTP()
        self.se_fm: SeFM = SeFM()
        self.xham: XHam = XHam()
        self.xeigpairs: XEigPairs = XEigPairs()
        
        self.se_bb: SeBB = SeBB()
        self.pham: PHam = PHam()
        self.peigpairs: PEigPairs = PEigPairs()

    @logtime
    def calc_init_guess(self):
        self.xctph.set_hole()
        self.xctph.rotG()
        self.se_tp.calc()
        self.se_fm.calc()
        self.xham.set_mats()
        self.xeigpairs = self.xham.diagonalize()
        self.xeigpairs.calc_orig_basis()

        self.xctph.set_elhole()

    @logtime
    def step(self):
        self.xctph.rotG()
        self.se_tp.calc()
        self.se_fm.calc()
        self.xham.set_mats()
        self.xeigpairs = self.xham.diagonalize()
        self.xeigpairs.calc_orig_basis()

    @logtime
    def ph_correction(self):
        self.se_bb.calc()
        self.pham.set_mats()
        self.peigpairs = self.pham.diagonalize()

    @logtime
    def run(self):
        self.prev_min: int = None

        for Ste.iter in range(self.max_iter):
            self.step()
            
            if self.prev_min == None:
                self.prev_min = self.xeigpairs.get_eig(self.xct_idx).real
                continue
            else:
                 self.current_min: float = self.xeigpairs.get_eig(self.xct_idx).real
                 self.error: float = np.abs(self.current_min - self.prev_min)
                 
                 if self.error < self.max_error:
                    break
                 else:
                    self.prev_min = self.current_min

        self.ph_correction()
    
    @logtime
    def write(self):
        self.xeigpairs.write()
        self.peigpairs.write() 

#endregion
