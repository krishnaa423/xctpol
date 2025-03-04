#region modules
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI 
import numpy as np 
from functools import wraps
import logging
import time 
import h5py 
from typing import List, Tuple, Dict, Sequence
from xctpol.utils import k2ry
#endregion

#region variables
comm: PETSc.Comm = PETSc.COMM_WORLD
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
    def __init__(self, shape: Sequence[int]):
        self.shape: np.ndarray = np.array(shape) 
        self.size: int = int(np.prod(self.shape).item())

    def get_local_range(self):
        local_size = self.size // comm.size  
        start = 0
        offset = 0
        if comm.rank < self.size % comm.size:
            local_size += 1
            start = local_size*comm.rank
            offset = local_size*(self.size % comm.size)
        else:
            start = offset + local_size*comm.rank
            
        end = start + local_size

        return start, end, local_size

    def get_grid_from_linear(self, lin_idx):
        grid_idx = np.zeros_like(self.shape)

        left_over = lin_idx 
        for dim in range(self.shape.ndim-1):
            grid_idx[dim] = left_over // np.prod(self.shape[dim+1:]).item()
            left_over = left_over % np.prod(self.shape[dim+1:]).item()
        grid_idx[0] = left_over 

        return grid_idx

class Sizes:
    def __init__(self, filename: str, num_evecs: int):
        with h5py.File(filename, 'r', driver='mpio', comm=MPI.COMM_WORLD)  as r:
            self.nQ: int = r['energies'].shape[1]
            self.ns: int = num_evecs
            self.nu: int = r['frequencies'].shape[0]

class XctEigs:
    def __init__(self, filename: str, num_evecs):
        self.filename: str = filename
        self.Q_plus_q_map: np.ndarray = None 
        self.Q_minus_q_map: np.ndarray = None 
        self.xct_eigs: PETSc.Vec = None 
        self.num_evecs: int = num_evecs

        self._read()

    def _read(self):
        with h5py.File(self.filename, 'r', driver='mpio', comm=MPI.COMM_WORLD) as r:
            self.Q_plus_q_map: np.ndarray = r['Q_plus_q_map'][:]
            self.Q_minus_q_map: np.ndarray = r['Q_minus_q_map'][:]
            xct_eigs_read: np.ndarray = r['energies'][:self.num_evecs, :].T
            nQ = xct_eigs_read.shape[0]
            ns = xct_eigs_read.shape[1]
            xct_eigs_read = xct_eigs_read.flatten()

            # Set the petsc vector.
            self.xct_eigs = PETSc.Vec().createMPI(nQ * ns, ns)
            self.xct_eigs.setValues(
                range(*self.xct_eigs.owner_range),
                xct_eigs_read[slice(*self.xct_eigs.owner_range)]
            )
            self.xct_eigs.assemble()

class PhEigs:
    def __init__(self, filename: str):
        self.filename: str = filename
        self.ph_eigs: PETSc.Vec = None 
        self._read()

    def _read(self):
        with h5py.File(self.filename, 'r', driver='mpio', comm=MPI.COMM_WORLD) as r:
            ph_eigs_read: np.ndarray = r['frequencies'][:].T
            nQ = ph_eigs_read.shape[0]
            nu = ph_eigs_read.shape[1]

            # Set the petsc vector.
            self.ph_eigs = PETSc.Vec().createMPI(nQ * nu, nu)
            self.ph_eigs.setValues(
                range(*self.ph_eigs.owner_range),
                ph_eigs_read[slice(*self.ph_eigs.owner_range)]
            )
            self.ph_eigs.assemble()

class XOcc:
    def __init__(self, sizes: Sizes, xct_idx):
        self.sizes: Sizes = sizes
        self.xct_idx: int = xct_idx 
        self.nx: PETSc.Vec = None 

        self._read()

    def _read(self):
        self.nx = PETSc.Vec().createMPI(self.sizes.nQ * self.sizes.ns, self.sizes.ns)
        self.nx.setValue(self.xct_idx, 1.0)
        self.nx.assemble()

class POcc:
    def __init__(self, sizes: Sizes, ph_idx):
        self.sizes: Sizes = sizes
        self.xct_idx: int = ph_idx 
        self.np: PETSc.Vec = None 

        self._read()

    def _read(self):
        self.np = PETSc.Vec().createMPI(self.sizes.nQ * self.sizes.nu, self.sizes.nu)
        self.np.setValue(self.xct_idx, 1.0)
        self.np.assemble()

class Xctph:
    def __init__(self, filename: str, sizes: Sizes, xcteigs: XctEigs):
        self.filename: str = filename
        self.sizes: Sizes = sizes
        self.xcteigs: XctEigs = xcteigs
        self.elec: PETSc.Mat = None 
        self.hole: PETSc.Mat = None 
        self.elhole: PETSc.Mat = None 
        self.buffer_p: PETSc.Mat = None 
        self.buffer: PETSc.Mat = None       # The one that holes the current attachment.

        self._read()

    def _read_one_array(self, source_name: str, dest_mat: PETSc.Mat):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 

        with h5py.File(self.filename, 'r', driver='mpio', comm=comm) as r:
            dist = Dist(shape=(nQ, nQ, nu))
            start, stop, local_size = dist.get_local_range()
            for idx in range(start, stop):
                q, Q, mu = dist.get_grid_from_linear()
                row_idx = q
                col_idx = self.xcteigs.Q_minus_q_map[q, Q]*nu + nQ 
                dest_mat.setValuesBlocked(
                    row_idx,
                    col_idx,
                    r[source_name][:ns, :ns, q, mu, Q]
                ) 
            dest_mat.assemble()

    def _read(self):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 
        self.elec = PETSc.Mat().createDense((nQ * ns, nQ * ns * nQ * nu), ns); self._read_one_array('xctph_e', self.elec)
        self.hole = PETSc.Mat().createDense((nQ * ns, nQ * ns * nQ * nu), ns); self._read_one_array('xctph_h', self.hole)
        self.elhole = PETSc.Mat().createDense((nQ * ns, nQ * ns * nQ * nu), ns); self._read_one_array('xctph_eh', self.elhole)
        self.buffer = PETSc.Mat().createDense((nQ * ns, nQ * ns * nQ * nu), ns)

    def set_hole(self):
        self.hole.copy(self.buffer)

    def set_elhole(self):
        self.elhole.copy(self.buffer)

    def rotG(self, xeigpairs):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 
        xeigpairs: XEigPairs = xeigpairs

        # Get the left and right rotation matrices.
        Aset = np.array([xeigpairs.xevecs], dtype='O').reshape(1, 1)
        Aset = np.repeat(Aset, nQ*nu, axis=0)
        rot_right: PETSc.Mat = PETSc.Mat().createNest(Aset)
        rot_right.assemble()
        rot_right.convert(PETSc.Mat.MPIDENSE)
        rot_right.assemble()
        rot_left: PETSc.Mat = xeigpairs.xevecs.copy()
        rot_left.hermitianTranspose()

        # Perform the rotation.
        self.buffer = rot_left @ self.buffer @ rot_right 

class SeTP:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        self.se_tp: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.se_tp.assemble()

    def calc(self, pheigs: PhEigs, nx: XOcc, xctph: Xctph):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 
        
        # Collect the PETSC inputs.
        pheigs_vec = pheigs.ph_eigs
        pheigs_inv_vec = pheigs_vec.duplicate(); pheigs_inv_vec.reciprocal(); pheigs_inv_vec.scale(-2.0); pheigs_inv_vec.assemble()
        xct_idx: int = nx.xct_idx
        xctph_mat = xctph.buffer

        # parallelize and calculate.
        #Calculate the right side.
        dist: Dist = Dist(shape=(nQ*nu))
        start, stop, local_size = dist.get_local_range()
        for vec_idx in range(start, stop):
            pheigs_inv_vec[vec_idx] *= xctph_mat[xct_idx, vec_idx*nQ*ns + xct_idx]
        pheigs_inv_vec.assemble()
        Amats = np.zeros((nQ*nu, 1), dtype='O')
        for mat_idx in range(nQ*nu):
            Amats[mat_idx, 0] = PETSc.createConstantDiagonal((nQ*ns, nQ*ns), pheigs_inv_vec.getValue(mat_idx))
        right_factor: PETSc.Mat = PETSc.Mat().createNest(Amats)
        right_factor.assemble()
        right_factor.convert(PETSc.Type.MPIDENSE)
        right_factor.assemble()

        # Calculate the total.
        self.se_tp = xctph_mat @ right_factor

class SeFM:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        self.se_fm: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.se_fm.assemble()

    def calc(self):
        pass 

class SeBB:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        self.se_bb: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.se_bb.assemble()

    def calc(self):
        pass 

class XHam:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        self.xham: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.xham.assemble()
        self.h0: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.xham.assemble()

        self.eps: SLEPc.EPS = SLEPc.EPS().create()
        self.eps.setOperators(self.xham)
        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        self.setUp()

    def set_mat(self, xct_eigs: XctEigs, se_tp: SeTP, se_fm: SeFM):
        self.h0.setDiagonal(xct_eigs.xct_eigs); self.h0.assemble()
        self.xham = self.h0 + se_tp.se_tp + se_fm.se_fm; self.xham.assemble()

    def diagonalize(self, xeigpairs, zero_out=False):
        xeigpairs: XEigPairs = xeigpairs
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        temp_evec: PETSc.Vec = PETSc.Vec().createMPI(nQ*ns, ns); temp_evec.assemble()
        
        # Zero out non diagonal blocks if requested.
        if zero_out:
            zeroes = np.zeros((nQ*ns*nQ*ns), dtype='c16')
            for row in range(nQ):
                for col in range(nQ):
                    if row==col: continue
                    self.xham.setValuesBlocked(row, col, zeroes)
            self.xham.assemble()

        # Solve.
        self.eps.solve()

        # Extract.
        for eig_idx in range(nQ*ns):
            eig = self.eps.getEigenpair(eig_idx, temp_evec)
            
            # Assemble the eigenvectors.
            xeigpairs.xevecs.setValues(range(*temp_evec.owner_range), eig_idx, temp_evec.array)
            xeigpairs.xevecs.assemble()

            # Assemble the eigenvalues.
            xeigpairs.xeigs.setValue(eig_idx, eig)
            xeigpairs.xeigs.assemble()

class XEigPairs:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        self.xeigs: PETSc.Vec = PETSc.Vec().createDense(nQ*ns, ns); self.xeigs.assemble()
        self.xevecs: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.xevecs.assemble()
        self.xevecs_orig: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.xevecs_orig.assemble()

    def set_normalized_ones(self):
        start, stop = self.xevecs.owner_range
        nrows = stop - start
        ncols = self.xevecs.size[1]
        ones = np.ones((nrows * ncols,), dtype='c16')
        self.xevecs.setValues(range(start, stop), range(ncols), ones)
        self.xevecs.assemble()

    def copy_eig_to_orig(self):
        self.xevecs.copy(self.xevecs_orig)
        self.xevecs.assemble()
        self.xevecs_orig.assemble()

    def calc_orig_basis(self):
        self.xevecs_orig = self.xevecs_orig @ self.xevecs
        self.xevecs_orig.assemble() 

    def write(self):
        with h5py.File('ste.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as w:
            ds_eigs = w.create_dataset('xeigs', shape=self.xeigs.size, dtype='c16')
            ds_eigs[slice(*self.xeigs.owner_range)] = self.xeigs.array

            ds_evecs = w.create_dataset('xevecs', shape=self.xevecs.size, dtype='c16')
            ds_evecs[slice(*self.xevecs.owner_range), :] = self.xevecs.getDenseArray()

class PHam:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        nu = self.sizes.nu
        self.pham: PETSc.Mat = PETSc.Mat().createDense((nQ*nu, nQ*nu), nu); self.pham.assemble()

    def set_mat(self):
        pass 

    def diagonalize(self):
        pass 

class PEigPairs:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        nu = self.sizes.nu
        self.peigs: PETSc.Vec = PETSc.Vec().createDense(nQ*nu, nu); self.peigs.assemble()
        self.pevecs: PETSc.Mat = PETSc.Mat().createDense((nQ*nu, nQ*nu), nu); self.pevecs.assemble()

    def write(self):
        pass 

class Ste:
    iter: int = -1 
    
    @logtime
    def __init__(
        self,
        temp: float = 300,
        input_filename: str = 'input.pkl',
        xctph_filename: str = 'xctph.h5',
        num_evecs: int = 10,
        xct_idx: int = 0,
        ph_idx: int = 0,
        delta: float = 1e-3,
        max_iter: int = 10,
        max_error: float = 1e-3,
        zero_out: bool  = False,    # Setting to true does xctpol calculation.
        add_fm: bool = False,       # Add Fanmigdal term.
    ):
        self.temp: float = temp
        self.input_filename: str =  input_filename
        self.xctph_filename: str = xctph_filename
        self.num_evecs: int = num_evecs
        self.xct_idx: int = xct_idx
        self.ph_idx: int = ph_idx
        self.delta: float = delta 
        self.max_iter: int = max_iter
        self.max_error: float = max_error 
        self.zero_out: bool = zero_out
        self.add_fm: bool = add_fm

        self.beta: float = 1 / (self.temp_K * k2ry)

    @logtime
    def init_vars(self):
        self.xcteigs: XctEigs = XctEigs(self.xctph_filename, self.num_evecs)
        self.pheigs: PhEigs = PhEigs(self.xctph_filename)
        self.sizes: Sizes = Sizes(self.xctph_filename, self.num_evecs)
        self.xocc: XOcc = XOcc(self.sizes, self.xct_idx)
        self.pocc: POcc = POcc(self.sizes, self.ph_idx)
        self.xctph: Xctph = Xctph(self.xctph_filename, sizes=self.sizes, xcteigs=self.xcteigs)
        
        self.se_tp: SeTP = SeTP(self.sizes)
        self.se_fm: SeFM = SeFM(self.sizes)
        self.xham: XHam = XHam(self.sizes)
        self.xeigpairs: XEigPairs = XEigPairs(self.sizes)
        
        self.se_bb: SeBB = SeBB(self.sizes)
        self.pham: PHam = PHam(self.sizes)
        self.peigpairs: PEigPairs = PEigPairs(self.sizes)

    @logtime
    def calc_init_guess(self):
        self.xeigpairs.set_normalized_ones()
        self.xctph.set_hole()
        self.xctph.rotG(self.xeigpairs)
        self.se_tp.calc(self.pheigs, self.xocc, self.xctph)
        self.se_fm.calc()
        self.xham.set_mat(self.xcteigs.xct_eigs, self.se_tp.se_tp, self.se_fm.se_fm)
        self.xeigpairs = self.xham.diagonalize(self.xeigpairs, zero_out=self.zero_out)
        self.xeigpairs.copy_eig_to_orig()

        self.xctph.set_elhole()

    @logtime
    def step(self):
        self.xctph.rotG(self.xeigpairs)
        self.se_tp.calc(self.pheigs, self.xocc, self.xctph)
        self.se_fm.calc()
        self.xham.set_mat(self.xcteigs.xct_eigs, self.se_tp.se_tp, self.se_fm.se_fm)
        self.xeigpairs = self.xham.diagonalize(zero_out=self.zero_out)
        self.xeigpairs.calc_orig_basis()

    @logtime
    def ph_correction(self):
        self.se_bb.calc()
        self.pham.set_mat()
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

        # self.ph_correction()
    
    @logtime
    def write(self):
        self.xeigpairs.write()
        # self.peigpairs.write() 

#endregion
