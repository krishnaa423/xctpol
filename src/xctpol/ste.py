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
import sys 
#endregion

#region variables
comm: PETSc.Comm = PETSc.COMM_WORLD
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Suppress stderr for non-zero ranks. Not sure if it is always a good idea, but makes reading error easier with a lotta processes.
if comm.rank != 0:
    sys.stderr = open('/dev/null', 'w')  # Suppress stderr for non-zero ranks
#endregion

#region functions
def logtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if comm.rank == 0:
            logging.info(f"ITERATION: {Ste.iter}, {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def print_zero_rank(msg):
    if comm.rank == 0:
        print(msg, flush=True)

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
        for dim in range(self.shape.size-1):
            grid_idx[dim] = left_over // np.prod(self.shape[dim+1:]).item()
            left_over = left_over % np.prod(self.shape[dim+1:]).item()
        grid_idx[-1] = left_over 

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

        self._read_xeigs()

    @logtime
    def _read_xeigs(self):
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

        self._read_peigs()

    @logtime
    def _read_peigs(self):
        with h5py.File(self.filename, 'r', driver='mpio', comm=MPI.COMM_WORLD) as r:
            ph_eigs_read: np.ndarray = r['frequencies'][:, :].T
            nQ = ph_eigs_read.shape[0]
            nu = ph_eigs_read.shape[1]
            ph_eigs_read = ph_eigs_read.flatten()

            # Set the petsc vector.
            self.ph_eigs = PETSc.Vec().createMPI(nQ * nu, nu)
            self.ph_eigs.assemble()
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

        self._read_xocc()

    @logtime
    def _read_xocc(self):
        self.nx = PETSc.Vec().createMPI(self.sizes.nQ * self.sizes.ns, self.sizes.ns)
        self.nx.setValue(self.xct_idx, 1.0)
        self.nx.assemble()

class POcc:
    def __init__(self, sizes: Sizes, pheigs: PhEigs, beta: float):
        self.sizes: Sizes = sizes
        self.pheigs: PhEigs = pheigs
        self.beta: float = beta
        self.np: PETSc.Vec = None 

        self._read_pocc()

    @logtime
    def _read_pocc(self):
        # Calculate the bose factors. 
        self.np = self.pheigs.ph_eigs.copy(); self.np.assemble()
        self.np.scale(self.beta); self.np.assemble()
        self.np.exp(); self.np.assemble()
        self.np -= 1.0; self.np.assemble()
        self.np.reciprocal(); self.np.assemble()

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

        self._read_xctph()

    @logtime
    def _read_one_array_xctph(self, virtual_source_name: str, source_linear_name: str, dest_mat: PETSc.Mat):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 

        with h5py.File(self.filename, 'r', driver='mpio', comm=MPI.COMM_WORLD) as r:
            dist = Dist(shape=(nQ, nQ, nu))
            start, stop, local_size = dist.get_local_range()
            for idx in range(start, stop):
                q, Q, mu = dist.get_grid_from_linear(idx)
                row_idx = q
                q_minus_Q = self.xcteigs.Q_minus_q_map[q, Q]
                col_idx = q_minus_Q*nu*nQ + Q 
                data_shape = r[virtual_source_name].shape 
                data = r[source_linear_name][:, q, mu, q_minus_Q].reshape(data_shape[0], data_shape[1])
                dest_mat.setValuesBlocked(
                    row_idx,
                    col_idx,
                    data[:ns, :ns]
                ) 
            dest_mat.assemble()

    @logtime
    def _read_xctph(self):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 
        self.elec = PETSc.Mat().createDense((nQ * ns, nQ * ns * nQ * nu), ns); self._read_one_array_xctph('xctph_e', 'xctph_e_linear', self.elec)
        self.hole = PETSc.Mat().createDense((nQ * ns, nQ * ns * nQ * nu), ns); self._read_one_array_xctph('xctph_h', 'xctph_h_linear', self.hole)
        self.elhole = PETSc.Mat().createDense((nQ * ns, nQ * ns * nQ * nu), ns); self._read_one_array_xctph('xctph_eh', 'xctph_eh_linear', self.elhole)
        self.buffer = PETSc.Mat().createDense((nQ * ns, nQ * ns * nQ * nu), ns)

    @logtime
    def set_hole(self):
        self.hole.copy(self.buffer)

    @logtime
    def set_elhole(self):
        self.elhole.copy(self.buffer)
        
    @logtime
    def rotG(self, xeigpairs):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 
        xeigpairs: XEigPairs = xeigpairs

        # Create rot left and right. 
        rot_right = xeigpairs.xevecs
        rot_left = xeigpairs.xevecs.copy()
        rot_left.hermitianTranspose()
        rot_left.assemble()
        
        # Create submatrices, rotate them. 
        submats = []
        for mat_idx in range(nQ*nu):
            mat_rows = nQ*ns
            mat_cols = nQ*ns
            submat = PETSc.Mat().createDense((mat_rows, mat_cols), ns)
            local_mat = self.buffer.getDenseArray()[:, mat_idx*mat_cols:(mat_idx+1)*mat_cols]
            submat[range(*submat.owner_range), :] = local_mat
            submat.assemble()
            submat = rot_left @ submat @ rot_right
            submat.assemble()
            submats.append(submat)
        
        # Set them back in the buffer.
        self.buffer = PETSc.Mat().createNest([submats])
        self.buffer.convert(PETSc.Mat.Type.MPIDENSE)
        self.buffer.assemble()

    @logtime
    def write_and_read_phonon_fmt(self):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 
        # Write the array out to a file. 
        with h5py.File('xctph_xct_fmt.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as w:
            ds_evecs = w.create_dataset('xctph_xct_fmt', shape=self.buffer.size, dtype='c16')
            ds_evecs[slice(*self.buffer.owner_range), :] = self.buffer.getDenseArray()

        # Now read it. 
        self.buffer_p = PETSc.Mat().createDense((nQ * nu, nQ * ns * nQ * ns), (nu, 1))
        start, stop = self.buffer_p.owner_range
        dist = Dist(shape=(nQ, nu))
        with h5py.File('xctph_xct_fmt.h5', 'r', driver='mpio', comm=MPI.COMM_WORLD) as r:
            ds = r['xctph_xct_fmt']
            for row_idx in range(start, stop):
                data = ds[:, row_idx*nQ*ns : (row_idx+1)*nQ*ns].flatten()
                self.buffer_p[row_idx, :] = data 
            self.buffer_p.assemble()

class SeTP:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        self.se_tp: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.se_tp.assemble()

    @logtime
    def calctp(self, pheigs: PhEigs, nx: XOcc, xctph: Xctph):
        nQ = self.sizes.nQ
        ns = self.sizes.ns 
        nu = self.sizes.nu 
        
        # Collect the PETSC inputs.
        pheigs_vec = pheigs.ph_eigs
        pheigs_inv_vec = pheigs_vec.copy(); pheigs_inv_vec.reciprocal(); pheigs_inv_vec.scale(-2.0); pheigs_inv_vec.assemble()
        xct_idx: int = nx.xct_idx
        xctph_mat = xctph.buffer


        #Calculate the right side vec.
        dist: Dist = Dist(shape=(nQ*nu))
        start, stop, local_size = dist.get_local_range()
        print(f'rank: {comm.rank}, pheigs_inv size: {pheigs_inv_vec.size}, start: {start}, stop: {stop}', flush=True)
        for vec_idx in range(start, stop):
            # TODO: Need to debug this for parallel case (mpirun -n <procs> where procs >= 2). 
            pheigs_inv_vec[vec_idx] *=  xctph_mat[xct_idx, vec_idx*nQ*ns + xct_idx]
            pheigs_inv_vec.assemble()

        # Calculate se_tp submatrices by diagonal scaling. 
        self.se_tp = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.se_tp.assemble()
        for mat_idx in range(nQ*nu):
            # Get submatrix. 
            mat_rows = nQ*ns
            mat_cols = nQ*ns
            submat = PETSc.Mat().createDense((mat_rows, mat_cols), ns)
            local_mat = xctph_mat.getDenseArray()[:, mat_idx*mat_cols:(mat_idx+1)*mat_cols]
            submat[range(*submat.owner_range), :] = local_mat
            submat.assemble()

            # Get diagonal vec. 
            rvec = PETSc.Vec().createMPI(submat.size[1])
            start, stop = rvec.owner_range
            local_size = stop - start  
            fill_value = pheigs_inv_vec[mat_idx]
            rvec[range(start, stop)] = np.full((local_size,), fill_value, dtype='c16')
            rvec.assemble()

            # Diagonal scale. 
            submat.diagonalScale(R=rvec)
            submat.assemble()

            # Add submat. 
            self.se_tp = self.se_tp + submat 
            self.se_tp.assemble()

class SeFM:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        self.se_fm: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.se_fm.assemble()

    @logtime
    def calcfm(self, pheigs: PhEigs, pocc: POcc, xcteigs: XctEigs, xct_idx: int, xctph: Xctph, delta: float):
        # Make sure to read left and right matrices. 
        nQ = self.sizes.nQ
        nu = self.sizes.nu
        ns = self.sizes.ns
        xctph_x_left = xctph.buffer.copy(); xctph_x_left.assemble()
        xctph_x_right = xctph.buffer.copy(); xctph_x_right.hermitianTranspose(); xctph_x_right.assemble()

        # Right vector. 
        rvec = PETSc.Vec().createMPI(nQ*nu*nQ*ns)
        start, stop = rvec.owner_range
        dist = Dist(shape=(nQ, nu, nQ, ns))
        for idx in range(start, stop):
            q, u, Q, s = dist.get_grid_from_linear(idx)
            
            # calculate value. 
            np = pocc.np[q*u]
            nx = 1.0 if Q*s==xct_idx else 0.0
            Ealpha = xcteigs.xct_eigs[xct_idx]
            E = xcteigs.xct_eigs[Q*s]
            omega = pheigs.ph_eigs[q*u]
            value = (np - nx)/(Ealpha + delta*1j - E + omega) + (np + 1 + nx)/(Ealpha + delta*1j - E - omega)
            
            # set value. 
            rvec[idx] = value 
        rvec.assemble()

        # Multiply and assign se_bb. 
        xctph_x_left.diagonalScale(R=rvec)
        xctph_x_left.assemble()
        self.se_fm = xctph_x_left @ xctph_x_right
        self.se_fm.assemble()

class SeBB:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        nu = self.sizes.nu
        self.se_bb: PETSc.Mat = PETSc.Mat().createDense((nQ*nu, nQ*nu), nu); self.se_bb.assemble()

    @logtime
    def calcbb(self, pheigs: PhEigs, ph_idx: int, xcteigs: XctEigs, xct_idx: int, xctph: Xctph, delta: float):
        # Make sure to read xctph for phonon calculation format. 
        nQ = self.sizes.nQ
        nu = self.sizes.nu
        ns = self.sizes.ns
        xctph.write_and_read_phonon_fmt()
        xctph_p_left = xctph.buffer_p.copy(); xctph_p_left.conjugate(); xctph_p_left.assemble()
        xctph_p_right = xctph.buffer_p.copy(); xctph_p_right.transpose(); xctph_p_right.assemble()

        # Right vector. 
        rvec = PETSc.Vec().createMPI(nQ*ns*nQ*ns)
        start, stop = rvec.owner_range
        dist = Dist(shape=(nQ, ns, nQ, ns))
        for idx in range(start, stop):
            Q1, s1, Q2, s2 = dist.get_grid_from_linear(idx)
            
            # calculate value. 
            n1 = 1.0 if Q1*s1==xct_idx else 0.0
            n2 = 1.0 if Q2*s2==xct_idx else 0.0
            walpha = pheigs.ph_eigs[ph_idx]
            E1 = xcteigs.xct_eigs[Q1*s1]
            E2 = xcteigs.xct_eigs[Q2*s2]
            value = (n1 - n2)/(walpha + delta*1j + E1 - E2)
            
            # set value. 
            rvec[idx] = value 
        rvec.assemble()

        # Multiply and assign se_bb. 
        xctph_p_left.diagonalScale(R=rvec)
        xctph_p_left.assemble()
        self.se_bb = xctph_p_left @ xctph_p_right
        self.se_bb.assemble()

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
        self.eps.setUp()

    @logtime
    def set_matx(self, xct_eigs: XctEigs, se_tp: SeTP, se_fm: SeFM):
        self.h0.setDiagonal(xct_eigs.xct_eigs); self.h0.assemble()
        self.xham = self.h0 + se_tp.se_tp + se_fm.se_fm; self.xham.assemble()

    @logtime
    def diagonalizex(self, xeigpairs, zero_out=False):
        xeigpairs: XEigPairs = xeigpairs
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        temp_evec: PETSc.Vec = PETSc.Vec().createMPI(nQ*ns, ns); temp_evec.assemble()
        
        # Zero out non diagonal blocks if requested.
        if zero_out:
            zeroes = np.zeros((ns*ns), dtype='c16')
            for row in range(nQ):
                for col in range(nQ):
                    if row==col: continue
                    self.xham.setValuesBlocked(row, col, zeroes)
            self.xham.assemble()

        # Solve.
        self.eps.solve()

        # Extract.
        nconv = self.eps.getConverged()
        xeigpairs.set_as_zero_except_orig()
        
        print_zero_rank(f'nQ: {nQ}, ns: {ns}, nQ*ns: {nQ*ns}, eps nconv: {nconv}')
        
        for eig_idx in range(nconv):
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
        self.xeigs: PETSc.Vec = PETSc.Vec().createMPI(nQ*ns, ns); self.xeigs.assemble()
        self.xevecs: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.xevecs.assemble()
        self.xevecs_orig: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.xevecs_orig.assemble()

    @logtime
    def set_as_zero_except_orig(self):
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        self.xeigs: PETSc.Vec = PETSc.Vec().createMPI(nQ*ns, ns); self.xeigs.assemble()
        self.xevecs: PETSc.Mat = PETSc.Mat().createDense((nQ*ns, nQ*ns), ns); self.xevecs.assemble()

    @logtime
    def set_normalized_ones(self):
        nQ = self.sizes.nQ
        ns = self.sizes.ns
        nu = self.sizes.nu
        start, stop = self.xevecs.owner_range
        nrows = stop - start
        ncols = self.xevecs.size[1]
        ones = np.ones((nrows * ncols,), dtype='c16') * np.sqrt(1/nQ/ns)
        self.xevecs.setValues(range(start, stop), range(ncols), ones)
        self.xevecs.assemble()

    @logtime
    def copy_eig_to_orig(self):
        self.xevecs.copy(self.xevecs_orig)
        self.xevecs.assemble()
        self.xevecs_orig.assemble()

    @logtime
    def calc_orig_basis(self):
        self.xevecs_orig = self.xevecs_orig @ self.xevecs
        self.xevecs_orig.assemble() 

    @logtime
    def get_eig(self, xct_idx: int):
        return self.xeigs[xct_idx]

    @logtime
    def writex(self):
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
        self.p0: PETSc.Mat = PETSc.Mat().createDense((nQ*nu, nQ*nu), nu); self.p0.assemble()
        self.pham: PETSc.Mat = PETSc.Mat().createDense((nQ*nu, nQ*nu), nu); self.pham.assemble()

        self.eps: SLEPc.EPS = SLEPc.EPS().create()
        self.eps.setOperators(self.pham)
        self.eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        self.eps.setUp()

    @logtime
    def set_matp(self, pheigs: PhEigs, se_bb: SeBB):
        self.p0.setDiagonal(pheigs.ph_eigs); self.p0.assemble()
        self.pham = self.p0 + se_bb.se_bb; self.pham.assemble()

    @logtime
    def diagonalizep(self, peigpairs, zero_out: False):
        peigpairs: PEigPairs = peigpairs
        nQ = self.sizes.nQ
        nu = self.sizes.nu
        temp_evec: PETSc.Vec = PETSc.Vec().createMPI(nQ*nu, nu); temp_evec.assemble()
        
        # Zero out non diagonal blocks if requested.
        if zero_out:
            zeroes = np.zeros((nu*nu), dtype='c16')
            for row in range(nQ):
                for col in range(nQ):
                    if row==col: continue
                    self.pham.setValuesBlocked(row, col, zeroes)
            self.pham.assemble()

        # Solve.
        self.eps.solve()

        # Extract.
        nconv = self.eps.getConverged()
        peigpairs.set_as_zero()
        
        print_zero_rank(f'nQ: {nQ}, nu: {nu}, nQ*nu: {nQ*nu}, eps nconv: {nconv}')
        
        for eig_idx in range(nconv):
            eig = self.eps.getEigenpair(eig_idx, temp_evec)
            
            # Assemble the eigenvectors.
            peigpairs.pevecs.setValues(range(*temp_evec.owner_range), eig_idx, temp_evec.array)
            peigpairs.pevecs.assemble()

            # Assemble the eigenvalues.
            peigpairs.peigs.setValue(eig_idx, eig)
            peigpairs.peigs.assemble()

class PEigPairs:
    def __init__(self, sizes: Sizes):
        self.sizes = sizes
        nQ = self.sizes.nQ
        nu = self.sizes.nu
        self.peigs: PETSc.Vec = PETSc.Vec().createMPI(nQ*nu, nu); self.peigs.assemble()
        self.pevecs: PETSc.Mat = PETSc.Mat().createDense((nQ*nu, nQ*nu), nu); self.pevecs.assemble()

    @logtime
    def set_as_zero(self):
        nQ = self.sizes.nQ
        nu = self.sizes.nu
        self.peigs: PETSc.Vec = PETSc.Vec().createMPI(nQ*nu, nu); self.peigs.assemble()
        self.pevecs: PETSc.Mat = PETSc.Mat().createDense((nQ*nu, nQ*nu), nu); self.pevecs.assemble()
    
    @logtime
    def writep(self):
        with h5py.File('ste.h5', 'a', driver='mpio', comm=MPI.COMM_WORLD) as w:
            ds_eigs = w.create_dataset('peigs', shape=self.peigs.size, dtype='c16')
            ds_eigs[slice(*self.peigs.owner_range)] = self.peigs.array

            ds_evecs = w.create_dataset('pevecs', shape=self.pevecs.size, dtype='c16')
            ds_evecs[slice(*self.pevecs.owner_range), :] = self.pevecs.getDenseArray()

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

        self.beta: float = 1 / (self.temp * k2ry)

    @logtime
    def STE_init_vars(self):
        self.xcteigs: XctEigs = XctEigs(self.xctph_filename, self.num_evecs)
        self.pheigs: PhEigs = PhEigs(self.xctph_filename)
        self.sizes: Sizes = Sizes(self.xctph_filename, self.num_evecs)
        self.xocc: XOcc = XOcc(self.sizes, self.xct_idx)
        self.pocc: POcc = POcc(self.sizes, self.pheigs, self.beta)
        self.xctph: Xctph = Xctph(self.xctph_filename, sizes=self.sizes, xcteigs=self.xcteigs)
        
        self.se_tp: SeTP = SeTP(self.sizes)
        self.se_fm: SeFM = SeFM(self.sizes)
        self.xham: XHam = XHam(self.sizes)
        self.xeigpairs: XEigPairs = XEigPairs(self.sizes)
        
        self.se_bb: SeBB = SeBB(self.sizes)
        self.pham: PHam = PHam(self.sizes)
        self.peigpairs: PEigPairs = PEigPairs(self.sizes)

    @logtime
    def STE_calc_init_guess(self):
        self.xeigpairs.set_normalized_ones()
        self.xctph.set_hole()
        self.xctph.rotG(self.xeigpairs)
        self.se_tp.calctp(self.pheigs, self.xocc, self.xctph)
        if self.add_fm: self.se_fm.calcfm(self.pheigs, self.pocc, self.xcteigs, self.xct_idx, self.xctph, self.delta)
        self.xham.set_matx(self.xcteigs, self.se_tp, self.se_fm)
        self.xham.diagonalizex(self.xeigpairs, zero_out=self.zero_out)
        self.xeigpairs.copy_eig_to_orig()
        self.xctph.set_elhole()

    @logtime
    def STE_step(self):
        self.xctph.rotG(self.xeigpairs)
        self.se_tp.calctp(self.pheigs, self.xocc, self.xctph)
        if self.add_fm: self.se_fm.calcfm(self.pheigs, self.pocc, self.xcteigs, self.xct_idx, self.xctph, self.delta)
        self.xham.set_matx(self.xcteigs, self.se_tp, self.se_fm)
        self.xham.diagonalizex(self.xeigpairs, zero_out=self.zero_out)
        self.xeigpairs.calc_orig_basis()

    @logtime
    def STE_ph_correction(self):
        self.se_bb.calcbb(self.pheigs, self.ph_idx, self.xcteigs, self.xct_idx, self.xctph, self.delta)
        self.pham.set_matp(self.pheigs, self.se_bb)
        self.pham.diagonalizep(self.peigpairs, zero_out=self.zero_out)

    @logtime
    def STE_run(self):
        self.STE_init_vars()
        self.STE_calc_init_guess()

        # Iterate till convergence. 
        self.prev_min: int = None
        self.error: float = None 
        for Ste.iter in range(self.max_iter):
            self.STE_step()
            
            if self.prev_min == None:
                self.prev_min = self.xeigpairs.get_eig(self.xct_idx).real
                print_zero_rank(f'\n ITERATION: {Ste.iter}, ERROR: {self.error}\n')
                continue
            else:
                self.current_min: float = self.xeigpairs.get_eig(self.xct_idx).real
                self.error: float = np.abs(self.current_min - self.prev_min)
                print_zero_rank(f'\n ITERATION: {Ste.iter}, ERROR: {self.error}\n')
                
                if self.error < self.max_error:
                    break
                else:
                    self.prev_min = self.current_min

        self.STE_ph_correction()
    
    @logtime
    def STE_write(self):
        self.xeigpairs.writex()
        self.peigpairs.writep() 

#endregion
