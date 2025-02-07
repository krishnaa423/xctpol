#region modules
import numpy as np 
from xctpol.utils import k2ry
import h5py 
from fp.inputs.input_main import Input 
from fp.io.pkl import load_obj
from mpi4py import MPI 
import time
from functools import wraps
import logging 
from ase.units import Ry
#endregion

#region variables
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
#region backup
# Units are Ry for energy and bohr for distance. 
# class Ste:
#     def __init__(
#         self,
#         temp: float,
#         input_filename: str = 'input.pkl',
#         xctph_filename: str = 'xctph.h5',
#     ):
#         self.input_filename: str = input_filename
#         self.xctph_filename: str = xctph_filename
#         self.temp_K: float = temp 

#         # Update.
#         self.input: Input = load_obj(self.input_filename)
#         self.input_dict: dict = self.input.input_dict
#         self.max_error: float = self.input_dict['ste']['max_error']
#         self.prev_lowest_energy: float = None 
#         self.current_lowest_energy: float = None 
#         self.iter: int = 0
#         self.error: float = self.max_error + 1
#         self.max_steps: int = self.input_dict['ste']['max_steps']
#         self.beta: float = 1 / (self.temp_K * k2ry)

#         self.start_time: float = 0.0

#     def get_elapsed_time(self, start_time=None):
#         if start_time  is None:
#             start_time = self.start_time
#         return time.time() - start_time

#     def read_xctph(self):
#         start_time = time.time()

#         with h5py.File(self.xctph_filename, 'r') as r:
#             self.Q_plus_q_map: np.ndarray = r['Q_plus_q_map'][:]
#             self.Q_minus_q_map: np.ndarray = r['Q_minus_q_map'][:]
#             self.ph_eigs: np.ndarray = r['frequencies'][:].T
#             self.xct_eigs: np.ndarray = r['energies'][:].T
#             self.xctph_eh: np.ndarray = r['xctph_eh'][:]
#             self.xctph_e: np.ndarray = r['xctph_e'][:]
#             self.xctph_h: np.ndarray = r['xctph_h'][:]
#             self.nq = self.xct_eigs.shape[0]
#             self.nS = self.xct_eigs.shape[1]
#             self.nu = self.ph_eigs.shape[1]
        
#         print(f'Finished reading xctph variables. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def init_var(self):
#         self.xctph: np.ndarray = np.zeros_like(self.xctph_eh)
#         self.ste_eigs: np.ndarray = np.zeros(
#             shape=(
#                 self.nq * self.nS,
#             ),
#             dtype='c16',        # Since we can have imaginary values for the eigenvalues.
#         )
#         self.ste_eigs = self.xct_eigs.flatten()
#         factor = 1/np.sqrt(self.nq * self.nS)
#         self.ste_evecs: np.ndarray = factor * np.ones(
#             shape=(
#                 self.nq, 
#                 self.nS,
#                 self.nq * self.nS, 
#             ),
#             dtype='c16',
#         )
#         self.ste_evecs_minus_plus: np.ndarray = factor * np.ones(
#             shape=(
#                 self.nq, 
#                 self.nq, 
#                 self.nq, 
#                 self.nS,
#                 self.nq * self.nS,
#             ),
#             dtype='c16',
#         )
#         self.xctph_minus: np.ndarray = np.zeros(
#             shape=(
#                 self.nS,
#                 self.nS,
#                 self.nu,
#                 self.nq,
#                 self.nq,
#                 self.nq,
#             ),
#             dtype='c16'
#         )
#         self.ste_se_tp: np.ndarray = np.zeros(shape=(self.nq * self.nS, self.nq * self.nS), dtype='c16')

#         # Calculate some stuff. 
#         self.ste_h0: np.ndarray = np.diag(self.xct_eigs.flatten()).reshape(*self.ste_se_tp.shape)
#         self.ste_h: np.ndarray = np.zeros_like(self.ste_h0)
#         self.eigs: np.ndarray = np.zeros(shape=(self.nq * self.nS), dtype='c16')
#         self.evecs: np.ndarray = np.zeros_like(self.ste_h0)
#         self.ste_occ_factor: np.ndarray = np.zeros(
#             shape=(self.nq * self.nS),
#             dtype='f8'
#         )

#         start_time = time.time()

#         self.ph_eigs_inv = np.zeros(shape=(self.nq, self.nq, self.nu), dtype='f8')
#         for q1 in range(self.nq):
#             for q2 in range(self.nq):
#                 for u1 in range(self.nu):
#                     value = self.ph_eigs[self.Q_minus_q_map[q1, q2], u1]
#                     self.ph_eigs_inv[q1, q2, u1] = 0.0 if value <= 0.0  else value 

#         print(f'Done calculating ph_eigs_inv. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def calc_ste_occ_factor(self):
#         start_time = time.time()
#         self.ste_occ_factor = 1/(np.exp(self.beta * self.ste_eigs.real) - 1)

#         print(f'Done calculating bose factor. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def calc_xctph_minus(self):
#         start_time = time.time()

#         for q1 in range(self.nq):
#             for q2 in range(self.nq):
#                 for q3 in range(self.nq):
#                     self.xctph_minus[:, :, :, q1, q2, q3] = self.xctph[:, :, q1, :, self.Q_minus_q_map[q2, q3]]

#         print(f'Finished ste minus calc. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def calc_ste_evecs_minus_plus(self):
#         start_time = time.time()

#         for q1 in range(self.nq):
#             for q2 in range(self.nq):
#                 for q3 in range(self.nq):
#                     self.ste_evecs_minus_plus[q1, q2, q3, :, :] = \
#                         self.ste_evecs[
#                             self.Q_plus_q_map[self.Q_minus_q_map[q1, q2], q3],
#                             :,
#                             :
#                         ]
        
#         print(f'Done calculating ste_evecs_minus_plus. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def build_self_energy_tp(self):
#         # a -> s4.
#         # b -> s5.
#         # c -> Q5.
#         # d -> lambda.

#         start_time = time.time()

#         self.ste_se_tp: np.ndarray = -2.0 * np.einsum(
#             'qQu,sSuQqQ,abucqQ,cbd,qQcad,d->qsQS',
#             self.ph_eigs_inv,
#             self.xctph_minus,
#             self.xctph_minus,
#             self.ste_evecs,
#             self.ste_evecs_minus_plus.conj(),
#             self.ste_occ_factor,
#         ).reshape(*self.ste_se_tp.shape)

#         print(f'Done calculating ste se tp. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def build_hamiltonian(self):
#         start_time = time.time()

#         self.ste_h = self.ste_h0 + self.ste_se_tp

#         print(f'Done calculating ste h. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def diagonalize(self):
#         start_time = time.time()

#         self.eigs,self.evecs = np.linalg.eig(self.ste_h)

#         self.ste_eigs = self.eigs.reshape(*self.ste_eigs.shape)
#         self.ste_evecs = self.evecs.reshape(*self.ste_evecs.shape)

#         print(f'Done diagonalizing hamiltonian. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def calc_init_guess(self):
#         start_time = time.time()

#         self.xctph[:] = self.xctph_e[:] 
#         self.calc_xctph_minus()
#         self.calc_ste_occ_factor()
#         self.build_self_energy_tp()
#         self.build_hamiltonian()
#         self.diagonalize()
#         self.calc_ste_evecs_minus_plus()
#         # self.calc_ste_occ_factor()
#         self.prev_lowest_energy = 0.0

#         self.xctph[:] = self.xctph_eh[:]
#         self.calc_xctph_minus()

#         print(f'Done initial guess. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def step(self):
#         start_time = time.time()

#         self.build_self_energy_tp()
#         self.build_hamiltonian()
#         self.diagonalize()
#         self.calc_ste_evecs_minus_plus()
#         self.calc_ste_occ_factor()

#         print(f'Done step {self.iter}. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#     def get_error(self) -> float :
#         self.current_lowest_energy = self.ste_eigs.real[0]
#         self.error = np.abs(self.current_lowest_energy - self.prev_lowest_energy)
#         self.prev_lowest_energy = self.current_lowest_energy

#     def run(self):
#         # Read.
#         self.start_time = time.time()
#         self.read_xctph()
#         self.init_var()

#         # Initial guess.
#         self.calc_init_guess()

#         # Iterate.
#         self.iter = 0
#         self.error = self.max_error + 1
#         while self.iter < self.max_steps:
#             self.step()

#             # Get error. Iterate or quit accordingly.
#             self.get_error()
#             print(f'\n\nITER: {self.iter}, ERROR: {self.error}, LOWEST_ENERGY: {self.ste_eigs.real[0]}, ELAPSED SO FAR: {self.get_elapsed_time()}.\n\n', flush=True)
#             if self.error < self.max_error:
#                 break
#             else:
#                 self.iter += 1

#         print(f'Done iterations. Elapsed: {self.get_elapsed_time()}.', flush=True)

#     def write(self):
#         start_time = time.time()

#         data = {
#             'ste_eigs': self.ste_eigs,
#             'ste_evecs': self.evecs,
#             'ste_se_tp': self.ste_se_tp,
#         }

#         with h5py.File('ste.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as w:
#             for name, data in data.items():
#                 w.create_dataset(name=name, data=data)


#         print(f'Done writing. Elapsed: {self.get_elapsed_time(start_time)}.', flush=True)

#endregion

class Ste:
    iter: int = -1
    def __init__(
        self,
        temp: float,
        input_filename: str = 'input.pkl',
        xctph_filename: str = 'xctph.h5',
        num_evecs: int = 10,
    ):
        self.input_filename: str = input_filename
        self.xctph_filename: str = xctph_filename
        self.temp_K: float = temp 
        self.num_evecs: int = num_evecs

        # Update. 
        self.input: Input = load_obj(self.input_filename)
        self.input_dict: dict = self.input.input_dict
        self.max_error: float = self.input_dict['ste']['max_error']
        self.prev_lowest_energy: float = None
        self.current_lowest_energy: float = None
        self.error: float = self.max_error + 1
        self.max_steps: int = self.input_dict['ste']['max_steps']
        self.beta: float = 1 / (self.temp_K * k2ry)
        self.partial_result: np.ndarray = None 

    @logtime
    def read_xctph(self):
        with h5py.File(self.xctph_filename, 'r') as r:
            self.Q_plus_q_map: np.ndarray = r['Q_plus_q_map'][:]
            self.Q_minus_q_map: np.ndarray = r['Q_minus_q_map'][:]
            self.ph_eigs: np.ndarray = r['frequencies'][:].T
            self.xct_eigs: np.ndarray = r['energies'][:self.num_evecs, :].T
            self.xctph_eh: np.ndarray = r['xctph_eh'][:self.num_evecs, :self.num_evecs, ...]
            self.xctph_e: np.ndarray = r['xctph_e'][:self.num_evecs, :self.num_evecs, ...]
            self.xctph_h: np.ndarray = r['xctph_h'][:self.num_evecs, :self.num_evecs, ...]
            self.nq = self.xct_eigs.shape[0]
            self.nS = self.xct_eigs.shape[1]
            self.nu = self.ph_eigs.shape[1]

        print(f'xct_eigs shape: {self.xct_eigs.shape}', flush=True)
        print(f'xctph shape: {self.xctph_eh.shape}', flush=True)
        print(f'nq: {self.nq}', flush=True)
        print(f'nS: {self.nS}', flush=True)
        print(f'nu: {self.nu}', flush=True)

    @logtime
    def init_var(self):
        # G. Exciton-phonon coupling. 
        self.xctph: np.ndarray = np.zeros_like(self.xctph_eh)
        self.xctph_minus: np.ndarray = np.zeros(shape=(self.nS,self.nS,self.nq,self.nu,self.nq,self.nq),dtype='c16')
        
        # E. Eigenvalues.
        self.ste_eigs: np.ndarray = np.zeros(shape=(self.nq * self.nS), dtype='c16')
        self.ste_eigs = self.xct_eigs.flatten()

        # A. Eigenvectors.
        factor = 1/np.sqrt(self.nq * self.nS)
        self.ste_evecs: np.ndarray = factor * np.ones(shape=(self.nq, self.nS, self.nq * self.nS), dtype='c16')
        self.ste_evecs_plus_minus: np.ndarray = factor * np.ones(shape=(self.nq, self.nq, self.nq, self.nS, self.nq * self.nS), dtype='c16')

        # n. Occupation factor.
        self.ste_occ_factor: np.ndarray = np.zeros(shape=(self.nq * self.nS), dtype='f8')
        self.calc_ste_occ_factor()
        print(f'Initial argmin: {np.argmin(self.ste_eigs)}', flush=True)

        # Sigma and H. Self-energy and Hamiltonian. 
        self.ste_se_tp: np.ndarray = np.zeros(shape=(self.nq * self.nS, self.nq * self.nS), dtype='c16')
        self.ste_h0: np.ndarray = np.zeros_like(self.ste_se_tp)
        self.ste_h0 = np.diag(self.xct_eigs.flatten())
        self.ste_h: np.ndarray = np.zeros_like(self.ste_se_tp)
        self.evecs: np.ndarray = np.zeros_like(self.ste_h0)
        self.eigs: np.ndarray = np.zeros(shape=(self.nq * self.nS), dtype='c16')

        # w. Phonon frequencies.
        self.ph_eigs_inv = np.zeros(shape=(self.nq, self.nq, self.nu), dtype='f8')
        for q1 in range(self.nq):
            for q2 in range(self.nq):
                for u1 in range(self.nu):
                    value = self.ph_eigs[self.Q_minus_q_map[q1, q2], u1]
                    self.ph_eigs_inv[q1, q2, u1] = 0.0 if value <= 0.0  else value 

    @logtime
    def calc_ste_occ_factor(self):
        # For now just looking at the minimum state. 
        self.ste_occ_factor[:] = 0.0
        self.ste_occ_factor[np.argmin(self.ste_eigs.real)] = 1.0

    @logtime
    def calc_xctph_minus(self):
        for q1 in range(self.nq):
            for q2 in range(self.nq):
                for q3 in range(self.nq):
                    self.xctph_minus[:, :, q1, :, q2, q3] = self.xctph[:, :, q1, :, self.Q_minus_q_map[q2, q3]]

    @logtime
    def calc_ste_evecs_plus_minus(self):
        for q1 in range(self.nq):
            for q2 in range(self.nq):
                for q3 in range(self.nq):
                    self.ste_evecs_plus_minus[q1, q2, q3, :, :] = \
                        self.ste_evecs[
                            self.Q_minus_q_map[self.Q_plus_q_map[q1, q2], q3],
                            :,
                            :
                        ]

    @logtime
    def build_self_energy_tp(self):
        if self.partial_result is None:
            self.partial_result = -2.0 * np.einsum(
                'qQu,sSQuqQ,bacuqQ->qsQSuabc',
                self.ph_eigs_inv,
                self.xctph_minus,
                self.xctph_minus.conj(),
            )
        
        self.ste_se_tp: np.ndarray = np.einsum(
            'qsQSuabc,cqQbd,cad,d->qsQS',
            self.partial_result,
            self.ste_evecs_plus_minus.conj(),
            self.ste_evecs,
            self.ste_occ_factor,
        ).reshape(*self.ste_se_tp.shape)

    @logtime
    def build_hamiltonian(self):
        self.ste_h = self.ste_h0 + self.ste_se_tp

    @logtime
    def diagonalize(self):
        self.eigs,self.evecs = np.linalg.eig(self.ste_h)

        self.ste_eigs = self.eigs.reshape(*self.ste_eigs.shape)
        self.ste_evecs = self.evecs.reshape(*self.ste_evecs.shape)

    @logtime
    def calc_init_guess(self):
        self.xctph[:] = self.xctph_e[:] 
        self.calc_xctph_minus()
        self.calc_ste_occ_factor()
        self.build_self_energy_tp()
        self.build_hamiltonian()
        self.diagonalize()
        self.calc_ste_evecs_plus_minus()
        self.calc_ste_occ_factor()
        self.prev_lowest_energy = 0.0

        self.xctph[:] = self.xctph_eh[:]
        self.calc_xctph_minus()
    
    @logtime
    def step(self):
        self.build_self_energy_tp()
        self.build_hamiltonian()
        self.diagonalize()
        self.calc_ste_evecs_plus_minus()
        self.calc_ste_occ_factor()

    @logtime
    def get_error(self) -> float:
        self.current_lowest_energy = self.ste_eigs.real.min()
        self.error = np.abs(self.current_lowest_energy - self.prev_lowest_energy)
        self.prev_lowest_energy = self.current_lowest_energy

    @logtime
    def run(self):
        self.read_xctph()
        self.init_var()

        # Init guess.
        self.calc_init_guess()

        # Iterate. 
        Ste.iter = 0
        self.error = self.max_error + 1
        while Ste.iter < self.max_steps:
            start_time = time.time()
            self.step()

            # Get error. Iterate or quit accordingly.
            self.get_error()
            elapsed_time = time.time() - start_time
            print(f'\n\nITER: {Ste.iter}, ERROR: {self.error}, LOWEST_ENERGY: {self.ste_eigs.real.min()}, STEP_TIME: {elapsed_time}\n\n', flush=True)
            if self.error < self.max_error:
                break
            else:
                Ste.iter += 1

    @logtime
    def write(self):
        # Output is in Ry. 
        # Sort and write them. 
        sort_indices = np.argsort(self.ste_eigs.real)
        print('Done sorting.', flush=True)

        data = {
            'ste_eigs': self.ste_eigs[sort_indices],
            'ste_eigs_eV': self.ste_eigs[sort_indices] * Ry,
            'ste_evecs': self.evecs[:, sort_indices],
            'ste_se_tp': self.ste_se_tp,
        }
        print('Done collecting sorted data.', flush=True)

        with h5py.File('ste.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as w:
            for name, data in data.items():
                w.create_dataset(name=name, data=data)

#endregion