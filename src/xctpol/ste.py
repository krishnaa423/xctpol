#region modules
import numpy as np 
from xctpol.utils import k2ry
import h5py 
from fp.inputs.input_main import Input 
from fp.io.pkl import load_obj
from mpi4py import MPI 
#endregion

#region variables
#endregion

#region functions
#endregion

#region classes
# Units are Ry for energy and bohr for distance. 
class Ste:
    def __init__(
        self,
        temp: float,
        input_filename: str = 'input.pkl',
        xctph_filename: str = 'xctph.h5',
    ):
        self.input_filename: str = input_filename
        self.xctph_filename: str = xctph_filename
        self.temp_K: float = temp 

        # Update.
        self.input: Input = load_obj(self.input_filename)
        self.input_dict: dict = self.input.input_dict
        self.max_error: float = self.input_dict['ste']['max_error']
        self.prev_lowest_energy: float = None 
        self.current_lowest_energy: float = None 
        self.iter: int = 0
        self.error: float = self.max_error + 1
        self.max_steps: int = self.input_dict['ste']['max_steps']
        self.beta: float = 1 / (self.temp_K * k2ry)

    def read_xctph(self):
        with h5py.File(self.xctph_filename, 'r') as r:
            self.Q_plus_q_map: np.ndarray = r['Q_plus_q_map'][:]
            self.Q_minus_q_map: np.ndarray = r['Q_minus_q_map'][:]
            self.ph_eigs: np.ndarray = r['frequencies'][:].T
            self.xct_eigs: np.ndarray = r['energies'][:].T
            self.xctph_eh: np.ndarray = r['xctph_eh'][:]
            self.xctph_e: np.ndarray = r['xctph_e'][:]
            self.xctph_h: np.ndarray = r['xctph_h'][:]
            self.nq = self.xct_eigs.shape[0]
            self.nS = self.xct_eigs.shape[1]
            self.nu = self.ph_eigs.shape[1]
        
        print('Finished reading xctph variables.\n\n', flush=True)

    def init_var(self):
        self.xctph: np.ndarray = np.zeros_like(self.xctph_eh)
        self.ste_eigs: np.ndarray = np.zeros(
            shape=(
                self.nq * self.nS,
            ),
            dtype='c16',        # Since we can have imaginary values for the eigenvalues.
        )
        factor = 1/np.sqrt(self.nq * self.nS)
        self.ste_evecs: np.ndarray = factor * np.ones(
            shape=(
                self.nq, 
                self.nS,
                self.nq * self.nS, 
            ),
            dtype='c16',
        )
        self.ste_evecs_minus_plus: np.ndarray = factor * np.ones(
            shape=(
                self.nq, 
                self.nq, 
                self.nq, 
                self.nS,
                self.nq * self.nS,
            ),
            dtype='c16',
        )
        self.xctph_minus: np.ndarray = np.zeros(
            shape=(
                self.nS,
                self.nS,
                self.nu,
                self.nq,
                self.nq,
            ),
            dtype='c16'
        )
        self.ste_se_tp: np.ndarray = np.zeros(shape=(self.nq * self.nS, self.nq * self.nS), dtype='c16')

        # Calculate some stuff. 
        self.ste_h0: np.ndarray = np.diag(self.xct_eigs.flatten()).reshape(*self.ste_se_tp.shape)
        self.ste_h: np.ndarray = np.zeros_like(self.ste_h0)
        self.eigs: np.ndarray = np.zeros(shape=(self.nq * self.nS), dtype='c16')
        self.evecs: np.ndarray = np.zeros_like(self.ste_h0)
        self.ste_occ_factor: np.ndarray = np.zeros(
            shape=(self.nS),
            dtype='f8'
        )

        self.ph_eigs_inv = np.zeros(shape=(self.nq, self.nq, self.nu), dtype='f8')
        for q1 in range(self.nq):
            for q2 in range(self.nq):
                for u1 in range(self.nu):
                    value = self.ph_eigs[self.Q_minus_q_map[q1, q2], u1]
                    self.ph_eigs_inv[q1, q2, u1] = 0.0 if value <= 0.0  else value 

        print('Done calculating ph_eigs_inv.\n\n', flush=True)

    def calc_ste_occ_factor(self):
        self.ste_occ_factor = 1/(np.exp(self.beta * self.ste_eigs.real) - 1)

        print('Done calculating bose factor.\n\n')

    def calc_xctph_minus(self):
        for q1 in range(self.nq):
            for q2 in range(self.nq):
                for q3 in range(self.nq):
                    self.xctph_minus[:, :, :, q1, q2, q3] = self.xctph[:, :, :, q1, self.Q_minus_q_map[q2, q3]]

        print('Finished xctph minus calc', flush=True)

    def calc_ste_evecs_minus_plus(self):
        for q1 in range(self.nq):
            for q2 in range(self.nq):
                for q3 in range(self.nq):
                    self.ste_evecs_minus_plus[q1, q2, q3, :, :] = \
                        self.ste_evecs[
                            self.Q_plus_q_map[self.Q_minus_q_map[q1, q2], q3],
                            :,
                            :
                        ]
        
        print('Done calculating ste_evecs_minus_plus. \n\n')

    def build_self_energy_tp(self):
        # a -> s4.
        # b -> s5.
        # c -> Q5.
        # d -> lambda.
        self.ste_se_tp: np.ndarray = -2.0 * np.einsum(
            'qQu,sSuQqQ,abucqQ,cbd,qQcad,d->qsQS',
            self.ph_eigs_inv,
            self.xctph_minus,
            self.xctph_minus,
            self.ste_evecs,
            self.ste_evecs_minus_plus.conj(),
            self.ste_occ_factor,
        ).reshape(*self.ste_se_tp.shape)

        print('Done calculating ste se tp.\n\n')

    def build_hamiltonian(self):
        self.ste_h = self.ste_h0 + self.ste_se_tp

    def diagonalize(self):
        self.eigs,self.evecs = np.linalg.eig(self.ste_h)

        self.ste_eigs = self.eigs.reshape(*self.ste_eigs.shape)
        self.ste_evecs = self.evecs.reshape(*self.ste_evecs.shape)

    def calc_init_guess(self):
        self.xctph = self.xctph_e 
        self.calc_xctph_minus()
        self.build_self_energy_tp()
        self.build_hamiltonian()
        self.diagonalize()
        self.calc_ste_evecs_minus_plus()
        # self.calc_ste_occ_factor()
        self.prev_lowest_energy = 0.0

    def step(self):
        self.build_self_energy_tp()
        self.build_hamiltonian()
        self.diagonalize()
        self.calc_ste_evecs_minus_plus()
        self.calc_ste_occ_factor()

    def get_error(self) -> float :
        self.current_lowest_energy = self.ste_eigs.real[0]
        self.error = np.abs(self.current_lowest_energy - self.prev_lowest_energy)

    def run(self):
        # Read.
        self.read_xctph()
        self.init_var()

        # Initial guess.
        self.calc_init_guess()

        # Iterate.
        self.iter = 0
        self.error = self.max_error + 1
        while self.iter < self.max_steps:
            self.step()

            # Get error. Iterate or quit accordingly.
            self.get_error()
            print(f'Iter: {iter}, error: {self.error}, lowest_energy: {self.ste_eigs.real[0]}')
            if self.error < self.max_error:
                break
            else:
                self.iter += 1

    def write(self):
        data = {
            'ste_eigs': self.ste_eigs,
            'ste_evecs': self.ste_evecs,
            'ste_se_tp': self.ste_se_tp,
        }

        with h5py.File('ste.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD) as w:
            for name, data in data.items():
                w.create_dataset(name=name, data=data)
#endregion