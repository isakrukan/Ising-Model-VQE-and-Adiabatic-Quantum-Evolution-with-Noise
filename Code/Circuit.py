from qiskit.circuit.library import efficient_su2
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, amplitude_damping_error, phase_damping_error, thermal_relaxation_error
from scipy.optimize import minimize, OptimizeResult
from qiskit.quantum_info import SparsePauliOp, Statevector, DensityMatrix
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from qiskit.synthesis import LieTrotter
from qiskit.circuit import Gate
from qiskit import transpile
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_algorithms.optimizers import SPSA
from IsingModel import IsingModel

from scipy.sparse.linalg import eigsh

import numpy as np 
import matplotlib.pyplot as plt

from copy import deepcopy

import time

class Circuit:

    def __init__(self, n_qubits:int, isingModel:IsingModel, transpile_gates=None, backend=None):
        r"""
        Constructs a parameterized quantum circuit for VQE simulations
        of a spin-1/2 Ising model.

        The ansatz is constructed using efficient_su2 with one repetition,
        preparing the variational state

            |psi(theta)> = U(theta)|0>^{\otimes n}.

        The circuit can optionally be transpiled into a specified gate
        basis. An optional backend can be stored for hardware-inspired
        noise simulations.

        Parameters:
            n_qubits:        Number of qubits (spin sites).
            isingModel:      IsingModel containing the Hamiltonian.
            transpile_gates: Optional list of basis gates for transpilation.
                            If None, the original ansatz is retained.
            backend:         Optional Qiskit backend (e.g. FakeSherbrooke()).
                            Used as the default backend in run_VQE().
                            Does not execute on real quantum hardware.
        """

        if n_qubits != isingModel.N:
            raise ValueError("Number of qubits must match number of spin sites.")
        
        self.n_qubits = n_qubits
        self.circuit = efficient_su2(n_qubits, reps=1)

        self.backend = backend

        if transpile_gates is not None:
            self.circuit = transpile(self.circuit, basis_gates=transpile_gates, optimization_level=1)

        self.backend = backend
        self.isingModel = isingModel
        
        self.noise = False 
        self.noise_model = None
        # self.cx_depolarizing = False
        self.noises = {} # Dict of all noises

        self._doc = {}

    def run_VQE(self, tol=1e-3, maxiter=10_000, theta0="RANDOM", method="COBYLA", backend=None) -> OptimizeResult:
        r"""
        Description: 
            Performs Variational Quantum Eigensolver (VQE) optimization.

            Starting from |psi(theta)> = U(theta)|0>^{\otimes n}, evaluates

                E(theta) = <psi(theta)|H|psi(theta)>

            and repeatedly adjusts theta to minimize the energy.

            The quantum circuit is simulated classically using Qiskit Aer.
            Without noise, the expectation value is calculated numerically
            from the simulated quantum state, without measurement sampling.

            When noise is included, Aer automatically selects a simulation
            method. Depending on the method, noise is modeled either by
            propagating density matrices or sampling stochastic trajectories.
            No finite-shot measurement noise is added by default.
            For N repeated measurements, this would be proportional to 1/sqrt(N).

        Optimization:
            COBYLA (SciPy):
                Derivative-free optimizer that constructs local linear
                approximations to the objective function.
                Not guaranteed to find the global minimum.

            SPSA (Qiskit Algorithms):
                Stochastic gradient-based optimizer, particularly useful
                for noisy objective functions. Not yet implemented.

        Parameters:
            tol:        Optimization tolerance (COBYLA or trust-constr only).
            maxiter:    Maximum function evaluations for COBYLA and trust-constr,
                        maximum iterations for SPSA.
            theta0:     Initial variational parameters, or "RANDOM".
            method:     Optimization method ("COBYLA", "trust-constr" or "SPSA").
            backend:    Optional Qiskit backend (e.g. FakeSherbrooke())
                        providing hardware calibration data for noise simulation.
                        If None, uses the custom noise model or ideal simulation.

        Returns:
            scipy.optimize.OptimizeResult containing the optimized
            parameters, energy, and convergence information.
    """

        if theta0 == "RANDOM":
            rng = np.random.default_rng(42)
            theta0 = rng.uniform(-np.pi, np.pi, self.circuit.num_parameters)
        elif np.size(theta0) != self.circuit.num_parameters:
                raise(ValueError(f"Initial angle theta0 is not the correct size {np.size(theta0)}/{self.circuit.num_parameters}"))
        

        # Use provided backend, or the default backend from __init__
        if backend is None:
            backend = self.backend

        qc = self.circuit
        H = self.isingModel.Hamiltonian

        if backend is not None:
            # Gets noise from backend
            noise_model = NoiseModel.from_backend(backend)
            self.noise = True

            # Transpiles circuit to gates used by backend
            # (also makes sure the circuit does not use the same amount of qubits as backend)
            qc = transpile(self.circuit, basis_gates=noise_model.basis_gates, optimization_level=1)

        elif self.noise or self.noise_model is not None: # Noise is provided

            if self.noise_model is not None: # Complete noise model has been provided
                noise_model = deepcopy(self.noise_model) # To avoid adding noise on top of already existing noise
            else:
                noise_model = NoiseModel()  # Construct new noise model and adds noise provided

            for error, gates in self.noises.values():
                noise_model.add_all_qubit_quantum_error(error, gates)

        if self.noise:
            estimator = Estimator(options={"backend_options": {"noise_model": noise_model}})

        else:
            estimator = Estimator()

        def energy(theta):
            result = estimator.run([(qc, H, theta)]).result()
            return float(result[0].data.evs)

        # Classical optimization
        if method == "COBYLA" or method == "trust-constr":
            result = minimize(
                energy,
                theta0,
                method=method,
                options={"maxiter": maxiter},
                tol=tol
            )

        elif method == "SPSA":
            optimizer = SPSA(maxiter=maxiter)

            spsa_result = optimizer.minimize(fun=energy, x0=theta0)

            # Convert Qiskit's result into SciPy's OptimizeResult
            result = OptimizeResult(
                x=spsa_result.x,
                fun=spsa_result.fun,
                nfev=spsa_result.nfev,
                nit=spsa_result.nit,
                success=None,
                message="SPSA finished; convergence not verified."
            )

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        return result

    def run_adiabatic(
            self, 
            g_initial=2.0, 
            T=50.0, 
            theta_initial="GROUND STATE", 
            steps=2000, 
            backend=None, 
            n_shots=50,
            print_progress=True
            ) -> tuple[list[DensityMatrix], float]:
        
        r"""
        Simulates adiabatic evolution from H(g_initial) (e.g. strong horizontal field) to final Hamiltonian.
        Assumes given Hamiltonian is final Hamiltonian.
        Make sure theta_initial gives the ground state of initial Hamiltonian.
        Since the evolution is adiabatic, the final state should be the final Hamiltonian's ground state.
        
        Parameters:
            g_initial:      initial value of g (appearing in - g sum_i X_i). 
            T:              Total evolution time (in units with hbar=1).
            theta_initial:  Initial angle which prepares the initial state through U(theta) 
                            (defaults to ground state |+^{\otimes n}> of H_initial = -g sum_i X_i)
            steps:          Number of time-evolution steps.
            backend:        Optional Qiskit backend (e.g. FakeSherbrooke())
                            providing hardware calibration data for noise simulation.
                            If None, uses the custom noise model or ideal simulation.
            n_shots:        Number of simulated trajectories, used when noise is added and 
                            AerSimulator simulates trajectories
            print_progress: If True, prints progress along the way

        Returns:
            E:              Expected energy from final state.
            psi:            Final state after adiabatic evolution.
        """

        if isinstance(theta_initial, str):
            if theta_initial != "GROUND STATE":
                raise ValueError("Unknown initial state.")

            theta_initial = np.zeros(self.circuit.num_parameters)
            theta_initial[:self.n_qubits] = np.pi / 2

        else:
            theta_initial = np.asarray(theta_initial, dtype=float)

            if theta_initial.size != self.circuit.num_parameters:
                raise ValueError("Incorrect number of initial parameters.")
            
        qc = self.circuit.assign_parameters(theta_initial)
        H_final = self.isingModel.Hamiltonian

        # Initial Hamiltonian
        H_initial = SparsePauliOp.from_sparse_list([("X", [i], -g_initial) for i in range(self.n_qubits)], num_qubits=self.n_qubits)

        dt = T / steps

        # Adiabatic evolution
        start_time = time.perf_counter()
        counter = 0
        if print_progress:
            print(f"Running adiabatic evolution, dt = {dt}")
            print(f"Setting up adiabatic evolution, {counter/steps*100:.2f}% complete, elapsed time: 0 seconds", end="\r")
        for i in range(steps):
            s = (i + 0.5) / steps

            H = (1 - s) * H_initial + s * H_final

            """
            e^{-iHt} as first-order product approximation, using
                e^{t(A+B)} = e^{tA}e^{tB} + O(t^2)
            """
            evolution_first_order = PauliEvolutionGate(H, time=dt, synthesis=LieTrotter(reps=1))
            qc.append(evolution_first_order, range(self.n_qubits)) # range(self.n_qubits): act on all qubits

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            counter += 1

            if print_progress:
                print(f"Setting up adiabatic evolution, {counter/steps*100:.2f}% complete, elapsed time: {elapsed_time:.2f} seconds", end="\r")

        current_time = time.perf_counter()
        elapsed_time = current_time - start_time

        if print_progress:
            print(f"Setting up adiabatic evolution, {counter/steps*100:.2f}% complete, elapsed time: {elapsed_time:.2f} seconds")


        if backend is not None:
            # Gets noise from backend
            noise_model = NoiseModel.from_backend(backend)
            self.noise = True

        elif self.noise: # Noise is provided

            noise_model = NoiseModel()  # Construct new noise model and adds noise provided

            for error, gates in self.noises.values():
                noise_model.add_all_qubit_quantum_error(error, gates)

        # Final quantum state and energy

        if not self.noise:
            # rho = DensityMatrix.from_instruction(qc)
            state = Statevector.from_instruction(qc)
            E = float(np.real(state.expectation_value(H_final)))

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time

            if print_progress:
                print(f"Finished adiabatic evolution, elapsed time: {elapsed_time:.2f} seconds")

            return [state], E

        else:
            # AerSimulator allows for simulating various trajectories 
            simulator = AerSimulator(method="statevector", noise_model=noise_model)

            # Transpiles circuit to gates used by backend
            # (also makes sure the circuit does not use the same amount of qubits as backend)
            qc = transpile(qc, basis_gates=noise_model.basis_gates, optimization_level=1)

            qc.save_statevector(pershot=True) # pershot=True requests a separate statevector for each simulated trajectory

            result = simulator.run(qc, shots=n_shots).result() # Runs n_shots trajectories
            states = result.data(0)["statevector"] # Gets all the states in |psi> = 1/n_shots sum_k=1^n_shots |psi_k><psi_k|

            E = []

            if print_progress:
                print(f"Calculating expectation value for simulated trajectories.")

            for i, state in enumerate(states):
                E.append(state.expectation_value(H_final))
                
            E = np.mean(E)

            current_time = time.perf_counter()
            elapsed_time = current_time - start_time

            if print_progress:
                print(f"Finished adiabatic evolution, elapsed time: {elapsed_time:.2f} seconds")

            return states, E


    def transpile(self, basis_gates: list) -> None:
        self.circuit = transpile(self.circuit, basis_gates=basis_gates, optimization_level=1)
        
    def display_circuit(self) -> None:
        self.circuit.draw(output="mpl")
        plt.show()

    def add_noise_model(self, noise_model:NoiseModel) -> None:
        self.noise_model = noise_model 
        self.noise = True

    def set_cx_depolarizing_prop(self, p:float, gate="cx") -> None:
        self.check_probability(p)
        self.noise = True 
        self.noises[f"deploarizing ({gate})"] = [depolarizing_error(p, 2), [gate]]
        self._doc[f"Depolarizing probability ({gate}): "] = p

    def set_phase_flip_prop(self, p:float, gate="ry") -> None:
        self.check_probability(p)
        self.noise = True 
        phase_flip = [pauli_error([("Z", p), ("I", 1-p)]), [gate]]
        self.noises[f"phase flip ({gate})"] = phase_flip
        self._doc[f"Phase flip probability ({gate}): "] = p

    def set_bit_flip_prop(self, p:float, gate="ry") -> None:
        self.check_probability(p)
        self.noise = True 
        phase_flip = [pauli_error([("X", p), ("I", 1-p)]), [gate]]
        self.noises[f"bit flip ({gate})"] = phase_flip
        self._doc[f"Bit flip probability ({gate}): "] = p

    def set_amplitude_damping_prop(self, p: float, gate="ry") -> None:
        self.check_probability(p)

        self.noise = True

        error = amplitude_damping_error(p)
        self._doc[f"Amplitude damping probability ({gate}): "] = p

        # Since CX is a two-qubit gate
        if gate == "cx":
            error = error.tensor(error)

        self.noises[f"amplitude damping ({gate})"] = [error, [gate]]

    def set_phase_damping_prop(self, p: float, gate="ry") -> None:
        self.check_probability(p)

        error = phase_damping_error(p)
        self._doc[f"Phase damping probability ({gate}): "] = p

        # Since CX is a two-qubit gate
        if gate == "cx":
            error = error.tensor(error)

        self.noise = True
        self.noises[f"phase damping ({gate})"] = [error, [gate]]


    def set_thermal_relaxation_prop(self, T1: float, T2: float, gate_time: float, gate="ry") -> None:

        if T1 <= 0 or T2 <= 0 or gate_time < 0 or T2 > 2*T1:
            raise ValueError("Require T1,T2 > 0, gate_time >= 0 and T2 <= 2*T1.")

        error = thermal_relaxation_error(T1, T2, gate_time)

        # Since CX is a two-qubit gate
        if gate == "cx":
            error = error.tensor(error)

        T_phi = 1 / (1 / T2 - 1 / (2 * T1))
        self.noise = True
        self.noises[f"thermal relaxation ({gate})"] = [error, [gate]]
        self._doc[f"T1 ({gate})"] = T1
        self._doc[f"T2 ({gate})"] = T2
        self._doc[f"T_phi ({gate})"] = T_phi
        self._doc[f"Gate time ({gate})"] = gate_time

    def update_ising_parameters(self, J=None, h=None, g=None) -> None:
        """
        Updates the Ising Hamiltonian without reconstructing the quantum circuit.
        Parameters set to None remain unchanged.
        """
        model = self.isingModel

        if J is not None:
            model.nearest_neighbor = not callable(J)
            model.J = model._as_callable(J, "J", pair=True)

        if h is not None:
            model.h = model._as_callable(h, "h")

        if g is not None:
            model.g = model._as_callable(g, "g")

        # Reconstruct only the Hamiltonian
        if model.dimension == 1:
            model._1D_Hamiltonian()
        else:
            model._2D_Hamiltonian()

    def get_linalg_E0(self) -> float:
        H = self.isingModel.Hamiltonian.to_matrix(sparse=True)

        E0 = eigsh(
            H,
            k=1,
            which="SA",
            return_eigenvectors=False
        )

        return float(E0[0])

    @staticmethod
    def check_probability(p:float):
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1.")

    @property
    def doc(self):
        return dict(sorted(self._doc.items()))

    @doc.setter
    def doc(self, value):
        self._doc = value