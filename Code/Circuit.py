from qiskit.circuit.library import efficient_su2
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, amplitude_damping_error, phase_damping_error, thermal_relaxation_error
from scipy.optimize import minimize, OptimizeResult
from qiskit.circuit import Gate
from qiskit import transpile
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit_algorithms.optimizers import SPSA
from IsingModel import TransverseIsingModel

from scipy.sparse.linalg import eigsh

import numpy as np 
import matplotlib.pyplot as plt

from copy import deepcopy

class Circuit:

    def __init__(self, n_qubits:int, isingModel:TransverseIsingModel, transpile_gates=None, backend=None):
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
            isingModel:      TransverseIsingModel containing the Hamiltonian.
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

    def run_VQE(self, tol=1e-10, maxiter=10_000, theta0="RANDOM", method="COBYLA", backend=None) -> OptimizeResult:
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
            """
            Checks what basis gates the backend uses, then transpiles
            and sets the correct error probabilities using qubit 0.
            """

            # Extract native quantum gates (exclude measurement, reset, etc.)
            basis_gates = []
            for name in backend.operation_names:
                if isinstance(backend.target.operation_from_name(name), Gate):
                    basis_gates.append(name)

            # Transpiling without changing the number of qubits
            qc = transpile(self.circuit, basis_gates=basis_gates, optimization_level=1)

            # Using qubit 0 as a reference for coherence times
            props = backend.qubit_properties(0)
            T1 = props.t1
            T2 = min(props.t2, 2*T1)

            # Creates noise model
            noise_model = NoiseModel()

            for gate in basis_gates:
                duration = None

                for info in backend.target[gate].values():
                    if info is not None and info.duration is not None:
                        duration = info.duration
                        break

                # Skip virtual or uncalibrated gates
                if duration is None or duration == 0:
                    continue

                # Determine how many qubits the gate acts on
                n = backend.target.operation_from_name(gate).num_qubits

                # Construct thermal relaxation error
                error = thermal_relaxation_error(T1, T2, duration)

                for i in range(n - 1):
                    error = error.tensor(thermal_relaxation_error(T1, T2, duration))

                noise_model.add_all_qubit_quantum_error(error, [gate])

            self._doc.update({
                "Backend": backend.name,
                "Reference qubit": 0,
                "T1": T1,
                "T2": T2,
                "Ry gate duration (assumed)": 50e-9,
                "CX gate duration (assumed)": 300e-9})

        elif self.noise or self.noise_model is not None:

            if self.noise_model is None:
                self.noise_model = NoiseModel()

            for error, gates in self.noises.values():
                self.noise_model.add_all_qubit_quantum_error(error, gates)

        estimator = Estimator(options={"backend_options": {"noise_model": self.noise_model}})

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

    def transpile(self, basis_gates: list) -> None:
        self.circuit = transpile(self.circuit, basis_gates=basis_gates, optimization_level=1)
        
    def display_circuit(self) -> None:
        self.circuit.draw(output="mpl")
        plt.show()

    def add_noise_model(self, noise_model:NoiseModel) -> None:
        self.noise_model = noise_model 

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