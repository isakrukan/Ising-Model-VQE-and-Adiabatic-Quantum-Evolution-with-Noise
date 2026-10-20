
from Circuit import *
from IsingModel import *
from parameters import *

import numpy as np

from qiskit.quantum_info import Statevector, state_fidelity


def compare_AQC_vs_VQE(D=2, n_qubits=4, g=0.01, steps=100, T=50.0, n_shots=200, backend=None):
    r"""
    Compares VQE and digitized adiabatic quantum evolution
    for ground-state preparation in the Ising model.
    """

    # Parameters and setup
    J = 1
    h = 0

    isingModel = IsingModel(D, n_qubits, J=J, h=h, g=g)
    circuit = Circuit(n_qubits, isingModel)

    E0 = circuit.get_linalg_E0()

    # Running both algorithms
    states, E_adiabatic = circuit.run_adiabatic(steps=steps, T=T, n_shots=n_shots, backend=backend)

    result_VQE = circuit.run_VQE(backend=backend)

    E_VQE = float(np.real(result_VQE.fun))
    E_adiabatic = float(np.real(E_adiabatic))

    # Energy errors
    error_VQE = abs(E_VQE - E0)
    error_AQC = abs(E_adiabatic - E0)
    difference = abs(E_VQE - E_adiabatic)

    # Exact GHZ state
    ghz = np.zeros(2**n_qubits, dtype=complex)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    ghz = Statevector(ghz)

    
    # VQE final state using optimized parameters
    state_VQE = circuit.get_VQE_state(result_VQE.x, backend=backend)

    # VQE GHZ fidelity
    fidelity_VQE = float(state_fidelity(state_VQE, ghz))

    # AQC GHZ fidelity
    fidelity_AQC = []
    for state in states:
        fidelity_AQC.append(float(abs(state.data[0] + state.data[-1])**2 / 2))

    fidelity_AQC = float(np.mean(fidelity_AQC))


    # Print results
    print("******************************************************")
    print("VQE vs AQC comparison")
    print("***************************")

    if backend is None:
        print(f"Simulation:       {'Ideal'}")
        print(f"Backend:          {'None'}")
    else:
        print(f"Simulation:       {'With noise'}")
        print(f"Backend:          {type(backend).__name__}")

    print(f"Dimension:        {D}")
    print(f"Number of qubits: {n_qubits}")
    print(f"J = {J}, h = {h}, g = {g}")
    print(f"Evolution time T: {T}")
    print(f"Trotter steps M:  {steps}")
    print(f"Time step dt:     {T / steps:.6f}")

    if backend is not None:
        print(f"Noise trajectories: {len(states)}")

    print()
    print("Energies:")
    print(f"  Exact E0:        {E0:.10f}")
    print(f"  VQE:             {E_VQE:.10f}")
    print(f"  AQC:             {E_adiabatic:.10f}")

    print()
    print("Absolute energy errors:")
    print(f"  |E_VQE - E0|:    {error_VQE:.8e}")
    print(f"  |E_AQC - E0|:    {error_AQC:.8e}")
    print(f"  |E_VQE - E_AQC|: {difference:.8e}")

    print()
    print("GHZ fidelity:")
    print(f"  VQE:             {fidelity_VQE:.8f}")
    print(f"  AQC:             {fidelity_AQC:.8f}")

    print()
    print("VQE optimization:")
    print(f"  Converged:       {result_VQE.success}")
    print(f"  Evaluations:     {result_VQE.nfev}")

    print("******************************************************")



if __name__ == "__main__":
    compare_AQC_vs_VQE(T=100)
    compare_AQC_vs_VQE(T=100, backend=BACKEND)