from Circuit import * 
from IsingModel import * 
from parameters import *


if __name__ == "__main__":

    # VQE for Ising model with default parameters and nearest neighbor interactions
    # Parameters
    D = 2 # Dimension 
    N = 10 # Number of qubits (square lattice)

    isingModel = IsingModel(D, N)
    circuit = Circuit(n_qubits=N, isingModel=isingModel)
    # circuit.display_circuit()

    circuit.set_phase_flip_prop(0.01)
    circuit.set_cx_depolarizing_prop(0.01)

    result = circuit.run_VQE(tol=1e-2, method="COBYLA")
    linalg_E0 = circuit.get_linalg_E0()

    print("Using custom gate parameters:")
    print(circuit.doc)
    print(f"E0 from VQE: {result.fun}")
    print(f"E0 from linalg: {linalg_E0}")
    print(f"diff = {abs(result.fun - linalg_E0)}")

    # Using the parameters of a backend:
    isingModel = IsingModel(D, N)
    circuit = Circuit(n_qubits=N, isingModel=isingModel, transpile_gates=BASIS_GATES)


    result = circuit.run_VQE(tol=1e-2, method="COBYLA", backend=FakeSherbrooke())
    # linalg_E0 = circuit.get_linalg_E0()

    print(f"Using parameters from {BACKEND.name}:")
    print(circuit.doc)
    print(f"E0 from VQE: {result.fun}")
    print(f"E0 from linalg: {linalg_E0}")
    print(f"diff = {abs(result.fun - linalg_E0)}")
