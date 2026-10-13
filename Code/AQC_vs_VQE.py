from Circuit import *
from IsingModel import * 
from parameters import *


def compare_AQC_vs_VQE(D=2, n_qubits=4, g=0.01, steps=100, T=50.0, n_shots=200, backend=None):
    r"""
    Computes the ground state of 

    H = -sum_{i<j} J_ij Z_i Z_j
            -sum_i h_i Z_i
            -sum_i g_i X_i

    which should be the GHZ state:
    
        |GHZ_n> = 1/sqrt{2}(|0^{\otimes n}> + |1^{\otimes n}>).

    Fidelity is computed by considering the first and last state entries.
    """
    # Parameters and setup
    J=1; h=0
    
    isingModel = IsingModel(D, n_qubits, J=J, h=h, g=g)
    circuit = Circuit(n_qubits, isingModel)
    E0 = -4.0001000037499 # calculated using circuit.get_linalg_E0() once
    if backend is None:
        states, E_adiabatic = circuit.run_adiabatic(steps=steps, T=T, n_shots=n_shots)
        E_VQE = circuit.run_VQE()

    else:
        states, E_adiabatic = circuit.run_adiabatic(steps=steps, T=T, n_shots=n_shots, backend=backend)
        E_VQE = circuit.run_VQE(backend=backend)

    E_VQE = E_VQE.fun
    # Since ghz has only two nonzero entries:
    fidelity = []
    for state in states:
        fidelity.append(abs(state.data[0] + state.data[-1])**2 / 2)

    fidelity = np.mean(fidelity)

    if len(states) == 1:
        print(f"Done running adiabatic evolution without noise")

    else:
        print(f"Done running adiabatic evolution with noise")

    print(f"GHZ fidelity: {fidelity}")

    print(f"Error |E_VQE - E_0| = {abs(E_VQE - E0)}")
    print(f"Error |E_adiabatic - E_0| = {abs(E0 - E_adiabatic)}")
    print(f"Diff |E_VQE - E_adiabatic| = {abs(E_VQE - E_adiabatic)}")


if __name__ == "__main__":
    compare_AQC_vs_VQE(T=100)
    compare_AQC_vs_VQE(T=100, backend=BACKEND)