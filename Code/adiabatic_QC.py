from Circuit import *
from IsingModel import * 


if __name__ == "__main__":

    r"""
    Aiming at preparing a known final ground state, namely, the GHZ state,
        
        |GHZ_n> = 1/sqrt{2}(|0^{\otimes n}> + |0^{\otimes n}>),
    
    for H(J=1, h=0, g<J)
    """
    D = 2; n_qubits = 10

    isingModel = IsingModel(D, n_qubits, J=1, h=0, g=0.01)

    circuit = Circuit(n_qubits, isingModel)

    E, psi = circuit.run_adiabatic(steps=2000)

    ghz = np.zeros(2**circuit.n_qubits, dtype=complex)
    ghz[0] = 1 / np.sqrt(2)
    ghz[-1] = 1 / np.sqrt(2)

    fidelity = abs(np.vdot(ghz, psi.data))**2

    print(f"GHZ fidelity: {fidelity}")
    print(f"E_final: {E}")

    E0 = circuit.get_linalg_E0()

    print(f"Exact ground-state energy: {E0}")
    print(f"error: {abs(E-E0)}")

