from Circuit import *
from IsingModel import * 
from parameters import *

if __name__ == "__main__":

    r"""
    Aiming at preparing a known final ground state, namely, the GHZ state,
        
        |GHZ_n> = 1/sqrt{2}(|0^{\otimes n}> + |0^{\otimes n}>),
    
    for H(J=1, h=0, g<J)
    """
    D = 2; n_qubits = 4

    isingModel = IsingModel(D, n_qubits, J=1, h=0, g=0.01)

    circuit = Circuit(n_qubits, isingModel)

    # states, E = circuit.run_adiabatic(steps=2000)
    states, E = circuit.run_adiabatic(steps=2000, backend=FakeSherbrooke())

    # # May take a long time
    # ghz = np.zeros(2**circuit.n_qubits, dtype=complex)
    # ghz[0] = 1 / np.sqrt(2)
    # ghz[-1] = 1 / np.sqrt(2)

    # fidelity = np.real(np.vdot(ghz, np.matmul(rho.data, ghz)))

    # Instead, since ghz has only two nonzero entries:
    fidelity = []
    for state in states:
        fidelity.append(abs(state.data[0] + state.data[-1])**2 / 2)

    fidelity = np.mean(fidelity)

    if len(states) == 1:
        print(f"Done running adiabatic evolution without noise")

    else:
        print(f"Done running adiabatic evolution with noise")

    print(f"GHZ fidelity: {fidelity}")
    print(f"E_final: {E}")

    # E0 = circuit.get_linalg_E0()

    # print(f"Exact ground-state energy: {E0}")
    # print(f"error: {abs(E-E0)}")

