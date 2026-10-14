from Circuit import *
from IsingModel import *
from parameters import *

from analysis import ghz_fidelity


def compare_AQC_vs_VQE(D=2, n_qubits=4, g=0.01, steps=100, T=50.0, n_shots=200, backend=None, E0=None, seed=42) -> dict:
    r"""
    Computes the ground state of

    H = -sum_{i<j} J_ij Z_i Z_j
            -sum_i h_i Z_i
            -sum_i g_i X_i

    which should be the GHZ state:

        |GHZ_n> = 1/sqrt{2}(|0^{\otimes n}> + |1^{\otimes n}>).

    Fidelity is computed by considering the first and last state entries.
    The energy is compared against the exact ground-state energy of the same
    Hamiltonian, and against the VQE estimate.

    Parameters:
        D:          Dimension (1 or 2).
        n_qubits:   Number of qubits (spin sites).
        g:          Transverse field of the final Hamiltonian (g < J).
        steps:      Number of time-evolution steps for the adiabatic run.
        T:          Total adiabatic evolution time.
        n_shots:    Number of simulated trajectories for the noisy run.
        backend:    Optional Qiskit backend (e.g. FakeSherbrooke()).
        E0:         Exact ground-state energy to compare against. If None it is
                    computed from this Hamiltonian with get_linalg_E0().
        seed:       Seed for the random VQE initial parameters.

    Returns:
        Dict with the GHZ fidelity, E_VQE, E_adiabatic and the reference E0.
    """
    # Parameters and setup
    J = 1
    h = 0

    isingModel = IsingModel(D, n_qubits, J=J, h=h, g=g)
    circuit = Circuit(n_qubits, isingModel)

    # Exact ground-state energy of *this* Hamiltonian. This used to be the
    # literal -4.0001000037499, which is only correct for the default system
    # (D=2, n_qubits=4, J=1, h=0, g=0.01); every other call, e.g.
    # compare_AQC_vs_VQE(n_qubits=6), reported the energy errors against a
    # reference three units away from the true one.
    if E0 is None:
        E0 = circuit.get_linalg_E0()

    if backend is None:
        states, E_adiabatic = circuit.run_adiabatic(steps=steps, T=T, n_shots=n_shots)
        E_VQE = circuit.run_VQE(seed=seed)

    else:
        states, E_adiabatic = circuit.run_adiabatic(steps=steps, T=T, n_shots=n_shots, backend=backend)
        E_VQE = circuit.run_VQE(backend=backend, seed=seed)

    E_VQE = E_VQE.fun

    # Since ghz has only two nonzero entries:
    fidelity = ghz_fidelity(states)

    if len(states) == 1:
        print("Done running adiabatic evolution without noise")

    else:
        print("Done running adiabatic evolution with noise")

    print(f"GHZ fidelity: {fidelity}")

    print(f"Error |E_VQE - E_0| = {abs(E_VQE - E0)}")
    print(f"Error |E_adiabatic - E_0| = {abs(E0 - E_adiabatic)}")
    print(f"Diff |E_VQE - E_adiabatic| = {abs(E_VQE - E_adiabatic)}")

    return {
        "fidelity": fidelity,
        "E_VQE": float(E_VQE),
        "E_adiabatic": float(E_adiabatic),
        "E0": float(E0),
    }


if __name__ == "__main__":
    compare_AQC_vs_VQE(T=100)
    compare_AQC_vs_VQE(T=100, backend=BACKEND)
