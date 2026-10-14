from Circuit import *
from IsingModel import *
from parameters import *

from analysis import ghz_fidelity, resolve_output_dir

import numpy as np
from pathlib import Path
import time


def analyze_AQC_fidelity(
        D=2,
        n_qubits=4,
        res=100,
        T_range=[10, 100],
        steps_range=[10, 1000],
        J=1,
        h=0,
        g=[0.01, 1],
        n_shots=30,
        g_initial=2.0,
        save_to="DEFAULT",
        backend=BACKEND) -> Path:

    r"""
    Maps the GHZ fidelity of the adiabatic evolution over a grid of total
    evolution times T and numbers of Trotter steps, and saves the result. The
    final energy of every grid point is stored alongside the fidelity.

    Parameters:
        D:           Dimension (1 or 2).
        n_qubits:    Number of qubits (spin sites).
        res:         Number of grid points along each axis.
        T_range:     [min, max] total evolution time.
        steps_range: [min, max] number of time-evolution steps.
        J, h:        Coupling and longitudinal field of the final Hamiltonian.
        g:           Transverse field of the final Hamiltonian. Only g[0] enters
                     the Hamiltonian; a two-element sequence is accepted for
                     backwards compatibility with the existing make target.
        n_shots:     Number of simulated trajectories for the noisy run.
        g_initial:   Transverse field of the initial Hamiltonian H_initial.
        save_to:     Output directory. "DEFAULT" (or None) means the repository's
                     Data/ directory, resolved independently of the current
                     working directory.
        backend:     Qiskit backend providing calibration-derived noise. Pass
                     None for an ideal simulation.

    Returns:
        The path of the written .npz file.
    """
    if res < 1:
        raise ValueError("res must be a positive integer.")

    if isinstance(g, (list, tuple)):
        g_final = g[0]
        g_label = "-".join(str(x) for x in g)
    else:
        g_final = g
        g_label = str(g)

    # Initial configuration
    isingModel = IsingModel(D, n_qubits, J=J, h=h, g=g_final)
    circuit = Circuit(n_qubits, isingModel)

    fidelities = np.zeros((res, res))
    energies = np.zeros((res, res))
    T_range = np.linspace(T_range[0], T_range[1], res)
    steps_range = np.linspace(steps_range[0], steps_range[1], res, dtype=int)

    print("Beginning fidelity analysis")
    start_time = time.perf_counter()
    counter = 0
    for i, steps in enumerate(steps_range):
        for j, T in enumerate(T_range):

            states, E = circuit.run_adiabatic(T=T, steps=steps, n_shots=n_shots, g_initial=g_initial, backend=backend)
            fidelities[i, j] = ghz_fidelity(states)
            energies[i, j] = float(np.real(E))

            counter += 1
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            print(f"{counter/res**2 * 100:.2f}% complete, time elapsed: {elapsed_time:.2f} seconds")

    current_time = time.perf_counter()
    elapsed_time = current_time - start_time
    print(f"Fidelity analysis done, time elapses: {elapsed_time:.2f} seconds")

    folder = resolve_output_dir(save_to)
    folder.mkdir(parents=True, exist_ok=True)

    if backend is not None:
        filename = f"fidelities_D{D}_N{n_qubits}_J{J}_h{h}_g{g_label}_T{T_range[0]}-{T_range[-1]}_steps{steps_range[0]}-{steps_range[-1]}_res{res}_backend.npz"

    else:
        filename = f"fidelities_D{D}_N{n_qubits}_J{J}_h{h}_g{g_label}_T{T_range[0]}-{T_range[-1]}_steps{steps_range[0]}-{steps_range[-1]}_res{res}_.npz"
    filepath = folder / filename

    np.savez_compressed(
        filepath,
        T_range=T_range,
        steps_range=steps_range,
        fidelities=fidelities,
        energies=energies)

    print(f"Results saved to {filepath}")

    return filepath


if __name__ == "__main__":

    r"""
    Aiming at preparing a known final ground state, namely, the GHZ state,

        |GHZ_n> = 1/sqrt{2}(|0^{\otimes n}> + |1^{\otimes n}>),

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
    fidelity = ghz_fidelity(states)

    if len(states) == 1:
        print("Done running adiabatic evolution without noise")

    else:
        print("Done running adiabatic evolution with noise")

    print(f"GHZ fidelity: {fidelity}")
    print(f"E_final: {E}")

    # E0 = circuit.get_linalg_E0()

    # print(f"Exact ground-state energy: {E0}")
    # print(f"error: {abs(E-E0)}")
