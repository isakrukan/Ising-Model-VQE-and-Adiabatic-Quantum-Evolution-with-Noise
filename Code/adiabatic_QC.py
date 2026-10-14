from Circuit import *
from IsingModel import * 
from parameters import *

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
        save_to="DEFAULT",
        backend=BACKEND) -> None:

    # Initial configuration
    isingModel = IsingModel(D, n_qubits, J=J, h=h, g=g[0])
    circuit = Circuit(n_qubits, isingModel)

    fidelities = np.zeros((res, res))
    T_range = np.linspace(T_range[0], T_range[1], res)
    steps_range = np.linspace(steps_range[0], steps_range[1], res)

    print(f"Beginning fidelity analysis")
    start_time = time.perf_counter()
    counter = 0
    for i, steps in enumerate(steps_range):
        for j, T in enumerate(T_range):

            states, E = circuit.run_adiabatic(T=T, steps=steps, n_shots=n_shots, backend=backend)
            fidelity = []
            for state in states:
                fidelity.append(abs(state.data[0] + state.data[-1])**2 / 2)

            fidelity = np.mean(fidelity); fidelities[i,j] = fidelity

            counter += 1
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            print(f"{counter/res**2 * 100:.2f}% complete, time elapsed: {elapsed_time:.2f} seconds")

    current_time = time.perf_counter()
    elapsed_time = current_time - start_time
    print(f"Fidelity analysis done, time elapses: {elapsed_time:.2f} seconds")

    if save_to == "DEFAULT":
        save_to = "Data"

    folder = Path(save_to)
    folder.mkdir(parents=True, exist_ok=True)

    if backend is not None:
        filename = f"fidelities_D{D}_N{n_qubits}_J{J}_h{h}_g{g}_T{T_range[0]}-{T_range[-1]}_steps{steps_range[0]}-{steps_range[-1]}_res{res}_backend.npz"

    else:
        filename = f"fidelities_D{D}_N{n_qubits}_J{J}_h{h}_g{g}_T{T_range[0]}-{T_range[-1]}_steps{steps_range[0]}-{steps_range[-1]}_res{res}_.npz"
    filepath = folder / filename

    np.savez_compressed(
        filepath,
        T_range=T_range,
        steps_range=steps_range,
        fidelities=fidelities)



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

