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
        backend=None) -> None:

    # Initial configuration
    isingModel = IsingModel(D, n_qubits, J=J, h=h, g=g[0])
    circuit = Circuit(n_qubits, isingModel)

    fidelities = np.zeros((res, res))
    T_range = np.linspace(T_range[0], T_range[1], res)
    steps_range = np.linspace(steps_range[0], steps_range[1], res, dtype=int)

    print(f"Beginning fidelity analysis")
    start_time = time.perf_counter()
    counter = 0
    for i, steps in enumerate(steps_range):
        for j, T in enumerate(T_range):

            states, E = circuit.run_adiabatic(T=T, steps=steps, n_shots=n_shots, backend=backend, print_progress=False)
            fidelity = []
            for state in states:
                fidelity.append(abs(state.data[0] + state.data[-1])**2 / 2)

            fidelity = np.mean(fidelity); fidelities[i,j] = fidelity

            counter += 1
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            print(f"{counter/res**2 * 100:.2f}% complete, time elapsed: {elapsed_time:.2f} seconds", end="\r", flush=True)

    current_time = time.perf_counter()
    elapsed_time = current_time - start_time
    print(" " * 100, end="\r")
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

    plot_fidelities(filename[0:-4], show=False)

def plot_fidelities(name:str, show=True):

    data1 = np.load(f"Data/{name}_backend.npz")
    data2 = np.load(f"Data/{name}_.npz")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    T_range1 = data1["T_range"]; steps_range1 = data1["steps_range"]
    fidelities1 = data1["fidelities"]
    im1 = axs[0].imshow(
        fidelities1,
        origin="lower",
        extent=[T_range1[0], T_range1[-1], steps_range1[0], steps_range1[-1]],
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
        vmin=0, vmax=1
    )

    T_range2 = data2["T_range"]; steps_range2 = data2["steps_range"]
    fidelities2 = data2["fidelities"]
    im2 = axs[1].imshow(
        fidelities2,
        origin="lower",
        extent=[T_range2[0], T_range2[-1], steps_range2[0], steps_range2[-1]],
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
        vmin=0, vmax=1
    )

    axs[0].set_title("With backend noise")
    axs[1].set_title("Without noise")

    for ax in axs:
        ax.set_xlabel("Evolution time T")

    axs[0].set_ylabel("Trotter steps")

    fig.colorbar(im2, ax=axs, label="GHZ fidelity", shrink=0.85)

    folder = Path("Figs")
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"{name}.pdf"
    filepath = folder / filename
    plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()


def find_best_values_with_noise(name):
    data = np.load(f"Data/{name}_backend.npz")
    T_range = data["T_range"]
    steps_range = data["steps_range"]
    fidelities = data["fidelities"]

    # Maximum fidelity
    index = np.unravel_index(np.argmax(fidelities), fidelities.shape)

    print("Maximum fidelity:")
    print(f"T = {T_range[index[1]]}, M = {steps_range[index[0]]}, F = {fidelities[index]}")

    # All fidelities >= 0.9
    print()
    print("Fidelities >= 0.9")
    indices_steps, indices_T = np.where(fidelities >= 0.9)

    T_values = T_range[indices_T]
    steps_values = steps_range[indices_steps]

    sort_indices = np.lexsort((steps_values, T_values))

    print("T values:", T_values[sort_indices])
    print("Steps values:", steps_values[sort_indices])
    print("Fidelities:", fidelities[indices_steps, indices_T][sort_indices])

    
    
    # Find the region where fidelity >= 0.9
    indices_steps, indices_T = np.where(fidelities >= 0.9)

    if len(indices_T) == 0:
        print("No fidelities >= 0.9")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        fidelities,
        origin="lower",
        extent=[T_range[0], T_range[-1], steps_range[0], steps_range[-1]],
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
        vmin=0, vmax=1
    )

    # Zoom to the region containing all fidelities >= 0.9
    ax.set_xlim(T_range[indices_T.min()], T_range[indices_T.max()])
    ax.set_ylim(steps_range[indices_steps.min()], steps_range[indices_steps.max()])

    ax.set_xlabel("Evolution time T")
    ax.set_ylabel("Trotter steps M")
    ax.set_title("GHZ fidelity with backend noise")

    fig.colorbar(im, ax=ax, label="GHZ fidelity")

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    

    name1 = "fidelities_D2_N4_J1_h0_g[0.01, 3]_T50.0-1300.0_steps50-1300_res25"
    name2 = "fidelities_D2_N4_J1_h0_g[0.01, 3]_T1.0-200.0_steps1-200_res20"
    name3 = "fidelities_D2_N4_J1_h0_g[0.01, 3]_T1.0-100.0_steps1-100_res100"
    name4 = "fidelities_D2_N4_J1_h0_g[0.01, 3]_T1.0-1000.0_steps1-1000_res20"

    find_best_values_with_noise(name3)
    
    plot_fidelities(name1)
    plot_fidelities(name2)
    plot_fidelities(name3)
    plot_fidelities(name4)

    """
    4.6 = 4.8a + b, 40.1 = 34.4a + b 
    a = (4.6-b)/4.8 = (40.1 - b)/34.4
    b = -1.15675
    a = 1.19932
    """

    


