from Circuit import * 
from IsingModel import * 

import numpy as np
from pathlib import Path

def analyze_Ry_phase_vs_CX_depolarizing_error(N:int, p_ry:list, p_cx:list, D=2, n_qubits=4, J=1, h=0, g=1, save_to="DEFAULT") -> None:

    p_ry = np.linspace(p_ry[0], p_ry[1], N)
    p_cx = np.linspace(p_cx[0], p_cx[1], N)

    isingModel = IsingModel(D, n_qubits, J=J, h=h, g=g)
    circuit = Circuit(n_qubits, isingModel)

    # Exact ground-state energy is independent of noise
    E0 = circuit.get_linalg_E0()
    E0_linalg = np.full((N, N), E0)
    E_VQE = np.zeros((N, N))
    error = np.zeros((N, N))

    counter = 0
    print(f"Running analyze_Ry_phase_vs_CX_depolarizing_error, {counter/N**2*100:.2f}% complete", end="\r")
    for i in range(N):
        for j in range(N):

            circuit.set_phase_flip_prop(float(p_ry[i]))
            circuit.set_cx_depolarizing_prop(float(p_cx[j]))

            result = circuit.run_VQE()

            E_VQE[i, j] = result.fun
            error[i, j] = abs(E0 - result.fun)

            counter += 1
            print(f"Running analyze_Ry_phase_vs_CX_depolarizing_error, {counter/N**2*100:.2f}% complete", end="\r")

    print(f"Running analyze_Ry_phase_vs_CX_depolarizing_error, {counter/N**2*100:.2f}% complete")
    # Default output directory
    if save_to == "DEFAULT":
        save_to = "Data"

    folder = Path(save_to)
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"Ry_phase_vs_CX_depolarizing_D{D}_N{n_qubits}_J{J}_h{h}_g{g}.npz"
    filepath = folder / filename

    np.savez_compressed(
        filepath,
        p_ry=p_ry,
        p_cx=p_cx,
        error=error,
        E_VQE=E_VQE,
        E0_linalg=E0_linalg)

    print(f"Results saved to {filepath}")


    


if __name__ == "__main__":
    analyze_Ry_phase_vs_CX_depolarizing_error(20, [0, 0.02], [0, 0.02])

    data = np.load("Data/Ry_phase_vs_CX_depolarizing_D2_N4_J1_h0_g1.npz")

    p_ry = data["p_ry"]
    p_cx = data["p_cx"]
    error = data["error"]

    plt.figure(figsize=(8, 6))

    im = plt.imshow(
        error,
        origin="lower",
        extent=[p_cx[0], p_cx[-1], p_ry[0], p_ry[-1]],
        aspect="auto",
        cmap="inferno",
        interpolation="nearest"
    )

    plt.colorbar(im, label=r"Energy error $|E_{\mathrm{VQE}}-E_0|$")

    plt.xlabel("CX depolarizing probability")
    plt.ylabel("RY phase-flip probability")
    plt.title("VQE ground-state energy error")

    plt.tight_layout()

    # Save figure alongside the data
    # plt.savefig(filename.with_suffix(".png"), dpi=300)
    plt.show()