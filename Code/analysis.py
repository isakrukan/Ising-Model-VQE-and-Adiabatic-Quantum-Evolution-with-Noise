r"""
Shared post-processing helpers for the Ising VQE / adiabatic experiments.

Two things used to be copy-pasted across the analysis scripts:

* the GHZ fidelity estimator

      F = |psi_0 + psi_{2^n - 1}|^2 / 2,

  which is exact because the GHZ state has only two non-zero amplitudes, and
* the ``"Data"`` output directory, which was resolved against the *current
  working directory* - so running any script from somewhere other than the
  repository root quietly wrote the results to the wrong place.

Both now live here, so every caller shares one implementation and one location.
"""

from pathlib import Path

import numpy as np

#: Repository root, independent of the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
#: Directory the analysis scripts write their ``.npz`` results to.
DATA_DIR = REPO_ROOT / "Data"
#: Directory the figure script writes to.
FIGS_DIR = REPO_ROOT / "Figs"

#: Sentinel meaning "use the default output directory" (kept for backwards
#: compatibility with the existing ``save_to="DEFAULT"`` call sites).
DEFAULT_OUTPUT = "DEFAULT"

#: The GHZ state is only ever compared against these two amplitudes.
GHZ_AMPLITUDE_ERROR = (
    "A state compatible with a GHZ state needs at least two amplitudes "
    "(2 or more qubits)."
)


def resolve_output_dir(path=None, default=None) -> Path:
    """
    Directory results should be written to.

    ``None`` and the ``"DEFAULT"`` sentinel both mean the default, which is the
    repository's ``Data/`` directory: never a path relative to the current
    working directory. Relative paths are still honoured, but they are resolved
    against the repository root rather than ``os.getcwd()``.

    Parameters:
        path:    None, "DEFAULT", or a filesystem path.
        default: Directory used for None / "DEFAULT" (``DATA_DIR`` if omitted).

    Returns:
        An absolute ``Path``.
    """
    if default is None:
        default = DATA_DIR

    if path is None or (isinstance(path, str) and path == DEFAULT_OUTPUT):
        return Path(default)

    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _is_single_state(states) -> bool:
    """True when ``states`` is one state vector rather than a sequence of them."""
    if isinstance(states, np.ndarray):
        # Checked before the `data` attribute below: every ndarray has a `.data`
        # memoryview, so a (n_states, dim) batch would otherwise look like a
        # single state and be flattened into one meaningless vector.
        return states.ndim == 1
    return hasattr(states, "data")  # e.g. qiskit.quantum_info.Statevector


def _amplitudes(state) -> np.ndarray:
    """Dense amplitude array of a state vector, accepting objects with ``.data``."""
    if isinstance(state, np.ndarray):
        data = state
    else:
        data = getattr(state, "data", state)
    return np.asarray(data).reshape(-1)


def ghz_fidelity(states) -> float:
    r"""
    Mean fidelity of ``states`` with the n-qubit GHZ state

        |GHZ_n> = (|0...0> + |1...1>) / sqrt(2).

    Because |GHZ_n> has only two non-zero amplitudes, the overlap reduces to

        F = |psi_0 + psi_{2^n - 1}|^2 / 2,

    which avoids building the full 2^n GHZ vector.

    Parameters:
        states: A single state vector (``Statevector`` or 1-D ``np.ndarray``),
                or a sequence of them as returned by
                ``Circuit.run_adiabatic`` for a noisy run.

    Returns:
        The mean fidelity over all given states, as a float.
    """
    if not _is_single_state(states):
        states = list(states)

        if not states:
            raise ValueError("ghz_fidelity() received no states.")

    else:
        states = [states]

    fidelities = []
    for state in states:
        amplitudes = _amplitudes(state)

        if amplitudes.size < 2:
            raise ValueError(GHZ_AMPLITUDE_ERROR)

        fidelities.append(abs(amplitudes[0] + amplitudes[-1]) ** 2 / 2)

    return float(np.mean(fidelities))
