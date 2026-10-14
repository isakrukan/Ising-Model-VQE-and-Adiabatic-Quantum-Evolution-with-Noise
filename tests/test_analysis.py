"""Tests for the shared GHZ-fidelity and output-directory helpers."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from analysis import DATA_DIR, DEFAULT_OUTPUT, FIGS_DIR, REPO_ROOT, ghz_fidelity, resolve_output_dir


def ghz_state(n):
    """( |0...0> + |1...1> ) / sqrt(2)."""
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1 / np.sqrt(2)
    state[-1] = 1 / np.sqrt(2)

    return Statevector(state)


def plus_state(n):
    return Statevector(np.ones(2 ** n) / np.sqrt(2 ** n))


# --- ghz_fidelity ------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4])
def test_ghz_state_has_unit_fidelity(n):
    assert ghz_fidelity(ghz_state(n)) == pytest.approx(1.0)


def test_product_state_has_half_fidelity():
    assert ghz_fidelity(Statevector.from_label("00")) == pytest.approx(0.5)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_plus_state_fidelity_has_a_closed_form(n):
    # |+>^n has 2^n equal amplitudes, so F = (2 / 2^(n/2))^2 / 2 = 2^(1 - n)
    assert ghz_fidelity(plus_state(n)) == pytest.approx(2 ** (1 - n))


def test_bell_state_has_unit_fidelity():
    assert ghz_fidelity(np.array([1, 0, 0, 1]) / np.sqrt(2)) == pytest.approx(1.0)


def test_mean_over_several_trajectories():
    states = [ghz_state(2), Statevector.from_label("00")]

    assert ghz_fidelity(states) == pytest.approx(0.75)


def test_dense_array_is_accepted():
    assert ghz_fidelity(ghz_state(2).data) == pytest.approx(1.0)


def test_batch_array_is_treated_as_many_states():
    batch = np.vstack([ghz_state(2).data, Statevector.from_label("00").data])

    assert ghz_fidelity(batch) == pytest.approx(0.75)


def test_single_amplitude_is_rejected():
    with pytest.raises(ValueError, match="at least two amplitudes"):
        ghz_fidelity(np.array([1.0]))


def test_empty_sequence_is_rejected():
    with pytest.raises(ValueError, match="no states"):
        ghz_fidelity([])


def test_global_phase_does_not_change_the_fidelity():
    state = ghz_state(2)
    rotated = Statevector(np.exp(1j * 0.7) * state.data)

    assert ghz_fidelity(rotated) == pytest.approx(1.0)


# --- resolve_output_dir ------------------------------------------------------

def test_default_output_directory_is_the_repository_data_directory():
    assert resolve_output_dir() == DATA_DIR
    assert resolve_output_dir(None) == DATA_DIR
    assert resolve_output_dir(DEFAULT_OUTPUT) == DATA_DIR


def test_the_data_directory_is_not_the_current_working_directory(tmp_path, monkeypatch):
    """
    Regression: "Data" used to be resolved against os.getcwd(), so running an
    analysis from anywhere else wrote the results outside the repository.
    """
    monkeypatch.chdir(tmp_path)

    assert resolve_output_dir(DEFAULT_OUTPUT) == DATA_DIR
    assert resolve_output_dir(DEFAULT_OUTPUT) != tmp_path / "Data"


def test_relative_paths_are_anchored_to_the_repository_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_output_dir("results") == REPO_ROOT / "results"


def test_absolute_paths_are_kept(tmp_path):
    assert resolve_output_dir(tmp_path) == tmp_path


def test_directory_arguments_are_accepted(tmp_path):
    assert resolve_output_dir(tmp_path / "nested") == tmp_path / "nested"


def test_compatible_default_override(tmp_path):
    assert resolve_output_dir(DEFAULT_OUTPUT, default=tmp_path) == tmp_path


def test_repository_layout_constants():
    assert REPO_ROOT.joinpath("Code", "analysis.py").is_file()
    assert DATA_DIR.parent == REPO_ROOT
    assert FIGS_DIR.parent == REPO_ROOT
    assert DATA_DIR.name == "Data"
    assert FIGS_DIR.name == "Figs"
