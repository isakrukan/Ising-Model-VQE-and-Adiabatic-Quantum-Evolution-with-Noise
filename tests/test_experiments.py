"""Tests for the analysis scripts and their saved output."""

import numpy as np
import pytest

import analysis
from AQC_vs_VQE import compare_AQC_vs_VQE
from Circuit import Circuit
from IsingModel import IsingModel
from adiabatic_QC import analyze_AQC_fidelity
from probability_analysis import analyze_Ry_phase_vs_CX_depolarizing_error


@pytest.fixture
def tiny(tmp_path):
    """Arguments for the cheapest possible sweep."""
    return {"D": 1, "n_qubits": 2, "res": 1, "T_range": [1, 1], "steps_range": [2, 2], "n_shots": 2, "save_to": tmp_path, "backend": None}


# --- analyze_AQC_fidelity ----------------------------------------------------

def test_fidelity_scan_writes_the_grid(tmp_path):
    path = analyze_AQC_fidelity(D=1, n_qubits=2, res=2, T_range=[1, 2], steps_range=[2, 4], n_shots=2, save_to=tmp_path, backend=None)

    assert path.is_file()
    assert path.parent == tmp_path

    data = np.load(path)

    assert set(data) == {"T_range", "steps_range", "fidelities", "energies"}
    assert data["fidelities"].shape == (2, 2)
    assert data["energies"].shape == (2, 2)
    assert np.all(np.isfinite(data["energies"]))
    assert np.all((data["fidelities"] >= 0) & (data["fidelities"] <= 1))


def test_fidelity_scan_default_directory_is_the_repository_data_directory(tiny, tmp_path, monkeypatch):
    """
    Regression: "Data" was relative to os.getcwd(), so a run started elsewhere
    wrote its results outside the repository.
    """
    monkeypatch.setattr(analysis, "DATA_DIR", tmp_path / "Data")

    path = analyze_AQC_fidelity(**{**tiny, "save_to": "DEFAULT"})

    assert path.parent == tmp_path / "Data"


def test_fidelity_scan_returns_the_written_path(tiny):
    path = analyze_AQC_fidelity(**tiny)

    assert path.name.endswith(".npz")
    assert path.exists()


def test_fidelity_scan_filename_does_not_embed_a_list_repr(tiny):
    """
    Regression: the filename interpolated the raw g list, producing names like
    "..._g[0.01, 1]_T...".
    """
    path = analyze_AQC_fidelity(**{**tiny, "g": [0.01, 1]})

    assert "g0.01-1" in path.name
    assert "[" not in path.name
    assert " " not in path.name


def test_fidelity_scan_accepts_a_scalar_g(tiny):
    path = analyze_AQC_fidelity(**{**tiny, "g": 0.5})

    assert "g0.5" in path.name


def test_fidelity_scan_marks_runs_that_used_a_backend(tiny):
    from parameters import FakeSherbrooke

    path = analyze_AQC_fidelity(**{**tiny, "backend": FakeSherbrooke()})

    assert path.name.endswith("_backend.npz")


def test_fidelity_scan_ideal_runs_have_no_backend_suffix(tiny):
    path = analyze_AQC_fidelity(**tiny)

    assert not path.name.endswith("_backend.npz")
    assert "noise" not in path.name


def test_fidelity_scan_validates_the_grid_size(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        analyze_AQC_fidelity(res=0, save_to=tmp_path)


def test_fidelity_scan_forwards_g_initial(tiny, monkeypatch):
    seen = {}
    real = Circuit.run_adiabatic

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Circuit, "run_adiabatic", spy)

    analyze_AQC_fidelity(**{**tiny, "g_initial": 3.5})

    assert seen["g_initial"] == 3.5


def test_fidelity_scan_measures_the_ghz_fidelity(tiny):
    """
    A slow sweep towards g -> 0 prepares the GHZ state, so the single grid point
    of the tiny configuration must be clearly non-zero.
    """
    path = analyze_AQC_fidelity(**{**tiny, "T_range": [40, 40], "steps_range": [400, 400]})

    assert np.load(path)["fidelities"][0, 0] > 0.9


# --- analyze_Ry_phase_vs_CX_depolarizing_error -------------------------------

def test_ry_phase_scan_writes_the_grid(tmp_path):
    path = analyze_Ry_phase_vs_CX_depolarizing_error(2, [0.0, 0.02], [0.0, 0.02], D=1, n_qubits=2, save_to=tmp_path)

    assert path.name == "Ry_phase_vs_CX_depolarizing_D1_N2_J1_h0_g1.npz"
    assert path.parent == tmp_path

    data = np.load(path)

    assert set(data) == {"p_ry", "p_cx", "error", "E_VQE", "E0_linalg"}
    assert data["E_VQE"].shape == (2, 2)
    assert np.allclose(data["E0_linalg"], data["E0_linalg"][0, 0])
    assert np.allclose(data["error"], np.abs(data["E0_linalg"] - data["E_VQE"]))
    assert data["p_ry"][0] == 0.0 and data["p_ry"][-1] == pytest.approx(0.02)


def test_ry_phase_scan_reference_energy_is_exact(tmp_path):
    path = analyze_Ry_phase_vs_CX_depolarizing_error(1, [0.0, 0.0], [0.0, 0.0], D=1, n_qubits=2, save_to=tmp_path)

    expected = Circuit(2, IsingModel(1, 2, J=1, h=0, g=1)).get_linalg_E0()

    assert np.load(path)["E0_linalg"][0, 0] == pytest.approx(expected)


def test_ry_phase_scan_validates_the_grid_size(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        analyze_Ry_phase_vs_CX_depolarizing_error(0, [0.0, 0.01], [0.0, 0.01], save_to=tmp_path)


@pytest.mark.parametrize("p_ry, p_cx", [([0.0], [0.0, 0.01]), ([0.0, 0.01], [0.0]), ([0.0, 0.01, 0.02], [0.0, 0.01])])
def test_ry_phase_scan_validates_the_probability_pairs(tmp_path, p_ry, p_cx):
    with pytest.raises(ValueError, match="pairs"):
        analyze_Ry_phase_vs_CX_depolarizing_error(2, p_ry, p_cx, save_to=tmp_path)


def test_ry_phase_scan_default_directory_is_the_repository_data_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "DATA_DIR", tmp_path / "Data")

    path = analyze_Ry_phase_vs_CX_depolarizing_error(1, [0.0, 0.0], [0.0, 0.0], D=1, n_qubits=2)

    assert path.parent == tmp_path / "Data"


# --- compare_AQC_vs_VQE ------------------------------------------------------

def test_compare_returns_the_measurements():
    result = compare_AQC_vs_VQE(D=1, n_qubits=2, g=0.01, steps=4, T=2.0, n_shots=2)

    assert set(result) == {"fidelity", "E_VQE", "E_adiabatic", "E0"}
    assert 0.0 <= result["fidelity"] <= 1.0
    assert result["E_VQE"] >= result["E0"] - 1e-9


def test_compare_reference_energy_follows_the_hamiltonian():
    """
    Regression: E0 was the literal -4.0001000037499, which is only right for the
    default four-site system, so any other call reported its energy errors
    against a reference three units away from the true one.
    """
    result = compare_AQC_vs_VQE(D=1, n_qubits=2, g=0.01, steps=3, T=1.0, n_shots=2)

    exact = Circuit(2, IsingModel(1, 2, J=1, h=0, g=0.01)).get_linalg_E0()

    assert result["E0"] == pytest.approx(exact)
    assert result["E0"] == pytest.approx(-1.000199980003999, rel=1e-9)
    assert result["E0"] != pytest.approx(-4.0001000037499)


def test_compare_accepts_an_explicit_reference_energy():
    result = compare_AQC_vs_VQE(D=1, n_qubits=2, g=0.01, steps=3, T=1.0, n_shots=2, E0=0.0)

    assert result["E0"] == 0.0


def test_compare_seed_is_reproducible():
    first = compare_AQC_vs_VQE(D=1, n_qubits=2, g=0.01, steps=3, T=1.0, n_shots=2, seed=5)
    second = compare_AQC_vs_VQE(D=1, n_qubits=2, g=0.01, steps=3, T=1.0, n_shots=2, seed=5)

    assert first["E_VQE"] == second["E_VQE"]


def test_compare_reports_the_noise_regime(capsys):
    compare_AQC_vs_VQE(D=1, n_qubits=2, g=0.01, steps=3, T=1.0, n_shots=2)

    assert "without noise" in capsys.readouterr().out
