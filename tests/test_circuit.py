"""Tests for the VQE / adiabatic runner."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import OptimizeResult

import Circuit as circuit_module
from Circuit import EXACT_DIAGONALIZATION_LIMIT, Circuit
from IsingModel import IsingModel


@pytest.fixture
def small():
    """Two-site chain at a moderate transverse field."""
    return Circuit(2, IsingModel(1, 2, J=1.0, h=0.0, g=0.5))


def adiabatic_energy(configure, g=0.01):
    """Final adiabatic energy of a fresh two-site circuit after ``configure``."""
    circuit = Circuit(2, IsingModel(1, 2, J=1.0, h=0.0, g=g))
    configure(circuit)
    _, energy = circuit.run_adiabatic(steps=4, T=1.0, n_shots=8)

    return float(np.real(energy))


def error_operations(error):
    """Gate names of the Kraus branches of a QuantumError."""
    return [branch[0]["name"] for branch in error.to_dict()["instructions"]]


# --- construction ------------------------------------------------------------

def test_qubit_count_must_match_the_model():
    with pytest.raises(ValueError, match="must match number of spin sites"):
        Circuit(3, IsingModel(1, 2))


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_ansatz_has_four_parameters_per_qubit(n_qubits):
    circuit = Circuit(n_qubits, IsingModel(1, n_qubits, J=1.0, h=0.0, g=0.5))

    assert circuit.circuit.num_parameters == 4 * n_qubits


def test_transpile_gates_restricts_the_basis():
    circuit = Circuit(2, IsingModel(1, 2, J=1.0, h=0.0, g=0.5), transpile_gates=["rx", "rz", "cx"])

    assert set(circuit.circuit.count_ops()) <= {"rx", "rz", "cx"}


def test_default_is_ideal():
    circuit = Circuit(2, IsingModel(1, 2))

    assert circuit.noise is False
    assert circuit.noise_model is None
    assert circuit.noises == {}
    assert circuit.doc == {}


def test_stored_backend_is_used_when_no_backend_is_passed(small):
    sentinel = object()
    circuit = Circuit(2, IsingModel(1, 2), backend=sentinel)

    assert circuit.backend is sentinel


# --- theta0 handling ---------------------------------------------------------

def test_random_theta0_is_reproducible(small):
    first = small.run_VQE(maxiter=5)
    second = small.run_VQE(maxiter=5)

    assert first.fun == second.fun
    assert np.array_equal(first.x, second.x)


def test_random_theta0_seed_changes_the_starting_point(small):
    first = small.run_VQE(maxiter=5, seed=1)
    second = small.run_VQE(maxiter=5, seed=2)

    assert not np.array_equal(first.x, second.x)


def test_random_keyword_is_case_insensitive(small):
    assert small.run_VQE(theta0="random", maxiter=5).fun == small.run_VQE(theta0="RANDOM", maxiter=5).fun


def test_numpy_theta0_is_accepted(small):
    """
    Regression: ``theta0 == "RANDOM"`` compared an array against a string, which
    yields an element-wise result, so passing the documented array-like theta0
    raised "truth value of an array with more than one element is ambiguous".
    """
    result = small.run_VQE(theta0=np.zeros(8), maxiter=6)

    # theta = 0 prepares |00>, whose energy is exactly -J for a two-site chain.
    assert result.fun <= -1.0 + 1e-9


def test_list_theta0_matches_array_theta0(small):
    from_list = small.run_VQE(theta0=[0.0] * 8, maxiter=6)
    from_array = small.run_VQE(theta0=np.zeros(8), maxiter=6)

    assert from_list.fun == pytest.approx(from_array.fun)


def test_theta0_wrong_size_raises(small):
    with pytest.raises(ValueError, match="not the correct size 3/8"):
        small.run_VQE(theta0=np.zeros(3))


def test_unknown_theta0_string_raises(small):
    with pytest.raises(ValueError, match="Unknown initial parameters"):
        small.run_VQE(theta0="NOT RANDOM")


def test_unknown_optimizer_raises(small):
    with pytest.raises(ValueError, match="Unknown optimization method"):
        small.run_VQE(method="GRADIENT_DESCENT")


# --- optimization methods ----------------------------------------------------

def test_cobyla_returns_an_optimize_result(small):
    result = small.run_VQE(method="COBYLA", maxiter=8)
    exact = small.get_linalg_E0()

    assert isinstance(result, OptimizeResult)
    # Better than the |00> starting point, and never below the exact minimum.
    assert exact <= result.fun <= -1.0


def test_trust_constr_returns_an_optimize_result(small):
    result = small.run_VQE(method="trust-constr", maxiter=4)

    assert isinstance(result, OptimizeResult)
    assert np.isfinite(result.fun)


def test_spsa_returns_an_optimize_result(small):
    result = small.run_VQE(method="SPSA", maxiter=3)

    assert isinstance(result, OptimizeResult)
    assert np.isfinite(result.fun)
    assert result.success is None
    assert "convergence not verified" in result.message


def test_vqe_energy_respects_the_variational_bound():
    """
    <psi|H|psi> >= E0 for any state, so the VQE estimate can never undershoot the
    exact ground-state energy.
    """
    circuit = Circuit(3, IsingModel(1, 3, J=1.0, h=0.0, g=0.5))
    exact = circuit.get_linalg_E0()

    assert circuit.run_VQE(maxiter=30).fun >= exact - 1e-9


def test_vqe_energy_respects_the_bound_under_noise():
    circuit = Circuit(2, IsingModel(1, 2, J=1.0, h=0.0, g=0.5))
    exact = circuit.get_linalg_E0()
    circuit.set_phase_flip_prop(0.05)

    assert circuit.run_VQE(maxiter=8).fun >= exact - 1e-9


# --- noise registration ------------------------------------------------------

@pytest.mark.parametrize("p", [-0.1, 1.1, 2.0, -1.0])
def test_probability_must_be_in_the_unit_interval(small, p):
    with pytest.raises(ValueError, match="between 0 and 1"):
        small.set_phase_flip_prop(p)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_probability_bounds_are_inclusive(small, p):
    small.set_phase_flip_prop(p)

    assert small.noise is True


def test_phase_flip_registers_a_z_error(small):
    small.set_phase_flip_prop(0.3, gate="ry")

    assert error_operations(small.noises["phase flip (ry)"][0]) == ["z", "id"]
    assert small.doc == {"Phase flip probability (ry): ": 0.3}


def test_bit_flip_registers_an_x_error(small):
    small.set_bit_flip_prop(0.3, gate="ry")

    assert error_operations(small.noises["bit flip (ry)"][0]) == ["x", "id"]
    assert "Bit flip probability (ry): " in small.doc


def test_cx_depolarizing_registers_a_two_qubit_error(small):
    small.set_cx_depolarizing_prop(0.1, gate="cx")

    error = small.noises["deploarizing (cx)"][0]

    assert error.num_qubits == 2
    assert small.doc["Depolarizing probability (cx): "] == 0.1


def test_single_qubit_gate_is_the_default(small):
    small.set_amplitude_damping_prop(0.1)

    assert small.noises["amplitude damping (ry)"][0].num_qubits == 1


def test_cx_damping_is_tensored_to_two_qubits(small):
    small.set_amplitude_damping_prop(0.1, gate="cx")
    small.set_phase_damping_prop(0.2, gate="cx")

    assert small.noises["amplitude damping (cx)"][0].num_qubits == 2
    assert small.noises["phase damping (cx)"][0].num_qubits == 2


@pytest.mark.parametrize("T1, T2, gate_time", [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, -1.0), (1.0, 2.1, 1.0)])
def test_thermal_relaxation_validates_its_arguments(small, T1, T2, gate_time):
    with pytest.raises(ValueError, match="Require T1,T2"):
        small.set_thermal_relaxation_prop(T1, T2, gate_time)


def test_thermal_relaxation_records_the_derived_pure_dephasing_time(small):
    small.set_thermal_relaxation_prop(60e-6, 90e-6, 0.5e-6)

    assert small.doc["T1 (ry)"] == 60e-6
    assert small.doc["T2 (ry)"] == 90e-6
    assert small.doc["T_phi (ry)"] == pytest.approx(1 / (1 / 90e-6 - 1 / (2 * 60e-6)))
    assert small.doc["Gate time (ry)"] == 0.5e-6


def test_add_noise_model_switches_noise_on(small):
    model = NoiseModel()

    small.add_noise_model(model)

    assert small.noise is True
    assert small.noise_model is model


def test_doc_is_sorted(small):
    small.set_phase_flip_prop(0.1)
    small.set_thermal_relaxation_prop(60e-6, 90e-6, 0.5e-6)

    assert list(small.doc) == sorted(small.doc)


# --- noise model resolution --------------------------------------------------

def test_resolve_noise_model_is_none_for_an_ideal_circuit(small):
    assert small._resolve_noise_model() is None


def test_resolve_noise_model_builds_from_registered_errors(small):
    small.set_phase_flip_prop(0.1)
    small.set_cx_depolarizing_prop(0.1)

    model = small._resolve_noise_model()

    assert isinstance(model, NoiseModel)
    assert len(model.to_dict()["errors"]) == 2


def test_resolve_noise_model_prefers_a_complete_model(monkeypatch):
    copied = []
    real_deepcopy = circuit_module.deepcopy

    def spy(obj, *args, **kwargs):
        copied.append(obj)
        return real_deepcopy(obj, *args, **kwargs)

    monkeypatch.setattr(circuit_module, "deepcopy", spy)

    circuit = Circuit(2, IsingModel(1, 2))
    circuit.set_phase_flip_prop(0.1)
    supplied = NoiseModel()
    circuit.add_noise_model(supplied)

    resolved = circuit._resolve_noise_model()

    # The complete model wins over the registered individual errors, and it is
    # copied so that a later run cannot pile noise on top of noise.
    assert resolved is not supplied
    assert copied == [supplied]


def test_run_adiabatic_applies_the_supplied_noise_model():
    """
    Regression: run_adiabatic() always built a fresh, empty NoiseModel() from the
    set_*_prop() registry and never looked at self.noise_model, so a model passed
    to add_noise_model() was silently inert - a 50 % depolarizing channel gave
    exactly the same energy as no noise at all.
    """
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.5, 1), ["ry"])

    with_model = adiabatic_energy(lambda circuit: circuit.add_noise_model(model))
    without_model = adiabatic_energy(lambda circuit: setattr(circuit, "noise", True))

    assert with_model != pytest.approx(without_model)
    # Depolarizing noise drives the state towards the maximally mixed one, whose
    # energy is larger than the ground-state energy.
    assert with_model > without_model


def test_run_adiabatic_applies_registered_errors():
    noisy = adiabatic_energy(lambda circuit: circuit.set_phase_flip_prop(0.3))
    ideal = adiabatic_energy(lambda circuit: None)

    assert noisy != pytest.approx(ideal)


# --- adiabatic runner --------------------------------------------------------

def test_noiseless_run_returns_one_statevector(small):
    states, energy = small.run_adiabatic(steps=6, T=2.0)

    assert isinstance(states, list)
    assert len(states) == 1
    assert isinstance(states[0], Statevector)
    assert states[0].num_qubits == 2


def test_noiseless_energy_is_the_expectation_value(small):
    states, energy = small.run_adiabatic(steps=6, T=2.0)
    expected = float(np.real(states[0].expectation_value(small.isingModel.Hamiltonian)))

    assert float(np.real(energy)) == pytest.approx(expected)


def test_adiabatic_evolution_reaches_the_ground_state():
    """
    For a slow enough sweep the final state of H(J=1, h=0, g->0) is the GHZ
    state, whose energy is E0 = -1 for a two-site chain.
    """
    circuit = Circuit(2, IsingModel(1, 2, J=1.0, h=0.0, g=0.0))

    states, energy = circuit.run_adiabatic(steps=200, T=10.0)

    assert float(np.real(energy)) == pytest.approx(circuit.get_linalg_E0(), abs=0.05)
    assert abs(states[0].data[0] + states[0].data[-1]) ** 2 / 2 > 0.9


def test_g_initial_changes_the_sweep(small):
    _, slow = small.run_adiabatic(g_initial=2.0, steps=10, T=2.0)
    _, fast = small.run_adiabatic(g_initial=0.5, steps=10, T=2.0)

    assert float(np.real(slow)) != pytest.approx(float(np.real(fast)))


def test_custom_initial_parameters_are_used(small):
    theta = np.zeros(small.circuit.num_parameters)

    _, energy = small.run_adiabatic(theta_initial=theta, steps=4, T=1.0)

    assert np.isfinite(float(np.real(energy)))


def test_custom_initial_parameters_must_have_the_right_size(small):
    with pytest.raises(ValueError, match="Incorrect number of initial parameters"):
        small.run_adiabatic(theta_initial=np.zeros(3))


def test_unknown_initial_state_name_raises(small):
    with pytest.raises(ValueError, match="Unknown initial state"):
        small.run_adiabatic(theta_initial="GHZ")


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": -5}, {"n_shots": 0}])
def test_positive_step_and_shot_counts_are_required(small, kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        small.run_adiabatic(T=1.0, **kwargs)


def test_noisy_run_returns_one_state_per_trajectory():
    circuit = Circuit(2, IsingModel(1, 2, J=1.0, h=0.0, g=0.01))
    circuit.set_phase_flip_prop(0.1)

    states, energy = circuit.run_adiabatic(steps=3, T=1.0, n_shots=5)

    assert len(states) == 5
    assert all(isinstance(state, Statevector) for state in states)
    assert np.isfinite(float(np.real(energy)))


# --- exact ground state ------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3, 4])
def test_linalg_E0_matches_dense_diagonalisation(N):
    circuit = Circuit(N, IsingModel(1, N, J=1.0, h=0.3, g=0.7))
    dense = np.linalg.eigvalsh(circuit.isingModel.Hamiltonian.to_matrix())

    assert circuit.get_linalg_E0() == pytest.approx(float(dense[0]))


def test_linalg_E0_is_deterministic(small):
    assert small.get_linalg_E0() == small.get_linalg_E0()


def test_linalg_E0_uses_the_sparse_solver_above_the_limit(small, monkeypatch):
    """
    Above EXACT_DIAGONALIZATION_LIMIT the Lanczos path is used; check it agrees
    with the dense result on a system small enough for either.
    """
    dense = small.get_linalg_E0()

    monkeypatch.setattr(circuit_module, "EXACT_DIAGONALIZATION_LIMIT", 1)

    assert small.get_linalg_E0() == pytest.approx(dense, abs=1e-8)


def test_documented_default_ground_state_energy():
    """
    The 2D four-site default system, the value the repository quoted as
    E0 = -4.0001000037499. It now comes from the Hamiltonian instead of a
    literal, and has to keep matching.
    """
    circuit = Circuit(4, IsingModel(2, 4, J=1.0, h=0.0, g=0.01))

    assert circuit.get_linalg_E0() == pytest.approx(-4.0001000037499, rel=1e-9)


def test_exact_limit_is_a_sane_power_of_two():
    assert EXACT_DIAGONALIZATION_LIMIT == 1024


# --- parameter updates -------------------------------------------------------

def test_update_ising_parameters_rebuilds_the_hamiltonian(small):
    before = small.isingModel.Hamiltonian.to_matrix()

    small.update_ising_parameters(h=0.7)

    after = small.isingModel.Hamiltonian.to_matrix()

    assert not np.allclose(before, after)
    assert small.isingModel.nearest_neighbor is True


def test_update_ising_parameters_leaves_untouched_fields_alone(small):
    small.update_ising_parameters(J=2.5)

    assert small.isingModel.J(0, 1) == 2.5


def test_update_ising_parameters_with_a_callable_J_clears_nearest_neighbour(small):
    small.update_ising_parameters(J=lambda i, j: 0.25)

    assert small.isingModel.nearest_neighbor is False
    assert small.isingModel.J(0, 1) == 0.25


def test_update_ising_parameters_updates_the_circuit_energy(small):
    before = small.get_linalg_E0()

    small.update_ising_parameters(g=2.0)

    assert small.get_linalg_E0() != pytest.approx(before)


# --- drawing -----------------------------------------------------------------

def test_display_circuit_writes_a_file(tmp_path):
    circuit = Circuit(3, IsingModel(1, 3, J=1.0, h=0.0, g=0.7))
    target = tmp_path / "nested" / "circuit.png"

    circuit.display_circuit(save_path=target)

    assert target.is_file()
    assert target.stat().st_size > 0


def test_display_circuit_closes_the_figure(tmp_path):
    import matplotlib.pyplot as plt

    circuit = Circuit(2, IsingModel(1, 2))
    circuit.display_circuit(save_path=tmp_path / "c.png")

    assert plt.get_fignums() == []


# --- backend-derived noise ---------------------------------------------------

def test_run_with_a_backend_uses_calibration_noise():
    from parameters import FakeSherbrooke

    circuit = Circuit(2, IsingModel(1, 2, J=1.0, h=0.0, g=0.5))

    result = circuit.run_VQE(maxiter=4, backend=FakeSherbrooke())

    assert circuit.noise is True
    assert np.isfinite(result.fun)


def test_run_adiabatic_with_a_backend_returns_trajectories():
    from parameters import FakeSherbrooke

    circuit = Circuit(2, IsingModel(1, 2, J=1.0, h=0.0, g=0.01))

    states, energy = circuit.run_adiabatic(steps=3, T=1.0, n_shots=3, backend=FakeSherbrooke())

    assert len(states) == 3
    assert np.isfinite(float(np.real(energy)))
