"""Tests for the FakeSherbrooke calibration data helpers."""

import pytest

from parameters import (
    AVAILABLE_OPERATIONS,
    BACKEND,
    BASIS_GATES,
    FREQUENCY,
    NOISE_MODEL,
    QUBIT,
    T1,
    T2,
    T_PHI,
    operation_is_gate,
    print_backend_info,
    print_gate_properties,
    single_qubit_gates,
    two_qubit_gates,
)


def test_backend_is_the_offline_fake_sherbrooke_snapshot():
    assert BACKEND.name == "fake_sherbrooke"
    assert BACKEND.num_qubits == 127


def test_basis_gates_contain_only_gates():
    assert "delay" not in BASIS_GATES
    assert "reset" not in BASIS_GATES
    assert "measure" not in BASIS_GATES
    assert "for_loop" not in BASIS_GATES
    assert "if_else" not in BASIS_GATES
    assert set(BASIS_GATES) == {"rz", "x", "sx", "id", "ecr"}


def test_operation_is_gate_distinguishes_instructions():
    assert operation_is_gate("sx") is True
    assert operation_is_gate("delay") is False
    assert operation_is_gate("measure") is False


def test_basis_gates_are_a_subset_of_the_available_operations():
    assert set(BASIS_GATES) <= set(AVAILABLE_OPERATIONS)


def test_gate_arity_helpers():
    assert set(single_qubit_gates()) == {"rz", "x", "sx", "id"}
    assert two_qubit_gates() == ["ecr"]


def test_noise_model_is_derived_from_the_backend():
    assert NOISE_MODEL.basis_gates
    assert set(BASIS_GATES) <= set(NOISE_MODEL.basis_gates)


def test_markovian_relaxation_relation_holds():
    assert T_PHI == pytest.approx(1 / (1 / T2 - 1 / (2 * T1)))
    assert 0 < T2 <= 2 * T1


def test_qubit_properties_are_physical():
    assert T1 > 0
    assert T2 > 0
    assert FREQUENCY > 0


def test_print_gate_properties_ignores_unknown_gates(capsys):
    assert print_gate_properties("not-a-gate", (QUBIT,)) is None
    assert capsys.readouterr().out == ""


def test_print_gate_properties_reports_a_known_gate(capsys):
    print_gate_properties("sx", (QUBIT,))

    out = capsys.readouterr().out

    assert "sx (0,)" in out
    assert "duration=" in out
    assert "error=" in out


def test_backend_info_prints_single_qubit_calibration(capsys):
    """
    Regression: the single-qubit section looped over the gate names used by newer
    backends ("sqrt(X)", "X", "Rz"), none of which exist on this snapshot, so it
    printed only the readout line.
    """
    print_backend_info()

    out = capsys.readouterr().out

    assert "sx (0,)" in out
    assert "x (0,)" in out
    assert "measure (0,)" in out


def test_backend_info_prints_a_two_qubit_example(capsys):
    """
    Regression: the two-qubit section looked for "ECR"/"CX"/"CZ", while this
    backend names its entangling gate "ecr", so nothing was printed at all.
    """
    print_backend_info()

    out = capsys.readouterr().out

    assert "Example two-qubit gate:" in out
    assert "ecr (" in out


def test_backend_info_reports_the_backend_identity(capsys):
    print_backend_info()

    out = capsys.readouterr().out

    assert "fake_sherbrooke" in out
    assert "Physical qubits: 127" in out
    assert "Basis gates:" in out
    assert "T1" in out and "T2" in out and "T_phi" in out
