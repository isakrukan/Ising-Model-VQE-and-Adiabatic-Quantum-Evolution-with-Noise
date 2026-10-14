"""Tests for the Ising Hamiltonian construction."""

import math

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from IsingModel import IsingModel

I2 = np.eye(2)
Z = np.diag([1.0, -1.0])
X = np.array([[0.0, 1.0], [1.0, 0.0]])


def terms(model):
    """
    Hamiltonian as {((qubit, pauli), ...): coefficient} with the support sorted
    by qubit, so tests can talk about bonds independently of Qiskit's endianness.
    """
    out = {}

    for pauli, coeff in zip(model.Hamiltonian.paulis, model.Hamiltonian.coeffs):
        label = pauli.to_label()  # qubit n-1 ... qubit 0
        ops = {model.N - 1 - k: ch for k, ch in enumerate(label) if ch != "I"}
        out[tuple(sorted(ops.items()))] = complex(coeff)

    return out


def zz_bonds(model):
    """Set of two-qubit Z-Z bonds, as ((i, 'Z'), (j, 'Z')) with i < j."""
    return {k for k in terms(model) if len(k) == 2 and all(ch == "Z" for _, ch in k)}


def expected_bonds(rows, cols):
    """Nearest-neighbour bonds of a rows x cols grid with open boundaries."""
    bonds = set()

    for i in range(rows * cols):
        ri, ci = divmod(i, cols)

        for j in range(i + 1, rows * cols):
            rj, cj = divmod(j, cols)

            if abs(ri - rj) + abs(ci - cj) == 1:
                bonds.add(((i, "Z"), (j, "Z")))

    return bonds


# --- lattice shape selection -------------------------------------------------

@pytest.mark.parametrize("N, shape", [(1, (1, 1)), (4, (2, 2)), (9, (3, 3)), (16, (4, 4)), (36, (6, 6))])
def test_best_lattice_shape_perfect_squares(N, shape):
    assert IsingModel.best_lattice_shape(N) == shape


@pytest.mark.parametrize("N, shape", [(2, (1, 2)), (6, (2, 3)), (8, (2, 4)), (10, (2, 5)), (12, (3, 4)), (18, (3, 6))])
def test_best_lattice_shape_non_squares(N, shape):
    assert IsingModel.best_lattice_shape(N) == shape


@pytest.mark.parametrize("N", [3, 5, 7, 11, 13, 29])
def test_best_lattice_shape_prime_degrades_to_chain(N):
    assert IsingModel.best_lattice_shape(N) == (1, N)


@pytest.mark.parametrize("N", range(1, 41))
def test_best_lattice_shape_is_a_valid_factorisation(N):
    rows, cols = IsingModel.best_lattice_shape(N)

    assert rows * cols == N
    assert 1 <= rows <= cols
    assert rows <= math.isqrt(N) + 1


@pytest.mark.parametrize("N", [4, 9, 10, 12, 15, 16])
def test_default_2d_shape_covers_every_spin(N):
    """
    Regression: the default shape used to be (isqrt(N), isqrt(N)), which for
    N = 10 gave a 3 x 3 lattice (nine sites) indexed with cols = 3 - so spin 9
    landed at a site the lattice does not have.
    """
    model = IsingModel(2, N)

    rows, cols = model.shape

    assert rows * cols == N
    assert zz_bonds(model) == expected_bonds(rows, cols)


def test_default_2d_shape_n10_is_2x5():
    assert IsingModel(2, 10).shape == (2, 5)


def test_1d_shape_is_none():
    assert IsingModel(1, 4).shape is None


def test_shape_is_stored_as_a_tuple():
    model = IsingModel(2, 6, shape=[2, 3])

    assert model.shape == (2, 3)


# --- shape validation --------------------------------------------------------

@pytest.mark.parametrize("shape", [(3, 4), (0, 10), (10, 0), (-2, -5), (2, 6), (1, 2, 3), (2,), ("a", 10), (2.5, 4)])
def test_invalid_explicit_shape_raises(shape):
    with pytest.raises(ValueError, match=r"rows\*cols=N"):
        IsingModel(2, 10, shape=shape)


def test_odd_shape_never_validated_when_dimension_is_1():
    # shape is a 2D concept; a 1D model ignores it entirely
    model = IsingModel(1, 4, shape=(99, 99))

    assert model.shape is None
    assert model.N == 4


# --- argument validation -----------------------------------------------------

@pytest.mark.parametrize("dimension", [0, 3, -1, 5])
def test_dimension_must_be_1_or_2(dimension):
    with pytest.raises(ValueError, match="Dimension"):
        IsingModel(dimension, 4)


@pytest.mark.parametrize("N", [0, -1, 1.5, "4", True])
def test_N_must_be_a_positive_integer(N):
    with pytest.raises(ValueError, match="positive integer"):
        IsingModel(1, N)


@pytest.mark.parametrize("name", ["J", "h", "g"])
def test_non_numeric_parameters_raise_type_error(name):
    with pytest.raises(TypeError, match=name):
        IsingModel(1, 3, **{name: "strong"})


# --- 1D Hamiltonian ----------------------------------------------------------

def test_1d_open_chain_bonds():
    model = IsingModel(1, 5, J=1.0, h=0.3, g=0.7)

    assert zz_bonds(model) == {((i, "Z"), (i + 1, "Z")) for i in range(4)}


def test_1d_coefficients():
    model = IsingModel(1, 5, J=2.0, h=0.3, g=0.7)
    got = terms(model)

    for i in range(4):
        assert got[((i, "Z"), (i + 1, "Z"))] == pytest.approx(-2.0)

    for i in range(5):
        assert got[((i, "Z"),)] == pytest.approx(-0.3)
        assert got[((i, "X"),)] == pytest.approx(-0.7)


def test_1d_no_periodic_wrapping():
    model = IsingModel(1, 4)

    assert ((0, "Z"), (3, "Z")) not in zz_bonds(model)
    assert len(zz_bonds(model)) == 3


def test_1d_callable_J_couples_every_pair():
    model = IsingModel(1, 4, J=lambda i, j: 0.5, h=0.3, g=0.7)
    got = terms(model)

    assert model.nearest_neighbor is False
    assert len(zz_bonds(model)) == 6  # complete graph on 4 sites
    assert all(coeff == pytest.approx(-0.5) for key, coeff in got.items() if len(key) == 2)


def test_nearest_neighbour_flag_tracks_callability():
    assert IsingModel(1, 4, J=1.0).nearest_neighbor is True
    assert IsingModel(1, 4, J=lambda i, j: 1.0).nearest_neighbor is False


def test_callable_h_and_g_are_used_per_site():
    model = IsingModel(1, 3, J=1.0, h=lambda i: 0.1 * (i + 1), g=lambda i: 0.2 * (i + 1))
    got = terms(model)

    for i in range(3):
        assert got[((i, "Z"),)] == pytest.approx(-0.1 * (i + 1))
        assert got[((i, "X"),)] == pytest.approx(-0.2 * (i + 1))


# --- 2D Hamiltonian ----------------------------------------------------------

@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (2, 5), (3, 4), (1, 6)])
def test_2d_bond_set_matches_the_grid(shape):
    model = IsingModel(2, shape[0] * shape[1], shape=shape)

    assert zz_bonds(model) == expected_bonds(*shape)


@pytest.mark.parametrize("shape", [(3, 3), (2, 5), (4, 4), (2, 6)])
def test_2d_bond_count(shape):
    rows, cols = shape
    model = IsingModel(2, rows * cols, shape=shape)

    assert len(zz_bonds(model)) == rows * (cols - 1) + cols * (rows - 1)


def test_2d_diagonal_is_not_a_neighbour():
    model = IsingModel(2, 4, shape=(2, 2))

    assert ((0, "Z"), (3, "Z")) not in zz_bonds(model)


def test_2d_no_periodic_wrapping():
    model = IsingModel(2, 6, shape=(2, 3))

    # 0 and 2 are on the same row but at opposite ends of it
    assert ((0, "Z"), (2, "Z")) not in zz_bonds(model)


# --- matrix equivalence ------------------------------------------------------

def test_1d_two_site_hamiltonian_matrix():
    """
    H = -J Z0 Z1 - h (Z0 + Z1) - g (X0 + X1), checked against a hand-built
    matrix. Qiskit is little-endian, so qubit 0 is the rightmost tensor factor.
    """
    J, h, g = 1.3, 0.4, 0.7

    model = IsingModel(1, 2, J=J, h=h, g=g)

    expected = (
        -J * np.kron(Z, Z)
        - h * (np.kron(Z, I2) + np.kron(I2, Z))
        - g * (np.kron(X, I2) + np.kron(I2, X))
    )

    assert np.allclose(model.Hamiltonian.to_matrix(), expected)


def test_2d_two_by_two_hamiltonian_matrix():
    """
    A 2 x 2 grid has four horizontal/vertical bonds and no diagonal ones, so it
    differs from the 1D four-site chain.
    """
    J, h, g = 1.0, 0.0, 0.5

    model = IsingModel(2, 4, J=J, h=h, g=g, shape=(2, 2))

    p = SparsePauliOp.from_sparse_list(
        [
            ("ZZ", [0, 1], -J),
            ("ZZ", [1, 3], -J),
            ("ZZ", [2, 3], -J),
            ("ZZ", [0, 2], -J),
            ("X", [0], -g),
            ("X", [1], -g),
            ("X", [2], -g),
            ("X", [3], -g),
        ],
        num_qubits=4,
    ).simplify()

    assert np.allclose(model.Hamiltonian.to_matrix(), p.to_matrix())


def test_zero_field_terms_are_not_duplicated():
    model = IsingModel(1, 3, J=1.0, h=0.0, g=0.0)

    labels = [p.to_label() for p in model.Hamiltonian.paulis]

    assert len(labels) == len(set(labels))


def test_hamiltonian_is_a_sparse_pauli_op():
    model = IsingModel(1, 3)

    assert isinstance(model.Hamiltonian, SparsePauliOp)
    assert model.Hamiltonian.num_qubits == 3


def test_repr_mentions_the_lattice():
    text = repr(IsingModel(2, 6))

    assert "shape=(2, 3)" in text
    assert "N=6" in text
