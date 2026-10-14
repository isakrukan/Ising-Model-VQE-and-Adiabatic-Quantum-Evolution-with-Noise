import math
import operator

from qiskit.quantum_info import SparsePauliOp


class IsingModel:

    def __init__(self, dimension: int, N: int, J=1.0, h=0.0, g=0.7, shape=None):
        r"""
        Ising model in one or two dimensions, with open boundaries.

        H = -sum_{i<j} J_ij Z_i Z_j
            -sum_i h_i Z_i
            -sum_i g_i X_i

        Parameters:
            dimension:  1 or 2
            N:          Number of spins
            J:          Coupling strength (number or callable J(i,j))
            h:          Longitudinal field (number or callable h(i))
            g:          Transverse field (number or callable g(i))
            shape:      (rows, cols) for a 2D lattice. If None, the most
                        square-like factorisation of N is used
                        (N = 10 -> (2, 5), N = 9 -> (3, 3), prime N -> (1, N)).

        A scalar J gives nearest-neighbor interactions.
        A callable J(i,j) allows arbitrary interactions.
        """
        if dimension not in (1, 2):
            raise ValueError("Dimension higher than 2 is not implemented!")

        if not isinstance(N, int) or isinstance(N, bool) or N < 1:
            raise ValueError("N must be a positive integer.")

        self.dimension = dimension
        self.N = N

        # Converting J, h, g to callables:
        self.nearest_neighbor = not callable(J)

        self.J = self._as_callable(J, "J", pair=True)
        self.h = self._as_callable(h, "h")
        self.g = self._as_callable(g, "g")

        # Determine lattice dimensions
        if dimension == 2:
            if shape is None:
                shape = self.best_lattice_shape(N)
            self.shape = self._validate_shape(shape, N)
        else:
            self.shape = None

        # Construct Hamiltonian
        if dimension == 1:
            self._1D_Hamiltonian()
        else:
            self._2D_Hamiltonian()

    @staticmethod
    def best_lattice_shape(N: int) -> tuple[int, int]:
        r"""
        Most square-like (rows, cols) factorisation of N, with rows <= cols.

        ``(isqrt(N), isqrt(N))`` is only a valid lattice when N is a perfect
        square: for N = 10 it gives (3, 3), i.e. nine sites for ten spins, so
        the row/column indexing in ``_2D_Hamiltonian`` placed the last spin
        outside the nominal lattice and the nearest-neighbour set no longer
        described any real 2D lattice. Non-squares now get a genuine
        factorisation, and a prime N degrades to the 1 x N chain, which is the
        honest 2D description of an N-site open chain.
        """
        rows = math.isqrt(N)
        while rows > 1 and N % rows != 0:
            rows -= 1
        return rows, N // rows

    @staticmethod
    def _validate_shape(shape, N: int) -> tuple[int, int]:
        """
        Returns ``shape`` as a (rows, cols) tuple of ints, or raises ValueError.
        """
        message = (
            "For 2D, shape=(rows, cols) must have positive integer dimensions "
            f"with rows*cols=N; got shape={shape!r} and N={N}."
        )

        try:
            rows, cols = shape
        except (TypeError, ValueError):
            raise ValueError(message) from None

        try:
            rows = operator.index(rows)
            cols = operator.index(cols)
        except TypeError:
            raise ValueError(message) from None

        if rows <= 0 or cols <= 0 or rows * cols != N:
            raise ValueError(message)

        return rows, cols

    @staticmethod
    def _as_callable(value, name, pair=False):
        if callable(value):
            return value
        if isinstance(value, (int, float)):
            if pair:
                return lambda i, j: value
            return lambda i: value
        raise TypeError(f"{name} must be numeric or callable.")

    def _1D_Hamiltonian(self):
        """
        Constructs the Hamiltonian for the 1D transverse Ising model.
        """
        terms = []

        for i in range(self.N):
            for j in range(i + 1, self.N):

                if self.nearest_neighbor and abs(i - j) != 1:
                    continue

                terms.append(("ZZ", [i, j], -self.J(i, j)))

        for i in range(self.N):
            terms.append(("X", [i], -self.g(i)))
            terms.append(("Z", [i], -self.h(i)))

        self.Hamiltonian = SparsePauliOp.from_sparse_list(terms, num_qubits=self.N)

    def _2D_Hamiltonian(self):
        """
        Constructs the Hamiltonian for the 2D transverse Ising model.
        """
        terms = []
        # shape is always set for a 2D model
        assert self.shape is not None
        cols = self.shape[1]

        for i in range(self.N):
            for j in range(i + 1, self.N):

                # Coordinates of spins i and j
                ri, ci = divmod(i, cols)
                rj, cj = divmod(j, cols)

                # Nearest-neighbor condition
                if self.nearest_neighbor and abs(ri - rj) + abs(ci - cj) != 1:
                    continue

                terms.append(("ZZ", [i, j], -self.J(i, j)))

        for i in range(self.N):
            terms.append(("X", [i], -self.g(i)))
            terms.append(("Z", [i], -self.h(i)))

        self.Hamiltonian = SparsePauliOp.from_sparse_list(terms, num_qubits=self.N)

    def __repr__(self) -> str:
        return (
            f"IsingModel(dimension={self.dimension}, N={self.N}, shape={self.shape}, "
            f"nearest_neighbor={self.nearest_neighbor})"
        )
