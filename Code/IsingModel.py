from qiskit.quantum_info import SparsePauliOp
import math

class TransverseIsingModel:

    def __init__(self, dimension:int, N:int, J=1.0, h=0.0, g=0.7, shape=None):
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
            shape:      (rows, cols) for a 2D lattice

        A scalar J gives nearest-neighbor interactions.
        A callable J(i,j) allows arbitrary interactions.
        """
        if dimension not in (1, 2):
            raise ValueError("Dimension higher than 2 is not implemented!")
        
        if not isinstance(N, int) or N < 1:
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
                side = math.isqrt(N)
                shape = (side, side)

            elif (len(shape) != 2
                or not all(isinstance(x, int) and x > 0 for x in shape)
                or shape[0] * shape[1] != N):

                raise ValueError("For 2D, shape=(rows, cols) must have positive integer dimensions with rows*cols=N.")

        self.shape = shape

        # Construct Hamiltonian
        if dimension == 1:
            self._1D_Hamiltonian()
        else:
            self._2D_Hamiltonian()

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
        rows, cols = self.shape

        for i in range(self.N):
            for j in range(i + 1, self.N):

                # Coordinates of spins i and j
                ri, ci = divmod(i, cols)
                rj, cj = divmod(j, cols)

                # Nearest-neighbor condition
                if self.nearest_neighbor and abs(ri-rj) + abs(ci-cj) != 1:
                    continue

                terms.append(("ZZ", [i, j], -self.J(i, j)))

        for i in range(self.N):
            terms.append(("X", [i], -self.g(i)))
            terms.append(("Z", [i], -self.h(i)))

        self.Hamiltonian = SparsePauliOp.from_sparse_list(terms, num_qubits=self.N)