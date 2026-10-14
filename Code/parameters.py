from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from qiskit.circuit import Gate


"""
Example calibration parameters from IBM's offline FakeSherbrooke snapshot.
"""

# Fake backend: hardware specifications and historical calibration data.
BACKEND = FakeSherbrooke()

# Approximate noise model derived from the backend.
NOISE_MODEL = NoiseModel.from_backend(BACKEND)

# Example: inspect one physical qubit.
QUBIT = 0
properties = BACKEND.qubit_properties(QUBIT)
T1 = properties.t1  # Energy relaxation time (s)
T2 = properties.t2  # Coherence time (s)
FREQUENCY = properties.frequency  # Hz

T_PHI = 1 / (1 / T2 - 1 / (2 * T1))

AVAILABLE_OPERATIONS = BACKEND.operation_names


def operation_is_gate(name: str) -> bool:
    """
    True when ``name`` is an actual gate rather than an instruction such as
    "delay", "reset", "measure" or a control-flow block.
    """
    return isinstance(BACKEND.target.operation_from_name(name), Gate)


# Filtering out e.g. "delay", "for_loop":
BASIS_GATES = [
    name for name in AVAILABLE_OPERATIONS
    if operation_is_gate(name)
]


def gates_by_arity(num_qubits: int) -> list[str]:
    """
    Native gates of this backend that act on exactly ``num_qubits`` qubits.

    Derived from the backend's target rather than hard-coded, because the gate
    names differ between backends ("sqrt(X)"/"X"/"Rz" on the newer ones, but
    "sx"/"x"/"rz" on this snapshot).
    """
    return [
        name for name in BASIS_GATES
        if any(qubits is not None and len(qubits) == num_qubits for qubits in BACKEND.target[name])
    ]


def single_qubit_gates() -> list[str]:
    """Native gates acting on a single qubit."""
    return gates_by_arity(1)


def two_qubit_gates() -> list[str]:
    """Native gates acting on two qubits."""
    return gates_by_arity(2)


def print_gate_properties(gate: str, qubits: tuple[int, ...]) -> None:
    """
    Display calibrated duration and error for a native instruction.
    """
    if gate not in BACKEND.target.operation_names:
        return

    instruction = BACKEND.target[gate].get(qubits)
    if instruction is None:
        return

    duration = f"{instruction.duration * 1e9:.1f} ns" if instruction.duration is not None else "unknown"
    error = f"{instruction.error:.6g}" if instruction.error is not None else "unknown"
    print(f"  {gate} {qubits}: duration={duration}, error={error}")


def print_backend_info():
    print(f"Backend: {BACKEND.name} ({type(BACKEND).__name__}, offline snapshot)")
    print(f"Physical qubits: {BACKEND.num_qubits}")
    print(f"Basis gates: {BASIS_GATES}")
    print(f"All supported operations: {AVAILABLE_OPERATIONS}")

    print(f"\nPhysical qubit {QUBIT}:")
    print(f"  T1    = {T1 * 1e6:.2f} us")
    print(f"  T2    = {T2 * 1e6:.2f} us")
    print(f"  T_phi = {T_PHI * 1e6:.2f} us")
    print(f"  Frequency = {FREQUENCY / 1e9:.4f} GHz")

    print("\nSingle-qubit gate and readout properties:")
    for gate in single_qubit_gates():
        print_gate_properties(gate, (QUBIT,))
    print_gate_properties("measure", (QUBIT,))

    # Print an actual connected pair from this backend's native gate set.
    for gate in two_qubit_gates():
        pair = next((qubits for qubits in BACKEND.target[gate]
                     if qubits is not None and len(qubits) == 2), None)
        if pair is not None:
            print("\nExample two-qubit gate:")
            print_gate_properties(gate, pair)
            break


if __name__ == "__main__":
    print_backend_info()
