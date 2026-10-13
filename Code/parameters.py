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

# Filtering out e.g. "delay", "for_loop":
BASIS_GATES = []
for name in AVAILABLE_OPERATIONS:
    if isinstance(BACKEND.target.operation_from_name(name), Gate):
        BASIS_GATES.append(name)


def print_gate_properties(gate:str, qubits:tuple[int, ...]) -> None:
    """
    Display calibrated duration and error for a native instruction.
    """
    if gate not in BACKEND.target.operation_names:
        return
    
    instruction = BACKEND.target[gate].get(qubits)
    if instruction is None:
        return

    duration = f"{instruction.duration * 1e9:.1f} ns"
    error = f"{instruction.error:.6g}"
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
    for gate in ("sqrt(X)", "X", "Rz", "measure"):
        print_gate_properties(gate, (QUBIT,))

    # Print an actual connected pair from this backend's native gate set.
    for gate in ("ECR", "CX", "CZ"):
        if gate in BACKEND.target.operation_names:
            pair = next((qubits for qubits in BACKEND.target[gate]
                         if qubits is not None and len(qubits) == 2), None)
            if pair is not None:
                print("\nExample two-qubit gate:")
                print_gate_properties(gate, pair)
                break

if __name__ == "__main__":
    print_backend_info()
