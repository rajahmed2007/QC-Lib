
# VectorQ: A Pure-Python Quantum Statevector Simulator

VectorQ is a lightweight, dependency-minimal quantum computing simulator written in Python and NumPy. It focuses on a clean, object-oriented API and efficient statevector evolution using tensor-based operations.

This simulator is built from three core components: `Statevector`, `QuantumGate`, and `QuantumCircuit`.

-----

## Features

  * **Object-Oriented Design:** Intuitive API with three core classes: `QuantumCircuit`, `Statevector`, and `QuantumGate`.
  * **Efficient Gate Application:** Uses `np.tensordot` for all gate operations. This avoids the explicit construction of large $2^n \times 2^n$ operator matrices, making it efficient for simulating a moderate number of qubits.
  * **Rigorous Validation:** Built-in runtime checks for:
      * Statevector normalization (`Statevector`).
      * Gate unitarity and correct dimensions (`QuantumGate`).
      * Qubit index and target count matching (`QuantumCircuit`).
  * **Dual Simulation Modes:**
      * **`run()`:** An ideal simulator that returns the final, "perfect" complex statevector.
      * **`sample()`:** A probabilistic simulator that mimics a real quantum computer by running a specified number of "shots" and returning the measurement counts.
  * **Core Gates Included:** Comes with a standard library of 1, 2, and 3-qubit gates, including `H`, `X`, `Y`, `Z`, `S`, `T`, `CNOT`, `CZ`, `SWAP`, and `TOFFOLI` (`CCNOT`).

-----

## Installation

The only dependency is **NumPy**.

```bash
pip install numpy
```

Currently, the project is not packaged. You can use it by cloning the repository and ensuring the `.py` files (`circuit.py`, `statev.py`, `gates.py`) are in your working directory or `PYTHONPATH`.

```bash
git clone https://github.com/your-username/vectorq.git  #<-- Replace with your repo URL
cd vectorq
```

-----

## Quick Start: Creating a Bell State

The following example creates a 2-qubit Bell state $|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}$, runs an ideal simulation to get the final state, and then samples the result 1024 times.

```python
from circuit import QuantumCircuit
from gates import H, CNOT
import numpy as np

# 1. Initialize a 2-qubit circuit
qc = QuantumCircuit(num_qubits=2)

# 2. Build the circuit by adding gates
#    Apply Hadamard to qubit 0
qc.add(H, [0])
#    Apply CNOT with control=0, target=1
qc.add(CNOT, [0, 1])

# 3. Print the circuit's operation list
print(qc)

# 4. Run the ideal simulation to get the final statevector
final_state = qc.run()
print("\nFinal Statevector:")
print(final_state)

# 5. Simulate a real measurement experiment
print("\nMeasurement Samples (1024 shots):")
counts = qc.sample(shots=1024)
print(counts)
```

### Expected Output

```
QuantumCircuit (2 qubits)
--------------------------------------
INITIAL STATE: |00⟩
  Step 0: [H] on qubit(s) [0]
  Step 1: [CNOT] on qubit(s) [0, 1]
--------------------------------------

Final Statevector:
2-qubit state |ψ⟩:
  (+0.707+0.000j) |00⟩
  (+0.000+0.000j) |01⟩
  (+0.000+0.000j) |10⟩
  (+0.707+0.000j) |11⟩

Measurement Samples (1024 shots):
{'00': 508, '11': 516}  (Note: Your counts will be probabilistic)
```

-----

## API Overview

### `QuantumGate`

A `QuantumGate` is a wrapper for a unitary $2^k \times 2^k$ NumPy array. It validates the matrix for unitarity and shape upon creation. All standard gates are pre-defined in `gates.py`.

  * `gate.matrix`: The raw NumPy array.
  * `gate.num_qubits`: The number of qubits ($k$) it acts on.
  * `gate.dagger`: Returns a new `QuantumGate` that is the conjugate transpose.
  * `gate.kron(other_gate)`: Returns the tensor product of two gates.

### `Statevector`

A `Statevector` is a wrapper for a $2^n$ complex NumPy array representing the quantum state $|\psi\rangle$. It enforces normalization at all times.

  * `sv.data`: The raw NumPy array of amplitudes.
  * `sv.num_qubits`: The number of qubits ($n$) in the state.
  * `sv.probabilities`: An array of the measurement probabilities ($|\psi_i|^2$).
  * `sv.apply(gate, targets)`: The core method. Applies a $k$-qubit `gate` to the `targets` list of qubit indices and returns a *new* `Statevector` object.
  * `sv.measure()`: Performs a probabilistic measurement, collapsing the state and returning the (outcome, new\_state) tuple.

### `QuantumCircuit`

This is the primary user-facing class. It manages an ordered list of operations and provides the main execution methods.

  * `qc = QuantumCircuit(num_qubits)`: Creates a new circuit for $n$ qubits.
  * `qc.add(gate, targets)`: Adds a `QuantumGate` operation to the circuit, to be applied on the `targets` list of qubits.
  * `qc.run() -> Statevector`: Executes the full circuit from the $|00...0\rangle$ state and returns the final, ideal `Statevector`. This result is cached; subsequent calls are instantaneous unless the circuit is modified.
  * `qc.sample(shots) -> dict`: Executes the circuit, calculates the final probabilities, and simulates `shots` number of measurements. Returns a dictionary of outcome counts (e.g., `{'01': 500, '10': 500}`).

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.