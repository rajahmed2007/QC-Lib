import numpy as np
import math
from typing import TYPE_CHECKING, Tuple, List

# Ye circular import error nhi hoga 
if TYPE_CHECKING:
    from .gates import QuantumGate  


class Statevector:
    """
    Represents a quantum statevector |ψ⟩ in the computational basis.

    This class is a wrapper around a 1D NumPy array. It enforces that the state is always complex, 1-dimensional, has a length that is a power of 2, and is normalized to 1.
    
    Defined in a way that the class doesn't fucking become an array, it stores an array and wraps it.
    
    """

    # ----------------------------------------------------------------------
    # ---  Dunders (Idk who the fuck is dunder..double under,...anyways )
    # ----------------------------------------------------------------------

    def __init__(self, data: np.ndarray):
        """
        Initializes the statevector. It assumes the input is normalized.

        Args:
            data: A 1D NumPy array of complex numbers.
                  Must have a length == 2**n and be normalized.
        
        Raises:
            ValueError: If data is not 1D, not complex,
                        shape is not 2**n, or not normalized.
        """
        # --- 1. Input Validation ---
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=complex)
            
        if data.ndim != 1:
            raise ValueError(f"Data must be a 1D array (got {data.ndim} dimensions).")
            
        if not np.iscomplexobj(data):
            # will cast to complex , (cast, not caste 😏) but assumes its numerical, dont input something like string, dictionary or your ex's address
            try:
                data = data.astype(complex)
            except (TypeError, ValueError):
                raise ValueError("Data must be complex or convertible to complex.")

        # --- 2. Qubit / Shape Validation ---
        n_elements = data.shape[0]
        if n_elements == 0:
            raise ValueError("Cannot create an empty statevector.")
            
        num_qubits = math.log2(n_elements)
        if not num_qubits.is_integer():
            raise ValueError(
                f"Data shape must be a power of 2 (got {n_elements})."
                # Aise nhi hoga QC. "Kyu nhi ho rahi padhai?"
            )
        
        self.num_qubits: int = int(num_qubits)

        # --- 3. Normalization Validation ---
        norm = np.sum(data.conj() * data).real
        if not math.isclose(norm, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"Statevector is not normalized (norm is {norm:.6f}). "
                "Use `Statevector.from_unnormalized()` to auto-normalize. Or fucking use a pen paper to normalize. Dont let go your BTech degree and pen paper writing habit. You digital ass."
            )
            
        # --- 4. Success ---
        # Finally Fuck
        self.data: np.ndarray = data

    def __repr__(self) -> str:
        """Clear, unambiguous representation for developers."""
        # Well developers doesnt give a fuck tho tbh
        return f"Statevector(n={self.num_qubits}, data={self.data})"

    def __str__(self) -> str:
        """User-friendly string representation (pretty print)."""
        # Another day, another assumption and requirement to babysit users
        s = f"{self.num_qubits}-qubit state |ψ⟩:\n"
        if self.num_qubits > 5:
            s += f"  (Vector dimension {2**self.num_qubits} too large to print)"
            return s
            
        for i, amp in enumerate(self.data):
            if math.isclose(amp, 0):
                continue  # 0 probability = 0 value...very philosophical
            
            # Format basis state as a binary string like |010 \rangle. Completely unnecessary but cute
            basis = format(i, f'0{self.num_qubits}b')
            s += f"  ({amp.real:+.3f}{amp.imag:+.3f}j) |{basis}⟩\n"
        return s.rstrip()  # Remove trailing newline. Thats just tantrums atp

    def __len__(self) -> int:
        """Returns the dimension of the vector space (2**n)."""
        return self.data.shape[0]

    def __eq__(self, other) -> bool:
        """
        Checks if two statevectors are (almost) equal.
        
        This ignores global population.... phase i mean. It checks if the
        fidelity |<self|other>|^2 is 1. (The vector must not be..well...infidel)
        """
        if not isinstance(other, Statevector):
            return False
            
        if self.data.shape != other.data.shape:
            return False
        
        # Calculate the inner product: <self|other>
        # np.vdot conjugates the first argument. Purr-fact.
        inner_product = np.vdot(self.data, other.data)
        
        # Get the fidelity (squared magnitude of the inner product)
        fidelity = (inner_product.conj() * inner_product).real
        
        # Check if fidelity is (close to) 1
        # I will sink those floating fuckers (nice alliteration)
        return math.isclose(fidelity, 1.0, abs_tol=1e-9)
        
    def __getitem__(self, key):
        """Gets the amplitude at a specific index or slice."""
        return self.data[key]

    # ----------------------------------------------------------------------
    # --- Classmethods (Alternate Constructors)
    # ----------------------------------------------------------------------

    @classmethod
    def from_unnormalized(cls, data: np.ndarray) -> 'Statevector': #that 'stateverctor'and not statevector shit is bulshitly pythonic nonsense
        """
        Creates a Statevector from unnormalized data by
        normalizing it first.
        
        Example:
          unnorm_vec = state1 + state2
          new_state = Statevector.from_unnormalized(unnorm_vec)
        """
        norm = np.linalg.norm(data)
        if math.isclose(norm, 0):
            raise ValueError("Cannot create a state from a zero vector.")
        
        # Call the main __init__ with the now-normalized data
        return cls(data / norm)

    # ----------------------------------------------------------------------
    # --- Properties (Computed values)
    # ----------------------------------------------------------------------

    @property
    def norm(self) -> float:
        """Calculates the L2 norm (magnitude) of the statevector."""
        # This should always be 1.0 due to __init__ checks. Redundant shit
        return np.linalg.norm(self.data)

    @property
    def probabilities(self) -> np.ndarray:
        """
        Returns the array of measurement probabilities (|\psi_i|^2)
        for each basis state.
        """
        # |amplitude|^2 = amplitude * conj(amplitude) ....in case the future me forgets inner products (well he wont)
        return (self.data.conj() * self.data).real

    # ----------------------------------------------------------------------
    # --- Core Quantum Operations
    # ----------------------------------------------------------------------

    def show(self):
        print(self)
    
    
    
    def apply(self, gate: 'QuantumGate', targets: List[int]) -> 'Statevector':
        """
        Applies a quantum gate to specific target qubits.
        
        It uses tensor magic to apply a
        small gate to a large statevector without building the
        giant (2**n x 2**n) matrix.
        
        |ψ'⟩ = G |ψ⟩
        
        Args:
            gate: A QuantumGate object (e.g., H, CNOT).
            targets: A list of qubit indices (e.g., [0] for H, [0, 1] for CNOT).
        
        Returns:
            A new Statevector object representing the resulting state.
        """
        # --- 1. Validation. No 8th grade shit here. ---
        k = gate.num_qubits
        n = self.num_qubits

        if k != len(targets):
            raise ValueError(
                f"Gate qubit count ({k}) does not match "
                f"number of targets provided ({len(targets)})."
                
                # KYU NHI HO PA RAHI CODING? Padhai Karni hai! Coding karni hai! Paisa kamana hai! Mar ja
            )
        if len(set(targets)) != len(targets):
            raise ValueError(f"Target qubit list contains duplicates: {targets}")
        if any(t < 0 or t >= n for t in targets):
            raise ValueError(
                f"Target qubit index out of range (0 to {n-1}): {targets}"
            )

        # --- 2. The Tensor Tartarus ---
        # I am rightfully assuming i am expert in alliteration atp
        
        # Reshape the (2**n,) state vector into a tensor of shape (2, 2, ..., 2)
        # This separates each qubit into its own dimension (axis)
        sv_tensor = self.data.reshape([2] * n)
        
        # Reshape the gate matrix (2**k, 2**k) into a tensor
        # It has 'k' output axes and 'k' input axes
        # Shape: (2, ..., 2,  2, ..., 2)
        #         <- k out -> <- k in ->
        gate_tensor = gate.matrix.reshape([2] * (2 * k))
        
        # The axes of the gate tensor we contract (the 'in' axes)
        gate_in_axes = list(range(k, 2 * k))
        
        # This is the black magic. My laptop is black.
        # It contracts the gate's 'in' axes with the statevector's 'target' axes.
        # The result has shape: (2, ..., 2,    2, ..., 2)
        #                       <- k out ->  <- n-k remaining ->
        new_tensor = np.tensordot(
            gate_tensor, sv_tensor, axes=(gate_in_axes, targets)
        )
        
        # --- 3. Un-shuffle the axes ---
        # The new tensor's axes are messed up. The new (output) axes
        # are at the *front*. We need to move them back to their
        # original 'target' positions to restore the qubit order.
        
        # Find all axes that *weren't* targeted
        remaining_axes = [i for i in range(n) if i not in targets]
        
        # Create the permutation map to put axes back in order
        # Example: n=4, targets=[1, 3]
        # 'new_tensor' axes are (new_1, new_3, old_0, old_2)
        # We want order (old_0, new_1, old_2, new_3)
        # The permutation will be [2, 0, 3, 1]
        permutation = [0] * n
        for i, target_idx in enumerate(targets):
            permutation[target_idx] = i      # Place new 'k' axes
        for i, rem_idx in enumerate(remaining_axes):
            permutation[rem_idx] = i + k     # Place remaining 'n-k' axes

        # np.transpose shuffles the axes back to the correct order
        transposed_tensor = np.transpose(new_tensor, permutation)
        
        # --- 4. Reshape back to a 1D vector and return ---
        # This should be normalized because the gate is unitary.
        # The __init__ will check and scream if it isn't.
        new_data = transposed_tensor.reshape(2**n)
        return Statevector(new_data)

    def kron(self, other: 'Statevector') -> 'Statevector':
        """
        Combines this state with another using the tensor (Kronecker) product.
        
        Usage: |ψ_c⟩ = |ψ_a⟩ ⊗ |ψ_b⟩  (self.kron(other))
        
        This places the 'other' state as the "less significant" qubit(s).
        E.g., |0⟩.kron(|1⟩) -> |01⟩
        
        Args:
            other: The other Statevector to combine with.
        
        Returns:
            A new, larger Statevector representing the combined system.
        """
        # Ahh another day of writing a function, writing docstring of 3839393 lines and function logic is one line return
        new_data = np.kron(self.data, other.data)
        return Statevector(new_data)

    # ----------------------------------------------------------------------
    # --- Measurement
    # ----------------------------------------------------------------------

    def measure(self) -> Tuple[int, 'Statevector']:
        """
        Simulates a measurement in the computational (Z) basis.
        
        This operation is probabilistic and collapses the state.
        
        Returns:
            A tuple containing:
            1. (int): The classical outcome (the decimal index of the
                       measured basis state, e.g., 5 for |101⟩).
            2. (Statevector): The new, collapsed Statevector (e.g., |101⟩).
        """
        probs = self.probabilities
        num_states = 2**self.num_qubits
        # Very pseudo quantum shit
        # 1. Choose a basis state index based on the probabilities
        # We must use `p=probs` as they are the probabilities.
        outcome = np.random.choice(range(num_states), p=probs)
        
        # 2. Create the new collapsed state
        # It's a "one-hot" vector, e.g., [0, 0, 0, 1, 0, 0]...well many other things (not necessarily things) are hot
        new_data = np.zeros(num_states, dtype=complex)
        new_data[outcome] = 1.0 + 0j
        
        # 3. Return the outcome and the new state
        return outcome, Statevector(new_data)

    # ----------------------------------------------------------------------
    # --- Algebraic Operations (Return unnormalized arrays)
    # ----------------------------------------------------------------------
    
    # These operations (add, sub, mul, div) are for superposition
    # and linear combinations. The result is *NOT* a valid,
    # normalized Statevector.
    # We return a raw numpy array. You fucking should and must use
    # `Statevector.from_unnormalized()` to turn it back into
    # a valid state. Absurd but iokay. We know the problem, we have weitten the problm, but i aint solving as i am assuming QC computer aint an infant

    def __add__(self, other: 'Statevector') -> np.ndarray:
        """
        Adds two statevectors, returning an *unnormalized* numpy array.
        
        To create a valid state from this, you must normalize the result.
        Example:
          unnormalized = state1 + state2
          new_state = Statevector.from_unnormalized(unnormalized)
        """
        if self.data.shape != other.data.shape:
             raise ValueError("Cannot add Statevectors with different shapes.")
        return self.data + other.data

    def __sub__(self, other: 'Statevector') -> np.ndarray:
        """Subtracts two statevectors, returning an *unnormalized* numpy array."""
        if self.data.shape != other.data.shape:
             raise ValueError("Cannot subtract Statevectors with different shapes.")
        return self.data - other.data

    def __mul__(self, scalar) -> np.ndarray:
        """Multiplies by a scalar, returning an *unnormalized* numpy array."""
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented  # Allows Python to try __rmul__
        return self.data * scalar
    
    def __rmul__(self, scalar) -> np.ndarray:
        """Multiplies by a scalar (e.g., 0.5 * state)."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar) -> np.ndarray:
        """Divides by a scalar, returning an *unnormalized* numpy array."""
        if not isinstance(scalar, (int, float, complex)):
            return NotImplemented
        if math.isclose(abs(scalar), 0):
            raise ZeroDivisionError("Cannot divide statevector by zero.")
        return self.data / scalar
    
    
    
    