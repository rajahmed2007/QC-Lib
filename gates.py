import numpy as np
import math
from typing import TYPE_CHECKING, List, Tuple

# No circular import possible here.
# The Gate doesn't know what a Statevector is.
# The Statevector is the one who has to deal with the Gate.
# This is a one-way, abusive relationship. Typical patriarchial fmailies.


class QuantumGate:
    """
    Represents a unitary quantum gate (a U matrix).

    This class is a wrapper around a 2D NumPy array. It enforces
    that the gate is always complex, square, has dimensions (2**k, 2**k),
    and is fucking unitary.
    """

    # ----------------------------------------------------------------------
    # --- Dunders (Still sounds stupid, still what they're called)
    # ----------------------------------------------------------------------

    def __init__(self, matrix, name: str):
        """
        Initializes the quantum gate.

        Args:
            matrix: A 2D (2**k, 2**k) NumPy array or list of lists.
            name: A human-readable name for the gate (e.g., "H", "CNOT", "MyAss").

        Raises:
            ValueError: If the matrix is not 2D, not square,
                        not a power-of-2 dimension, or not unitary.
        """
        
        # --- 1. Input Validation (My non existent god, the validation...) ---
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, dtype=complex)

        if not np.iscomplexobj(matrix):
            # If you give me [0, 1]... I'll assume you mean [0.+0.j, 1.+0.j]
            # I'm a good guy. Sometimes.Default : Fuck yourself (If we define fuck is an act of n individuals, thats n=1 base case)
            try:
                matrix = matrix.astype(complex)
            except (TypeError, ValueError):
                raise ValueError("Gate matrix must be complex or convertible to complex.") #Dont put your ex's address
        
        if matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D (got {matrix.ndim} dimensions).") #Ahh bro dont generalize tensor sht yet (d = 0, scalar, d = 1, vector, d = 2 matrix, d = 3 cube,...etc NO )
            
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"This isn't a square dance. Matrix must be square (got {matrix.shape})." #Bruh you aint applying rectangular matrix...forget that XOR XNOR shit
            )

        # --- 2. Qubit / Shape Validation ---
        n_dim = matrix.shape[0]
        if n_dim < 2:
            raise ValueError(
                f"What is this, a gate for my cat? I mean what would u do with a 0 qubit state."
                f"Must be at least 2x2 (got {n_dim}x{n_dim})."
            )

        num_qubits = math.log2(n_dim)
        if not num_qubits.is_integer():
            raise ValueError(
                f"Gate dimension must be a power of 2 (got {n_dim}x{n_dim})." #Again bruteforcing...just like u did with ur love life
            )
            
        self.num_qubits: int = int(num_qubits)

        # --- 3. Unitarity Validation (The REAL test) ---
        # U must be unitary, meaning U_dagger * U = I (dagger is bra i mean harmitian conjugate....dont think murder weapon or somethjig...python (both the snake and the lanuage) is enough to murder)
        identity = np.eye(n_dim, dtype=complex)
        u_dagger_u = matrix.conj().T @ matrix
        
        if not np.allclose(u_dagger_u, identity, atol=1e-9):
            raise ValueError(
                f"Matrix for gate '{name}' is not unitary. "
                f"U†U != I. This thing is leaking probability. Fix it." #Rule no 1 : Leakage is bad...anywhere
            )
            
        # --- 4. Success ---
        # Fine. You can exist. i am god
        self.matrix: np.ndarray = matrix
        self.name: str = name

    def __repr__(self) -> str:
        """For the dev who's too busy to read and still won't honsetly give a slanting fuck __str__."""
        return f"QuantumGate(name='{self.name}', qubits={self.num_qubits})"

    def __str__(self) -> str:
        """Because `print(gate.matrix)` is for peasants. Classism ahh code."""
        s = f"Gate: {self.name} ({self.num_qubits}-qubit)\n"
        if self.num_qubits > 4: # A 16x16 is too much to ask
            s += " (Matrix too large to print)"
            return s
            
        # Pretty print the matrix. thats the only pretty thing u have
        for row in self.matrix:
            s += " ["
            for val in row:
                s += f"{val.real:+.2f}{val.imag:+.2f}j  "
            s = s.rstrip() + "]\n"
        return s.rstrip()

    # ----------------------------------------------------------------------
    # --- Properties
    # ----------------------------------------------------------------------

    @property
    def dagger(self) -> 'QuantumGate':
        """
        Returns the adjoint (conjugate transpose) of this gate.
        Un-does whatever the hell this gate did. I wish i had a dagger in life.
        
        Returns:
            A new QuantumGate object (U†).
        """
        # The name† is just *chef's kiss*
        return QuantumGate(self.matrix.conj().T, f"{self.name}†")

    # ----------------------------------------------------------------------
    # --- Gate Operations
    # ----------------------------------------------------------------------

    def kron(self, other: 'QuantumGate') -> 'QuantumGate':
        """
        Combines this gate with another using the tensor (Kronecker) product.
        
        Usage: G_new = G_a.kron(G_b)  (G_a ⊗ G_b)
        
        This is for building bigger, badder gates. Baddies!
        
        Args:
            other: The other QuantumGate to combine with.
            
        Returns:
            A new, larger QuantumGate.
        """
        # For when you need to tensor-product your problems into bigger problems
        new_matrix = np.kron(self.matrix, other.matrix)
        new_name = f"({self.name} ⊗ {other.name})"
        
        # The new gate will be re-validated by __init__...
        # which is good, because np.kron is not guaranteed to be unitary
        # if the inputs aren't. (But ours are. We checked.)
        return QuantumGate(new_matrix, new_name)


# ======================================================================
# --- STANDARD GATES
# ---
# --- 
# ======================================================================

# --- 1-Qubit Gates ----------------------------------------------------

# Pauli-X (NOT) Gate
# |0⟩ -> |1⟩
# |1⟩ -> |0⟩
PAULI_X = QuantumGate(
    matrix=[
        [0, 1],
        [1, 0]
    ],
    name="X"
)

# Pauli-Y gate
PAULI_Y = QuantumGate(
    [
        [0, -1j]
        ,[1j,0]
    ],
    "Y"
)

# Pauli-Z
PAULI_Z = QuantumGate(
    [
        [1,0],
        [0,-1]
    ],
    "Z"
)

# Hadamard
H = QuantumGate(
    [
        [np.sqrt(0.5), np.sqrt(0.5)],
        [np.sqrt(0.5), - np.sqrt(0.5)]
    ], "H"
)


# Sigma_V

def Sigma_v(v : list) -> QuantumGate:
    X =[ 
        [v[2], complex(v[0], v[1])],
        [complex(v[0], - v[1]),  -v[2]]
]
    return QuantumGate(X, "S_V")



# S_gate

S = QuantumGate(
    [
        [1,0],
        [0,1j]
    ], "S"
)

#pi/8 gate

T = QuantumGate(
    [
        [1,0],
        [0, np.pow(np.e, 1j * np.pi/4 )]
    ], "T"
)

# Phase

def phase_gate(phi : float) -> QuantumGate:
    X = [
        [1,0],
        [0, np.pow(np.e, 1j * phi/4 )]
    ]
    return QuantumGate(X,"P_phi")


# U_3, the god

def U_3_gate(theta, phi, lamb):
    m = np.cos(theta/2)
    n = np.sin(theta/2)
    x = np.pow(np.e, 1j * lamb)
    y = np.pow(np.e, 1j * phi)
    X  = [
        [m, -n*x],
        [y*n, x*y*m]
    ]
    print(X)
    print()
    return QuantumGate(X, "U_3")


# Identity (eat 5 star, do nothing)

I = QuantumGate(np.identity(2),"I")

# --- 2-Qubit States ----------

# CNOT
CNOT = QuantumGate(
    [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0], 
    ],"CNOT"
)

#CZ

CZ = QuantumGate(
    [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,-1]
    ], "CZ"
)


#CY

CY = QuantumGate(
    [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,-1j],
        [0,0,1j,0]
    ], "CY"
)


#SWAP

SWAP = QuantumGate(
    [
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ], "SWAP"
)

# iSWAP

iSWAP = QuantumGate(
    [
        [1,0,0,0],
        [0,0,1j,0],
        [0,1j,0,0],
        [0,0,0,1]
    ], "iSWAP"
)



# 3 Cubit

TOffoli = QuantumGate(
    [[1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0],]
    
    ,"CCNOT"
)

Fredkin = QuantumGate(
    [[1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1],]
    
    ,"CSWAP"
)