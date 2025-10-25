from statev import Statevector
from gates import * # Fuck Tim Peters...for now
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

class QuantumCircuit:
    """
    Manages a sequence of quantum operations and the simulation of their
    evolution and measurement.
    
    This is the "recipe" manager. It holds the list of instructions,
    and can either give you the final "perfect" state (run()) or
    the "real world" measured results (sample()).
    """

    # I hate doctrings I hate both docs and strings. In fact I hate to do ct(cycle test) on rings 

    def __init__(self, num_qubits: int):
        if num_qubits <= 0:
            raise ValueError(f"So you want {num_qubits} qubit states. Go fucking develop the theory. Digital Ass 2.0")
            
        self.num_qubits: int = num_qubits
        self.dim: int = 2**num_qubits
        
        # (gate , target) tuples
        self.operations: List[tuple] = []
        
        # This is a cache. We're not fuckers. wont re-calculate the final state if we ddint change the circuit at all
        self.ca: Optional[Statevector] = None

    def get_initial_state(self) -> Statevector:
        """
        Generates the |00...0> state.
        This is a vector with 1.0 at index 0 and 0.0 everywhere else.
        """
        # Create a |0> state for one qubit
        q0 = Statevector(np.array([1.0 + 0j, 0.0 + 0j]))
        
        # As simple as fuck (actually fuck is not simple but okay) for 1 qubit
        if self.num_qubits == 1:
            return q0
            
        # For n qubits, we kron |0> with itself n times
        # |0> ⊗ |0> ⊗ ... ⊗ |0>
        state = q0
        for _ in range(self.num_qubits - 1):
            state = state.kron(q0) # Using our bad-ass kron method ....we could have yk...made a kron power method but fuck it for now (will update this comment once i make)
            
        return state

    def add(self, gate: QuantumGate, targets: List[int]) -> 'QuantumCircuit':
        """
        Adds a gate operation to the circuit's todo list.
        
        Args:
            gate: The QuantumGate object (from gates.py).
            targets: The list of qubit indices it acts on.
            
        Returns:
            self (for "method chaining", which is just a fancy way
            to look cool, e.g., circuit.add(...).add(...))
        """
        # --- Validation (because we're not savages) ---
        if gate.num_qubits != len(targets):
            raise ValueError(
                f"Gate '{gate.name}' needs {gate.num_qubits} targets, "
                f"but you gave {len(targets)}."
                f"You fucking moron"
            )
        if any(t < 0 or t >= self.num_qubits for t in targets):
            raise ValueError(
                f"Target qubit index is out of bounds (0 to {self.num_qubits-1}). "
                f"Got: {targets}"
                f"🤦"
            )
            
        # --- Add to the todofucking list ---
        self.operations.append((gate, targets))
        
        # --- CRITICAL: Invalidate the cache ---
        # We just changed the recipe, so the old "final state" is garbage.
        self.ca = None
        
        # --- Return self for chaining --- Well chaining and stning are very mediavel shits
        return self

    def run(self) -> Statevector:
        """
        Executes the "ideal simulator" run.
        
        This starts from |00...0> and applies every gate in order,
        returning the final, perfect, complex statevector.
        
        It's smart and uses a cache. If you run() twice without
        adding gates, it just gives you the saved answer.
        """
        # 1. Check the cache
        if self.ca is not None:
            # print("--ddddd") # buggy buggy
            return self.ca
            
        # 2. No cache? We do the work.
        # print("my ass") # fucky buggy
        
        # Start with the fresh |00...0> state
        current_state = self.get_initial_state()
        
        # 3. Apply every gate in the recipe, one by one
        for gate, targets in self.operations:
            # hello my cute apply method
            current_state = current_state.apply(gate, targets)
            
        # My hard earned fruity result. Sabar ka fall qubits hota hai
        self.ca = current_state
        
        return current_state

    def sample(self, shots: int = 2**14) -> Dict[str, int]:
        """
        Executes the "real quantum computer" simulation.
        
        This simulates running the circuit and *measuring* the
        result `shots` times, then returns the dictionary of
        how many times each outcome (e.g., "01") was seen.
        
        Args:
            shots: The number of "experiments" to run (default 1024).
            
        Returns:
            A dictionary of { 'binary_string': count }, 
            e.g., {'00': 501, '11': 523}
        """
        # 1. Use the perfect result
        final_state = self.run()
        
        # 2. Easy to get probabilities. Fuck you. I mean i wrote the method
        probs = final_state.probabilities
        
        # 3. Simulate all "shots" at once.
        # This is fucking faster fucker than a fucking for-loop.
        # We're forcing, under gunpoint,  numpy to pick `shots` number of outcomes,
        # from the range [0, 1, ..., 2**n - 1], using the probabilities we just calculated.
        
        # The possible outcomes are just the indices
        possible_outcomes = np.arange(self.dim)
        
        measured_outcomes = np.random.choice(
            possible_outcomes,  # e.g., [0, 1, 2, 3]
            size=shots,         # e.g., 1024
            p=probs             # e.g., [0.5, 0, 0, 0.5]
        )
        
        # 4. Tally the results
        # `measured_outcomes` is now an array like [0, 3, 3, 0, 0, 3, ...]
        # `Counter` turns this into a dict: {0: 501, 3: 523}
        counts = Counter(measured_outcomes) #hello collections
        
        # 5. Format the keys cutely, just like me
        # Users don't want `3`. They want `'11'`. Another day of babysitting users
        
        formatted_counts = {
            format(outcome, f'0{self.num_qubits}b'): count
            for outcome, count in counts.items()
        }
        
        return formatted_counts

    def __str__(self) -> str:
        """
        A simple, non-fancy printout of the circuit.
        A real "draw()" method with ASCII art is a *whole* other
        project involving 2D arrays. This is the usable version. My cat will write the draw
        """
        s = f"QuantumCircuit ({self.num_qubits} qubits)\n"
        s += "--------------------------------------\n"
        s += "INITIAL STATE: |" + "0" * self.num_qubits + "⟩\n"
        
        if not self.operations:
            s += "  (No operations added. A bit boring, eh?)\n"
            return s
            
        for i, (gate, targets) in enumerate(self.operations):
            s += f"  Step {i}: [{gate.name}] on qubit(s) {targets}\n"
            
        s += "--------------------------------------"
        return s
    


