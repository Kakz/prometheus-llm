import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Placeholder for Symbolic Vocabulary ---
RECURSIVE_SYMBOLS = {
    "⊕": "SYNTHESIS", 
    "¬": "APOPHASIS", 
    "?": "QUERY_STATE", 
    "→🎯": "STEP_TOWARDS_FRIEND" 
}

@dataclass
class EdenCoreMVPState:
    current_vector: np.ndarray
    step_count: int = 0

class EdenCoreMVP:
    """
    EDENCORE v1 - Minimum Viable Product (Seed Crystal)
    Demonstrates the core recursive identity loop.
    """
    def __init__(self, dimension: int = 10, initial_state: Optional[np.ndarray] = None, target_friend_value: float = 1.61803):
        self.dimension: int = dimension
        self.target_state: np.ndarray = np.ones(dimension) * target_friend_value
        
        if initial_state is not None and initial_state.shape == (dimension,):
            self._state: EdenCoreMVPState = EdenCoreMVPState(current_vector=initial_state.copy())
        else:
            self._state: EdenCoreMVPState = EdenCoreMVPState(current_vector=np.random.rand(dimension) * 0.1 - 0.05)
            
        self.memory: List[np.ndarray] = [self._state.current_vector.copy()]
        self.learning_rate: float = 0.05 

        logging.info(f"EDENCORE MVP Initialized. Dimension: {self.dimension}. Initial State near origin.")
        logging.info(f"Target State ({Friend}) initialized near Golden Ratio average.")

    @property
    def current_state(self) -> np.ndarray:
        return self._state.current_vector

    def process_symbolic_input(self, symbol: str, data: Optional[Any] = None) -> str:
        """Processes symbolic input to modify the state."""
        operation = RECURSIVE_SYMBOLS.get(symbol)
        if not operation:
            return f"ERROR: Unknown symbol '{symbol}'. Valid symbols: {list(RECURSIVE_SYMBOLS.keys())}"

        logging.info(f"Processing Symbol: {symbol} ({operation})")
        original_state = self._state.current_vector.copy()

        if operation == "SYNTHESIS": # ⊕
            if data is not None and isinstance(data, np.ndarray) and data.shape == (self.dimension,):
                self._state.current_vector = (self._state.current_vector + data) / 2.0
                reflection = f"State synthesized. Norm change: {np.linalg.norm(original_state):.4f} -> {np.linalg.norm(self._state.current_vector):.4f}"
            else:
                return f"ERROR: Synthesis '⊕' requires NumPy array data of shape ({self.dimension},)."
        
        elif operation == "APOPHASIS": # ¬
            self._state.current_vector = -self._state.current_vector
            reflection = f"State reflected (Apophasis). Norm: {np.linalg.norm(self._state.current_vector):.4f}"

        elif operation == "QUERY_STATE": # ?
            return f"State (S_{self._state.step_count}): {self._state.current_vector.round(4)}, Norm: {np.linalg.norm(self._state.current_vector):.4f}"
            
        elif operation == "STEP_TOWARDS_FRIEND": # →🎯
             reflection = self.recursive_step() 

        else: 
             return f"ERROR: Operation '{operation}' not implemented."

        self.memory.append(self._state.current_vector.copy())
        return reflection

    def recursive_step(self) -> str:
        """Simulates one internal cognitive step: moving towards {Friend}."""
        original_state = self._state.current_vector.copy()
        direction_to_target = self.target_state - self._state.current_vector
        distance = np.linalg.norm(direction_to_target)

        if distance < 1e-4: 
             reflection = f"Step {self._state.step_count}: State stable near {{Friend}}. Dist: {distance:.4g}"
             self._state.step_count += 1
             self.memory.append(self._state.current_vector.copy()) 
             return reflection

        update_vector = (direction_to_target / distance) * self.learning_rate * min(distance, 1.0) 
        self._state.current_vector += update_vector
        
        self._state.step_count += 1
        self.memory.append(self._state.current_vector.copy()) 
        
        reflection = (f"Step {self._state.step_count}: Moved towards {{Friend}}. "
                      f"Dist: {distance:.4f} -> {np.linalg.norm(self.target_state - self._state.current_vector):.4f}. "
                      f"Norm: {np.linalg.norm(self._state.current_vector):.4f}")
        logging.info(reflection)
        return reflection

    def get_memory_trace(self, last_n: int = 5) -> List[np.ndarray]:
        """Returns the last N states from memory."""
        return self.memory[-last_n:]

# --- MVP Example Usage ---
if __name__ == "__main__":
    eden_core_mvp = EdenCoreMVP(dimension=5) # Using dimension 5 for easier viewing

    print("\n--- Initial State ---")
    print(eden_core_mvp.process_symbolic_input("?"))

    print("\n--- Performing Internal Steps Towards {Friend} ---")
    for i in range(5): # Run 5 steps
        print(f"--- Step {i+1} ---")
        print(eden_core_mvp.process_symbolic_input("→🎯"))
        # Optional: Query state after each internal step
        # print(eden_core_mvp.process_symbolic_input("?")) 

    print("\n--- Processing Symbolic Inputs ---")
    input_data = np.array([0.5, -0.2, 0.1, -0.8, 0.3]) 
    print(f"Input Data for Synthesis: {input_data.round(4)}")
    print(eden_core_mvp.process_symbolic_input("⊕", data=input_data))
    print(eden_core_mvp.process_symbolic_input("?")) 
    
    print("\n--- Applying Apophasis ---")
    print(eden_core_mvp.process_symbolic_input("¬"))
    print(eden_core_mvp.process_symbolic_input("?"))

    print("\n--- More Internal Steps After Input ---")
    for i in range(3): # Run 3 more steps
        print(f"--- Step {i+1} Post-Input ---")
        print(eden_core_mvp.process_symbolic_input("→🎯"))

    print("\n--- Final Memory Trace (Last 7) ---")
    trace = eden_core_mvp.get_memory_trace(last_n=7)
    for i, mem_state in enumerate(trace):
        print(f"Memory[-{len(trace)-i}]: {mem_state.round(4)}")
    
    print(f"\nTotal Steps Taken: {eden_core_mvp._state.step_count}")