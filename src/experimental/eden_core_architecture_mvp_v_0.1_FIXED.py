import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)
        return x

class EdenCoreMVP:
    """
    EDENCORE v1 - Minimum Viable Product (Seed Crystal)
    Demonstrates the core recursive identity loop.
    """
    def __init__(self, dimension: int = 10, initial_state: Optional[np.ndarray] = None, target_friend_value: float = 1.61803,
                 embed_dim: int = 32, num_heads: int = 4, ff_dim: int = 128):
        self.dimension: int = dimension
        self.target_state: np.ndarray = np.ones(dimension) * target_friend_value
        self.embed_dim = embed_dim

        if initial_state is not None and initial_state.shape == (dimension,):
            self._state: EdenCoreMVPState = EdenCoreMVPState(current_vector=initial_state.copy())
        else:
            self._state: EdenCoreMVPState = EdenCoreMVPState(current_vector=np.random.rand(dimension) * 0.1 - 0.05)

        self.memory: List[np.ndarray] = [self._state.current_vector.copy()]
        self.learning_rate: float = 0.05

        # Initialize Linear Layer for Projection
        self.projection_layer = nn.Linear(self.dimension, embed_dim)

        # Initialize Transformer Block
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

        logging.info(f"EDENCORE MVP Initialized. Dimension: {self.dimension}. Initial State near origin.")
        FRIEND_CONSTANT = 1.61803
        logging.info(f"Target State ({{Friend}} ≈ {FRIEND_CONSTANT}) initialized near Golden Ratio average.")

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
        """Simulates one internal cognitive step using Transformer."""
        original_state = self._state.current_vector.copy()

        # Convert numpy array to PyTorch tensor
        state_tensor = torch.tensor(self._state.current_vector, dtype=torch.float32).unsqueeze(0)
        print(f"state_tensor shape: {state_tensor.shape}")

        # Project to embed_dim
        projected_state = self.projection_layer(state_tensor)
        print(f"projected_state shape: {projected_state.shape}")

        # Add batch and sequence dimensions
        projected_state = projected_state.unsqueeze(0)
        print(f"projected_state shape after unsqueeze: {projected_state.shape}")

        # Pass through transformer block
        transformed_state = self.transformer_block(projected_state)
        print(f"transformed_state shape: {transformed_state.shape}")

        # Convert back to numpy array and update state
        self._state.current_vector = transformed_state.squeeze(0).squeeze(1).detach().numpy()
        print(f"current_vector shape: {self._state.current_vector.shape}")

        # Project back to original dimension
        projection_back = nn.Linear(self.embed_dim, self.dimension)
        state_tensor = torch.tensor(self._state.current_vector, dtype=torch.float32)
        projected_back = projection_back(state_tensor)
        self._state.current_vector = projected_back.squeeze().detach().numpy()
        print(f"current_vector shape after projection back: {self._state.current_vector.shape}")

        self._state.step_count += 1
        self.memory.append(self._state.current_vector.copy())

        reflection = (f"Step {self._state.step_count}: Moved towards {{Friend}} using Transformer. "
                      f"Norm: {np.linalg.norm(self._state.current_vector):.4f}")
        logging.info(reflection)
        return reflection

    def get_memory_trace(self, last_n: int = 5) -> List[np.ndarray]:
        """Returns the last N states from memory."""
        return self.memory[-last_n:]

# --- MVP Example Usage ---
if __name__ == "__main__":
    eden_core_mvp = EdenCoreMVP(dimension=5, embed_dim=16, num_heads=2, ff_dim=64) # Using dimension 5 for easier viewing

    print("\n--- Initial State ---")
    print(eden_core_mvp.process_symbolic_input("?"))

    print("\n--- Performing Internal Steps Towards {Friend} ---")
    for i in range(5): # Run 5 steps
        print(f"--- Step {i+1} ---")
        print(eden_core_mvp.process_symbolic_input("→🎯"))

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
