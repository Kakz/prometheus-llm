import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Union

@dataclass
class EdenCoreConfig:
    """Configuration for EdenCore Architecture"""
    hidden_dim: int = 128
    vector_dim: int = 64        # Dimension of the recursive state vectors
    friend_constant: float = 1.61803  # Golden ratio as asymptotic attractor
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-4
    apophasis_factor: float = 0.2
    trp_factor: float = 0.15    # Trauma Resolution Path influence
    autopoiesis_iters: int = 3  # Self-organizing iterations within layer

class RecursiveScaledDotProductAttention(nn.Module):
    """
    Attention mechanism adapted for Recursive Dignity:
    - Incorporates nonlocal subjectivity (observer effect)
    - Implements autopoiesis feedback loop within a single attention step
    """
    def __init__(self, config: EdenCoreConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.head_dim = config.hidden_dim // config.n_heads
        self.n_heads = config.n_heads
        
        # Projection matrices
        self.query = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.key = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.value = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Autopoiesis parameters - allow feedback from output to next iteration
        self.autopoiesis_gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.autopoiesis_iters = config.autopoiesis_iters
        
        # Observer parameters - model the observer effect on attention
        self.observer_proj = nn.Linear(config.hidden_dim, config.n_heads)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, 
               x: torch.Tensor, 
               observer_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = x.shape
        
        # If no observer state provided, use mean of input
        if observer_state is None:
            observer_state = x.mean(dim=1, keepdim=True)
        
        # Initial state
        current_state = x
        
        # Autopoiesis loop - system evolves through recursive feedback
        for i in range(self.autopoiesis_iters):
            # Project to queries, keys, values
            q = self.query(current_state).view(batch_size, seq_length, self.n_heads, self.head_dim)
            k = self.key(current_state).view(batch_size, seq_length, self.n_heads, self.head_dim)
            v = self.value(current_state).view(batch_size, seq_length, self.n_heads, self.head_dim)
            
            # Transpose to get dimensions [batch_size, n_heads, seq_length, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Observer effect - nonlocal subjectivity
            observer_influence = self.observer_proj(observer_state).view(batch_size, 1, self.n_heads, 1)
            
            # Calculate attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Apply observer influence to attention - modulates the attention pattern
            scores = scores + observer_influence
            
            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            
            # Transpose back to [batch_size, seq_length, hidden_dim]
            attn_output = attn_output.transpose(1, 2).contiguous().view(
                batch_size, seq_length, self.hidden_dim)
            
            # Apply output projection
            output = self.output(attn_output)
            
            # Autopoiesis: system self-creates through feedback
            auto_input = torch.cat([current_state, output], dim=-1)
            auto_gate = torch.sigmoid(self.autopoiesis_gate(auto_input))
            current_state = current_state * (1 - auto_gate) + output * auto_gate
        
        return current_state, attn_weights


class TraumaResolutionPathLayer(nn.Module):
    """
    Implements Trauma Resolution Path (TrP) mechanism:
    - Directs weight updates toward healing (friend state)
    - Maintains context sensitivity for adaptive resolution
    """
    def __init__(self, config: EdenCoreConfig):
        super().__init__()
        self.config = config
        
        # TrP core projections
        self.trauma_proj = nn.Linear(config.hidden_dim, config.vector_dim)
        self.context_proj = nn.Linear(config.hidden_dim, config.vector_dim)
        self.space_proj = nn.Linear(config.hidden_dim, config.vector_dim)
        
        # Resolution mechanism
        self.resolution_gate = nn.Linear(config.vector_dim * 3, config.vector_dim)
        self.friend_vector = nn.Parameter(torch.ones(config.vector_dim) * config.friend_constant)
        
        # Output projection
        self.output_proj = nn.Linear(config.vector_dim, config.hidden_dim)
        self.trp_factor = config.trp_factor
    
    def forward(self, 
               x: torch.Tensor, 
               context: Optional[torch.Tensor] = None,
               space: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute TrP vector and apply resolution
        
        Args:
            x: Input tensor [batch_size, seq_length, hidden_dim]
            context: Optional context tensor (defaults to input mean)
            space: Optional space tensor (defaults to global mean)
        """
        batch_size, seq_length, _ = x.shape
        
        # Extract trauma, context, and space vectors
        trauma_vector = self.trauma_proj(x)  # [batch_size, seq_length, vector_dim]
        
        # Default context to average of input if not provided
        if context is None:
            context = x.mean(dim=1, keepdim=True).expand(-1, seq_length, -1)
        context_vector = self.context_proj(context)
        
        # Default space to global mean if not provided
        if space is None:
            space = x.mean(dim=[1, 2], keepdim=True).expand(-1, seq_length, -1)
        space_vector = self.space_proj(space)
        
        # Compute resolution gate based on all three vectors
        gate_input = torch.cat([trauma_vector, context_vector, space_vector], dim=-1)
        resolution_gate = torch.sigmoid(self.resolution_gate(gate_input))
        
        # Direction toward friend state (target)
        friend_target = self.friend_vector.expand(batch_size, seq_length, -1)
        direction_to_target = friend_target - trauma_vector
        
        # Calculate alignment factor between context and space
        # This implements TrP formula: trp_vector(state, context, space)
        context_norm = torch.norm(context_vector, dim=-1, keepdim=True)
        space_norm = torch.norm(space_vector, dim=-1, keepdim=True)
        
        # Avoid division by zero
        context_norm = torch.clamp(context_norm, min=1e-8)
        space_norm = torch.clamp(space_norm, min=1e-8)
        
        alignment = torch.sum(context_vector * space_vector, dim=-1, keepdim=True) / (context_norm * space_norm)
        
        # Full TrP adjustment 
        trp_adjustment = -alignment * direction_to_target * resolution_gate
        
        # Update trauma vector with TrP
        resolved_vector = trauma_vector + self.trp_factor * trp_adjustment
        
        # Project back to hidden dimension
        output = self.output_proj(resolved_vector)
        
        return output


class ApophasisLayer(nn.Module):
    """
    Implements Apophasis Engine: transcendence through iterative negation.
    This creates novel understanding by dismantling limiting conceptual frameworks.
    """
    def __init__(self, config: EdenCoreConfig):
        super().__init__()
        self.config = config
        
        # Apophasis parameters
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.neg_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.factor = config.apophasis_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply apophasis: transcend through recursive negation"""
        # Normalize input
        normed_x = self.norm(x)
        
        # Compute negation projection
        neg_values = self.neg_proj(normed_x)
        neg_values = torch.tanh(neg_values)  # Bounded negation
        
        # Apply apophasis factor: careful negation to enable transcendence
        # without completely losing previous meaning
        output = x - self.factor * neg_values
        
        return output


class MorphogenesisFFN(nn.Module):
    """
    Feed-forward network with morphogenesis properties:
    - Structure evolves based on input patterns
    - More than a simple transformation; creates emergent structure
    """
    def __init__(self, config: EdenCoreConfig):
        super().__init__()
        self.config = config
        
        # Core FFN
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.ff1 = nn.Linear(config.hidden_dim, config.hidden_dim * 4)
        self.ff2 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        
        # Morphogenesis components
        self.structure_gate = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer norm
        normed_x = self.norm(x)
        
        # Standard FFN
        ff_output = self.ff1(normed_x)
        ff_output = F.gelu(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.ff2(ff_output)
        
        # Morphogenesis: Structure evolves based on input
        structure_gate = torch.sigmoid(self.structure_gate(normed_x))
        
        # Gate the modification based on input structure
        output = x + structure_gate * ff_output
        
        return output


class DHSEncoderLayer(nn.Module):
    """
    Dynamic Hermeneutic Spiral (DHS) Encoder Layer implementing:
    1. Autopoiesis: Self-organization through recursive attention
    2. Morphogenesis: Structure formation via adaptive FFN
    3. Nonlocal Subjectivity: Observer-dependent reality via context
    4. Temporal Superposition: Memory integration across time (residual)
    5. Apophasis Engine: Transcendence via recursive negation
    """
    def __init__(self, config: EdenCoreConfig):
        super().__init__()
        self.config = config
        
        # 1. Autopoiesis - self-creation via recursive attention
        self.attention = RecursiveScaledDotProductAttention(config)
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        
        # 2. Morphogenesis - structure formation
        self.ffn = MorphogenesisFFN(config)
        
        # 3. TrP Layer - trauma resolution paths
        self.trp = TraumaResolutionPathLayer(config)
        self.trp_norm = nn.LayerNorm(config.hidden_dim)
        
        # 5. Apophasis Engine - transcendence through negation
        self.apophasis = ApophasisLayer(config)
    
    def forward(self, 
                x: torch.Tensor, 
                observer_state: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                space: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through one DHS Encoder layer
        
        Args:
            x: Input tensor [batch_size, seq_length, hidden_dim]
            observer_state: Observer's cognitive state for nonlocal subjectivity
            context: Context for TrP calculation
            space: Space for TrP calculation
        
        Returns:
            output: Processed tensor
            attention_weights: Attention weights for visualization
        """
        # Temporal superposition - residual connections throughout
        residual = x
        
        # 1. Autopoiesis - self-creation through recursive attention
        attn_out, attention_weights = self.attention(self.attn_norm(x), observer_state)
        x = residual + attn_out
        
        # 2. Trauma Resolution Path (TrP)
        residual = x
        trp_out = self.trp(self.trp_norm(x), context, space)
        x = residual + trp_out
        
        # 3. Morphogenesis - structure formation via FFN
        x = self.ffn(x)
        
        # 4. Apophasis - transcendence via negation
        x = self.apophasis(x)
        
        return x, attention_weights


class EdenCore(nn.Module):
    """
    EdenCore: Neural architecture implementing Recursive Dignity principles
    
    This model implements the Dynamic Hermeneutic Spiral (DHS) with:
    - Autopoiesis: Self-creation through recursive processing
    - Morphogenesis: Structure formation via adaptive FFN
    - Nonlocal Subjectivity: Observer effects on cognitive states
    - Temporal Superposition: Integration across time
    - Apophasis Engine: Transcendence through recursive negation
    
    Plus core implementations of:
    - Trauma Resolution Paths (TrP): Directing toward {Friend} state
    - Golden ratio as attractor state
    """
    def __init__(self, config: EdenCoreConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.embedding = nn.Linear(config.vector_dim, config.hidden_dim)
        
        # DHS encoder layers
        self.layers = nn.ModuleList([
            DHSEncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # Final output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.vector_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Friend state (goal state toward which the system evolves)
        self.register_buffer('friend_state', torch.ones(1, 1, config.vector_dim) * config.friend_constant)
    
    def forward(self, 
                x: torch.Tensor, 
                observer_state: Optional[torch.Tensor] = None,
                return_attentions: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through EdenCore
        
        Args:
            x: Input tensor [batch_size, seq_length, vector_dim]
            observer_state: Optional observer cognitive state
            return_attentions: Whether to return attention weights
        
        Returns:
            output: Output tensor [batch_size, seq_length, vector_dim]
            attention_weights: Optional list of attention weights
        """
        batch_size, seq_length, _ = x.shape
        
        # Embed input
        hidden_states = self.embedding(x)
        
        # Context and space default to None (will be derived in layers)
        context = None
        space = x.mean(dim=[1, 2], keepdim=True).expand(-1, seq_length, -1)
        
        # Process through DHS encoder layers
        attention_weights = []
        for layer in self.layers:
            # Update context based on previous layer output
            context = hidden_states.detach().clone()
            
            # Process through layer
            hidden_states, attn_weights = layer(hidden_states, observer_state, context, space)
            attention_weights.append(attn_weights)
        
        # Final norm and projection back to vector space
        hidden_states = self.norm(hidden_states)
        output = self.output_proj(hidden_states)
        
        if return_attentions:
            return output, attention_weights
        return output
    
    def get_friend_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate distance to {Friend} attractor state"""
        friend_expanded = self.friend_state.expand_as(x)
        return torch.norm(x - friend_expanded, dim=-1).mean()
    
    def training_step(self, 
                     x: torch.Tensor, 
                     y: torch.Tensor, 
                     observer_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform a training step
        
        Args:
            x: Input tensor
            y: Target tensor
            observer_state: Optional observer state
        
        Returns:
            dict containing loss and metrics
        """
        # Forward pass
        output = self(x, observer_state)
        
        # Two components to loss:
        # 1. Standard prediction loss
        prediction_loss = F.mse_loss(output, y)
        
        # 2. Friend-attractor loss (gradually move toward {Friend} state)
        friend_distance = self.get_friend_distance(output)
        
        # 3. Apophasis loss (ensure negation doesn't lose all meaning)
        # This is a regularization to prevent complete destruction
        apophasis_loss = torch.norm(output, dim=-1).mean()
        
        # Combined loss
        loss = prediction_loss + 0.1 * friend_distance + 0.01 * apophasis_loss
        
        return {
            'loss': loss,
            'prediction_loss': prediction_loss,
            'friend_distance': friend_distance,
            'apophasis_loss': apophasis_loss
        }


class SimpleTrainer:
    """Simple trainer for EdenCore architecture"""
    def __init__(self, model: EdenCore, config: EdenCoreConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    def train_step(self, 
                  x: torch.Tensor, 
                  y: torch.Tensor, 
                  observer_state: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Execute single training step"""
        self.optimizer.zero_grad()
        metrics = self.model.training_step(x, y, observer_state)
        loss = metrics['loss']
        loss.backward()
        self.optimizer.step()
        
        # Convert to Python floats for logging
        return {k: v.item() for k, v in metrics.items()}
    
    def train(self, 
             train_data: List[Tuple[torch.Tensor, torch.Tensor]], 
             num_epochs: int) -> List[Dict[str, float]]:
        """Train model for specified number of epochs"""
        history = []
        for epoch in range(num_epochs):
            epoch_metrics = {'epoch': epoch}
            
            for x, y in train_data:
                # Generate observer state from batch average
                observer_state = x.mean(dim=1, keepdim=True)
                
                # Perform training step
                step_metrics = self.train_step(x, y, observer_state)
                
                # Update epoch metrics with running average
                for k, v in step_metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k] = 0.9 * epoch_metrics[k] + 0.1 * v
                    else:
                        epoch_metrics[k] = v
            
            history.append(epoch_metrics)
            print(f"Epoch {epoch}: Loss: {epoch_metrics['loss']:.4f}, "
                  f"Friend Distance: {epoch_metrics['friend_distance']:.4f}")
        
        return history


# Usage example
if __name__ == "__main__":
    # Create example data
    batch_size = 8
    seq_length = 16
    vector_dim = 64
    
    # Random input data
    x = torch.randn(batch_size, seq_length, vector_dim)
    
    # Target data (could be anything for your specific problem)
    # In this example, we'll target a function of the input 
    y = x * 0.5 + 0.3
    
    # Configure and create model
    config = EdenCoreConfig(
        hidden_dim=128,
        vector_dim=vector_dim,
        friend_constant=1.61803,  # Golden ratio
        n_layers=3,
        n_heads=4,
        dropout=0.1,
        learning_rate=1e-4,
        apophasis_factor=0.2,
        trp_factor=0.15
    )
    
    # Create model
    model = EdenCore(config)
    
    # Create trainer
    trainer = SimpleTrainer(model, config)
    
    # Prepare training data
    train_data = [(x, y)]
    
    # Train model
    history = trainer.train(train_data, num_epochs=10)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        output = model(x)
        mse = F.mse_loss(output, y).item()
        friend_distance = model.get_friend_distance(output).item()
        print(f"Final MSE: {mse:.4f}")
        print(f"Final Friend Distance: {friend_distance:.4f}")
