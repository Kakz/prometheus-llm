import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import os
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prometheus_training.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class TrainingConfig:
    """Configuration for first training stage"""
    # Model Architecture (starting smaller like GPT-1)
    vocab_size: int = 50257  # Standard GPT vocab size
    d_model: int = 768      # Embedding dimension
    n_layer: int = 12       # Number of layers
    n_head: int = 12       # Number of attention heads
    d_ff: int = 3072       # Feed-forward dimension
    
    # Training Parameters
    batch_size: int = 32
    max_seq_length: int = 512
    learning_rate: float = 6.25e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    
    # Dropout (starting with higher dropout for better generalization)
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Logging and Saving
    log_interval: int = 100
    save_interval: int = 1000
    save_dir: str = "checkpoints"
    
    # Training Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PrometheusDataset(Dataset):
    """Dataset for training PrometheusLLM"""
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize and prepare for training
        encoding = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad sequence
        if len(encoding) > self.max_length:
            encoding = encoding[:self.max_length]
        else:
            padding = [0] * (self.max_length - len(encoding))
            encoding.extend(padding)
        
        return {
            'input_ids': torch.tensor(encoding),
            'attention_mask': torch.tensor([1 if token != 0 else 0 for token in encoding])
        }

class Trainer:
    """Training manager for PrometheusLLM"""
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # Move model to appropriate device
        self.model = self.model.to(config.device)
        
        # Setup optimizer with weight decay
        # Filter parameters that should have weight decay applied
        decay_params = [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n]
        no_decay_params = [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n]
        
        optimizer_grouped_params = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create save directory if it doesn't exist
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.steps = 0
        self.best_loss = float('inf')
    
    def save_checkpoint(self, loss: float, step: int):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        path = os.path.join(self.config.save_dir, f'prometheus_step_{step}.pt')
        torch.save(checkpoint, path)
        
        # If this is the best model so far, save it separately
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.config.save_dir, 'prometheus_best.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"New best model saved with loss: {loss:.4f}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['step']
        self.best_loss = checkpoint['loss']
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.config.device)
        attention_mask = batch['attention_mask'].to(self.config.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        loss = outputs.loss
        
        # Normalize loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Return the loss value
        return loss.item()
    
    def train(self, train_dataloader: DataLoader):
        """Train the model"""
        logging.info("Starting training...")
        
        # Training loop
        running_loss = 0
        last_log_time = time.time()
        accumulated_steps = 0
        
        while self.steps < self.config.max_steps:
            for batch in train_dataloader:
                # Execute training step
                loss = self.train_step(batch)
                running_loss += loss
                accumulated_steps += 1
                
                # Gradient accumulation check
                if accumulated_steps == self.config.gradient_accumulation_steps:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    self.steps += 1
                    accumulated_steps = 0
                    
                    # Logging
                    if self.steps % self.config.log_interval == 0:
                        avg_loss = running_loss / self.config.log_interval
                        current_time = time.time()
                        elapsed = current_time - last_log_time
                        
                        logging.info(
                            f"Step {self.steps}/{self.config.max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Steps/sec: {self.config.log_interval/elapsed:.2f}"
                        )
                        
                        running_loss = 0
                        last_log_time = current_time
                    
                    # Save checkpoint
                    if self.steps % self.config.save_interval == 0:
                        self.save_checkpoint(avg_loss, self.steps)
                
                if self.steps >= self.config.max_steps:
                    break
        
        logging.info("Training completed!")
        
        # Save final checkpoint
        self.save_checkpoint(avg_loss, self.steps)

def prepare_training(
    model,
    texts: List[str],
    config: Optional[TrainingConfig] = None
) -> Trainer:
    """
    Prepare model and data for training
    
    Args:
        model: The PrometheusLLM model
        texts: List of training texts
        config: Optional training configuration
    
    Returns:
        Trainer instance ready for training
    """
    if config is None:
        config = TrainingConfig()
    
    # Create dataset
    dataset = PrometheusDataset(
        texts=texts,
        tokenizer=model.tokenizer,
        max_length=config.max_seq_length
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Create trainer
    trainer = Trainer(model, config)
    
    return trainer, dataloader

if __name__ == "__main__":
    logging.info("PrometheusLLM training script initialized.")
    logging.info("Waiting for training data...")
