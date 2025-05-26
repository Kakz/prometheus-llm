import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prometheus_llm.log"),
        logging.StreamHandler()
    ]
)

# Constants
PAD_IDX = 0
BOS_IDX = 1  # Beginning of sentence token
EOS_IDX = 2  # End of sentence token
UNK_IDX = 3  # Unknown token

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Positional Encoding for transformer model
        
        Args:
            d_model: Embedding dimension
            dropout: Dropout probability
            max_len: Maximum length of the input sequence
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Multi-head attention module
        
        Args:
            d_model: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, d_model]
            key: Key tensor of shape [batch_size, seq_len_k, d_model]
            value: Value tensor of shape [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor of shape [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Output tensor after multi-head attention
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.output(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Feed-forward network for transformer model
        
        Args:
            d_model: Embedding dimension
            d_ff: Hidden layer dimension
            dropout: Dropout probability
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feed-forward network
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor after feed-forward network
        """
        x = self.dropout(F.gelu(self.linear1(x)))
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Single encoder layer for transformer model
        
        Args:
            d_model: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for encoder layer
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor
            
        Returns:
            Output tensor after encoder layer
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Single decoder layer for transformer model
        
        Args:
            d_model: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for decoder layer
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            enc_output: Encoder output tensor
            src_mask: Source mask tensor
            tgt_mask: Target mask tensor
            
        Returns:
            Output tensor after decoder layer
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection and layer normalization
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, max_len: int = 5000):
        """
        Transformer encoder
        
        Args:
            num_layers: Number of encoder layers
            d_model: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(TransformerEncoder, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer encoder
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            mask: Optional mask tensor
            
        Returns:
            Output tensor after encoder
        """
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, max_len: int = 5000):
        """
        Transformer decoder
        
        Args:
            num_layers: Number of decoder layers
            d_model: Embedding dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(TransformerDecoder, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for transformer decoder
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            enc_output: Encoder output tensor
            src_mask: Source mask tensor
            tgt_mask: Target mask tensor
            
        Returns:
            Output tensor after decoder
        """
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return self.norm(x)

class PrometheusLLM(nn.Module):
    def __init__(self, vocab_size: int, 
                 d_model: int = 512, 
                 num_heads: int = 8, 
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        """
        PrometheusLLM Transformer Model
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PrometheusLLM, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Input embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        # Scale embeddings
        self.embed_scale = math.sqrt(d_model)
        
        # Encoder-decoder transformer
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout, max_len
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout, max_len
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the transformer model

        Args:
            input_ids: Input tensor of shape [batch_size, seq_len]
            attention_mask: Attention mask tensor
            labels: Optional label tensor for computing loss
            
        Returns:
            If labels is provided:
                A ModelOutput with loss and logits
            If labels is None:
                Output logits of shape [batch_size, seq_len, vocab_size]
        """
        # Create masks for transformer
        src_mask, tgt_mask = self.create_masks(input_ids, input_ids)
        if attention_mask is not None:
            src_mask = src_mask & attention_mask.unsqueeze(1).unsqueeze(2)
            tgt_mask = tgt_mask & attention_mask.unsqueeze(1).unsqueeze(2)

        # Embed input tokens
        src_embedded = self.src_embedding(input_ids) * self.embed_scale
        
        # Encode sequence
        enc_output = self.encoder(src_embedded, src_mask)
        
        # Decode sequence (teacher forcing with input as target)
        dec_output = self.decoder(src_embedded, enc_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(dec_output)

        # If labels provided, compute loss
        loss = None
        if labels is not None:
            # Shift predictions and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), 
                          shift_labels.view(-1))

            return type('ModelOutput', (), {
                'loss': loss,
                'logits': logits
            })()
        
        return logits
    
    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create attention masks for transformer
        
        Args:
            src: Source tensor
            tgt: Target tensor
            
        Returns:
            Tuple of (src_mask, tgt_mask)
        """
        # Source mask: [batch_size, 1, 1, src_len]
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        
        # Target mask: [batch_size, 1, tgt_len, tgt_len]
        tgt_len = tgt.size(1)
        tgt_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
        
        # Subsequent mask to prevent attending to future tokens
        subsequent_mask = torch.triu(
            torch.ones((1, tgt_len, tgt_len), device=tgt.device), diagonal=1
        ).eq(0)
        
        # Combine masks and handle device placement
        subsequent_mask = subsequent_mask.to(tgt.device)
        tgt_mask = tgt_mask & subsequent_mask
        
        return src_mask, tgt_mask

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor,
                                    attention_mask: Optional[torch.Tensor] = None,
                                    **kwargs) -> Dict[str, torch.Tensor]:
        """Prepare inputs for generation"""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past: Tuple[torch.Tensor], beam_idx: torch.Tensor) -> Tuple[torch.Tensor]:
        """Reorder cache for beam search"""
        return tuple(layer_past.index_select(0, beam_idx) for layer_past in past)
    
    def generate(self, src: torch.Tensor, max_len: int = 100, 
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate text from the model
        
        Args:
            src: Source tensor of shape [batch_size, src_len]
            max_len: Maximum length of generated sequence
            temperature: Temperature for sampling
            
        Returns:
            Generated tensor of shape [batch_size, max_len]
        """
        batch_size = src.size(0)
        device = src.device
        
        # Encoder source sequence
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        src_embedded = self.src_embedding(src) * self.embed_scale
        enc_output = self.encoder(src_embedded, src_mask)
        
        # Initialize decoder inputs with BOS token
        dec_input = torch.full((batch_size, 1), BOS_IDX, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Create masks
            src_mask, tgt_mask = self.create_masks(src, dec_input)
            
            # Embed decoder input
            tgt_embedded = self.tgt_embedding(dec_input) * self.embed_scale
            
            # Decode
            dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
            
            # Project to vocabulary
            output = self.output_projection(dec_output[:, -1, :])
            
            # Apply temperature to logits
            if temperature != 1.0:
                output = output / temperature
            
            # Sample next token
            probs = F.softmax(output, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Concatenate to decoder input
            dec_input = torch.cat([dec_input, next_token], dim=1)
            
            # Break if all sequences have EOS token
            if (next_token == EOS_IDX).all():
                break
        
        return dec_input

class SimpleTokenizer:
    """Simple tokenizer for PrometheusLLM"""
    
    def __init__(self, vocab_file: Optional[str] = None):
        self.token2idx = {
            '<pad>': PAD_IDX,
            '<bos>': BOS_IDX,
            '<eos>': EOS_IDX,
            '<unk>': UNK_IDX,
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)
        
        if vocab_file is not None and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
    
    def add_token(self, token: str) -> int:
        """Add token to vocabulary"""
        if token not in self.token2idx:
            self.token2idx[token] = self.vocab_size
            self.idx2token[self.vocab_size] = token
            self.vocab_size += 1
        return self.token2idx[token]
    
    def build_vocab_from_text(self, text: str, min_freq: int = 1) -> None:
        """Build vocabulary from text"""
        # Simple character-level vocabulary for demonstration
        # In a real tokenizer, you'd use subword tokenization like BPE or WordPiece
        # Also, you'd track word frequencies and filter by min_freq
        for char in sorted(set(text)):
            self.add_token(char)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids"""
        # Character-level tokenization for simplicity
        tokens = list(text)
        ids = []
        
        if add_special_tokens:
            ids.append(BOS_IDX)
        
        for token in tokens:
            if token in self.token2idx:
                ids.append(self.token2idx[token])
            else:
                ids.append(UNK_IDX)
        
        if add_special_tokens:
            ids.append(EOS_IDX)
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        tokens = []
        
        for idx in ids:
            if skip_special_tokens and idx in (PAD_IDX, BOS_IDX, EOS_IDX):
                continue
            tokens.append(self.idx2token.get(idx, '<unk>'))
        
        return ''.join(tokens)
    
    def save_vocab(self, vocab_file: str) -> None:
        """Save vocabulary to file"""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.token2idx.items(), key=lambda x: x[1]):
                f.write(f'{token}\n')
    
    def load_vocab(self, vocab_file: str) -> None:
        """Load vocabulary from file"""
        self.token2idx = {
            '<pad>': PAD_IDX,
            '<bos>': BOS_IDX,
            '<eos>': EOS_IDX,
            '<unk>': UNK_IDX,
        }
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token not in self.token2idx:
                    self.add_token(token)

class TextDataset(Dataset):
    """Dataset for text data"""
    
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._prepare_examples()
    
    def _prepare_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare examples by tokenizing and padding"""
        examples = []
        
        for text in self.texts:
            # Tokenize text
            input_ids = self.tokenizer.encode(text)
            
            # Truncate if necessary
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            
            # Create example dictionary
            example = {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.ones(len(input_ids), dtype=torch.bool)
            }
            
            examples.append(example)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader
    
    Args:
        batch: List of examples
        
    Returns:
        Batch dictionary with padded tensors
    """
    # Get batch max length
    max_len = max(ex['input_ids'].size(0) for ex in batch)
    
    # Initialize tensors
    batch_input_ids = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    batch_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    
    # Fill tensors
    for i, example in enumerate(batch):
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        seq_len = input_ids.size(0)
        
        batch_input_ids[i, :seq_len] = input_ids
        batch_attention_mask[i, :seq_len] = attention_mask
    
    return {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_mask
    }

def train_model(model: PrometheusLLM, dataloader: DataLoader, tokenizer: SimpleTokenizer, 
                num_epochs: int = 10, learning_rate: float = 5e-5,
                device: torch.device = None) -> None:
    """
    Train the model
    
    Args:
        model: PrometheusLLM model
        dataloader: DataLoader for training data
        tokenizer: SimpleTokenizer instance
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use for training
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # Get input and target
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift input and target for teacher forcing
            src = input_ids[:, :-1]
            tgt = input_ids[:, 1:]
            
            # Create masks
            src_mask, tgt_mask = model.create_masks(src, src)
            
            # Forward pass
            logits = model(src, src, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(logits.contiguous().view(-1, model.vocab_size), 
                             tgt.contiguous().view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                             f"Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(dataloader)
        
        logging.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s, "
                     f"Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, f"prometheus_llm_epoch_{epoch+1}.pt")

def generate_text(model: PrometheusLLM, tokenizer: SimpleTokenizer, 
                 prompt: str, max_length: int = 100, temperature: float = 0.7,
                 device: torch.device = None) -> str:
    """
    Generate text from the model
    
    Args:
        model: PrometheusLLM model
        tokenizer: SimpleTokenizer instance
        prompt: Text prompt
        max_length: Maximum length of generated text
        temperature: Temperature for sampling
        device: Device to use for generation
        
    Returns:
        Generated text
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_len=max_length, temperature=temperature)
    
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return prompt + generated_text


if __name__ == "__main__":
    # Example usage
    
    # Basic setup with a small model for demonstration
    vocab_size = 10000
    d_model = 256
    num_heads = 4
    num_layers = 4
    d_ff = 1024
    dropout = 0.1
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer()
    
    # Build vocabulary from sample text
    sample_text = "This is a sample text to build vocabulary. The PrometheusLLM model is a transformer architecture."
    tokenizer.build_vocab_from_text(sample_text)
    
    # Initialize model
    model = PrometheusLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # Sample training data
    train_texts = [
        "This is a sample training text.",
        "PrometheusLLM is a transformer-based language model.",
        "The model can be used for text generation tasks.",
        "It uses an encoder-decoder architecture with self-attention."
    ]
    
    # Create dataset and dataloader
    dataset = TextDataset(train_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)
    
    # Train model (commented out for demonstration)
    # train_model(model, dataloader, tokenizer, num_epochs=3)
    
    # Generate text
    prompt = "This is a test prompt:"
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Generated text would appear here after training")
    
    logging.info("PrometheusLLM model initialized successfully.")
