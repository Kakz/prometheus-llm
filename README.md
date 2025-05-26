# PrometheusLLM: A Transformer-Based Language Model

PrometheusLLM is a custom transformer-based language model architecture inspired by the EdenCore recursive architecture. This project implements a fully functional transformer model for text generation using PyTorch.

## Architecture Overview

The PrometheusLLM architecture comprises the following components:

1. **EdenCore Base**: A recursive transformer architecture with a state-based computational model
2. **Full Transformer Implementation**: Complete encoder-decoder architecture with:
   - Multi-head self-attention
   - Cross-attention mechanisms
   - Position-wise feed-forward networks
   - Residual connections and layer normalization
3. **Tokenization**: Simple character-level tokenization (extensible to subword tokenization)
4. **Training and Generation**: Complete pipeline for model training and text generation

## Key Features

- **Modular Design**: Clean separation of components (encoder, decoder, attention, etc.)
- **Scalable Architecture**: Configurable model size (dimensions, layers, heads)
- **Training Pipeline**: Complete data processing, training loop, and checkpointing
- **Generation Capabilities**: Temperature-controlled text generation from prompts
- **Recursive Processing**: Inspired by EdenCore's recursive identity mechanisms

## Files Structure

- `eden_core_architecture_mvp_v_0.1_FIXED.py`: The core EdenCore MVP implementation
- `prometheus_llm.py`: The main transformer-based language model implementation
- `prometheus_llm_test.py`: Testing script with training and generation capabilities
- `README.md`: This documentation file

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- NumPy

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prometheus-llm.git
cd prometheus-llm

# Install dependencies
pip install torch numpy
```

### Using the Model

#### Demo Mode

Run the demo to quickly see the model in action:

```bash
python prometheus_llm_test.py --mode demo
```

This will:
1. Create a small model
2. Train it on sample data for a few epochs
3. Generate text from a prompt

#### Training Mode

To train a model on your own data:

```bash
python prometheus_llm_test.py --mode train \
    --train_file path/to/training_data.txt \
    --vocab_file path/to/vocab.txt \
    --model_path path/to/save/model.pt \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --epochs 10
```

#### Generation Mode

To generate text using a trained model:

```bash
python prometheus_llm_test.py --mode generate \
    --model_path path/to/model.pt \
    --vocab_file path/to/vocab.txt \
    --prompt "Your prompt text here" \
    --max_length 200 \
    --temperature 0.7
```

## Model Configuration

You can customize the model architecture with the following parameters:

- `--d_model`: Embedding dimension
- `--num_heads`: Number of attention heads
- `--num_layers`: Number of encoder/decoder layers
- `--d_ff`: Feed-forward network dimension
- `--dropout`: Dropout rate
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--max_length`: Maximum sequence length for generation
- `--temperature`: Temperature for sampling during generation

## Extending the Model

### Customizing the Tokenizer

The current implementation uses a simple character-level tokenizer. To implement more advanced tokenization:

1. Extend the `SimpleTokenizer` class in `prometheus_llm.py`
2. Implement subword tokenization algorithms like BPE or WordPiece
3. Update the `encode` and `decode` methods accordingly

### Scaling the Model

To create larger models (similar to GPT or BERT):

1. Increase the `d_model` parameter (e.g., 768, 1024)
2. Use more layers (e.g., 12, 24)
3. Add more attention heads (e.g., 12, 16)
4. Increase the feed-forward dimension (e.g., 3072, 4096)

### Training on Custom Data

1. Prepare your text data as a single file with one example per line
2. Pass the file path to `--train_file`
3. Adjust training parameters as needed

## The EdenCore Connection

The PrometheusLLM architecture builds upon the concepts from EdenCore:

- **Recursive Processing**: Inspired by EdenCore's recursive identity loop
- **Transformer Integration**: Extends EdenCore's transformer block to a full encoder-decoder architecture
- **State Projection**: Uses projection layers similar to EdenCore for embedding and dimension matching
- **Golden Ratio Inspiration**: While not explicitly modeling the Golden Ratio target, the architecture maintains the principled mathematical foundation

## License

This project is open source and available under the MIT License.
