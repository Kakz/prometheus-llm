import torch
from prometheus_llm import PrometheusLLM, SimpleTokenizer, TextDataset, collate_batch, train_model, generate_text
from torch.utils.data import DataLoader
import logging
import argparse
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prometheus_llm_test.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test PrometheusLLM Model")
    parser.add_argument("--mode", type=str, default="demo", choices=["train", "generate", "demo"],
                        help="Mode to run: train, generate, or demo")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to load or save model checkpoint")
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="Path to vocabulary file")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Path to training data file")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length for generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of encoder/decoder layers")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    return parser.parse_args()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_dummy_training_data():
    """Create dummy training data for demo"""
    dummy_texts = [
        "The PrometheusLLM model is a transformer architecture that can generate text.",
        "This model uses self-attention mechanisms to process sequential data efficiently.",
        "Transformers are powerful models for natural language processing tasks.",
        "This implementation includes an encoder-decoder architecture with multi-head attention.",
        "The model can be trained on various text datasets to generate coherent text.",
        "Attention mechanisms allow the model to focus on relevant parts of the input sequence.",
        "Positional encoding helps the model understand the order of tokens in the sequence.",
        "The feed-forward network in each layer processes the attention outputs.",
        "Layer normalization helps stabilize the training of deep transformer networks.",
        "Residual connections help with gradient flow during backpropagation.",
    ]
    return dummy_texts

def load_training_data(train_file):
    """Load training data from a file"""
    if not os.path.exists(train_file):
        logging.warning(f"Training file {train_file} not found. Using dummy data.")
        return create_dummy_training_data()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    return texts

def initialize_model(tokenizer, args):
    """Initialize the PrometheusLLM model"""
    model = PrometheusLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout
    )
    
    # Load checkpoint if provided
    if args.model_path and os.path.exists(args.model_path):
        logging.info(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=get_device())
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def run_training(args):
    """Run model training"""
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(args.vocab_file)
    
    # Load training data
    train_texts = load_training_data(args.train_file)
    
    # Build vocabulary if needed
    if tokenizer.vocab_size <= 4:  # Only special tokens
        logging.info("Building vocabulary from training data")
        all_text = " ".join(train_texts)
        tokenizer.build_vocab_from_text(all_text)
        
        if args.vocab_file:
            logging.info(f"Saving vocabulary to {args.vocab_file}")
            tokenizer.save_vocab(args.vocab_file)
    
    logging.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Initialize model
    model = initialize_model(tokenizer, args)
    
    # Create dataset and dataloader
    dataset = TextDataset(train_texts, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    
    # Train model
    train_model(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        num_epochs=args.epochs,
        device=device
    )
    
    # Save final model
    if args.model_path:
        logging.info(f"Saving model to {args.model_path}")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, args.model_path)
    
    return model, tokenizer

def run_generation(args):
    """Run text generation"""
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(args.vocab_file)
    if tokenizer.vocab_size <= 4:  # Only special tokens
        logging.error("Vocabulary not loaded. Please provide a valid vocabulary file.")
        return
    
    # Initialize model
    model = initialize_model(tokenizer, args)
    
    # Generate text
    logging.info(f"Generating text with prompt: {args.prompt}")
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        device=device
    )
    
    logging.info(f"Generated text: {generated_text}")
    return generated_text

def run_demo():
    """Run a demonstration of the model"""
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Create dummy training data
    train_texts = create_dummy_training_data()
    
    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer()
    all_text = " ".join(train_texts)
    tokenizer.build_vocab_from_text(all_text)
    logging.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Initialize a small model for demo
    model = PrometheusLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        dropout=0.1
    )
    
    # Create dataset and dataloader
    dataset = TextDataset(train_texts, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    
    # Train for a few steps
    logging.info("Training the model for 2 epochs (demo only)")
    train_model(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        num_epochs=2,
        device=device
    )
    
    # Generate text
    prompt = "The model can"
    logging.info(f"Generating text with prompt: {prompt}")
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=50,
        temperature=0.8,
        device=device
    )
    
    logging.info(f"Generated text: {generated_text}")
    return generated_text

def main():
    args = parse_args()
    
    if args.mode == "train":
        run_training(args)
    elif args.mode == "generate":
        run_generation(args)
    else:  # demo
        run_demo()

if __name__ == "__main__":
    main()
