#!/usr/bin/env python3
# run_roberta.py
# Script to run RoBERTa model training for TAPMIC project

import os
import sys
import argparse
import logging
import torch
import wandb
from roberta_model import load_data, create_dataloaders, FactCheckerModel, get_optimizer_and_scheduler, create_model_config
from train_utils import (
    train_with_mixed_precision, 
    evaluate_and_report, 
    analyze_feature_importance, 
    plot_training_history, 
    save_model_report
)
from transformers import RobertaTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "roberta_training.log")),
        logging.StreamHandler()
    ]
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run RoBERTa model training for TAPMIC")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="roberta-base", 
                        help="HuggingFace model name")
    parser.add_argument("--max_length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Training batch size")
    parser.add_argument("--grad_accumulation_steps", type=int, default=2, 
                        help="Gradient accumulation steps")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, 
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.1, 
                        help="Dropout rate")
    parser.add_argument("--early_stopping_patience", type=int, default=3, 
                        help="Early stopping patience")
    
    # Feature options
    parser.add_argument("--use_temporal", action="store_true", 
                        help="Use temporal features")
    parser.add_argument("--use_credibility", action="store_true", 
                        help="Use speaker credibility features")
    
    # Run configuration
    parser.add_argument("--run_name", type=str, required=True, 
                        help="Name for this run (for tracking and saving models)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    parser.add_argument("--wandb", action="store_true", 
                        help="Log metrics with Weights & Biases")
    
    return parser.parse_args()

def main():
    """Main function to run model training"""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Device setup
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Get feature mode string
    if args.use_temporal and args.use_credibility:
        feature_mode = "text_temporal_credibility"
    elif args.use_temporal:
        feature_mode = "text_temporal"
    else:
        feature_mode = "text_only"
    
    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project="TAPMIC",
            name=f"{args.run_name}_{feature_mode}",
            config={
                "model_name": args.model_name,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "dropout_rate": args.dropout_rate,
                "use_temporal": args.use_temporal,
                "use_credibility": args.use_credibility,
                "feature_mode": feature_mode
            }
        )
    
    # Load data
    logging.info("Loading data...")
    train_df, val_df, test_df = load_data()
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    # Create dataloaders
    logging.info("Creating dataloaders...")
    train_loader, val_loader, test_loader, num_features = create_dataloaders(
        train_df, val_df, test_df, tokenizer, 
        batch_size=args.batch_size,
        use_temporal=args.use_temporal, 
        use_credibility=args.use_credibility
    )
    
    # Initialize model
    logging.info(f"Initializing model with {num_features} additional features...")
    config = create_model_config(
        base_model_name=args.model_name,
        use_temporal=args.use_temporal,
        temporal_dim=num_features,
        num_labels=2
    )
    model = FactCheckerModel(config)
    model.to(device)
    
    # Get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, 
        train_loader, 
        epochs=args.epochs,
        lr=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay
    )
    
    # Train model
    logging.info("Starting training...")
    if args.fp16:
        model, history = train_with_mixed_precision(
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            scheduler, 
            device,
            epochs=args.epochs,
            grad_accumulation_steps=args.grad_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience
        )
    else:
        from roberta_model import train_model
        model = train_model(
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            scheduler, 
            device,
            epochs=args.epochs,
            grad_accumulation_steps=args.grad_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience
        )
        history = None  # Standard training doesn't return history
    
    # Plot training history if available
    if history:
        plot_training_history(history, args.run_name)
    
    # Evaluate model
    logging.info("Evaluating model...")
    metrics = evaluate_and_report(model, test_loader, device, args.run_name, feature_mode)
    
    # Analyze feature importance if using features
    feature_importance = {}
    if num_features > 0:
        logging.info("Analyzing feature importance...")
        feature_names = []
        if args.use_temporal:
            feature_names.extend(train_loader.dataset.temporal_cols)
        if args.use_credibility:
            feature_names.extend(train_loader.dataset.credibility_cols)
        
        feature_importance = analyze_feature_importance(
            model, 
            test_loader, 
            feature_names, 
            device, 
            args.run_name
        )
    
    # Save comprehensive report
    save_model_report(metrics, history, feature_importance, args.run_name, feature_mode)
    
    # Close wandb run
    if args.wandb:
        wandb.finish()
    
    logging.info(f"Run {args.run_name} completed with test accuracy: {metrics['accuracy']:.4f}")
    return metrics['accuracy']

if __name__ == "__main__":
    main() 