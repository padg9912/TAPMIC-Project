#!/usr/bin/env python3
# evaluate.py
# Evaluation script for TAPMIC models

import os
import sys
import argparse
import logging
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import wandb

# Import ModelEvaluator
from evaluation_metrics import ModelEvaluator
from roberta_model import load_pretrained_model, load_data, create_dataloaders
from wandb_setup import initialize_wandb, finish_run

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Define file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "models", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate TAPMIC models")
    
    # Model configuration
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to model checkpoint to evaluate")
    parser.add_argument("--model_name", type=str, default="roberta-base", 
                        help="Base model name for tokenizer")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for evaluation")
    
    # Feature options
    parser.add_argument("--use_temporal", action="store_true", 
                        help="Use temporal features")
    parser.add_argument("--use_credibility", action="store_true", 
                        help="Use speaker credibility features")
    
    # Evaluation options
    parser.add_argument("--run_name", type=str, required=True, 
                        help="Name for this evaluation run")
    parser.add_argument("--eval_temporal_breakdown", action="store_true", 
                        help="Evaluate with temporal feature breakdown")
    parser.add_argument("--cross_validation", action="store_true", 
                        help="Perform cross-validation evaluation")
    parser.add_argument("--cv_folds", type=int, default=5, 
                        help="Number of folds for cross-validation")
    
    # Run configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--wandb", action="store_true", 
                        help="Log metrics with Weights & Biases")
    
    return parser.parse_args()

def main():
    """Main function to run model evaluation"""
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
        initialize_wandb(
            run_name=f"eval_{args.run_name}",
            feature_mode=feature_mode,
            config={
                "model_path": args.model_path,
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "use_temporal": args.use_temporal,
                "use_credibility": args.use_credibility,
                "feature_mode": feature_mode,
                "eval_temporal_breakdown": args.eval_temporal_breakdown,
                "cross_validation": args.cross_validation
            }
        )
    
    # Load data
    logging.info("Loading data...")
    train_df, val_df, test_df = load_data()
    
    # Load model
    logging.info(f"Loading model from {args.model_path}...")
    model = load_pretrained_model(args.model_path, device)
    
    # Create dataloaders
    logging.info("Creating dataloaders...")
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    train_loader, val_loader, test_loader, num_features = create_dataloaders(
        train_df, val_df, test_df, tokenizer, 
        batch_size=args.batch_size,
        use_temporal=args.use_temporal, 
        use_credibility=args.use_credibility
    )
    
    # Initialize model evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        run_name=args.run_name,
        feature_mode=feature_mode
    )
    
    # Standard evaluation on test set
    logging.info("Evaluating model on test set...")
    metrics = evaluator.evaluate_model(test_loader, output_report=True)
    
    # Print main metrics
    logging.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Test F1 Score: {metrics['f1']:.4f}")
    
    # Temporal feature breakdown if requested
    if args.eval_temporal_breakdown and args.use_temporal:
        logging.info("Performing temporal feature breakdown evaluation...")
        
        # Key temporal features to evaluate
        temporal_features = [
            "mean_publication_year",
            "publication_date_range_days",
            "days_to_nearest_election"
        ]
        
        # Evaluate with breakdown for each feature
        for feature in temporal_features:
            if feature in test_loader.dataset.temporal_cols:
                logging.info(f"Evaluating breakdown by {feature}...")
                temporal_metrics = evaluator.evaluate_with_temporal_breakdown(test_loader, feature)
                
                # Log summary for this feature
                logging.info(f"Completed temporal breakdown for {feature}")
            else:
                logging.warning(f"Feature {feature} not found in dataset")
    
    # Cross-validation if requested
    if args.cross_validation:
        logging.info(f"Performing {args.cv_folds}-fold cross-validation...")
        
        # For cross-validation, we need the full dataset
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset, test_loader.dataset])
        
        # Run cross-validation
        cv_metrics = evaluator.evaluate_with_cross_validation(
            dataset=full_dataset,
            n_splits=args.cv_folds,
            batch_size=args.batch_size
        )
        
        # Log cross-validation results
        logging.info(f"Cross-validation results:")
        logging.info(f"Average accuracy: {cv_metrics['average']['accuracy']:.4f} ± {cv_metrics['std_dev']['accuracy']:.4f}")
        logging.info(f"Average F1 score: {cv_metrics['average']['f1']:.4f} ± {cv_metrics['std_dev']['f1']:.4f}")
    
    # Finish wandb run if enabled
    if args.wandb:
        finish_run()
    
    logging.info(f"Evaluation completed. Reports saved in {os.path.join(REPORTS_DIR, args.run_name)}")
    return metrics['accuracy']

if __name__ == "__main__":
    main() 