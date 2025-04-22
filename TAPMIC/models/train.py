#!/usr/bin/env python3
"""
Training script for the fact checking model
"""

import os
import json
import logging
import argparse
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    RobertaTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter

# Import local modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.roberta_model import FactCheckerModel, create_model_config, load_pretrained_model
from models.data_utils import (
    load_claims_data,
    create_data_loaders,
    format_results,
    save_results
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fact_checker_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_epoch(
    model, 
    data_loader, 
    optimizer, 
    scheduler, 
    device, 
    epoch
):
    """
    Train the model for one epoch
    
    Args:
        model: Model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        train_loss: Average loss for the epoch
    """
    model.train()
    losses = []
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch} [Training]")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Get temporal features if available
        temporal_features = None
        if 'temporal_features' in batch:
            temporal_features = batch['temporal_features'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            temporal_features=temporal_features
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimize
        optimizer.step()
        scheduler.step()
        
        # Update progress bar
        losses.append(loss.item())
        progress_bar.set_postfix({'loss': sum(losses) / len(losses)})
    
    # Calculate average loss
    train_loss = sum(losses) / len(losses)
    logger.info(f"Epoch {epoch} - Training Loss: {train_loss:.4f}")
    
    return train_loss


def evaluate(model, data_loader, device, epoch=None):
    """
    Evaluate the model
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to use
        epoch: Current epoch number (for logging)
        
    Returns:
        eval_loss: Average loss for the evaluation
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    losses = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Create progress bar description
    desc = f"Epoch {epoch} [Evaluation]" if epoch is not None else "Evaluation"
    progress_bar = tqdm(data_loader, desc=desc)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get temporal features if available
            temporal_features = None
            if 'temporal_features' in batch:
                temporal_features = batch['temporal_features'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                temporal_features=temporal_features
            )
            
            # Get loss
            loss = outputs.loss
            losses.append(loss.item())
            
            # Get predictions
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability of TRUE
            preds = np.where(probs >= 0.5, 1, 0)
            
            # Add to lists
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': sum(losses) / len(losses)})
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    # Calculate average loss
    eval_loss = sum(losses) / len(losses)
    
    # Log metrics
    metrics = {
        'loss': eval_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if epoch is not None:
        logger.info(f"Epoch {epoch} - Evaluation Metrics: {metrics}")
    else:
        logger.info(f"Evaluation Metrics: {metrics}")
    
    return eval_loss, metrics, all_probs


def save_model(model, tokenizer, output_dir, config=None):
    """
    Save the model and tokenizer
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Output directory
        config: Model configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save config if provided
    if config is not None:
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    logger.info(f"Model saved to {output_dir}")


def train(args):
    """
    Train the model
    
    Args:
        args: Command-line arguments
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.base_model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model_name)
    
    # Load data
    logger.info(f"Loading training data from {args.train_data}")
    train_claims, train_labels, train_claim_ids = load_claims_data(
        args.train_data,
        label_key=args.label_key,
        text_key=args.text_key,
        id_key=args.id_key
    )
    
    logger.info(f"Loading validation data from {args.val_data}")
    val_claims, val_labels, val_claim_ids = load_claims_data(
        args.val_data,
        label_key=args.label_key,
        text_key=args.text_key,
        id_key=args.id_key
    )
    
    # Create data loaders
    train_loader = create_data_loaders(
        claims=train_claims,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_temporal_features=args.use_temporal,
        labels=train_labels,
        claim_ids=train_claim_ids,
        shuffle=True
    )
    
    val_loader = create_data_loaders(
        claims=val_claims,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_temporal_features=args.use_temporal,
        labels=val_labels,
        claim_ids=val_claim_ids,
        shuffle=False
    )
    
    # Create model config
    model_config = create_model_config(
        base_model_name=args.base_model_name,
        use_temporal=args.use_temporal,
        temporal_dim=3,  # Number of temporal features
        num_labels=2
    )
    
    # Create model
    logger.info(f"Creating model based on {args.base_model_name}")
    model = FactCheckerModel.from_pretrained(
        args.base_model_name,
        config=model_config
    )
    model.to(device)
    
    # Set up optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.num_epochs
    
    # Set up scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_proportion),
        num_training_steps=total_steps
    )
    
    # Set up TensorBoard
    tb_writer = SummaryWriter(
        log_dir=os.path.join(args.output_dir, 'tensorboard')
    )
    
    # Training loop
    logger.info("Starting training")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch
        )
        
        # Evaluate
        val_loss, val_metrics, _ = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )
        
        # Log to TensorBoard
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        for metric, value in val_metrics.items():
            if metric != 'loss':
                tb_writer.add_scalar(f'Metrics/{metric}', value, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best model (val_loss={val_loss:.4f})")
            
            # Save model
            save_model(
                model=model,
                tokenizer=tokenizer,
                output_dir=os.path.join(args.output_dir, 'best_model'),
                config={
                    'base_model_name': args.base_model_name,
                    'use_temporal': args.use_temporal,
                    'max_length': args.max_length,
                    'num_labels': 2,
                    'val_metrics': val_metrics
                }
            )
    
    # Save final model
    save_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=os.path.join(args.output_dir, 'final_model'),
        config={
            'base_model_name': args.base_model_name,
            'use_temporal': args.use_temporal,
            'max_length': args.max_length,
            'num_labels': 2,
            'val_metrics': val_metrics
        }
    )
    
    # Close TensorBoard writer
    tb_writer.close()
    
    logger.info("Training completed")


def predict(args):
    """
    Make predictions using a trained model
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_dir}")
    model, tokenizer = load_pretrained_model(args.model_dir, device)
    
    # Load config
    config_path = os.path.join(args.model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        use_temporal = config.get('use_temporal', args.use_temporal)
        max_length = config.get('max_length', args.max_length)
    else:
        use_temporal = args.use_temporal
        max_length = args.max_length
    
    # Load data
    logger.info(f"Loading test data from {args.test_data}")
    test_claims, test_labels, test_claim_ids = load_claims_data(
        args.test_data,
        label_key=args.label_key,
        text_key=args.text_key,
        id_key=args.id_key
    )
    
    # Create data loader
    test_loader = create_data_loaders(
        claims=test_claims,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=max_length,
        include_temporal_features=use_temporal,
        labels=test_labels,
        claim_ids=test_claim_ids,
        shuffle=False
    )
    
    # Evaluate
    logger.info("Making predictions")
    _, metrics, probabilities = evaluate(
        model=model,
        data_loader=test_loader,
        device=device
    )
    
    # Convert probabilities to predictions
    predictions = [1 if prob >= 0.5 else 0 for prob in probabilities]
    
    # Format results
    results = format_results(
        claim_ids=test_claim_ids,
        claims=test_claims,
        predictions=predictions,
        probabilities=probabilities,
        labels=test_labels,
        output_format='json'
    )
    
    # Add metadata
    results['metadata'] = {
        'model_dir': args.model_dir,
        'test_data': args.test_data,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics
    }
    
    # Save results
    save_results(results, args.output_file)
    
    logger.info(f"Predictions saved to {args.output_file}")
    logger.info(f"Final metrics: {metrics}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train and evaluate fact checking model')
    subparsers = parser.add_subparsers(dest='mode', help='Mode')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    train_parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    train_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    train_parser.add_argument('--base_model_name', type=str, default='roberta-base', help='Base model name')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    train_parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    train_parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    train_parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion')
    train_parser.add_argument('--use_temporal', action='store_true', help='Use temporal features')
    train_parser.add_argument('--label_key', type=str, default='label', help='Key for labels in data')
    train_parser.add_argument('--text_key', type=str, default='claim_text', help='Key for claim text in data')
    train_parser.add_argument('--id_key', type=str, default='claim_id', help='Key for claim ID in data')
    
    # Predict parser
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    predict_parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    predict_parser.add_argument('--output_file', type=str, required=True, help='Output file')
    predict_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    predict_parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    predict_parser.add_argument('--use_temporal', action='store_true', help='Use temporal features')
    predict_parser.add_argument('--label_key', type=str, default='label', help='Key for labels in data')
    predict_parser.add_argument('--text_key', type=str, default='claim_text', help='Key for claim text in data')
    predict_parser.add_argument('--id_key', type=str, default='claim_id', help='Key for claim ID in data')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate function
    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 