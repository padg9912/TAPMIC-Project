#!/usr/bin/env python3
# wandb_setup.py
# Configuration and setup for Weights & Biases tracking for TAPMIC project

import os
import sys
import logging
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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

def initialize_wandb(run_name, feature_mode, config=None, resume=False):
    """
    Initialize Weights & Biases for tracking experiments
    
    Args:
        run_name (str): Name for this run
        feature_mode (str): Feature mode (text_only, text_temporal, text_temporal_credibility)
        config (dict): Configuration dictionary for wandb run
        resume (bool): Whether to resume a previous run
        
    Returns:
        run: Wandb run object
    """
    # Default configuration
    default_config = {
        "model_name": "roberta-base",
        "max_length": 128,
        "batch_size": 16,
        "epochs": 5,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "dropout_rate": 0.1,
        "grad_accumulation_steps": 2,
        "early_stopping_patience": 3,
        "feature_mode": feature_mode
    }
    
    # Use provided config if available, otherwise use default
    run_config = config if config is not None else default_config
    
    # Initialize wandb run
    run = wandb.init(
        project="TAPMIC",
        name=f"{run_name}_{feature_mode}",
        config=run_config,
        resume=resume
    )
    
    logging.info(f"Initialized wandb run: {run.name}")
    
    return run

def log_batch_metrics(batch_idx, batch_loss, epoch, total_batches):
    """
    Log batch-level metrics to wandb
    
    Args:
        batch_idx (int): Batch index
        batch_loss (float): Loss for this batch
        epoch (int): Current epoch
        total_batches (int): Total number of batches
    """
    wandb.log({
        "batch": batch_idx,
        "batch_loss": batch_loss,
        "epoch_progress": batch_idx / total_batches,
        "epoch": epoch
    })

def log_epoch_metrics(epoch, train_loss, val_loss, accuracy, f1, precision, recall, learning_rate):
    """
    Log epoch-level metrics to wandb
    
    Args:
        epoch (int): Current epoch
        train_loss (float): Training loss
        val_loss (float): Validation loss
        accuracy (float): Validation accuracy
        f1 (float): Validation F1 score
        precision (float): Validation precision
        recall (float): Validation recall
        learning_rate (float): Current learning rate
    """
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "learning_rate": learning_rate
    })

def log_confusion_matrix(y_true, y_pred, epoch=None):
    """
    Log confusion matrix visualization to wandb
    
    Args:
        y_true (array): Ground truth labels
        y_pred (array): Predicted labels
        epoch (int): Current epoch (optional)
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['False', 'True'],
                yticklabels=['False', 'True'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Log to wandb
    if epoch is not None:
        wandb.log({f"confusion_matrix_epoch_{epoch}": wandb.Image(plt)})
    else:
        wandb.log({"confusion_matrix": wandb.Image(plt)})
    
    plt.close()

def log_feature_importance(feature_names, importance_scores):
    """
    Log feature importance visualization to wandb
    
    Args:
        feature_names (list): List of feature names
        importance_scores (array): Feature importance scores
    """
    # Create dataframe for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"feature_importance": wandb.Image(plt)})
    
    # Also log as a table
    wandb.log({"feature_importance_table": wandb.Table(dataframe=importance_df)})
    
    plt.close()

def log_temporal_feature_analysis(feature_name, feature_values, labels):
    """
    Log analysis of temporal feature vs. truth labels
    
    Args:
        feature_name (str): Name of the temporal feature
        feature_values (array): Feature values
        labels (array): Truth labels
    """
    # Create dataframe for visualization
    df = pd.DataFrame({
        'Feature': feature_values,
        'Label': labels
    })
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Feature', hue='Label', bins=20, multiple='dodge')
    plt.title(f'Distribution of {feature_name} by Label')
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({f"temporal_feature_{feature_name}": wandb.Image(plt)})
    
    # Calculate correlation
    correlation = np.corrcoef(feature_values, labels)[0, 1]
    wandb.log({f"correlation_{feature_name}": correlation})
    
    plt.close()

def log_model_summary(model):
    """
    Log model architecture summary to wandb
    
    Args:
        model: PyTorch model
    """
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log parameter counts
    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    })
    
    # Log model architecture
    wandb.log({"model_summary": wandb.Table(data=[[str(model)]], columns=["Model Architecture"])})

def log_prediction_samples(texts, true_labels, predicted_labels, probabilities, sample_count=10):
    """
    Log sample predictions to wandb
    
    Args:
        texts (list): List of input texts
        true_labels (list): List of true labels
        predicted_labels (list): List of predicted labels
        probabilities (list): List of prediction probabilities
        sample_count (int): Number of samples to log
    """
    # Get indices of correct and incorrect predictions
    correct_indices = [i for i, (t, p) in enumerate(zip(true_labels, predicted_labels)) if t == p]
    incorrect_indices = [i for i, (t, p) in enumerate(zip(true_labels, predicted_labels)) if t != p]
    
    # Sample from correct and incorrect predictions
    sample_indices = []
    if len(correct_indices) > 0:
        sample_indices.extend(np.random.choice(correct_indices, min(sample_count // 2, len(correct_indices)), replace=False))
    if len(incorrect_indices) > 0:
        sample_indices.extend(np.random.choice(incorrect_indices, min(sample_count // 2, len(incorrect_indices)), replace=False))
    
    # Create table data
    table_data = []
    for idx in sample_indices:
        table_data.append([
            texts[idx],
            "True" if true_labels[idx] == 1 else "False",
            "True" if predicted_labels[idx] == 1 else "False",
            f"{probabilities[idx]:.4f}",
            "Correct" if true_labels[idx] == predicted_labels[idx] else "Incorrect"
        ])
    
    # Log to wandb
    columns = ["Text", "True Label", "Predicted Label", "Probability", "Status"]
    wandb.log({"prediction_samples": wandb.Table(columns=columns, data=table_data)})

def finish_run():
    """
    Finish the wandb run
    """
    wandb.finish()
    logging.info("Wandb run completed")

def main():
    """Example usage of wandb_setup.py"""
    # Initialize wandb
    run = initialize_wandb("example_run", "text_temporal")
    
    # Log some metrics
    for epoch in range(5):
        log_epoch_metrics(
            epoch=epoch,
            train_loss=0.5 - epoch * 0.1,
            val_loss=0.6 - epoch * 0.1,
            accuracy=0.7 + epoch * 0.05,
            f1=0.65 + epoch * 0.06,
            precision=0.7 + epoch * 0.04,
            recall=0.65 + epoch * 0.05,
            learning_rate=0.001 * (0.9 ** epoch)
        )
    
    # Log confusion matrix
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    log_confusion_matrix(y_true, y_pred)
    
    # Log feature importance
    feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    importance_scores = np.random.rand(5)
    log_feature_importance(feature_names, importance_scores)
    
    # Finish run
    finish_run()

if __name__ == "__main__":
    main() 