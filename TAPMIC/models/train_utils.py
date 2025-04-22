#!/usr/bin/env python3
# train_utils.py
# Utility functions for training RoBERTa models for TAPMIC project

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import wandb
import logging
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Import wandb setup functions
from wandb_setup import (
    log_batch_metrics, 
    log_epoch_metrics, 
    log_confusion_matrix,
    log_feature_importance,
    log_model_summary,
    log_prediction_samples,
    log_temporal_feature_analysis
)

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

def train_with_mixed_precision(model, train_loader, val_loader, optimizer, scheduler, 
                               device, epochs=5, grad_accumulation_steps=2, early_stopping_patience=3):
    """
    Train the model with mixed precision for faster training on GPUs
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        epochs: Number of epochs
        grad_accumulation_steps: Gradient accumulation steps
        early_stopping_patience: Early stopping patience
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rate': []
    }
    
    # Log model summary to wandb if wandb is initialized
    if wandb.run is not None:
        log_model_summary(model)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                # Forward pass with or without additional features
                if 'features' in batch:
                    features = batch['features'].to(device)
                    outputs = model(input_ids, attention_mask, features)
                else:
                    outputs = model(input_ids, attention_mask)
                    
                # Calculate loss
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accumulation_steps
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            
            # Update weights every grad_accumulation_steps
            if (step + 1) % grad_accumulation_steps == 0:
                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            # Track loss
            train_loss += loss.item() * grad_accumulation_steps
            train_steps += 1
            
            # Log batch metrics
            if step % 100 == 0:
                logging.info(f"Batch {step}/{len(train_loader)} - Loss: {loss.item() * grad_accumulation_steps:.4f}")
                if wandb.run is not None:
                    log_batch_metrics(
                        batch_idx=step, 
                        batch_loss=loss.item() * grad_accumulation_steps,
                        epoch=epoch,
                        total_batches=len(train_loader)
                    )
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_texts = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Store texts for sample predictions
                if 'text' in batch:
                    all_texts.extend(batch['text'])
                
                # Forward pass with or without additional features (no mixed precision for evaluation)
                if 'features' in batch:
                    features = batch['features'].to(device)
                    outputs = model(input_ids, attention_mask, features)
                else:
                    outputs = model(input_ids, attention_mask)
                
                # Calculate loss
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                
                # Track loss
                val_loss += loss.item()
                val_steps += 1
                
                # Calculate predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions, labels, and probabilities
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate average validation loss and metrics
        avg_val_loss = val_loss / val_steps
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(accuracy)
        history['val_f1'].append(f1)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        # Log epoch metrics
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        # Log metrics to wandb
        if wandb.run is not None:
            log_epoch_metrics(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
                accuracy=accuracy,
                f1=f1,
                precision=precision,
                recall=recall,
                learning_rate=scheduler.get_last_lr()[0]
            )
            
            # Log confusion matrix every epoch
            log_confusion_matrix(all_labels, all_preds, epoch=epoch+1)
            
            # Log sample predictions
            if len(all_texts) > 0:
                log_prediction_samples(
                    texts=all_texts[:100],  # Limit to 100 samples
                    true_labels=all_labels[:100],
                    predicted_labels=all_preds[:100],
                    probabilities=all_probs[:100]
                )
            
            # Log temporal feature analysis if available
            if hasattr(train_loader.dataset, 'temporal_cols') and 'features' in batch:
                for i, feature_name in enumerate(train_loader.dataset.temporal_cols):
                    if i < features.shape[1]:  # Make sure we're within bounds
                        feature_values = features[:, i].cpu().numpy()
                        log_temporal_feature_analysis(feature_name, feature_values, labels.cpu().numpy())
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            logging.info(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}")
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the model
            model_path = os.path.join(BASE_DIR, "models", "checkpoints", f"roberta_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            logging.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logging.info("Early stopping triggered")
                break
    
    return model, history

def evaluate_and_report(model, test_loader, device, run_name, feature_mode):
    """
    Evaluate the model and generate comprehensive reports
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on (cuda/cpu)
        run_name: Name of the run for saving reports
        feature_mode: String describing the feature mode (e.g., "text_only", "text_temporal")
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    logging.info("Evaluating model...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass with or without additional features
            if 'features' in batch:
                features = batch['features'].to(device)
                outputs = model(input_ids, attention_mask, features)
            else:
                outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Track loss
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions, probabilities, and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    
    # Generate classification report
    class_report = classification_report(all_labels, all_preds, target_names=['FALSE', 'TRUE'], output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    
    # Log results
    logging.info(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    logging.info(f"F1-Score (TRUE): {class_report['TRUE']['f1-score']:.4f}")
    logging.info(f"F1-Score (FALSE): {class_report['FALSE']['f1-score']:.4f}")
    logging.info(f"AUC: {roc_auc:.4f}")
    
    if wandb.run is not None:
        wandb.log({
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "f1_true": class_report['TRUE']['f1-score'],
            "f1_false": class_report['FALSE']['f1-score'],
            "precision_true": class_report['TRUE']['precision'],
            "recall_true": class_report['TRUE']['recall'],
            "auc": roc_auc
        })
    
    # Create visualizations
    plot_confusion_matrix(cm, run_name, feature_mode)
    plot_roc_curve(fpr, tpr, roc_auc, run_name, feature_mode)
    plot_precision_recall_curve(precision, recall, run_name, feature_mode)
    
    # Save detailed report to file
    report_path = os.path.join(REPORTS_DIR, f"{run_name}_{feature_mode}_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"=== EVALUATION REPORT: {run_name} - {feature_mode} ===\n\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=['FALSE', 'TRUE']))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'f1_true': class_report['TRUE']['f1-score'],
        'f1_false': class_report['FALSE']['f1-score'],
        'precision_true': class_report['TRUE']['precision'],
        'recall_true': class_report['TRUE']['recall'],
        'auc': roc_auc,
        'confusion_matrix': cm,
        'roc': {
            'fpr': fpr,
            'tpr': tpr
        },
        'pr': {
            'precision': precision,
            'recall': recall
        }
    }
    
    return metrics

def analyze_feature_importance(model, dataloader, feature_names, device, run_name):
    """
    Analyze feature importance using a permutation-based approach
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        feature_names: List of feature names
        device: Device to run on
        run_name: Name of the run
        
    Returns:
        feature_importance: Dictionary mapping feature names to importance scores
    """
    if not hasattr(dataloader.dataset, 'feature_cols') or len(dataloader.dataset.feature_cols) == 0:
        logging.info("No numerical features to analyze")
        return {}
    
    logging.info("Analyzing feature importance...")
    
    # Get baseline performance
    model.eval()
    baseline_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Baseline"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            features = batch['features'].to(device)
            
            outputs = model(input_ids, attention_mask, features)
            _, predicted = torch.max(outputs, 1)
            baseline_correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    baseline_accuracy = baseline_correct / total
    logging.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Initialize feature importance dictionary
    feature_importance = {}
    
    # Analyze importance of each feature
    for i, feature_name in enumerate(feature_names):
        logging.info(f"Analyzing importance of {feature_name}...")
        
        # Store original feature values
        original_values = []
        for batch in dataloader:
            original_values.append(batch['features'][:, i].clone())
        
        # Permute the feature
        permuted_correct = 0
        batch_idx = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Permuting {feature_name}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                features = batch['features'].clone().to(device)
                
                # Permute the specific feature
                perm_indices = torch.randperm(features.size(0))
                features[:, i] = features[perm_indices, i]
                
                outputs = model(input_ids, attention_mask, features)
                _, predicted = torch.max(outputs, 1)
                permuted_correct += (predicted == labels).sum().item()
                batch_idx += 1
        
        permuted_accuracy = permuted_correct / total
        importance = baseline_accuracy - permuted_accuracy
        feature_importance[feature_name] = importance
        
        logging.info(f"Importance of {feature_name}: {importance:.4f}")
    
    # Sort features by importance
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    # Plot feature importance
    plot_feature_importance(sorted_importance, run_name)
    
    # Log feature importance to wandb
    if wandb.run is not None:
        log_feature_importance(feature_names, sorted_importance)
    
    return sorted_importance

def plot_confusion_matrix(cm, run_name, feature_mode):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['FALSE', 'TRUE'], 
                yticklabels=['FALSE', 'TRUE'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {feature_mode}')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(REPORTS_DIR, f"{run_name}_{feature_mode}_confusion_matrix.png"), dpi=300)
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.Image(plt)})
    
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, run_name, feature_mode):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {feature_mode}')
    plt.legend(loc="lower right")
    
    # Save figure
    plt.savefig(os.path.join(REPORTS_DIR, f"{run_name}_{feature_mode}_roc_curve.png"), dpi=300)
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({"roc_curve": wandb.Image(plt)})
    
    plt.close()

def plot_precision_recall_curve(precision, recall, run_name, feature_mode):
    """Plot precision-recall curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {feature_mode}')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    # Save figure
    plt.savefig(os.path.join(REPORTS_DIR, f"{run_name}_{feature_mode}_pr_curve.png"), dpi=300)
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({"pr_curve": wandb.Image(plt)})
    
    plt.close()

def plot_feature_importance(feature_importance, run_name):
    """Plot feature importance"""
    plt.figure(figsize=(12, 8))
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Limit to top 20 features for readability
    if len(features) > 20:
        features = features[:20]
        importance = importance[:20]
    
    # Create horizontal bar chart
    colors = ['green' if imp > 0 else 'red' for imp in importance]
    plt.barh(features, importance, color=colors)
    plt.xlabel('Importance (Drop in Accuracy)')
    plt.title('Feature Importance')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(REPORTS_DIR, f"{run_name}_feature_importance.png"), dpi=300)
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({"feature_importance_plot": wandb.Image(plt)})
    
    plt.close()

def plot_training_history(history, run_name):
    """Plot training history"""
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 subplot
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(REPORTS_DIR, f"{run_name}_training_history.png"), dpi=300)
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log({"training_history": wandb.Image(plt)})
    
    plt.close()

def save_model_report(metrics, history, feature_importance, run_name, feature_mode):
    """Save comprehensive model report to CSV"""
    # Create dataframe for model performances
    df = pd.DataFrame({
        'run_name': [run_name],
        'feature_mode': [feature_mode],
        'accuracy': [metrics['accuracy']],
        'loss': [metrics['loss']],
        'f1_true': [metrics['f1_true']],
        'f1_false': [metrics['f1_false']],
        'precision_true': [metrics['precision_true']],
        'recall_true': [metrics['recall_true']],
        'auc': [metrics['auc']]
    })
    
    # Save to CSV
    csv_path = os.path.join(REPORTS_DIR, "model_performance.csv")
    
    # Append to existing file if it exists
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
    logging.info(f"Model performance saved to {csv_path}")
    
    # Also save feature importance if available
    if feature_importance:
        fi_df = pd.DataFrame({
            'feature': list(feature_importance.keys()),
            'importance': list(feature_importance.values())
        })
        fi_df['run_name'] = run_name
        fi_df['feature_mode'] = feature_mode
        
        fi_csv_path = os.path.join(REPORTS_DIR, f"{run_name}_{feature_mode}_feature_importance.csv")
        fi_df.to_csv(fi_csv_path, index=False)
        logging.info(f"Feature importance saved to {fi_csv_path}")
    
    # Save training history
    if history:
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(os.path.join(REPORTS_DIR, f"{run_name}_{feature_mode}_training_history.csv"), index=False)
        logging.info(f"Training history saved to CSV") 