#!/usr/bin/env python3
# evaluation_metrics.py
# Comprehensive evaluation metrics for TAPMIC models

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc
)
from sklearn.model_selection import StratifiedKFold
import logging
import torch
import json
import wandb
from typing import Dict, List, Tuple, Union, Optional

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

class ModelEvaluator:
    """Comprehensive model evaluator for TAPMIC models"""
    
    def __init__(self, model, device, run_name=None, feature_mode=None):
        """
        Initialize evaluator
        
        Args:
            model: PyTorch model to evaluate
            device: Device to run evaluation on
            run_name: Name of the run for saving reports
            feature_mode: Feature mode (text_only, text_temporal, etc.)
        """
        self.model = model
        self.device = device
        self.run_name = run_name or "model_evaluation"
        self.feature_mode = feature_mode or "unknown"
        
        # Create report directory
        self.report_dir = os.path.join(REPORTS_DIR, self.run_name)
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize result dictionaries
        self.metrics = {}
        self.temporal_metrics = {}
        self.cross_val_metrics = {}
    
    def evaluate_model(self, dataloader, output_report=True):
        """
        Evaluate model on the given dataloader
        
        Args:
            dataloader: PyTorch dataloader containing evaluation data
            output_report: Whether to save the report
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Initialize lists to store outputs
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Collect texts if available for sample prediction analysis
        all_texts = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Store texts if available
                if 'text' in batch:
                    all_texts.extend(batch['text'])
                
                # Forward pass
                if 'features' in batch:
                    features = batch['features'].to(self.device)
                    outputs = self.model(input_ids, attention_mask, features)
                else:
                    outputs = self.model(input_ids, attention_mask)
                
                # Calculate predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions, labels, and probabilities
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Output report if requested
        if output_report:
            self._save_report(all_labels, all_preds, all_probs, all_texts)
        
        return self.metrics
    
    def evaluate_with_temporal_breakdown(self, dataloader, temporal_feature_name):
        """
        Evaluate model with breakdown by temporal feature values
        
        Args:
            dataloader: PyTorch dataloader containing evaluation data
            temporal_feature_name: Name of the temporal feature to use for breakdown
            
        Returns:
            temporal_metrics: Dictionary of metrics broken down by temporal categories
        """
        self.model.eval()
        
        # Initialize lists to store outputs and feature values
        all_preds = []
        all_labels = []
        all_probs = []
        all_feature_values = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get feature index
                if 'features' in batch and hasattr(dataloader.dataset, 'temporal_cols'):
                    features = batch['features'].to(self.device)
                    try:
                        feature_idx = dataloader.dataset.temporal_cols.index(temporal_feature_name)
                        all_feature_values.extend(features[:, feature_idx].cpu().numpy())
                    except (ValueError, IndexError):
                        logging.warning(f"Feature {temporal_feature_name} not found in temporal_cols")
                        return {}
                    
                    # Forward pass with features
                    outputs = self.model(input_ids, attention_mask, features)
                else:
                    logging.warning("Features not available for temporal breakdown")
                    return {}
                
                # Calculate predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store predictions, labels, and probabilities
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Create dataframe for analysis
        eval_df = pd.DataFrame({
            'label': all_labels,
            'pred': all_preds,
            'prob': all_probs,
            'feature': all_feature_values
        })
        
        # Define categories based on the feature values
        if temporal_feature_name == 'days_to_nearest_election':
            # Group by election proximity
            bins = [-np.inf, 30, 90, 180, np.inf]
            labels = ["< 30 days", "30-90 days", "90-180 days", "> 180 days"]
            eval_df['category'] = pd.cut(eval_df['feature'], bins=bins, labels=labels)
        elif 'year' in temporal_feature_name.lower():
            # Group by year ranges
            bins = [1990, 2000, 2010, 2020, 2025]
            labels = ["1990-2000", "2000-2010", "2010-2020", "2020-2025"]
            eval_df['category'] = pd.cut(eval_df['feature'], bins=bins, labels=labels)
        else:
            # Generic quantile-based grouping for other features
            quantiles = np.quantile(eval_df['feature'], [0, 0.25, 0.5, 0.75, 1])
            eval_df['category'] = pd.qcut(eval_df['feature'], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        
        # Calculate metrics for each category
        temporal_metrics = {}
        categories = eval_df['category'].dropna().unique()
        
        for category in categories:
            category_df = eval_df[eval_df['category'] == category]
            if len(category_df) > 0:
                metrics = self._calculate_metrics(
                    category_df['label'].values,
                    category_df['pred'].values,
                    category_df['prob'].values
                )
                temporal_metrics[str(category)] = metrics
        
        # Save temporal breakdown
        self.temporal_metrics[temporal_feature_name] = temporal_metrics
        
        # Generate and save visualization
        self._visualize_temporal_breakdown(temporal_feature_name, eval_df)
        
        return temporal_metrics
    
    def evaluate_with_cross_validation(self, dataset, n_splits=5, batch_size=16):
        """
        Evaluate model using stratified k-fold cross-validation
        
        Args:
            dataset: Dataset to evaluate on
            n_splits: Number of folds for cross-validation
            batch_size: Batch size for evaluation
            
        Returns:
            cross_val_metrics: Dictionary of cross-validation metrics
        """
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Get all data from dataset
        all_data = []
        all_labels = []
        for i in range(len(dataset)):
            data = dataset[i]
            all_data.append(data)
            all_labels.append(data['label'].item())
        
        # Initialize metrics storage
        fold_metrics = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
            logging.info(f"Evaluating fold {fold+1}/{n_splits}")
            
            # Create validation dataloader for this fold
            from torch.utils.data import DataLoader, Subset
            val_dataset = Subset(dataset, val_idx)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Evaluate on this fold
            metrics = self.evaluate_model(val_loader, output_report=False)
            fold_metrics.append(metrics)
            
            logging.info(f"Fold {fold+1} accuracy: {metrics['accuracy']:.4f}")
        
        # Calculate average metrics across folds
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            if isinstance(fold_metrics[0][key], (int, float)):
                avg_metrics[key] = sum(fold[key] for fold in fold_metrics) / len(fold_metrics)
        
        # Calculate standard deviation
        std_metrics = {}
        for key in avg_metrics.keys():
            std_metrics[key] = np.std([fold[key] for fold in fold_metrics])
        
        # Store cross-validation results
        self.cross_val_metrics = {
            'average': avg_metrics,
            'std_dev': std_metrics,
            'folds': fold_metrics
        }
        
        # Save cross-validation report
        self._save_cross_val_report()
        
        return self.cross_val_metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            
        Returns:
            metrics: Dictionary of metrics
        """
        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'average_precision': average_precision_score(y_true, y_prob)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # Calculate derived metrics
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics
    
    def _save_report(self, y_true, y_pred, y_prob, texts=None):
        """
        Save comprehensive evaluation report
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            texts: Optional list of input texts
        """
        # Create report path
        report_path = os.path.join(self.report_dir, f"evaluation_{self.feature_mode}.json")
        
        # Add classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create full results dictionary
        results = {
            'run_name': self.run_name,
            'feature_mode': self.feature_mode,
            'metrics': self.metrics,
            'classification_report': report
        }
        
        # Save report to JSON
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save confusion matrix visualization
        cm = confusion_matrix(y_true, y_pred)
        self._plot_confusion_matrix(cm)
        
        # Save ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        self._plot_roc_curve(fpr, tpr, roc_auc)
        
        # Save precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        self._plot_precision_recall_curve(precision, recall)
        
        # Log to wandb if available
        if wandb.run is not None:
            from wandb_setup import log_confusion_matrix, log_prediction_samples
            log_confusion_matrix(y_true, y_pred)
            
            if texts:
                log_prediction_samples(
                    texts=texts[:100],  # Limit to 100 samples
                    true_labels=y_true[:100],
                    predicted_labels=y_pred[:100],
                    probabilities=y_prob[:100]
                )
        
        logging.info(f"Evaluation report saved to {report_path}")
        
        # Print summary metrics
        self._print_metrics_summary()
    
    def _save_cross_val_report(self):
        """Save cross-validation report"""
        # Create report path
        report_path = os.path.join(self.report_dir, f"cross_val_{self.feature_mode}.json")
        
        # Save report to JSON
        with open(report_path, 'w') as f:
            json.dump(self.cross_val_metrics, f, indent=2)
        
        # Create visualization of cross-validation results
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        means = [self.cross_val_metrics['average'][m] for m in metrics_to_plot]
        stds = [self.cross_val_metrics['std_dev'][m] for m in metrics_to_plot]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_to_plot, means, yerr=stds, capsize=10, color='skyblue')
        
        # Add mean values on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.01,
                    f"{means[i]:.4f}", ha='center', va='bottom', fontsize=10)
        
        plt.title('Cross-Validation Results')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        cv_plot_path = os.path.join(self.report_dir, f"cross_val_plot_{self.feature_mode}.png")
        plt.savefig(cv_plot_path)
        plt.close()
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "cross_val_metrics": wandb.Table(
                    columns=["Metric", "Mean", "Std Dev"],
                    data=[[m, self.cross_val_metrics['average'][m], self.cross_val_metrics['std_dev'][m]] 
                          for m in metrics_to_plot]
                ),
                "cross_val_plot": wandb.Image(cv_plot_path)
            })
        
        logging.info(f"Cross-validation report saved to {report_path}")
    
    def _visualize_temporal_breakdown(self, feature_name, eval_df):
        """
        Create visualization of metrics by temporal category
        
        Args:
            feature_name: Name of the temporal feature
            eval_df: DataFrame with evaluation results
        """
        # Extract metrics for each category
        categories = []
        accuracies = []
        f1_scores = []
        
        for category, metrics in self.temporal_metrics[feature_name].items():
            categories.append(category)
            accuracies.append(metrics['accuracy'])
            f1_scores.append(metrics['f1'])
        
        # Create grouped bar chart
        plt.figure(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        plt.bar(x + width/2, f1_scores, width, label='F1 Score', color='lightcoral')
        
        plt.xlabel('Temporal Category')
        plt.ylabel('Score')
        plt.title(f'Performance by {feature_name}')
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.report_dir, f"temporal_{feature_name}_{self.feature_mode}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                f"temporal_{feature_name}": wandb.Image(plot_path),
                f"temporal_metrics_{feature_name}": wandb.Table(
                    columns=["Category", "Accuracy", "F1 Score"],
                    data=[[cat, acc, f1] for cat, acc, f1 in zip(categories, accuracies, f1_scores)]
                )
            })
    
    def _plot_confusion_matrix(self, cm):
        """
        Plot and save confusion matrix
        
        Args:
            cm: Confusion matrix array
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['False', 'True'],
                    yticklabels=['False', 'True'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.report_dir, f"confusion_matrix_{self.feature_mode}.png")
        plt.savefig(plot_path)
        plt.close()
    
    def _plot_roc_curve(self, fpr, tpr, roc_auc):
        """
        Plot and save ROC curve
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under ROC curve
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.report_dir, f"roc_curve_{self.feature_mode}.png")
        plt.savefig(plot_path)
        plt.close()
    
    def _plot_precision_recall_curve(self, precision, recall):
        """
        Plot and save precision-recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
        """
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.report_dir, f"precision_recall_{self.feature_mode}.png")
        plt.savefig(plot_path)
        plt.close()
    
    def _print_metrics_summary(self):
        """Print summary of evaluation metrics"""
        print("\n" + "="*50)
        print(f"Evaluation Summary ({self.feature_mode})")
        print("="*50)
        print(f"Accuracy:    {self.metrics['accuracy']:.4f}")
        print(f"F1 Score:    {self.metrics['f1']:.4f}")
        print(f"Precision:   {self.metrics['precision']:.4f}")
        print(f"Recall:      {self.metrics['recall']:.4f}")
        print(f"ROC AUC:     {self.metrics['roc_auc']:.4f}")
        print(f"Avg Prec:    {self.metrics['average_precision']:.4f}")
        print(f"Balanced Acc: {self.metrics['balanced_accuracy']:.4f}")
        print("="*50)
        print(f"Confusion Matrix:")
        print(f"TN: {self.metrics['tn']}, FP: {self.metrics['fp']}")
        print(f"FN: {self.metrics['fn']}, TP: {self.metrics['tp']}")
        print("="*50 + "\n")

def main():
    """Example usage of evaluation_metrics.py"""
    # This would be typically called from another script
    print("This script provides evaluation metrics for TAPMIC models.")
    print("Import and use the ModelEvaluator class in your training scripts.")

if __name__ == "__main__":
    main() 