#!/usr/bin/env python3
# intermediate1_dataset_analysis_41.py
# Analysis of binary classification dataset - Class Distribution (4.1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "intermediate_1")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define file paths
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
VALID_PATH = os.path.join(DATA_DIR, "valid.csv")
REPORT_PATH = os.path.join(SCRIPT_DIR, "intermediate1_dataset_analysis_41.txt")

def load_data():
    """Load train, test, and validation datasets"""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    
    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")
    print(f"Validation set: {valid_df.shape[0]} samples")
    
    return train_df, test_df, valid_df

def analyze_class_distribution(train_df, test_df, valid_df):
    """
    4.1 Analyze and visualize the class distribution in the binary classification dataset
    """
    section_content = []
    section_content.append(f"# 4.1 Binary Class Distribution Analysis\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Count binary labels in each dataset
    train_counts = train_df['label'].value_counts()
    test_counts = test_df['label'].value_counts()
    valid_counts = valid_df['label'].value_counts()
    
    # Combine all datasets for overall counts
    combined_df = pd.concat([train_df, test_df, valid_df])
    total_counts = combined_df['label'].value_counts()
    
    # Calculate percentages
    train_pcts = (train_counts / train_counts.sum() * 100).round(2)
    test_pcts = (test_counts / test_counts.sum() * 100).round(2)
    valid_pcts = (valid_counts / valid_counts.sum() * 100).round(2)
    total_pcts = (total_counts / total_counts.sum() * 100).round(2)
    
    # Create a report
    section_content.append("## Binary Class Distribution\n")
    
    section_content.append("### Train Set")
    for label, count in train_counts.items():
        section_content.append(f"- {label}: {count} samples ({train_pcts[label]}%)")
    section_content.append(f"- Total: {train_counts.sum()} samples\n")
    
    section_content.append("### Test Set")
    for label, count in test_counts.items():
        section_content.append(f"- {label}: {count} samples ({test_pcts[label]}%)")
    section_content.append(f"- Total: {test_counts.sum()} samples\n")
    
    section_content.append("### Validation Set")
    for label, count in valid_counts.items():
        section_content.append(f"- {label}: {count} samples ({valid_pcts[label]}%)")
    section_content.append(f"- Total: {valid_counts.sum()} samples\n")
    
    section_content.append("### Overall")
    for label, count in total_counts.items():
        section_content.append(f"- {label}: {count} samples ({total_pcts[label]}%)")
    section_content.append(f"- Total: {total_counts.sum()} samples\n")
    
    # Check for class imbalance
    majority_class = total_counts.idxmax()
    minority_class = total_counts.idxmin()
    imbalance_ratio = (total_counts[majority_class] / total_counts[minority_class]).round(2)
    
    section_content.append("### Class Imbalance Analysis")
    section_content.append(f"- Majority class: {majority_class} with {total_counts[majority_class]} samples")
    section_content.append(f"- Minority class: {minority_class} with {total_counts[minority_class]} samples")
    section_content.append(f"- Imbalance ratio (majority:minority): {imbalance_ratio}")
    
    if imbalance_ratio > 1.5:
        section_content.append("- **Recommendation**: Consider balancing classes using techniques like oversampling, undersampling, or SMOTE")
    else:
        section_content.append("- **Note**: Class distribution is relatively balanced (imbalance ratio < 1.5)\n")
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Prepare data for plotting
    datasets = ['Train', 'Test', 'Validation', 'Overall']
    true_counts = [train_counts.get('TRUE', 0), test_counts.get('TRUE', 0), 
                   valid_counts.get('TRUE', 0), total_counts.get('TRUE', 0)]
    false_counts = [train_counts.get('FALSE', 0), test_counts.get('FALSE', 0), 
                    valid_counts.get('FALSE', 0), total_counts.get('FALSE', 0)]
    
    # Create grouped bar chart
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, true_counts, width, label='TRUE', color='green', alpha=0.7)
    rects2 = ax.bar(x + width/2, false_counts, width, label='FALSE', color='red', alpha=0.7)
    
    # Add labels and title
    ax.set_title('Binary Class Distribution across Datasets', fontsize=16)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    
    # Add count labels on bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(rects1)
    add_labels(rects2)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(PLOTS_DIR, "4.1_binary_class_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Binary Class Distribution]({os.path.relpath(plot_path, SCRIPT_DIR)})\n")
    
    # Create pie charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Define data for pie charts
    pie_data = [
        (train_counts, 'Train Set'),
        (test_counts, 'Test Set'),
        (valid_counts, 'Validation Set'),
        (total_counts, 'Overall')
    ]
    
    # Create each pie chart
    for i, (counts, title) in enumerate(pie_data):
        axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', 
                   colors=['green', 'red'], explode=[0.05, 0.05], 
                   shadow=True, startangle=90)
        axes[i].set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    pie_chart_path = os.path.join(PLOTS_DIR, "4.1_binary_class_distribution_pie.png")
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Binary Class Distribution - Pie Charts]({os.path.relpath(pie_chart_path, SCRIPT_DIR)})\n")
    
    # Save the report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(section_content))
    
    print(f"Class distribution analysis report saved to: {REPORT_PATH}")
    
    return combined_df

if __name__ == "__main__":
    print("Starting analysis of binary classification dataset - Class Distribution (4.1)...")
    
    # Load the data
    print("Loading data...")
    train_df, test_df, valid_df = load_data()
    
    # Analyze class distribution
    print("Analyzing class distribution...")
    combined_df = analyze_class_distribution(train_df, test_df, valid_df)
    
    print("Analysis complete!") 