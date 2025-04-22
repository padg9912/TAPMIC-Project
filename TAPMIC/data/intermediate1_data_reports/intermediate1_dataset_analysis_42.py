#!/usr/bin/env python3
# intermediate1_dataset_analysis_42.py
# Analysis of binary classification dataset - Basic Statistics (4.2)

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
REPORT_PATH = os.path.join(SCRIPT_DIR, "intermediate1_dataset_analysis_42.txt")

def load_data():
    """Load train, test, and validation datasets"""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    
    # Combine all datasets for overall statistics
    combined_df = pd.concat([train_df, test_df, valid_df])
    
    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")
    print(f"Validation set: {valid_df.shape[0]} samples")
    print(f"Combined dataset: {combined_df.shape[0]} samples")
    
    return train_df, test_df, valid_df, combined_df

def analyze_basic_statistics(train_df, test_df, valid_df, combined_df):
    """
    4.2 Calculate and visualize basic statistics for the dataset
    """
    # DEBUG: Print unique label values
    print("Unique label values in dataset:", combined_df['label'].unique())
    
    section_content = []
    section_content.append(f"# 4.2 Basic Statistics Analysis\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get a list of numerical and categorical columns
    numerical_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Remove ID and specific count columns for separate analysis
    count_cols = [col for col in numerical_cols if 'counts' in col]
    other_numerical = [col for col in numerical_cols if col not in count_cols and col != 'id']
    
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    # Exclude label, original_label, and statement for separate analysis
    special_categorical = ['label', 'original_label', 'statement']
    categorical_cols = [col for col in categorical_cols if col not in special_categorical]
    
    # 1. Dataset Shape Information
    section_content.append("## 1. Dataset Shape Information")
    section_content.append(f"- Train set: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    section_content.append(f"- Test set: {test_df.shape[0]} rows × {test_df.shape[1]} columns")
    section_content.append(f"- Validation set: {valid_df.shape[0]} rows × {valid_df.shape[1]} columns")
    section_content.append(f"- Combined dataset: {combined_df.shape[0]} rows × {combined_df.shape[1]} columns\n")
    
    # List all columns
    section_content.append("### Column List")
    columns_by_type = {
        "Numerical": other_numerical,
        "Count-based": count_cols,
        "Categorical": categorical_cols,
        "Special": special_categorical
    }
    
    for col_type, cols in columns_by_type.items():
        section_content.append(f"**{col_type} columns:** {', '.join(cols)}\n")
    
    # 2. Statement Length Statistics
    section_content.append("## 2. Statement Length Analysis")
    
    # Calculate statement length statistics
    combined_df['statement_length'] = combined_df['statement'].apply(len)
    combined_df['word_count'] = combined_df['statement'].apply(lambda x: len(str(x).split()))
    
    # Generate statistics by label
    length_by_label = combined_df.groupby('label')[['statement_length', 'word_count']].agg(
        ['mean', 'median', 'min', 'max', 'std']).round(1)
    
    # Print label index for debugging
    print("Labels in the groupby index:", length_by_label.index.tolist())
    
    # Convert the statistics to a more readable format
    section_content.append("### Statement Length Statistics by Label (characters)")
    for label in combined_df['label'].unique():
        stats = length_by_label.loc[label, 'statement_length']
        section_content.append(f"**{label} statements:**")
        section_content.append(f"- Mean: {stats['mean']} chars")
        section_content.append(f"- Median: {stats['median']} chars")
        section_content.append(f"- Min: {stats['min']} chars")
        section_content.append(f"- Max: {stats['max']} chars")
        section_content.append(f"- Standard Deviation: {stats['std']} chars\n")
    
    section_content.append("### Word Count Statistics by Label")
    for label in combined_df['label'].unique():
        stats = length_by_label.loc[label, 'word_count']
        section_content.append(f"**{label} statements:**")
        section_content.append(f"- Mean: {stats['mean']} words")
        section_content.append(f"- Median: {stats['median']} words")
        section_content.append(f"- Min: {stats['min']} words")
        section_content.append(f"- Max: {stats['max']} words")
        section_content.append(f"- Standard Deviation: {stats['std']} words\n")
    
    # Visualize statement length and word count distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Statement length histogram by label
    sns.histplot(data=combined_df, x='statement_length', hue='label', 
                 bins=30, kde=True, ax=axes[0, 0], palette=['green', 'red'])
    axes[0, 0].set_title('Statement Length Distribution by Label', fontsize=14)
    axes[0, 0].set_xlabel('Statement Length (characters)', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    
    # Word count histogram by label
    sns.histplot(data=combined_df, x='word_count', hue='label', 
                 bins=30, kde=True, ax=axes[0, 1], palette=['green', 'red'])
    axes[0, 1].set_title('Word Count Distribution by Label', fontsize=14)
    axes[0, 1].set_xlabel('Word Count', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    
    # Boxplot for statement length
    sns.boxplot(data=combined_df, x='label', y='statement_length', 
                ax=axes[1, 0], palette=['green', 'red'])
    axes[1, 0].set_title('Statement Length by Label', fontsize=14)
    axes[1, 0].set_xlabel('Label', fontsize=12)
    axes[1, 0].set_ylabel('Statement Length (characters)', fontsize=12)
    
    # Boxplot for word count
    sns.boxplot(data=combined_df, x='label', y='word_count', 
                ax=axes[1, 1], palette=['green', 'red'])
    axes[1, 1].set_title('Word Count by Label', fontsize=14)
    axes[1, 1].set_xlabel('Label', fontsize=12)
    axes[1, 1].set_ylabel('Word Count', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    statement_stats_path = os.path.join(PLOTS_DIR, "4.2_statement_statistics.png")
    plt.savefig(statement_stats_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Statement Statistics]({os.path.relpath(statement_stats_path, SCRIPT_DIR)})\n")
    
    # 3. Speaker Truth History Analysis
    section_content.append("## 3. Speaker Truth History Analysis")
    
    # Analyze truth history columns
    truth_history_cols = [col for col in combined_df.columns if 'counts' in col]
    
    # Summary statistics for truth history
    truth_history_stats = combined_df[truth_history_cols].describe().round(2)
    
    section_content.append("### Truth History Statistics")
    section_content.append("These columns represent the speaker's history of truthfulness in previous statements:")
    
    for col in truth_history_cols:
        col_name = col.replace('_', ' ').title()
        section_content.append(f"**{col_name}:**")
        section_content.append(f"- Mean: {truth_history_stats.loc['mean', col]}")
        section_content.append(f"- Median: {truth_history_stats.loc['50%', col]}")
        section_content.append(f"- Min: {truth_history_stats.loc['min', col]}")
        section_content.append(f"- Max: {truth_history_stats.loc['max', col]}")
        section_content.append(f"- Standard Deviation: {truth_history_stats.loc['std', col]}\n")
    
    # Check for missing values in truth history
    missing_counts = combined_df[truth_history_cols].isnull().sum()
    section_content.append("### Missing Values in Truth History")
    for col, missing in missing_counts.items():
        percent_missing = round((missing / len(combined_df)) * 100, 2)
        section_content.append(f"- {col}: {missing} missing values ({percent_missing}%)")
    
    # Create correlation heatmap for truth history
    plt.figure(figsize=(12, 10))
    truth_history_corr = combined_df[truth_history_cols].corr()
    sns.heatmap(truth_history_corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation between Truth History Features', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    truth_corr_path = os.path.join(PLOTS_DIR, "4.2_truth_history_correlation.png")
    plt.savefig(truth_corr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Truth History Correlation]({os.path.relpath(truth_corr_path, SCRIPT_DIR)})\n")
    
    # Create a grouped bar chart of average truth history by binary label
    truth_by_label = combined_df.groupby('label')[truth_history_cols].mean().reset_index()
    truth_by_label_melted = pd.melt(truth_by_label, id_vars='label', value_vars=truth_history_cols)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=truth_by_label_melted, x='variable', y='value', hue='label', palette=['green', 'red'])
    plt.title('Average Truth History by Label', fontsize=16)
    plt.xlabel('Truth History Feature', fontsize=12)
    plt.ylabel('Average Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Label')
    plt.tight_layout()
    
    # Save figure
    truth_by_label_path = os.path.join(PLOTS_DIR, "4.2_truth_history_by_label.png")
    plt.savefig(truth_by_label_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Truth History by Label]({os.path.relpath(truth_by_label_path, SCRIPT_DIR)})\n")
    
    # 4. Key Observations and Findings
    section_content.append("## 4. Key Observations and Findings")
    
    # Get the actual label values
    label_values = list(combined_df['label'].unique())
    true_label = label_values[0]
    false_label = label_values[1] if len(label_values) > 1 else None
    
    # Statement length findings - using actual labels instead of hardcoded 'TRUE'/'FALSE'
    true_avg_length = length_by_label.loc[true_label, 'statement_length']['mean']
    false_avg_length = length_by_label.loc[false_label, 'statement_length']['mean'] if false_label else 0
    length_diff = abs(true_avg_length - false_avg_length).round(1)
    
    if true_avg_length > false_avg_length:
        section_content.append(f"- {true_label} statements are on average {length_diff} characters longer than {false_label} statements")
    else:
        section_content.append(f"- {false_label} statements are on average {length_diff} characters longer than {true_label} statements")
    
    true_avg_words = length_by_label.loc[true_label, 'word_count']['mean']
    false_avg_words = length_by_label.loc[false_label, 'word_count']['mean'] if false_label else 0
    word_diff = abs(true_avg_words - false_avg_words).round(1)
    
    if true_avg_words > false_avg_words:
        section_content.append(f"- {true_label} statements have on average {word_diff} more words than {false_label} statements")
    else:
        section_content.append(f"- {false_label} statements have on average {word_diff} more words than {true_label} statements")
    
    # Truth history findings
    # Calculate average total counts
    combined_df['total_truth_counts'] = combined_df[truth_history_cols].sum(axis=1)
    avg_counts_by_label = combined_df.groupby('label')['total_truth_counts'].mean().round(2)
    
    section_content.append(f"- Speakers making {true_label} statements have an average of {avg_counts_by_label[true_label]} previous fact-checked statements")
    if false_label:
        section_content.append(f"- Speakers making {false_label} statements have an average of {avg_counts_by_label[false_label]} previous fact-checked statements")
    
    # Add recommendations based on basic statistics
    section_content.append("\n### Recommendations Based on Basic Statistics")
    
    # Statement length recommendations
    if abs(true_avg_length - false_avg_length) > 20 or abs(true_avg_words - false_avg_words) > 5:
        section_content.append(f"- **Include statement length features**: The difference in statement length between {true_label} and {false_label} classes suggests that including statement length (characters) and word count as features may be beneficial for classification.")
    
    # Normalize truth history
    has_missing_truth = any(missing > 0 for missing in missing_counts)
    if has_missing_truth:
        section_content.append("- **Handle missing truth history**: Fill missing truth history values with appropriate methods (e.g., zero for new speakers, mean/median for others).")
    
    section_content.append("- **Create speaker credibility feature**: Calculate a credibility score based on the speaker's truth history (e.g., ratio of true statements to total statements).")
    
    # Save the report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(section_content))
    
    print(f"Basic statistics analysis report saved to: {REPORT_PATH}")
    
    return combined_df

if __name__ == "__main__":
    print("Starting analysis of binary classification dataset - Basic Statistics (4.2)...")
    
    # Load the data
    print("Loading data...")
    train_df, test_df, valid_df, combined_df = load_data()
    
    # Analyze basic statistics
    print("Analyzing basic statistics...")
    analyze_basic_statistics(train_df, test_df, valid_df, combined_df)
    
    print("Analysis complete!") 