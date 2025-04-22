#!/usr/bin/env python3
# intermediate1_dataset_analysis.py
# Comprehensive analysis script for binary classification dataset (Task 4.7)
# Implements high and medium priority preprocessing from task 4.6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os
import re
import string
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

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
REPORT_PATH = os.path.join(SCRIPT_DIR, "intermediate1_dataset_analysis_report.txt")

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

def analyze_class_distribution(train_df, test_df, valid_df, combined_df, report_sections):
    """
    4.1 Analyze class distribution in binary classification dataset
    """
    section_content = []
    section_content.append(f"# 4.1 Binary Class Distribution Analysis\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define datasets and their names
    datasets = {
        "Train": train_df,
        "Test": test_df,
        "Validation": valid_df,
        "Overall": combined_df
    }
    
    section_content.append("## Binary Class Distribution\n")
    
    # Analyze class distribution for each dataset
    class_counts = {}
    for name, df in datasets.items():
        # Count samples by label
        label_counts = df['label'].value_counts()
        
        # Calculate percentages
        total = label_counts.sum()
        percentages = (label_counts / total * 100).round(2)
        
        # Store counts for later
        class_counts[name] = {
            'True': label_counts.get(1, 0),
            'False': label_counts.get(0, 0),
            'Total': total
        }
        
        # Add to report
        section_content.append(f"### {name} Set")
        section_content.append(f"- True: {label_counts.get(1, 0)} samples ({percentages.get(1, 0)}%)")
        section_content.append(f"- False: {label_counts.get(0, 0)} samples ({percentages.get(0, 0)}%)")
        section_content.append(f"- Total: {total} samples\n")
    
    # Calculate class imbalance
    majority_class = "True" if class_counts["Overall"]["True"] > class_counts["Overall"]["False"] else "False"
    minority_class = "False" if majority_class == "True" else "True"
    imbalance_ratio = class_counts["Overall"][majority_class] / class_counts["Overall"][minority_class]
    
    section_content.append("### Class Imbalance Analysis")
    section_content.append(f"- Majority class: {majority_class} with {class_counts['Overall'][majority_class]} samples")
    section_content.append(f"- Minority class: {minority_class} with {class_counts['Overall'][minority_class]} samples")
    section_content.append(f"- Imbalance ratio (majority:minority): {imbalance_ratio:.2f}")
    
    if imbalance_ratio < 1.5:
        section_content.append(f"- **Note**: Class distribution is relatively balanced (imbalance ratio < 1.5)\n")
    else:
        section_content.append(f"- **Note**: Class distribution shows imbalance (imbalance ratio >= 1.5)\n")
    
    # Create visualizations
    # Bar chart of class distribution
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    dataset_names = list(datasets.keys())
    true_counts = [class_counts[name]["True"] for name in dataset_names]
    false_counts = [class_counts[name]["False"] for name in dataset_names]
    
    # Set the positions of the bars on the x-axis
    x = np.arange(len(dataset_names))
    width = 0.35
    
    # Create the bars
    plt.bar(x - width/2, true_counts, width, label='TRUE', color='green', alpha=0.7)
    plt.bar(x + width/2, false_counts, width, label='FALSE', color='red', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Binary Class Distribution Across Datasets', fontsize=14)
    plt.xticks(x, dataset_names)
    plt.legend()
    
    # Add counts on top of bars
    for i, v in enumerate(true_counts):
        plt.text(i - width/2, v + 50, str(v), ha='center')
    
    for i, v in enumerate(false_counts):
        plt.text(i + width/2, v + 50, str(v), ha='center')
    
    plt.tight_layout()
    
    # Save bar chart
    bar_chart_path = os.path.join(PLOTS_DIR, "4.1_binary_class_distribution.png")
    plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Binary Class Distribution]({os.path.relpath(bar_chart_path, SCRIPT_DIR)})\n")
    
    # Create pie charts for class distribution
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for i, (name, counts) in enumerate(class_counts.items()):
        sizes = [counts["True"], counts["False"]]
        labels = ['TRUE', 'FALSE']
        colors = ['green', 'red']
        axs[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        axs[i].set_title(f"{name} Set: {counts['Total']} samples", fontsize=12)
    
    plt.tight_layout()
    
    # Save pie charts
    pie_charts_path = os.path.join(PLOTS_DIR, "4.1_binary_class_distribution_pie.png")
    plt.savefig(pie_charts_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Binary Class Distribution - Pie Charts]({os.path.relpath(pie_charts_path, SCRIPT_DIR)})\n")
    
    # Add to report sections
    report_sections["4.1_class_distribution"] = "\n".join(section_content)
    
    return class_counts

def analyze_missing_values(train_df, test_df, valid_df, combined_df, report_sections):
    """
    4.3 Analyze missing values in the binary classification dataset and implement handling strategies
    """
    section_content = []
    section_content.append(f"# 4.3 Missing Value Analysis and Handling\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Define datasets and their names
    datasets = {
        "Train": train_df,
        "Test": test_df,
        "Valid": valid_df,
        "Combined": combined_df
    }
    
    # 1. Overall Missing Value Summary
    section_content.append("## 1. Overall Missing Value Summary\n")
    
    # Calculate missing values for each dataset
    missing_data = pd.DataFrame()
    
    for name, df in datasets.items():
        # Count missing values
        missing_count = df.isnull().sum()
        
        # Calculate percentages
        missing_percent = (missing_count / len(df) * 100).round(2)
        
        # Add to dataframe
        missing_data[f"{name} Missing"] = missing_count
        missing_data[f"{name} Missing %"] = missing_percent
    
    # Filter to only show columns with missing values
    missing_data = missing_data.loc[missing_data["Combined Missing"] > 0]
    
    # Sort by combined missing percentage (descending)
    missing_data = missing_data.sort_values("Combined Missing %", ascending=False)
    
    # Add to report
    section_content.append("### Missing Value Counts and Percentages\n")
    section_content.append("```")
    section_content.append(missing_data.to_string())
    section_content.append("```\n")
    
    # List columns with missing values
    section_content.append("### Columns with Missing Values\n")
    
    for index, row in missing_data.iterrows():
        section_content.append(f"- **{index}**: {row['Combined Missing']} values ({row['Combined Missing %']}%) missing in the combined dataset")
    
    # 2. Missing Values by Label
    section_content.append("\n## 2. Missing Values by Label\n")
    
    # Split combined dataset by label
    true_df = combined_df[combined_df['label'] == 1]
    false_df = combined_df[combined_df['label'] == 0]
    
    # Calculate missing values for each label
    missing_by_label = pd.DataFrame()
    
    # For TRUE label
    missing_count_true = true_df.isnull().sum()
    missing_percent_true = (missing_count_true / len(true_df) * 100).round(2)
    missing_by_label["True Missing"] = missing_count_true
    missing_by_label["True Missing %"] = missing_percent_true
    
    # For FALSE label
    missing_count_false = false_df.isnull().sum()
    missing_percent_false = (missing_count_false / len(false_df) * 100).round(2)
    missing_by_label["False Missing"] = missing_count_false
    missing_by_label["False Missing %"] = missing_percent_false
    
    # Filter to only show columns with missing values
    missing_by_label = missing_by_label.loc[(missing_by_label["True Missing"] > 0) | (missing_by_label["False Missing"] > 0)]
    
    # Sort by true missing percentage (descending)
    missing_by_label = missing_by_label.sort_values("True Missing %", ascending=False)
    
    # Add to report
    section_content.append("```")
    section_content.append(missing_by_label.to_string())
    section_content.append("```\n")
    
    # Create bar chart for missing values by label
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    columns_with_missing = missing_by_label.index.tolist()
    true_missing_pct = missing_by_label["True Missing %"].values
    false_missing_pct = missing_by_label["False Missing %"].values
    
    # Set the positions of the bars on the x-axis
    x = np.arange(len(columns_with_missing))
    width = 0.35
    
    # Create the bars
    plt.barh(x - width/2, true_missing_pct, width, label='TRUE', color='green', alpha=0.7)
    plt.barh(x + width/2, false_missing_pct, width, label='FALSE', color='red', alpha=0.7)
    
    # Add labels and title
    plt.ylabel('Column', fontsize=12)
    plt.xlabel('Missing Values (%)', fontsize=12)
    plt.title('Missing Values by Label', fontsize=14)
    plt.yticks(x, columns_with_missing)
    plt.legend()
    
    plt.tight_layout()
    
    # Save bar chart
    missing_by_label_path = os.path.join(PLOTS_DIR, "4.3_missing_values_by_label.png")
    plt.savefig(missing_by_label_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Missing Values by Label]({os.path.relpath(missing_by_label_path, SCRIPT_DIR)})\n")
    
    # Analysis of missing values by label
    section_content.append("### Analysis of Missing Values by Label\n")
    
    for column in columns_with_missing:
        true_pct = missing_by_label.loc[column, "True Missing %"]
        false_pct = missing_by_label.loc[column, "False Missing %"]
        diff = abs(true_pct - false_pct)
        
        if diff > 1.0:  # Only highlight differences greater than 1%
            higher = "TRUE" if true_pct > false_pct else "FALSE"
            section_content.append(f"- **{column}**: Missing values are {diff:.2f}% more common in {higher} statements")
    
    # 3. Missing Value Patterns
    section_content.append("\n## 3. Missing Value Patterns\n")
    
    # Create correlation heatmap of missing values
    plt.figure(figsize=(12, 10))
    
    # Create binary indicators for missingness
    missing_indicators = combined_df.isnull().astype(int)
    
    # Only keep columns with missing values
    missing_indicators = missing_indicators[missing_indicators.columns[missing_indicators.sum() > 0]]
    
    # Calculate correlation matrix
    corr_matrix = missing_indicators.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        fmt='.2f',
        linewidths=0.5
    )
    
    plt.title('Correlation Between Missing Values', fontsize=16)
    plt.tight_layout()
    
    # Save heatmap
    corr_heatmap_path = os.path.join(PLOTS_DIR, "4.3_missing_value_correlation.png")
    plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Missing Value Correlation]({os.path.relpath(corr_heatmap_path, SCRIPT_DIR)})\n")
    
    # Create percentage missing bar chart
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    missing_pct = (combined_df.isnull().sum() / len(combined_df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]
    
    # Create bar chart
    sns.barplot(x=missing_pct.index, y=missing_pct.values, palette='viridis')
    
    plt.title('Percentage of Missing Values by Column', fontsize=14)
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Missing Values (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save bar chart
    missing_pct_path = os.path.join(PLOTS_DIR, "4.3_missing_value_percentage.png")
    plt.savefig(missing_pct_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Missing Value Percentage]({os.path.relpath(missing_pct_path, SCRIPT_DIR)})\n")
    
    # 4. Missing Value Handling Implementation
    section_content.append("\n## 4. Missing Value Handling Implementation\n")
    
    section_content.append("### Column-Specific Handling Strategies")
    
    # Define handling strategies based on missing value analysis
    handling_strategies = {
        'speaker_job': "Fill with special 'UNKNOWN' category + binary flag feature",
        'state': "Fill with special 'UNKNOWN' category + binary flag feature",
        'context': "Fill with special 'UNKNOWN' category",
        'subject': "Fill with most common value",
        'speaker': "Fill with special 'UNKNOWN' value",
        'party': "Fill with special 'UNKNOWN' value",
        'barely_true_counts': "Fill with 0 (assuming no previous statements)",
        'false_counts': "Fill with 0 (assuming no previous statements)",
        'half_true_counts': "Fill with 0 (assuming no previous statements)",
        'mostly_true_counts': "Fill with 0 (assuming no previous statements)",
        'pants_on_fire_counts': "Fill with 0 (assuming no previous statements)"
    }
    
    # Apply missing value handling strategies
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    valid_processed = valid_df.copy()
    
    # Process each dataset with the same strategy
    for dataset_name, dataset in [("Train", train_processed), ("Test", test_processed), ("Validation", valid_processed)]:
        section_content.append(f"\n#### {dataset_name} Dataset Processing")
        
        # Create binary flag features for high missing columns
        for column in ['speaker_job', 'state']:
            if column in dataset.columns:
                flag_column = f"{column}_missing"
                dataset[flag_column] = dataset[column].isnull().astype(int)
                section_content.append(f"- Created binary flag feature '{flag_column}' to indicate missing values")
        
        # Fill missing values according to strategies
        for column in dataset.columns:
            if column in handling_strategies and dataset[column].isnull().sum() > 0:
                missing_count = dataset[column].isnull().sum()
                
                if column in ['speaker_job', 'state', 'context', 'speaker', 'party']:
                    # Fill categorical columns with 'UNKNOWN'
                    dataset[column] = dataset[column].fillna('UNKNOWN')
                    section_content.append(f"- Filled {missing_count} missing values in '{column}' with 'UNKNOWN'")
                
                elif column == 'subject':
                    # Fill with most common value
                    most_common = dataset[column].mode()[0]
                    dataset[column] = dataset[column].fillna(most_common)
                    section_content.append(f"- Filled {missing_count} missing values in '{column}' with most common value: '{most_common}'")
                
                elif column in ['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']:
                    # Fill numeric truth history columns with 0
                    dataset[column] = dataset[column].fillna(0)
                    section_content.append(f"- Filled {missing_count} missing values in '{column}' with 0")
    
    section_content.append("\n### Summary of Missing Value Handling")
    section_content.append("- Created binary flags for columns with significant missing values (speaker_job, state)")
    section_content.append("- Filled categorical missing values with 'UNKNOWN' placeholder")
    section_content.append("- Filled truth history missing values with 0")
    section_content.append("- Filled subject missing values with the most common value")
    
    # Add to report sections
    report_sections["4.3_missing_values"] = "\n".join(section_content)
    
    return train_processed, test_processed, valid_processed

def create_speaker_credibility_features(train_df, test_df, valid_df, report_sections):
    """
    Create speaker credibility features based on historical truth data (high priority)
    """
    section_content = []
    section_content.append(f"# Speaker Credibility Feature Engineering\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Process each dataset
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    valid_processed = valid_df.copy()
    
    datasets = {
        "Train": train_processed,
        "Test": test_processed,
        "Validation": valid_processed
    }
    
    section_content.append("## Speaker Credibility Features\n")
    section_content.append("Creating speaker credibility features based on historical truth data:\n")
    
    # Define truth and lie columns
    truth_cols = ['mostly_true_counts', 'half_true_counts']
    lie_cols = ['barely_true_counts', 'false_counts', 'pants_on_fire_counts']
    
    for name, df in datasets.items():
        section_content.append(f"### {name} Dataset Processing")
        
        # Ensure all count columns are numeric and filled
        for col in truth_cols + lie_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Calculate aggregated counts
        df['total_statements'] = df[truth_cols + lie_cols].sum(axis=1)
        df['truth_statements'] = df[truth_cols].sum(axis=1)
        df['lie_statements'] = df[lie_cols].sum(axis=1)
        
        # Create credibility score (ratio of true statements to total statements)
        df['credibility_score'] = np.where(
            df['total_statements'] > 0,
            df['truth_statements'] / df['total_statements'],
            0.5  # Default to neutral credibility for speakers with no history
        )
        
        # Create weighted credibility score (giving different weights to different categories)
        df['weighted_mostly_true'] = df['mostly_true_counts'] * 1.0
        df['weighted_half_true'] = df['half_true_counts'] * 0.5
        df['weighted_barely_true'] = df['barely_true_counts'] * 0.25
        df['weighted_false'] = df['false_counts'] * 0.0
        df['weighted_pants_fire'] = df['pants_on_fire_counts'] * -0.5
        
        weighted_sum = (df['weighted_mostly_true'] + df['weighted_half_true'] + 
                        df['weighted_barely_true'] + df['weighted_false'] + 
                        df['weighted_pants_fire'])
        
        df['weighted_credibility'] = np.where(
            df['total_statements'] > 0,
            (weighted_sum + df['total_statements']) / (2 * df['total_statements']),  # Normalize to 0-1 range
            0.5  # Default to neutral credibility for speakers with no history
        )
        
        # Calculate standard deviation of speaker truth history
        df['truth_history_std'] = df[truth_cols + lie_cols].std(axis=1)
        
        # Record features created
        section_content.append("Created features:")
        section_content.append("- `total_statements`: Total historical statements by speaker")
        section_content.append("- `truth_statements`: Total TRUE, MOSTLY-TRUE, and HALF-TRUE statements")
        section_content.append("- `lie_statements`: Total FALSE, BARELY-TRUE, and PANTS-ON-FIRE statements")
        section_content.append("- `credibility_score`: Ratio of truth statements to total statements")
        section_content.append("- `weighted_credibility`: Weighted credibility score with different weights per category")
        section_content.append("- `truth_history_std`: Standard deviation of truth history, indicating consistency\n")
    
    # Analyze the correlation of new features with the label
    combined_processed = pd.concat([train_processed, test_processed, valid_processed])
    
    # Calculate correlations of new features with the label
    credibility_features = ['total_statements', 'truth_statements', 'lie_statements', 
                           'credibility_score', 'weighted_credibility', 'truth_history_std']
    
    corr_with_label = combined_processed[credibility_features + ['label']].corr()['label'].drop('label')
    
    section_content.append("## Correlation with Truth Label\n")
    for feature, corr in corr_with_label.sort_values(ascending=False).items():
        section_content.append(f"- **{feature}**: {corr:.4f}")
    
    # Create visualization of the correlation of speaker credibility with truth label
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    corr_sorted = corr_with_label.sort_values(ascending=False)
    bars = plt.bar(
        corr_sorted.index, 
        corr_sorted.values,
        color=[
            'green' if x > 0 else 'red' for x in corr_sorted.values
        ],
        alpha=0.7
    )
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title('Correlation of Speaker Credibility Features with Truth Label', fontsize=14)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add correlation values on top of bars
    for bar in bars:
        height = bar.get_height()
        if height < 0:
            va = 'top'
            y_pos = height - 0.02
        else:
            va = 'bottom'
            y_pos = height + 0.02
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            y_pos,
            f'{height:.3f}',
            ha='center',
            va=va,
            fontsize=9
        )
    
    plt.tight_layout()
    
    # Save correlation chart
    corr_chart_path = os.path.join(PLOTS_DIR, "speaker_credibility_correlation.png")
    plt.savefig(corr_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Speaker Credibility Correlation]({os.path.relpath(corr_chart_path, SCRIPT_DIR)})\n")
    
    # Create a scatter plot of credibility_score vs label
    plt.figure(figsize=(10, 6))
    
    # Add jitter to binary label for better visualization
    jittered_labels = combined_processed['label'] + np.random.normal(0, 0.05, size=len(combined_processed))
    
    plt.scatter(
        combined_processed['credibility_score'],
        jittered_labels,
        alpha=0.3,
        c=combined_processed['label'].map({0: 'red', 1: 'green'}),
        s=30
    )
    
    plt.xlabel('Speaker Credibility Score', fontsize=12)
    plt.ylabel('Statement Truth (jittered)', fontsize=12)
    plt.title('Relationship Between Speaker Credibility and Statement Truth', fontsize=14)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Save scatter plot
    scatter_path = os.path.join(PLOTS_DIR, "credibility_vs_truth_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Credibility vs Truth]({os.path.relpath(scatter_path, SCRIPT_DIR)})\n")
    
    # Summary and conclusions
    section_content.append("## Summary of Speaker Credibility Features\n")
    section_content.append("The speaker credibility features show strong correlation with the truth label:")
    section_content.append(f"- `credibility_score` has {corr_with_label['credibility_score']:.4f} correlation with truth")
    section_content.append(f"- `weighted_credibility` has {corr_with_label['weighted_credibility']:.4f} correlation with truth")
    section_content.append("These features capture the historical truthfulness of speakers and are strong predictors of statement veracity.")
    
    # Add to report sections
    report_sections["speaker_credibility_features"] = "\n".join(section_content)
    
    return train_processed, test_processed, valid_processed

def process_text_features(train_df, test_df, valid_df, report_sections):
    """
    Implement basic text preprocessing and feature extraction (high priority)
    """
    section_content = []
    section_content.append(f"# Text Preprocessing and Feature Extraction\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Process each dataset
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    valid_processed = valid_df.copy()
    
    datasets = {
        "Train": train_processed,
        "Test": test_processed,
        "Validation": valid_processed
    }
    
    section_content.append("## Text Feature Engineering\n")
    
    # Ensure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    for name, df in datasets.items():
        section_content.append(f"### {name} Dataset Text Processing")
        
        # Create basic text features
        # 1. Statement length
        df['statement_length'] = df['statement'].str.len()
        
        # 2. Word count
        df['word_count'] = df['statement'].str.split().str.len()
        
        # 3. Average word length
        df['avg_word_length'] = df['statement'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # 4. Preprocessing for further analysis
        def preprocess_text(text):
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation
            text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        df['processed_text'] = df['statement'].apply(preprocess_text)
        
        # 5. Count specific terms
        common_terms = ['percent', 'says', 'billion', 'million', 'tax', 'government', 'obama', 'president']
        for term in common_terms:
            df[f'contains_{term}'] = df['processed_text'].str.contains(r'\b' + term + r'\b').astype(int)
        
        # 6. Create features for specific patterns
        df['contains_number'] = df['processed_text'].str.contains(r'\d').astype(int)
        df['question_mark_count'] = df['statement'].str.count(r'\?')
        df['exclamation_mark_count'] = df['statement'].str.count(r'\!')
        
        # Log the features created
        section_content.append("Created basic text features:")
        section_content.append("- `statement_length`: Character count of statement")
        section_content.append("- `word_count`: Number of words in statement")
        section_content.append("- `avg_word_length`: Average length of words in statement")
        section_content.append("- `processed_text`: Lowercase, punctuation-free version of statement")
        section_content.append("- Term presence indicators for common terms")
        section_content.append("- `contains_number`: Whether statement contains numeric values")
        section_content.append("- `question_mark_count`: Number of question marks in statement")
        section_content.append("- `exclamation_mark_count`: Number of exclamation marks in statement\n")
    
    # Analyze the correlation of new features with the label
    combined_processed = pd.concat([train_processed, test_processed, valid_processed])
    
    # Get all new text feature columns
    text_feature_cols = [col for col in combined_processed.columns 
                         if col not in train_df.columns 
                         and col != 'processed_text']
    
    # Calculate correlations with label
    text_feature_corr = combined_processed[text_feature_cols + ['label']].corr()['label'].drop('label')
    
    section_content.append("## Correlation of Text Features with Truth Label\n")
    for feature, corr in text_feature_corr.sort_values(ascending=False).items():
        section_content.append(f"- **{feature}**: {corr:.4f}")
    
    # Create visualization of correlations
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    corr_sorted = text_feature_corr.sort_values(ascending=False)
    bars = plt.bar(
        corr_sorted.index, 
        corr_sorted.values,
        color=['green' if x > 0 else 'red' for x in corr_sorted.values],
        alpha=0.7
    )
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title('Correlation of Text Features with Truth Label', fontsize=14)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xticks(rotation=90)
    
    plt.tight_layout()
    
    # Save correlation chart
    text_corr_path = os.path.join(PLOTS_DIR, "text_features_correlation.png")
    plt.savefig(text_corr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Text Features Correlation]({os.path.relpath(text_corr_path, SCRIPT_DIR)})\n")
    
    # Summary and conclusions
    section_content.append("## Summary of Text Feature Engineering\n")
    
    # Find top correlated features
    top_features = text_feature_corr.abs().sort_values(ascending=False).head(5).index.tolist()
    
    section_content.append("Top text features by correlation magnitude:")
    for feature in top_features:
        section_content.append(f"- `{feature}`: {text_feature_corr[feature]:.4f}")
    
    section_content.append("\nThese text features capture important characteristics of statements that are associated with truthfulness.")
    
    # Add to report sections
    report_sections["text_feature_engineering"] = "\n".join(section_content)
    
    return train_processed, test_processed, valid_processed

def encode_categorical_features(train_df, test_df, valid_df, report_sections):
    """
    Implement appropriate encoding for categorical features (high priority)
    """
    section_content = []
    section_content.append(f"# Categorical Feature Encoding\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Process each dataset
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    valid_processed = valid_df.copy()
    
    combined_df = pd.concat([train_processed, test_processed, valid_processed])
    
    section_content.append("## Categorical Feature Encoding\n")
    
    # Define categorical columns (high and low cardinality)
    high_cardinality_cols = ['subject', 'speaker', 'speaker_job', 'context']
    low_cardinality_cols = ['state', 'party']
    
    # Get cardinality of each column
    cardinality = {}
    for col in high_cardinality_cols + low_cardinality_cols:
        if col in combined_df.columns:
            cardinality[col] = combined_df[col].nunique()
    
    section_content.append("### Categorical Field Cardinality\n")
    for col, unique_count in sorted(cardinality.items(), key=lambda x: x[1], reverse=True):
        section_content.append(f"- **{col}**: {unique_count} unique values")
    
    # Apply target encoding to high cardinality columns
    # We need to be careful to prevent data leakage, so we'll use the train set for fitting
    
    section_content.append("\n### High Cardinality Encoding (Target Encoding)\n")
    
    for col in high_cardinality_cols:
        if col in train_processed.columns:
            # Calculate target encoding mapping from train set
            target_encoding = train_processed.groupby(col)['label'].mean().to_dict()
            
            # Apply encoding to all datasets
            for name, df in [("Train", train_processed), ("Test", test_processed), ("Validation", valid_processed)]:
                # Create encoded column
                encoding_col = f"{col}_target_encoded"
                df[encoding_col] = df[col].map(target_encoding)
                
                # Fill missing values with global mean (for values not seen in training)
                global_mean = train_processed['label'].mean()
                df[encoding_col] = df[encoding_col].fillna(global_mean)
            
            section_content.append(f"- Applied target encoding to '{col}' -> '{col}_target_encoded'")
    
    section_content.append("\n### Low Cardinality Encoding (One-Hot Encoding)\n")
    
    for col in low_cardinality_cols:
        if col in train_processed.columns and cardinality[col] < 30:  # Only one-hot encode if less than 30 categories
            # Get all unique values from combined dataset
            all_categories = combined_df[col].dropna().unique()
            
            # Apply one-hot encoding to all datasets
            for name, df in [("Train", train_processed), ("Test", test_processed), ("Validation", valid_processed)]:
                for category in all_categories:
                    # Create binary column for each category
                    binary_col = f"{col}_{category}"
                    df[binary_col] = (df[col] == category).astype(int)
            
            section_content.append(f"- Applied one-hot encoding to '{col}' -> {len(all_categories)} binary features")
    
    # Create special binary features for strongly predictive categories
    section_content.append("\n### Special Predictive Category Features\n")
    
    # Define strongly predictive categories based on previous analysis
    predictive_categories = {
        'speaker': {
            'high_credibility': ['kasim-reed', 'dennis-kucinich', 'bill-nelson', 'rob-portman', 'cory-booker'],
            'low_credibility': ['chain-email', 'viral-image', 'blog-posting', 'democratic-congressional-campaign-committee', 'ben-carson']
        },
        'context': {
            'formal_context': ['the State of the Union address', 'comments on CNN\'s "State of the Union"', 'a newspaper column', 'a letter', 'an interview on MSNBC'],
            'informal_context': ['a chain email', 'a chain e-mail', 'a blog post', 'an email', 'a campaign mailer']
        }
    }
    
    # Create binary features for each group
    for field, category_groups in predictive_categories.items():
        for group_name, categories in category_groups.items():
            # Create binary feature for each dataset
            for name, df in [("Train", train_processed), ("Test", test_processed), ("Validation", valid_processed)]:
                if field in df.columns:
                    binary_col = f"{field}_{group_name}"
                    df[binary_col] = df[field].isin(categories).astype(int)
            
            section_content.append(f"- Created binary feature '{field}_{group_name}' for categories: {', '.join(categories)}")
    
    # Analyze the correlation of encoded features with the target
    section_content.append("\n## Correlation of Encoded Features with Truth Label\n")
    
    # Get all encoded columns
    encoded_cols = []
    for col in train_processed.columns:
        if col.endswith('_target_encoded') or col.endswith('_credibility') or \
           col.startswith('state_') or col.startswith('party_') or \
           '_credibility' in col or '_high_credibility' in col or '_low_credibility' in col or \
           '_formal_context' in col or '_informal_context' in col:
            encoded_cols.append(col)
    
    # Combine datasets for correlation analysis
    combined_processed = pd.concat([train_processed, test_processed, valid_processed])
    
    # Calculate correlations
    if encoded_cols:
        encoded_corr = combined_processed[encoded_cols + ['label']].corr()['label'].drop('label')
        
        # Display top correlations
        section_content.append("### Top Positive Correlations (indicative of TRUE statements)")
        for feature, corr in encoded_corr.sort_values(ascending=False).head(10).items():
            section_content.append(f"- **{feature}**: {corr:.4f}")
        
        section_content.append("\n### Top Negative Correlations (indicative of FALSE statements)")
        for feature, corr in encoded_corr.sort_values().head(10).items():
            section_content.append(f"- **{feature}**: {corr:.4f}")
        
        # Create correlation visualization
        plt.figure(figsize=(14, 10))
        
        # Create correlation heatmap for encoded features
        corr_matrix = combined_processed[encoded_cols].corr()
        
        # Use a mask to show only the lower triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='coolwarm',
            annot=False,
            fmt='.2f',
            linewidths=0.5,
            vmin=-1,
            vmax=1
        )
        
        plt.title('Correlation Between Encoded Categorical Features', fontsize=16)
        plt.tight_layout()
        
        # Save heatmap
        encoded_corr_path = os.path.join(PLOTS_DIR, "encoded_features_correlation.png")
        plt.savefig(encoded_corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content.append(f"\n![Encoded Features Correlation]({os.path.relpath(encoded_corr_path, SCRIPT_DIR)})\n")
    
    # Summary and conclusions
    section_content.append("## Summary of Categorical Encoding\n")
    section_content.append("- **Target Encoding**: Applied to high-cardinality fields (subject, speaker, speaker_job, context)")
    section_content.append("- **One-Hot Encoding**: Applied to low-cardinality fields (state, party)")
    section_content.append("- **Special Features**: Created binary features for predictive category groups")
    
    section_content.append("\nThese encoding strategies reduce dimensionality while preserving the predictive power of categorical fields.")
    
    # Add to report sections
    report_sections["categorical_encoding"] = "\n".join(section_content)
    
    return train_processed, test_processed, valid_processed

# Main function with consolidated processing steps
if __name__ == "__main__":
    print("Starting comprehensive analysis of binary classification dataset...")
    
    # Load the data
    print("Loading data...")
    train_df, test_df, valid_df, combined_df = load_data()
    
    # Dictionary to store report sections
    report_sections = {}
    
    # 4.1 Analyze class distribution
    print("Analyzing class distribution...")
    class_counts = analyze_class_distribution(train_df, test_df, valid_df, combined_df, report_sections)
    
    # 4.3 Analyze and handle missing values (high priority)
    print("Analyzing and handling missing values...")
    train_processed, test_processed, valid_processed = analyze_missing_values(train_df, test_df, valid_df, combined_df, report_sections)
    
    # Create speaker credibility features (high priority)
    print("Creating speaker credibility features...")
    train_processed, test_processed, valid_processed = create_speaker_credibility_features(
        train_processed, test_processed, valid_processed, report_sections
    )
    
    # Process text features (high priority)
    print("Processing text features...")
    train_processed, test_processed, valid_processed = process_text_features(
        train_processed, test_processed, valid_processed, report_sections
    )
    
    # Encode categorical features (high priority)
    print("Encoding categorical features...")
    train_processed, test_processed, valid_processed = encode_categorical_features(
        train_processed, test_processed, valid_processed, report_sections
    )
    
    # Save processed datasets
    print("Saving processed datasets...")
    train_processed.to_csv(os.path.join(DATA_DIR, "processed_train.csv"), index=False)
    test_processed.to_csv(os.path.join(DATA_DIR, "processed_test.csv"), index=False)
    valid_processed.to_csv(os.path.join(DATA_DIR, "processed_validation.csv"), index=False)
    
    # Write the comprehensive report
    print("Generating comprehensive analysis report...")
    
    with open(REPORT_PATH, "w") as f:
        f.write("# Comprehensive Binary Classification Dataset Analysis\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This report includes analysis and preprocessing of the binary classification dataset.\n\n")
        
        # Add each section to the report
        for section_name, section_content in report_sections.items():
            f.write(f"{section_content}\n\n")
        
        f.write("\n## Preprocessing Summary\n\n")
        f.write("The following preprocessing steps have been implemented:\n\n")
        f.write("1. **Missing Value Handling**:\n")
        f.write("   - Created binary flags for high-missing columns (speaker_job, state)\n")
        f.write("   - Filled categorical missing values with 'UNKNOWN'\n")
        f.write("   - Filled numeric truth history missing values with 0\n\n")
        
        f.write("2. **Speaker Credibility Features**:\n")
        f.write("   - Created credibility_score (truth statements / total statements)\n")
        f.write("   - Created weighted_credibility with category-specific weights\n")
        f.write("   - Generated aggregated truth and lie counts\n\n")
        
        f.write("3. **Text Feature Engineering**:\n")
        f.write("   - Created basic text features (length, word count, avg word length)\n")
        f.write("   - Created term presence indicators for common terms\n")
        f.write("   - Added features for syntactic patterns (question marks, numbers)\n\n")
        
        f.write("4. **Categorical Encoding**:\n")
        f.write("   - Applied target encoding to high-cardinality fields\n")
        f.write("   - Applied one-hot encoding to low-cardinality fields\n")
        f.write("   - Created special binary features for predictive category groups\n\n")
        
        f.write("The processed datasets have been saved to:\n")
        f.write(f"- Train: {os.path.join(DATA_DIR, 'processed_train.csv')}\n")
        f.write(f"- Test: {os.path.join(DATA_DIR, 'processed_test.csv')}\n")
        f.write(f"- Validation: {os.path.join(DATA_DIR, 'processed_validation.csv')}\n")
    
    print(f"Analysis complete! Report saved to {REPORT_PATH}") 