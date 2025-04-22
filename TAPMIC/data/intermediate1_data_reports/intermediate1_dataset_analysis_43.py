#!/usr/bin/env python3
# intermediate1_dataset_analysis_43.py
# Analysis of binary classification dataset - Missing Value Analysis (4.3)

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
REPORT_PATH = os.path.join(SCRIPT_DIR, "intermediate1_dataset_analysis_43.txt")

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

def analyze_missing_values(train_df, test_df, valid_df, combined_df):
    """
    4.3 Analyze missing values in the binary classification dataset
    """
    section_content = []
    section_content.append(f"# 4.3 Missing Value Analysis\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Calculate missing values for each dataset
    train_missing = train_df.isnull().sum()
    test_missing = test_df.isnull().sum()
    valid_missing = valid_df.isnull().sum()
    combined_missing = combined_df.isnull().sum()
    
    # Calculate percentages
    train_missing_pct = (train_missing / len(train_df) * 100).round(2)
    test_missing_pct = (test_missing / len(test_df) * 100).round(2)
    valid_missing_pct = (valid_missing / len(valid_df) * 100).round(2)
    combined_missing_pct = (combined_missing / len(combined_df) * 100).round(2)
    
    # Create a summary dataframe for missing values
    missing_df = pd.DataFrame({
        'Train Missing': train_missing,
        'Train Missing %': train_missing_pct,
        'Test Missing': test_missing,
        'Test Missing %': test_missing_pct,
        'Valid Missing': valid_missing,
        'Valid Missing %': valid_missing_pct,
        'Combined Missing': combined_missing,
        'Combined Missing %': combined_missing_pct
    })
    
    # Sort by the highest percentage of missing values in the combined dataset
    missing_df = missing_df.sort_values('Combined Missing %', ascending=False)
    
    # Keep only columns with at least one missing value
    missing_df = missing_df[missing_df['Combined Missing'] > 0]
    
    # 1. Overall Missing Value Summary
    section_content.append("## 1. Overall Missing Value Summary\n")
    
    if missing_df.empty:
        section_content.append("**No missing values detected in any columns across all datasets.**\n")
    else:
        section_content.append("### Missing Value Counts and Percentages\n")
        # Format the dataframe content as a table for the report
        missing_table = missing_df.to_string()
        section_content.append(f"```\n{missing_table}\n```\n")
        
        section_content.append("### Columns with Missing Values\n")
        for column in missing_df.index:
            section_content.append(f"- **{column}**: {missing_df.loc[column, 'Combined Missing']} values ({missing_df.loc[column, 'Combined Missing %']}%) missing in the combined dataset")
    
    # 2. Missing Values by Label
    section_content.append("\n## 2. Missing Values by Label\n")
    
    # Create a visualization of missing values by label
    if not missing_df.empty:
        # Get columns with missing values
        cols_with_missing = missing_df.index.tolist()
        
        # Calculate missing values by label for those columns
        true_missing = combined_df[combined_df['label'] == True][cols_with_missing].isnull().sum()
        false_missing = combined_df[combined_df['label'] == False][cols_with_missing].isnull().sum()
        
        # Calculate percentages
        true_count = (combined_df['label'] == True).sum()
        false_count = (combined_df['label'] == False).sum()
        
        true_missing_pct = (true_missing / true_count * 100).round(2)
        false_missing_pct = (false_missing / false_count * 100).round(2)
        
        # Create dataframe for label-wise missing values
        label_missing_df = pd.DataFrame({
            'True Missing': true_missing,
            'True Missing %': true_missing_pct,
            'False Missing': false_missing,
            'False Missing %': false_missing_pct
        })
        
        # Format the dataframe content as a table for the report
        label_missing_table = label_missing_df.to_string()
        section_content.append(f"```\n{label_missing_table}\n```\n")
        
        # Create bar chart for missing value percentages by label
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        label_missing_plot = pd.DataFrame({
            'True': true_missing_pct,
            'False': false_missing_pct
        })
        
        # Plot
        ax = label_missing_plot.plot(kind='bar', figsize=(14, 8), color=['green', 'red'])
        plt.title('Missing Value Percentages by Label', fontsize=16)
        plt.xlabel('Column', fontsize=12)
        plt.ylabel('Missing Value Percentage', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Label')
        plt.tight_layout()
        
        # Save figure
        label_missing_path = os.path.join(PLOTS_DIR, "4.3_missing_values_by_label.png")
        plt.savefig(label_missing_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content.append(f"\n![Missing Values by Label]({os.path.relpath(label_missing_path, SCRIPT_DIR)})\n")
        
        # Add analysis of any significant differences between missing values by label
        section_content.append("### Analysis of Missing Values by Label\n")
        
        for column in label_missing_df.index:
            true_pct = label_missing_df.loc[column, 'True Missing %']
            false_pct = label_missing_df.loc[column, 'False Missing %']
            diff = abs(true_pct - false_pct)
            
            if diff > 2:  # If there's more than 2% difference
                if true_pct > false_pct:
                    section_content.append(f"- **{column}**: Missing values are {diff:.2f}% more common in TRUE statements")
                else:
                    section_content.append(f"- **{column}**: Missing values are {diff:.2f}% more common in FALSE statements")
    else:
        section_content.append("No missing values to analyze by label.")
    
    # 3. Visualize Missing Value Patterns
    section_content.append("\n## 3. Missing Value Patterns\n")
    
    if not missing_df.empty:
        # Create visualization for missing value patterns
        plt.figure(figsize=(12, 8))
        # Use missingno library-like visualization with seaborn heatmap
        cols_with_missing = missing_df.index.tolist()
        missing_pattern = combined_df[cols_with_missing].isnull()
        
        # Downsample if dataset is too large
        if len(missing_pattern) > 1000:
            missing_pattern = missing_pattern.sample(1000, random_state=42)
        
        # Create correlation matrix for missing values
        corr_matrix = missing_pattern.corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
        plt.title('Correlation of Missing Values Between Columns', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        missing_corr_path = os.path.join(PLOTS_DIR, "4.3_missing_value_correlation.png")
        plt.savefig(missing_corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content.append(f"![Missing Value Correlation]({os.path.relpath(missing_corr_path, SCRIPT_DIR)})\n")
        
        # Create bar chart for overall missing values 
        plt.figure(figsize=(14, 8))
        ax = missing_df['Combined Missing %'].plot(kind='bar', color='skyblue')
        plt.title('Percentage of Missing Values by Column', fontsize=16)
        plt.xlabel('Column', fontsize=12)
        plt.ylabel('Missing Value Percentage', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for i, v in enumerate(missing_df['Combined Missing %']):
            ax.text(i, v + 0.5, f"{v}%", ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        missing_bar_path = os.path.join(PLOTS_DIR, "4.3_missing_value_percentage.png")
        plt.savefig(missing_bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content.append(f"![Missing Value Percentage]({os.path.relpath(missing_bar_path, SCRIPT_DIR)})\n")
    else:
        section_content.append("No missing value patterns to visualize as the dataset is complete.")
    
    # 4. Recommendations for Missing Value Handling
    section_content.append("\n## 4. Recommendations for Missing Value Handling\n")
    
    if not missing_df.empty:
        # Add specific recommendations for each column with missing values
        section_content.append("### Column-Specific Recommendations\n")
        
        for column in missing_df.index:
            missing_pct = missing_df.loc[column, 'Combined Missing %']
            
            if 'counts' in column.lower():
                section_content.append(f"- **{column}**: Fill with 0 (assuming missing means no previous statements)")
            elif column in ['subject', 'context']:
                section_content.append(f"- **{column}**: Fill with a special placeholder like 'UNKNOWN' or 'OTHER'")
                section_content.append(f"  - Alternative: Create a binary feature '{column}_missing' to flag missing values")
            elif column in ['speaker_job', 'state', 'party']:
                section_content.append(f"- **{column}**: Fill with the most frequent value or a special 'UNKNOWN' category")
            else:
                section_content.append(f"- **{column}**: Consider imputing with mean/median for numeric or mode for categorical variables")
        
        # Add general recommendations
        section_content.append("\n### General Recommendations\n")
        
        if missing_df['Combined Missing %'].max() > 20:
            section_content.append("- **High Missing Value Columns**: Consider dropping columns with extremely high missing value rates (>80%)")
        
        section_content.append("- **Preprocessing Pipeline**: Create a consistent missing value handling strategy in the preprocessing pipeline")
        section_content.append("- **Missing Value Flags**: Create binary flags for columns where missingness might be informative")
        section_content.append("- **Document Strategy**: Document the missing value handling strategy for each field to ensure consistency")
    else:
        section_content.append("The dataset is complete with no missing values. No handling strategies are necessary.")
    
    # Save the report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(section_content))
    
    print(f"Missing value analysis report saved to: {REPORT_PATH}")
    
    return combined_df

if __name__ == "__main__":
    print("Starting analysis of binary classification dataset - Missing Value Analysis (4.3)...")
    
    # Load the data
    print("Loading data...")
    train_df, test_df, valid_df, combined_df = load_data()
    
    # Analyze missing values
    print("Analyzing missing values...")
    analyze_missing_values(train_df, test_df, valid_df, combined_df)
    
    print("Analysis complete!") 