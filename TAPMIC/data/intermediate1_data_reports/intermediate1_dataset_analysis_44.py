#!/usr/bin/env python3
# intermediate1_dataset_analysis_44.py
# Analysis of binary classification dataset - Field Analysis (4.4)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from collections import Counter

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
REPORT_PATH = os.path.join(SCRIPT_DIR, "intermediate1_dataset_analysis_44.txt")

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

def analyze_fields(train_df, test_df, valid_df, combined_df):
    """
    4.4 Analyze individual fields in the binary classification dataset
    """
    section_content = []
    section_content.append(f"# 4.4 Field Analysis\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get categorical and numerical columns
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = combined_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Separate special columns for individual analysis
    special_categorical = ['statement']
    regular_categorical = [col for col in categorical_cols if col not in special_categorical and col != 'original_label']
    
    # Remove id from numerical for separate handling
    count_cols = [col for col in numerical_cols if 'counts' in col]
    other_numerical = [col for col in numerical_cols if col not in count_cols and col != 'id']
    
    # 1. Dataset Structure
    section_content.append("## 1. Dataset Structure\n")
    section_content.append(f"The binary classification dataset consists of {combined_df.shape[0]} samples and {combined_df.shape[1]} columns.\n")
    
    section_content.append("### Column Types")
    section_content.append(f"- **Categorical Columns ({len(categorical_cols)}):** {', '.join(categorical_cols)}")
    section_content.append(f"- **Numerical Columns ({len(numerical_cols)}):** {', '.join(numerical_cols)}\n")
    
    # 2. Categorical Field Analysis
    section_content.append("## 2. Categorical Field Analysis\n")
    
    # Loop through each categorical column (excluding statement which will be analyzed separately)
    for col in regular_categorical:
        section_content.append(f"### {col.replace('_', ' ').title()} Analysis")
        
        # Handle missing values in the analysis
        value_counts = combined_df[col].fillna('MISSING').value_counts()
        value_pcts = (value_counts / len(combined_df) * 100).round(2)
        
        # Number of unique values
        unique_count = value_counts.shape[0]
        section_content.append(f"- **Unique Values:** {unique_count}")
        
        # Show top N values with percentages
        top_n = min(10, unique_count)
        section_content.append(f"- **Top {top_n} Values:**")
        for val, count in value_counts.head(top_n).items():
            pct = value_pcts[val]
            section_content.append(f"  - {val}: {count} ({pct}%)")
        
        # Create distribution by binary label
        cross_tab = pd.crosstab(
            combined_df[col].fillna('MISSING'), 
            combined_df['label'],
            normalize='columns'
        ).round(4) * 100
        
        # Check if there are more than 25 unique values
        if unique_count > 25:
            # For many values, focus on top categories only for visualization
            top_vals = value_counts.head(15).index.tolist()
            filtered_df = combined_df[combined_df[col].isin(top_vals)]
            
            # Plot percentage distribution for top categories
            plt.figure(figsize=(12, 8))
            ax = sns.countplot(
                x=col, 
                hue='label', 
                data=filtered_df,
                palette=['red', 'green'],
                order=top_vals
            )
            plt.title(f'Top {len(top_vals)} {col.replace("_", " ").title()} Categories by Label', fontsize=16)
            plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Label', labels=['False', 'True'])
            
            # Add percentage annotations on bars
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        height + 5,
                        f'{int(height)}',
                        ha="center", fontsize=9
                    )
            
            plt.tight_layout()
            
            # Save plot
            cat_plot_path = os.path.join(PLOTS_DIR, f"4.4_{col}_top_categories.png")
            plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            section_content.append(f"\n![{col.replace('_', ' ').title()} Top Categories]({os.path.relpath(cat_plot_path, SCRIPT_DIR)})\n")
            
            # For columns with many values, check for predictive power
            section_content.append("\n#### Predictive Value Analysis")
            
            # Calculate ratio of True to False for each category
            true_false_ratio = {}
            for val in combined_df[col].fillna('MISSING').unique():
                true_count = ((combined_df[col] == val) & (combined_df['label'] == True)).sum()
                false_count = ((combined_df[col] == val) & (combined_df['label'] == False)).sum()
                
                # Avoid division by zero
                if false_count > 0:
                    ratio = true_count / false_count
                else:
                    ratio = float('inf') if true_count > 0 else 0
                
                # Only include if category has minimum samples
                if true_count + false_count >= 20:
                    true_false_ratio[val] = (ratio, true_count + false_count)
            
            # Sort by ratio
            sorted_ratios = sorted(true_false_ratio.items(), key=lambda x: x[1][0], reverse=True)
            
            # Show categories with high True:False ratio
            section_content.append("**Categories with High TRUE:FALSE Ratio:**")
            for val, (ratio, count) in sorted_ratios[:5]:
                if ratio > 1.5:  # Threshold for "high ratio"
                    section_content.append(f"- {val}: {ratio:.2f} ({count} samples)")
            
            # Show categories with low True:False ratio (more False labels)
            section_content.append("\n**Categories with High FALSE:TRUE Ratio:**")
            for val, (ratio, count) in sorted(true_false_ratio.items(), key=lambda x: x[1][0])[:5]:
                if ratio < 0.67:  # Threshold for "low ratio" (1/1.5)
                    section_content.append(f"- {val}: {(1/ratio):.2f} ({count} samples)")
        else:
            # For fewer unique values, create a proper stacked percentage plot
            plt.figure(figsize=(14, 8))
            
            # Plot distribution
            cross_tab_plot = cross_tab.copy()
            cross_tab_plot.columns = ['False', 'True']
            cross_tab_plot.plot(kind='bar', stacked=False, figsize=(14, 8), colormap='RdYlGn')
            plt.title(f'{col.replace("_", " ").title()} Distribution by Label (%)', fontsize=16)
            plt.xlabel(col.replace('_', ' ').title(), fontsize=12)
            plt.ylabel('Percentage', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Label')
            
            # Add percentage annotations
            for i, p in enumerate(plt.gca().patches):
                width, height = p.get_width(), p.get_height()
                x, y = p.get_xy() 
                plt.gca().annotate(f'{height:.1f}%', (x + width/2, y + height/2), ha='center', va='center')
            
            plt.tight_layout()
            
            # Save plot
            cat_plot_path = os.path.join(PLOTS_DIR, f"4.4_{col}_distribution.png")
            plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            section_content.append(f"\n![{col.replace('_', ' ').title()} Distribution]({os.path.relpath(cat_plot_path, SCRIPT_DIR)})\n")
        
        # Check association with truth value using chi-square test
        from scipy.stats import chi2_contingency
        
        # Create contingency table
        contingency = pd.crosstab(combined_df[col].fillna('MISSING'), combined_df['label'])
        
        # Apply chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        section_content.append("#### Statistical Significance")
        section_content.append(f"- **Chi-Square Test:** χ² = {chi2:.2f}, p-value = {p:.6f}")
        
        if p < 0.05:
            section_content.append(f"- **Result:** {col.replace('_', ' ').title()} has a statistically significant association with the truth label (p < 0.05)")
        else:
            section_content.append(f"- **Result:** No statistically significant association detected between {col.replace('_', ' ').title()} and truth label (p >= 0.05)")
        
        section_content.append("")  # Add space between categorical column analyses
    
    # 3. Special Analysis for Statement Field
    section_content.append("## 3. Statement Field Analysis\n")
    
    # Text length distribution by label
    combined_df['statement_length'] = combined_df['statement'].str.len()
    combined_df['word_count'] = combined_df['statement'].str.split().str.len()
    
    # Word count visualization
    plt.figure(figsize=(14, 6))
    sns.histplot(
        data=combined_df, 
        x='word_count',
        hue='label',
        bins=50,
        kde=True,
        palette=['red', 'green']
    )
    plt.title('Word Count Distribution by Label', fontsize=16)
    plt.xlabel('Number of Words', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xlim(0, combined_df['word_count'].quantile(0.99))  # Limit to 99% quantile to avoid outliers
    
    plt.tight_layout()
    wc_plot_path = os.path.join(PLOTS_DIR, "4.4_word_count_distribution.png")
    plt.savefig(wc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Word Count Distribution]({os.path.relpath(wc_plot_path, SCRIPT_DIR)})\n")
    
    # Vocabulary analysis
    # Get all words
    all_words = " ".join(combined_df['statement'].fillna("")).lower().split()
    word_counts = Counter(all_words)
    
    # Remove very common words (simple stopwords)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are'}
    filtered_word_counts = {word: count for word, count in word_counts.items() if word not in stopwords}
    
    # Top 20 words
    top_words = sorted(filtered_word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    section_content.append("### Common Words Analysis")
    section_content.append("**Top 20 Words in Statements:**")
    for word, count in top_words:
        section_content.append(f"- {word}: {count} occurrences")
    
    # Create word frequency plot
    plt.figure(figsize=(14, 8))
    x, y = zip(*[(word, count) for word, count in top_words])
    sns.barplot(x=list(y), y=list(x), palette='viridis')
    plt.title('Top 20 Words in Statements', fontsize=16)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Word', fontsize=12)
    
    plt.tight_layout()
    word_plot_path = os.path.join(PLOTS_DIR, "4.4_top_words.png")
    plt.savefig(word_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Top Words]({os.path.relpath(word_plot_path, SCRIPT_DIR)})\n")
    
    # 4. Numerical Field Analysis
    section_content.append("## 4. Numerical Field Analysis\n")
    
    # First analyze count-based columns
    section_content.append("### Speaker Truth History Features\n")
    
    # Descriptive statistics for count columns
    count_stats = combined_df[count_cols].describe().round(2)
    stats_table = count_stats.to_string()
    section_content.append("**Descriptive Statistics:**")
    section_content.append(f"```\n{stats_table}\n```\n")
    
    # Distribution of count columns by label
    # Create a boxplot for each count column by label
    plt.figure(figsize=(16, 10))
    
    # Melt the dataframe for easier plotting
    count_df_melted = pd.melt(
        combined_df[count_cols + ['label']],
        id_vars=['label'],
        var_name='Count Type',
        value_name='Count'
    )
    
    # Create boxplot
    ax = sns.boxplot(
        x='Count Type',
        y='Count',
        hue='label',
        data=count_df_melted,
        palette=['red', 'green']
    )
    
    plt.title('Distribution of Speaker Truth History by Label', fontsize=16)
    plt.xlabel('Count Type', fontsize=12)
    plt.ylabel('Count Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Label', labels=['False', 'True'])
    
    # Set y-axis to log scale to better visualize distribution
    ax.set_yscale('symlog')  # Use symlog to handle zeros
    
    plt.tight_layout()
    counts_box_path = os.path.join(PLOTS_DIR, "4.4_truth_history_boxplot.png")
    plt.savefig(counts_box_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Truth History Boxplot]({os.path.relpath(counts_box_path, SCRIPT_DIR)})\n")
    
    # Calculate a "credibility score" for speakers
    section_content.append("### Speaker Credibility Analysis\n")
    
    # Create a simple credibility score (true statements / all statements)
    truth_cols = ['mostly_true_counts', 'half_true_counts']
    lie_cols = ['barely_true_counts', 'false_counts', 'pants_on_fire_counts']
    
    # Handle NaN values by filling with 0
    for col in truth_cols + lie_cols:
        combined_df[col] = combined_df[col].fillna(0)
    
    # Calculate total statements and truth ratio
    combined_df['total_statements'] = combined_df[truth_cols + lie_cols].sum(axis=1)
    combined_df['truth_statements'] = combined_df[truth_cols].sum(axis=1)
    combined_df['lie_statements'] = combined_df[lie_cols].sum(axis=1)
    
    # Calculate credibility score (ratio of true statements to total statements)
    # Handle division by zero by assigning 0 to speakers with no previous statements
    combined_df['credibility_score'] = np.where(
        combined_df['total_statements'] > 0,
        combined_df['truth_statements'] / combined_df['total_statements'],
        0
    )
    
    # Analyze credibility distribution by label
    plt.figure(figsize=(14, 8))
    
    # Only include speakers with at least one statement for meaningful analysis
    credibility_df = combined_df[combined_df['total_statements'] > 0]
    
    ax = sns.violinplot(
        x='label',
        y='credibility_score',
        data=credibility_df,
        palette=['red', 'green'],
        inner='quart'
    )
    
    plt.title('Speaker Credibility Score Distribution by Statement Label', fontsize=16)
    plt.xlabel('Statement Label', fontsize=12)
    plt.ylabel('Credibility Score (True Statements / Total Statements)', fontsize=12)
    plt.xticks([0, 1], ['False', 'True'])
    
    # Add mean lines
    for i, label in enumerate([False, True]):
        mean_val = credibility_df[credibility_df['label'] == label]['credibility_score'].mean()
        plt.axhline(mean_val, ls='--', color='blue', alpha=0.6, xmin=i/2, xmax=(i+1)/2)
        plt.text(i, mean_val + 0.01, f'Mean: {mean_val:.3f}', ha='center')
    
    plt.tight_layout()
    credibility_path = os.path.join(PLOTS_DIR, "4.4_credibility_score_distribution.png")
    plt.savefig(credibility_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Credibility Score Distribution]({os.path.relpath(credibility_path, SCRIPT_DIR)})\n")
    
    # Correlation between credibility score and label
    point_biserial = credibility_df['credibility_score'].corr(credibility_df['label'].astype(int))
    section_content.append(f"**Correlation with Statement Truth:** {point_biserial:.4f}")
    
    if abs(point_biserial) > 0.1:
        section_content.append("- There is a meaningful correlation between speaker credibility and statement truth label")
        section_content.append(f"- {'Higher' if point_biserial > 0 else 'Lower'} credibility scores are associated with TRUE statements")
    else:
        section_content.append("- The correlation between speaker credibility and statement truth is weak")
    
    # 5. Key Findings and Feature Recommendations
    section_content.append("\n## 5. Key Findings and Feature Recommendations\n")
    
    section_content.append("### Key Findings from Field Analysis")
    
    # For categorical fields
    section_content.append("**Categorical Fields:**")
    for col in regular_categorical:
        # Create contingency table
        contingency = pd.crosstab(combined_df[col].fillna('MISSING'), combined_df['label'])
        # Apply chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        if p < 0.05:
            section_content.append(f"- **{col.replace('_', ' ').title()}** has a statistically significant association with truth label (p={p:.6f})")
    
    # For statement field
    # Check if there's a difference in length between true and false statements
    true_mean = combined_df[combined_df['label'] == True]['word_count'].mean()
    false_mean = combined_df[combined_df['label'] == False]['word_count'].mean()
    length_diff = abs(true_mean - false_mean)
    
    section_content.append("\n**Statement Field:**")
    if length_diff > 1:  # If there's more than a 1-word difference
        section_content.append(f"- {'TRUE' if true_mean > false_mean else 'FALSE'} statements are on average {length_diff:.1f} words longer")
    else:
        section_content.append("- Statement length shows minimal difference between TRUE and FALSE labels")
    
    # For numerical/count fields
    section_content.append("\n**Speaker Truth History:**")
    section_content.append(f"- Speaker credibility (ratio of true to total statements) has a {abs(point_biserial):.4f} correlation with statement truth")
    section_content.append(f"- {'Higher' if point_biserial > 0 else 'Lower'} credibility scores tend to be associated with TRUE statements")
    
    # Feature recommendations based on analysis
    section_content.append("\n### Feature Engineering Recommendations")
    
    # Categorical features
    section_content.append("**Categorical Features:**")
    section_content.append("- Encode categorical variables with high cardinality using techniques like:")
    section_content.append("  - Target encoding (replace categories with their mean target value)")
    section_content.append("  - Frequency encoding (replace categories with their frequency)")
    section_content.append("  - One-hot encoding for low-cardinality categorical variables")
    section_content.append("- Create grouped categories for sparse categorical values")
    
    # Text features
    section_content.append("\n**Statement Features:**")
    section_content.append("- Extract text length features (character count, word count)")
    section_content.append("- Create features for presence of specific keywords identified in analysis")
    section_content.append("- Implement text preprocessing (lowercase, remove stopwords, stemming/lemmatization)")
    section_content.append("- Consider TF-IDF or word embeddings for the statement text")
    
    # Speaker history features
    section_content.append("\n**Speaker History Features:**")
    section_content.append("- Implement the credibility score (true statements / total statements)")
    section_content.append("- Create features for the ratio of each truth category to total statements")
    section_content.append("- Generate a weighted credibility score (giving higher weight to extreme categories)")
    
    # Save the report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(section_content))
    
    print(f"Field analysis report saved to: {REPORT_PATH}")
    
    return combined_df

if __name__ == "__main__":
    print("Starting analysis of binary classification dataset - Field Analysis (4.4)...")
    
    # Load the data
    print("Loading data...")
    train_df, test_df, valid_df, combined_df = load_data()
    
    # Analyze fields
    print("Analyzing fields...")
    analyze_fields(train_df, test_df, valid_df, combined_df)
    
    print("Analysis complete!") 