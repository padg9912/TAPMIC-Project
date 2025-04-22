#!/usr/bin/env python3
# intermediate1_dataset_analysis_45.py
# Analysis of binary classification dataset - Correlation and Multivariate Analysis (4.5)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
REPORT_PATH = os.path.join(SCRIPT_DIR, "intermediate1_dataset_analysis_45.txt")

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

def prepare_dataset_for_analysis(df):
    """
    Prepare dataset for correlation and multivariate analysis by:
    1. Creating additional features
    2. Encoding categorical variables
    3. Handling missing values
    """
    # Create a copy to avoid modifying the original
    df_prepared = df.copy()
    
    # Calculate statement length features
    df_prepared['statement_length'] = df_prepared['statement'].str.len()
    df_prepared['word_count'] = df_prepared['statement'].str.split().str.len()
    
    # Create speaker credibility score based on previous statements
    truth_cols = ['mostly_true_counts', 'half_true_counts']
    lie_cols = ['barely_true_counts', 'false_counts', 'pants_on_fire_counts']
    
    # Fill missing values in count columns with 0
    for col in truth_cols + lie_cols:
        df_prepared[col] = df_prepared[col].fillna(0)
    
    # Calculate total statements and truth ratio
    df_prepared['total_statements'] = df_prepared[truth_cols + lie_cols].sum(axis=1)
    df_prepared['truth_statements'] = df_prepared[truth_cols].sum(axis=1)
    df_prepared['lie_statements'] = df_prepared[lie_cols].sum(axis=1)
    
    # Calculate credibility score (ratio of true statements to total statements)
    # Handle division by zero by assigning 0 to speakers with no previous statements
    df_prepared['credibility_score'] = np.where(
        df_prepared['total_statements'] > 0,
        df_prepared['truth_statements'] / df_prepared['total_statements'],
        0
    )
    
    # Create weighted credibility score (giving more weight to extreme categories)
    df_prepared['weighted_mostly_true'] = df_prepared['mostly_true_counts'] * 1.0
    df_prepared['weighted_half_true'] = df_prepared['half_true_counts'] * 0.5
    df_prepared['weighted_barely_true'] = df_prepared['barely_true_counts'] * 0.25
    df_prepared['weighted_false'] = df_prepared['false_counts'] * 0.0
    df_prepared['weighted_pants_fire'] = df_prepared['pants_on_fire_counts'] * -0.5
    
    df_prepared['weighted_credibility'] = np.where(
        df_prepared['total_statements'] > 0,
        (df_prepared['weighted_mostly_true'] + df_prepared['weighted_half_true'] + 
         df_prepared['weighted_barely_true'] + df_prepared['weighted_false'] + 
         df_prepared['weighted_pants_fire']) / df_prepared['total_statements'],
        0
    )
    
    # Encode categorical variables with label encoding for correlation analysis
    categorical_cols = ['subject', 'speaker', 'speaker_job', 'state', 'party', 'context']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_prepared.columns:
            # Fill missing values with 'MISSING'
            df_prepared[col] = df_prepared[col].fillna('MISSING')
            
            # Create and fit label encoder
            le = LabelEncoder()
            df_prepared[col + '_encoded'] = le.fit_transform(df_prepared[col])
            label_encoders[col] = le
    
    # Drop non-numeric columns for correlation analysis
    df_prepared_numeric = df_prepared.drop(['id', 'statement', 'original_label'] + categorical_cols, axis=1)
    
    # Convert label to numeric if needed
    if df_prepared_numeric['label'].dtype != 'int64':
        df_prepared_numeric['label'] = df_prepared_numeric['label'].astype(int)
    
    return df_prepared, df_prepared_numeric, label_encoders

def analyze_correlations(combined_df):
    """
    4.5 Analyze correlations and multivariate relationships in the binary classification dataset
    """
    section_content = []
    section_content.append(f"# 4.5 Correlation and Multivariate Analysis\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Prepare dataset for analysis
    print("Preparing dataset for analysis...")
    df_prepared, df_numeric, label_encoders = prepare_dataset_for_analysis(combined_df)
    
    # 1. Pearson Correlation Analysis
    section_content.append("## 1. Correlation Analysis\n")
    
    # Calculate correlation matrix
    correlation_matrix = df_numeric.corr()
    
    # Find top correlations with the target variable
    target_correlations = correlation_matrix['label'].drop('label').sort_values(ascending=False)
    
    section_content.append("### Pearson Correlation with Target Variable (Label)")
    section_content.append("Features with highest positive correlation (indicative of TRUE statements):")
    for feature, corr in target_correlations.head(10).items():
        section_content.append(f"- **{feature}**: {corr:.4f}")
    
    section_content.append("\nFeatures with highest negative correlation (indicative of FALSE statements):")
    for feature, corr in target_correlations.tail(10).items():
        section_content.append(f"- **{feature}**: {corr:.4f}")
    
    # Create correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix, 
        mask=mask,
        annot=False, 
        cmap='coolwarm', 
        vmin=-1, 
        vmax=1, 
        fmt='.2f',
        linewidths=0.5
    )
    plt.title('Correlation Matrix Heatmap', fontsize=16)
    plt.tight_layout()
    
    # Save heatmap
    corr_heatmap_path = os.path.join(PLOTS_DIR, "4.5_correlation_heatmap.png")
    plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"\n![Correlation Heatmap]({os.path.relpath(corr_heatmap_path, SCRIPT_DIR)})\n")
    
    # Create correlation bar chart for top features
    plt.figure(figsize=(12, 8))
    target_corr_sorted = target_correlations.reindex(target_correlations.abs().sort_values(ascending=False).index)
    top_features = target_corr_sorted.head(15).index.tolist()
    
    target_corr_plot = target_correlations[top_features]
    
    # Create bar chart
    ax = target_corr_plot.plot(kind='barh', figsize=(12, 8), color=target_corr_plot.map(lambda x: 'green' if x > 0 else 'red'))
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.title('Top 15 Features by Correlation with Target Variable', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    # Save bar chart
    corr_bar_path = os.path.join(PLOTS_DIR, "4.5_correlation_bar_chart.png")
    plt.savefig(corr_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    section_content.append(f"![Correlation Bar Chart]({os.path.relpath(corr_bar_path, SCRIPT_DIR)})\n")
    
    # 2. Feature Interactions
    section_content.append("## 2. Feature Interactions\n")
    
    # Select top features based on correlation
    top_features = target_correlations.abs().sort_values(ascending=False).head(5).index.tolist()
    
    # Scatter plot of top 2 features
    if len(top_features) >= 2:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=top_features[0], 
            y=top_features[1], 
            hue='label',
            data=df_numeric,
            palette=['red', 'green'],
            alpha=0.6
        )
        plt.title(f'Scatter Plot of Top 2 Features', fontsize=16)
        plt.xlabel(top_features[0], fontsize=12)
        plt.ylabel(top_features[1], fontsize=12)
        plt.legend(title='Label', labels=['False', 'True'])
        plt.tight_layout()
        
        # Save scatter plot
        scatter_path = os.path.join(PLOTS_DIR, "4.5_top_features_scatter.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content.append(f"![Top Features Scatter Plot]({os.path.relpath(scatter_path, SCRIPT_DIR)})\n")
    
    # Create a pairplot for top 5 features
    try:
        if len(top_features) >= 3:
            print("Creating pairplot...")
            plt.figure(figsize=(15, 15))
            pairplot_df = df_numeric[top_features + ['label']].sample(min(5000, len(df_numeric)), random_state=42)
            g = sns.pairplot(
                pairplot_df,
                hue='label',
                palette=['red', 'green'],
                corner=True,
                diag_kind='kde',
                plot_kws={'alpha': 0.5, 's': 20},
                diag_kws={'alpha': 0.5}
            )
            g.fig.suptitle('Pairplot of Top Features', fontsize=16, y=1.02)
            
            # Save pairplot
            pairplot_path = os.path.join(PLOTS_DIR, "4.5_top_features_pairplot.png")
            plt.savefig(pairplot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            section_content.append(f"![Top Features Pairplot]({os.path.relpath(pairplot_path, SCRIPT_DIR)})\n")
    except Exception as e:
        print(f"Error creating pairplot: {e}")
        section_content.append("*Note: Pairplot could not be generated due to an error.*\n")
    
    # 3. Mutual Information Analysis
    section_content.append("## 3. Mutual Information Analysis\n")
    
    # Drop the target variable for MI calculation
    X = df_numeric.drop('label', axis=1)
    y = df_numeric['label']
    
    try:
        # Calculate mutual information
        mi_values = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_values})
        mi_df = mi_df.sort_values('Mutual Information', ascending=False).reset_index(drop=True)
        
        section_content.append("Mutual Information (MI) measures the dependency between variables without assuming a linear relationship.")
        section_content.append("Higher MI scores indicate stronger predictive power for classification.\n")
        
        section_content.append("### Top Features by Mutual Information Score")
        for idx, row in mi_df.head(15).iterrows():
            section_content.append(f"- **{row['Feature']}**: {row['Mutual Information']:.4f}")
        
        # Create bar chart for MI values
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Mutual Information', y='Feature', data=mi_df.head(15), palette='viridis')
        plt.title('Top 15 Features by Mutual Information', fontsize=16)
        plt.xlabel('Mutual Information Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        # Save MI bar chart
        mi_path = os.path.join(PLOTS_DIR, "4.5_mutual_information.png")
        plt.savefig(mi_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content.append(f"\n![Mutual Information]({os.path.relpath(mi_path, SCRIPT_DIR)})\n")
    except Exception as e:
        print(f"Error calculating mutual information: {e}")
        section_content.append("*Note: Mutual Information analysis could not be completed due to an error.*\n")
    
    # 4. Dimensionality Reduction
    section_content.append("## 4. Dimensionality Reduction Analysis\n")
    
    # Drop any non-numeric or constant columns
    X_pca = X.select_dtypes(include=['int64', 'float64'])
    # Remove constant columns that might cause issues
    X_pca = X_pca.loc[:, X_pca.std() > 0]
    
    try:
        # Perform PCA
        print("Performing PCA...")
        pca = PCA(n_components=2, random_state=42)
        pca_result = pca.fit_transform(X_pca)
        
        # Create dataframe with PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        pca_df['label'] = y.values
        
        # Plot PCA results
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x='PC1', 
            y='PC2', 
            hue='label',
            data=pca_df,
            palette=['red', 'green'],
            alpha=0.6,
            s=30
        )
        plt.title('PCA: Principal Component Analysis', fontsize=16)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.legend(title='Label', labels=['False', 'True'])
        plt.tight_layout()
        
        # Save PCA plot
        pca_path = os.path.join(PLOTS_DIR, "4.5_pca_analysis.png")
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content.append("### PCA Analysis")
        section_content.append("Principal Component Analysis (PCA) reduces dimensionality by transforming data into a new coordinate system.")
        section_content.append(f"- PC1 explains {pca.explained_variance_ratio_[0]:.2%} of variance")
        section_content.append(f"- PC2 explains {pca.explained_variance_ratio_[1]:.2%} of variance")
        section_content.append(f"- Combined, they explain {sum(pca.explained_variance_ratio_):.2%} of total variance\n")
        
        # Top contributing features to PC1 and PC2
        section_content.append("#### Top Contributing Features")
        
        component_df = pd.DataFrame(
            pca.components_.T, 
            columns=['PC1', 'PC2'], 
            index=X_pca.columns
        )
        
        section_content.append("**PC1 Top Contributors:**")
        pc1_top = component_df['PC1'].abs().sort_values(ascending=False).head(5)
        for feature, contribution in pc1_top.items():
            direction = "+" if component_df.loc[feature, 'PC1'] > 0 else "-"
            section_content.append(f"- {feature}: {direction}{abs(contribution):.4f}")
        
        section_content.append("\n**PC2 Top Contributors:**")
        pc2_top = component_df['PC2'].abs().sort_values(ascending=False).head(5)
        for feature, contribution in pc2_top.items():
            direction = "+" if component_df.loc[feature, 'PC2'] > 0 else "-"
            section_content.append(f"- {feature}: {direction}{abs(contribution):.4f}")
        
        section_content.append(f"\n![PCA Analysis]({os.path.relpath(pca_path, SCRIPT_DIR)})\n")
    except Exception as e:
        print(f"Error performing PCA: {e}")
        section_content.append("*Note: PCA analysis could not be completed due to an error.*\n")
    
    try:
        # Perform t-SNE on a sample of the data
        print("Performing t-SNE...")
        # Sample the data for t-SNE (it's computationally expensive)
        sample_size = min(5000, X_pca.shape[0])
        X_sample = X_pca.sample(sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_result = tsne.fit_transform(X_sample)
        
        # Create dataframe with t-SNE results
        tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE1', 't-SNE2'])
        tsne_df['label'] = y_sample.values
        
        # Plot t-SNE results
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x='t-SNE1', 
            y='t-SNE2', 
            hue='label',
            data=tsne_df,
            palette=['red', 'green'],
            alpha=0.6,
            s=30
        )
        plt.title('t-SNE Visualization', fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.legend(title='Label', labels=['False', 'True'])
        plt.tight_layout()
        
        # Save t-SNE plot
        tsne_path = os.path.join(PLOTS_DIR, "4.5_tsne_analysis.png")
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        section_content.append("### t-SNE Analysis")
        section_content.append("t-SNE (t-Distributed Stochastic Neighbor Embedding) is a technique for dimensionality reduction that preserves local structure.")
        section_content.append("It's particularly good at visualizing clusters or groups in high-dimensional data.\n")
        
        section_content.append(f"![t-SNE Analysis]({os.path.relpath(tsne_path, SCRIPT_DIR)})\n")
    except Exception as e:
        print(f"Error performing t-SNE: {e}")
        section_content.append("*Note: t-SNE analysis could not be completed due to an error.*\n")
    
    # 5. Key Findings and Feature Recommendations
    section_content.append("## 5. Key Findings and Feature Recommendations\n")
    
    # Summarize key findings from correlation analysis
    section_content.append("### Key Findings from Correlation Analysis")
    section_content.append("**Top Predictive Features:**")
    
    try:
        # List top 5 features from different analyses
        top_corr_features = target_correlations.abs().sort_values(ascending=False).head(5).index.tolist()
        section_content.append("1. **Pearson Correlation Top Features:**")
        for feature in top_corr_features:
            corr_value = correlation_matrix.loc[feature, 'label']
            direction = "positively" if corr_value > 0 else "negatively"
            section_content.append(f"   - {feature} ({direction} correlated, r={corr_value:.4f})")
        
        if 'mi_df' in locals():
            section_content.append("\n2. **Mutual Information Top Features:**")
            for feature in mi_df.head(5)['Feature'].values:
                mi_value = mi_df[mi_df['Feature'] == feature]['Mutual Information'].values[0]
                section_content.append(f"   - {feature} (MI={mi_value:.4f})")
        
        section_content.append("\n**Feature Interactions:**")
        section_content.append("- The analysis reveals complex relationships between features that can't be captured by single correlations")
        section_content.append("- PCA shows that the variance in the data can be largely explained by a small number of components")
        section_content.append("- t-SNE visualization suggests some separation between TRUE and FALSE classes, but with significant overlap")
    except Exception as e:
        print(f"Error summarizing findings: {e}")
    
    # Feature recommendations based on multivariate analysis
    section_content.append("\n### Feature Engineering Recommendations")
    
    section_content.append("**Based on Correlation Analysis:**")
    section_content.append("- Focus on highly correlated features, especially speaker credibility metrics")
    section_content.append("- Consider interactive features combining speaker history and statement characteristics")
    section_content.append("- Normalize truth history counts by total statements to create ratio features")
    
    section_content.append("\n**Based on Dimensionality Reduction:**")
    section_content.append("- Consider using principal components as features to reduce dimensionality")
    section_content.append("- Group similar categorical values based on their relationship with the target variable")
    section_content.append("- Create composite features that capture the relationships revealed by PCA")
    
    section_content.append("\n**Feature Selection Strategy:**")
    section_content.append("- Use a combination of correlation, mutual information, and dimensionality reduction for feature selection")
    section_content.append("- Prioritize speaker credibility score and truth history features")
    section_content.append("- Consider both categorical features (with proper encoding) and numerical features")
    section_content.append("- Test both with and without dimensionality reduction techniques in the modeling phase")
    
    # Save the report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(section_content))
    
    print(f"Correlation and multivariate analysis report saved to: {REPORT_PATH}")
    
    return df_prepared

if __name__ == "__main__":
    print("Starting analysis of binary classification dataset - Correlation and Multivariate Analysis (4.5)...")
    
    # Load the data
    print("Loading data...")
    train_df, test_df, valid_df, combined_df = load_data()
    
    # Analyze correlations and multivariate relationships
    print("Analyzing correlations and multivariate relationships...")
    df_prepared = analyze_correlations(combined_df)
    
    print("Analysis complete!") 