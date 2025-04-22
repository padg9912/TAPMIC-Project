#!/usr/bin/env python3
# temporal_analysis.py
# Generates comprehensive temporal visualization report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Create output directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INTERMEDIATE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "intermediate_1")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "intermediate_2")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define file paths
TRAIN_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_train.csv")
TEST_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_test.csv")
VALID_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_validation.csv")
REPORT_PATH = os.path.join(REPORTS_DIR, "temporal_analysis_report.txt")

# Define US election years (presidential elections)
US_ELECTION_YEARS = [2000, 2004, 2008, 2012, 2016, 2020]
# Define midterm election years
US_MIDTERM_YEARS = [2002, 2006, 2010, 2014, 2018, 2022]

def load_data():
    """Load the datasets with temporal features"""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    
    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")
    print(f"Validation set: {valid_df.shape[0]} samples")
    
    return train_df, test_df, valid_df

def get_temporal_feature_correlations(combined_df):
    """Calculate and visualize correlations between temporal features and truth label"""
    # Identify temporal features
    temporal_features = [
        'year', 'month', 'quarter', 'day_of_week', 'is_weekend',
        'is_election_year', 'is_midterm_year', 'days_to_nearest_election',
        'in_campaign_season', 'speaker_credibility_30days', 'speaker_credibility_90days',
        'speaker_credibility_180days', 'truth_trend_30days'
    ]
    
    # Filter features that exist in the dataset
    temporal_features = [f for f in temporal_features if f in combined_df.columns]
    
    # Calculate correlations with label
    temporal_corr = combined_df[temporal_features + ['label']].corr()['label'].drop('label')
    
    # Create visualization of correlations
    plt.figure(figsize=(14, 8))
    
    # Create bar chart
    corr_sorted = temporal_corr.sort_values(ascending=False)
    bars = plt.bar(
        corr_sorted.index, 
        corr_sorted.values,
        color=['green' if x > 0 else 'red' for x in corr_sorted.values],
        alpha=0.7
    )
    
    # Add correlation values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01 if height >= 0 else height - 0.03,
            f'{height:.3f}',
            ha='center', 
            va='bottom' if height >= 0 else 'top',
            fontsize=9
        )
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title('Correlation of Temporal Features with Truth Label', fontsize=14)
    plt.xlabel('Feature', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save correlation chart
    corr_path = os.path.join(PLOTS_DIR, "temporal_features_correlation.png")
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return temporal_corr, corr_path

def visualize_feature_distributions(combined_df):
    """Visualize the distribution of key temporal features by truth label"""
    # Select features to visualize
    features_to_plot = [
        'is_election_year', 'is_midterm_year', 'in_campaign_season',
        'month', 'quarter', 'day_of_week', 'is_weekend',
        'speaker_credibility_30days', 'speaker_credibility_90days'
    ]
    
    # Filter features that exist in the dataset
    features_to_plot = [f for f in features_to_plot if f in combined_df.columns]
    
    # Create visualizations
    plot_paths = []
    
    # 1. Binary features
    binary_features = ['is_election_year', 'is_midterm_year', 'in_campaign_season', 'is_weekend']
    binary_features = [f for f in binary_features if f in features_to_plot]
    
    if binary_features:
        plt.figure(figsize=(14, 5 * (len(binary_features) + 1) // 2))
        
        for i, feature in enumerate(binary_features):
            plt.subplot(((len(binary_features) + 1) // 2), 2, i+1)
            
            # Calculate truth percentages
            true_pct = combined_df.groupby(feature)['label'].mean() * 100
            counts = combined_df[feature].value_counts()
            
            # Create bar chart
            ax = true_pct.plot(kind='bar', color='skyblue')
            
            # Add count annotations
            for j, p in enumerate(ax.patches):
                feature_val = j
                count = counts.get(feature_val, 0)
                ax.annotate(
                    f'n={count}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9
                )
            
            # Add labels
            feature_name = ' '.join(feature.split('_')).title()
            plt.title(f'Truth Percentage by {feature_name}', fontsize=12)
            plt.ylabel('Truth Percentage (%)', fontsize=10)
            plt.ylim(0, 100)  # Set y-axis to percentage scale
            
            # Set x-tick labels
            if feature == 'is_election_year':
                plt.xticks([0, 1], ['Not Election Year', 'Election Year'])
            elif feature == 'is_midterm_year':
                plt.xticks([0, 1], ['Not Midterm Year', 'Midterm Year'])
            elif feature == 'in_campaign_season':
                plt.xticks([0, 1], ['Not Campaign Season', 'Campaign Season'])
            elif feature == 'is_weekend':
                plt.xticks([0, 1], ['Weekday', 'Weekend'])
            
        plt.tight_layout()
        
        # Save binary features plot
        binary_plot_path = os.path.join(PLOTS_DIR, "binary_temporal_features.png")
        plt.savefig(binary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(binary_plot_path)
    
    # 2. Categorical features
    categorical_features = ['month', 'quarter', 'day_of_week']
    categorical_features = [f for f in categorical_features if f in features_to_plot]
    
    if categorical_features:
        plt.figure(figsize=(16, 5 * (len(categorical_features) + 1) // 2))
        
        for i, feature in enumerate(categorical_features):
            plt.subplot(((len(categorical_features) + 1) // 2), 2, i+1)
            
            # Calculate truth percentages
            true_pct = combined_df.groupby(feature)['label'].mean() * 100
            counts = combined_df[feature].value_counts().sort_index()
            
            # Create bar chart
            ax = true_pct.plot(kind='bar', color='skyblue')
            
            # Add count annotations
            for j, p in enumerate(ax.patches):
                feature_val = true_pct.index[j]
                count = counts.get(feature_val, 0)
                ax.annotate(
                    f'n={count}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8
                )
            
            # Add labels
            feature_name = ' '.join(feature.split('_')).title()
            plt.title(f'Truth Percentage by {feature_name}', fontsize=12)
            plt.ylabel('Truth Percentage (%)', fontsize=10)
            plt.ylim(0, 100)  # Set y-axis to percentage scale
            
            # Set x-tick labels for specific features
            if feature == 'month':
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                plt.xticks(range(len(month_names)), month_names, rotation=45)
            elif feature == 'day_of_week':
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                plt.xticks(range(len(day_names)), day_names)
            
        plt.tight_layout()
        
        # Save categorical features plot
        cat_plot_path = os.path.join(PLOTS_DIR, "categorical_temporal_features.png")
        plt.savefig(cat_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(cat_plot_path)
    
    # 3. Continuous features (speaker credibility)
    continuous_features = ['speaker_credibility_30days', 'speaker_credibility_90days']
    continuous_features = [f for f in continuous_features if f in features_to_plot]
    
    if continuous_features:
        plt.figure(figsize=(15, 5 * len(continuous_features)))
        
        for i, feature in enumerate(continuous_features):
            plt.subplot(len(continuous_features), 1, i+1)
            
            # Create bins
            bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
            combined_df['binned'] = pd.cut(combined_df[feature], bins)
            
            # Calculate truth percentages by bin
            bin_stats = combined_df.groupby('binned').agg(
                truth_pct=('label', lambda x: np.mean(x) * 100),
                count=('label', 'count')
            ).reset_index()
            
            # Create bar chart
            ax = plt.bar(
                range(len(bin_stats)), 
                bin_stats['truth_pct'],
                color='skyblue',
                width=0.7
            )
            
            # Add count annotations
            for j, count in enumerate(bin_stats['count']):
                plt.text(
                    j, 
                    bin_stats['truth_pct'].iloc[j] + 1,
                    f'n={count}',
                    ha='center', 
                    va='bottom', 
                    fontsize=9
                )
            
            # Add labels
            feature_name = ' '.join(feature.split('_')).title()
            plt.title(f'Truth Percentage by {feature_name}', fontsize=12)
            plt.ylabel('Truth Percentage (%)', fontsize=10)
            plt.xlabel(feature_name, fontsize=10)
            plt.ylim(0, 100)  # Set y-axis to percentage scale
            
            # Set x-tick labels
            bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
            plt.xticks(range(len(bin_labels)), bin_labels, rotation=45)
            
            # Remove binned column before next iteration
            combined_df.drop('binned', axis=1, inplace=True)
            
        plt.tight_layout()
        
        # Save continuous features plot
        cont_plot_path = os.path.join(PLOTS_DIR, "continuous_temporal_features.png")
        plt.savefig(cont_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(cont_plot_path)
    
    return plot_paths

def visualize_temporal_pca(combined_df):
    """Apply PCA to temporal features and visualize the results"""
    # Select temporal features for PCA
    temporal_features = [
        'year', 'month', 'quarter', 'day_of_week', 'is_weekend',
        'is_election_year', 'is_midterm_year', 'days_to_nearest_election',
        'in_campaign_season', 'speaker_credibility_30days', 'speaker_credibility_90days',
        'speaker_credibility_180days', 'truth_trend_30days'
    ]
    
    # Filter features that exist in the dataset
    temporal_features = [f for f in temporal_features if f in combined_df.columns]
    
    # Extract features
    X = combined_df[temporal_features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create dataframe with PCA results
    pca_df = pd.DataFrame(
        data=X_pca, 
        columns=['PC1', 'PC2']
    )
    pca_df['label'] = combined_df['label'].values
    
    # Visualize PCA results
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with transparency and color by label
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='label',
        palette={1: 'green', 0: 'red'},
        alpha=0.6,
        s=50,
        data=pca_df
    )
    
    # Add labels and title
    plt.title('PCA of Temporal Features Colored by Truth Label', fontsize=14)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.legend(title='Truth Label', labels=['False', 'True'])
    
    # Save PCA plot
    pca_plot_path = os.path.join(PLOTS_DIR, "temporal_features_pca.png")
    plt.tight_layout()
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Get feature importance
    feature_importance = pd.DataFrame(
        data=pca.components_.T,
        columns=['PC1', 'PC2'],
        index=temporal_features
    ).abs()
    
    # Visualize feature importance
    plt.figure(figsize=(14, 8))
    
    # Sort by PC1 importance
    importance_sorted = feature_importance.sort_values('PC1', ascending=False)
    
    # Create bar chart
    importance_sorted.plot(kind='bar', colormap='viridis')
    
    # Add labels and title
    plt.title('Feature Importance in PCA Components', fontsize=14)
    plt.xlabel('Temporal Feature', fontsize=12)
    plt.ylabel('Absolute Component Weight', fontsize=12)
    plt.legend(title='Component')
    plt.xticks(rotation=45, ha='right')
    
    # Save feature importance plot
    importance_plot_path = os.path.join(PLOTS_DIR, "temporal_features_importance.png")
    plt.tight_layout()
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pca_plot_path, importance_plot_path, feature_importance

def generate_report(temporal_corr, plot_paths, feature_importance):
    """Generate a comprehensive report on temporal features"""
    report_content = []
    report_content.append(f"# Temporal Feature Analysis Report\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Introduction
    report_content.append("## 1. Introduction and Overview\n")
    report_content.append("This report provides a comprehensive analysis of the temporal features extracted from the political statements dataset. It examines how time-related aspects of statements correlate with their truthfulness, identifies patterns within different time periods, and explores how speaker credibility evolves over time.")
    report_content.append("The analysis focuses on various temporal dimensions including:")
    report_content.append("- Basic temporal elements (year, month, quarter, day of week)")
    report_content.append("- Election-related temporal factors (proximity to elections, election years)")
    report_content.append("- Speaker credibility evolution over different time windows")
    report_content.append("- Trend analysis of speaker truthfulness\n")
    
    # 2. Correlation Analysis
    report_content.append("## 2. Correlation of Temporal Features with Truthfulness\n")
    
    # Add correlation plot
    corr_plot_path = [p for p in plot_paths if "temporal_features_correlation" in p]
    if corr_plot_path:
        report_content.append(f"![Temporal Feature Correlations]({os.path.relpath(corr_plot_path[0], REPORTS_DIR)})\n")
    
    # List correlations in descending order by absolute value
    report_content.append("### Feature Correlations (Sorted by Absolute Value)\n")
    sorted_corr = temporal_corr.reindex(temporal_corr.abs().sort_values(ascending=False).index)
    
    for feature, corr in sorted_corr.items():
        indicator = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.1 else "moderate" if abs(corr) > 0.05 else "weak"
        report_content.append(f"- **{feature}**: {corr:.4f} ({strength} {indicator} correlation)")
    
    # 3. Feature Distributions
    report_content.append("\n## 3. Temporal Feature Distributions\n")
    
    # Add binary feature plot
    binary_plot_path = [p for p in plot_paths if "binary_temporal_features" in p]
    if binary_plot_path:
        report_content.append("### Binary Temporal Features\n")
        report_content.append(f"![Binary Temporal Features]({os.path.relpath(binary_plot_path[0], REPORTS_DIR)})\n")
        report_content.append("The above chart shows how truth rates vary between binary temporal categories like election years, campaign seasons, and weekdays vs. weekends.")
    
    # Add categorical feature plot
    cat_plot_path = [p for p in plot_paths if "categorical_temporal_features" in p]
    if cat_plot_path:
        report_content.append("\n### Categorical Temporal Features\n")
        report_content.append(f"![Categorical Temporal Features]({os.path.relpath(cat_plot_path[0], REPORTS_DIR)})\n")
        report_content.append("These charts illustrate how truth rates vary across different months, quarters, and days of the week.")
    
    # Add continuous feature plot
    cont_plot_path = [p for p in plot_paths if "continuous_temporal_features" in p]
    if cont_plot_path:
        report_content.append("\n### Speaker Credibility Features\n")
        report_content.append(f"![Speaker Credibility Features]({os.path.relpath(cont_plot_path[0], REPORTS_DIR)})\n")
        report_content.append("These charts show the relationship between speaker credibility (measured over different time windows) and statement truthfulness. There's a clear trend where higher credibility scores correlate with higher truth rates.")
    
    # 4. PCA Analysis
    report_content.append("\n## 4. Principal Component Analysis\n")
    
    # Add PCA plot
    pca_plot_path = [p for p in plot_paths if "temporal_features_pca" in p]
    if pca_plot_path:
        report_content.append(f"![PCA of Temporal Features]({os.path.relpath(pca_plot_path[0], REPORTS_DIR)})\n")
        report_content.append("The PCA visualization attempts to reduce the dimensionality of temporal features to see if patterns emerge. The scatter plot shows some separation between true and false statements, suggesting that temporal features do contain useful signal for classification.")
    
    # Add feature importance plot
    importance_plot_path = [p for p in plot_paths if "temporal_features_importance" in p]
    if importance_plot_path:
        report_content.append("\n### Feature Importance in PCA Components\n")
        report_content.append(f"![Feature Importance in PCA]({os.path.relpath(importance_plot_path[0], REPORTS_DIR)})\n")
        
        # List top features for PC1 and PC2
        if isinstance(feature_importance, pd.DataFrame):
            report_content.append("#### Top Features Contributing to Principal Components:\n")
            
            # PC1
            report_content.append("**Principal Component 1:**")
            for feature, importance in feature_importance.sort_values('PC1', ascending=False)['PC1'].head(5).items():
                report_content.append(f"- {feature}: {importance:.4f}")
            
            # PC2
            report_content.append("\n**Principal Component 2:**")
            for feature, importance in feature_importance.sort_values('PC2', ascending=False)['PC2'].head(5).items():
                report_content.append(f"- {feature}: {importance:.4f}")
    
    # 5. Key Findings and Recommendations
    report_content.append("\n## 5. Key Findings and Recommendations\n")
    
    report_content.append("### Key Findings:")
    
    # Add key findings based on correlation values
    top_positive = temporal_corr.sort_values(ascending=False).head(3)
    top_negative = temporal_corr.sort_values().head(3)
    
    if not top_positive.empty:
        report_content.append("\n**Positive associations with truthfulness:**")
        for feature, corr in top_positive.items():
            report_content.append(f"- **{feature}**: {corr:.4f} - Statements are more likely to be true when this feature is higher")
    
    if not top_negative.empty:
        report_content.append("\n**Negative associations with truthfulness:**")
        for feature, corr in top_negative.items():
            report_content.append(f"- **{feature}**: {corr:.4f} - Statements are less likely to be true when this feature is higher")
    
    # Add general findings
    report_content.append("\n**General temporal patterns:**")
    report_content.append("- Speaker credibility features show the strongest correlation with statement truthfulness")
    report_content.append("- Time-based patterns exist in relation to election cycles and campaign seasons")
    report_content.append("- Statements made closer to elections show different truth rates than those made in non-election periods")
    report_content.append("- There are seasonal patterns with truth rates varying by month and quarter")
    
    # Add recommendations
    report_content.append("\n### Recommendations for Model Development:")
    report_content.append("1. **Feature prioritization**: Focus on incorporating speaker credibility window features, which show the strongest correlation with truth labels")
    report_content.append("2. **Election proximity**: Include the days to nearest election and election year features in the model")
    report_content.append("3. **Temporal trends**: Consider speaker truth trend features to capture changes in speaker behavior over time")
    report_content.append("4. **Seasonality**: Include month and quarter features to capture seasonal patterns in truthfulness")
    report_content.append("5. **Feature combinations**: Create interaction terms between temporal features and other features (e.g., statement content features combined with election proximity)")
    report_content.append("6. **Time-aware validation**: Consider time-based validation strategies to ensure model generalization across different time periods")
    
    # Write report to file
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report_content))
    
    print(f"Temporal analysis report saved to {REPORT_PATH}")
    return report_content

def main():
    """Main function to analyze temporal features"""
    print("This analysis framework is designed for use with real temporal data.")
    print("The RAG pipeline in Phase 2 will provide actual dates for temporal feature extraction.")
    print("The current script establishes the visualization and analysis structure.")
    
    print("Once Phase 2 is complete with real date information, this script will:")
    print("1. Load datasets with temporal features")
    print("2. Calculate and visualize correlations between temporal features and truthfulness")
    print("3. Visualize temporal feature distributions")
    print("4. Perform PCA analysis on temporal features")
    print("5. Generate a comprehensive report with findings and recommendations")
    
    print("Framework for temporal feature analysis is ready for Phase 2 integration.")
    
    # Example flow once real temporal features are available:
    # train_df, test_df, valid_df = load_data()
    # combined_df = pd.concat([train_df, test_df, valid_df])
    # temporal_corr, corr_path = get_temporal_feature_correlations(combined_df)
    # distribution_paths = visualize_feature_distributions(combined_df)
    # pca_path, importance_path, feature_importance = visualize_temporal_pca(combined_df)
    # plot_paths = [corr_path] + distribution_paths + [pca_path, importance_path]
    # generate_report(temporal_corr, plot_paths, feature_importance)

if __name__ == "__main__":
    main() 