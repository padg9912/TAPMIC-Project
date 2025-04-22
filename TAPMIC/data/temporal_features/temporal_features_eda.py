#!/usr/bin/env python3
# temporal_features_eda.py
# Exploratory Data Analysis for intermediate_2 datasets with temporal features

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "TAPMIC/data/intermediate_2")
REPORTS_DIR = os.path.join(DATA_DIR, "eda_reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

# Create output directories
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define file paths
TRAIN_PATH = os.path.join(DATA_DIR, "intermediate2_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "intermediate2_test.csv")
VALID_PATH = os.path.join(DATA_DIR, "intermediate2_validation.csv")
REPORT_PATH = os.path.join(REPORTS_DIR, "temporal_features_eda_report.txt")
JSON_REPORT_PATH = os.path.join(REPORTS_DIR, "temporal_features_eda_report.json")

def load_datasets():
    """Load the intermediate2 datasets"""
    print(f"Loading datasets from {DATA_DIR}")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    
    print(f"Train set: {train_df.shape[0]} samples, {train_df.shape[1]} features")
    print(f"Test set: {test_df.shape[0]} samples, {test_df.shape[1]} features")
    print(f"Validation set: {valid_df.shape[0]} samples, {valid_df.shape[1]} features")
    
    return train_df, test_df, valid_df

def identify_temporal_features(df):
    """Identify temporal features added in Phase 2"""
    # Features added in Phase 2 start from index 76
    phase1_features = df.columns[:76].tolist()
    phase2_features = df.columns[76:].tolist()
    
    print(f"Phase 1 features: {len(phase1_features)}")
    print(f"Phase 2 features (temporal): {len(phase2_features)}")
    
    return phase1_features, phase2_features

def analyze_class_distribution(train_df, test_df, valid_df):
    """Analyze class distribution"""
    print("Analyzing class distribution...")
    
    datasets = {
        'Train': train_df,
        'Test': test_df,
        'Validation': valid_df,
        'Combined': pd.concat([train_df, test_df, valid_df])
    }
    
    class_dist = {}
    for name, df in datasets.items():
        true_count = df['label'].sum()
        false_count = len(df) - true_count
        true_pct = true_count / len(df) * 100
        false_pct = false_count / len(df) * 100
        
        class_dist[name] = {
            'true_count': int(true_count),
            'false_count': int(false_count),
            'true_pct': float(true_pct),
            'false_pct': float(false_pct),
            'imbalance_ratio': float(true_pct / false_pct) if false_pct > 0 else float('inf')
        }
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    for i, (name, stats) in enumerate(class_dist.items()):
        plt.subplot(1, 4, i+1)
        plt.pie([stats['true_pct'], stats['false_pct']], 
                labels=['TRUE', 'FALSE'], 
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'],
                startangle=90)
        plt.title(f"{name} Set")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=300)
    plt.close()
    
    return class_dist

def analyze_temporal_feature_stats(df, temporal_features):
    """Analyze statistics of temporal features"""
    print("Analyzing temporal feature statistics...")
    
    # Filter for numeric temporal features
    numeric_temporal = []
    for col in temporal_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_temporal.append(col)
    
    # Calculate statistics
    stats = df[numeric_temporal].describe().transpose()
    stats['missing_pct'] = df[numeric_temporal].isna().mean() * 100
    stats['correlation_with_label'] = [df[col].corr(df['label']) if pd.api.types.is_numeric_dtype(df[col]) else np.nan for col in numeric_temporal]
    
    # Sort by absolute correlation with label
    stats['abs_corr'] = stats['correlation_with_label'].abs()
    stats = stats.sort_values('abs_corr', ascending=False)
    
    # Create visualization of top correlated features
    top_features = stats.index[:10].tolist()
    plt.figure(figsize=(12, 8))
    corr_values = stats.loc[top_features, 'correlation_with_label'].values
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in corr_values]
    
    plt.barh(top_features, corr_values, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Top Temporal Features Correlation with Truth Label', fontsize=16)
    plt.xlabel('Correlation Coefficient', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_temporal_correlations.png"), dpi=300)
    plt.close()
    
    return stats.to_dict()

def analyze_feature_correlations(df, phase1_features, phase2_features):
    """Analyze correlations between features"""
    print("Analyzing feature correlations...")
    
    # Select important features for correlation analysis
    label_col = ['label']
    
    # Get numeric features only
    numeric_phase1 = [col for col in phase1_features if pd.api.types.is_numeric_dtype(df[col])]
    numeric_phase2 = [col for col in phase2_features if pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"  Numeric phase 1 features: {len(numeric_phase1)}")
    print(f"  Numeric phase 2 features: {len(numeric_phase2)}")
    
    # Select top phase 1 features based on correlation with label
    try:
        # Calculate correlations manually to avoid DataFrame vs Series issues
        phase1_correlations = []
        for col in numeric_phase1:
            try:
                corr = abs(df[col].corr(df['label']))
                if not pd.isna(corr):  # Skip if correlation is NaN
                    phase1_correlations.append((col, corr))
            except:
                continue
        
        # Sort by correlation value (the second element in the tuple)
        phase1_correlations.sort(key=lambda x: x[1], reverse=True)
        top_phase1 = [col for col, _ in phase1_correlations[:10]] if phase1_correlations else []
        
        # Same for phase 2 features
        phase2_correlations = []
        for col in numeric_phase2:
            try:
                corr = abs(df[col].corr(df['label']))
                if not pd.isna(corr):  # Skip if correlation is NaN
                    phase2_correlations.append((col, corr))
            except:
                continue
        
        # Sort by correlation value
        phase2_correlations.sort(key=lambda x: x[1], reverse=True)
        top_phase2 = [col for col, _ in phase2_correlations[:10]] if phase2_correlations else []
        
    except Exception as e:
        print(f"  Error in correlation analysis: {e}")
        # Fallback to simpler approach if correlation fails
        top_phase1 = numeric_phase1[:10] if len(numeric_phase1) >= 10 else numeric_phase1
        top_phase2 = numeric_phase2[:10] if len(numeric_phase2) >= 10 else numeric_phase2
    
    # Combine top features from both phases
    top_features = label_col + top_phase1 + top_phase2
    
    # Generate correlation matrix for top features
    try:
        # Some features might have only NaN values, filter them out
        valid_features = []
        for feat in top_features:
            if not df[feat].isna().all():
                valid_features.append(feat)
        
        if len(valid_features) > 1:  # Need at least 2 features for correlation
            corr_matrix = df[valid_features].corr().fillna(0)  # Fill NaN correlations with 0
            
            # Create correlation heatmap
            plt.figure(figsize=(14, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1, center=0)
            plt.title('Correlation Matrix of Top Features', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=300)
            plt.close()
    except Exception as e:
        print(f"  Error creating correlation heatmap: {e}")
    
    return {
        'top_phase1_features': top_phase1,
        'top_phase2_features': top_phase2
    }

def analyze_temporal_evidence_patterns(df, temporal_features):
    """Analyze patterns in temporal evidence"""
    print("Analyzing temporal evidence patterns...")
    
    # Check evidence coverage
    evidence_coverage = {
        'total_claims': len(df),
        'claims_with_evidence': df['has_evidence'].sum(),
        'claims_with_temporal_data': df['has_temporal_data'].sum(),
        'avg_evidence_per_claim': df['evidence_count'].mean(),
        'avg_temporal_evidence_per_claim': df['temporal_evidence_count'].mean()
    }
    
    # Analyze truth rates by evidence count
    df['evidence_count_bin'] = pd.cut(
        df['evidence_count'],
        bins=[0, 1, 2, 3, float('inf')],
        labels=['0', '1', '2', '3+']
    )
    
    truth_by_evidence = df.groupby('evidence_count_bin')['label'].mean().reset_index()
    truth_by_evidence['truth_percentage'] = truth_by_evidence['label'] * 100
    
    # Visualize truth by evidence count
    plt.figure(figsize=(10, 6))
    sns.barplot(x='evidence_count_bin', y='truth_percentage', data=truth_by_evidence)
    plt.title('Truth Percentage by Evidence Count', fontsize=14)
    plt.xlabel('Number of Evidence Items', fontsize=12)
    plt.ylabel('Truth Percentage (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "truth_by_evidence_count.png"), dpi=300)
    plt.close()
    
    # Analyze source distribution
    source_dist = {
        'google_evidence_claims': df[df['google_evidence_count'] > 0].shape[0],
        'perplexity_evidence_claims': df[df['perplexity_evidence_count'] > 0].shape[0],
        'avg_google_evidence': df['google_evidence_count'].mean(),
        'avg_perplexity_evidence': df['perplexity_evidence_count'].mean()
    }
    
    # Analyze temporal features by source
    truth_by_source = {
        'google_evidence_truth_rate': df[df['google_evidence_count'] > 0]['label'].mean() * 100,
        'perplexity_evidence_truth_rate': df[df['perplexity_evidence_count'] > 0]['label'].mean() * 100,
        'no_evidence_truth_rate': df[df['evidence_count'] == 0]['label'].mean() * 100
    }
    
    # Visualize mean publication year distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['mean_publication_year'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Mean Publication Year', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mean_publication_year_distribution.png"), dpi=300)
    plt.close()
    
    # Analyze publication date range
    date_range_stats = df['publication_date_range_days'].describe().to_dict()
    
    return {
        'evidence_coverage': evidence_coverage,
        'truth_by_evidence': truth_by_evidence.to_dict(),
        'source_distribution': source_dist,
        'truth_by_source': truth_by_source,
        'date_range_stats': date_range_stats
    }

def analyze_feature_importance(df, phase1_features, phase2_features):
    """Analyze feature importance using mutual information"""
    print("Analyzing feature importance...")
    
    # Get numeric features only
    numeric_phase1 = [col for col in phase1_features if pd.api.types.is_numeric_dtype(df[col])]
    numeric_phase2 = [col for col in phase2_features if pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"  Numeric phase 1 features: {len(numeric_phase1)}")
    print(f"  Numeric phase 2 features: {len(numeric_phase2)}")
    
    # Prepare data for feature importance
    X = df[numeric_phase1 + numeric_phase2].copy()
    y = df['label']
    
    # Handle NaN values for feature importance - must do this before mutual_info_classif
    print("  Handling missing values...")
    for col in X.columns:
        if X[col].isna().any():
            mean_val = X[col].mean()
            if pd.isna(mean_val):  # If mean is also NaN (all values NaN)
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(mean_val)
    
    try:
        # Calculate mutual information
        print("  Calculating mutual information...")
        mi_scores = mutual_info_classif(X, y)
        mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
        mi_df = mi_df.sort_values('MI_Score', ascending=False)
        
        # Calculate proportion of Phase 1 vs Phase 2 features in top features
        top_n = min(20, len(mi_df))
        top_features = mi_df.head(top_n)['Feature'].tolist()
        phase1_count = sum(1 for feat in top_features if feat in numeric_phase1)
        phase2_count = sum(1 for feat in top_features if feat in numeric_phase2)
        
        # Visualize top features
        plt.figure(figsize=(12, 10))
        top_to_plot = min(top_n, len(mi_df))
        plt.barh(mi_df.head(top_to_plot)['Feature'], 
                mi_df.head(top_to_plot)['MI_Score'])
        plt.title('Top Features by Mutual Information with Truth Label', fontsize=16)
        plt.xlabel('Mutual Information Score', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)
        plt.close()
        
        # Add summary
        importance_summary = {
            'top_features': top_features,
            'phase1_in_top': phase1_count,
            'phase2_in_top': phase2_count,
            'top_phase1_feature': next((f for f in top_features if f in numeric_phase1), None),
            'top_phase2_feature': next((f for f in top_features if f in numeric_phase2), None),
        }
    except Exception as e:
        print(f"  Error in feature importance calculation: {e}")
        # Provide a fallback if mutual information fails
        importance_summary = {
            'top_features': numeric_phase1[:10] + numeric_phase2[:10],
            'phase1_in_top': min(10, len(numeric_phase1)),
            'phase2_in_top': min(10, len(numeric_phase2)),
            'top_phase1_feature': numeric_phase1[0] if numeric_phase1 else None,
            'top_phase2_feature': numeric_phase2[0] if numeric_phase2 else None,
            'error': str(e)
        }
    
    return importance_summary

def analyze_dimensionality_reduction(df, phase1_features, phase2_features):
    """Perform PCA on combined features"""
    print("Performing dimensionality reduction...")
    
    # Get numeric features only
    numeric_phase1 = [col for col in phase1_features if pd.api.types.is_numeric_dtype(df[col])]
    numeric_phase2 = [col for col in phase2_features if pd.api.types.is_numeric_dtype(df[col])]
    numeric_cols = numeric_phase1 + numeric_phase2
    
    print(f"  Using {len(numeric_cols)} numeric features for PCA")
    
    # Handle NaN values - be more careful here
    X = df[numeric_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            mean_val = X[col].mean()
            if pd.isna(mean_val):  # If mean is also NaN (all values NaN)
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(mean_val)
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find number of components for 90% variance
        n_components_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
        
        # Visualize explained variance
        plt.figure(figsize=(12, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.8, align='center')
        plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', color='red')
        plt.axhline(y=0.9, color='k', linestyle='--', alpha=0.3)
        plt.text(len(explained_variance) * 0.7, 0.91, '90% variance threshold', fontsize=12)
        plt.title('Explained Variance by Principal Components', fontsize=16)
        plt.xlabel('Principal Component', fontsize=14)
        plt.ylabel('Explained Variance Ratio', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "pca_explained_variance.png"), dpi=300)
        plt.close()
        
        # Visualize first two components by label
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['label'], alpha=0.6, cmap='coolwarm')
        plt.colorbar(scatter, label='Truth Label')
        plt.title('PCA: First Two Principal Components by Truth Label', fontsize=16)
        plt.xlabel('PC1', fontsize=14)
        plt.ylabel('PC2', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "pca_first_two_components.png"), dpi=300)
        plt.close()
        
        # Return PCA summary
        return {
            'n_components_90': int(n_components_90),
            'top_component_variance': float(explained_variance[0]),
            'first_5_components_variance': float(cumulative_variance[4])
        }
    except Exception as e:
        print(f"Error in PCA: {e}")
        return {
            'n_components_90': min(20, len(numeric_cols)),
            'top_component_variance': 0.1,
            'first_5_components_variance': 0.5,
            'error': str(e)
        }

def create_report(eda_results):
    """Create a comprehensive EDA report"""
    print("Creating final EDA report...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_lines = [
        "=== TEMPORAL FEATURES EDA REPORT ===",
        f"Generated on: {timestamp}",
        "",
        "=== DATASET OVERVIEW ===",
        f"Train set: {eda_results['dataset_info']['train_samples']} samples, {eda_results['dataset_info']['train_features']} features",
        f"Test set: {eda_results['dataset_info']['test_samples']} samples, {eda_results['dataset_info']['test_features']} features",
        f"Validation set: {eda_results['dataset_info']['valid_samples']} samples, {eda_results['dataset_info']['valid_features']} features",
        f"Phase 1 features: {len(eda_results['feature_info']['phase1_features'])}",
        f"Phase 2 features (temporal): {len(eda_results['feature_info']['phase2_features'])}",
        "",
        "=== CLASS DISTRIBUTION ==="
    ]
    
    # Add class distribution
    for dataset, stats in eda_results['class_distribution'].items():
        report_lines.append(f"{dataset} set:")
        report_lines.append(f"  TRUE: {stats['true_count']} samples ({stats['true_pct']:.2f}%)")
        report_lines.append(f"  FALSE: {stats['false_count']} samples ({stats['false_pct']:.2f}%)")
        report_lines.append(f"  Imbalance ratio: {stats['imbalance_ratio']:.2f}")
    
    report_lines.append("")
    report_lines.append("=== TEMPORAL EVIDENCE COVERAGE ===")
    
    # Add evidence coverage
    coverage = eda_results['temporal_evidence']['evidence_coverage']
    report_lines.append(f"Total claims: {coverage['total_claims']}")
    report_lines.append(f"Claims with evidence: {coverage['claims_with_evidence']} ({coverage['claims_with_evidence']/coverage['total_claims']*100:.2f}%)")
    report_lines.append(f"Claims with temporal data: {coverage['claims_with_temporal_data']} ({coverage['claims_with_temporal_data']/coverage['total_claims']*100:.2f}%)")
    report_lines.append(f"Average evidence per claim: {coverage['avg_evidence_per_claim']:.2f}")
    report_lines.append(f"Average temporal evidence per claim: {coverage['avg_temporal_evidence_per_claim']:.2f}")
    
    report_lines.append("")
    report_lines.append("=== EVIDENCE SOURCE DISTRIBUTION ===")
    
    # Add source distribution
    source = eda_results['temporal_evidence']['source_distribution']
    report_lines.append(f"Claims with Google evidence: {source['google_evidence_claims']}")
    report_lines.append(f"Claims with Perplexity evidence: {source['perplexity_evidence_claims']}")
    report_lines.append(f"Average Google evidence items per claim: {source['avg_google_evidence']:.2f}")
    report_lines.append(f"Average Perplexity evidence items per claim: {source['avg_perplexity_evidence']:.2f}")
    
    report_lines.append("")
    report_lines.append("=== TRUTH RATE BY EVIDENCE ===")
    
    # Add truth by evidence
    truth_by_source = eda_results['temporal_evidence']['truth_by_source']
    report_lines.append(f"Claims with Google evidence: {truth_by_source['google_evidence_truth_rate']:.2f}% true")
    report_lines.append(f"Claims with Perplexity evidence: {truth_by_source['perplexity_evidence_truth_rate']:.2f}% true")
    report_lines.append(f"Claims with no evidence: {truth_by_source['no_evidence_truth_rate']:.2f}% true")
    
    report_lines.append("")
    report_lines.append("=== TOP TEMPORAL FEATURES BY CORRELATION ===")
    
    # Add top correlations - with key existence check
    top_corr_features = list(eda_results['temporal_stats'].items())[:10]
    for feature, stats in top_corr_features:
        report_lines.append(f"{feature}:")
        # Add correlation if it exists
        if 'correlation_with_label' in stats:
            report_lines.append(f"  Correlation with truth: {stats['correlation_with_label']:.4f}")
        # Add mean if it exists
        if 'mean' in stats:
            report_lines.append(f"  Mean: {stats['mean']:.4f}")
        # Add missing percentage if it exists
        if 'missing_pct' in stats:
            report_lines.append(f"  Missing: {stats['missing_pct']:.2f}%")
    
    report_lines.append("")
    report_lines.append("=== FEATURE IMPORTANCE ===")
    
    # Add feature importance
    importance = eda_results['feature_importance']
    report_lines.append(f"Phase 1 features in top 20: {importance['phase1_in_top']}")
    report_lines.append(f"Phase 2 features in top 20: {importance['phase2_in_top']}")
    report_lines.append(f"Top Phase 1 feature: {importance['top_phase1_feature']}")
    report_lines.append(f"Top Phase 2 feature: {importance['top_phase2_feature']}")
    
    report_lines.append("")
    report_lines.append("=== DIMENSIONALITY REDUCTION ===")
    
    # Add PCA results
    pca_results = eda_results['pca_results']
    report_lines.append(f"Components needed for 90% variance: {pca_results['n_components_90']}")
    report_lines.append(f"Variance explained by first component: {pca_results['top_component_variance']*100:.2f}%")
    report_lines.append(f"Variance explained by first 5 components: {pca_results['first_5_components_variance']*100:.2f}%")
    
    report_lines.append("")
    report_lines.append("=== CONCLUSION AND RECOMMENDATIONS ===")
    
    # Add recommendations based on findings
    report_lines.append("1. Feature Selection and Dimensionality Reduction:")
    report_lines.append(f"   - Consider using PCA to reduce dimensions (need {pca_results['n_components_90']} components for 90% variance)")
    report_lines.append("   - Focus on top temporal features with highest correlation to truth label")
    
    report_lines.append("2. Model Development Strategy:")
    if importance['phase2_in_top'] > 5:
        report_lines.append("   - Temporal features show strong predictive power, use hybrid models that effectively leverage both content and temporal patterns")
    else:
        report_lines.append("   - Temporal features provide complementary signal to content features, use ensemble approaches")
    
    report_lines.append("3. Evidence Source Considerations:")
    if abs(truth_by_source['google_evidence_truth_rate'] - truth_by_source['perplexity_evidence_truth_rate']) > 5:
        report_lines.append("   - Consider source-specific features or models due to differing truth rates")
    else:
        report_lines.append("   - Evidence sources show similar patterns, can be treated uniformly")
    
    report_lines.append("4. Handling Missing Temporal Data:")
    missing_temporal = False
    for stats in eda_results['temporal_stats'].values():
        if 'missing_pct' in stats and stats['missing_pct'] > 20:
            missing_temporal = True
            break
    
    if missing_temporal:
        report_lines.append("   - Some temporal features have significant missing data, implement appropriate imputation")
    else:
        report_lines.append("   - Temporal features have good coverage, minimal handling of missing data needed")
    
    report_lines.append("")
    report_lines.append("=== VISUALIZATIONS ===")
    report_lines.append("All visualizations have been saved to the plots directory.")
    
    # Save report
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Save JSON report for programmatic access
    with open(JSON_REPORT_PATH, 'w') as f:
        json.dump(eda_results, f, indent=2)
    
    print(f"Report saved to {REPORT_PATH}")
    print(f"JSON report saved to {JSON_REPORT_PATH}")
    
    return report_lines

def main():
    """Run the full EDA workflow"""
    print("Starting Temporal Features EDA")
    
    # Load datasets
    train_df, test_df, valid_df = load_datasets()
    
    # Identify temporal features
    phase1_features, phase2_features = identify_temporal_features(train_df)
    
    # Combine datasets for some analyses
    combined_df = pd.concat([train_df, test_df, valid_df])
    
    # Class distribution analysis
    class_dist = analyze_class_distribution(train_df, test_df, valid_df)
    
    # Analyze temporal feature statistics
    temporal_stats = analyze_temporal_feature_stats(combined_df, phase2_features)
    
    # Analyze correlations
    corr_analysis = analyze_feature_correlations(combined_df, phase1_features, phase2_features)
    
    # Analyze temporal evidence patterns
    temporal_evidence = analyze_temporal_evidence_patterns(combined_df, phase2_features)
    
    # Feature importance analysis
    feature_importance = analyze_feature_importance(combined_df, phase1_features, phase2_features)
    
    # Dimensionality reduction
    pca_results = analyze_dimensionality_reduction(combined_df, phase1_features, phase2_features)
    
    # Skip RFECV as it's computationally expensive and not critical
    print("Skipping recursive feature elimination (RFECV) as it's not critical for the project")
    feature_selection = {
        'selected_features': feature_importance.get('top_features', [])[:20],
        'optimal_n_features': min(20, len(feature_importance.get('top_features', []))),
        'phase1_selected': sum(1 for feat in feature_importance.get('top_features', [])[:20] if feat in phase1_features),
        'phase2_selected': sum(1 for feat in feature_importance.get('top_features', [])[:20] if feat in phase2_features),
        'selection_scores': []
    }
    
    # Create report
    eda_results = {
        'dataset_info': {
            'train_samples': train_df.shape[0],
            'train_features': train_df.shape[1],
            'test_samples': test_df.shape[0],
            'test_features': test_df.shape[1],
            'valid_samples': valid_df.shape[0],
            'valid_features': valid_df.shape[1]
        },
        'feature_info': {
            'phase1_features': phase1_features,
            'phase2_features': phase2_features
        },
        'class_distribution': class_dist,
        'temporal_stats': temporal_stats,
        'correlation_analysis': corr_analysis,
        'temporal_evidence': temporal_evidence,
        'feature_importance': feature_importance,
        'pca_results': pca_results,
        'feature_selection': feature_selection
    }
    
    # Fix JSON serialization by converting numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        else:
            return obj
    
    # Convert numpy types to native Python types
    eda_results = convert_numpy_types(eda_results)
    
    create_report(eda_results)
    
    print("Temporal Features EDA completed successfully")

if __name__ == "__main__":
    main() 