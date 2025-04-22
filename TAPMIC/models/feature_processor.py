#!/usr/bin/env python3
"""
Streamlined feature processor for TAPMIC RoBERTa model.
Focuses on the essential components while preserving temporal logic data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "intermediate_2")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "final")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define essential features to keep
TEXT_FIELD = 'processed_text'

# Core metadata fields needed for model evaluation and analysis
METADATA_FIELDS = ['id', 'label', 'speaker', 'subject', 'context']

# Essential text features from Phase 1
TEXT_FEATURES = [
    'statement_length', 'word_count', 'avg_word_length',
    'contains_number', 'contains_president', 'contains_government'
]

# Essential credibility features from Phase 1
CREDIBILITY_FEATURES = [
    'credibility_score', 'weighted_credibility', 
    'speaker_high_credibility', 'speaker_low_credibility'
]

# Essential categorical features from Phase 1 (one-hot encoded)
CATEGORICAL_FEATURES = [
    'party_republican', 'party_democrat', 'party_none',
    'context_formal_context', 'context_informal_context'
]

# Essential temporal features from Phase 2
TEMPORAL_FEATURES = [
    # Evidence counts and temporal coverage
    'evidence_count', 'temporal_evidence_count',
    
    # Publication date information
    'mean_publication_year', 'publication_date_range_days', 'multiple_dates_percentage',
    
    # Election and campaign features
    'is_election_year_claim', 'days_to_nearest_election', 'in_campaign_season',
    
    # Evidence source features
    'perplexity_evidence_count', 'google_evidence_count',
    
    # Temporal relationship features
    'days_between_claim_and_earliest_evidence', 'days_between_claim_and_latest_evidence',
    'temporal_consistency_score',
    
    # Speaker timeline features
    'speaker_30day_true_ratio', 'speaker_90day_true_ratio', 'speaker_180day_true_ratio',
    'speaker_30day_claim_count', 'speaker_90day_claim_count', 'speaker_180day_claim_count',
    'speaker_evidence_consistency'
]

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the intermediate2 datasets
    
    Returns:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        valid_df: Validation DataFrame
    """
    logger.info(f"Loading datasets from {DATA_DIR}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "intermediate2_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "intermediate2_test.csv"))
    valid_df = pd.read_csv(os.path.join(DATA_DIR, "intermediate2_validation.csv"))
    
    logger.info(f"Train set: {train_df.shape[0]} samples, {train_df.shape[1]} features")
    logger.info(f"Test set: {test_df.shape[0]} samples, {test_df.shape[1]} features")
    logger.info(f"Validation set: {valid_df.shape[0]} samples, {valid_df.shape[1]} features")
    
    return train_df, test_df, valid_df

def select_features(train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Select essential features for the RoBERTa model
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        valid_df: Validation DataFrame
        
    Returns:
        train_df: Filtered training DataFrame
        test_df: Filtered testing DataFrame
        valid_df: Filtered validation DataFrame
    """
    logger.info("Selecting essential features")
    
    # Check which features are actually available in the dataset
    all_features = set(METADATA_FIELDS + [TEXT_FIELD] + 
                      TEXT_FEATURES + CREDIBILITY_FEATURES + 
                      CATEGORICAL_FEATURES + TEMPORAL_FEATURES)
    available_features = [col for col in all_features if col in train_df.columns]
    
    # Select features
    train_df = train_df[available_features].copy()
    test_df = test_df[available_features].copy()
    valid_df = valid_df[available_features].copy()
    
    logger.info(f"Selected {len(available_features)} essential features")
    
    return train_df, test_df, valid_df

def handle_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Handle missing values with simple methods
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        valid_df: Validation DataFrame
        
    Returns:
        train_df: DataFrame with handled missing values
        test_df: DataFrame with handled missing values
        valid_df: DataFrame with handled missing values
    """
    logger.info("Handling missing values")
    
    # Create missing indicators for important temporal features
    key_temporal_features = [
        'mean_publication_year', 'temporal_consistency_score', 
        'speaker_30day_true_ratio', 'speaker_180day_true_ratio'
    ]
    
    for feat in key_temporal_features:
        if feat in train_df.columns:
            if train_df[feat].isna().any():
                missing_col = f"{feat}_missing"
                train_df[missing_col] = train_df[feat].isna().astype(int)
                test_df[missing_col] = test_df[feat].isna().astype(int)
                valid_df[missing_col] = valid_df[feat].isna().astype(int)
    
    # Fill missing values for different types of columns
    for col in train_df.columns:
        if col == 'label' or col == TEXT_FIELD or col in METADATA_FIELDS:
            # Skip these fields, should not have missing values
            continue
            
        if train_df[col].isna().any():
            if pd.api.types.is_numeric_dtype(train_df[col]):
                # For numeric features, fill with median from training set
                median_value = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_value)
                test_df[col] = test_df[col].fillna(median_value)
                valid_df[col] = valid_df[col].fillna(median_value)
            else:
                # For categorical features, fill with most common value
                most_common = train_df[col].mode()[0]
                train_df[col] = train_df[col].fillna(most_common)
                test_df[col] = test_df[col].fillna(most_common)
                valid_df[col] = valid_df[col].fillna(most_common)
    
    missing_counts = train_df.isna().sum()
    if missing_counts.sum() > 0:
        logger.warning(f"There are still missing values after imputation: {missing_counts[missing_counts > 0]}")
    
    return train_df, test_df, valid_df

def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Normalize numerical features while preserving temporal logic
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        valid_df: Validation DataFrame
        
    Returns:
        train_df: DataFrame with normalized features
        test_df: DataFrame with normalized features
        valid_df: DataFrame with normalized features
        feature_info: Dictionary with scaling information (for model interpretation)
    """
    logger.info("Normalizing numerical features")
    
    # Get all numeric features excluding metadata and label
    numeric_features = [col for col in train_df.columns 
                       if pd.api.types.is_numeric_dtype(train_df[col]) 
                       and col != 'label' 
                       and col not in METADATA_FIELDS
                       and not col.endswith('_missing')]
    
    # Store feature info for later interpretation
    feature_info = {
        'numeric_features': numeric_features,
        'scaling_stats': {}
    }
    
    # Store original statistics for interpretation
    for feat in numeric_features:
        feature_info['scaling_stats'][feat] = {
            'mean': float(train_df[feat].mean()),
            'std': float(train_df[feat].std()),
            'min': float(train_df[feat].min()),
            'max': float(train_df[feat].max())
        }
    
    # Apply standardization
    scaler = StandardScaler()
    train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
    test_df[numeric_features] = scaler.transform(test_df[numeric_features])
    valid_df[numeric_features] = scaler.transform(valid_df[numeric_features])
    
    logger.info(f"Normalized {len(numeric_features)} numerical features")
    
    return train_df, test_df, valid_df, feature_info

def create_feature_report(train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame, feature_info: Dict) -> Dict:
    """
    Generate report on features
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        valid_df: Validation DataFrame
        feature_info: Dictionary with feature information
        
    Returns:
        report: Dictionary with feature statistics
    """
    logger.info("Generating feature report")
    
    # Count features by category
    text_feature_count = sum(1 for feat in TEXT_FEATURES if feat in train_df.columns)
    credibility_feature_count = sum(1 for feat in CREDIBILITY_FEATURES if feat in train_df.columns)
    categorical_feature_count = sum(1 for feat in CATEGORICAL_FEATURES if feat in train_df.columns)
    temporal_feature_count = sum(1 for feat in TEMPORAL_FEATURES if feat in train_df.columns)
    missing_indicator_count = sum(1 for col in train_df.columns if col.endswith('_missing'))
    
    # Initialize report
    report = {
        "dataset_info": {
            "train_samples": train_df.shape[0],
            "test_samples": test_df.shape[0],
            "validation_samples": valid_df.shape[0],
            "total_features": train_df.shape[1] - 1  # Exclude label
        },
        "feature_counts": {
            "metadata_fields": len([col for col in METADATA_FIELDS if col in train_df.columns]),
            "text_features": text_feature_count,
            "credibility_features": credibility_feature_count,
            "categorical_features": categorical_feature_count,
            "temporal_features": temporal_feature_count,
            "missing_indicators": missing_indicator_count,
            "total_features": train_df.shape[1] - 1  # Exclude label
        },
        "feature_types": {
            "numeric_features": len(feature_info['numeric_features']),
            "categorical_features": sum(1 for col in train_df.columns if not pd.api.types.is_numeric_dtype(train_df[col]))
        },
        "temporal_feature_coverage": {
            feature: {
                "non_null": int(train_df[feature].notna().sum()),
                "coverage_pct": float(train_df[feature].notna().mean() * 100),
                "min": float(feature_info['scaling_stats'][feature]['min']) if feature in feature_info['scaling_stats'] else None,
                "max": float(feature_info['scaling_stats'][feature]['max']) if feature in feature_info['scaling_stats'] else None,
                "mean": float(feature_info['scaling_stats'][feature]['mean']) if feature in feature_info['scaling_stats'] else None,
                "std": float(feature_info['scaling_stats'][feature]['std']) if feature in feature_info['scaling_stats'] else None
            }
            for feature in TEMPORAL_FEATURES if feature in train_df.columns
        },
        "class_distribution": {
            "train": {
                "true_count": int(train_df['label'].sum()),
                "false_count": int(train_df.shape[0] - train_df['label'].sum()),
                "true_pct": float(train_df['label'].mean() * 100),
                "false_pct": float((1 - train_df['label'].mean()) * 100)
            }
        }
    }
    
    return report

def save_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame, report: Dict, feature_info: Dict):
    """
    Save the processed datasets and reports
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        valid_df: Validation DataFrame
        report: Feature report dictionary
        feature_info: Feature information dictionary
    """
    logger.info("Saving processed datasets and reports")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save datasets
    train_df.to_csv(os.path.join(OUTPUT_DIR, "final_train.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "final_test.csv"), index=False)
    valid_df.to_csv(os.path.join(OUTPUT_DIR, "final_validation.csv"), index=False)
    
    # Save feature info (for model interpretation)
    with open(os.path.join(OUTPUT_DIR, "feature_info.json"), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Save feature report
    with open(os.path.join(OUTPUT_DIR, "feature_report.json"), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create human-readable report
    with open(os.path.join(OUTPUT_DIR, "feature_report.txt"), 'w') as f:
        f.write("=== FEATURE PROCESSING REPORT ===\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== DATASET STATISTICS ===\n")
        f.write(f"Train set: {train_df.shape[0]} samples, {train_df.shape[1]} features\n")
        f.write(f"Test set: {test_df.shape[0]} samples, {test_df.shape[1]} features\n")
        f.write(f"Validation set: {valid_df.shape[0]} samples, {valid_df.shape[1]} features\n\n")
        
        f.write("=== FEATURE COUNTS ===\n")
        for key, value in report["feature_counts"].items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("=== TEMPORAL FEATURES SUMMARY ===\n")
        for feature in TEMPORAL_FEATURES:
            if feature in train_df.columns:
                stats = report["temporal_feature_coverage"][feature]
                f.write(f"{feature}: {stats['coverage_pct']:.2f}% coverage\n")
        
        f.write("\n=== CLASS DISTRIBUTION ===\n")
        f.write(f"TRUE samples: {report['class_distribution']['train']['true_count']} ({report['class_distribution']['train']['true_pct']:.2f}%)\n")
        f.write(f"FALSE samples: {report['class_distribution']['train']['false_count']} ({report['class_distribution']['train']['false_pct']:.2f}%)\n")
    
    logger.info(f"Datasets and reports saved to {OUTPUT_DIR}")

def main():
    """Main function to process features for the RoBERTa model"""
    logger.info("Starting streamlined feature processing")
    
    # Load datasets
    train_df, test_df, valid_df = load_datasets()
    
    # Select essential features
    train_df, test_df, valid_df = select_features(train_df, test_df, valid_df)
    
    # Handle missing values
    train_df, test_df, valid_df = handle_missing_values(train_df, test_df, valid_df)
    
    # Normalize features
    train_df, test_df, valid_df, feature_info = normalize_features(train_df, test_df, valid_df)
    
    # Generate feature report
    report = create_feature_report(train_df, test_df, valid_df, feature_info)
    
    # Save datasets and reports
    save_datasets(train_df, test_df, valid_df, report, feature_info)
    
    logger.info("Feature processing completed successfully")
    
    return 0

if __name__ == "__main__":
    main() 