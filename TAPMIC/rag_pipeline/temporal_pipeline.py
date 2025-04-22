#!/usr/bin/env python3
# temporal_pipeline.py
# Connect collected evidence data with temporal feature extraction framework

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import logging
from tqdm import tqdm

# Add parent directory to path to import temporal_features module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "data/temporal_features"))

# Import temporal features module
try:
    import temporal_features as tf
except ImportError as e:
    print(f"Error importing temporal_features: {e}")
    print("Please ensure you're running this script from the correct directory and that temporal_features.py exists.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, "temporal_pipeline.log")),
        logging.StreamHandler()
    ]
)

# Define paths
EVIDENCE_DIR = os.path.join(SCRIPT_DIR, "evidence_data")
EVIDENCE_FILE = os.path.join(EVIDENCE_DIR, "collected_evidence.json")
INTERMEDIATE_DIR = os.path.join(PROJECT_ROOT, "data/intermediate_1")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/intermediate_2")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data/temporal_features/reports")
PLOTS_DIR = os.path.join(REPORTS_DIR, "plots")

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Define intermediate dataset paths
TRAIN_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_train.csv")
TEST_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_test.csv")
VALID_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_validation.csv")

# Define output dataset paths
OUTPUT_TRAIN_PATH = os.path.join(OUTPUT_DIR, "intermediate2_train.csv")
OUTPUT_TEST_PATH = os.path.join(OUTPUT_DIR, "intermediate2_test.csv")
OUTPUT_VALID_PATH = os.path.join(OUTPUT_DIR, "intermediate2_validation.csv")

# Define report path
REPORT_PATH = os.path.join(REPORTS_DIR, "temporal_pipeline_report.txt")

def load_evidence_data():
    """Load the collected evidence data"""
    logging.info(f"Loading evidence data from {EVIDENCE_FILE}")
    try:
        with open(EVIDENCE_FILE, 'r') as f:
            evidence_data = json.load(f)
        logging.info(f"Successfully loaded {len(evidence_data)} evidence items")
        return evidence_data
    except Exception as e:
        logging.error(f"Error loading evidence data: {e}")
        return []

def load_intermediate_datasets():
    """Load the intermediate datasets"""
    logging.info("Loading intermediate datasets")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        valid_df = pd.read_csv(VALID_PATH)
        
        logging.info(f"Train set: {train_df.shape[0]} samples")
        logging.info(f"Test set: {test_df.shape[0]} samples")
        logging.info(f"Validation set: {valid_df.shape[0]} samples")
        
        return train_df, test_df, valid_df
    except Exception as e:
        logging.error(f"Error loading intermediate datasets: {e}")
        return None, None, None

def extract_evidence_dates(evidence_data):
    """Extract dates from evidence and organize by claim_id"""
    logging.info("Extracting dates from evidence")
    evidence_dates = defaultdict(list)
    
    for item in tqdm(evidence_data):
        claim_id = item.get('claim_id')
        if not claim_id:
            continue
        
        date_info = {
            'publication_date': item.get('publication_date'),
            'all_dates_found': item.get('all_dates_found', []),
            'has_temporal_data': item.get('has_temporal_data', False),
            'source': item.get('source', 'unknown'),
            'title': item.get('title', ''),
            'domain': item.get('domain', '')
        }
        
        evidence_dates[claim_id].append(date_info)
    
    logging.info(f"Extracted dates for {len(evidence_dates)} unique claims")
    return evidence_dates

def add_temporal_features(df, evidence_dates):
    """Add temporal features to dataframe based on evidence dates"""
    logging.info("Adding temporal features from evidence dates")
    
    # Initialize new columns
    df['has_evidence'] = 0
    df['evidence_count'] = 0
    df['has_temporal_data'] = 0
    df['temporal_evidence_count'] = 0
    df['min_publication_date'] = pd.NaT
    df['max_publication_date'] = pd.NaT
    df['mean_publication_year'] = np.nan
    df['publication_date_range_days'] = np.nan
    df['multiple_dates_percentage'] = 0.0
    
    # Initialize election-related features
    df['is_election_year_claim'] = 0
    df['days_to_nearest_election'] = np.nan
    df['in_campaign_season'] = 0
    
    # Add source features
    df['perplexity_evidence_count'] = 0
    df['google_evidence_count'] = 0
    
    # US Election years (presidential and midterm)
    us_election_years = [2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022]
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        claim_id = row['id']
        evidence_items = evidence_dates.get(claim_id, [])
        
        if not evidence_items:
            continue
        
        # Basic evidence counts
        df.at[idx, 'has_evidence'] = 1
        df.at[idx, 'evidence_count'] = len(evidence_items)
        
        # Temporal evidence counts
        temporal_items = [item for item in evidence_items if item['has_temporal_data']]
        df.at[idx, 'temporal_evidence_count'] = len(temporal_items)
        df.at[idx, 'has_temporal_data'] = 1 if temporal_items else 0
        
        # Source counts
        perplexity_count = len([item for item in evidence_items if item['source'] == 'perplexity_generated'])
        google_count = len([item for item in evidence_items if item['source'] == 'google_generated'])
        df.at[idx, 'perplexity_evidence_count'] = perplexity_count
        df.at[idx, 'google_evidence_count'] = google_count
        
        # Publication dates
        publication_dates = []
        for item in evidence_items:
            if item['publication_date']:
                try:
                    pub_date = pd.to_datetime(item['publication_date'])
                    publication_dates.append(pub_date)
                except:
                    pass
        
        if publication_dates:
            df.at[idx, 'min_publication_date'] = min(publication_dates)
            df.at[idx, 'max_publication_date'] = max(publication_dates)
            df.at[idx, 'mean_publication_year'] = np.mean([date.year for date in publication_dates])
            df.at[idx, 'publication_date_range_days'] = (max(publication_dates) - min(publication_dates)).days
        
        # Multiple dates percentage
        multiple_dates_items = [item for item in evidence_items if len(item.get('all_dates_found', [])) > 1]
        if evidence_items:
            df.at[idx, 'multiple_dates_percentage'] = len(multiple_dates_items) / len(evidence_items) * 100
        
        # Election-related features
        claim_date = row.get('date')
        if claim_date and not pd.isna(claim_date):
            try:
                claim_date = pd.to_datetime(claim_date)
                claim_year = claim_date.year
                
                # Is election year
                df.at[idx, 'is_election_year_claim'] = 1 if claim_year in us_election_years else 0
                
                # Days to nearest election (November 3rd of election years)
                days_to_elections = [abs((datetime(year, 11, 3) - claim_date).days) 
                                    for year in us_election_years]
                df.at[idx, 'days_to_nearest_election'] = min(days_to_elections) if days_to_elections else np.nan
                
                # In campaign season (within 270 days of a presidential election)
                presidential_election_years = [2000, 2004, 2008, 2012, 2016, 2020]
                days_to_presidential_elections = [abs((datetime(year, 11, 3) - claim_date).days) 
                                                for year in presidential_election_years]
                df.at[idx, 'in_campaign_season'] = 1 if min(days_to_presidential_elections) <= 270 else 0 if days_to_presidential_elections else 0
            except:
                pass
    
    # Create temporal distance features
    logging.info("Creating temporal distance features")
    df['days_between_claim_and_earliest_evidence'] = np.nan
    df['days_between_claim_and_latest_evidence'] = np.nan
    df['temporal_consistency_score'] = np.nan
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        claim_date = row.get('date')
        min_pub_date = row['min_publication_date']
        max_pub_date = row['max_publication_date']
        
        if claim_date and not pd.isna(claim_date) and not pd.isna(min_pub_date):
            try:
                claim_date = pd.to_datetime(claim_date)
                df.at[idx, 'days_between_claim_and_earliest_evidence'] = abs((claim_date - min_pub_date).days)
                df.at[idx, 'days_between_claim_and_latest_evidence'] = abs((claim_date - max_pub_date).days)
                
                # Temporal consistency score (lower is better)
                # Measures how consistent the evidence dates are with the claim date
                # Formula: 1 / (1 + log(1 + days_between_claim_and_earliest_evidence))
                # This gives values between 0 and 1, with 1 indicating perfect temporal consistency
                if not pd.isna(df.at[idx, 'days_between_claim_and_earliest_evidence']):
                    days_diff = df.at[idx, 'days_between_claim_and_earliest_evidence']
                    df.at[idx, 'temporal_consistency_score'] = 1 / (1 + np.log1p(days_diff))
            except:
                pass
    
    return df

def generate_speaker_timeline_features(df, evidence_dates):
    """Generate speaker timeline features based on evidence dates"""
    logging.info("Generating speaker timeline features")
    
    # Group claims by speaker and sort by date
    speaker_claims = defaultdict(list)
    
    for idx, row in df.iterrows():
        speaker = row.get('speaker', 'unknown')
        claim_date = row.get('date')
        claim_id = row.get('id')
        label = row.get('label')  # binary truth label
        
        if claim_date and not pd.isna(claim_date) and claim_id:
            try:
                claim_date = pd.to_datetime(claim_date)
                speaker_claims[speaker].append({
                    'id': claim_id,
                    'date': claim_date,
                    'label': label,
                    'idx': idx
                })
            except:
                pass
    
    # Sort claims for each speaker by date
    for speaker in speaker_claims:
        speaker_claims[speaker].sort(key=lambda x: x['date'])
    
    # Initialize timeline features
    df['speaker_30day_true_ratio'] = np.nan
    df['speaker_90day_true_ratio'] = np.nan
    df['speaker_180day_true_ratio'] = np.nan
    df['speaker_30day_claim_count'] = 0
    df['speaker_90day_claim_count'] = 0
    df['speaker_180day_claim_count'] = 0
    df['speaker_evidence_consistency'] = np.nan
    
    # Generate timeline features
    for speaker, claims in speaker_claims.items():
        for i, current_claim in enumerate(claims):
            idx = current_claim['idx']
            current_date = current_claim['date']
            
            # Find claims in the previous time windows
            previous_30days = []
            previous_90days = []
            previous_180days = []
            
            for j in range(i):
                prev_claim = claims[j]
                days_diff = (current_date - prev_claim['date']).days
                
                if days_diff <= 30:
                    previous_30days.append(prev_claim)
                if days_diff <= 90:
                    previous_90days.append(prev_claim)
                if days_diff <= 180:
                    previous_180days.append(prev_claim)
            
            # Calculate true ratios for each time window
            if previous_30days:
                df.at[idx, 'speaker_30day_true_ratio'] = np.mean([c['label'] for c in previous_30days])
                df.at[idx, 'speaker_30day_claim_count'] = len(previous_30days)
            
            if previous_90days:
                df.at[idx, 'speaker_90day_true_ratio'] = np.mean([c['label'] for c in previous_90days])
                df.at[idx, 'speaker_90day_claim_count'] = len(previous_90days)
            
            if previous_180days:
                df.at[idx, 'speaker_180day_true_ratio'] = np.mean([c['label'] for c in previous_180days])
                df.at[idx, 'speaker_180day_claim_count'] = len(previous_180days)
            
            # Calculate evidence consistency
            # This measures how consistent the evidence is for a speaker over time
            # by calculating the standard deviation of temporal_consistency_score
            # for the speaker's recent claims
            if previous_90days:
                consistency_scores = []
                for prev_claim in previous_90days:
                    prev_idx = prev_claim['idx']
                    score = df.at[prev_idx, 'temporal_consistency_score']
                    if not pd.isna(score):
                        consistency_scores.append(score)
                
                if consistency_scores:
                    df.at[idx, 'speaker_evidence_consistency'] = 1 - np.std(consistency_scores)
    
    return df

def create_temporal_feature_report(train_df, test_df, valid_df):
    """Create a comprehensive report on the temporal features"""
    logging.info("Creating temporal feature report")
    
    # Combine datasets
    combined_df = pd.concat([train_df, test_df, valid_df])
    
    # Initialize report content
    report_lines = [
        "=== TEMPORAL FEATURE PIPELINE REPORT ===",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== EVIDENCE COVERAGE ===",
        f"Total claims: {combined_df.shape[0]}",
        f"Claims with evidence: {combined_df['has_evidence'].sum()} ({combined_df['has_evidence'].mean()*100:.2f}%)",
        f"Claims with temporal data: {combined_df['has_temporal_data'].sum()} ({combined_df['has_temporal_data'].mean()*100:.2f}%)",
        f"Average evidence items per claim: {combined_df['evidence_count'].mean():.2f}",
        f"Average temporal evidence items per claim: {combined_df['temporal_evidence_count'].mean():.2f}",
        "",
        "=== TEMPORAL FEATURE STATISTICS ==="
    ]
    
    # Calculate statistics for numeric features
    temporal_features = [
        'evidence_count', 'temporal_evidence_count', 'multiple_dates_percentage',
        'mean_publication_year', 'publication_date_range_days',
        'days_between_claim_and_earliest_evidence', 'days_between_claim_and_latest_evidence',
        'temporal_consistency_score', 'speaker_30day_true_ratio', 'speaker_90day_true_ratio',
        'speaker_180day_true_ratio', 'speaker_30day_claim_count', 'speaker_90day_claim_count',
        'speaker_180day_claim_count', 'speaker_evidence_consistency'
    ]
    
    for feature in temporal_features:
        if feature in combined_df.columns:
            stats = combined_df[feature].describe()
            report_lines.append(f"Feature: {feature}")
            report_lines.append(f"  Count: {stats['count']:.0f}")
            report_lines.append(f"  Mean: {stats['mean']:.4f}")
            report_lines.append(f"  Std: {stats['std']:.4f}")
            report_lines.append(f"  Min: {stats['min']:.4f}")
            report_lines.append(f"  25%: {stats['25%']:.4f}")
            report_lines.append(f"  50%: {stats['50%']:.4f}")
            report_lines.append(f"  75%: {stats['75%']:.4f}")
            report_lines.append(f"  Max: {stats['max']:.4f}")
            report_lines.append("")
    
    # Truth rate by temporal features
    report_lines.append("=== TRUTH RATE BY TEMPORAL FEATURES ===")
    
    # Truth rate by evidence count bins
    combined_df['evidence_count_bin'] = pd.cut(
        combined_df['evidence_count'],
        bins=[0, 1, 2, 3, 5, float('inf')],
        labels=['0', '1', '2', '3-5', '5+']
    )
    
    evidence_count_truth = combined_df.groupby('evidence_count_bin')['label'].mean().reset_index()
    evidence_count_truth['truth_percentage'] = evidence_count_truth['label'] * 100
    
    report_lines.append("Truth percentage by evidence count:")
    for _, row in evidence_count_truth.iterrows():
        report_lines.append(f"  {row['evidence_count_bin']}: {row['truth_percentage']:.2f}%")
    report_lines.append("")
    
    # Truth rate by temporal consistency score bins
    combined_df['consistency_bin'] = pd.cut(
        combined_df['temporal_consistency_score'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    consistency_truth = combined_df.groupby('consistency_bin')['label'].mean().reset_index()
    consistency_truth['truth_percentage'] = consistency_truth['label'] * 100
    
    report_lines.append("Truth percentage by temporal consistency score:")
    for _, row in consistency_truth.iterrows():
        report_lines.append(f"  {row['consistency_bin']}: {row['truth_percentage']:.2f}%")
    report_lines.append("")
    
    # Truth rate by election proximity
    combined_df['election_proximity_bin'] = pd.cut(
        combined_df['days_to_nearest_election'],
        bins=[0, 30, 90, 180, 365, float('inf')],
        labels=['< 1 month', '1-3 months', '3-6 months', '6-12 months', '> 1 year']
    )
    
    proximity_truth = combined_df.groupby('election_proximity_bin')['label'].mean().reset_index()
    proximity_truth['truth_percentage'] = proximity_truth['label'] * 100
    
    report_lines.append("Truth percentage by election proximity:")
    for _, row in proximity_truth.iterrows():
        report_lines.append(f"  {row['election_proximity_bin']}: {row['truth_percentage']:.2f}%")
    report_lines.append("")
    
    # Create visualizations
    report_lines.append("=== VISUALIZATIONS ===")
    report_lines.append("Visualizations have been saved to the reports/plots directory.")
    
    # Create a directory for visualizations
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Correlation heatmap of temporal features with label
    plt.figure(figsize=(14, 10))
    temporal_features.append('label')
    corr_df = combined_df[temporal_features].corr()
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation of Temporal Features with Truth Label', fontsize=14)
    plt.tight_layout()
    corr_plot_path = os.path.join(PLOTS_DIR, "temporal_features_correlation.png")
    plt.savefig(corr_plot_path)
    plt.close()
    
    # Truth rate by evidence count
    plt.figure(figsize=(12, 6))
    sns.barplot(x='evidence_count_bin', y='truth_percentage', data=evidence_count_truth)
    plt.title('Truth Percentage by Evidence Count', fontsize=14)
    plt.xlabel('Number of Evidence Items', fontsize=12)
    plt.ylabel('Truth Percentage (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    evidence_plot_path = os.path.join(PLOTS_DIR, "truth_by_evidence_count.png")
    plt.savefig(evidence_plot_path)
    plt.close()
    
    # Truth rate by temporal consistency
    plt.figure(figsize=(12, 6))
    sns.barplot(x='consistency_bin', y='truth_percentage', data=consistency_truth)
    plt.title('Truth Percentage by Temporal Consistency', fontsize=14)
    plt.xlabel('Temporal Consistency', fontsize=12)
    plt.ylabel('Truth Percentage (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    consistency_plot_path = os.path.join(PLOTS_DIR, "truth_by_consistency.png")
    plt.savefig(consistency_plot_path)
    plt.close()
    
    # Truth rate by election proximity
    plt.figure(figsize=(12, 6))
    sns.barplot(x='election_proximity_bin', y='truth_percentage', data=proximity_truth)
    plt.title('Truth Percentage by Election Proximity', fontsize=14)
    plt.xlabel('Proximity to Election', fontsize=12)
    plt.ylabel('Truth Percentage (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    proximity_plot_path = os.path.join(PLOTS_DIR, "truth_by_election_proximity.png")
    plt.savefig(proximity_plot_path)
    plt.close()
    
    # Feature importance based on correlation with label
    label_corr = corr_df['label'].drop('label').abs().sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=label_corr.values, y=label_corr.index)
    plt.title('Temporal Feature Importance (Correlation with Truth Label)', fontsize=14)
    plt.xlabel('Absolute Correlation', fontsize=12)
    plt.tight_layout()
    importance_plot_path = os.path.join(PLOTS_DIR, "temporal_feature_importance.png")
    plt.savefig(importance_plot_path)
    plt.close()
    
    # Save report
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Report saved to {REPORT_PATH}")
    return report_lines

def main():
    """Main function that runs the temporal pipeline"""
    logging.info("Starting temporal pipeline")
    
    # Load evidence data
    evidence_data = load_evidence_data()
    if not evidence_data:
        logging.error("Failed to load evidence data")
        return
    
    # Load intermediate datasets
    train_df, test_df, valid_df = load_intermediate_datasets()
    if train_df is None or test_df is None or valid_df is None:
        logging.error("Failed to load intermediate datasets")
        return
    
    # Extract dates from evidence
    evidence_dates = extract_evidence_dates(evidence_data)
    
    # Add temporal features
    logging.info("Adding temporal features to train set")
    train_df = add_temporal_features(train_df, evidence_dates)
    
    logging.info("Adding temporal features to test set")
    test_df = add_temporal_features(test_df, evidence_dates)
    
    logging.info("Adding temporal features to validation set")
    valid_df = add_temporal_features(valid_df, evidence_dates)
    
    # Generate speaker timeline features
    logging.info("Generating speaker timeline features for train set")
    train_df = generate_speaker_timeline_features(train_df, evidence_dates)
    
    logging.info("Generating speaker timeline features for test set")
    test_df = generate_speaker_timeline_features(test_df, evidence_dates)
    
    logging.info("Generating speaker timeline features for validation set")
    valid_df = generate_speaker_timeline_features(valid_df, evidence_dates)
    
    # Create report
    logging.info("Creating temporal feature report")
    create_temporal_feature_report(train_df, test_df, valid_df)
    
    # Save enhanced datasets
    logging.info(f"Saving enhanced train dataset to {OUTPUT_TRAIN_PATH}")
    train_df.to_csv(OUTPUT_TRAIN_PATH, index=False)
    
    logging.info(f"Saving enhanced test dataset to {OUTPUT_TEST_PATH}")
    test_df.to_csv(OUTPUT_TEST_PATH, index=False)
    
    logging.info(f"Saving enhanced validation dataset to {OUTPUT_VALID_PATH}")
    valid_df.to_csv(OUTPUT_VALID_PATH, index=False)
    
    logging.info("Temporal pipeline completed successfully")
    logging.info(f"Enhanced datasets saved to {OUTPUT_DIR}")
    logging.info(f"Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main() 