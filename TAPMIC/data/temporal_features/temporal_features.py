#!/usr/bin/env python3
# temporal_features.py
# Extracts temporal features from the political statements dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from matplotlib.ticker import MaxNLocator
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
REPORT_PATH = os.path.join(REPORTS_DIR, "temporal_features_report.txt")

# Define US election years (presidential elections)
US_ELECTION_YEARS = [2000, 2004, 2008, 2012, 2016, 2020]
# Define midterm election years
US_MIDTERM_YEARS = [2002, 2006, 2010, 2014, 2018, 2022]

def load_data():
    """Load the preprocessed datasets"""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    
    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")
    print(f"Validation set: {valid_df.shape[0]} samples")
    
    # Note: In Phase 2, the RAG pipeline will provide real date information
    # This framework is designed to work with actual temporal data
    
    return train_df, test_df, valid_df

def parse_dates(df):
    """Convert date strings to datetime objects and extract date components"""
    # Ensure the date column exists
    if 'date' not in df.columns:
        print("Error: 'date' column not found in the dataset")
        return df
    
    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Extract date components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Get reference date (using current time as a fixed reference)
    reference_date = datetime(2023, 1, 1)
    
    # Calculate days since reference
    df['days_since_reference'] = (reference_date - df['date']).dt.days
    
    # Election year features
    df['is_election_year'] = df['year'].apply(lambda x: 1 if x in US_ELECTION_YEARS else 0)
    df['is_midterm_year'] = df['year'].apply(lambda x: 1 if x in US_MIDTERM_YEARS else 0)
    df['days_to_nearest_election'] = df.apply(
        lambda row: min([abs((datetime(year, 11, 3) - row['date']).days) 
                         for year in US_ELECTION_YEARS + US_MIDTERM_YEARS]), axis=1
    )
    
    # Designate campaign seasons (9 months before election)
    df['in_campaign_season'] = df.apply(
        lambda row: 1 if min([
            abs((datetime(year, 11, 3) - row['date']).days) 
            for year in US_ELECTION_YEARS]) <= 270 else 0, axis=1
    )
    
    return df

def analyze_temporal_patterns(train_df, test_df, valid_df):
    """Analyze temporal patterns in truth values"""
    # Combine datasets for analysis
    combined_df = pd.concat([train_df, test_df, valid_df])
    
    # Ensure date is parsed
    combined_df = parse_dates(combined_df)
    
    # Initialize report content
    report_content = []
    report_content.append(f"# Temporal Feature Analysis Report\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Truth rates over time (yearly)
    report_content.append("## 1. Truth Rates Over Time (Yearly)")
    
    yearly_truth = combined_df.groupby('year')['label'].mean().reset_index()
    yearly_counts = combined_df.groupby('year').size().reset_index(name='count')
    yearly_data = pd.merge(yearly_truth, yearly_counts, on='year')
    yearly_data['truth_percentage'] = yearly_data['label'] * 100
    
    # Create visualization
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot truth rate
    ax1.plot(yearly_data['year'], yearly_data['truth_percentage'], 'b-', linewidth=2, marker='o')
    ax1.set_ylabel('Truth Percentage (%)', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0, 100)
    
    # Plot sample count
    ax2.bar(yearly_data['year'], yearly_data['count'], alpha=0.3, color='gray')
    ax2.set_ylabel('Number of Statements', color='gray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Add vertical lines for election years
    for year in US_ELECTION_YEARS:
        if year >= min(yearly_data['year']) and year <= max(yearly_data['year']):
            plt.axvline(x=year, color='r', linestyle='--', alpha=0.5)
            plt.text(year, 5, 'Election', rotation=90, color='r', alpha=0.7)
    
    # Add labels and title
    plt.title('Truth Percentage by Year', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show each year
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    yearly_plot_path = os.path.join(PLOTS_DIR, "truth_by_year.png")
    plt.tight_layout()
    plt.savefig(yearly_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add to report
    report_content.append("Truth percentage by year with election year indicators:\n")
    report_content.append(f"![Truth by Year]({os.path.relpath(yearly_plot_path, REPORTS_DIR)})\n")
    
    # Add yearly stats to report
    report_content.append("Yearly Truth Percentages:")
    for _, row in yearly_data.sort_values('year').iterrows():
        election_status = ""
        if row['year'] in US_ELECTION_YEARS:
            election_status = " (Presidential Election Year)"
        elif row['year'] in US_MIDTERM_YEARS:
            election_status = " (Midterm Election Year)"
        report_content.append(f"- **{int(row['year'])}**: {row['truth_percentage']:.2f}% true statements from {int(row['count'])} samples{election_status}")
    
    # 2. Monthly patterns
    report_content.append("\n## 2. Truth Rates by Month")
    
    monthly_truth = combined_df.groupby('month')['label'].mean().reset_index()
    monthly_counts = combined_df.groupby('month').size().reset_index(name='count')
    monthly_data = pd.merge(monthly_truth, monthly_counts, on='month')
    monthly_data['truth_percentage'] = monthly_data['label'] * 100
    
    # Sort by month
    monthly_data = monthly_data.sort_values('month')
    
    # Create visualization
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot truth rate
    ax1.plot(monthly_data['month'], monthly_data['truth_percentage'], 'g-', linewidth=2, marker='o')
    ax1.set_ylabel('Truth Percentage (%)', color='g', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylim(0, 100)
    
    # Plot sample count
    ax2.bar(monthly_data['month'], monthly_data['count'], alpha=0.3, color='gray')
    ax2.set_ylabel('Number of Statements', color='gray', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Add labels and title
    plt.title('Truth Percentage by Month', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show each month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(1, 13), month_names)
    
    monthly_plot_path = os.path.join(PLOTS_DIR, "truth_by_month.png")
    plt.tight_layout()
    plt.savefig(monthly_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add to report
    report_content.append("Truth percentage by month:\n")
    report_content.append(f"![Truth by Month]({os.path.relpath(monthly_plot_path, REPORTS_DIR)})\n")
    
    # Add monthly stats to report
    report_content.append("Monthly Truth Percentages:")
    for _, row in monthly_data.iterrows():
        month_name = month_names[int(row['month'])-1]
        report_content.append(f"- **{month_name}**: {row['truth_percentage']:.2f}% true statements from {int(row['count'])} samples")
    
    # 3. Election proximity analysis
    report_content.append("\n## 3. Truth Rates by Proximity to Elections")
    
    # Create bins for days to election
    combined_df['election_proximity_bin'] = pd.cut(
        combined_df['days_to_nearest_election'],
        bins=[0, 30, 90, 180, 365, float('inf')],
        labels=['< 1 month', '1-3 months', '3-6 months', '6-12 months', '> 1 year']
    )
    
    proximity_truth = combined_df.groupby('election_proximity_bin')['label'].mean().reset_index()
    proximity_counts = combined_df.groupby('election_proximity_bin').size().reset_index(name='count')
    proximity_data = pd.merge(proximity_truth, proximity_counts, on='election_proximity_bin')
    proximity_data['truth_percentage'] = proximity_data['label'] * 100
    
    # Create visualization
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot truth rate
    bars = ax1.bar(
        proximity_data['election_proximity_bin'], 
        proximity_data['truth_percentage'], 
        color='purple', 
        alpha=0.7
    )
    ax1.set_ylabel('Truth Percentage (%)', color='purple', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{height:.1f}%',
            ha='center', 
            va='bottom',
            color='purple'
        )
    
    # Plot sample count (line)
    ax2.plot(
        proximity_data['election_proximity_bin'], 
        proximity_data['count'], 
        'ko-', 
        linewidth=2
    )
    ax2.set_ylabel('Number of Statements', color='black', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Add labels and title
    plt.title('Truth Percentage by Proximity to Elections', fontsize=14)
    plt.xlabel('Days to Nearest Election', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    proximity_plot_path = os.path.join(PLOTS_DIR, "truth_by_election_proximity.png")
    plt.tight_layout()
    plt.savefig(proximity_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add to report
    report_content.append("Truth percentage by proximity to elections:\n")
    report_content.append(f"![Truth by Election Proximity]({os.path.relpath(proximity_plot_path, REPORTS_DIR)})\n")
    
    # Add proximity stats to report
    report_content.append("Election Proximity Truth Percentages:")
    for _, row in proximity_data.iterrows():
        report_content.append(f"- **{row['election_proximity_bin']}**: {row['truth_percentage']:.2f}% true statements from {int(row['count'])} samples")
    
    # Save report content
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(report_content))
    
    print(f"Temporal analysis report saved to {REPORT_PATH}")
    return combined_df

def generate_speaker_timeline_features(df):
    """Generate speaker-specific temporal features"""
    # Ensure speaker and date columns exist
    if 'speaker' not in df.columns or 'date' not in df.columns:
        print("Error: Required columns not found in the dataset")
        return df
    
    # Sort by speaker and date
    df = df.sort_values(['speaker', 'date'])
    
    # Initialize speaker timeline features
    df['speaker_statements_before'] = 0
    df['speaker_true_statements_before'] = 0
    df['speaker_false_statements_before'] = 0
    df['speaker_credibility_30days'] = np.nan
    df['speaker_credibility_90days'] = np.nan
    df['speaker_credibility_180days'] = np.nan
    df['truth_trend_30days'] = np.nan
    
    # Group by speaker
    speaker_groups = df.groupby('speaker')
    
    # Process each speaker
    for speaker, group in speaker_groups:
        # Sort by date
        group = group.sort_values('date')
        
        # Loop through each statement
        for i, (idx, row) in enumerate(group.iterrows()):
            # Get statements before current one
            statements_before = group.iloc[:i]
            
            # Count previous statements
            df.at[idx, 'speaker_statements_before'] = len(statements_before)
            df.at[idx, 'speaker_true_statements_before'] = sum(statements_before['label'] == 1)
            df.at[idx, 'speaker_false_statements_before'] = sum(statements_before['label'] == 0)
            
            # Calculate rolling credibility for different time windows
            if len(statements_before) > 0:
                # 30-day window
                window_30days = statements_before[statements_before['date'] >= row['date'] - pd.Timedelta(days=30)]
                if len(window_30days) > 0:
                    df.at[idx, 'speaker_credibility_30days'] = window_30days['label'].mean()
                
                # 90-day window
                window_90days = statements_before[statements_before['date'] >= row['date'] - pd.Timedelta(days=90)]
                if len(window_90days) > 0:
                    df.at[idx, 'speaker_credibility_90days'] = window_90days['label'].mean()
                
                # 180-day window
                window_180days = statements_before[statements_before['date'] >= row['date'] - pd.Timedelta(days=180)]
                if len(window_180days) > 0:
                    df.at[idx, 'speaker_credibility_180days'] = window_180days['label'].mean()
                
                # Calculate trend (change in truth rate over last 30 days)
                if len(window_30days) >= 3:  # Need at least 3 statements to calculate trend
                    window_30days = window_30days.sort_values('date')
                    # Simple trend calculation: difference between later and earlier truth rates
                    half_len = len(window_30days) // 2
                    if half_len > 0:
                        earlier_half = window_30days.iloc[:half_len]
                        later_half = window_30days.iloc[half_len:]
                        
                        earlier_truth_rate = earlier_half['label'].mean()
                        later_truth_rate = later_half['label'].mean()
                        
                        df.at[idx, 'truth_trend_30days'] = later_truth_rate - earlier_truth_rate
    
    # Fill missing values for trend features
    df['speaker_credibility_30days'] = df['speaker_credibility_30days'].fillna(0.5)
    df['speaker_credibility_90days'] = df['speaker_credibility_90days'].fillna(0.5)
    df['speaker_credibility_180days'] = df['speaker_credibility_180days'].fillna(0.5)
    df['truth_trend_30days'] = df['truth_trend_30days'].fillna(0)
    
    return df

def save_enhanced_datasets(train_df, test_df, valid_df):
    """Save datasets with temporal features"""
    # Define output paths
    train_output = os.path.join(OUTPUT_DIR, "intermediate2_train.csv")
    test_output = os.path.join(OUTPUT_DIR, "intermediate2_test.csv")
    valid_output = os.path.join(OUTPUT_DIR, "intermediate2_validation.csv")
    
    # Save to CSV
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    valid_df.to_csv(valid_output, index=False)
    
    print(f"Enhanced datasets saved to {OUTPUT_DIR}")
    print(f"- Train: {train_output}")
    print(f"- Test: {test_output}")
    print(f"- Validation: {valid_output}")

def main():
    """Main function to extract and analyze temporal features"""
    print("Loading datasets...")
    train_df, test_df, valid_df = load_data()
    
    print("NOTE: This framework is designed for use with real temporal data.")
    print("The RAG pipeline in Phase 2 will provide actual dates for temporal feature extraction.")
    print("The current scripts establish the structure for temporal feature generation.")
    
    print("Framework for temporal feature extraction is ready for Phase 2 integration.")
    
    # The functions in this script are designed to be used with real date data:
    # 1. parse_dates() - Will extract temporal components from real dates
    # 2. analyze_temporal_patterns() - Will analyze truth rates over time 
    # 3. generate_speaker_timeline_features() - Will track speaker credibility evolution
    # 4. save_enhanced_datasets() - Will save datasets with real temporal features
    
    # Example flow once RAG pipeline provides dates:
    # train_df = parse_dates(train_df)
    # test_df = parse_dates(test_df)
    # valid_df = parse_dates(valid_df)
    # analyze_temporal_patterns(train_df, test_df, valid_df)
    # train_df = generate_speaker_timeline_features(train_df)
    # test_df = generate_speaker_timeline_features(test_df)
    # valid_df = generate_speaker_timeline_features(valid_df)
    # save_enhanced_datasets(train_df, test_df, valid_df)

if __name__ == "__main__":
    main() 