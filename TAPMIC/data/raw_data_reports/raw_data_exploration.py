import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class LiarDataExplorer:
    """
    A class to explore and analyze the LIAR dataset, generate visualizations,
    and create a comprehensive EDA report.
    """
    
    def __init__(self, data_dir='data/raw', output_dir='data/raw_data_reports'):
        """
        Initialize the data explorer with paths and load the datasets.
        
        Args:
            data_dir: Directory containing raw data files
            output_dir: Directory to save reports and visualizations
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load datasets
        self.train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        self.test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        self.valid_df = pd.read_csv(os.path.join(data_dir, 'valid.csv'))
        
        # Set style for plots
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Report data
        self.report_data = {
            "dataset_info": {},
            "label_distribution": {},
            "missing_values": {},
            "statement_statistics": {},
            "speaker_statistics": {},
            "temporal_insights": {},
            "preprocessing_recommendations": []
        }
    
    def get_basic_info(self):
        """Get basic information about the datasets"""
        sizes = {
            'train': len(self.train_df),
            'test': len(self.test_df),
            'valid': len(self.valid_df),
            'total': len(self.train_df) + len(self.test_df) + len(self.valid_df)
        }
        
        columns = list(self.train_df.columns)
        
        self.report_data["dataset_info"] = {
            "sizes": sizes,
            "columns": columns,
            "train_sample": self.train_df.head(2).to_dict('records'),
            "data_types": {col: str(self.train_df[col].dtype) for col in columns}
        }
        
        print(f"Dataset sizes: Train={sizes['train']}, Test={sizes['test']}, Validation={sizes['valid']}")
        print(f"Total samples: {sizes['total']}")
        print(f"Columns: {', '.join(columns)}")
    
    def analyze_labels(self):
        """Analyze the distribution of truthfulness labels"""
        # Count labels in each dataset
        train_labels = self.train_df['label'].value_counts().to_dict()
        test_labels = self.test_df['label'].value_counts().to_dict()
        valid_labels = self.valid_df['label'].value_counts().to_dict()
        
        # Combine all datasets for overall distribution
        all_df = pd.concat([self.train_df, self.test_df, self.valid_df])
        all_labels = all_df['label'].value_counts().to_dict()
        
        self.report_data["label_distribution"] = {
            "train": train_labels,
            "test": test_labels,
            "valid": valid_labels,
            "all": all_labels
        }
        
        # Visualize label distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        sns.countplot(y=self.train_df['label'], order=self.train_df['label'].value_counts().index)
        plt.title('Label Distribution in Training Set')
        plt.xlabel('Count')
        plt.ylabel('Label')
        
        plt.subplot(2, 2, 2)
        sns.countplot(y=all_df['label'], order=all_df['label'].value_counts().index)
        plt.title('Label Distribution in All Data')
        plt.xlabel('Count')
        plt.ylabel('Label')
        
        # Prepare for binary classification visualization
        all_df['binary_label'] = all_df['label'].apply(
            lambda x: 'TRUE' if x in ['true', 'mostly-true', 'half-true'] 
            else 'FALSE'
        )
        binary_counts = all_df['binary_label'].value_counts()
        
        plt.subplot(2, 2, 3)
        plt.pie(binary_counts, labels=binary_counts.index, autopct='%1.1f%%', startangle=90, 
                colors=['#ff9999','#66b3ff'])
        plt.title('Binary Label Distribution (TRUE vs FALSE)')
        
        plt.subplot(2, 2, 4)
        sns.countplot(x='binary_label', data=all_df)
        plt.title('Binary Label Counts')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'label_distribution.png'))
        plt.close()
        
        # Calculate binary distribution statistics
        binary_distribution = all_df['binary_label'].value_counts().to_dict()
        class_imbalance = max(binary_distribution.values()) / min(binary_distribution.values())
        
        self.report_data["label_distribution"]["binary"] = binary_distribution
        self.report_data["label_distribution"]["class_imbalance"] = class_imbalance
        
        print(f"Original label distribution: {all_labels}")
        print(f"Binary label distribution: {binary_distribution}")
        print(f"Class imbalance ratio: {class_imbalance:.2f}")
    
    def check_missing_values(self):
        """Check for missing values in the datasets"""
        # Check for nulls in each dataset
        train_nulls = self.train_df.isnull().sum().to_dict()
        test_nulls = self.test_df.isnull().sum().to_dict()
        valid_nulls = self.valid_df.isnull().sum().to_dict()
        
        # Check for empty strings in text fields
        text_columns = ['statement', 'subject', 'speaker', 'context']
        train_empty = {col: (self.train_df[col] == '').sum() for col in text_columns if col in self.train_df.columns}
        test_empty = {col: (self.test_df[col] == '').sum() for col in text_columns if col in self.test_df.columns}
        valid_empty = {col: (self.valid_df[col] == '').sum() for col in text_columns if col in self.valid_df.columns}
        
        self.report_data["missing_values"] = {
            "null_values": {
                "train": train_nulls,
                "test": test_nulls,
                "valid": valid_nulls
            },
            "empty_strings": {
                "train": train_empty,
                "test": test_empty,
                "valid": valid_empty
            }
        }
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        
        missing_data = pd.DataFrame({
            'Train null': pd.Series(train_nulls),
            'Test null': pd.Series(test_nulls),
            'Valid null': pd.Series(valid_nulls),
            'Train empty': pd.Series(train_empty),
            'Test empty': pd.Series(test_empty),
            'Valid empty': pd.Series(valid_empty)
        })
        
        sns.heatmap(missing_data.transpose(), cmap='viridis', annot=True, fmt='g')
        plt.title('Missing Values and Empty Strings')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'missing_values.png'))
        plt.close()
        
        print("Missing value analysis completed")
    
    def analyze_statements(self):
        """Analyze statements (length, complexity, etc.)"""
        # Combine datasets for analysis
        all_df = pd.concat([self.train_df, self.test_df, self.valid_df])
        
        # Calculate statement lengths
        all_df['statement_length'] = all_df['statement'].str.len()
        all_df['statement_word_count'] = all_df['statement'].str.split().str.len()
        
        # Analyze by truthfulness
        statement_stats_by_label = all_df.groupby('label')[
            ['statement_length', 'statement_word_count']
        ].agg(['mean', 'median', 'min', 'max']).reset_index()
        
        # Add binary label for analysis
        all_df['binary_label'] = all_df['label'].apply(
            lambda x: 'TRUE' if x in ['true', 'mostly-true', 'half-true'] 
            else 'FALSE'
        )
        
        statement_stats_by_binary = all_df.groupby('binary_label')[
            ['statement_length', 'statement_word_count']
        ].agg(['mean', 'median', 'min', 'max']).reset_index()
        
        # Convert DataFrame to dict and then convert all numpy types to Python types
        statement_stats_by_label_dict = {}
        for col in statement_stats_by_label.columns:
            if isinstance(col, tuple):
                col_name = f"{col[0]}_{col[1]}"
            else:
                col_name = col
            statement_stats_by_label_dict[col_name] = statement_stats_by_label[col].tolist()
            
        statement_stats_by_binary_dict = {}
        for col in statement_stats_by_binary.columns:
            if isinstance(col, tuple):
                col_name = f"{col[0]}_{col[1]}"
            else:
                col_name = col
            statement_stats_by_binary_dict[col_name] = statement_stats_by_binary[col].tolist()
        
        self.report_data["statement_statistics"] = {
            "overall": {
                "mean_length": float(all_df['statement_length'].mean()),
                "median_length": float(all_df['statement_length'].median()),
                "mean_word_count": float(all_df['statement_word_count'].mean()),
                "median_word_count": float(all_df['statement_word_count'].median()),
            },
            "by_label": statement_stats_by_label_dict,
            "by_binary_label": statement_stats_by_binary_dict
        }
        
        # Visualize statement lengths by truthfulness
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x='label', y='statement_word_count', data=all_df, order=all_df['label'].value_counts().index)
        plt.title('Statement Word Count by Label')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='binary_label', y='statement_word_count', data=all_df)
        plt.title('Statement Word Count by Binary Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'statement_analysis.png'))
        plt.close()
        
        print("Statement analysis completed")
    
    def analyze_speakers(self):
        """Analyze speakers and their truthfulness patterns"""
        # Combine datasets for analysis
        all_df = pd.concat([self.train_df, self.test_df, self.valid_df])
        
        # Top speakers by count
        top_speakers = all_df['speaker'].value_counts().head(20)
        
        # Speaker truthfulness analysis
        speaker_truthfulness = all_df.groupby('speaker')['label'].value_counts().unstack().fillna(0)
        speaker_truthfulness['total'] = speaker_truthfulness.sum(axis=1)
        
        # Filter for speakers with at least 10 statements
        active_speakers = speaker_truthfulness[speaker_truthfulness['total'] >= 10]
        
        # Calculate truth ratio for each speaker
        active_speakers['truth_ratio'] = (
            active_speakers.get('true', 0) + 
            active_speakers.get('mostly-true', 0) + 
            active_speakers.get('half-true', 0)
        ) / active_speakers['total']
        
        # Select top 15 active speakers by statement count
        top_active_speakers = active_speakers.sort_values('total', ascending=False).head(15)
        
        self.report_data["speaker_statistics"] = {
            "total_unique_speakers": int(all_df['speaker'].nunique()),
            "top_speakers": {k: int(v) for k, v in top_speakers.to_dict().items()},
            "top_active_speakers": {k: int(v) for k, v in top_active_speakers['total'].to_dict().items()},
            "speaker_truth_ratios": {k: float(v) for k, v in top_active_speakers['truth_ratio'].to_dict().items()}
        }
        
        # Visualize speaker statistics
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 1, 1)
        top_speakers.plot(kind='bar')
        plt.title('Top 20 Speakers by Number of Statements')
        plt.ylabel('Number of Statements')
        plt.xlabel('Speaker')
        plt.xticks(rotation=45, ha='right')
        
        plt.subplot(2, 1, 2)
        top_active_speakers['truth_ratio'].sort_values().plot(kind='barh')
        plt.title('Truth Ratio for Top 15 Active Speakers')
        plt.xlabel('Truth Ratio (true + mostly-true + half-true) / total')
        plt.ylabel('Speaker')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'speaker_analysis.png'))
        plt.close()
        
        print("Speaker analysis completed")
    
    def analyze_temporal_patterns(self):
        """Analyze any temporal patterns in the data"""
        # This function is limited since we don't have explicit date information
        # but we can analyze patterns in the party, state, etc. over time
        
        # Combine datasets for analysis
        all_df = pd.concat([self.train_df, self.test_df, self.valid_df])
        
        # Party distribution
        party_distribution = all_df['party'].value_counts()
        
        # Party and truthfulness
        party_truthfulness = all_df.groupby('party')['label'].value_counts().unstack().fillna(0)
        party_truthfulness['total'] = party_truthfulness.sum(axis=1)
        
        for party in party_truthfulness.index:
            if party != 'none':
                party_truthfulness.loc[party, 'truth_ratio'] = (
                    party_truthfulness.loc[party].get('true', 0) + 
                    party_truthfulness.loc[party].get('mostly-true', 0) + 
                    party_truthfulness.loc[party].get('half-true', 0)
                ) / party_truthfulness.loc[party, 'total']
        
        # Temporal insights from available context data
        # (While we don't have explicit dates in the dataset, we can check context for temporal patterns)
        context_terms = ['recent', 'current', 'past', 'future', 'year', 'month', 'week', 'day']
        temporal_context_counts = {term: int(all_df['context'].str.contains(term, case=False).sum()) 
                                   for term in context_terms}
        
        # Convert party distribution to serializable format
        party_dist_dict = {k: int(v) for k, v in party_distribution.to_dict().items()}
        
        # Convert party truthfulness to serializable format
        party_truth_dict = {}
        for party, row in party_truthfulness.iterrows():
            party_truth_dict[party] = {col: float(val) if isinstance(val, (np.integer, np.floating)) else val 
                                      for col, val in row.items()}
        
        self.report_data["temporal_insights"] = {
            "party_distribution": party_dist_dict,
            "party_truthfulness": party_truth_dict,
            "temporal_context_counts": temporal_context_counts
        }
        
        # Visualize party distribution and truthfulness
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        party_distribution.plot(kind='bar')
        plt.title('Distribution by Political Party')
        plt.ylabel('Count')
        plt.xlabel('Party')
        
        plt.subplot(2, 2, 2)
        if 'truth_ratio' in party_truthfulness.columns:
            filtered_parties = party_truthfulness[party_truthfulness.index != 'none']
            if not filtered_parties.empty:
                filtered_parties['truth_ratio'].sort_values().plot(kind='barh')
                plt.title('Truth Ratio by Political Party')
                plt.xlabel('Truth Ratio')
                plt.ylabel('Party')
        
        plt.subplot(2, 2, 3)
        pd.Series(temporal_context_counts).sort_values().plot(kind='barh')
        plt.title('Temporal Terms in Statement Context')
        plt.xlabel('Count')
        plt.ylabel('Term')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_analysis.png'))
        plt.close()
        
        print("Temporal pattern analysis completed")
    
    def generate_preprocessing_recommendations(self):
        """Generate recommendations for preprocessing based on the analysis"""
        recommendations = []
        
        # Class imbalance recommendation
        class_imbalance = self.report_data["label_distribution"].get("class_imbalance", 0)
        if class_imbalance > 1.5:
            recommendations.append({
                "issue": "Class Imbalance",
                "description": f"The class imbalance ratio is {class_imbalance:.2f}",
                "recommendation": "Consider using SMOTE or class weights during model training to address the imbalance."
            })
        
        # Missing values recommendations
        null_values = self.report_data["missing_values"]["null_values"]["train"]
        empty_strings = self.report_data["missing_values"]["empty_strings"]["train"]
        
        for col, count in null_values.items():
            if count > 0:
                recommendations.append({
                    "issue": f"Missing Values in {col}",
                    "description": f"{count} null values in {col} column",
                    "recommendation": f"Handle missing values in {col} through imputation or removal."
                })
        
        for col, count in empty_strings.items():
            if count > 0:
                recommendations.append({
                    "issue": f"Empty Strings in {col}",
                    "description": f"{count} empty strings in {col} column",
                    "recommendation": f"Handle empty strings in {col} through imputation or special encoding."
                })
        
        # Statement length recommendation
        all_df = pd.concat([self.train_df, self.test_df, self.valid_df])
        if all_df['statement'].str.len().max() > 512:  # BERT typically handles 512 tokens
            recommendations.append({
                "issue": "Long Statements",
                "description": "Some statements exceed typical BERT token limits",
                "recommendation": "Consider truncating or summarizing long statements to fit within BERT token limits."
            })
        
        # Feature extraction recommendation
        recommendations.append({
            "issue": "Feature Extraction",
            "description": "Temporal information may be embedded in the statement context",
            "recommendation": "Extract temporal features from statements and context for incorporation into the model."
        })
        
        self.report_data["preprocessing_recommendations"] = recommendations
        print(f"Generated {len(recommendations)} preprocessing recommendations")
    
    def save_report(self):
        """Save the report data to a JSON file"""
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_data["report_timestamp"] = timestamp
        
        # Save report data as JSON
        report_path = os.path.join(self.output_dir, 'eda_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.report_data, f, cls=NumpyEncoder, indent=2)
        
        # Generate a human-readable summary
        summary_path = os.path.join(self.output_dir, 'eda_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"LIAR Dataset Exploratory Data Analysis Summary\n")
            f.write(f"Generated on: {timestamp}\n\n")
            
            # Dataset Info
            info = self.report_data["dataset_info"]
            f.write(f"Dataset Information:\n")
            f.write(f"- Train set: {info['sizes']['train']} samples\n")
            f.write(f"- Test set: {info['sizes']['test']} samples\n")
            f.write(f"- Validation set: {info['sizes']['valid']} samples\n")
            f.write(f"- Total: {info['sizes']['total']} samples\n\n")
            
            # Label Distribution
            f.write(f"Label Distribution:\n")
            all_labels = self.report_data["label_distribution"]["all"]
            for label, count in all_labels.items():
                f.write(f"- {label}: {count} samples\n")
            
            f.write(f"\nBinary Label Distribution:\n")
            binary = self.report_data["label_distribution"]["binary"]
            for label, count in binary.items():
                f.write(f"- {label}: {count} samples\n")
            f.write(f"- Class imbalance ratio: {self.report_data['label_distribution']['class_imbalance']:.2f}\n\n")
            
            # Missing Values
            f.write(f"Missing Values Analysis:\n")
            null_train = self.report_data["missing_values"]["null_values"]["train"]
            empty_train = self.report_data["missing_values"]["empty_strings"]["train"]
            
            null_cols = [col for col, count in null_train.items() if count > 0]
            if null_cols:
                f.write(f"- Columns with null values: {', '.join(null_cols)}\n")
            else:
                f.write(f"- No null values found\n")
                
            empty_cols = [col for col, count in empty_train.items() if count > 0]
            if empty_cols:
                f.write(f"- Columns with empty strings: {', '.join(empty_cols)}\n\n")
            else:
                f.write(f"- No empty strings found\n\n")
            
            # Statement Statistics
            stmt_stats = self.report_data["statement_statistics"]["overall"]
            f.write(f"Statement Statistics:\n")
            f.write(f"- Mean statement length: {stmt_stats['mean_length']:.1f} characters\n")
            f.write(f"- Mean word count: {stmt_stats['mean_word_count']:.1f} words\n\n")
            
            # Speaker Statistics
            speaker_stats = self.report_data["speaker_statistics"]
            f.write(f"Speaker Statistics:\n")
            f.write(f"- Total unique speakers: {speaker_stats['total_unique_speakers']}\n")
            f.write(f"- Top speaker: {next(iter(speaker_stats['top_speakers']))}\n\n")
            
            # Preprocessing Recommendations
            f.write(f"Preprocessing Recommendations:\n")
            for i, rec in enumerate(self.report_data["preprocessing_recommendations"], 1):
                f.write(f"{i}. {rec['issue']}: {rec['recommendation']}\n")
        
        print(f"Report saved to {report_path}")
        print(f"Summary saved to {summary_path}")
    
    def run_all_analyses(self):
        """Run all analyses and generate the complete report"""
        print("Starting LIAR dataset exploratory data analysis...")
        
        self.get_basic_info()
        self.analyze_labels()
        self.check_missing_values()
        self.analyze_statements()
        self.analyze_speakers()
        self.analyze_temporal_patterns()
        self.generate_preprocessing_recommendations()
        self.save_report()
        
        print("EDA completed successfully!")

if __name__ == "__main__":
    explorer = LiarDataExplorer()
    explorer.run_all_analyses() 