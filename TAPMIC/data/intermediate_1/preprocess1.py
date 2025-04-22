#!/usr/bin/env python3
# preprocess1.py
# Convert 6-class classification to binary classification

import pandas as pd
import os
import sys

# Define the mapping from the 6 classes to binary
# TRUE labels: "true", "mostly-true", "half-true"
# FALSE labels: "barely-true", "false", "pants-fire"
def map_label_to_binary(label):
    if label in ["true", "mostly-true", "half-true"]:
        return "TRUE"
    elif label in ["barely-true", "false", "pants-fire"]:
        return "FALSE"
    else:
        return None  # Handle unexpected labels

def process_file(input_path, output_path):
    """Process a single CSV file and save with binary labels"""
    print(f"Processing {input_path}...")
    
    # Read CSV file
    df = pd.read_csv(input_path)
    
    # Check if 'label' column exists
    if 'label' not in df.columns:
        print(f"Error: 'label' column not found in {input_path}")
        return False
    
    # Keep original label
    df['original_label'] = df['label']
    
    # Map 6 classes to binary
    df['label'] = df['label'].apply(map_label_to_binary)
    
    # Check if any labels couldn't be mapped
    unmapped = df[df['label'].isna()]
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} rows have labels that couldn't be mapped to binary")
        print(unmapped['original_label'].value_counts())
        # Fill unmapped labels as "FALSE" (conservative approach)
        df['label'] = df['label'].fillna("FALSE")
    
    # Save to output file
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Print class counts
    class_counts = df['label'].value_counts()
    print(f"Class distribution:")
    for label, count in class_counts.items():
        print(f"  {label}: {count} samples")
    
    return True

def main():
    # Define paths
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'raw'))
    output_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for filename in ['train.csv', 'test.csv', 'valid.csv']:
        input_path = os.path.join(data_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"Error: Input file {input_path} not found")
            continue
        
        process_file(input_path, output_path)
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 