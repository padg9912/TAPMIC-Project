#!/usr/bin/env python3
# claim_extractor.py
# Extracts key claims from preprocessed datasets for the RAG pipeline

import pandas as pd
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
import json
from datetime import datetime

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERMEDIATE_DIR = os.path.join(BASE_DIR, "data", "intermediate_1")
CLAIMS_DIR = os.path.join(BASE_DIR, "rag_pipeline", "claims_data")
os.makedirs(CLAIMS_DIR, exist_ok=True)

# Input files
TRAIN_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_train.csv")
TEST_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_test.csv")
VALID_PATH = os.path.join(INTERMEDIATE_DIR, "intermediate1_validation.csv")

# Output files
CLAIMS_FILE = os.path.join(CLAIMS_DIR, "political_claims.json")
CLAIMS_STATS_FILE = os.path.join(CLAIMS_DIR, "political_claims_stats.txt")

def load_data():
    """Load the preprocessed datasets"""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    valid_df = pd.read_csv(VALID_PATH)
    
    print(f"Loaded {train_df.shape[0]} training samples")
    print(f"Loaded {test_df.shape[0]} testing samples")
    print(f"Loaded {valid_df.shape[0]} validation samples")
    
    return train_df, test_df, valid_df

def clean_claim(text):
    """Clean and normalize the claim text"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase and remove excess whitespace
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but preserve important punctuation
    text = re.sub(r'[^\w\s.,?!;:-]', '', text)
    
    return text

def extract_keywords(text, subject, speaker, context):
    """Extract keywords from the claim for search optimization"""
    keywords = []
    
    # Extract from the claim text (main statements)
    if text:
        # Use the first sentence as it often contains the main claim
        sentences = sent_tokenize(text)
        main_sentence = sentences[0] if sentences else text
        
        # Extract key nouns and entities (simplified approach)
        words = main_sentence.split()
        # Take the most important words (first 5-8 words are often key)
        keywords.extend(words[:min(8, len(words))])
    
    # Add subject as it's highly relevant
    if not pd.isna(subject) and subject:
        keywords.append(subject)
    
    # Add speaker name
    if not pd.isna(speaker) and speaker:
        keywords.append(speaker)
    
    # Add context if available
    if not pd.isna(context) and context:
        keywords.append(context)
    
    # Clean and deduplicate keywords
    cleaned_keywords = [k.strip().lower() for k in keywords if k.strip()]
    cleaned_keywords = [re.sub(r'[^\w\s]', '', k) for k in cleaned_keywords]
    unique_keywords = list(set([k for k in cleaned_keywords if len(k) > 2]))
    
    return unique_keywords

def extract_claims(df, source_name):
    """Extract claims from a dataset with their context"""
    claims = []
    
    for idx, row in df.iterrows():
        # Skip rows with missing statement
        if pd.isna(row['statement']):
            continue
            
        # Clean the statement text
        cleaned_text = clean_claim(row['statement'])
        if not cleaned_text:
            continue
            
        # Extract subject, date and other metadata
        subject = row['subject'] if 'subject' in row and not pd.isna(row['subject']) else ""
        date_str = row['date'] if 'date' in row and not pd.isna(row['date']) else ""
        speaker = row['speaker'] if 'speaker' in row and not pd.isna(row['speaker']) else ""
        context = row['context'] if 'context' in row and not pd.isna(row['context']) else ""
        
        # Get search keywords
        keywords = extract_keywords(cleaned_text, subject, speaker, context)
        
        # Build search query
        search_query = f"{speaker} {cleaned_text[:100]}"
        if subject:
            search_query += f" {subject}"
            
        # Create claim object
        claim = {
            'id': str(row['id']) if 'id' in row else f"{source_name}_{idx}",
            'text': cleaned_text,
            'speaker': speaker,
            'date': date_str,
            'subject': subject,
            'context': context,
            'truth_label': "TRUE" if row['label'] == 1 else "FALSE",
            'original_label': row['original_label'] if 'original_label' in row else "",
            'keywords': keywords,
            'search_query': search_query.strip(),
            'source': source_name,
            'credibility_score': row['credibility_score'] if 'credibility_score' in row else None
        }
        
        claims.append(claim)
    
    print(f"Extracted {len(claims)} claims from {source_name}")
    return claims

def save_claims(claims):
    """Save extracted claims to JSON file"""
    with open(CLAIMS_FILE, 'w') as f:
        json.dump(claims, f, indent=2)
    
    print(f"Saved {len(claims)} claims to {CLAIMS_FILE}")

def analyze_claims(claims):
    """Generate statistics about extracted claims"""
    stats = []
    stats.append(f"Claim Extraction Statistics - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats.append(f"Total claims extracted: {len(claims)}")
    
    # Count by truth label
    truth_counts = {}
    for claim in claims:
        truth_label = claim['truth_label']
        truth_counts[truth_label] = truth_counts.get(truth_label, 0) + 1
    
    stats.append("\nDistribution by truth label:")
    for label, count in truth_counts.items():
        percentage = (count / len(claims)) * 100
        stats.append(f"  {label}: {count} claims ({percentage:.2f}%)")
    
    # Count by source
    source_counts = {}
    for claim in claims:
        source = claim['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    stats.append("\nDistribution by source:")
    for source, count in source_counts.items():
        percentage = (count / len(claims)) * 100
        stats.append(f"  {source}: {count} claims ({percentage:.2f}%)")
    
    # Subject distribution
    subject_counts = {}
    for claim in claims:
        subject = claim['subject']
        if subject:  # Skip empty subjects
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    stats.append("\nTop subjects:")
    top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for subject, count in top_subjects:
        percentage = (count / len(claims)) * 100
        stats.append(f"  {subject}: {count} claims ({percentage:.2f}%)")
    
    # Text length stats
    text_lengths = [len(claim['text']) for claim in claims]
    avg_length = sum(text_lengths) / len(text_lengths)
    max_length = max(text_lengths)
    min_length = min(text_lengths)
    
    stats.append("\nClaim text statistics:")
    stats.append(f"  Average length: {avg_length:.2f} characters")
    stats.append(f"  Maximum length: {max_length} characters")
    stats.append(f"  Minimum length: {min_length} characters")
    
    # Save statistics
    with open(CLAIMS_STATS_FILE, 'w') as f:
        f.write('\n'.join(stats))
    
    print(f"Saved claim statistics to {CLAIMS_STATS_FILE}")
    return stats

def main():
    """Main execution function"""
    print("Starting claim extraction process...")
    
    # Load datasets
    train_df, test_df, valid_df = load_data()
    
    # Extract claims from each dataset
    train_claims = extract_claims(train_df, "train")
    test_claims = extract_claims(test_df, "test")
    valid_claims = extract_claims(valid_df, "validation")
    
    # Combine all claims
    all_claims = train_claims + test_claims + valid_claims
    
    # Save to file
    save_claims(all_claims)
    
    # Generate statistics
    stats = analyze_claims(all_claims)
    for line in stats:
        print(line)
    
    print("Claim extraction completed successfully!")

if __name__ == "__main__":
    main() 