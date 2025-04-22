#!/usr/bin/env python3
"""
Data utilities for the fact checking model
"""

import os
import json
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import re
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for temporal feature extraction
MONTH_NAMES = [
    'january', 'february', 'march', 'april', 'may', 'june', 
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
]

RELATIVE_TIME_WORDS = [
    'yesterday', 'today', 'tomorrow', 'last week', 'last month', 'last year',
    'next week', 'next month', 'next year', 'ago', 'since', 'previous',
    'earlier', 'later', 'recent', 'recently', 'current', 'currently',
    'now', 'present', 'past', 'future', 'before', 'after', 'during'
]


class FactCheckingDataset(Dataset):
    """Dataset for fact checking tasks"""
    
    def __init__(
        self, 
        claims: List[str], 
        tokenizer: RobertaTokenizer, 
        labels: Optional[List[int]] = None, 
        max_length: int = 512,
        include_temporal_features: bool = True,
        claim_ids: Optional[List[str]] = None
    ):
        """
        Initialize dataset
        
        Args:
            claims: List of claim texts
            tokenizer: RoBERTa tokenizer
            labels: Optional list of labels (0 for FALSE, 1 for TRUE)
            max_length: Maximum sequence length
            include_temporal_features: Whether to include temporal features
            claim_ids: Optional list of claim IDs
        """
        self.claims = claims
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.include_temporal_features = include_temporal_features
        self.claim_ids = claim_ids if claim_ids is not None else [f"claim_{i}" for i in range(len(claims))]
        
        # Extract temporal features if enabled
        if self.include_temporal_features:
            self.temporal_features = extract_batch_temporal_features(self.claims)
        
    def __len__(self):
        return len(self.claims)
    
    def __getitem__(self, idx):
        # Tokenize claim
        encoding = self.tokenizer(
            self.claims[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        
        # Add claim ID
        item['claim_id'] = self.claim_ids[idx]
        
        # Add temporal features if enabled
        if self.include_temporal_features:
            item['temporal_features'] = self.temporal_features[idx]
        
        return item


def extract_temporal_features(text: str) -> torch.Tensor:
    """
    Extract temporal features from text
    
    Features:
    1. Presence of years (e.g., 2020, '98)
    2. Presence of month names
    3. Presence of relative time words
    
    Args:
        text: Input text
        
    Returns:
        tensor: Tensor of temporal features
    """
    # Convert to lowercase for matching
    text_lower = text.lower()
    
    # Feature 1: Years (e.g., 2020, '98)
    years_pattern = r'\b(19|20)\d{2}\b|\'\d{2}\b'
    has_years = 1.0 if re.search(years_pattern, text) else 0.0
    
    # Feature 2: Month names
    has_months = 0.0
    for month in MONTH_NAMES:
        if month in text_lower:
            has_months = 1.0
            break
    
    # Feature 3: Relative time words
    has_relative_time = 0.0
    for word in RELATIVE_TIME_WORDS:
        if word in text_lower:
            has_relative_time = 1.0
            break
    
    # Create tensor
    features = torch.tensor([has_years, has_months, has_relative_time], dtype=torch.float)
    
    return features


def extract_batch_temporal_features(texts: List[str]) -> torch.Tensor:
    """
    Extract temporal features for a batch of texts
    
    Args:
        texts: List of input texts
        
    Returns:
        tensor: Tensor of temporal features for the batch
    """
    features_list = [extract_temporal_features(text) for text in texts]
    return torch.stack(features_list)


def load_claims_data(
    file_path: str, 
    label_key: str = 'label',
    text_key: str = 'claim_text',
    id_key: str = 'claim_id'
) -> Tuple[List[str], Optional[List[int]], List[str]]:
    """
    Load claims data from JSON file
    
    Args:
        file_path: Path to the JSON file
        label_key: Key for labels in the JSON
        text_key: Key for claim text in the JSON
        id_key: Key for claim ID in the JSON
        
    Returns:
        claims: List of claim texts
        labels: List of labels if available, else None
        claim_ids: List of claim IDs
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Claims file not found: {file_path}")
    
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dictionary formats
    if isinstance(data, list):
        # List of dictionaries
        claims = [item.get(text_key, "") for item in data if text_key in item]
        
        # Extract labels if they exist
        labels = None
        if all(label_key in item for item in data):
            labels = [item.get(label_key) for item in data if text_key in item]
            
            # Convert string labels to integers if needed
            if labels and isinstance(labels[0], str):
                label_map = {'FALSE': 0, 'TRUE': 1, 'false': 0, 'true': 1, 'False': 0, 'True': 1}
                labels = [label_map.get(label, label) for label in labels]
        
        # Extract claim IDs
        claim_ids = [item.get(id_key, f"claim_{i}") for i, item in enumerate(data) if text_key in item]
    else:
        # Dictionary format
        claims = []
        labels = []
        claim_ids = []
        
        for key, item in data.items():
            if isinstance(item, dict) and text_key in item:
                claims.append(item.get(text_key, ""))
                
                # Extract label if it exists
                if label_key in item:
                    label = item.get(label_key)
                    if isinstance(label, str):
                        label_map = {'FALSE': 0, 'TRUE': 1, 'false': 0, 'true': 1, 'False': 0, 'True': 1}
                        label = label_map.get(label, label)
                    labels.append(label)
                
                # Use key as claim ID, or field if available
                claim_id = item.get(id_key, key)
                claim_ids.append(claim_id)
        
        # If no labels found, set to None
        if not labels:
            labels = None
    
    # Log statistics
    logger.info(f"Loaded {len(claims)} claims from {file_path}")
    if labels:
        if len(set(labels)) <= 2:  # Binary classification
            num_true = sum(1 for label in labels if label == 1)
            num_false = sum(1 for label in labels if label == 0)
            logger.info(f"  TRUE claims: {num_true} ({num_true/len(labels):.1%})")
            logger.info(f"  FALSE claims: {num_false} ({num_false/len(labels):.1%})")
    
    return claims, labels, claim_ids


def create_data_loaders(
    claims: List[str],
    tokenizer: RobertaTokenizer,
    batch_size: int = 16,
    max_length: int = 512,
    include_temporal_features: bool = True,
    labels: Optional[List[int]] = None,
    claim_ids: Optional[List[str]] = None,
    shuffle: bool = False
) -> DataLoader:
    """
    Create data loader for the claims
    
    Args:
        claims: List of claim texts
        tokenizer: RoBERTa tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        include_temporal_features: Whether to include temporal features
        labels: Optional list of labels
        claim_ids: Optional list of claim IDs
        shuffle: Whether to shuffle the data
        
    Returns:
        data_loader: DataLoader object
    """
    # Create dataset
    dataset = FactCheckingDataset(
        claims=claims,
        tokenizer=tokenizer,
        labels=labels,
        max_length=max_length,
        include_temporal_features=include_temporal_features,
        claim_ids=claim_ids
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return data_loader


def format_results(
    claim_ids: List[str],
    claims: List[str],
    predictions: List[int],
    probabilities: Optional[List[float]] = None,
    labels: Optional[List[int]] = None,
    output_format: str = 'json'
) -> Union[Dict, pd.DataFrame]:
    """
    Format the results of inference
    
    Args:
        claim_ids: List of claim IDs
        claims: List of claim texts
        predictions: List of predictions (0 for FALSE, 1 for TRUE)
        probabilities: Optional list of probabilities for the predicted class
        labels: Optional list of true labels
        output_format: Output format ('json' or 'dataframe')
        
    Returns:
        results: Formatted results
    """
    # Create result dictionary
    results_list = []
    for i, (claim_id, claim, pred) in enumerate(zip(claim_ids, claims, predictions)):
        item = {
            'claim_id': claim_id,
            'claim_text': claim,
            'prediction': 'TRUE' if pred == 1 else 'FALSE',
            'prediction_score': int(pred)  # 0 or 1
        }
        
        # Add probability if available
        if probabilities is not None:
            prob = probabilities[i]
            item['probability'] = float(prob)
            
            # Calculate confidence score (distance from 0.5)
            confidence = abs(prob - 0.5) * 2  # Scale to [0, 1]
            item['confidence'] = float(confidence)
        
        # Add ground truth if available
        if labels is not None:
            item['ground_truth'] = 'TRUE' if labels[i] == 1 else 'FALSE'
            item['correct'] = (pred == labels[i])
        
        results_list.append(item)
    
    # Return in requested format
    if output_format == 'dataframe':
        return pd.DataFrame(results_list)
    else:
        return {'results': results_list}


def save_results(results: Union[Dict, pd.DataFrame], output_file: str):
    """
    Save the results to a file
    
    Args:
        results: Results to save
        output_file: Output file path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Handle different input types
    if isinstance(results, pd.DataFrame):
        # Convert DataFrame to dictionary
        results_dict = {'results': results.to_dict(orient='records')}
    else:
        results_dict = results
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


def main():
    """Test the functionality of this module"""
    # Example claims
    claims = [
        "NASA spent $28 million on a space toilet that doesn't work.",
        "The Earth is flat according to all scientific evidence.",
        "In 2019, the United States had the lowest unemployment rate in 50 years.",
        "The COVID-19 vaccine was released in April 2020."
    ]
    
    # Extract temporal features
    features = extract_batch_temporal_features(claims)
    
    print("Claims and their temporal features:")
    for claim, feature in zip(claims, features):
        print(f"Claim: {claim}")
        print(f"Features: {feature.tolist()}")
        print()


if __name__ == "__main__":
    main() 