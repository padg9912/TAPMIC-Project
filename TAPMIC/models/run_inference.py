#!/usr/bin/env python3
"""
Inference script for RoBERTa-based fact checking model
"""

import os
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from transformers import RobertaTokenizer

from roberta_model import FactCheckerModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_temporal_features(claim):
    """
    Extract temporal features from claim text
    
    Args:
        claim: Claim text
        
    Returns:
        features: List of temporal features
    """
    features = [0, 0, 0]  # Initialize with zeros
    
    # Feature 1: Contains year (4 digits)
    import re
    year_pattern = r'\b(19|20)\d{2}\b'
    if re.search(year_pattern, claim):
        features[0] = 1
        
    # Feature 2: Contains month name
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    if any(month in claim.lower() for month in months):
        features[1] = 1
        
    # Feature 3: Contains relative time words
    time_words = ['yesterday', 'today', 'tomorrow', 'last week', 'next month', 
                  'recently', 'ago', 'later', 'soon']
    if any(word in claim.lower() for word in time_words):
        features[2] = 1
        
    return features


def load_claims(claims_file):
    """
    Load claims from file
    
    Args:
        claims_file: Path to claims file (JSON)
        
    Returns:
        claims: List of claims
        claim_ids: List of claim IDs (if available)
    """
    logger.info(f"Loading claims from {claims_file}")
    
    with open(claims_file, 'r') as f:
        data = json.load(f)
    
    claims = []
    claim_ids = []
    
    # Handle different JSON formats
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # Extract claim text
                if 'claim' in item:
                    claims.append(item['claim'])
                elif 'claim_text' in item:
                    claims.append(item['claim_text'])
                else:
                    logger.warning(f"Claim text not found in item: {item}")
                    continue
                
                # Extract claim ID if available
                if 'id' in item:
                    claim_ids.append(item['id'])
                elif 'claim_id' in item:
                    claim_ids.append(item['claim_id'])
                else:
                    claim_ids.append(str(len(claim_ids)))
            else:
                claims.append(str(item))
                claim_ids.append(str(len(claim_ids)))
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and ('claim' in value or 'claim_text' in value):
                # Extract claim text
                if 'claim' in value:
                    claims.append(value['claim'])
                else:
                    claims.append(value['claim_text'])
                
                claim_ids.append(key)
            else:
                claims.append(str(value))
                claim_ids.append(key)
    
    logger.info(f"Loaded {len(claims)} claims")
    
    return claims, claim_ids


def run_inference(model, tokenizer, claims, use_temporal=False, batch_size=16, device='cuda'):
    """
    Run inference on claims
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        claims: List of claims
        use_temporal: Whether to use temporal features
        batch_size: Batch size
        device: Device to run inference on
        
    Returns:
        predictions: List of predictions (0 for FALSE, 1 for TRUE)
        probabilities: List of probabilities for the TRUE class
    """
    logger.info(f"Running inference on {len(claims)} claims")
    
    model.eval()
    predictions = []
    probabilities = []
    
    # Process in batches
    for i in tqdm(range(0, len(claims), batch_size), desc="Processing batches"):
        batch_claims = claims[i:i + batch_size]
        
        # Tokenize claims
        batch_encodings = tokenizer.batch_encode_plus(
            batch_claims,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        batch_encodings = {k: v.to(device) for k, v in batch_encodings.items()}
        
        # Extract temporal features if enabled
        if use_temporal:
            temporal_features = []
            for claim in batch_claims:
                features = extract_temporal_features(claim)
                temporal_features.append(features)
            
            temporal_features = torch.tensor(temporal_features, dtype=torch.float).to(device)
        else:
            temporal_features = None
        
        # Run inference
        with torch.no_grad():
            if temporal_features is not None:
                outputs = model(
                    input_ids=batch_encodings['input_ids'],
                    attention_mask=batch_encodings['attention_mask'],
                    temporal_features=temporal_features
                )
            else:
                outputs = model(
                    input_ids=batch_encodings['input_ids'],
                    attention_mask=batch_encodings['attention_mask']
                )
        
        # Get predictions and probabilities
        batch_probs = torch.softmax(outputs, dim=1)
        batch_preds = torch.argmax(outputs, dim=1)
        
        # Add to results
        predictions.extend(batch_preds.cpu().numpy())
        probabilities.extend(batch_probs[:, 1].cpu().numpy())  # Probability for TRUE class
    
    logger.info(f"Inference complete. {sum(predictions)} TRUE, {len(predictions) - sum(predictions)} FALSE")
    
    return predictions, probabilities


def save_results(output_file, claim_ids, claims, predictions, probabilities):
    """
    Save inference results to file
    
    Args:
        output_file: Path to output file
        claim_ids: List of claim IDs
        claims: List of claims
        predictions: List of predictions
        probabilities: List of probabilities
    """
    logger.info(f"Saving results to {output_file}")
    
    results = []
    
    for i, (claim_id, claim, pred, prob) in enumerate(zip(claim_ids, claims, predictions, probabilities)):
        results.append({
            'id': claim_id,
            'claim': claim,
            'prediction': 'TRUE' if pred == 1 else 'FALSE',
            'probability': float(prob),
            'confidence': float(max(prob, 1 - prob))
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)


def load_evidence(evidence_file, claim_ids=None):
    """
    Load evidence for claims if available
    
    Args:
        evidence_file: Path to evidence file
        claim_ids: List of claim IDs to filter evidence
        
    Returns:
        evidence_dict: Dictionary mapping claim IDs to evidence
    """
    logger.info(f"Loading evidence from {evidence_file}")
    
    with open(evidence_file, 'r') as f:
        data = json.load(f)
    
    evidence_dict = {}
    
    for item in data:
        claim_id = item.get('claim_id')
        
        if claim_id is None:
            continue
        
        if claim_ids is not None and claim_id not in claim_ids:
            continue
        
        if claim_id not in evidence_dict:
            evidence_dict[claim_id] = []
        
        evidence_dict[claim_id].append({
            'text': item.get('text', ''),
            'source': item.get('source', ''),
            'date': item.get('date', {})
        })
    
    logger.info(f"Loaded evidence for {len(evidence_dict)} claims")
    
    return evidence_dict


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run inference with a trained fact-checking model")
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--claims_file', type=str, required=True,
                        help='Path to the claims file (JSON)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output file for results')
    
    # Optional arguments
    parser.add_argument('--evidence_file', type=str, default=None,
                        help='Path to evidence file (JSON, optional)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help=f'Device to run inference on (default: {"cuda" if torch.cuda.is_available() else "cpu"})')
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = FactCheckerModel.from_pretrained(args.model_path)
    model.to(args.device)
    
    # Check if model uses temporal features
    use_temporal = hasattr(model.config, 'use_temporal') and model.config.use_temporal
    logger.info(f"Model uses temporal features: {use_temporal}")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    
    # Load claims
    claims, claim_ids = load_claims(args.claims_file)
    
    # Load evidence if provided
    evidence_dict = None
    if args.evidence_file:
        evidence_dict = load_evidence(args.evidence_file, claim_ids)
    
    # Run inference
    predictions, probabilities = run_inference(
        model=model,
        tokenizer=tokenizer,
        claims=claims,
        use_temporal=use_temporal,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Save results
    save_results(args.output_file, claim_ids, claims, predictions, probabilities)
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main() 