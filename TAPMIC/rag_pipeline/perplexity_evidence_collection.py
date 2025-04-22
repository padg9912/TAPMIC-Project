#!/usr/bin/env python3
# perplexity_evidence_collection.py
# Process remaining claims using Perplexity API to generate evidence with temporal information

import os
import json
import time
import logging
import re
import argparse
import requests
from datetime import datetime
import sys
from tqdm import tqdm
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_data", "evidence_collection.log"), mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS_DIR = os.path.join(BASE_DIR, "rag_pipeline", "claims_data")
EVIDENCE_DIR = os.path.join(BASE_DIR, "rag_pipeline", "evidence_data")
EVIDENCE_FILE = os.path.join(EVIDENCE_DIR, "collected_evidence.json")
REPORT_FILE = os.path.join(EVIDENCE_DIR, "evidence_collection_report.txt")

# Configuration
RESULTS_PER_CLAIM = 3  # Number of evidence items per claim
BATCH_SIZE = 10  # Process claims in larger batches for API efficiency
MAX_CLAIMS_TO_PROCESS = 0  # 0 means process all remaining claims
PERPLEXITY_API_KEY = None  # Will be set via command line arg
API_ENDPOINT = "https://api.perplexity.ai/chat/completions"

# System prompt for evidence generation with temporal focus
SYSTEM_PROMPT = """You are an expert fact-checker tasked with providing objective and reliable search results for political claims.
Your job is to generate realistic search results that might appear when someone searches for information about a political claim.
For each claim, generate potential search results that include:
1. A mix of sources (news sites, fact-checking sites, political websites)
2. A range of publication dates spanning different time periods relevant to the claim
3. Both supporting and contradicting evidence when appropriate
4. A realistic title, URL, and snippet for each result

IMPORTANT: Each result MUST include specific dates within the content (exact day, month, and year).
Make the temporal information (dates) a central part of the evidence.

Format each result as:
Title: [title]
URL: [url with date in format YYYY/MM/DD]
Snippet: [content with specific dates mentioned]
"""

def load_existing_evidence():
    """Load existing evidence to analyze patterns and avoid duplicates"""
    try:
        if os.path.exists(EVIDENCE_FILE):
            with open(EVIDENCE_FILE, 'r') as f:
                existing_evidence = json.load(f)
            logging.info(f"Loaded {len(existing_evidence)} existing evidence items")
            return existing_evidence
        else:
            logging.warning(f"No existing evidence file found at {EVIDENCE_FILE}")
            return []
    except Exception as e:
        logging.error(f"Error loading existing evidence: {e}")
        return []

def extract_processed_claim_ids(existing_evidence):
    """Extract claim IDs that have already been processed"""
    processed_claim_ids = set()
    for evidence in existing_evidence:
        if 'claim_id' in evidence:
            processed_claim_ids.add(evidence['claim_id'])
    
    logging.info(f"Found {len(processed_claim_ids)} claims already processed")
    return processed_claim_ids

def analyze_temporal_patterns(existing_evidence):
    """Analyze temporal patterns in existing evidence to guide generation"""
    patterns = {
        'publication_years': {},
        'domains': {},
        'has_temporal_data': 0,
        'common_sources': []
    }
    
    for evidence in existing_evidence:
        # Count publication years
        if evidence.get('publication_date'):
            year = evidence['publication_date'].split('-')[0]
            patterns['publication_years'][year] = patterns['publication_years'].get(year, 0) + 1
        
        # Count domains
        if evidence.get('domain'):
            domain = evidence['domain']
            patterns['domains'][domain] = patterns['domains'].get(domain, 0) + 1
        
        # Count evidence with temporal data
        if evidence.get('has_temporal_data', False) or evidence.get('publication_date'):
            patterns['has_temporal_data'] += 1
    
    # Get top domains
    top_domains = sorted(patterns['domains'].items(), key=lambda x: x[1], reverse=True)[:10]
    patterns['common_sources'] = [domain for domain, count in top_domains]
    
    logging.info(f"Analyzed temporal patterns in {len(existing_evidence)} evidence items")
    if existing_evidence:
        temporal_percentage = (patterns['has_temporal_data'] / len(existing_evidence)) * 100
        logging.info(f"Temporal data percentage: {temporal_percentage:.2f}%")
    
    if patterns['common_sources']:
        logging.info(f"Common sources: {', '.join(patterns['common_sources'][:5])}")
    
    return patterns

def load_claims(claims_file):
    """Load claims from a JSON file"""
    try:
        with open(claims_file, 'r') as f:
            claims_data = json.load(f)
        
        logging.info(f"Loaded claims file with {len(claims_data)} entries")
        
        # Convert to list of claim objects
        processed_claims = []
        
        # Handle different JSON structures
        if isinstance(claims_data, list):
            for i, item in enumerate(claims_data):
                # Try different possible field names for claims
                claim_text = None
                for field in ['claim', 'statement', 'text']:
                    if field in item:
                        claim_text = item[field]
                        break
                
                if claim_text is None:
                    continue
                
                # Use item ID if available, otherwise generate one
                claim_id = item.get('id', f"claim_{i}")
                subject = item.get('subject', '')
                
                processed_claims.append({
                    'id': claim_id,
                    'text': claim_text,
                    'subject': subject
                })
        elif isinstance(claims_data, dict):
            for key, value in claims_data.items():
                if isinstance(value, dict):
                    # Try different possible field names for claims
                    claim_text = None
                    for field in ['claim', 'statement', 'text']:
                        if field in value:
                            claim_text = value[field]
                            break
                    
                    if claim_text is None:
                        claim_text = str(value)
                    
                    subject = value.get('subject', '')
                    
                    processed_claims.append({
                        'id': key,
                        'text': claim_text,
                        'subject': subject
                    })
                else:
                    processed_claims.append({
                        'id': key,
                        'text': str(value),
                        'subject': ''
                    })
        
        logging.info(f"Processed {len(processed_claims)} claims")
        return processed_claims
    except Exception as e:
        logging.error(f"Error loading claims file: {e}")
        raise

def extract_date_from_url(url):
    """Extract date from URL if it follows common patterns"""
    date_patterns = [
        r'(\d{4})/(\d{1,2})/(\d{1,2})',  # YYYY/MM/DD
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        r'/(\d{4})(\d{2})(\d{2})/',      # YYYYMMDD
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, url)
        if match:
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                if 1990 <= year <= 2025 and 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year}-{month:02d}-{day:02d}"
            except:
                pass
    
    return None

def extract_dates_from_text(text):
    """Extract potential dates from text content"""
    # Common date formats
    date_patterns = [
        r'\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b',  # DD/MM/YYYY or MM/DD/YYYY
        r'\b(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})\b',     # YYYY/MM/DD
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',  # Month DD, YYYY
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[\.|\s]+(\d{1,2})[,|\s]+(\d{4})\b',  # Abbreviated months
    ]
    
    month_names = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12,
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'Jun': 6, 'Jul': 7, 'Aug': 8, 
        'Sep': 9, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    extracted_dates = []
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                # Handle month name format
                if match.group(1) in month_names:
                    month = month_names[match.group(1)]
                    day = int(match.group(2))
                    year = int(match.group(3))
                    if 1990 <= year <= 2025:
                        extracted_dates.append(f"{year}-{month:02d}-{day:02d}")
                else:
                    # Handle numeric patterns
                    part1 = int(match.group(1))
                    part2 = int(match.group(2))
                    part3 = int(match.group(3))
                    
                    # Determine format based on first number
                    if part1 > 1000:  # YYYY/MM/DD
                        year, month, day = part1, part2, part3
                    elif part2 > 12:  # DD/MM/YYYY (assuming European format)
                        day, month, year = part1, part2, part3
                    else:  # MM/DD/YYYY (default US format)
                        month, day, year = part1, part2, part3
                    
                    # Fix 2-digit years
                    if year < 100:
                        if year < 50:  # Assume 20xx for lower numbers
                            year += 2000
                        else:  # Assume 19xx for higher numbers
                            year += 1900
                    
                    # Validate date
                    if 1990 <= year <= 2025 and 1 <= month <= 12 and 1 <= day <= 31:
                        extracted_dates.append(f"{year}-{month:02d}-{day:02d}")
            except:
                continue
    
    return extracted_dates

def get_domain(url):
    """Extract domain from URL"""
    try:
        domain = urlparse(url).netloc
        return domain if domain else url
    except:
        return url

def generate_evidence_with_perplexity(api_key, claim, subject=""):
    """Generate evidence for a claim using Perplexity API"""
    # Create prompt with system message
    user_message = f"Political claim: {claim}\n"
    if subject:
        user_message += f"Subject: {subject}\n"
    user_message += f"\nGenerate {RESULTS_PER_CLAIM} search results with temporal information for this claim."
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "sonar-pro",  # Updated to a valid Perplexity model
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        # Make the API request
        response = requests.post(API_ENDPOINT, headers=headers, json=payload)
        
        # Check for successful response
        if response.status_code == 200:
            response_data = response.json()
            generated_text = response_data["choices"][0]["message"]["content"]
            
            # Parse generated evidence
            results = []
            
            # Split into separate evidence items
            evidence_blocks = generated_text.split("\n\n")
            for block in evidence_blocks:
                if not block.strip():
                    continue
                    
                evidence = {}
                current_field = None
                
                for line in block.strip().split("\n"):
                    if line.startswith("Title:"):
                        evidence["title"] = line.replace("Title:", "").strip()
                        current_field = "title"
                    elif line.startswith("URL:"):
                        evidence["link"] = line.replace("URL:", "").strip()
                        current_field = "link"
                    elif line.startswith("Snippet:"):
                        evidence["snippet"] = line.replace("Snippet:", "").strip()
                        current_field = "snippet"
                    elif current_field:
                        # Append to current field if it's a continuation
                        evidence[current_field] += " " + line.strip()
                
                # Only add if we have title and link
                if "title" in evidence and "link" in evidence:
                    results.append(evidence)
                
                # Only keep the requested number of results
                if len(results) >= RESULTS_PER_CLAIM:
                    break
            
            return results
        else:
            logging.error(f"API request failed with status code {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        logging.error(f"Error generating evidence with Perplexity API: {e}")
        return []

def process_evidence_item(claim_id, result, claim_text, subject, result_idx):
    """Process a single evidence item"""
    title = result.get('title', '')
    link = result.get('link', '')
    snippet = result.get('snippet', '')
    
    # Skip if no link or no content
    if not link or (not title and not snippet):
        return None
    
    # Extract dates
    url_date = extract_date_from_url(link)
    text_dates = extract_dates_from_text(title + " " + snippet)
    
    # Combine all found dates
    all_dates = [d for d in ([url_date] + text_dates) if d]
    publication_date = all_dates[0] if all_dates else None
    
    # Create unique ID for evidence
    evidence_id = f"{claim_id}_ev{result_idx}"
    
    # Create evidence item
    evidence = {
        "id": evidence_id,
        "claim_id": claim_id,
        "claim_text": claim_text,
        "title": title,
        "content": snippet,
        "source_url": link,
        "domain": get_domain(link),
        "publication_date": publication_date,
        "retrieval_date": datetime.now().strftime("%Y-%m-%d"),
        "all_dates_found": all_dates,
        "subject": subject,
        "has_temporal_data": bool(publication_date or all_dates),
        "source": "perplexity_generated"
    }
    
    return evidence

def save_evidence(evidence_items, existing_evidence):
    """Save evidence to JSON file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(EVIDENCE_FILE), exist_ok=True)
    
    # Combine existing and new evidence
    combined_evidence = existing_evidence + evidence_items
    
    try:
        with open(EVIDENCE_FILE, 'w') as f:
            json.dump(combined_evidence, f, indent=2)
        logging.info(f"Saved {len(combined_evidence)} evidence items to {EVIDENCE_FILE}")
    except Exception as e:
        logging.error(f"Error saving evidence: {e}")
        # Try to save to backup file
        try:
            backup_file = f"{EVIDENCE_FILE}.backup.{int(time.time())}.json"
            with open(backup_file, 'w') as f:
                json.dump(combined_evidence, f, indent=2)
            logging.info(f"Saved backup to {backup_file}")
        except:
            logging.error("Failed to save backup file")

def generate_report(evidence_items, existing_evidence):
    """Generate a report about the evidence collection process"""
    all_evidence = existing_evidence + evidence_items
    
    # Count by subject
    subject_counts = {}
    for item in all_evidence:
        subject = item.get('subject', 'unknown')
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    # Count by domain
    domain_counts = {}
    for item in all_evidence:
        domain = item.get('domain', 'unknown')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    # Temporal analysis
    with_dates = sum(1 for item in all_evidence if item.get('publication_date'))
    with_temporal = sum(1 for item in all_evidence if item.get('has_temporal_data', False))
    
    # Create report
    report_lines = [
        "=== EVIDENCE COLLECTION REPORT ===",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total evidence items: {len(all_evidence)}",
        f"Items with publication dates: {with_dates} ({with_dates/len(all_evidence)*100:.2f}%)",
        f"Items with any temporal data: {with_temporal} ({with_temporal/len(all_evidence)*100:.2f}%)",
        "",
        "=== SUBJECT DISTRIBUTION ===",
    ]
    
    for subject, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        if subject:  # Skip empty subjects
            percentage = (count / len(all_evidence)) * 100
            report_lines.append(f"  {subject}: {count} items ({percentage:.2f}%)")
    
    report_lines.extend([
        "",
        "=== TOP SOURCES ===",
    ])
    
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(all_evidence)) * 100
        report_lines.append(f"  {domain}: {count} items ({percentage:.2f}%)")
    
    # Save report to file
    try:
        with open(REPORT_FILE, 'w') as f:
            f.write('\n'.join(report_lines))
        logging.info(f"Saved report to {REPORT_FILE}")
    except Exception as e:
        logging.error(f"Error saving report: {e}")
    
    return report_lines

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate evidence for political claims using Perplexity API")
    parser.add_argument("--claims_file", type=str, default=os.path.join(CLAIMS_DIR, "political_claims.json"),
                      help="Path to the claims JSON file")
    parser.add_argument("--start_index", type=int, default=0,
                      help="Starting index for claims to process")
    parser.add_argument("--end_index", type=int, default=None,
                      help="Ending index for claims to process")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                      help="Number of claims to process in each batch")
    parser.add_argument("--max_claims", type=int, default=MAX_CLAIMS_TO_PROCESS,
                      help="Maximum number of claims to process (0 for all)")
    parser.add_argument("--api_key", type=str, required=True,
                      help="Perplexity API key")
    return parser.parse_args()

def main():
    """Main function to process claims and generate evidence"""
    args = parse_args()
    
    # Set API key
    global PERPLEXITY_API_KEY
    PERPLEXITY_API_KEY = args.api_key
    
    logging.info("Starting evidence collection with Perplexity API...")
    
    # Load existing evidence
    existing_evidence = load_existing_evidence()
    
    # Extract claim IDs that have already been processed
    processed_claim_ids = extract_processed_claim_ids(existing_evidence)
    
    # Analyze temporal patterns
    temporal_patterns = analyze_temporal_patterns(existing_evidence)
    
    # Load claims
    all_claims = load_claims(args.claims_file)
    logging.info(f"Loaded {len(all_claims)} claims from file")
    
    # Filter out already processed claims
    claims_to_process = [c for c in all_claims if c['id'] not in processed_claim_ids]
    logging.info(f"Found {len(claims_to_process)} claims that haven't been processed yet")
    
    # Apply start and end index
    if args.end_index:
        claims_to_process = claims_to_process[args.start_index:args.end_index]
    else:
        claims_to_process = claims_to_process[args.start_index:]
    
    # Limit to max_claims if specified
    if args.max_claims > 0 and len(claims_to_process) > args.max_claims:
        claims_to_process = claims_to_process[:args.max_claims]
    
    logging.info(f"Will process {len(claims_to_process)} claims")
    
    # Process claims in batches
    all_new_evidence = []
    
    for batch_start in range(0, len(claims_to_process), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(claims_to_process))
        batch = claims_to_process[batch_start:batch_end]
        
        logging.info(f"Processing batch {batch_start//args.batch_size + 1}/{(len(claims_to_process) + args.batch_size - 1)//args.batch_size}")
        batch_evidence = []
        
        for claim in tqdm(batch):
            claim_id = claim['id']
            claim_text = claim['text']
            subject = claim.get('subject', '')
            
            # Generate evidence with Perplexity API
            results = generate_evidence_with_perplexity(PERPLEXITY_API_KEY, claim_text, subject)
            
            # Process and store evidence items
            for idx, result in enumerate(results):
                evidence = process_evidence_item(claim_id, result, claim_text, subject, idx)
                if evidence:
                    batch_evidence.append(evidence)
            
            # Brief delay to avoid hitting API rate limits
            time.sleep(0.2)
        
        # Add batch evidence to all evidence
        all_new_evidence.extend(batch_evidence)
        
        # Save intermediate results every batch
        if batch_evidence:
            intermediate_file = os.path.join(EVIDENCE_DIR, f"evidence_batch_{batch_start//args.batch_size + 1}.json")
            with open(intermediate_file, 'w') as f:
                json.dump(batch_evidence, f, indent=2)
            logging.info(f"Saved {len(batch_evidence)} evidence items to {intermediate_file}")
    
    # Save all new evidence
    save_evidence(all_new_evidence, existing_evidence)
    
    # Generate report
    report_lines = generate_report(all_new_evidence, existing_evidence)
    for line in report_lines[:10]:  # Print first 10 lines of report
        logging.info(line)
    
    logging.info(f"Evidence collection completed successfully! Added {len(all_new_evidence)} new items.")

if __name__ == "__main__":
    main() 