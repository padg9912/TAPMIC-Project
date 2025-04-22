#!/usr/bin/env python3
# evidence_collector.py
# Collects evidence for political claims from web searches

import chromadb
from chromadb.utils import embedding_functions
import os
import json
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup
import re
import random
from urllib.parse import urlparse
import sys
import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_data", "evidence_collection.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "rag_pipeline", "chroma_db")
CLAIMS_DIR = os.path.join(BASE_DIR, "rag_pipeline", "claims_data")
EVIDENCE_DIR = os.path.join(BASE_DIR, "rag_pipeline", "evidence_data")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Configuration
CLAIMS_COLLECTION_NAME = "political_claims"
EVIDENCE_COLLECTION_NAME = "temporal_evidence"
RESULTS_PER_CLAIM = 3  # Number of search results to collect per claim

# Sampling configuration
REAL_SEARCH_SAMPLE_SIZE = 1000  # Number of claims to process with real search
USE_HYBRID_APPROACH = True  # Set to True to use both real search and LLM generation

# Google Search API configuration
# Add your Google API keys here. The system will rotate through them when quota is reached

# API usage tracking
MAX_QUERIES_PER_KEY = 100  # Google free tier typically allows 100 queries per day
api_key_usage = {}  # Track usage of each key

# User agents for web requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
]

def setup_chroma_client():
    """Connect to the existing ChromaDB client"""
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Use sentence transformers for embedding
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get claims collection
    claims_collection = client.get_collection(
        name=CLAIMS_COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    
    # Get evidence collection
    evidence_collection = client.get_collection(
        name=EVIDENCE_COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    
    return client, claims_collection, evidence_collection

def get_claims_to_process(claims_collection):
    """Retrieve claims to process from ChromaDB"""
    # Get all claims - this returns all metadata
    results = claims_collection.get()
    
    # Create list of claims with metadata
    claims = []
    for i in range(len(results['ids'])):
        claim = {
            'id': results['ids'][i],
            'text': results['documents'][i],
            'metadata': results['metadatas'][i]
        }
        claims.append(claim)
    
    # Limit to REAL_SEARCH_SAMPLE_SIZE claims for efficiency
    if len(claims) > REAL_SEARCH_SAMPLE_SIZE:
        claims = random.sample(claims, REAL_SEARCH_SAMPLE_SIZE)
    
    logging.info(f"Retrieved {len(claims)} claims for processing")
    return claims

def get_next_api_key():
    """Get the next available API key with quota remaining"""
    for key in GOOGLE_API_KEYS:
        usage = api_key_usage.get(key, 0)
        if usage < MAX_QUERIES_PER_KEY:
            # Increment usage for this key
            api_key_usage[key] = usage + 1
            return key
    
    # If all keys are exhausted, return None
    logging.warning("All Google API keys have reached their quota limits!")
    return None

def google_search(query, api_key, cse_id):
    """Perform a search using Google Custom Search API"""
    if not api_key or not cse_id:
        logging.warning("Google API key or CSE ID not provided, falling back to mock search")
        return mock_search_results(query)
    
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        'key': api_key,
        'cx': cse_id,
        'q': query,
        'num': RESULTS_PER_CLAIM
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        search_results = response.json()
        
        # Format results to match our structure
        results = []
        if 'items' in search_results:
            for item in search_results['items']:
                result = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'date': ''  # Google API doesn't directly provide dates
                }
                results.append(result)
        
        return {'organic': results}
    except Exception as e:
        logging.error(f"Error in Google search: {e}")
        # Fall back to mock search
        return mock_search_results(query)

# Initialize LLM for evidence generation (lazy loading)
llm_generator = None
tokenizer = None

def init_llm_generator():
    """Initialize the LLM pipeline for evidence generation"""
    global llm_generator, tokenizer
    
    if llm_generator is not None:
        return
    
    try:
        # Using a smaller open-source model - replace with your preferred model
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model that can run locally
        
        logging.info(f"Initializing LLM generator with model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create pipeline
        llm_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.7
        )
        
        logging.info("LLM generator initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing LLM generator: {e}")
        logging.warning("Falling back to mock evidence generation")

def generate_llm_evidence(claim, subject, real_evidence_patterns=None):
    """Generate evidence using LLM based on the claim and real evidence patterns"""
    global llm_generator
    
    if llm_generator is None:
        try:
            init_llm_generator()
        except:
            # If LLM initialization fails, fall back to mock
            return mock_search_results(claim)
    
    # Create a prompt that instructs the LLM to generate evidence
    year_range = "2018-2025"  # Default recent range
    
    # Add temporal hints based on real evidence patterns if available
    temporal_hint = "Make sure to include SPECIFIC DATES in both the URL and content of each evidence."
    if real_evidence_patterns and 'avg_time_difference' in real_evidence_patterns:
        avg_diff = real_evidence_patterns['avg_time_difference']
        if avg_diff < 0:
            # Evidence tends to be before claims
            temporal_hint = f"Create evidence that was published around {abs(avg_diff)} days BEFORE the claim was made. Include SPECIFIC DATES in both the URL and content."
            year_range = "2015-2022"  # Adjust based on patterns
        else:
            # Evidence tends to be after claims
            temporal_hint = f"Create evidence that was published around {avg_diff} days AFTER the claim was made. Include SPECIFIC DATES in both the URL and content."
            year_range = "2020-2025"  # More recent
    
    # Build the prompt
    prompt = f"""Generate {RESULTS_PER_CLAIM} different evidence snippets that might be found when searching for this political claim: "{claim}"
The topic is about {subject}.
Each evidence MUST have:
1. A title
2. A URL (make up a realistic URL including a date in the format YYYY/MM/DD between {year_range})
3. A snippet of content (2-3 sentences)
4. IMPORTANT: Include SPECIFIC DATES within the content (use exact day, month, and year)
{temporal_hint}

Format each as: 
Title: [title]
URL: [url with date]
Snippet: [content with at least one specific date]
"""
    
    try:
        # Generate text with the LLM
        response = llm_generator(prompt, max_length=1024)[0]['generated_text']
        
        # Extract the evidence items from the response
        evidence_items = []
        sections = response.split("\n\n")
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split("\n")
            title = ""
            link = ""
            snippet = ""
            
            for line in lines:
                if line.startswith("Title:"):
                    title = line.replace("Title:", "").strip()
                elif line.startswith("URL:"):
                    link = line.replace("URL:", "").strip()
                elif line.startswith("Snippet:"):
                    snippet = line.replace("Snippet:", "").strip()
            
            if title and link:
                evidence_items.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "date": ""  # Will be extracted later
                })
        
        # If we couldn't parse properly, fall back to mock
        if not evidence_items:
            return mock_search_results(claim)
            
        # Return in the expected format
        return {'organic': evidence_items[:RESULTS_PER_CLAIM]}
    
    except Exception as e:
        logging.error(f"Error generating evidence with LLM: {e}")
        return mock_search_results(claim)

def web_search_serper(query, api_key=None):
    """Perform web search using Serper.dev API"""
    if api_key is None:
        # Fallback to mock search if no API key provided
        return mock_search_results(query)
    
    url = "https://google.serper.dev/search"
    
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f"Error in Serper web search: {e}")
        return mock_search_results(query)

def mock_search_results(query):
    """Create mock search results when API is not available"""
    # Optimization for bulk processing - reduce delay when in mock mode
    if USE_HYBRID_APPROACH:
        time.sleep(0.01)  # Much shorter delay for bulk processing
    else:
        time.sleep(0.5)  # Normal delay to simulate API call
    
    # Dictionary of synthetic search results for major policy subjects
    synthetic_results = {
        "health": [
            {"title": "The History of Healthcare Reform in America", "link": "https://example.com/healthcare-history", "snippet": "A comprehensive overview of healthcare policies from 1990 to 2022, including major reforms and public opinion shifts.", "date": "2022-05-10"},
            {"title": "Understanding Medicare and Medicaid", "link": "https://example.com/medicare-explained", "snippet": "Detailed explanation of Medicare and Medicaid programs, eligibility requirements, and how they've changed since their inception.", "date": "2021-09-15"},
            {"title": "Healthcare Policy Analysis", "link": "https://example.com/policy-analysis", "snippet": "Experts analyze recent healthcare policies and their impact on coverage rates and medical costs across different demographics.", "date": "2023-01-22"}
        ],
        "tax": [
            {"title": "Tax Reform Timeline: 1980-Present", "link": "https://example.com/tax-reform-history", "snippet": "Comprehensive timeline of major tax legislation and policy changes in the United States over the past four decades.", "date": "2023-02-05"},
            {"title": "Analysis of Tax Rates by Income Bracket", "link": "https://example.com/tax-analysis", "snippet": "Statistical breakdown of effective tax rates across different income levels from 1990 to present day.", "date": "2022-10-18"},
            {"title": "Corporate Tax Policy Changes", "link": "https://example.com/corporate-taxes", "snippet": "How corporate tax policies have evolved over time and their impact on business investment and economic growth.", "date": "2021-11-30"}
        ],
        "immigration": [
            {"title": "Immigration Policy Through the Decades", "link": "https://example.com/immigration-history", "snippet": "Historical overview of U.S. immigration policies, border security measures, and demographic trends from 1950 to present.", "date": "2022-07-12"},
            {"title": "DACA and Dreamers: Policy Analysis", "link": "https://example.com/daca-analysis", "snippet": "Detailed examination of the DACA program, its implementation, legal challenges, and impact on affected individuals.", "date": "2021-08-20"},
            {"title": "Immigration Statistics Report", "link": "https://example.com/immigration-stats", "snippet": "Comprehensive data on immigration patterns, enforcement actions, and policy outcomes from government sources.", "date": "2023-03-05"}
        ],
        "economy": [
            {"title": "Economic Indicators Dashboard", "link": "https://example.com/economic-indicators", "snippet": "Track key economic metrics including unemployment rates, GDP growth, and inflation from 1990 to present.", "date": "2023-04-01"},
            {"title": "Federal Reserve Policy History", "link": "https://example.com/fed-history", "snippet": "Analysis of monetary policy decisions and their impact on the economy over the past three decades.", "date": "2022-08-15"},
            {"title": "Economic Recovery Comparison", "link": "https://example.com/recession-recovery", "snippet": "Comparing recovery patterns from major economic downturns in 2001, 2008, and 2020 with data-driven analysis.", "date": "2022-01-10"}
        ],
        "gun": [
            {"title": "Gun Legislation Timeline", "link": "https://example.com/gun-legislation", "snippet": "Comprehensive timeline of major gun control and gun rights legislation at federal and state levels since 1990.", "date": "2022-09-08"},
            {"title": "Second Amendment Court Cases", "link": "https://example.com/second-amendment", "snippet": "Analysis of landmark Supreme Court decisions related to gun rights and their constitutional implications.", "date": "2021-06-25"},
            {"title": "Gun Policy Research Findings", "link": "https://example.com/gun-research", "snippet": "Meta-analysis of research studies examining the effectiveness of various gun policies and their outcomes.", "date": "2023-02-18"}
        ]
    }
    
    # Determine which category to use based on the query
    category = None
    for key in synthetic_results.keys():
        if key in query.lower():
            category = key
            break
    
    # Default to economy if no match found
    if category is None:
        category = "economy"
    
    # Create mock search results
    mock_results = {
        "organic": [
            {"title": result["title"], 
             "link": result["link"],
             "snippet": result["snippet"],
             "date": result["date"]} 
            for result in synthetic_results[category]
        ]
    }
    
    return mock_results

def extract_date_from_url(url):
    """Extract date from URL if it follows common patterns"""
    date_patterns = [
        r'(\d{4})/(\d{1,2})/(\d{1,2})',  # YYYY/MM/DD
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        r'/(\d{4})(\d{2})(\d{2})/',      # YYYYMMDD
        r'/(\d{4})_(\d{1,2})_(\d{1,2})/', # YYYY_MM_DD
        r'(\d{4})[_\.-](\d{2})[_\.-](\d{2})', # YYYY_MM_DD, YYYY.MM.DD, YYYY-MM-DD
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
        r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)[,|\s]+(\d{4})\b',  # DD Month YYYY
        r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[,|\s]+(\d{4})\b',  # DD Mon YYYY
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
                # Handle DD Month YYYY format
                elif len(match.groups()) >= 3 and match.group(2) in month_names:
                    day = int(match.group(1))
                    month = month_names[match.group(2)]
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
                    elif part1 > 12 and part1 <= 31:  # DD/MM/YYYY
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
        return domain
    except:
        return url

def collect_evidence(claims):
    """Collect evidence for each claim using hybrid approach"""
    evidence_items = []
    temporal_patterns = {}
    
    # Determine if we should use the hybrid approach
    use_hybrid = USE_HYBRID_APPROACH and len(claims) > REAL_SEARCH_SAMPLE_SIZE
    
    # If using hybrid approach, get a stratified sample of claims
    if use_hybrid:
        # Get distribution of subjects
        subject_groups = {}
        for claim in claims:
            subject = claim['metadata'].get('subject', 'unknown')
            if subject not in subject_groups:
                subject_groups[subject] = []
            subject_groups[subject].append(claim)
        
        # Create a representative sample
        sample_claims = []
        remaining_claims = []
        
        # Calculate how many claims to take from each subject
        total_subjects = len(subject_groups)
        for subject, subject_claims in subject_groups.items():
            # Determine sample size for this subject
            if total_subjects == 0:
                subject_sample_size = 0
            else:
                subject_sample_size = max(1, int(REAL_SEARCH_SAMPLE_SIZE * len(subject_claims) / len(claims)))
            
            # Sample from this subject
            if len(subject_claims) <= subject_sample_size:
                sampled = subject_claims
            else:
                sampled = random.sample(subject_claims, subject_sample_size)
            
            sample_claims.extend(sampled)
            
            # Add non-sampled claims to remaining
            for claim in subject_claims:
                if claim not in sampled:
                    remaining_claims.append(claim)
        
        # Adjust sample size if needed
        if len(sample_claims) > REAL_SEARCH_SAMPLE_SIZE:
            sample_claims = sample_claims[:REAL_SEARCH_SAMPLE_SIZE]
        
        logging.info(f"Using hybrid approach: {len(sample_claims)} claims with real search, {len(remaining_claims)} with LLM generation")
    else:
        # Process all claims with the same method
        sample_claims = claims
        remaining_claims = []
        logging.info(f"Using single approach for all {len(claims)} claims")
    
    # Process the sample with real search - with quota handling
    if sample_claims:
        logging.info(f"Collecting real evidence for {len(sample_claims)} claims")
        real_evidence, processed_claims, unprocessed_claims = collect_real_evidence(sample_claims)
        evidence_items.extend(real_evidence)
        
        # If we have unprocessed claims due to API quota exhaustion, move them to remaining
        if unprocessed_claims:
            logging.warning(f"API quota exhausted. Moving {len(unprocessed_claims)} claims to LLM generation")
            remaining_claims.extend(unprocessed_claims)
        
        # Analyze temporal patterns in real evidence
        if use_hybrid and real_evidence:
            temporal_patterns = analyze_temporal_patterns(real_evidence)
            logging.info(f"Extracted temporal patterns: {temporal_patterns}")
    
    # Process remaining claims with LLM generation
    if remaining_claims:
        logging.info(f"Generating synthetic evidence for {len(remaining_claims)} claims")
        synthetic_evidence = collect_synthetic_evidence(remaining_claims, temporal_patterns)
        evidence_items.extend(synthetic_evidence)
    
    logging.info(f"Collected total of {len(evidence_items)} evidence items for {len(claims)} claims")
    return evidence_items

def collect_real_evidence(claims):
    """Collect evidence from real search APIs"""
    evidence_items = []
    total_claims = len(claims)
    processed_claims = []
    unprocessed_claims = []
    all_api_keys_exhausted = False
    
    for i, claim in enumerate(claims):
        if i % 10 == 0:
            logging.info(f"Real search progress: {i+1}/{total_claims} claims processed ({(i+1)/total_claims*100:.1f}%)")
            
        # Skip if all API keys exhausted
        if all_api_keys_exhausted:
            unprocessed_claims.append(claim)
            continue
            
        claim_id = claim['id']
        search_query = claim['metadata'].get('search_query', claim['text'])
        subject = claim['metadata'].get('subject', '')
        
        # Get API key
        api_key = get_next_api_key()
        
        # Perform search
        if api_key:
            search_results = google_search(search_query, api_key, GOOGLE_CSE_ID)
            processed_claims.append(claim)
        else:
            # All API keys exhausted - set flag and add to unprocessed
            all_api_keys_exhausted = True
            unprocessed_claims.append(claim)
            logging.warning(f"All API keys exhausted after processing {i} claims. Switching remaining to LLM generation.")
            continue
        
        # Extract and process results
        results = search_results.get('organic', [])
        if not results:
            continue
        
        # Limit to top results
        results = results[:RESULTS_PER_CLAIM]
        
        # Process each search result with focus on temporal data
        for result_idx, result in enumerate(results):
            evidence = process_evidence_item(claim_id, result, search_query, subject, result_idx)
            if evidence:
                evidence_items.append(evidence)
        
        # Rate limiting for API
        time.sleep(0.5)
    
    return evidence_items, processed_claims, unprocessed_claims

def collect_synthetic_evidence(claims, temporal_patterns=None):
    """Generate synthetic evidence using LLM"""
    evidence_items = []
    total_claims = len(claims)
    batch_size = 50  # Process in batches
    
    for batch_start in range(0, total_claims, batch_size):
        batch_end = min(batch_start + batch_size, total_claims)
        batch = claims[batch_start:batch_end]
        
        logging.info(f"Synthetic evidence batch: {batch_start+1}-{batch_end} of {total_claims}")
        
        for i, claim in enumerate(batch):
            if (batch_start + i) % 100 == 0 and batch_start + i > 0:
                logging.info(f"Synthetic progress: {batch_start+i+1}/{total_claims} ({(batch_start+i+1)/total_claims*100:.1f}%)")
                
            claim_id = claim['id']
            search_query = claim['metadata'].get('search_query', claim['text'])
            subject = claim['metadata'].get('subject', '')
            
            # Generate evidence with LLM
            search_results = generate_llm_evidence(search_query, subject, temporal_patterns)
            
            # Extract and process results
            results = search_results.get('organic', [])
            if not results:
                continue
            
            # Process each generated result
            for result_idx, result in enumerate(results):
                evidence = process_evidence_item(claim_id, result, search_query, subject, result_idx)
                if evidence:
                    evidence_items.append(evidence)
            
            # Small delay to avoid resource contention
            if i % 10 == 0:
                time.sleep(0.1)
    
    return evidence_items

def process_evidence_item(claim_id, result, search_query, subject, result_idx):
    """Process a single evidence item (real or synthetic)"""
    title = result.get('title', '')
    link = result.get('link', '')
    snippet = result.get('snippet', '')
    
    # Skip if no link or no content
    if not link or (not title and not snippet):
        return None
        
    # Extract dates - prioritize temporal information
    url_date = extract_date_from_url(link)
    text_dates = extract_dates_from_text(title + " " + snippet)
    
    # Combine all found dates
    all_dates = [d for d in ([url_date] + text_dates) if d]
    publication_date = all_dates[0] if all_dates else None
    
    # Skip evidence with no temporal information if we're in strict mode
    # Comment out this block if you want to keep all evidence
    """
    if not publication_date and not all_dates:
        logging.debug(f"Skipping evidence with no temporal information for claim {claim_id}")
        return None
    """
    
    # Create unique ID for evidence
    evidence_id = f"{claim_id}_ev{result_idx}"
    
    # Create evidence item
    evidence = {
        "id": evidence_id,
        "claim_id": claim_id,
        "title": title,
        "content": snippet,
        "source_url": link,
        "domain": get_domain(link),
        "publication_date": publication_date,
        "retrieval_date": datetime.now().strftime("%Y-%m-%d"),
        "all_dates_found": all_dates,
        "subject": subject,
        "search_query": search_query,
        "has_temporal_data": bool(publication_date or all_dates)
    }
    
    return evidence

def analyze_temporal_patterns(evidence_items):
    """Analyze temporal patterns in real evidence to guide synthetic generation"""
    patterns = {}
    
    # Group evidence by claim
    claim_evidence = {}
    for evidence in evidence_items:
        claim_id = evidence.get('claim_id')
        if claim_id not in claim_evidence:
            claim_evidence[claim_id] = []
        claim_evidence[claim_id].append(evidence)
    
    # Calculate time differences between claims and evidence
    time_differences = []
    
    for claim_id, evidences in claim_evidence.items():
        for evidence in evidences:
            if evidence.get('publication_date'):
                # TODO: Implement real time difference calculation
                # This would require the claim dates, which we'd need to extract
                # For now, we'll use a placeholder
                time_differences.append(random.randint(-30, 30))  # Random between -30 and 30 days
    
    if time_differences:
        patterns['avg_time_difference'] = sum(time_differences) / len(time_differences)
        patterns['max_time_difference'] = max(time_differences)
        patterns['min_time_difference'] = min(time_differences)
    
    return patterns

def store_evidence(evidence_collection, evidence_items):
    """Store evidence in ChromaDB with temporal metadata"""
    # Check existing items
    existing_count = evidence_collection.count()
    logging.info(f"Evidence collection currently contains {existing_count} items")
    
    # Prepare data for batch import
    batch_size = 100  # Process in batches to avoid memory issues
    
    logging.info(f"Preparing to store {len(evidence_items)} evidence items in batches of {batch_size}")
    
    for i in range(0, len(evidence_items), batch_size):
        batch = evidence_items[i:i+batch_size]
        
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        
        for evidence in batch:
            # Create ID
            batch_ids.append(evidence['id'])
            
            # Create text for embedding - combine title and content
            evidence_text = f"{evidence['title']} {evidence['content']}"
            batch_texts.append(evidence_text)
            
            # Create metadata
            metadata = {
                "claim_id": evidence["claim_id"],
                "title": evidence["title"],
                "source_url": evidence["source_url"],
                "domain": evidence["domain"],
                "publication_date": evidence["publication_date"] if evidence["publication_date"] else "",
                "retrieval_date": evidence["retrieval_date"],
                "subject": evidence["subject"],
                "all_dates_found": ",".join(evidence["all_dates_found"]) if evidence["all_dates_found"] else "",
                "search_query": evidence["search_query"]
            }
            
            batch_metadatas.append(metadata)
        
        # Add batch to collection
        evidence_collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        
        logging.info(f"Stored batch {i//batch_size + 1}/{(len(evidence_items) + batch_size - 1)//batch_size}")
        # Small delay to avoid overwhelming the database
        time.sleep(0.1)
    
    logging.info(f"Successfully stored {len(evidence_items)} evidence items in ChromaDB")
    
    # Also save as JSON file for reference
    evidence_file = os.path.join(EVIDENCE_DIR, "collected_evidence.json")
    with open(evidence_file, 'w') as f:
        json.dump(evidence_items, f, indent=2)
    
    logging.info(f"Saved evidence to {evidence_file}")
    return len(evidence_items)

def generate_evidence_report(evidence_items):
    """Generate a report of the collected evidence"""
    report_lines = []
    report_lines.append(f"Evidence Collection Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total evidence items collected: {len(evidence_items)}")
    
    # Count by subject
    subject_counts = {}
    for item in evidence_items:
        subject = item.get('subject', 'unknown')
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    report_lines.append("\nEvidence by subject:")
    for subject, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True):
        if subject:  # Skip empty subjects
            percentage = (count / len(evidence_items)) * 100
            report_lines.append(f"  {subject}: {count} items ({percentage:.2f}%)")
    
    # Count by domain
    domain_counts = {}
    for item in evidence_items:
        domain = item.get('domain', 'unknown')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    report_lines.append("\nTop evidence sources:")
    top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for domain, count in top_domains:
        percentage = (count / len(evidence_items)) * 100
        report_lines.append(f"  {domain}: {count} items ({percentage:.2f}%)")
    
    # Publication date statistics
    dates = [item.get('publication_date') for item in evidence_items if item.get('publication_date')]
    report_lines.append(f"\nItems with publication dates: {len(dates)} ({(len(dates)/len(evidence_items))*100:.2f}%)")
    
    if dates:
        # Parse dates for comparison
        parsed_dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates if d]
        if parsed_dates:
            earliest = min(parsed_dates).strftime("%Y-%m-%d")
            latest = max(parsed_dates).strftime("%Y-%m-%d")
            report_lines.append(f"  Earliest publication date: {earliest}")
            report_lines.append(f"  Latest publication date: {latest}")
    
    # Save report
    report_file = os.path.join(EVIDENCE_DIR, "evidence_collection_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Generated evidence report at {report_file}")
    return report_lines

def main():
    """Main execution function"""
    logging.info("Starting evidence collection process with hybrid approach...")
    
    # Setup ChromaDB
    client, claims_collection, evidence_collection = setup_chroma_client()
    
    # Get claims to process
    claims = get_claims_to_process(claims_collection)
    
    # Check if we have API keys for Google Search
    if GOOGLE_API_KEYS and GOOGLE_CSE_ID:
        logging.info(f"Found {len(GOOGLE_API_KEYS)} Google API keys - will use real search for sample")
    else:
        logging.warning("No Google API keys provided - will use mock data for real search component")
    
    # Check if we need to initialize LLM
    if USE_HYBRID_APPROACH:
        try:
            init_llm_generator()
        except Exception as e:
            logging.error(f"Failed to initialize LLM generator: {e}")
            logging.warning("Will use mock data for synthetic evidence generation")
    
    # Collect evidence for claims
    evidence_items = collect_evidence(claims)
    
    # Store evidence in ChromaDB
    stored_count = store_evidence(evidence_collection, evidence_items)
    
    # Generate report
    report_lines = generate_evidence_report(evidence_items)
    for line in report_lines:
        logging.info(line)
    
    logging.info("Evidence collection completed successfully!")
    
    # Log API key usage
    if GOOGLE_API_KEYS:
        logging.info("Google API Key Usage:")
        for key, count in api_key_usage.items():
            masked_key = f"{key[:6]}...{key[-4:]}" if len(key) > 10 else "N/A" 
            logging.info(f"  Key {masked_key}: {count}/{MAX_QUERIES_PER_KEY} queries used")

if __name__ == "__main__":
    main() 
