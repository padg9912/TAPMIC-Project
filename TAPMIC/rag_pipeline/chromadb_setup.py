#!/usr/bin/env python3
# chromadb_setup.py
# Sets up ChromaDB for storing claims and evidence with temporal metadata

import chromadb
from chromadb.utils import embedding_functions
import json
import os
import pandas as pd
from datetime import datetime
import time
import random

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS_DIR = os.path.join(BASE_DIR, "rag_pipeline", "claims_data")
DB_DIR = os.path.join(BASE_DIR, "rag_pipeline", "chroma_db")
os.makedirs(DB_DIR, exist_ok=True)

# Input files
CLAIMS_FILE = os.path.join(CLAIMS_DIR, "political_claims.json")

# Configuration
CLAIMS_COLLECTION_NAME = "political_claims"
EVIDENCE_COLLECTION_NAME = "temporal_evidence"

def setup_chroma_client():
    """Initialize and configure the ChromaDB client"""
    print(f"Initializing ChromaDB client with persistence at: {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Check if using a pre-existing DB or creating a new one
    collections = client.list_collections()
    collection_names = [c.name for c in collections]
    
    if CLAIMS_COLLECTION_NAME in collection_names:
        print(f"Found existing '{CLAIMS_COLLECTION_NAME}' collection")
    
    if EVIDENCE_COLLECTION_NAME in collection_names:
        print(f"Found existing '{EVIDENCE_COLLECTION_NAME}' collection")
    
    return client

def create_collections(client):
    """Create or get collections for claims and evidence"""
    # Use sentence transformers for embedding
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or get claims collection
    try:
        claims_collection = client.get_collection(
            name=CLAIMS_COLLECTION_NAME,
            embedding_function=sentence_transformer_ef
        )
        print(f"Retrieved existing collection: {CLAIMS_COLLECTION_NAME}")
    except:
        claims_collection = client.create_collection(
            name=CLAIMS_COLLECTION_NAME,
            embedding_function=sentence_transformer_ef,
            metadata={"description": "Political claims with speaker and context information"}
        )
        print(f"Created new collection: {CLAIMS_COLLECTION_NAME}")
    
    # Create or get evidence collection
    try:
        evidence_collection = client.get_collection(
            name=EVIDENCE_COLLECTION_NAME,
            embedding_function=sentence_transformer_ef
        )
        print(f"Retrieved existing collection: {EVIDENCE_COLLECTION_NAME}")
    except:
        evidence_collection = client.create_collection(
            name=EVIDENCE_COLLECTION_NAME,
            embedding_function=sentence_transformer_ef,
            metadata={"description": "Evidence for political claims with temporal information"}
        )
        print(f"Created new collection: {EVIDENCE_COLLECTION_NAME}")
    
    return claims_collection, evidence_collection

def load_claims():
    """Load extracted political claims"""
    try:
        with open(CLAIMS_FILE, 'r') as f:
            claims = json.load(f)
        print(f"Loaded {len(claims)} claims from {CLAIMS_FILE}")
        return claims
    except Exception as e:
        print(f"Error loading claims: {e}")
        return []

def store_claims(claims_collection, claims):
    """Store claims in ChromaDB with appropriate metadata"""
    # Check if collection already has documents
    existing_count = claims_collection.count()
    if existing_count > 0:
        print(f"Collection already contains {existing_count} claims. Skipping import.")
        return
    
    # Prepare data for batch import
    ids = []
    texts = []
    metadatas = []
    
    batch_size = 500  # Process in batches to avoid memory issues
    
    print(f"Preparing to store {len(claims)} claims in batches of {batch_size}")
    
    for i in range(0, len(claims), batch_size):
        batch = claims[i:i+batch_size]
        
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        
        for claim in batch:
            # Create ID
            claim_id = claim['id']
            batch_ids.append(str(claim_id))
            
            # Create text for embedding
            claim_text = claim['text']
            if claim['speaker']:
                claim_text = f"{claim['speaker']}: {claim_text}"
            batch_texts.append(claim_text)
            
            # Create metadata
            metadata = {
                "speaker": claim["speaker"],
                "date": claim["date"],
                "subject": claim["subject"],
                "context": claim["context"],
                "truth_label": claim["truth_label"],
                "original_label": claim["original_label"],
                "source": claim["source"],
                "keywords": ", ".join(claim["keywords"]) if "keywords" in claim else "",
                "search_query": claim["search_query"] if "search_query" in claim else "",
            }
            
            # Add credibility score if available
            if "credibility_score" in claim and claim["credibility_score"] is not None:
                metadata["credibility_score"] = float(claim["credibility_score"])
            
            batch_metadatas.append(metadata)
        
        # Add batch to collection
        claims_collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        
        print(f"Stored batch {i//batch_size + 1}/{(len(claims) + batch_size - 1)//batch_size}")
        # Small delay to avoid overwhelming the database
        time.sleep(0.1)
    
    print(f"Successfully stored {len(claims)} claims in ChromaDB")

def setup_evidence_schema():
    """Document the evidence schema for the RAG pipeline"""
    schema = {
        "id": "Unique identifier for the evidence",
        "claim_id": "ID of the claim this evidence relates to",
        "content": "Text content of the evidence",
        "source_url": "URL source of the evidence",
        "publication_date": "Date when the evidence was published",
        "retrieval_date": "Date when the evidence was retrieved",
        "context_before_date": "Dates mentioned before the claim date",
        "context_after_date": "Dates mentioned after the claim date",
        "time_difference": "Time difference between claim and evidence in days",
        "temporal_relevance": "Score indicating temporal relevance to the claim",
        "source_reliability": "Score indicating reliability of the source",
        "search_query": "Query used to retrieve this evidence"
    }
    
    # Save schema for reference
    schema_file = os.path.join(CLAIMS_DIR, "evidence_schema.json")
    with open(schema_file, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"Evidence schema documentation saved to {schema_file}")
    return schema

def test_query(claims_collection, evidence_collection):
    """Run test queries to verify collections are working"""
    print("\nTesting claim queries:")
    
    # Query by subject
    results = claims_collection.query(
        query_texts=["healthcare policy"],
        n_results=3,
        where={"subject": "health-care"}
    )
    
    print(f"Found {len(results['ids'][0])} health-care related claims")
    
    # Query by speaker 
    results = claims_collection.query(
        query_texts=["taxes economy"],
        n_results=3,
        where={"speaker": {"$ne": ""}}
    )
    
    print(f"Found {len(results['ids'][0])} claims with speaker information about taxes/economy")
    
    # Test evidence collection if it has data
    if evidence_collection.count() > 0:
        print("\nTesting evidence queries:")
        
        results = evidence_collection.query(
            query_texts=["recent evidence"],
            n_results=3
        )
        
        print(f"Found {len(results['ids'][0])} evidence entries")

def main():
    """Main execution function"""
    print("Starting ChromaDB setup for TAPMIC project...")
    
    # Setup ChromaDB client
    client = setup_chroma_client()
    
    # Create collections
    claims_collection, evidence_collection = create_collections(client)
    
    # Load and store claims
    claims = load_claims()
    if claims:
        store_claims(claims_collection, claims)
    
    # Setup evidence schema
    evidence_schema = setup_evidence_schema()
    
    # Test query operations
    test_query(claims_collection, evidence_collection)
    
    print("\nChromaDB setup completed successfully!")
    print(f"Claims collection: {claims_collection.count()} documents")
    print(f"Evidence collection: {evidence_collection.count()} documents")
    print(f"Database location: {DB_DIR}")

if __name__ == "__main__":
    main() 