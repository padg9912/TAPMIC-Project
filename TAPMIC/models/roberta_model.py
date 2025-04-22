#!/usr/bin/env python3
# roberta_model.py
# RoBERTa model implementation for Temporally Aware Political Misinformation Classification (TAPMIC)

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaPreTrainedModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Define file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "intermediate_2")
MODELS_DIR = os.path.join(BASE_DIR, "models", "checkpoints")
os.makedirs(MODELS_DIR, exist_ok=True)

# Define constants
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.1
EARLY_STOPPING_PATIENCE = 3
GRAD_ACCUMULATION_STEPS = 2

class ClaimDataset(Dataset):
    """Dataset for claims with text and numerical features"""
    
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH, use_temporal=True, use_credibility=True):
        """
        Initialize dataset
        
        Args:
            data (DataFrame): Dataframe containing claims and features
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            use_temporal: Whether to use temporal features
            use_credibility: Whether to use speaker credibility features
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_temporal = use_temporal
        self.use_credibility = use_credibility
        
        # Identify categorical and numerical columns
        self.text_col = 'processed_text'  # The processed claim text from Phase 1
        
        # Identify temporal features (from Phase 2)
        self.temporal_cols = [
            'evidence_count', 'temporal_evidence_count', 'google_evidence_count', 
            'perplexity_evidence_count', 'multiple_dates_percentage', 
            'mean_publication_year', 'publication_date_range_days',
            'days_between_claim_and_earliest_evidence', 'days_between_claim_and_latest_evidence',
            'temporal_consistency_score', 'temporal_claim_percentage', 
            'speaker_30day_true_ratio', 'speaker_90day_true_ratio', 
            'speaker_180day_true_ratio', 'speaker_30day_claim_count',
            'speaker_90day_claim_count', 'speaker_180day_claim_count',
            'speaker_evidence_consistency', 'election_year_flag',
            'campaign_season_flag', 'days_to_nearest_election'
        ]
        
        # Identify speaker credibility features (from Phase 1)
        self.credibility_cols = [
            'speaker_credibility_score', 'weighted_credibility_score',
            'count_true', 'count_false', 'count_half_true', 
            'count_mostly_true', 'count_barely_true', 'count_pants_fire'
        ]
        
        # Prepare feature list based on flags
        self.feature_cols = []
        if self.use_temporal:
            self.feature_cols.extend(self.temporal_cols)
        if self.use_credibility:
            self.feature_cols.extend(self.credibility_cols)
        
        # Normalize numerical features
        if len(self.feature_cols) > 0:
            self.scaler = StandardScaler()
            # Fill NaN values with 0 and scale
            features = self.data[self.feature_cols].fillna(0)
            self.scaled_features = self.scaler.fit_transform(features)
        else:
            self.scaled_features = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get text data
        text = self.data.iloc[idx][self.text_col]
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Create item dictionary
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
        }
        
        # Add numerical features if using them
        if len(self.feature_cols) > 0:
            features = torch.tensor(self.scaled_features[idx], dtype=torch.float)
            item['features'] = features
        
        return item

class FactCheckerModel(RobertaPreTrainedModel):
    """
    RoBERTa-based model for fact checking
    
    Uses RoBERTa as the base model with optional temporal features
    """
    
    def __init__(self, config):
        """
        Initialize the model
        
        Args:
            config: RoBERTa config with additional parameters
        """
        super().__init__(config)
        
        # Set up RoBERTa base model
        self.roberta = RobertaModel(config)
        
        # Get hidden size
        hidden_size = config.hidden_size
        
        # Check if we're using temporal features
        self.use_temporal = getattr(config, 'use_temporal', False)
        
        # If temporal features are enabled, modify architecture
        if self.use_temporal:
            self.temporal_dim = getattr(config, 'temporal_dim', 3)  # Default is 3 features
            
            # Set up layers for temporal feature processing
            self.temporal_encoder = nn.Sequential(
                nn.Linear(self.temporal_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128)
            )
            
            # Classification head with combined features
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size + 128, hidden_size // 2),
                nn.Dropout(config.hidden_dropout_prob),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)  # Binary classification (TRUE/FALSE)
            )
        else:
            # Standard classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Dropout(config.hidden_dropout_prob),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 2)  # Binary classification (TRUE/FALSE)
            )
        
        # Initialize weights
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, temporal_features=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            token_type_ids: Token type IDs
            temporal_features: Optional temporal features tensor
            
        Returns:
            logits: Output logits (2 classes)
        """
        # Get RoBERTa features
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # If using temporal features, process and combine
        if self.use_temporal and temporal_features is not None:
            # Process temporal features
            temporal_encoded = self.temporal_encoder(temporal_features)
            
            # Concatenate with RoBERTa features
            combined_features = torch.cat([cls_output, temporal_encoded], dim=1)
            
            # Pass through classifier
            logits = self.classifier(combined_features)
        else:
            # Use only RoBERTa features
            logits = self.classifier(cls_output)
        
        return logits

class TemporalFeatureExtractor:
    """
    Extracts and normalizes temporal features from evidence.
    """
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or [
            "days_since_earliest_evidence",
            "days_since_latest_evidence",
            "evidence_date_range",
            "avg_evidence_recency",
            "temporal_consistency"
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract_features(self, evidence_items):
        """
        Extract temporal features from a list of evidence items.
        
        Args:
            evidence_items: List of evidence dictionaries
            
        Returns:
            features: Tensor of normalized temporal features
        """
        # Implementation would extract dates from evidence
        # and compute temporal features
        features = torch.zeros(1, self.feature_dim)
        # This would be replaced with actual feature extraction logic
        return features
    
class SpeakerFeatureExtractor:
    """
    Extracts and normalizes speaker credibility features.
    """
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or [
            "speaker_reliability_score",
            "speaker_expertise_level",
            "prior_false_claims_count",
            "speaker_consistency"
        ]
        self.feature_dim = len(self.feature_names)
    
    def extract_features(self, speaker_data):
        """
        Extract speaker credibility features.
        
        Args:
            speaker_data: Dictionary containing speaker information
            
        Returns:
            features: Tensor of normalized speaker features
        """
        # Implementation would extract credibility features
        # from speaker data
        features = torch.zeros(1, self.feature_dim)
        # This would be replaced with actual feature extraction logic
        return features

def load_data():
    """Load training, validation, and test data"""
    try:
        train = pd.read_csv(os.path.join(DATA_DIR, "intermediate2_train.csv"))
        val = pd.read_csv(os.path.join(DATA_DIR, "intermediate2_validation.csv"))
        test = pd.read_csv(os.path.join(DATA_DIR, "intermediate2_test.csv"))
        
        logging.info(f"Loaded {len(train)} training samples")
        logging.info(f"Loaded {len(val)} validation samples")
        logging.info(f"Loaded {len(test)} test samples")
        
        return train, val, test
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=BATCH_SIZE, 
                       use_temporal=True, use_credibility=True):
    """Create DataLoader objects for training, validation, and testing"""
    
    # Create datasets
    train_dataset = ClaimDataset(train_df, tokenizer, use_temporal=use_temporal, 
                                use_credibility=use_credibility)
    val_dataset = ClaimDataset(val_df, tokenizer, use_temporal=use_temporal, 
                              use_credibility=use_credibility)
    test_dataset = ClaimDataset(test_df, tokenizer, use_temporal=use_temporal, 
                               use_credibility=use_credibility)
    
    # Calculate number of features
    num_features = 0
    if use_temporal:
        num_features += len(train_dataset.temporal_cols)
    if use_credibility:
        num_features += len(train_dataset.credibility_cols)
    
    logging.info(f"Using {num_features} numerical features")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader, num_features

def get_optimizer_and_scheduler(model, train_dataloader, epochs=EPOCHS, lr=LEARNING_RATE,
                              warmup_ratio=WARMUP_RATIO, weight_decay=WEIGHT_DECAY):
    """Configure optimizer and learning rate scheduler"""
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * epochs // GRAD_ACCUMULATION_STEPS
    
    # Calculate warmup steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Group parameters for optimization
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    # Create optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, 
               epochs=EPOCHS, grad_accumulation_steps=GRAD_ACCUMULATION_STEPS,
               early_stopping_patience=EARLY_STOPPING_PATIENCE):
    """Train the model with early stopping"""
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass with or without additional features
            if 'features' in batch:
                features = batch['features'].to(device)
                outputs = model(input_ids, attention_mask, features)
            else:
                outputs = model(input_ids, attention_mask)
                
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every grad_accumulation_steps
            if (step + 1) % grad_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Track loss
            train_loss += loss.item() * grad_accumulation_steps
            train_steps += 1
            
            # Log batch metrics
            if step % 100 == 0:
                logging.info(f"Batch {step}/{len(train_loader)} - Loss: {loss.item() * grad_accumulation_steps:.4f}")
                if wandb.run is not None:
                    wandb.log({"batch_loss": loss.item() * grad_accumulation_steps})
        
        # Calculate average training loss
        avg_train_loss = train_loss / train_steps
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass with or without additional features
                if 'features' in batch:
                    features = batch['features'].to(device)
                    outputs = model(input_ids, attention_mask, features)
                else:
                    outputs = model(input_ids, attention_mask)
                
                # Calculate loss
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                
                # Track loss
                val_loss += loss.item()
                val_steps += 1
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / val_steps
        accuracy = correct / total
        
        # Log epoch metrics
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "accuracy": accuracy,
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            logging.info(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}")
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the model
            model_path = os.path.join(MODELS_DIR, f"roberta_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            logging.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logging.info("Early stopping triggered")
                break
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass with or without additional features
            if 'features' in batch:
                features = batch['features'].to(device)
                outputs = model(input_ids, attention_mask, features)
            else:
                outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            # Track loss
            test_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    
    # Log results
    logging.info(f"Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {accuracy:.4f}")
    
    if wandb.run is not None:
        wandb.log({
            "test_loss": test_loss/len(test_loader),
            "test_accuracy": accuracy
        })
    
    return predictions, true_labels, accuracy

def create_model_config(
    base_model_name="roberta-base",
    use_temporal=True,
    temporal_dim=3,
    num_labels=2
):
    """
    Create a configuration for the FactCheckerModel
    
    Args:
        base_model_name: Base model name for RoBERTa
        use_temporal: Whether to use temporal features
        temporal_dim: Dimension of temporal features
        num_labels: Number of output labels
        
    Returns:
        config: Model configuration
    """
    # Load base RoBERTa config
    config = RobertaConfig.from_pretrained(base_model_name)
    
    # Add custom parameters
    config.use_temporal = use_temporal
    config.temporal_dim = temporal_dim
    config.num_labels = num_labels
    
    return config

def load_pretrained_model(model_path, device=None):
    """
    Load a pretrained FactCheckerModel
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        model: Loaded model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = FactCheckerModel.from_pretrained(model_path)
    model.to(device)
    
    return model

def main():
    """Main function to execute the model training and evaluation"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Create configurations for the three runs
    run_configs = [
        {"name": "text_only", "use_temporal": False, "use_credibility": False},
        {"name": "text_temporal", "use_temporal": True, "use_credibility": False},
        {"name": "text_temporal_credibility", "use_temporal": True, "use_credibility": True}
    ]
    
    # Execute runs
    for config in run_configs:
        run_name = config["name"]
        use_temporal = config["use_temporal"]
        use_credibility = config["use_credibility"]
        
        logging.info(f"Starting run: {run_name}")
        logging.info(f"Using temporal features: {use_temporal}")
        logging.info(f"Using credibility features: {use_credibility}")
        
        # Initialize wandb
        wandb.init(
            project="TAPMIC", 
            name=run_name,
            config={
                "model_name": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "dropout_rate": DROPOUT_RATE,
                "use_temporal": use_temporal,
                "use_credibility": use_credibility
            }
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader, num_features = create_dataloaders(
            train_df, val_df, test_df, tokenizer, 
            use_temporal=use_temporal, 
            use_credibility=use_credibility
        )
        
        # Initialize model
        model = FactCheckerModel(
            config=RobertaConfig(
                use_temporal=use_temporal,
                temporal_feature_size=num_features
            )
        )
        model.to(device)
        
        # Get optimizer and scheduler
        optimizer, scheduler = get_optimizer_and_scheduler(model, train_loader)
        
        # Train model
        model = train_model(model, train_loader, val_loader, optimizer, scheduler, device)
        
        # Evaluate model
        predictions, true_labels, accuracy = evaluate_model(model, test_loader, device)
        
        # Log final results
        logging.info(f"Run {run_name} completed with test accuracy: {accuracy:.4f}")
        
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main() 