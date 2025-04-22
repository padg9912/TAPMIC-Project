# TAPMIC - RoBERTa Model Implementation

This directory contains the RoBERTa model implementation for the Temporally Aware Political Misinformation Classification (TAPMIC) project. The model leverages both text features and temporal features to classify political claims as TRUE or FALSE.

## Directory Structure

```
models/
├── checkpoints/        # Model checkpoints saved during training
├── reports/            # Evaluation reports and visualizations
├── roberta_model.py    # Core model implementation
├── data_utils.py       # Data preprocessing utilities
├── train.py            # Main training script
├── run_inference.py    # Inference script
├── evaluate.py         # Evaluation script
├── run_roberta.py      # Entry point for training with advanced features
├── train_utils.py      # Training utilities for advanced features
└── README.md           # This file
```

## Key Components

### Model Architecture (`roberta_model.py`)

The `FactCheckerModel` class provides a RoBERTa-based model with optional temporal feature integration:

- Extends `RobertaPreTrainedModel` from HuggingFace Transformers
- Processes claim text through RoBERTa
- Optionally processes temporal features through additional layers
- Combines text and temporal representations for final classification

### Data Processing (`data_utils.py`)

Handles data loading, preprocessing, and dataset creation:

- `FactCheckingDataset` class for PyTorch dataset implementation
- Temporal feature extraction from claim text
- Data loading utilities for various input formats
- Result formatting and saving

### Training (`train.py`)

Provides comprehensive training functionality:

- Command-line interface for training configuration
- Training and evaluation loops
- TensorBoard integration
- Model saving and checkpoint management

### Inference (`run_inference.py`)

Handles model inference on new claims:

- Loads trained models
- Processes claim texts
- Generates predictions with confidence scores
- Saves results in structured format

### Evaluation (`evaluate.py`)

Detailed model evaluation and analysis:

- Comprehensive metrics calculation
- Confusion matrix visualization
- Error analysis and reporting
- Feature importance analysis

### Advanced Training (`run_roberta.py` and `train_utils.py`)

Additional training functionality with advanced features:

- Integration with Weights & Biases for experiment tracking
- Mixed precision training
- Feature importance analysis
- Visualization utilities

## Model Features

The model can use two types of features:

1. **Text Features**: The processed claim text (required)
2. **Temporal Features**: Features extracted from claim text (optional)
   - Years (e.g., 2020, '98)
   - Month names (e.g., January, Feb)
   - Relative time words (e.g., yesterday, recently)

## Running the Model

### Basic Training

```bash
python train.py train \
  --train_data /path/to/train_data.json \
  --val_data /path/to/val_data.json \
  --output_dir ./model_output \
  --base_model_name roberta-base \
  --use_temporal
```

### Model Inference

```bash
python train.py predict \
  --model_dir ./model_output/best_model \
  --test_data /path/to/test_data.json \
  --output_file ./predictions.json \
  --use_temporal
```

### Advanced Training

```bash
python run_roberta.py \
  --run_name experiment1 \
  --use_temporal \
  --wandb
```

### Detailed Evaluation

```bash
python evaluate.py \
  --model_path ./model_output/best_model \
  --claims_file /path/to/test_data.json \
  --output_dir ./evaluation_results
```

## Training Parameters

The model supports various training parameters:

- `--base_model_name`: Base model name (default: "roberta-base")
- `--batch_size`: Batch size (default: 16)
- `--max_length`: Maximum sequence length (default: 512)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--num_epochs`: Number of training epochs (default: 3)
- `--use_temporal`: Whether to use temporal features

Run with `-h` for a complete list of options.

## Model Architecture

The model uses a hybrid architecture that combines:

1. **RoBERTa for Text Processing**: Processes the claim text to extract contextual representations
2. **Feed-forward Networks for Numerical Features**: Processes temporal and speaker credibility features
3. **Combined Classification Layer**: Integrates text and numerical features for final classification

## Feature Types

The model can use three types of features:

1. **Text Features**: The processed claim text (required)
2. **Temporal Features**: Features extracted from evidence (optional)
   - Evidence counts, publication dates, temporal consistency metrics, etc.
3. **Speaker Credibility Features**: Historical truthfulness of the speaker (optional)
   - Speaker credibility scores, historical truth ratios, etc.

## Running the Model

### Installation Requirements

```bash
pip install torch transformers pandas numpy matplotlib seaborn scikit-learn wandb tqdm
```

### Basic Usage

To run the model with text features only:

```bash
python run_roberta.py --run_name baseline
```

To include temporal features:

```bash
python run_roberta.py --run_name temporal --use_temporal
```

To use all features:

```bash
python run_roberta.py --run_name full --use_temporal --use_credibility
```

### Command-line Arguments

```
# Model configuration
--model_name            HuggingFace model name (default: "roberta-base")
--max_length            Maximum sequence length (default: 128)
--batch_size            Training batch size (default: 16)
--grad_accumulation_steps  Gradient accumulation steps (default: 2)

# Training parameters
--epochs                Number of training epochs (default: 5)
--learning_rate         Learning rate (default: 2e-5)
--warmup_ratio          Warmup ratio (default: 0.1)
--weight_decay          Weight decay (default: 0.01)
--dropout_rate          Dropout rate (default: 0.1)
--early_stopping_patience  Early stopping patience (default: 3)

# Feature options
--use_temporal          Use temporal features
--use_credibility       Use speaker credibility features

# Run configuration
--run_name              Name for this run (required)
--seed                  Random seed for reproducibility (default: 42)
--device                Device to use (default: cuda if available, else cpu)
--fp16                  Use mixed precision training
--wandb                 Log metrics with Weights & Biases
```

## Evaluation Metrics

The model is evaluated on:

- Accuracy
- Precision, Recall, and F1-Score
- AUC-ROC
- Feature importance analysis

## Weights & Biases Integration

To track experiments with Weights & Biases:

```bash
# Login to wandb
wandb login

# Run with wandb tracking
python run_roberta.py --run_name experiment1 --use_temporal --wandb
```

## Recommended Hardware

- GPU with at least 8GB of VRAM
- 16GB+ of system RAM
- Mixed precision training (--fp16) recommended for faster training 