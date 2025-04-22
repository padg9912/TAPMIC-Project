# TAPMIC: Temporally Aware Political Misinformation Classification

TAPMIC is a machine learning project focused on enhancing political fact-checking by incorporating temporal context into misinformation detection models.

## Project Overview

This project leverages temporal data and evidence to improve the accuracy of political claim verification. It implements both traditional machine learning models like RoBERTa and evidence-based approaches that consider the timing and chronology of claims and supporting information.

## Key Features

- **Temporal Feature Extraction**: Analyzes dates, timeframes, and temporal relationships in claims and evidence
- **Evidence Retrieval Pipeline**: Collects and processes supporting evidence with temporal information
- **RoBERTa-based Classification**: Fine-tuned transformer model with temporal feature integration
- **Comprehensive Evaluation**: Multi-metric assessment of model performance

## Project Structure

The project is organized as follows:

```
TAPMIC/
├── data/                      # Data processing and analysis
│   ├── intermediate_1/        # Preliminary data processing
│   ├── intermediate_2/        # Feature-enhanced datasets
│   ├── raw_data_reports/      # EDA on raw data
│   └── temporal_features/     # Temporal feature extraction
├── models/                    # Model implementations
│   ├── roberta_model.py       # RoBERTa with temporal features
│   ├── run_roberta.py         # Training script for RoBERTa
│   └── evaluate.py            # Evaluation utilities
├── rag_pipeline/              # Evidence retrieval system
│   ├── claim_extractor.py     # Extract claims from text
│   ├── evidence_collector.py  # Collect supporting evidence
│   └── temporal_pipeline.py   # Extract temporal info from evidence
└── requirements.txt           # Project dependencies
```

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment: 
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install dependencies: `pip install -r TAPMIC/requirements.txt`

## Usage

To train the RoBERTa model with temporal features:

```bash
python TAPMIC/models/run_roberta.py --use_temporal --run_name "roberta_temporal" --wandb
```

## Contributing

Contributions to improve the models or dataset processing are welcome. Please feel free to submit a pull request or open an issue to discuss potential improvements.

## License

[MIT License](LICENSE) 