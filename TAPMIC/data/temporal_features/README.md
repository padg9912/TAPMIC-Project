# Temporal Features Framework

This directory contains the framework for temporal feature extraction and analysis for the TAPMIC (Temporally Aware Political Misinformation Classification) project.

## Overview

The temporal features are a core component of the TAPMIC project, designed to capture how political statements' truthfulness relates to temporal factors such as:

- Basic time elements (year, month, day of week)
- Election cycles and campaign seasons
- Speaker credibility evolution over time
- Temporal patterns in truth rates

## Files

- `temporal_features.py`: Core feature extraction framework that will process real date information
- `temporal_analysis.py`: Analysis and visualization framework for temporal features
- `reports/`: Directory for generated reports and visualizations

## Usage

These scripts are designed to be integrated with the RAG pipeline in Phase 2, which will provide real date information from web scraping and search API results.

The current implementation serves as the structural framework for:
1. Extracting temporal components from dates
2. Creating election-related features (election year, campaign season)
3. Generating speaker timeline features (credibility evolution)
4. Analyzing temporal patterns in truth values
5. Creating enhanced datasets with temporal features

## Integration with RAG Pipeline

In Phase 2, the RAG pipeline will:
1. Collect evidence with dates from the web
2. Provide this temporal information to the feature extraction framework
3. Allow for the creation of real temporal features
4. Enable meaningful analysis of how time affects political statement truthfulness

## Future Enhancements

Once real temporal data is available, these scripts will:
- Generate enhanced datasets with temporal features (intermediate2_*.csv)
- Create comprehensive visualizations of temporal patterns
- Provide detailed reports on the relationship between time and truthfulness
- Identify key temporal features for the classification models 