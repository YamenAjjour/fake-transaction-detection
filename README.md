# Fake Transaction Detection System - User Guide

## Overview

A machine learning system that detects fraudulent transactions using Random Forest Classification. The system analyzes product descriptions and average prices to identify suspicious patterns.

## System Requirements

- Python 3.7 or higher
- Required Python packages: pandas, numpy, scikit-learn
- Sufficient disk space for CSV data storage
- Minimum 4GB RAM recommended

## Data Requirements

The system expects a CSV file named "data_fakes.csv" with the following columns:

- ProductName: Name of the product
- product_description: Detailed description of the product
- AveragePrice: Average price of the product
- Fake: Binary indicator (0 for legitimate, 1 for fake transactions)

## Running the System

### Initial Setup

- Place your data_fakes.csv file in the same directory as the main script
- Ensure all required Python packages are installed

### Execution Steps

1. First Run:
- Execute the main script to perform initial data splitting
- The system will automatically create test.csv and train.csv files
1. Subsequent Runs:
- The system will use existing test.csv and train.csv files if available
- Delete these files if you want to generate a new data split

### Output and Results

The system will display:

- F1 score of the baseline random classifier
- F1 score of the optimized Random Forest model
- Best depth parameter found during optimization

## Performance Notes

- Model depth is automatically optimized between 10 and 22 levels
- Results focus on fake transaction detection accuracy
- Performance metrics are saved for reproducibility

## Troubleshooting

- Ensure input CSV file follows the required format
- Check for sufficient memory when processing large datasets
- Verify all required Python packages are correctly installed
