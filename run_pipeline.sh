#!/bin/bash

# TELEClass Pipeline Execution Script
# This script runs training and inference sequentially

set -e  # Exit on error

echo "=========================================="
echo "TELEClass Pipeline Execution"
echo "=========================================="

# Install dependencies
echo ""
echo "=========================================="
echo "Installing Dependencies"
echo "=========================================="
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with OPENAI_API_KEY"
    echo "Example: echo 'OPENAI_API_KEY=your_key_here' > .env"
    exit 1
fi

# Check if Amazon_products directory exists
if [ ! -d "Amazon_products" ]; then
    echo "ERROR: Amazon_products directory not found!"
    exit 1
fi

# Change to src directory
cd src

echo ""
echo "=========================================="
echo "Step 1: Training Model"
echo "=========================================="
python train_model.py

echo ""
echo "=========================================="
echo "Step 2: Running Inference"
echo "=========================================="
python inference_model.py

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Check outputs/submission.csv for results"

