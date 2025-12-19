#!/bin/bash
# Setup script for LoRA evaluation project

echo "====================================================================="
echo "LoRA Fine-Tuning Evaluation for Phi-2 - Setup Script"
echo "====================================================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt
echo ""

echo "====================================================================="
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run evaluation: python main.py"
echo "3. View results: cat evaluation_results/evaluation_log.txt"
echo "====================================================================="
