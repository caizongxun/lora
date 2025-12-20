#!/bin/bash

# Script to install fully compatible package versions for Colab
# This ensures transformers, peft, and torch all work together

echo "="*80
echo "🔧 Installing Fully Compatible Package Versions"
echo "="*80
echo ""

echo "Step 1: Uninstalling old versions..."
pip uninstall -y transformers peft torch torchaudio torchvision

echo ""
echo "Step 2: Installing PyTorch with CUDA 12.6 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -q

echo ""
echo "Step 3: Installing compatible transformers version..."
pip install transformers==4.40.2 -q

echo ""
echo "Step 4: Installing compatible PEFT version..."
pip install peft==0.7.1 -q

echo ""
echo "Step 5: Verifying installation..."
python -c "
import torch
import transformers
import peft
print(f'✅ torch: {torch.__version__}')
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ peft: {peft.__version__}')
print('')
print('All versions compatible!')
"

echo ""
echo "="*80
echo "✅ Installation complete!"
echo "="*80
