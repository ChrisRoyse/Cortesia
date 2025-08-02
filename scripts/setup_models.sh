#!/bin/bash
# Setup script for downloading and preparing models

echo "LLMKG Model Setup"
echo "================="

# Create directories
echo "Creating model directories..."
mkdir -p model_weights/{bert-base-uncased,minilm-l6-v2,bert-large-ner}

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Install required Python packages
echo "Installing Python dependencies..."
pip install -q huggingface_hub

# Run the download script
echo "Downloading models..."
python3 scripts/download_models.py

# Create a marker file to indicate models are ready
if [ $? -eq 0 ]; then
    echo "Models downloaded successfully!"
    touch model_weights/.models_ready
else
    echo "Failed to download models"
    exit 1
fi

echo "Setup complete!"