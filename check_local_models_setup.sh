#!/bin/bash
# Check local models setup

echo "=== Checking Local Model Setup ==="
echo

# Check if model_weights directory exists
if [ -d "model_weights" ]; then
    echo "✓ model_weights directory exists"
    
    # Check for .models_ready marker
    if [ -f "model_weights/.models_ready" ]; then
        echo "✓ .models_ready marker found"
    else
        echo "✗ .models_ready marker missing"
    fi
    
    # Check BERT model
    if [ -f "model_weights/bert-base-uncased/model.safetensors" ]; then
        SIZE=$(du -h "model_weights/bert-base-uncased/model.safetensors" | cut -f1)
        echo "✓ BERT model found (size: $SIZE)"
    else
        echo "✗ BERT model missing"
    fi
    
    # Check MiniLM model
    if [ -f "model_weights/minilm-l6-v2/model.safetensors" ]; then
        SIZE=$(du -h "model_weights/minilm-l6-v2/model.safetensors" | cut -f1)
        echo "✓ MiniLM model found (size: $SIZE)"
    else
        echo "✗ MiniLM model missing"
    fi
    
    echo
    echo "Directory structure:"
    find model_weights -type f -name "*.safetensors" -o -name "config.json" | sort
    
else
    echo "✗ model_weights directory not found!"
fi

echo
echo "=== Setup Summary ==="
echo "To use local models:"
echo "1. Set environment variable: export LLMKG_USE_LOCAL_MODELS=true"
echo "2. Run tests with: cargo test --features ai"
echo "3. CI/CD will cache model_weights directory"