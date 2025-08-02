# LLMKG Model Weights

This directory contains locally cached model weights for the Enhanced Knowledge Storage System.

## Setup

To download and set up the required models:

### Windows
```bash
scripts\setup_models.bat
```

### Linux/macOS
```bash
./scripts/setup_models.sh
```

## Models

The system uses the following models:

1. **bert-base-uncased** (440MB)
   - General purpose BERT model
   - Used for medium complexity tasks
   - Embeddings and text understanding

2. **sentence-transformers/all-MiniLM-L6-v2** (90MB)
   - Lightweight sentence embeddings
   - Used for low complexity tasks
   - Fast inference

3. **dbmdz/bert-large-cased-finetuned-conll03-english** (1.3GB)
   - Named Entity Recognition (NER)
   - Used for high complexity tasks
   - Entity extraction

## Directory Structure

```
model_weights/
├── bert-base-uncased/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── minilm-l6-v2/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab.txt
├── bert-large-ner/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
└── .models_ready  # Marker file indicating models are downloaded
```

## Using Local Models

To use local models instead of downloading from HuggingFace:

```bash
# Set environment variable
export LLMKG_USE_LOCAL_MODELS=true

# Run your application
cargo run --features ai
```

## CI/CD

The CI/CD pipeline caches these models to speed up testing. The cache key is based on the `scripts/download_models.py` file hash.

## Troubleshooting

If models fail to download:

1. Check your internet connection
2. Ensure you have enough disk space (requires ~2GB)
3. Try downloading individual models manually from HuggingFace
4. Check the HuggingFace Hub status

## Manual Download

If automated download fails, you can manually download models:

```python
from huggingface_hub import snapshot_download

# Download a specific model
snapshot_download(
    repo_id="bert-base-uncased",
    local_dir="model_weights/bert-base-uncased"
)
```