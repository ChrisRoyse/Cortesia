# Local Model Setup Guide

This guide explains how to set up and use local models with LLMKG's Enhanced Knowledge Storage System.

## Overview

The Enhanced Knowledge Storage System supports running AI models locally using the Candle framework (Rust-based deep learning). This allows for offline operation and better performance for production deployments.

## Pre-downloaded Models

The following models have been downloaded and converted to SafeTensors format for use with Candle:

1. **bert-base-uncased** (417.7 MB)
   - General-purpose BERT model for embeddings and text understanding
   - 768-dimensional embeddings
   - Used for standard complexity tasks

2. **minilm-l6-v2** (86.7 MB)  
   - Lightweight sentence transformer model
   - 384-dimensional embeddings
   - Optimized for semantic similarity tasks
   - Lower memory footprint

3. **bert-large-ner** (Partially downloaded)
   - Specialized for Named Entity Recognition
   - Currently missing weight files

## Directory Structure

```
model_weights/
├── .models_ready              # Marker file indicating models are set up
├── bert-base-uncased/
│   ├── config.json            # Model configuration
│   ├── model.safetensors      # Converted weights for Candle
│   ├── tokenizer.json         # Tokenizer configuration
│   ├── vocab.txt              # Vocabulary file
│   └── candle_metadata.json   # Candle-specific metadata
├── minilm-l6-v2/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── candle_metadata.json
└── bert-large-ner/
    ├── config.json
    └── vocab.txt
```

## Using Local Models

### 1. Environment Variable

To enable local model usage, set the environment variable:

```bash
# Linux/Mac
export LLMKG_USE_LOCAL_MODELS=true

# Windows
set LLMKG_USE_LOCAL_MODELS=true
```

### 2. In Code

The system automatically detects and uses local models when available:

```rust
use llmkg::enhanced_knowledge_storage::{
    model_management::{ModelResourceManager, ModelResourceConfig},
    types::ComplexityLevel,
};

// The resource manager will use local models if LLMKG_USE_LOCAL_MODELS is set
let config = ModelResourceConfig::default();
let manager = ModelResourceManager::new(config).await?;

// Process a task - will use local models
let task = ProcessingTask {
    complexity_level: ComplexityLevel::Medium,
    content: "Your text here".to_string(),
    task_type: "embeddings".to_string(),
    timeout: Some(30000),
};

let result = manager.process_with_optimal_model(task).await?;
```

### 3. Direct Local Model Usage

For direct control over local models:

```rust
use llmkg::enhanced_knowledge_storage::ai_components::{
    local_model_backend::{LocalModelBackend, LocalModelConfig},
};

let config = LocalModelConfig {
    model_weights_dir: PathBuf::from("model_weights"),
    device: candle_core::Device::Cpu,
    max_sequence_length: 512,
    use_cache: true,
};

let backend = LocalModelBackend::new(config)?;
let embeddings = backend.generate_embeddings("bert-base-uncased", "Hello world").await?;
```

## Model Selection

The system automatically selects models based on task complexity:

- **Low Complexity**: MiniLM-L6-v2 (fast, lightweight)
- **Medium Complexity**: BERT-base-uncased (balanced)
- **High Complexity**: BERT-large models (when available)

## Verifying Setup

Run the verification script to check model status:

```bash
python scripts/check_models.py
```

Expected output:
```
Checking local models...
==================================================

bert-base-uncased:
  [OK] model.safetensors (417.7 MB)
  [OK] config.json (0.0 MB)
  [OK] vocab.txt (0.2 MB)

minilm-l6-v2:
  [OK] model.safetensors (86.7 MB)
  [OK] config.json (0.0 MB)
  [OK] vocab.txt (0.2 MB)

==================================================
[OK] Models are ready for local use!
```

## CI/CD Integration

The GitHub Actions workflow caches model weights to speed up CI builds:

```yaml
- name: Cache Model Weights
  uses: actions/cache@v3
  with:
    path: model_weights/
    key: ${{ runner.os }}-models-v1-${{ hashFiles('scripts/download_models.py') }}
```

## Troubleshooting

### Models Not Loading

1. Check the `.models_ready` marker file exists
2. Verify SafeTensors files are present and not corrupted
3. Ensure `LLMKG_USE_LOCAL_MODELS=true` is set
4. Check logs for specific error messages

### Performance Issues

1. Ensure you're using release builds (`cargo build --release`)
2. Consider using GPU acceleration if available
3. Monitor memory usage - models require ~500MB-1GB RAM each

### Missing Models

To download missing models:

```bash
python scripts/convert_to_candle.py
```

This will:
1. Download models from HuggingFace
2. Convert to SafeTensors format
3. Generate Candle metadata
4. Create the `.models_ready` marker

## Technical Details

### Candle Framework

Candle is a Rust-based deep learning framework that provides:
- Native Rust performance
- No Python dependencies at runtime
- SafeTensors format for secure weight loading
- CPU and GPU support

### Model Formats

- **PyTorch weights** (`.bin`): Original HuggingFace format
- **SafeTensors** (`.safetensors`): Secure, memory-mapped format used by Candle
- **Config files**: JSON configuration for model architecture
- **Tokenizer files**: Vocabulary and tokenization rules

### Integration Architecture

The system uses a local-only approach:
1. **LocalModelBackend**: Exclusively loads and runs Candle models from the local filesystem

This ensures the system works completely offline with all required models present locally. **The system will fail if models are not available locally as there are no remote fallbacks.**