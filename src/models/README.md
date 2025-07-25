# LLMKG AI Models

This directory contains the AI models required for Phase 1 of the LLMKG Human-Like Memory System.

## Required Models

### Primary Models
1. **DistilBERT-NER** (66M params) - High-accuracy named entity recognition
2. **TinyBERT-NER** (14.5M params) - Lightweight NER for batch processing
3. **T5-Small** (60M params) - Text generation for answer synthesis
4. **all-MiniLM-L6-v2** (22M params) - Semantic similarity and embeddings

### Additional Models
5. **DistilBERT-Relation** - Relationship extraction
6. **Dependency Parser** - Syntactic analysis
7. **Intent Classifier** - Question intent understanding
8. **Relation Classifier** - Relationship type classification

## Setup Instructions

### Quick Setup
```bash
cd src/models
python download_models.py
```

This will:
- Create `pretrained/` and `tokenizers/` directories
- Download available models from HuggingFace
- Create placeholders for models that need custom training

### Manual Model Preparation

If you need to prepare models manually:

1. **Convert to ONNX format:**
   ```bash
   python -m transformers.onnx --model=distilbert-base-cased --feature=token-classification distilbert onnx/
   ```

2. **Quantize to INT8 for performance:**
   ```bash
   python -m onnxruntime.quantization.preprocess --input model.onnx --output model_preprocessed.onnx
   python -m onnxruntime.quantization.quantize --input model_preprocessed.onnx --output model_int8.onnx
   ```

## Performance Optimization

The models are optimized for Intel i9 processors with:
- INT8 quantization (6x speedup)
- MKL-DNN acceleration
- AVX-512 instructions
- Parallel batch processing

## Expected Performance (Intel i9)

| Model | Single Inference | Batch (32) | Memory |
|-------|-----------------|------------|--------|
| DistilBERT-NER | <5ms | <50ms | 134MB |
| TinyBERT-NER | <3ms | <30ms | 29MB |
| T5-Small | <20ms | <200ms | 120MB |
| MiniLM | <2ms | <20ms | 44MB |

## Directory Structure

```
models/
├── mod.rs                    # Module definition
├── model_loader.rs          # Model loading utilities
├── onnx_runtime.rs          # ONNX Runtime integration
├── embeddings.rs            # Embedding generation
├── download_models.py       # Model download script
├── README.md               # This file
├── pretrained/             # ONNX model files
│   ├── distilbert_ner_int8.onnx
│   ├── tinybert_ner_int8.onnx
│   ├── t5_small_int8.onnx
│   └── ...
└── tokenizers/             # Tokenizer files
    ├── bert_tokenizer.json
    ├── t5_tokenizer.json
    └── ...
```

## Troubleshooting

### ONNX Runtime Issues
- Ensure ONNX Runtime 1.17+ is installed
- For CUDA support, install `onnxruntime-gpu`
- Check that models are in INT8 format for best performance

### Memory Issues
- Use `TinyBERT` for memory-constrained environments
- Enable lazy loading in `ModelConfig`
- Clear model cache periodically with `ModelLoader::clear_cache()`

### Performance Issues
- Verify MKL-DNN is enabled in `ModelConfig`
- Adjust batch sizes based on your hardware
- Use caching for repeated queries
