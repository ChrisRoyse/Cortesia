# LLMKG Model Integration Master Roadmap

## Executive Summary

This document outlines the comprehensive strategy for integrating HuggingFace Candle with all model variants in the LLMKG system, enabling real ML inference capabilities while maintaining performance targets and WASM compatibility.

## Architecture Integration Strategy

### Phase 1: Foundation (Week 1)
- **Dependencies**: Add Candle ecosystem crates
- **Core Integration**: Create CandleModelBackend trait implementation
- **Infrastructure**: Extend existing model loading/registry systems
- **Target**: SmolLM-135M as proof of concept

### Phase 2: Model Coverage (Week 2-3)
- **SmolLM Family**: All 9 variants (135M, 360M, 1.7B, v2 series, instruct)
- **TinyLlama Family**: All 7 variants (base, chat versions)
- **OpenELM Family**: All 8 variants (270M-3B range)
- **MiniLM Family**: All 6 embedding variants

### Phase 3: WASM Optimization (Week 4)
- **Bundle Optimization**: Quantization, streaming, progressive loading
- **Performance Validation**: Meet <1ms query, <10ms inference targets
- **Browser Compatibility**: Cross-platform testing

### Phase 4: Production Integration (Week 5)
- **Cognitive Integration**: Connect to attention/memory systems
- **Production Deployment**: MCP server integration
- **Monitoring**: Performance metrics integration

## Technical Requirements

### Core Dependencies
```toml
[dependencies]
# Candle ML Framework
candle-core = "0.6"
candle-nn = "0.6"
candle-transformers = "0.6"
candle-hub = "0.6"

# HuggingFace Integration
hf-hub = "0.3"
tokenizers = "0.19"

# Quantization Support
candle-onnx = "0.6"  # For ONNX model support
candle-wasm-utils = "0.6"  # WASM optimization utilities

# Additional ML utilities
safetensors = "0.4"
```

### Memory Architecture Integration
- **Zero-Copy Compatibility**: Integrate with existing bumpalo arena
- **Memory Pools**: Dedicated inference memory management
- **Cache Strategy**: LRU cache for tokenization and embeddings
- **Quantization**: Int4/Int8 support for all models

### Performance Targets
- **Inference Latency**: <50ms first token, <10ms subsequent
- **Memory Usage**: 100-800MB depending on model size
- **Bundle Size**: <5MB base + progressive model loading
- **Query Integration**: Maintain <1ms graph query performance

## Model Conversion Strategy

### Universal Conversion Pipeline
1. **Download**: HuggingFace Hub → Local cache
2. **Format Check**: Safetensors preferred, fallback to PyTorch
3. **Quantization**: Generate Int4/Int8 variants for WASM
4. **Validation**: Inference testing against reference outputs
5. **Integration**: Connect to LLMKG model registry

### Model Prioritization
1. **Tier 1 (Week 1)**: SmolLM-135M, SmolLM-360M
2. **Tier 2 (Week 2)**: SmolLM-1.7B, TinyLlama-1.1B
3. **Tier 3 (Week 3)**: All remaining generative models
4. **Tier 4 (Week 4)**: MiniLM embedding models

## Implementation Validation

### Testing Strategy
- **Unit Tests**: Each model conversion pipeline
- **Integration Tests**: LLMKG system compatibility  
- **Performance Tests**: Latency and throughput benchmarks
- **WASM Tests**: Browser compatibility across devices
- **Memory Tests**: Leak detection and usage optimization

### Success Criteria
- [ ] All 30 model variants successfully converted
- [ ] <10ms inference latency for small models (<500M params)
- [ ] <500MB memory usage for largest models
- [ ] WASM bundles <100MB per quantized model
- [ ] 100% compatibility with existing LLMKG API

## Risk Mitigation

### Technical Risks
- **Bundle Size**: Progressive loading + quantization fallback
- **Browser Compatibility**: Feature detection + server fallback
- **Performance**: Device-specific model selection
- **Memory**: Streaming inference for large models

### Contingency Plans
- **Candle Issues**: Fallback to ONNX Runtime (ort crate)
- **WASM Problems**: Server-side inference API endpoint
- **Performance**: Smaller model variants for constrained devices

## Directory Structure
```
docs/modelplan/
├── 00_MASTER_IMPLEMENTATION_ROADMAP.md (this file)
├── 01_CANDLE_DEPENDENCIES_SETUP.md
├── 02_WASM_DEPLOYMENT_STRATEGY.md
├── smollm/
│   ├── smollm_135m.md
│   ├── smollm_360m.md
│   ├── smollm_1_7b.md
│   ├── smollm_135m_instruct.md
│   ├── smollm_360m_instruct.md
│   ├── smollm_1_7b_instruct.md
│   ├── smollm2_135m.md
│   ├── smollm2_360m.md
│   └── smollm2_1_7b.md
├── tinyllama/
│   ├── tinyllama_1_1b.md
│   ├── tinyllama_1_1b_chat.md
│   ├── tinyllama_1_1b_chat_v0_1.md
│   ├── tinyllama_1_1b_chat_v0_3.md
│   ├── tinyllama_1_1b_chat_v0_6.md
│   ├── tinyllama_1_1b_chat_v1_0.md
│   └── tinyllama_1_1b_intermediate.md
├── openelm/
│   ├── openelm_270m.md
│   ├── openelm_450m.md
│   ├── openelm_1_1b.md
│   ├── openelm_3b.md
│   ├── openelm_270m_instruct.md
│   ├── openelm_450m_instruct.md
│   ├── openelm_1_1b_instruct.md
│   └── openelm_3b_instruct.md
└── minilm/
    ├── minilm_l12_h384.md
    ├── minilm_multilingual_l12_h384.md
    ├── all_minilm_l6_v2.md
    ├── all_minilm_l12_v2.md
    ├── ms_marco_minilm_l6_v2.md
    └── ms_marco_minilm_l12_v2.md
```

## Next Steps

1. **Review Dependencies**: Examine `01_CANDLE_DEPENDENCIES_SETUP.md`
2. **WASM Strategy**: Study `02_WASM_DEPLOYMENT_STRATEGY.md` 
3. **Model Plans**: Follow individual model conversion guides
4. **Implementation**: Execute in priority order (Tier 1 → Tier 4)
5. **Validation**: Continuous testing throughout implementation

## Success Metrics

- **Development Velocity**: Complete Tier 1 in 5 days
- **Performance**: All targets met for production models
- **Compatibility**: 100% WASM deployment success
- **Integration**: Seamless LLMKG system operation
- **Quality**: Zero regression in existing functionality