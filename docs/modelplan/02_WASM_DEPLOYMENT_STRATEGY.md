# WASM Deployment Strategy for LLMKG Models

## Executive Summary

This document outlines the comprehensive strategy for deploying all LLMKG models via WebAssembly (WASM), enabling client-side ML inference while maintaining performance targets of <1ms query latency and <10ms inference time.

## WASM Architecture Overview

### Core Components
- **Candle WASM Runtime**: Pure Rust ML inference engine
- **Progressive Loading**: Stream model weights on-demand
- **Quantization Pipeline**: Int4/Int8 compression for size optimization
- **Memory Management**: Integration with existing zero-copy architecture
- **WebGPU Acceleration**: When available in browser

### Performance Targets
- **Bundle Size**: <5MB initial load, <100MB per quantized model
- **Memory Usage**: <500MB peak for largest models (1.7B parameters)
- **Inference Speed**: <50ms first token, <10ms subsequent tokens
- **Query Integration**: Maintain <1ms knowledge graph query performance

## Bundle Optimization Strategy

### 1. Core Bundle Structure
```
llmkg-wasm/
├── llmkg.js              # WASM bindings (500KB)
├── llmkg_bg.wasm         # Core runtime (2MB)
├── llmkg.d.ts           # TypeScript definitions (50KB)
└── models/               # Progressive model loading
    ├── smollm/
    │   ├── smollm_135m_q4.bin    # Quantized weights (60MB)
    │   ├── smollm_135m_config.json
    │   └── smollm_135m_tokenizer.json
    └── ...
```

### 2. Quantization Strategy

#### Int4 Quantization (Primary)
- **Size Reduction**: 75% smaller than FP16
- **Quality**: Minimal degradation for models >360M parameters
- **Memory**: ~4 bits per parameter + overhead
- **Target Models**: All models >270M parameters

#### Int8 Quantization (Fallback)
- **Size Reduction**: 50% smaller than FP16
- **Quality**: Negligible degradation
- **Memory**: ~8 bits per parameter + overhead
- **Target Models**: Models with sensitive quality requirements

#### Implementation
```rust
// src/models/quantization.rs
pub struct ModelQuantizer {
    target_bits: u8,
    calibration_data: Vec<String>,
}

impl ModelQuantizer {
    pub fn quantize_model(&self, model: &CandleModel, output_path: &Path) -> Result<QuantizedModel> {
        match self.target_bits {
            4 => self.quantize_to_int4(model, output_path),
            8 => self.quantize_to_int8(model, output_path),
            _ => Err(GraphError::InvalidState("Unsupported quantization bits".to_string())),
        }
    }

    fn quantize_to_int4(&self, model: &CandleModel, output_path: &Path) -> Result<QuantizedModel> {
        // Implementation using Candle's quantization utilities
        // Focus on weight-only quantization for WASM deployment
        todo!("Implement Int4 quantization")
    }
}
```

### 3. Progressive Loading Implementation

#### Model Streaming Architecture
```rust
// src/models/wasm_streaming.rs
pub struct StreamingModelLoader {
    base_url: String,
    cache: Arc<RwLock<HashMap<String, ModelChunk>>>,
    loading_queue: Arc<RwLock<VecDeque<LoadingRequest>>>,
}

impl StreamingModelLoader {
    pub async fn load_model_progressive(&self, model_id: &str) -> Result<StreamingModel> {
        // 1. Load model config and tokenizer (small, <1MB)
        let config = self.fetch_config(model_id).await?;
        let tokenizer = self.fetch_tokenizer(model_id).await?;
        
        // 2. Create streaming model with lazy weight loading
        let streaming_model = StreamingModel::new(config, tokenizer, self.clone());
        
        // 3. Start background loading of critical layers
        self.preload_critical_layers(model_id).await?;
        
        Ok(streaming_model)
    }

    async fn preload_critical_layers(&self, model_id: &str) -> Result<()> {
        // Load embedding layer and first few transformer layers
        // These are needed for first token generation
        let critical_layers = vec!["embed_tokens", "layers.0", "layers.1"];
        
        for layer in critical_layers {
            self.fetch_layer_async(model_id, layer).await?;
        }
        
        Ok(())
    }
}

pub struct StreamingModel {
    config: ModelConfig,
    tokenizer: Arc<Tokenizer>,
    loader: StreamingModelLoader,
    loaded_layers: Arc<RwLock<HashMap<String, Tensor>>>,
}

impl CandleModel for StreamingModel {
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        // Ensure required layers are loaded before computation
        self.ensure_layers_loaded(&self.get_required_layers(input_ids))?;
        
        // Proceed with normal forward pass
        self.forward_with_loaded_layers(input_ids, attention_mask)
    }
}
```

### 4. Memory Management Integration

#### WASM Memory Architecture
```rust
// src/models/wasm_memory.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmMemoryManager {
    arena: bumpalo::Bump,
    model_memory: Vec<u8>,
    inference_cache: lru::LruCache<String, Vec<f32>>,
    memory_limit: usize,
}

#[wasm_bindgen]
impl WasmMemoryManager {
    #[wasm_bindgen(constructor)]
    pub fn new(memory_limit_mb: usize) -> Self {
        Self {
            arena: bumpalo::Bump::new(),
            model_memory: Vec::new(),
            inference_cache: lru::LruCache::new(NonZeroUsize::new(1000).unwrap()),
            memory_limit: memory_limit_mb * 1024 * 1024,
        }
    }

    pub fn allocate_model_weights(&mut self, size: usize) -> Result<*mut u8, JsValue> {
        if self.get_memory_usage() + size > self.memory_limit {
            return Err(JsValue::from_str("Memory limit exceeded"));
        }

        self.model_memory.resize(self.model_memory.len() + size, 0);
        Ok(self.model_memory.as_mut_ptr())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.arena.allocated_bytes() + self.model_memory.len()
    }

    pub fn cleanup(&mut self) {
        self.arena.reset();
        self.inference_cache.clear();
    }
}
```

## Model-Specific WASM Deployment Plans

### Tier 1: Small Models (<400M parameters)
- **Target**: SmolLM-135M, SmolLM-360M
- **Bundle Size**: 60-150MB quantized
- **Loading**: Full model in single request
- **Memory**: 200-400MB peak usage
- **Use Case**: Real-time chat, instant responses

### Tier 2: Medium Models (400M-1B parameters)
- **Target**: TinyLlama-1.1B, OpenELM-450M
- **Bundle Size**: 150-400MB quantized
- **Loading**: Progressive with 2-3 chunks
- **Memory**: 400-600MB peak usage
- **Use Case**: Document analysis, code generation

### Tier 3: Large Models (1B+ parameters)
- **Target**: SmolLM-1.7B, OpenELM-3B
- **Bundle Size**: 400MB-1GB quantized
- **Loading**: Streaming with 5+ chunks
- **Memory**: 600-800MB peak usage
- **Use Case**: Complex reasoning, research tasks

### Tier 4: Embedding Models
- **Target**: All MiniLM variants
- **Bundle Size**: 20-100MB
- **Loading**: Single request, fast loading
- **Memory**: 100-200MB peak usage
- **Use Case**: Semantic search, embeddings

## Browser Compatibility Matrix

### Modern Browsers (Full Support)
- **Chrome 91+**: WebAssembly threads, SIMD, WebGPU (experimental)
- **Firefox 89+**: WebAssembly threads, SIMD
- **Safari 15+**: WebAssembly, limited threading
- **Edge 91+**: Same as Chrome

### Optimizations by Browser
```rust
// src/models/browser_detection.rs
pub struct BrowserOptimizer;

impl BrowserOptimizer {
    pub fn detect_capabilities() -> BrowserCapabilities {
        let mut caps = BrowserCapabilities::default();
        
        #[cfg(target_arch = "wasm32")]
        {
            caps.wasm_threads = Self::check_wasm_threads();
            caps.wasm_simd = Self::check_wasm_simd();
            caps.webgpu = Self::check_webgpu();
            caps.shared_array_buffer = Self::check_shared_array_buffer();
        }
        
        caps
    }

    pub fn optimize_for_browser(&self, caps: &BrowserCapabilities) -> DeploymentConfig {
        DeploymentConfig {
            use_threads: caps.wasm_threads,
            use_simd: caps.wasm_simd,
            use_webgpu: caps.webgpu,
            max_memory_mb: if caps.wasm_threads { 2048 } else { 1024 },
            chunk_size: if caps.shared_array_buffer { 50 } else { 20 }, // MB
        }
    }
}

#[derive(Default)]
pub struct BrowserCapabilities {
    pub wasm_threads: bool,
    pub wasm_simd: bool,
    pub webgpu: bool,
    pub shared_array_buffer: bool,
}
```

## Performance Optimization Techniques

### 1. SIMD Acceleration
```rust
// Enable SIMD for compatible browsers
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

pub fn optimized_matrix_multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
    if is_simd_available() {
        simd_matrix_multiply(a, b)
    } else {
        fallback_matrix_multiply(a, b)
    }
}

fn simd_matrix_multiply(a: &[f32], b: &[f32]) -> Vec<f32> {
    // Use WASM SIMD instructions for 4x speedup
    todo!("Implement SIMD matrix operations")
}
```

### 2. WebGPU Acceleration (Future)
```rust
// Experimental WebGPU support
#[cfg(target_arch = "wasm32")]
pub struct WebGpuAccelerator {
    device: Option<web_sys::GpuDevice>,
    queue: Option<web_sys::GpuQueue>,
}

impl WebGpuAccelerator {
    pub async fn new() -> Result<Self, JsValue> {
        if Self::is_webgpu_available() {
            // Initialize WebGPU device
            Self::init_webgpu().await
        } else {
            Ok(Self { device: None, queue: None })
        }
    }

    pub fn accelerate_inference(&self, tensors: &[Tensor]) -> Result<Vec<Tensor>, JsValue> {
        if let (Some(device), Some(queue)) = (&self.device, &self.queue) {
            // Use WebGPU for tensor operations
            self.gpu_inference(device, queue, tensors)
        } else {
            // Fallback to CPU
            self.cpu_inference(tensors)
        }
    }
}
```

## Deployment Pipeline

### 1. Build Process
```bash
#!/bin/bash
# scripts/build_wasm_models.sh

set -e

echo "Building WASM models with quantization..."

# Build base WASM runtime
wasm-pack build --target web --features "wasm,quantization" --release

# Create model deployment directory
mkdir -p dist/models

# Process each model tier
for tier in tier1 tier2 tier3 tier4; do
    echo "Processing $tier models..."
    
    # Download and quantize models for this tier
    cargo run --bin quantize_models -- --tier $tier --output dist/models/
    
    # Optimize WASM bundles
    wasm-opt -Os -o dist/models/$tier/optimized.wasm dist/models/$tier/model.wasm
done

echo "WASM deployment ready in dist/"
```

### 2. CDN Distribution
```javascript
// CDN structure for model distribution
const MODEL_CDN_BASE = 'https://cdn.llmkg.ai/models/v1/';

const MODEL_URLS = {
    'smollm-135m': `${MODEL_CDN_BASE}smollm/135m_q4.bin`,
    'smollm-360m': `${MODEL_CDN_BASE}smollm/360m_q4.bin`,
    'tinyllama-1.1b': `${MODEL_CDN_BASE}tinyllama/1.1b_q4.bin`,
    // ... other models
};

// Progressive loading with caching
class ModelLoader {
    constructor() {
        this.cache = new Map();
        this.loadingPromises = new Map();
    }

    async loadModel(modelId) {
        if (this.cache.has(modelId)) {
            return this.cache.get(modelId);
        }

        if (this.loadingPromises.has(modelId)) {
            return this.loadingPromises.get(modelId);
        }

        const promise = this._loadModelInternal(modelId);
        this.loadingPromises.set(modelId, promise);
        
        try {
            const model = await promise;
            this.cache.set(modelId, model);
            return model;
        } finally {
            this.loadingPromises.delete(modelId);
        }
    }

    async _loadModelInternal(modelId) {
        const url = MODEL_URLS[modelId];
        if (!url) {
            throw new Error(`Unknown model: ${modelId}`);
        }

        // Show loading progress
        const response = await fetch(url);
        const reader = response.body.getReader();
        const contentLength = +response.headers.get('Content-Length');
        
        let receivedLength = 0;
        const chunks = [];
        
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            chunks.push(value);
            receivedLength += value.length;
            
            // Update progress
            const progress = (receivedLength / contentLength) * 100;
            this.onProgress?.(modelId, progress);
        }

        // Combine chunks and initialize model
        const allChunks = new Uint8Array(receivedLength);
        let position = 0;
        for (const chunk of chunks) {
            allChunks.set(chunk, position);
            position += chunk.length;
        }

        return this.initializeModel(modelId, allChunks);
    }
}
```

## Testing Strategy

### 1. Performance Benchmarks
```rust
// benches/wasm_inference.rs
use criterion::{criterion_group, criterion_main, Criterion};
use llmkg::models::*;

fn benchmark_model_inference(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    
    c.bench_function("smollm_135m_inference", |b| {
        b.to_async(&rt).iter(|| async {
            let model = load_test_model("smollm-135m").await;
            model.generate_text("Hello world", &ModelConfig::default()).await
        });
    });
}

criterion_group!(benches, benchmark_model_inference);
criterion_main!(benches);
```

### 2. Browser Testing
```javascript
// tests/browser_compatibility.js
describe('WASM Model Deployment', () => {
    test('loads on Chrome with threading', async () => {
        const model = await loadModel('smollm-135m');
        expect(model).toBeDefined();
        
        const result = await model.generate('Hello');
        expect(result).toContain('Hello');
    });

    test('graceful degradation on older browsers', async () => {
        // Mock older browser environment
        delete window.SharedArrayBuffer;
        
        const model = await loadModel('smollm-135m');
        expect(model).toBeDefined();
        // Should still work, just slower
    });
});
```

## Success Metrics

### Performance Targets
- [ ] Bundle size <5MB initial, <100MB per model
- [ ] Memory usage <500MB peak
- [ ] First token <50ms, subsequent <10ms
- [ ] 95th percentile loading time <30 seconds

### Compatibility Targets
- [ ] Chrome 91+ full performance
- [ ] Firefox 89+ full performance
- [ ] Safari 15+ basic functionality
- [ ] Mobile browsers functional

### Quality Targets
- [ ] <5% quality degradation with Int4 quantization
- [ ] 100% API compatibility with native models
- [ ] Zero memory leaks in 24-hour testing

This WASM deployment strategy provides the foundation for deploying all LLMKG models in browsers while maintaining performance and compatibility targets.