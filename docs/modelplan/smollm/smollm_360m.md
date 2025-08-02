# SmolLM-360M Integration Plan

## Model Overview

**Model ID**: `HuggingFaceTB/SmolLM-360M`  
**Parameters**: 360,000,000 (360M)  
**Architecture**: Transformer (Llama-based)  
**Size Category**: Medium  
**Priority**: Tier 1 - High Performance  
**Use Case**: Balanced performance and efficiency, document analysis

## Technical Specifications

### Model Details
- **Family**: SmolLM
- **Version**: 1.0
- **Context Length**: 2048 tokens
- **Vocabulary Size**: 49,152 tokens  
- **Training Data**: 600B tokens from SmolLM corpus
- **License**: Apache-2.0
- **Release Date**: July 2024

### Performance Targets
- **Inference Speed**: 50-80 tokens/second (CPU)
- **Memory Usage**: 400-600MB peak
- **First Token Latency**: <40ms
- **Subsequent Tokens**: <12ms
- **WASM Bundle Size**: ~150MB quantized (Int4)

## Implementation Plan

### Phase 1: Model Download & Conversion (Day 1)

#### Step 1.1: Download Model Files
```bash
# scripts/download_smollm_360m.sh
#!/bin/bash
set -e

MODEL_ID="HuggingFaceTB/SmolLM-360M"
OUTPUT_DIR="./models/smollm/360m"

echo "Downloading SmolLM-360M from HuggingFace Hub..."

mkdir -p "$OUTPUT_DIR"

huggingface-cli download "$MODEL_ID" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

echo "Model downloaded to: $OUTPUT_DIR"

# Validate download
echo "Validating model files..."
python3 scripts/validate_model.py "$OUTPUT_DIR" --expected-size 1400000000 --vocab-size 49152
```

**Expected Files**:
- `config.json` (model configuration)
- `model.safetensors` (model weights ~1.4GB)
- `tokenizer.json` (tokenizer configuration)
- `tokenizer_config.json` (tokenizer metadata)

#### Step 1.2: Model Configuration Verification
```rust
// src/models/smollm/smollm_360m.rs
pub struct SmolLM360M {
    model: Llama,
    device: Device,
    cache: Cache,
    config: LlamaConfig,
}

impl SmolLM360M {
    pub fn load(model_path: &Path, device: Device) -> Result<Self> {
        // Verify this is indeed the 360M model
        let config_path = model_path.join("config.json");
        let config_json = std::fs::read_to_string(config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_json)?;
        
        // Verify model dimensions
        let hidden_size = config["hidden_size"].as_u64().unwrap();
        let num_layers = config["num_hidden_layers"].as_u64().unwrap();
        
        // SmolLM-360M specific architecture validation
        if hidden_size != 1024 || num_layers != 30 {
            return Err(GraphError::InvalidState(
                format!("Invalid SmolLM-360M architecture: hidden_size={}, layers={}", 
                       hidden_size, num_layers)
            ));
        }
        
        // Load model (same pattern as 135M)
        let config: LlamaConfig = serde_json::from_str(&config_json)?;
        let weights_path = model_path.join("model.safetensors");
        let weights_data = std::fs::read(weights_path)?;
        let weights = SafeTensors::deserialize(&weights_data)?;
        
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let vb = candle_nn::VarBuilder::from_tensors(weights, dtype, &device);
        
        let model = Llama::load(vb, &config)?;
        let cache = Cache::new(true, dtype, &config, &device)?;

        Ok(Self { model, device, cache, config })
    }

    // Same interface as SmolLM135M but optimized for larger model
    pub fn generate_text(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_with_config(prompt, max_tokens, &GenerationConfig::default())
    }
    
    pub fn generate_with_config(&mut self, prompt: &str, max_tokens: usize, config: &GenerationConfig) -> Result<String> {
        let tokens = self.tokenize(prompt)?;
        let mut generated_tokens = tokens.clone();
        
        // Use better sampling for 360M model
        let mut logits_processor = LogitsProcessor::new(
            config.seed,
            Some(config.temperature),
            config.top_p,
        );
        
        for _ in 0..max_tokens {
            let input_ids = Tensor::from_slice(
                &generated_tokens, 
                (1, generated_tokens.len()), 
                &self.device
            )?;

            let logits = self.model.forward(&input_ids, 0, &mut self.cache)?;
            
            // Advanced sampling (temperature, top-p)
            let next_token = logits_processor.sample(&logits.squeeze(0)?)?;
            
            if next_token == self.config.eos_token_id.unwrap_or(2) {
                break;
            }
            
            generated_tokens.push(next_token);
        }

        self.detokenize(&generated_tokens[tokens.len()..])
    }
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: Some(0.9),
            seed: None,
            repetition_penalty: 1.1,
        }
    }
}
```

### Phase 2: Performance Optimization (Day 1-2)

#### Step 2.1: Memory-Efficient Loading
```rust
// src/models/smollm/memory_optimized.rs
pub struct MemoryOptimizedSmolLM360M {
    layers: Vec<TransformerBlock>,
    embed_tokens: Embedding,
    norm: LayerNorm,
    lm_head: Linear,
    device: Device,
    current_layer_cache: Option<usize>, // Only keep one layer in memory at a time
}

impl MemoryOptimizedSmolLM360M {
    pub fn load_streaming(model_path: &Path, device: Device) -> Result<Self> {
        // Load only essential components first
        let config = Self::load_config(model_path)?;
        let weights = Self::load_weights_lazy(model_path)?;
        
        // Load embedding and output layers (always needed)
        let embed_tokens = Self::load_embedding_layer(&weights, &config, &device)?;
        let lm_head = Self::load_output_layer(&weights, &config, &device)?;
        let norm = Self::load_norm_layer(&weights, &config, &device)?;
        
        // Create placeholder for transformer layers (load on-demand)
        let layers = vec![]; // Will be populated during inference
        
        Ok(Self {
            layers,
            embed_tokens,
            norm,
            lm_head,
            device,
            current_layer_cache: None,
        })
    }
    
    fn load_layer_on_demand(&mut self, layer_idx: usize, weights: &SafeTensors) -> Result<()> {
        if self.current_layer_cache == Some(layer_idx) {
            return Ok(()); // Already loaded
        }
        
        // Unload previous layer to save memory
        if let Some(prev_idx) = self.current_layer_cache {
            self.unload_layer(prev_idx)?;
        }
        
        // Load requested layer
        let layer = Self::load_transformer_block(weights, layer_idx, &self.device)?;
        
        // Update cache
        if self.layers.len() <= layer_idx {
            self.layers.resize(layer_idx + 1, TransformerBlock::default());
        }
        self.layers[layer_idx] = layer;
        self.current_layer_cache = Some(layer_idx);
        
        Ok(())
    }
}
```

#### Step 2.2: Advanced Quantization
```rust
// src/models/smollm/quantization_360m.rs
pub struct SmolLM360MQuantizer;

impl SmolLM360MQuantizer {
    pub fn quantize_to_int4_grouped(model_path: &Path, output_path: &Path) -> Result<()> {
        println!("Quantizing SmolLM-360M to Int4 with group quantization...");
        
        let weights_path = model_path.join("model.safetensors");
        let weights_data = std::fs::read(weights_path)?;
        let tensors = SafeTensors::deserialize(&weights_data)?;
        
        let device = Device::Cpu;
        let mut quantized_data = QuantizedModelData::new();
        
        // Use group-wise quantization for better quality
        for (name, tensor_view) in tensors.tensors() {
            let tensor = tensor_view.load(&device)?;
            
            if Self::should_quantize_tensor(name) {
                let quantized = Self::quantize_tensor_grouped(&tensor, 128)?; // 128 elements per group
                quantized_data.add_tensor(name, quantized);
            } else {
                // Keep embeddings and layer norms in FP16
                let fp16_tensor = tensor.to_dtype(DType::F16)?;
                quantized_data.add_tensor(name, QuantizedTensor::FP16(fp16_tensor));
            }
        }
        
        // Save with compression
        quantized_data.save_compressed(output_path)?;
        
        // Copy configuration files
        Self::copy_config_files(model_path, output_path)?;
        
        println!("360M quantization complete. Size reduction: ~75%");
        Ok(())
    }
    
    fn quantize_tensor_grouped(tensor: &Tensor, group_size: usize) -> Result<QuantizedTensor> {
        let shape = tensor.shape();
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        
        let mut quantized_groups = Vec::new();
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        
        // Process in groups for better quality
        for chunk in data.chunks(group_size) {
            let min_val = chunk.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = chunk.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            let scale = (max_val - min_val) / 15.0; // 4-bit range
            let zero_point = (-min_val / scale).round() as i8;
            
            let quantized_chunk: Vec<u8> = chunk.iter()
                .map(|&x| ((x / scale + zero_point as f32).round().clamp(0.0, 15.0) as u8))
                .collect();
                
            quantized_groups.extend(quantized_chunk);
            scales.push(scale);
            zero_points.push(zero_point);
        }
        
        Ok(QuantizedTensor::Int4Grouped {
            data: quantized_groups,
            scales,
            zero_points,
            shape: shape.dims().to_vec(),
            group_size,
        })
    }
}

#[derive(Debug)]
pub enum QuantizedTensor {
    Int4Grouped {
        data: Vec<u8>,
        scales: Vec<f32>,
        zero_points: Vec<i8>,
        shape: Vec<usize>,
        group_size: usize,
    },
    FP16(Tensor),
}
```

### Phase 3: WASM Deployment (Day 2-3)

#### Step 3.1: Progressive Loading for WASM
```rust
// src/models/smollm/wasm_360m.rs
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmSmolLM360M {
    model: Option<SmolLM360M>,
    loading_state: LoadingState,
    loaded_layers: HashSet<usize>,
    base_url: String,
}

#[derive(Debug, Clone, PartialEq)]
enum LoadingState {
    Uninitialized,
    LoadingConfig,
    LoadingEmbeddings,
    LoadingLayers(usize), // Current layer being loaded
    Ready,
    Error(String),
}

#[wasm_bindgen]
impl WasmSmolLM360M {
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self {
            model: None,
            loading_state: LoadingState::Uninitialized,
            loaded_layers: HashSet::new(),
            base_url,
        }
    }
    
    #[wasm_bindgen]
    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        self.loading_state = LoadingState::LoadingConfig;
        
        // Load configuration and tokenizer first (small files)
        let config = self.fetch_config().await?;
        let tokenizer = self.fetch_tokenizer().await?;
        
        self.loading_state = LoadingState::LoadingEmbeddings;
        
        // Load embedding layer
        let embeddings = self.fetch_embeddings().await?;
        
        // Initialize model with basic components
        self.model = Some(SmolLM360M::new_with_components(config, tokenizer, embeddings)?);
        
        // Start loading first few layers in background
        self.preload_critical_layers().await?;
        
        self.loading_state = LoadingState::Ready;
        Ok(())
    }
    
    #[wasm_bindgen]
    pub async fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String, JsValue> {
        if self.loading_state != LoadingState::Ready {
            return Err(JsValue::from_str("Model not ready"));
        }
        
        // Ensure all layers are loaded for generation
        self.ensure_all_layers_loaded().await?;
        
        let model = self.model.as_mut().unwrap();
        model.generate_text(prompt, max_tokens)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    async fn fetch_config(&self) -> Result<ModelConfig, JsValue> {
        let url = format!("{}/config.json", self.base_url);
        let response = self.fetch_json(&url).await?;
        serde_json::from_value(response)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    async fn preload_critical_layers(&mut self) -> Result<(), JsValue> {
        // Load first 3 layers for faster first token
        for layer_idx in 0..3 {
            self.load_layer(layer_idx).await?;
        }
        Ok(())
    }
    
    async fn load_layer(&mut self, layer_idx: usize) -> Result<(), JsValue> {
        if self.loaded_layers.contains(&layer_idx) {
            return Ok(());
        }
        
        let url = format!("{}/layers/layer_{}.bin", self.base_url, layer_idx);
        let layer_data = self.fetch_binary(&url).await?;
        
        // Deserialize and load layer into model
        if let Some(model) = &mut self.model {
            model.load_layer_from_bytes(layer_idx, &layer_data)?;
            self.loaded_layers.insert(layer_idx);
        }
        
        Ok(())
    }
}
```

#### Step 3.2: Bundle Size Optimization
```bash
# scripts/build_smollm_360m_wasm.sh
#!/bin/bash
set -e

MODEL_NAME="smollm_360m"
OUTPUT_DIR="./dist/wasm/models/$MODEL_NAME"

echo "Building optimized WASM bundle for SmolLM-360M..."

# Quantize model with aggressive settings for WASM
cargo run --bin quantize_model -- \
    --model-path ./models/smollm/360m \
    --output-path ./models/smollm/360m_q4 \
    --format int4-grouped \
    --group-size 128 \
    --calibration-data ./data/calibration_samples.txt

# Split model into layers for progressive loading
python3 scripts/split_model_layers.py \
    ./models/smollm/360m_q4 \
    --output ./models/smollm/360m_split \
    --layers-per-chunk 3

# Build WASM with streaming support
wasm-pack build \
    --target web \
    --out-dir "$OUTPUT_DIR" \
    --features "wasm,streaming,quantization" \
    --release

# Optimize WASM bundle
wasm-opt -Os --enable-bulk-memory -o "$OUTPUT_DIR/llmkg_bg.wasm" "$OUTPUT_DIR/llmkg_bg.wasm"

# Create layer chunks
mkdir -p "$OUTPUT_DIR/layers"
cp ./models/smollm/360m_split/layer_*.bin "$OUTPUT_DIR/layers/"

# Copy configuration files
cp ./models/smollm/360m_q4/*.json "$OUTPUT_DIR/"

# Create optimized manifest
cat > "$OUTPUT_DIR/model_manifest.json" << EOF
{
  "name": "SmolLM-360M",
  "version": "1.0.0",
  "architecture": "llama",
  "parameters": 360000000,
  "quantization": "int4-grouped",
  "bundle_size_mb": $(du -m "$OUTPUT_DIR" | tail -1 | cut -f1),
  "streaming": {
    "enabled": true,
    "total_layers": 30,
    "layers_per_chunk": 3,
    "critical_layers": [0, 1, 2]
  },
  "performance": {
    "target_tokens_per_second": 60,
    "max_memory_mb": 600,
    "first_token_latency_ms": 40,
    "context_length": 2048
  }
}
EOF

echo "WASM bundle created at: $OUTPUT_DIR"
echo "Total bundle size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo "Core runtime: $(du -sh "$OUTPUT_DIR"/*.wasm "$OUTPUT_DIR"/*.js | tail -1)"
```

### Phase 4: Testing & Validation (Day 3)

#### Step 4.1: Quality Comparison Tests
```rust
// tests/models/smollm_360m_quality.rs
#[tokio::test]
async fn test_smollm_360m_vs_reference() {
    let test_prompts = vec![
        "The future of artificial intelligence",
        "Explain quantum computing in simple terms",
        "Write a short story about a robot",
        "What is the capital of France?",
    ];
    
    let our_model = load_test_model("smollm-360m").await;
    let reference_outputs = load_reference_outputs("smollm_360m_reference.json");
    
    for (prompt, expected) in test_prompts.iter().zip(reference_outputs) {
        let our_output = our_model.generate_text(prompt, 50).unwrap();
        
        // Compare BLEU score, semantic similarity
        let bleu_score = calculate_bleu_score(&our_output, &expected);
        let semantic_sim = calculate_semantic_similarity(&our_output, &expected);
        
        // Quality thresholds
        assert!(bleu_score > 0.6, "BLEU score too low: {}", bleu_score);
        assert!(semantic_sim > 0.8, "Semantic similarity too low: {}", semantic_sim);
    }
}

#[tokio::test]
async fn test_quantized_vs_fp16_quality() {
    let fp16_model = load_fp16_model("smollm-360m").await;
    let quantized_model = load_quantized_model("smollm-360m-q4").await;
    
    let test_prompt = "The significance of renewable energy";
    
    let fp16_output = fp16_model.generate_text(&test_prompt, 100).unwrap();
    let q4_output = quantized_model.generate_text(&test_prompt, 100).unwrap();
    
    let quality_degradation = calculate_quality_degradation(&fp16_output, &q4_output);
    
    // Should be <5% quality loss with Int4 quantization
    assert!(quality_degradation < 0.05, "Quality degradation too high: {:.2}%", quality_degradation * 100.0);
}
```

## Success Criteria

### Functional Requirements
- [ ] Successfully load SmolLM-360M with Candle
- [ ] Generate high-quality text with advanced sampling
- [ ] Create Int4 quantized version <160MB
- [ ] Implement progressive loading for WASM
- [ ] Achieve streaming inference with layer-by-layer loading

### Performance Requirements
- [ ] Memory usage <600MB peak (native), <400MB (WASM)
- [ ] Inference speed >50 tokens/second (CPU)
- [ ] First token latency <40ms
- [ ] WASM bundle core <20MB, total <160MB
- [ ] Quality degradation <5% with quantization

### Integration Requirements
- [ ] Full API compatibility with LLMKG system
- [ ] Browser deployment with progressive loading
- [ ] Memory-efficient streaming inference
- [ ] Comprehensive test suite with quality benchmarks

## Risk Mitigation

1. **Large Model Size**: Implement aggressive quantization and layer streaming
2. **Memory Usage**: Use memory-mapped loading and layer unloading
3. **WASM Limitations**: Progressive loading with critical path optimization
4. **Quality Loss**: Group-wise quantization and careful layer selection

## Timeline

- **Day 1**: Download, validation, architecture verification, memory optimization
- **Day 2**: Advanced quantization, streaming inference implementation
- **Day 3**: WASM deployment, progressive loading, quality testing

This plan builds on the SmolLM-135M foundation while addressing the unique challenges of the larger 360M parameter model.