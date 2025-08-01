# SmolLM-135M Integration Plan

## Model Overview

**Model ID**: `HuggingFaceTB/SmolLM-135M`  
**Parameters**: 135,000,000 (135M)  
**Architecture**: Transformer (Llama-based)  
**Size Category**: Small  
**Priority**: Tier 1 - Proof of Concept  
**Use Case**: Real-time chat, instant responses, edge deployment

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
- **Inference Speed**: 80-120 tokens/second (CPU)
- **Memory Usage**: 200-300MB peak
- **First Token Latency**: <30ms
- **Subsequent Tokens**: <8ms
- **WASM Bundle Size**: ~60MB quantized (Int4)

## Implementation Plan

### Phase 1: Model Download & Conversion (Day 1)

#### Step 1.1: Download Model Files
```bash
# Create download script: scripts/download_smollm_135m.sh
#!/bin/bash
set -e

MODEL_ID="HuggingFaceTB/SmolLM-135M"
CACHE_DIR="$HOME/.cache/huggingface/hub"
OUTPUT_DIR="./models/smollm/135m"

echo "Downloading SmolLM-135M from HuggingFace Hub..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download using huggingface-hub CLI
huggingface-cli download "$MODEL_ID" \
    --local-dir "$OUTPUT_DIR" \
    --local-dir-use-symlinks False

echo "Model downloaded to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
```

**Expected Files**:
- `config.json` (model configuration)
- `model.safetensors` (model weights ~540MB)
- `tokenizer.json` (tokenizer configuration)
- `tokenizer_config.json` (tokenizer metadata)
- `special_tokens_map.json` (special tokens)

#### Step 1.2: Validate Download
```rust
// src/models/validation.rs - Add validation function
pub fn validate_smollm_135m_download(model_path: &Path) -> Result<()> {
    let required_files = [
        "config.json",
        "model.safetensors", 
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    for file in &required_files {
        let file_path = model_path.join(file);
        if !file_path.exists() {
            return Err(GraphError::StorageError(
                format!("Missing required file: {}", file)
            ));
        }
    }

    // Validate file sizes
    let weights_path = model_path.join("model.safetensors");
    let weights_size = std::fs::metadata(&weights_path)?.len();
    
    // SmolLM-135M should be ~540MB in FP16
    if weights_size < 500_000_000 || weights_size > 600_000_000 {
        return Err(GraphError::StorageError(
            format!("Unexpected model size: {} bytes", weights_size)
        ));
    }

    Ok(())
}
```

### Phase 2: Candle Integration (Day 1-2)

#### Step 2.1: Create Model Wrapper
```rust
// src/models/smollm/mod.rs
use candle_core::{Device, Tensor, DType, Result as CandleResult};
use candle_nn::{linear, embedding, LayerNorm, Linear, Embedding};
use candle_transformers::models::llama::{Llama, LlamaConfig, Cache};
use safetensors::SafeTensors;

pub struct SmolLM135M {
    model: Llama,
    device: Device,
    cache: Cache,
    config: LlamaConfig,
}

impl SmolLM135M {
    pub fn load(model_path: &Path, device: Device) -> Result<Self> {
        // Load configuration
        let config_path = model_path.join("config.json");
        let config_json = std::fs::read_to_string(config_path)?;
        let config: LlamaConfig = serde_json::from_str(&config_json)?;

        // Load model weights
        let weights_path = model_path.join("model.safetensors");
        let weights_data = std::fs::read(weights_path)?;
        let weights = SafeTensors::deserialize(&weights_data)?;
        
        // Create variable builder
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let vb = candle_nn::VarBuilder::from_tensors(weights, dtype, &device);

        // Initialize model
        let model = Llama::load(vb, &config)?;
        let cache = Cache::new(true, dtype, &config, &device)?;

        Ok(Self {
            model,
            device,
            cache,
            config,
        })
    }

    pub fn generate_text(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Tokenize input
        let tokens = self.tokenize(prompt)?;
        let mut generated_tokens = tokens.clone();
        
        // Generation loop
        for _ in 0..max_tokens {
            let input_ids = Tensor::from_slice(
                &generated_tokens, 
                (1, generated_tokens.len()), 
                &self.device
            )?;

            // Forward pass
            let logits = self.model.forward(&input_ids, 0, &mut self.cache)?;
            
            // Sample next token (greedy for now)
            let next_token = self.sample_token(&logits)?;
            
            // Check for end token
            if next_token == self.config.eos_token_id.unwrap_or(2) {
                break;
            }
            
            generated_tokens.push(next_token);
        }

        // Decode tokens to text
        self.detokenize(&generated_tokens[tokens.len()..])
    }

    fn sample_token(&self, logits: &Tensor) -> Result<u32> {
        // Simple greedy sampling - get token with highest probability
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        
        let max_idx = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap();
            
        Ok(max_idx as u32)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        // Implementation depends on tokenizer integration
        // This is a placeholder
        todo!("Implement tokenization")
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String> {
        // Implementation depends on tokenizer integration
        // This is a placeholder
        todo!("Implement detokenization")
    }
}

impl CandleModel for SmolLM135M {
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        // Delegate to internal Llama model
        // Note: This is simplified - real implementation needs cache management
        todo!("Implement CandleModel trait")
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn reset_kv_cache(&mut self) {
        self.cache = Cache::new(true, DType::F32, &self.config, &self.device)
            .expect("Failed to reset cache");
    }
}
```

#### Step 2.2: Tokenizer Integration
```rust
// src/models/smollm/tokenizer.rs
use tokenizers::{Tokenizer, Encoding};

pub struct SmolLMTokenizer {
    tokenizer: Tokenizer,
}

impl SmolLMTokenizer {
    pub fn load(tokenizer_path: &Path) -> Result<Self> {
        let tokenizer_file = tokenizer_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| GraphError::StorageError(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| GraphError::StorageError(format!("Tokenization failed: {}", e)))?;
            
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let text = self.tokenizer
            .decode(token_ids, false)
            .map_err(|e| GraphError::StorageError(format!("Detokenization failed: {}", e)))?;
            
        Ok(text)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}
```

### Phase 3: WASM Optimization (Day 2-3)

#### Step 3.1: Quantization Pipeline
```rust
// src/models/smollm/quantization.rs
use candle_core::{Tensor, DType, Device};

pub struct SmolLMQuantizer;

impl SmolLMQuantizer {
    pub fn quantize_to_int4(model_path: &Path, output_path: &Path) -> Result<()> {
        println!("Quantizing SmolLM-135M to Int4...");
        
        // Load original weights
        let weights_path = model_path.join("model.safetensors");
        let weights_data = std::fs::read(weights_path)?;
        let tensors = SafeTensors::deserialize(&weights_data)?;
        
        let device = Device::Cpu;
        let mut quantized_tensors = HashMap::new();
        
        // Quantize each tensor
        for (name, tensor_view) in tensors.tensors() {
            if Self::should_quantize_tensor(name) {
                let tensor = tensor_view.load(&device)?;
                let quantized = Self::quantize_tensor_int4(&tensor)?;
                quantized_tensors.insert(name.to_string(), quantized);
            } else {
                // Keep as FP16 for important tensors (embeddings, layernorms)
                let tensor = tensor_view.load(&device)?;
                quantized_tensors.insert(name.to_string(), tensor);
            }
        }
        
        // Save quantized model
        Self::save_quantized_model(&quantized_tensors, output_path)?;
        
        // Copy other files
        for file in &["config.json", "tokenizer.json", "tokenizer_config.json"] {
            std::fs::copy(
                model_path.join(file),
                output_path.join(file)
            )?;
        }
        
        println!("Quantization complete. Output: {:?}", output_path);
        Ok(())
    }

    fn should_quantize_tensor(name: &str) -> bool {
        // Don't quantize embeddings and layer norms for quality
        !name.contains("embed_tokens") && 
        !name.contains("norm") &&
        !name.contains("lm_head")
    }

    fn quantize_tensor_int4(tensor: &Tensor) -> Result<Tensor> {
        // Group-wise quantization with 128 elements per group
        let group_size = 128;
        let shape = tensor.shape();
        
        // This is a simplified implementation
        // Real implementation would use proper Int4 packing
        let min_val = tensor.min(0)?.min(0)?.to_scalar::<f32>()?;
        let max_val = tensor.max(0)?.max(0)?.to_scalar::<f32>()?;
        
        let scale = (max_val - min_val) / 15.0; // 4-bit range: 0-15
        let quantized = ((tensor - min_val)? / scale)?
            .round()?
            .clamp(0.0, 15.0)?
            .to_dtype(DType::U8)?; // Store as U8, interpret as 4-bit
            
        Ok(quantized)
    }

    fn save_quantized_model(tensors: &HashMap<String, Tensor>, output_path: &Path) -> Result<()> {
        std::fs::create_dir_all(output_path)?;
        
        // Save in custom format optimized for loading
        let quantized_path = output_path.join("model_q4.bin");
        let mut file = std::fs::File::create(quantized_path)?;
        
        // Write header
        file.write_all(b"SMOLLM_Q4")?; // Magic bytes
        file.write_all(&(tensors.len() as u32).to_le_bytes())?;
        
        // Write tensors
        for (name, tensor) in tensors {
            // Write name length and name
            file.write_all(&(name.len() as u32).to_le_bytes())?;
            file.write_all(name.as_bytes())?;
            
            // Write tensor data
            Self::write_tensor_to_file(&mut file, tensor)?;
        }
        
        Ok(())
    }

    fn write_tensor_to_file(file: &mut std::fs::File, tensor: &Tensor) -> Result<()> {
        let shape = tensor.shape();
        let dtype = tensor.dtype();
        
        // Write shape
        file.write_all(&(shape.dims().len() as u32).to_le_bytes())?;
        for dim in shape.dims() {
            file.write_all(&(*dim as u32).to_le_bytes())?;
        }
        
        // Write dtype
        let dtype_id = match dtype {
            DType::F32 => 0u8,
            DType::F16 => 1u8,
            DType::U8 => 2u8,
            _ => return Err(GraphError::StorageError("Unsupported dtype".to_string())),
        };
        file.write_all(&[dtype_id])?;
        
        // Write data
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;
        for value in data {
            file.write_all(&value.to_le_bytes())?;
        }
        
        Ok(())
    }
}
```

#### Step 3.2: WASM Bundle Creation
```bash
# scripts/build_smollm_135m_wasm.sh
#!/bin/bash
set -e

MODEL_NAME="smollm_135m"
OUTPUT_DIR="./dist/wasm/models/$MODEL_NAME"

echo "Building WASM bundle for SmolLM-135M..."

# Create quantized model
cargo run --bin quantize_model -- \
    --model-path ./models/smollm/135m \
    --output-path ./models/smollm/135m_q4 \
    --format int4

# Build WASM with model-specific features
wasm-pack build \
    --target web \
    --out-dir "$OUTPUT_DIR" \
    --features "wasm,smollm_135m" \
    --release

# Optimize WASM
wasm-opt -Os -o "$OUTPUT_DIR/llmkg_bg.wasm" "$OUTPUT_DIR/llmkg_bg.wasm"

# Copy quantized model
cp ./models/smollm/135m_q4/model_q4.bin "$OUTPUT_DIR/"
cp ./models/smollm/135m_q4/*.json "$OUTPUT_DIR/"

# Create model manifest
cat > "$OUTPUT_DIR/model_manifest.json" << EOF
{
  "name": "SmolLM-135M",
  "version": "1.0.0",
  "architecture": "llama",
  "parameters": 135000000,
  "quantization": "int4",
  "bundle_size_mb": $(du -m "$OUTPUT_DIR/model_q4.bin" | cut -f1),
  "files": {
    "weights": "model_q4.bin",
    "config": "config.json",
    "tokenizer": "tokenizer.json"
  },
  "performance": {
    "target_tokens_per_second": 100,
    "max_memory_mb": 300,
    "context_length": 2048
  }
}
EOF

echo "WASM bundle created at: $OUTPUT_DIR"
echo "Bundle size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
```

### Phase 4: Integration Testing (Day 3-4)

#### Step 4.1: Unit Tests
```rust
// tests/models/smollm_135m_test.rs
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio_test;

    #[tokio::test]
    async fn test_smollm_135m_loading() {
        let model_path = download_test_model("smollm-135m").await;
        let device = Device::Cpu;
        
        let model = SmolLM135M::load(&model_path, device).unwrap();
        assert!(model.config.vocab_size == 49152);
    }

    #[tokio::test]
    async fn test_smollm_135m_inference() {
        let mut model = load_test_model("smollm-135m").await;
        
        let prompt = "The capital of France is";
        let result = model.generate_text(prompt, 10).unwrap();
        
        assert!(result.len() > prompt.len());
        assert!(result.starts_with(prompt));
    }

    #[tokio::test]
    async fn test_smollm_135m_quantization() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = download_test_model("smollm-135m").await;
        
        SmolLMQuantizer::quantize_to_int4(&model_path, temp_dir.path()).unwrap();
        
        let quantized_model_path = temp_dir.path().join("model_q4.bin");
        assert!(quantized_model_path.exists());
        
        // Check that quantized model is smaller
        let original_size = std::fs::metadata(model_path.join("model.safetensors")).unwrap().len();
        let quantized_size = std::fs::metadata(quantized_model_path).unwrap().len();
        
        assert!(quantized_size < original_size / 2); // Should be <50% of original
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test]
    async fn test_wasm_model_loading() {
        use wasm_bindgen_test::*;
        
        let model = WasmModelLoader::new()
            .load_quantized_model("/models/smollm_135m/model_q4.bin")
            .await
            .unwrap();
            
        assert!(model.is_ready());
    }
}
```

#### Step 4.2: Performance Benchmarking
```rust
// benches/smollm_135m_bench.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn benchmark_smollm_135m(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut model = rt.block_on(load_test_model("smollm-135m"));
    
    let prompts = vec![
        "Hello",
        "The quick brown fox",
        "In the year 2024",
        "Machine learning is",
    ];
    
    for prompt in prompts {
        c.bench_with_input(
            BenchmarkId::new("smollm_135m_inference", prompt),
            &prompt,
            |b, prompt| {
                b.iter(|| {
                    model.generate_text(prompt, 20).unwrap()
                });
            },
        );
    }
    
    // Memory usage benchmark
    c.bench_function("smollm_135m_memory_usage", |b| {
        b.iter(|| {
            let memory_before = get_memory_usage();
            let _model = rt.block_on(load_test_model("smollm-135m"));
            let memory_after = get_memory_usage();
            memory_after - memory_before
        });
    });
}

criterion_group!(benches, benchmark_smollm_135m);
criterion_main!(benches);
```

### Phase 5: Production Integration (Day 4-5)

#### Step 5.1: Model Registry Integration
```rust
// Update src/models/registry.rs for SmolLM-135M
impl EnhancedModelRegistry {
    fn populate_smollm_models(&mut self) {
        let smollm_135m = ModelMetadata {
            name: "SmolLM-135M".to_string(),
            family: "SmolLM".to_string(),
            parameters: 135_000_000,
            size_category: ModelSize::Small,
            huggingface_id: "HuggingFaceTB/SmolLM-135M".to_string(),
            architecture: "Transformer".to_string(),
            capabilities: ModelCapabilities {
                text_generation: true,
                instruction_following: false,
                chat: false,
                code_generation: true,
                reasoning: true,
                multilingual: false,
            },
            context_length: 2048,
            vocab_size: 49152,
            training_tokens: Some(600_000_000_000),
            release_date: "2024-07".to_string(),
            license: "Apache-2.0".to_string(),
            description: "SmolLM-135M is a compact language model with 135M parameters, optimized for efficiency and edge deployment.".to_string(),
        };
        
        self.models.insert(smollm_135m.huggingface_id.clone(), smollm_135m);
    }
    
    async fn load_smollm_135m(&self) -> Result<Arc<dyn ModelBackend>> {
        let model_path = self.get_model_cache_path("HuggingFaceTB/SmolLM-135M");
        let device = Device::Cpu; // or detect best device
        
        let smollm_model = SmolLM135M::load(&model_path, device)?;
        let tokenizer = SmolLMTokenizer::load(&model_path)?;
        
        let backend = CandleModelBackend::new(
            Box::new(smollm_model),
            Arc::new(tokenizer),
            self.get_model("HuggingFaceTB/SmolLM-135M").unwrap().clone(),
        )?;
        
        Ok(Arc::new(backend))
    }
}
```

#### Step 5.2: API Integration
```rust
// Update src/models/mod.rs to export SmolLM
impl Model {
    pub fn generate_text(&self, prompt: &str, max_tokens: Option<u32>) -> Result<String> {
        if !self.loaded {
            return Err(GraphError::InvalidState("Model not loaded".to_string()));
        }
        
        // Use the actual Candle backend instead of placeholder
        match self.metadata.family.as_str() {
            "SmolLM" => {
                let backend = self.get_backend()?;
                backend.generate_text(prompt, &self.config)
            },
            _ => {
                // Fallback to placeholder for now
                Ok(format!("Generated response for: {}", prompt))
            }
        }
    }
}
```

## Success Criteria

### Functional Requirements
- [ ] Successfully download SmolLM-135M from HuggingFace Hub
- [ ] Load model using Candle with <5 second initialization
- [ ] Generate coherent text with <30ms first token latency
- [ ] Achieve >80 tokens/second throughput on modern CPU
- [ ] Create Int4 quantized version <70MB
- [ ] Deploy to WASM with full functionality

### Performance Requirements
- [ ] Memory usage <300MB peak
- [ ] WASM bundle size <70MB total
- [ ] First token latency <30ms
- [ ] Subsequent tokens <10ms average
- [ ] Text quality comparable to reference implementation

### Integration Requirements
- [ ] Full compatibility with existing LLMKG API
- [ ] Integration with model registry
- [ ] WASM deployment working in major browsers
- [ ] Unit tests passing with >95% coverage
- [ ] Performance benchmarks meeting targets

## Risk Mitigation

### Technical Risks
1. **Candle Compatibility**: Test with multiple Candle versions
2. **WASM Size**: Aggressive quantization if needed
3. **Performance**: Profile and optimize hot paths
4. **Memory Usage**: Implement streaming inference if needed

### Mitigation Strategies
1. **Fallback Implementation**: Keep working placeholder during development
2. **Progressive Quantization**: Start with Int8, move to Int4 if size allows
3. **Device Detection**: Optimize based on available hardware
4. **Error Handling**: Graceful degradation for unsupported features

## Timeline

- **Day 1**: Model download, validation, basic Candle integration
- **Day 2**: Complete inference pipeline, tokenizer integration
- **Day 3**: Quantization, WASM optimization, bundle creation
- **Day 4**: Integration testing, performance benchmarking
- **Day 5**: Production integration, API updates, documentation

This plan provides the foundation for SmolLM-135M integration and serves as a template for other SmolLM variants.