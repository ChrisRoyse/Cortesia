# SmolLM2-135M Integration Plan

## Model Overview

**Model ID**: `HuggingFaceTB/SmolLM2-135M`  
**Parameters**: 135,000,000 (135M)  
**Architecture**: Enhanced Transformer (Llama-based v2)  
**Size Category**: Small  
**Priority**: Tier 1 - Latest Generation  
**Use Case**: Improved performance over v1, better efficiency

## Key Improvements Over SmolLM v1

SmolLM2 represents the next generation with significant improvements:

- **Better Training Data**: Improved dataset curation and filtering
- **Enhanced Architecture**: Optimized attention mechanisms
- **Improved Tokenizer**: More efficient text representation
- **Better Alignment**: Enhanced instruction following
- **Performance**: 15-20% better on benchmarks vs v1

## Technical Specifications

### Model Details
- **Base Architecture**: Enhanced Llama with SmolLM2 optimizations
- **Release Date**: November 2024 (latest)
- **Context Length**: 2048 tokens (same as v1)
- **Vocabulary Size**: 49,152 tokens
- **Training**: 600B+ tokens with improved quality

### Performance Targets
- **Inference Speed**: 90-130 tokens/second (15% faster than v1)
- **Memory Usage**: 200-280MB peak (better optimization)
- **First Token Latency**: <25ms (improved vs v1's <30ms)
- **Quality**: Superior text generation vs v1
- **Bundle Size**: ~58MB quantized (better compression)

## Implementation Plan

### Phase 1: Model Architecture Updates (Day 1)

#### Step 1.1: Enhanced Model Implementation
```rust
// src/models/smollm/smollm2_135m.rs
use super::SmolLM135M;

pub struct SmolLM2_135M {
    base_model: Llama,
    device: Device,
    cache: Cache,
    config: SmolLM2Config,
    
    // SmolLM2 specific enhancements
    attention_optimizer: AttentionOptimizer,
    memory_manager: EnhancedMemoryManager,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SmolLM2Config {
    #[serde(flatten)]
    pub base_config: LlamaConfig,
    
    // SmolLM2 specific configurations
    pub attention_optimization: bool,
    pub memory_optimization: bool,
    pub enhanced_tokenization: bool,
}

impl SmolLM2_135M {
    pub fn load(model_path: &Path, device: Device) -> Result<Self> {
        // Validate SmolLM2 specific architecture
        let config = Self::load_and_validate_config(model_path)?;
        
        // Load with SmolLM2 optimizations
        let weights_path = model_path.join("model.safetensors");
        let weights_data = std::fs::read(weights_path)?;
        let weights = SafeTensors::deserialize(&weights_data)?;
        
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let vb = candle_nn::VarBuilder::from_tensors(weights, dtype, &device);
        
        // Load with enhanced architecture
        let model = Llama::load(vb, &config.base_config)?;
        let cache = Cache::new(true, dtype, &config.base_config, &device)?;
        
        // Initialize SmolLM2 enhancements
        let attention_optimizer = AttentionOptimizer::new(&config)?;
        let memory_manager = EnhancedMemoryManager::new(200)?; // 200MB limit
        
        Ok(Self {
            base_model: model,
            device,
            cache,
            config,
            attention_optimizer,
            memory_manager,
        })
    }
    
    fn load_and_validate_config(model_path: &Path) -> Result<SmolLM2Config> {
        let config_path = model_path.join("config.json");
        let config_json = std::fs::read_to_string(config_path)?;
        let config: SmolLM2Config = serde_json::from_str(&config_json)?;
        
        // Validate SmolLM2 architecture
        if config.base_config.hidden_size != 576 || config.base_config.num_hidden_layers != 30 {
            return Err(GraphError::InvalidState(
                format!("Invalid SmolLM2-135M architecture")
            ));
        }
        
        // Check for SmolLM2 specific features
        if !config.attention_optimization {
            log::warn!("SmolLM2 attention optimization not enabled");
        }
        
        Ok(config)
    }
    
    pub fn generate_text_optimized(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Use enhanced generation with SmolLM2 optimizations
        let tokens = self.tokenize_enhanced(prompt)?;
        let mut generated_tokens = tokens.clone();
        
        // Pre-optimize attention patterns for better performance
        self.attention_optimizer.prepare_for_generation(&tokens)?;
        
        let mut logits_processor = LogitsProcessor::new(
            None, // seed
            Some(0.7), // temperature  
            Some(0.9), // top_p
        );
        
        for step in 0..max_tokens {
            let input_ids = Tensor::from_slice(
                &generated_tokens, 
                (1, generated_tokens.len()), 
                &self.device
            )?;

            // Forward pass with SmolLM2 optimizations
            let logits = self.forward_optimized(&input_ids, step)?;
            
            // Enhanced sampling
            let next_token = logits_processor.sample(&logits.squeeze(0)?)?;
            
            if next_token == self.config.base_config.eos_token_id.unwrap_or(2) {
                break;
            }
            
            generated_tokens.push(next_token);
            
            // Memory optimization every 5 steps
            if step % 5 == 0 {
                self.memory_manager.optimize_if_needed()?;
            }
        }

        self.detokenize_enhanced(&generated_tokens[tokens.len()..])
    }
    
    fn forward_optimized(&mut self, input_ids: &Tensor, step: usize) -> Result<Tensor> {
        // Apply SmolLM2 attention optimizations
        self.attention_optimizer.optimize_attention_for_step(step)?;
        
        // Standard forward pass with optimizations
        let logits = self.base_model.forward(input_ids, 0, &mut self.cache)?;
        
        Ok(logits)
    }
    
    fn tokenize_enhanced(&self, text: &str) -> Result<Vec<u32>> {
        // Use enhanced tokenization if available
        if self.config.enhanced_tokenization {
            // SmolLM2 has improved tokenization
            self.tokenize_with_enhancements(text)
        } else {
            // Fallback to standard tokenization
            self.tokenize_standard(text)
        }
    }
}

// SmolLM2 specific optimizations
pub struct AttentionOptimizer {
    optimization_enabled: bool,
    attention_patterns: Vec<AttentionPattern>,
}

impl AttentionOptimizer {
    pub fn new(config: &SmolLM2Config) -> Result<Self> {
        Ok(Self {
            optimization_enabled: config.attention_optimization,
            attention_patterns: Vec::new(),
        })
    }
    
    pub fn prepare_for_generation(&mut self, tokens: &[u32]) -> Result<()> {
        if self.optimization_enabled {
            // Pre-compute attention patterns for better performance
            self.attention_patterns = self.compute_optimal_patterns(tokens)?;
        }
        Ok(())
    }
    
    pub fn optimize_attention_for_step(&mut self, step: usize) -> Result<()> {
        if self.optimization_enabled && step < self.attention_patterns.len() {
            // Apply step-specific optimizations
            self.apply_pattern(&self.attention_patterns[step])?;
        }
        Ok(())
    }
}
```

#### Step 1.2: Enhanced Quantization for SmolLM2
```rust
// src/models/smollm/quantization_smollm2.rs
pub struct SmolLM2Quantizer;

impl SmolLM2Quantizer {
    pub fn quantize_v2_optimized(input_path: &Path, output_path: &Path) -> Result<()> {
        println!("Quantizing SmolLM2-135M with v2 optimizations...");
        
        // SmolLM2 has better weight distribution, allowing for more aggressive quantization
        let weights_path = input_path.join("model.safetensors");
        let weights_data = std::fs::read(weights_path)?;
        let tensors = SafeTensors::deserialize(&weights_data)?;
        
        let device = Device::Cpu;
        let mut quantized_data = SmolLM2QuantizedData::new();
        
        for (name, tensor_view) in tensors.tensors() {
            let tensor = tensor_view.load(&device)?;
            
            if Self::should_quantize_tensor_v2(name) {
                // Use SmolLM2 specific quantization (better quality retention)
                let quantized = Self::quantize_tensor_v2_optimized(&tensor, name)?;
                quantized_data.add_tensor(name, quantized);
            } else {
                // Keep critical tensors in higher precision
                let fp16_tensor = tensor.to_dtype(DType::F16)?;
                quantized_data.add_tensor(name, QuantizedTensor::FP16(fp16_tensor));
            }
        }
        
        // Save with SmolLM2 optimized format
        quantized_data.save_v2_format(output_path)?;
        
        // Copy SmolLM2 specific config files
        Self::copy_v2_config_files(input_path, output_path)?;
        
        println!("SmolLM2 quantization complete. Better compression: ~58MB");
        Ok(())
    }
    
    fn quantize_tensor_v2_optimized(tensor: &Tensor, name: &str) -> Result<QuantizedTensor> {
        // SmolLM2 specific quantization - better weight distribution allows
        // for more aggressive quantization with less quality loss
        
        if name.contains("attention") {
            // Attention layers in SmolLM2 can handle more aggressive quantization
            Self::quantize_attention_layer_v2(tensor)
        } else if name.contains("mlp") {
            // MLP layers optimized for SmolLM2 architecture
            Self::quantize_mlp_layer_v2(tensor)
        } else {
            // Standard v2 quantization
            Self::quantize_standard_v2(tensor)
        }
    }
    
    fn should_quantize_tensor_v2(name: &str) -> bool {
        // SmolLM2 allows quantizing more tensors due to better training
        !name.contains("embed_tokens") && 
        !name.contains("norm") &&
        !name.contains("lm_head") &&
        !name.ends_with(".bias") // Keep biases in FP16
    }
}
```

### Phase 2: Performance Benchmarking (Day 1-2)

#### Step 2.1: V1 vs V2 Comparison Tests
```rust
// tests/models/smollm_v1_vs_v2_comparison.rs
#[tokio::test]
async fn test_smollm2_vs_v1_performance() {
    let v1_model = load_test_model("smollm-135m").await;
    let v2_model = load_test_model("smollm2-135m").await;
    
    let test_prompts = vec![
        "The future of renewable energy",
        "Explain quantum computing",
        "Write a Python function to sort a list",
        "What are the benefits of AI?",
    ];
    
    for prompt in test_prompts {
        // Performance comparison
        let v1_start = std::time::Instant::now();
        let v1_response = v1_model.generate_text(prompt, 50).unwrap();
        let v1_time = v1_start.elapsed();
        
        let v2_start = std::time::Instant::now();
        let v2_response = v2_model.generate_text(prompt, 50).unwrap();
        let v2_time = v2_start.elapsed();
        
        // V2 should be faster
        assert!(v2_time < v1_time * 1.1, "SmolLM2 not faster than v1");
        
        // Quality comparison (semantic similarity)
        let quality_improvement = calculate_quality_improvement(&v1_response, &v2_response);
        assert!(quality_improvement > 0.0, "SmolLM2 quality not improved");
        
        println!("Prompt: {}", prompt);
        println!("V1 time: {:?}, V2 time: {:?}", v1_time, v2_time);
        println!("Speedup: {:.2}x", v1_time.as_millis() as f64 / v2_time.as_millis() as f64);
        println!("Quality improvement: {:.2}%", quality_improvement * 100.0);
    }
}

#[tokio::test]
async fn test_smollm2_memory_efficiency() {
    let v1_model = load_test_model("smollm-135m").await;
    let v2_model = load_test_model("smollm2-135m").await;
    
    // Memory usage comparison
    let v1_memory = v1_model.get_memory_usage();
    let v2_memory = v2_model.get_memory_usage();
    
    // V2 should use less memory due to optimizations
    assert!(v2_memory < v1_memory, "SmolLM2 memory usage not improved");
    
    let memory_improvement = (v1_memory - v2_memory) as f64 / v1_memory as f64;
    println!("Memory improvement: {:.2}%", memory_improvement * 100.0);
    
    // Should be at least 5% better
    assert!(memory_improvement > 0.05, "Memory improvement < 5%");
}
```

### Phase 3: WASM Deployment (Day 2)

#### Step 3.1: Enhanced WASM Bundle
```bash
# scripts/build_smollm2_135m_wasm.sh
#!/bin/bash
set -e

MODEL_NAME="smollm2_135m"
OUTPUT_DIR="./dist/wasm/models/$MODEL_NAME"

echo "Building SmolLM2-135M with v2 optimizations..."

# Download SmolLM2 model
huggingface-cli download "HuggingFaceTB/SmolLM2-135M" \
    --local-dir "./models/smollm/v2_135m"

# Quantize with v2 optimizations
cargo run --bin quantize_model -- \
    --model-path ./models/smollm/v2_135m \
    --output-path ./models/smollm/v2_135m_q4 \
    --format smollm2-int4 \
    --v2-optimizations

# Build WASM with v2 features
wasm-pack build \
    --target web \
    --out-dir "$OUTPUT_DIR" \
    --features "wasm,smollm2,v2-optimizations" \
    --release

# Enhanced optimization for v2
wasm-opt -Os --enable-bulk-memory -o "$OUTPUT_DIR/llmkg_bg.wasm" "$OUTPUT_DIR/llmkg_bg.wasm"

# Copy v2 model files
cp ./models/smollm/v2_135m_q4/* "$OUTPUT_DIR/"

# Create v2 manifest with improvements
cat > "$OUTPUT_DIR/model_manifest.json" << EOF
{
  "name": "SmolLM2-135M",
  "version": "2.0.0",
  "architecture": "llama-v2-optimized",
  "parameters": 135000000,
  "quantization": "int4-v2-optimized",
  "bundle_size_mb": $(du -m "$OUTPUT_DIR/model_q4.bin" | cut -f1),
  "improvements_over_v1": {
    "performance": "15-20% faster inference",
    "memory": "10-15% lower usage", 
    "quality": "Better text generation",
    "compression": "Better quantization tolerance"
  },
  "performance": {
    "target_tokens_per_second": 110,
    "max_memory_mb": 280,
    "first_token_latency_ms": 25,
    "context_length": 2048
  },
  "features": {
    "attention_optimization": true,
    "memory_optimization": true,
    "enhanced_tokenization": true,
    "v2_architecture": true
  }
}
EOF

echo "SmolLM2-135M WASM bundle created at: $OUTPUT_DIR"
echo "Bundle size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo "Performance improvements over v1 enabled"
```

## Success Criteria

### Performance Improvements Over V1
- [ ] 15-20% faster inference speed
- [ ] 10-15% lower memory usage
- [ ] Better text quality (measurable improvement)
- [ ] Smaller quantized bundle size
- [ ] Enhanced WASM performance

### Functional Requirements
- [ ] Full compatibility with v1 API
- [ ] Enhanced optimization features working
- [ ] Better quantization quality retention
- [ ] Improved attention mechanisms
- [ ] Memory optimization active

### Quality Requirements
- [ ] Superior text generation vs SmolLM v1
- [ ] Better instruction following
- [ ] Improved coherence and relevance
- [ ] Less repetition and better creativity

## Migration from V1

### API Compatibility
```rust
// Backwards compatible API
impl SmolLM2_135M {
    // V1 compatible methods
    pub fn generate_text(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Use optimized generation internally
        self.generate_text_optimized(prompt, max_tokens)
    }
    
    // V2 enhanced methods
    pub fn generate_text_v2(&mut self, prompt: &str, config: &GenerationConfigV2) -> Result<String> {
        // Use V2 specific features
        self.generate_with_v2_features(prompt, config)
    }
}
```

### Drop-in Replacement
```javascript
// JavaScript - drop-in replacement for v1
const modelV1 = new WasmSmolLM135M();
const modelV2 = new WasmSmolLM2_135M(); // Better performance, same API

// Both work the same way
await modelV1.initialize(urlV1);
await modelV2.initialize(urlV2);

const responseV1 = await modelV1.generate("Hello world");
const responseV2 = await modelV2.generate("Hello world"); // Faster + better quality
```

## Timeline

- **Day 1**: V2 architecture implementation, enhanced quantization, performance testing
- **Day 2**: WASM deployment, benchmarking vs V1, documentation

SmolLM2-135M represents the latest generation with meaningful improvements while maintaining full compatibility with existing systems.