# TinyLlama-1.1B Integration Plan

## Model Overview

**Model ID**: `TinyLlama/TinyLlama_v1.1`  
**Parameters**: 1,100,000,000 (1.1B)  
**Architecture**: Llama 2 Architecture with optimizations  
**Size Category**: Large  
**Priority**: Tier 2 - Large Model Performance  
**Use Case**: Complex reasoning, code generation, comprehensive text tasks

## Technical Specifications

### Model Details
- **Architecture**: Llama 2 with TinyLlama optimizations
- **Context Length**: 2048 tokens
- **Vocabulary Size**: 32,000 tokens (different from SmolLM)
- **Training Data**: ~1 trillion tokens from SlimPajama dataset
- **Attention**: Grouped Query Attention (GQA) for efficiency
- **License**: Apache 2.0
- **Release Date**: Early 2024

### Key Architectural Differences from SmolLM
- **Larger Model**: 1.1B vs 135M-1.7B parameters
- **Different Tokenizer**: SentencePiece vs SmolLM tokenizer  
- **GQA Attention**: More efficient than standard multi-head attention
- **Llama 2 Base**: Direct Llama 2 architecture vs SmolLM modifications

### Performance Targets
- **Inference Speed**: 15-30 tokens/second (CPU), 60-100 (GPU)
- **Memory Usage**: 1.2-1.8GB peak (native), <1GB (WASM quantized) 
- **First Token Latency**: <120ms (native), <250ms (WASM)
- **Subsequent Tokens**: <35ms average
- **WASM Bundle Size**: ~450MB quantized (Int4 + streaming)

## Implementation Strategy

TinyLlama requires a different approach due to its Llama 2 base and larger size:

1. **Llama 2 Integration**: Use Candle's Llama 2 implementation
2. **SentencePiece Tokenizer**: Different tokenization approach
3. **GQA Attention**: Specialized attention mechanism
4. **Aggressive Optimization**: Required for practical deployment

### Phase 1: Llama 2 Architecture Integration (Day 1-2)

#### Step 1.1: TinyLlama-Specific Implementation
```rust
// src/models/tinyllama/tinyllama_1_1b.rs
use candle_core::{Device, Tensor, DType, Result as CandleResult};
use candle_transformers::models::llama2_c::{Llama, Config as LlamaConfig};
use candle_transformers::generation::LogitsProcessor;

pub struct TinyLlama1_1B {
    model: Llama,
    device: Device,
    config: TinyLlamaConfig,
    tokenizer: SentencePieceTokenizer,
    
    // TinyLlama optimizations
    gqa_optimizer: GroupedQueryAttentionOptimizer,
    memory_manager: TinyLlamaMemoryManager,
    streaming_state: Option<StreamingState>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TinyLlamaConfig {
    #[serde(flatten)]
    pub llama_config: LlamaConfig,
    
    // TinyLlama specific
    pub use_gqa: bool,
    pub num_key_value_heads: usize,
    pub rope_scaling: Option<RopeScaling>,
}

impl TinyLlama1_1B {
    pub fn load(model_path: &Path, device: Device) -> Result<Self> {
        // Load TinyLlama config (different from SmolLM)
        let config = Self::load_tinyllama_config(model_path)?;
        
        // Validate TinyLlama architecture
        Self::validate_tinyllama_architecture(&config)?;
        
        // Load model weights with Llama 2 loader
        let weights_path = model_path.join("pytorch_model.bin");
        let weights = Self::load_pytorch_weights(&weights_path, &device)?;
        
        // Create TinyLlama model
        let model = Llama::load(weights, &config.llama_config)?;
        
        // Load SentencePiece tokenizer (different from SmolLM)
        let tokenizer = SentencePieceTokenizer::load(model_path)?;
        
        // Initialize TinyLlama optimizations
        let gqa_optimizer = GroupedQueryAttentionOptimizer::new(&config)?;
        let memory_manager = TinyLlamaMemoryManager::new(1200)?; // 1.2GB limit
        
        Ok(Self {
            model,
            device,
            config,
            tokenizer,
            gqa_optimizer,
            memory_manager,
            streaming_state: None,
        })
    }
    
    fn load_tinyllama_config(model_path: &Path) -> Result<TinyLlamaConfig> {
        let config_path = model_path.join("config.json");
        let config_json = std::fs::read_to_string(config_path)?;
        let config: TinyLlamaConfig = serde_json::from_str(&config_json)?;
        
        Ok(config)
    }
    
    fn validate_tinyllama_architecture(config: &TinyLlamaConfig) -> Result<()> {
        // TinyLlama 1.1B specific validation
        if config.llama_config.hidden_size != 2048 {
            return Err(GraphError::InvalidState(
                format!("Invalid TinyLlama hidden_size: {}", config.llama_config.hidden_size)
            ));
        }
        
        if config.llama_config.num_hidden_layers != 22 {
            return Err(GraphError::InvalidState(
                format!("Invalid TinyLlama num_layers: {}", config.llama_config.num_hidden_layers)
            ));
        }
        
        // Validate GQA configuration
        if config.use_gqa && config.num_key_value_heads == 0 {
            return Err(GraphError::InvalidState(
                "GQA enabled but num_key_value_heads not set".to_string()
            ));
        }
        
        Ok(())
    }
    
    fn load_pytorch_weights(weights_path: &Path, device: &Device) -> Result<VarBuilder> {
        // TinyLlama uses PyTorch format, not SafeTensors
        use candle_nn::VarBuilder;
        
        if weights_path.exists() {
            // Load PyTorch format
            let weights_data = std::fs::read(weights_path)?;
            let weights = candle_core::pickle::read_pth_tensor_info(&weights_data)?;
            
            let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
            Ok(VarBuilder::from_pth(&weights_data, dtype, device)?)
        } else {
            // Try SafeTensors as fallback
            let safetensors_path = weights_path.parent().unwrap().join("model.safetensors");
            if safetensors_path.exists() {
                let weights_data = std::fs::read(safetensors_path)?;
                let weights = safetensors::SafeTensors::deserialize(&weights_data)?;
                
                let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
                Ok(VarBuilder::from_tensors(weights, dtype, device))
            } else {
                Err(GraphError::StorageError("No model weights found".to_string()))
            }
        }
    }
    
    pub fn generate_with_gqa(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Use GQA optimizations for better efficiency
        let tokens = self.tokenizer.encode(prompt)?;
        let mut generated_tokens = tokens.clone();
        
        // Initialize GQA state
        self.gqa_optimizer.prepare_generation(&tokens)?;
        
        let mut logits_processor = LogitsProcessor::new(
            None,
            Some(0.7), // temperature
            Some(0.9), // top_p
        );
        
        for step in 0..max_tokens {
            let input_ids = Tensor::from_slice(
                &generated_tokens,
                (1, generated_tokens.len()),
                &self.device,
            )?;
            
            // Forward pass with GQA optimization
            let logits = self.forward_with_gqa(&input_ids, step)?;
            
            let next_token = logits_processor.sample(&logits.squeeze(0)?)?;
            
            if next_token == 2 { // EOS token for TinyLlama
                break;
            }
            
            generated_tokens.push(next_token);
            
            // Memory management for large model
            if step % 10 == 0 {
                self.memory_manager.optimize_memory()?;
            }
        }
        
        // Decode with SentencePiece
        let generated_text = self.tokenizer.decode(&generated_tokens[tokens.len()..])?;
        Ok(generated_text)
    }
    
    fn forward_with_gqa(&mut self, input_ids: &Tensor, step: usize) -> Result<Tensor> {
        // Apply GQA optimizations
        self.gqa_optimizer.optimize_for_step(step)?;
        
        // Forward pass through Llama model
        let logits = self.model.forward(input_ids)?;
        
        Ok(logits)
    }
}

// SentencePiece tokenizer integration (different from SmolLM)
pub struct SentencePieceTokenizer {
    processor: sentencepiece::SentencePieceProcessor,
    vocab_size: usize,
    bos_token: u32,
    eos_token: u32,
    pad_token: u32,
}

impl SentencePieceTokenizer {
    pub fn load(model_path: &Path) -> Result<Self> {
        let tokenizer_path = model_path.join("tokenizer.model");
        
        if !tokenizer_path.exists() {
            return Err(GraphError::StorageError(
                "SentencePiece tokenizer model not found".to_string()
            ));
        }
        
        let processor = sentencepiece::SentencePieceProcessor::open(tokenizer_path)
            .map_err(|e| GraphError::StorageError(format!("Failed to load tokenizer: {}", e)))?;
        
        let vocab_size = processor.get_piece_size();
        
        Ok(Self {
            processor,
            vocab_size,
            bos_token: 1,  // TinyLlama BOS token
            eos_token: 2,  // TinyLlama EOS token  
            pad_token: 0,  // TinyLlama PAD token
        })
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let pieces = self.processor.encode(text)
            .map_err(|e| GraphError::StorageError(format!("Tokenization failed: {}", e)))?;
            
        let token_ids: Vec<u32> = pieces.iter().map(|&id| id as u32).collect();
        Ok(token_ids)
    }
    
    pub fn decode(&self, token_ids: &[u32]) -> Result<String> {
        let ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        let text = self.processor.decode(&ids)
            .map_err(|e| GraphError::StorageError(format!("Detokenization failed: {}", e)))?;
            
        Ok(text)
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// Grouped Query Attention optimizer
pub struct GroupedQueryAttentionOptimizer {
    enabled: bool,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: HashMap<usize, (Tensor, Tensor)>, // layer -> (k_cache, v_cache)
}

impl GroupedQueryAttentionOptimizer {
    pub fn new(config: &TinyLlamaConfig) -> Result<Self> {
        Ok(Self {
            enabled: config.use_gqa,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.llama_config.hidden_size / config.llama_config.num_attention_heads,
            kv_cache: HashMap::new(),
        })
    }
    
    pub fn prepare_generation(&mut self, tokens: &[u32]) -> Result<()> {
        if self.enabled {
            // Pre-allocate KV cache for GQA
            self.kv_cache.clear();
            
            // Initialize cache for all layers
            for layer_idx in 0..22 { // TinyLlama has 22 layers
                let seq_len = tokens.len();
                let k_cache = Tensor::zeros((1, self.num_kv_heads, seq_len, self.head_dim), DType::F32, &Device::Cpu)?;
                let v_cache = Tensor::zeros((1, self.num_kv_heads, seq_len, self.head_dim), DType::F32, &Device::Cpu)?;
                
                self.kv_cache.insert(layer_idx, (k_cache, v_cache));
            }
        }
        
        Ok(())
    }
    
    pub fn optimize_for_step(&mut self, step: usize) -> Result<()> {
        if self.enabled {
            // Update KV cache sizes for current step
            for (_, (k_cache, v_cache)) in self.kv_cache.iter_mut() {
                // Extend cache if needed
                // This is where GQA-specific optimizations would go
            }
        }
        
        Ok(())
    }
}
```

#### Step 1.2: PyTorch Weight Loading
```rust
// src/models/tinyllama/pytorch_loader.rs
use candle_core::{Device, DType, Tensor};
use std::collections::HashMap;

pub struct PyTorchWeightLoader;

impl PyTorchWeightLoader {
    pub fn load_tinyllama_weights(model_path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        let weights_path = model_path.join("pytorch_model.bin");
        
        if weights_path.exists() {
            Self::load_from_pytorch(&weights_path, device)
        } else {
            // Try model shards
            Self::load_from_shards(model_path, device)
        }
    }
    
    fn load_from_pytorch(weights_path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        // Load PyTorch pickle format
        let weights_data = std::fs::read(weights_path)?;
        
        // Parse PyTorch format with candle
        let tensors = candle_core::pickle::read_pth_tensor_info(&weights_data)?;
        
        let mut loaded_tensors = HashMap::new();
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        
        for (name, tensor_info) in tensors {
            let tensor = Tensor::from_raw_buffer(
                &tensor_info.data,
                tensor_info.dtype,
                &tensor_info.shape,
                device,
            )?;
            
            // Convert to target dtype if needed
            let tensor = if tensor.dtype() != dtype {
                tensor.to_dtype(dtype)?
            } else {
                tensor
            };
            
            loaded_tensors.insert(name, tensor);
        }
        
        Ok(loaded_tensors)
    }
    
    fn load_from_shards(model_path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        // TinyLlama might be split into multiple files
        let mut all_tensors = HashMap::new();
        
        // Look for shard files
        for entry in std::fs::read_dir(model_path)? {
            let entry = entry?;
            let file_name = entry.file_name().to_string_lossy();
            
            if file_name.starts_with("pytorch_model-") && file_name.ends_with(".bin") {
                let shard_tensors = Self::load_from_pytorch(&entry.path(), device)?;
                all_tensors.extend(shard_tensors);
            }
        }
        
        if all_tensors.is_empty() {
            return Err(GraphError::StorageError("No PyTorch model weights found".to_string()));
        }
        
        Ok(all_tensors)
    }
    
    pub fn convert_to_safetensors(pytorch_path: &Path, output_path: &Path) -> Result<()> {
        // Convert PyTorch weights to SafeTensors for faster loading
        let device = Device::Cpu;
        let tensors = Self::load_tinyllama_weights(pytorch_path, &device)?;
        
        // Save as SafeTensors
        let safetensors_path = output_path.join("model.safetensors");
        
        // Convert tensors to SafeTensors format
        let mut tensor_data = Vec::new();
        let mut metadata = HashMap::new();
        
        for (name, tensor) in tensors {
            let tensor_bytes = tensor.to_vec1::<f32>()?;
            let byte_data: Vec<u8> = tensor_bytes.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            
            tensor_data.extend(byte_data);
            metadata.insert(name, tensor.shape().dims().to_vec());
        }
        
        // Write SafeTensors file
        std::fs::write(safetensors_path, tensor_data)?;
        
        println!("Converted TinyLlama weights to SafeTensors format");
        Ok(())
    }
}
```

### Phase 2: Streaming & Quantization (Day 2-3)

#### Step 2.1: TinyLlama Streaming Implementation
```rust
// src/models/tinyllama/streaming.rs
pub struct StreamingTinyLlama {
    core_components: TinyLlamaCoreComponents,
    layer_loader: TinyLlamaLayerLoader,
    layer_cache: LruCache<usize, TransformerLayer>,
    streaming_config: StreamingConfig,
    device: Device,
}

pub struct TinyLlamaCoreComponents {
    embed_tokens: Embedding,
    norm: RMSNorm,
    lm_head: Linear,
    tokenizer: SentencePieceTokenizer,
}

impl StreamingTinyLlama {
    pub fn load_streaming(model_path: &Path, device: Device, config: StreamingConfig) -> Result<Self> {
        // Load core components that are always needed
        let core_components = Self::load_core_components(model_path, &device)?;
        
        // Initialize layer streaming
        let layer_loader = TinyLlamaLayerLoader::new(model_path, &device)?;
        
        // Cache for active layers (keep 2-3 layers in memory)
        let layer_cache = LruCache::new(NonZeroUsize::new(config.max_cached_layers).unwrap());
        
        Ok(Self {
            core_components,
            layer_loader,
            layer_cache,
            streaming_config: config,
            device,
        })
    }
    
    pub fn generate_streaming(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let tokens = self.core_components.tokenizer.encode(prompt)?;
        let mut generated_tokens = tokens.clone();
        
        // Pre-load first few layers
        self.preload_layers(&[0, 1, 21]).await?; // First 2 + last layer
        
        for step in 0..max tokens {
            // Ensure required layers are loaded
            let required_layers = self.get_required_layers_for_step(step);
            self.ensure_layers_loaded(&required_layers).await?;
            
            // Forward pass through streaming layers
            let next_token = self.forward_streaming(&generated_tokens)?;
            
            if next_token == 2 { // TinyLlama EOS
                break;
            }
            
            generated_tokens.push(next_token);
            
            // Adaptive layer management
            if step % 5 == 0 {
                self.manage_layer_cache(step)?;
            }
        }
        
        let response = self.core_components.tokenizer.decode(&generated_tokens[tokens.len()..])?;
        Ok(response)
    }
    
    async fn forward_streaming(&mut self, tokens: &[u32]) -> Result<u32> {
        let input_ids = Tensor::from_slice(tokens, (1, tokens.len()), &self.device)?;
        
        // Embedding
        let mut hidden_states = self.core_components.embed_tokens.forward(&input_ids)?;
        
        // Process through 22 layers with streaming
        for layer_idx in 0..22 {
            // Get layer (load if not cached)
            let layer = self.get_or_load_layer(layer_idx).await?;
            
            // Forward through layer
            hidden_states = layer.forward(&hidden_states)?;
            
            // Memory management for large model
            if layer_idx > 2 && layer_idx % 4 == 0 {
                self.maybe_unload_distant_layers(layer_idx)?;
            }
        }
        
        // Final norm and projection
        let hidden_states = self.core_components.norm.forward(&hidden_states)?;
        let logits = self.core_components.lm_head.forward(&hidden_states)?;
        
        // Sample (greedy for now)
        let last_logits = logits.get(0)?.get(tokens.len() - 1)?;
        let token_id = last_logits.argmax(0)?.to_scalar::<u32>()?;
        
        Ok(token_id)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub max_cached_layers: usize,
    pub memory_limit_mb: usize,
    pub prefetch_layers: usize,
    pub unload_threshold: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_cached_layers: 3,
            memory_limit_mb: 1000, // 1GB limit
            prefetch_layers: 2,
            unload_threshold: 5, // Unload layers not used for 5 steps
        }
    }
}
```

### Phase 3: WASM Deployment (Day 3-4)

```bash
# scripts/build_tinyllama_1_1b_wasm.sh
#!/bin/bash
set -e

MODEL_NAME="tinyllama_1_1b"
OUTPUT_DIR="./dist/wasm/models/$MODEL_NAME"

echo "Building TinyLlama-1.1B for WASM with extreme optimization..."

# Download TinyLlama model
huggingface-cli download "TinyLlama/TinyLlama_v1.1" \
    --local-dir "./models/tinyllama/1_1b"

# Convert PyTorch to SafeTensors for better loading
cargo run --bin convert_pytorch -- \
    --input ./models/tinyllama/1_1b \
    --output ./models/tinyllama/1_1b_safetensors

# Extreme quantization for WASM
cargo run --bin quantize_model -- \
    --model-path ./models/tinyllama/1_1b_safetensors \
    --output-path ./models/tinyllama/1_1b_q4_extreme \
    --format int4-extreme \
    --target wasm \
    --memory-limit 800MB

# Split into streaming chunks (mandatory for 1.1B)
python3 scripts/create_tinyllama_chunks.py \
    ./models/tinyllama/1_1b_q4_extreme \
    --output ./models/tinyllama/1_1b_streaming \
    --chunk-size 40MB \
    --layers-per-chunk 2

# Build WASM with streaming + sentencepiece
wasm-pack build \
    --target web \
    --out-dir "$OUTPUT_DIR" \
    --features "wasm,streaming,sentencepiece,tinyllama" \
    --release \
    -- --max-memory 2147483648  # 2GB max

# Aggressive WASM optimization
wasm-opt -Os --enable-bulk-memory --enable-threads \
    --strip-debug --strip-producers \
    -o "$OUTPUT_DIR/llmkg_bg.wasm" "$OUTPUT_DIR/llmkg_bg.wasm"

# Create streaming chunk distribution
mkdir -p "$OUTPUT_DIR/chunks"
cp ./models/tinyllama/1_1b_streaming/chunk_*.bin "$OUTPUT_DIR/chunks/"

# Core components (always loaded)
mkdir -p "$OUTPUT_DIR/core"
cp ./models/tinyllama/1_1b_streaming/core_*.bin "$OUTPUT_DIR/core/"

# Copy tokenizer
cp ./models/tinyllama/1_1b/tokenizer.model "$OUTPUT_DIR/"
cp ./models/tinyllama/1_1b/*.json "$OUTPUT_DIR/"

# Create TinyLlama manifest
cat > "$OUTPUT_DIR/model_manifest.json" << EOF
{
  "name": "TinyLlama-1.1B",
  "version": "1.1.0",
  "architecture": "llama2-tinyllama",
  "parameters": 1100000000,
  "quantization": "int4-extreme",
  "bundle_size_mb": $(du -m "$OUTPUT_DIR" | tail -1 | cut -f1),
  "tokenizer": "sentencepiece",
  "streaming": {
    "enabled": true,
    "mandatory": true,
    "total_chunks": 11,
    "chunk_size_mb": 40,
    "layers_per_chunk": 2,
    "preload_chunks": [0, 10],
    "memory_management": "extreme"
  },
  "performance": {
    "target_tokens_per_second": 20,
    "max_memory_mb": 1000,
    "first_token_latency_ms": 200,
    "context_length": 2048,
    "device_requirements": {
      "min_memory_gb": 2,
      "recommended_memory_gb": 4,
      "mobile": "not_supported",
      "tablet": "limited_functionality",  
      "laptop": "full_performance",
      "desktop": "optimal"
    }
  },
  "features": {
    "grouped_query_attention": true,
    "rope_scaling": true,
    "flash_attention": false,
    "streaming_required": true
  }
}
EOF

echo "TinyLlama-1.1B WASM build complete"
echo "WARNING: This is a large model requiring significant resources"
echo "Core bundle: $(du -sh "$OUTPUT_DIR"/*.wasm "$OUTPUT_DIR"/*.js | tail -1)"
echo "Total with chunks: $(du -sh "$OUTPUT_DIR" | cut -f1)"
```

## Success Criteria

### Functional Requirements
- [ ] Load TinyLlama with PyTorch/SafeTensors weights
- [ ] SentencePiece tokenization working correctly
- [ ] GQA attention optimization active
- [ ] Streaming inference for memory management
- [ ] WASM deployment with extreme optimization

### Performance Requirements
- [ ] Memory usage <1GB (WASM), <1.8GB (native)
- [ ] Inference speed >20 tokens/second WASM, >30 native
- [ ] Bundle size <500MB total with streaming
- [ ] Quality comparable to reference implementation

### Technical Requirements
- [ ] PyTorch weight conversion to SafeTensors
- [ ] Grouped Query Attention implementation
- [ ] Streaming layer management
- [ ] SentencePiece integration
- [ ] Cross-browser WASM compatibility

## Risk Mitigation

1. **Large Model Size**: Mandatory streaming, extreme quantization
2. **Memory Requirements**: Aggressive layer management, device detection
3. **PyTorch Compatibility**: Robust weight conversion pipeline
4. **Browser Limitations**: Feature detection, graceful degradation
5. **Performance**: GQA optimization, SIMD acceleration where available

## Timeline

- **Day 1-2**: Llama 2 architecture, PyTorch loading, GQA implementation
- **Day 3**: Streaming architecture, extreme quantization
- **Day 4**: WASM deployment, browser testing, optimization

TinyLlama-1.1B represents a significant scaling challenge requiring advanced optimization techniques while maintaining the quality expected from a 1.1B parameter model.