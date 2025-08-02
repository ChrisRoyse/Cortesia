# SmolLM-1.7B Integration Plan

## Model Overview

**Model ID**: `HuggingFaceTB/SmolLM-1.7B`  
**Parameters**: 1,700,000,000 (1.7B)  
**Architecture**: Transformer (Llama-based)  
**Size Category**: Large  
**Priority**: Tier 2 - High Quality  
**Use Case**: Complex reasoning, research tasks, code generation

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
- **Inference Speed**: 20-40 tokens/second (CPU), 80-120 (GPU)
- **Memory Usage**: 800MB-1.2GB peak (native), <800MB (WASM quantized)
- **First Token Latency**: <100ms (native), <200ms (WASM)
- **Subsequent Tokens**: <25ms average
- **WASM Bundle Size**: ~400MB quantized (Int4 + streaming)

## Implementation Strategy

Due to the large size (1.7B parameters), this model requires advanced optimization techniques:

1. **Mandatory Quantization**: Int4 required for practical deployment
2. **Streaming Architecture**: Layer-by-layer loading essential
3. **Memory Management**: Aggressive memory optimization
4. **Device-Specific Loading**: Different strategies for different hardware

### Phase 1: Advanced Model Architecture (Day 1-2)

#### Step 1.1: Streaming Model Implementation
```rust
// src/models/smollm/smollm_1_7b.rs
pub struct SmolLM1_7B {
    config: LlamaConfig,
    device: Device,
    
    // Core components (always loaded)
    embed_tokens: Embedding,
    norm: LayerNorm,
    lm_head: Linear,
    
    // Streaming components
    layer_loader: LayerStreamLoader,
    layer_cache: LruCache<usize, TransformerBlock>,
    kv_cache: KVCache,
    
    // Memory management
    memory_manager: ModelMemoryManager,
    max_memory_mb: usize,
}

impl SmolLM1_7B {
    pub fn load_streaming(model_path: &Path, device: Device, max_memory_mb: usize) -> Result<Self> {
        let config = Self::load_config(model_path)?;
        
        // Validate architecture (1.7B specific)
        if config.hidden_size != 2048 || config.num_hidden_layers != 24 {
            return Err(GraphError::InvalidState(
                format!("Invalid SmolLM-1.7B architecture: hidden_size={}, layers={}", 
                       config.hidden_size, config.num_hidden_layers)
            ));
        }
        
        // Load only essential components
        let weights = Self::open_weights_file(model_path)?;
        let embed_tokens = Self::load_embedding(&weights, &config, &device)?;
        let norm = Self::load_norm(&weights, &config, &device)?;
        let lm_head = Self::load_lm_head(&weights, &config, &device)?;
        
        // Initialize streaming components
        let layer_loader = LayerStreamLoader::new(model_path, &device)?;
        let cache_size = (max_memory_mb * 1024 * 1024) / 4; // Reserve 1/4 for layer cache
        let layer_cache = LruCache::new(NonZeroUsize::new(3).unwrap()); // Keep 3 layers max
        
        let kv_cache = KVCache::new(&config, &device, 32)?; // Batch size 32
        let memory_manager = ModelMemoryManager::new(max_memory_mb)?;
        
        Ok(Self {
            config,
            device,
            embed_tokens,
            norm,
            lm_head,
            layer_loader,
            layer_cache,
            kv_cache,
            memory_manager,
            max_memory_mb,
        })
    }
    
    pub fn generate_text_streaming(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let tokens = self.tokenize(prompt)?;
        let mut generated_tokens = tokens.clone();
        let mut output_text = String::new();
        
        // Pre-load critical layers
        self.preload_layers(&[0, 1, self.config.num_hidden_layers - 1]).await?;
        
        for step in 0..max_tokens {
            let start_time = std::time::Instant::now();
            
            // Forward pass with streaming
            let next_token = self.forward_step(&generated_tokens)?;
            
            if next_token == self.config.eos_token_id.unwrap_or(2) {
                break;
            }
            
            generated_tokens.push(next_token);
            
            // Decode incrementally for streaming response
            let new_text = self.decode_incremental(next_token)?;
            output_text.push_str(&new_text);
            
            // Memory management
            if step % 10 == 0 {
                self.memory_manager.cleanup_if_needed()?;
            }
            
            let elapsed = start_time.elapsed();
            log::debug!("Token {} generated in {:?}", step, elapsed);
        }
        
        Ok(output_text)
    }
    
    async fn forward_step(&mut self, tokens: &[u32]) -> Result<u32> {
        let seq_len = tokens.len();
        let input_ids = Tensor::from_slice(tokens, (1, seq_len), &self.device)?;
        
        // Embedding layer
        let mut hidden_states = self.embed_tokens.forward(&input_ids)?;
        
        // Process through transformer layers with streaming
        for layer_idx in 0..self.config.num_hidden_layers {
            // Load layer if not in cache
            if !self.layer_cache.contains(&layer_idx) {
                self.load_layer_on_demand(layer_idx).await?;
            }
            
            // Forward through layer
            let layer = self.layer_cache.get(&layer_idx).unwrap();
            hidden_states = layer.forward(&hidden_states, &mut self.kv_cache, layer_idx)?;
            
            // Aggressive memory management for large model
            if layer_idx > 2 && (layer_idx % 3 == 0) {
                self.maybe_unload_old_layers(layer_idx)?;
            }
        }
        
        // Final normalization and projection
        let hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        
        // Sample next token
        let next_token = self.sample_token(&logits)?;
        Ok(next_token)
    }
    
    async fn load_layer_on_demand(&mut self, layer_idx: usize) -> Result<()> {
        if self.layer_cache.len() >= 3 {
            // Remove oldest layer to make space
            self.layer_cache.pop_lru();
        }
        
        let layer = self.layer_loader.load_layer(layer_idx).await?;
        self.layer_cache.put(layer_idx, layer);
        
        Ok(())
    }
}

pub struct LayerStreamLoader {
    model_path: PathBuf,
    device: Device,
    weights_mmap: memmap2::Mmap,
    layer_offsets: HashMap<usize, (usize, usize)>, // layer_idx -> (offset, size)
}

impl LayerStreamLoader {
    pub fn new(model_path: &Path, device: &Device) -> Result<Self> {
        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weights_mmap = unsafe { memmap2::Mmap::map(&weights_file)? };
        
        // Parse safetensors header to get layer offsets
        let layer_offsets = Self::parse_layer_offsets(&weights_mmap)?;
        
        Ok(Self {
            model_path: model_path.to_path_buf(),
            device: device.clone(),
            weights_mmap,
            layer_offsets,
        })
    }
    
    pub async fn load_layer(&self, layer_idx: usize) -> Result<TransformerBlock> {
        let (offset, size) = self.layer_offsets.get(&layer_idx)
            .ok_or_else(|| GraphError::InvalidState(format!("Layer {} not found", layer_idx)))?;
        
        // Extract layer weights from memory-mapped file
        let layer_data = &self.weights_mmap[*offset..(*offset + *size)];
        
        // Deserialize and create layer
        let layer_tensors = self.deserialize_layer_tensors(layer_data)?;
        let layer = TransformerBlock::from_tensors(layer_tensors, &self.device)?;
        
        Ok(layer)
    }
}
```

#### Step 1.2: Memory-Mapped Quantization
```rust
// src/models/smollm/quantization_1_7b.rs
pub struct SmolLM1_7BQuantizer;

impl SmolLM1_7BQuantizer {
    pub fn quantize_mmap(input_path: &Path, output_path: &Path) -> Result<()> {
        println!("Quantizing SmolLM-1.7B with memory mapping...");
        
        // Use memory mapping for large model processing
        let input_file = File::open(input_path.join("model.safetensors"))?;
        let mmap = unsafe { memmap2::Mmap::map(&input_file)? };
        
        let tensors = SafeTensors::deserialize(&mmap)?;
        let mut quantized_file = BufWriter::new(File::create(output_path.join("model_q4.bin"))?);
        
        // Process in chunks to avoid memory explosion
        for (name, tensor_view) in tensors.tensors() {
            println!("Quantizing tensor: {} (shape: {:?})", name, tensor_view.shape());
            
            if Self::should_quantize_tensor(name) {
                Self::quantize_tensor_streaming(&tensor_view, &mut quantized_file, name)?;
            } else {
                Self::preserve_tensor_fp16(&tensor_view, &mut quantized_file, name)?;
            }
        }
        
        quantized_file.flush()?;
        
        // Create index file for fast layer loading
        Self::create_layer_index(output_path)?;
        
        println!("1.7B quantization complete. Expected size: ~680MB");
        Ok(())
    }
    
    fn quantize_tensor_streaming(
        tensor_view: &TensorView, 
        output: &mut BufWriter<File>,
        name: &str
    ) -> Result<()> {
        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();
        
        // Write tensor header
        Self::write_tensor_header(output, name, shape, QuantizationType::Int4Grouped)?;
        
        // Process tensor in chunks to manage memory
        let chunk_size = 1024 * 1024; // 1M elements per chunk
        let total_elements: usize = shape.iter().product();
        
        for chunk_start in (0..total_elements).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(total_elements);
            let chunk_size_actual = chunk_end - chunk_start;
            
            // Load chunk
            let chunk_data = tensor_view.data();
            let chunk_slice = &chunk_data[chunk_start * 4..(chunk_end * 4)]; // Assuming F32
            let chunk_floats = Self::bytes_to_f32_slice(chunk_slice);
            
            // Quantize chunk
            let quantized_chunk = Self::quantize_chunk_int4(chunk_floats, 128)?;
            
            // Write quantized chunk
            output.write_all(&quantized_chunk)?;
        }
        
        Ok(())
    }
    
    fn create_layer_index(output_path: &Path) -> Result<()> {
        // Create index for fast layer loading
        let index_path = output_path.join("layer_index.json");
        
        let layer_info = serde_json::json!({
            "total_layers": 24,
            "layer_names": (0..24).map(|i| format!("model.layers.{}", i)).collect::<Vec<_>>(),
            "quantization": "int4_grouped",
            "group_size": 128,
            "streaming": true
        });
        
        std::fs::write(index_path, serde_json::to_string_pretty(&layer_info)?)?;
        Ok(())
    }
}
```

### Phase 2: WASM Streaming Architecture (Day 2-3)

#### Step 2.1: Progressive Loading System
```rust
// src/models/smollm/wasm_1_7b.rs
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSmolLM1_7B {
    model: Option<SmolLM1_7B>,
    loading_state: LoadingState,
    loaded_chunks: HashSet<usize>,
    base_url: String,
    total_chunks: usize,
    memory_limit_mb: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum LoadingState {
    Uninitialized,
    LoadingCore,      // Essential components
    LoadingChunks(usize), // Layer chunks
    Ready,
    Error(String),
}

#[wasm_bindgen]
impl WasmSmolLM1_7B {
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String, memory_limit_mb: usize) -> Self {
        Self {
            model: None,
            loading_state: LoadingState::Uninitialized,
            loaded_chunks: HashSet::new(),
            base_url,
            total_chunks: 8, // 24 layers / 3 layers per chunk
            memory_limit_mb,
        }
    }
    
    #[wasm_bindgen]
    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        self.loading_state = LoadingState::LoadingCore;
        
        // Load core components first (embeddings, final layer)
        let manifest = self.fetch_manifest().await?;
        let config = self.fetch_config().await?;
        let tokenizer = self.fetch_tokenizer().await?;
        let core_weights = self.fetch_core_weights().await?;
        
        // Initialize streaming model
        self.model = Some(SmolLM1_7B::new_streaming(
            config,
            tokenizer,
            core_weights,
            self.memory_limit_mb,
        )?);
        
        // Pre-load first chunk for faster initial response
        self.load_chunk(0).await?;
        
        self.loading_state = LoadingState::Ready;
        self.log_performance_info();
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub async fn generate_streaming(&mut self, prompt: &str, max_tokens: usize) -> Result<js_sys::ReadableStream, JsValue> {
        if self.loading_state != LoadingState::Ready {
            return Err(JsValue::from_str("Model not ready"));
        }
        
        // Create streaming response
        let (stream, writer) = self.create_response_stream();
        
        // Start generation in background
        let generation_future = self.generate_with_streaming_response(prompt, max_tokens, writer);
        wasm_bindgen_futures::spawn_local(generation_future);
        
        Ok(stream)
    }
    
    async fn generate_with_streaming_response(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        writer: web_sys::WritableStreamDefaultWriter,
    ) {
        let tokens = match self.tokenize(prompt).await {
            Ok(tokens) => tokens,
            Err(e) => {
                let _ = writer.write_with_chunk(&JsValue::from_str(&format!("Error: {}", e)));
                return;
            }
        };
        
        let mut generated_tokens = tokens.clone();
        
        for step in 0..max_tokens {
            // Ensure required layers are loaded
            let required_chunk = self.get_required_chunk_for_step(step);
            if !self.loaded_chunks.contains(&required_chunk) {
                if let Err(e) = self.load_chunk(required_chunk).await {
                    let _ = writer.write_with_chunk(&JsValue::from_str(&format!("Loading error: {}", e)));
                    return;
                }
            }
            
            // Generate next token
            match self.model.as_mut().unwrap().forward_step(&generated_tokens).await {
                Ok(next_token) => {
                    if next_token == 2 { // EOS token
                        break;
                    }
                    
                    generated_tokens.push(next_token);
                    
                    // Decode and stream token
                    if let Ok(token_text) = self.decode_token(next_token).await {
                        let _ = writer.write_with_chunk(&JsValue::from_str(&token_text));
                    }
                },
                Err(e) => {
                    let _ = writer.write_with_chunk(&JsValue::from_str(&format!("Generation error: {}", e)));
                    return;
                }
            }
            
            // Memory management
            if step % 20 == 0 {
                self.manage_memory().await;
            }
        }
        
        let _ = writer.close();
    }
    
    async fn load_chunk(&mut self, chunk_idx: usize) -> Result<(), JsValue> {
        if self.loaded_chunks.contains(&chunk_idx) {
            return Ok(());
        }
        
        // Check memory limit
        if self.loaded_chunks.len() >= 3 {
            // Unload oldest chunk
            self.unload_least_used_chunk().await?;
        }
        
        let url = format!("{}/chunks/chunk_{}.bin", self.base_url, chunk_idx);
        let chunk_data = self.fetch_binary(&url).await?;
        
        // Load chunk into model
        if let Some(model) = &mut self.model {
            model.load_chunk_from_bytes(chunk_idx, &chunk_data)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        
        self.loaded_chunks.insert(chunk_idx);
        Ok(())
    }
    
    fn log_performance_info(&self) {
        web_sys::console::log_1(&format!(
            "SmolLM-1.7B initialized: Memory limit: {}MB, Chunks: {}, Streaming: enabled",
            self.memory_limit_mb, self.total_chunks
        ).into());
    }
}
```

#### Step 2.2: Chunk-Based Distribution
```bash
# scripts/build_smollm_1_7b_wasm.sh
#!/bin/bash
set -e

MODEL_NAME="smollm_1_7b"
OUTPUT_DIR="./dist/wasm/models/$MODEL_NAME"

echo "Building streaming WASM bundle for SmolLM-1.7B..."

# Quantize with extreme compression for WASM
cargo run --bin quantize_model -- \
    --model-path ./models/smollm/1_7b \
    --output-path ./models/smollm/1_7b_q4 \
    --format int4-grouped \
    --group-size 64 \
    --extreme-compression

# Split into streamable chunks
python3 scripts/create_streaming_chunks.py \
    ./models/smollm/1_7b_q4 \
    --output ./models/smollm/1_7b_chunks \
    --chunk-size 50MB \
    --overlap-layers 1

# Build WASM with streaming support
wasm-pack build \
    --target web \
    --out-dir "$OUTPUT_DIR" \
    --features "wasm,streaming,large-model" \
    --release \
    -- --max-memory 4194304  # 4GB max memory

# Optimize for streaming
wasm-opt -Os --enable-bulk-memory --enable-threads -o "$OUTPUT_DIR/llmkg_bg.wasm" "$OUTPUT_DIR/llmkg_bg.wasm"

# Create chunk distribution
mkdir -p "$OUTPUT_DIR/chunks"
cp ./models/smollm/1_7b_chunks/*.bin "$OUTPUT_DIR/chunks/"

# Core components (always loaded)
mkdir -p "$OUTPUT_DIR/core"
cp ./models/smollm/1_7b_chunks/core_*.bin "$OUTPUT_DIR/core/"

# Configuration and metadata
cp ./models/smollm/1_7b_q4/*.json "$OUTPUT_DIR/"

# Create comprehensive manifest
cat > "$OUTPUT_DIR/model_manifest.json" << EOF
{
  "name": "SmolLM-1.7B",
  "version": "1.0.0",
  "architecture": "llama",
  "parameters": 1700000000,
  "quantization": "int4-grouped-extreme",
  "bundle_size_mb": $(du -m "$OUTPUT_DIR" | tail -1 | cut -f1),
  "streaming": {
    "enabled": true,
    "total_chunks": 8,
    "chunk_size_mb": 50,
    "overlap_layers": 1,
    "preload_chunks": [0],
    "memory_management": "aggressive"
  },
  "performance": {
    "target_tokens_per_second": 25,
    "max_memory_mb": 800,
    "first_token_latency_ms": 150,
    "context_length": 2048,
    "device_recommendations": {
      "mobile": "not_recommended",
      "tablet": "basic_functionality",
      "laptop": "full_performance",
      "desktop": "optimal"
    }
  },
  "requirements": {
    "min_memory_mb": 600,
    "recommended_memory_mb": 1200,
    "webgl": "recommended",
    "wasm_threads": "required",
    "shared_array_buffer": "required"
  }
}
EOF

echo "Streaming WASM bundle created at: $OUTPUT_DIR"
echo "Core runtime: $(du -sh "$OUTPUT_DIR"/*.wasm "$OUTPUT_DIR"/*.js | tail -1)"
echo "Total with chunks: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo "Chunks: $(ls -la "$OUTPUT_DIR/chunks/" | wc -l) files"
```

### Phase 3: Advanced Optimization (Day 3-4)

#### Step 3.1: Device-Specific Optimization
```rust
// src/models/smollm/device_optimization.rs
pub struct DeviceOptimizer;

impl DeviceOptimizer {
    pub fn optimize_for_device(model: &mut SmolLM1_7B, device_info: &DeviceInfo) -> Result<()> {
        match device_info.category {
            DeviceCategory::HighEnd => {
                model.set_memory_limit(1200); // 1.2GB
                model.enable_layer_prefetching(true);
                model.set_batch_size(32);
                model.enable_kv_cache_optimization(true);
            },
            DeviceCategory::MidRange => {
                model.set_memory_limit(800); // 800MB
                model.enable_layer_prefetching(false);
                model.set_batch_size(16);
                model.enable_aggressive_memory_management(true);
            },
            DeviceCategory::LowEnd => {
                model.set_memory_limit(600); // 600MB
                model.enable_layer_streaming_only(true);
                model.set_batch_size(8);
                model.enable_extreme_memory_optimization(true);
            },
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub category: DeviceCategory,
    pub available_memory_mb: usize,
    pub cpu_cores: usize,
    pub supports_wasm_threads: bool,
    pub supports_simd: bool,
}

#[derive(Debug, Clone)]
pub enum DeviceCategory {
    HighEnd,   // Desktop, high-end laptop
    MidRange,  // Standard laptop, high-end tablet
    LowEnd,    // Basic devices, older hardware
}

impl DeviceInfo {
    #[cfg(target_arch = "wasm32")]
    pub fn detect() -> Self {
        let navigator = web_sys::window().unwrap().navigator();
        let memory_info = navigator.device_memory().unwrap_or(4.0); // GB
        let cpu_cores = navigator.hardware_concurrency().unwrap_or(4) as usize;
        
        let available_memory_mb = (memory_info * 1024.0) as usize;
        let supports_wasm_threads = js_sys::eval("typeof SharedArrayBuffer !== 'undefined'")
            .unwrap()
            .as_bool()
            .unwrap_or(false);
            
        let supports_simd = Self::check_wasm_simd();
        
        let category = if available_memory_mb >= 8192 && cpu_cores >= 8 {
            DeviceCategory::HighEnd
        } else if available_memory_mb >= 4096 && cpu_cores >= 4 {
            DeviceCategory::MidRange
        } else {
            DeviceCategory::LowEnd
        };
        
        Self {
            category,
            available_memory_mb,
            cpu_cores,
            supports_wasm_threads,
            supports_simd,
        }
    }
}
```

## Success Criteria

### Functional Requirements
- [ ] Successfully load and run SmolLM-1.7B with streaming
- [ ] Memory usage <800MB peak (WASM), <1.2GB (native)
- [ ] Generate high-quality text with advanced reasoning
- [ ] Streaming inference with <200ms first token (WASM)
- [ ] Progressive loading working across browsers

### Performance Requirements
- [ ] Native: >30 tokens/second CPU, >100 tokens/second GPU  
- [ ] WASM: >20 tokens/second with streaming
- [ ] Bundle size: Core <50MB, total <450MB
- [ ] Memory efficiency: Aggressive layer management
- [ ] Quality: <5% degradation vs. FP16

### Advanced Features
- [ ] Device-specific optimization
- [ ] Streaming response generation
- [ ] Aggressive memory management
- [ ] Cross-browser compatibility
- [ ] Graceful degradation on low-end devices

## Risk Mitigation

1. **Memory Constraints**: Multi-tier memory management, streaming-only mode
2. **Loading Time**: Progressive chunks, pre-loading critical components
3. **Browser Compatibility**: Feature detection, fallback modes
4. **Quality Loss**: Careful quantization with quality validation
5. **Performance**: Device-specific optimization, SIMD acceleration

## Timeline

- **Day 1-2**: Streaming architecture, memory mapping, quantization pipeline
- **Day 2-3**: WASM streaming implementation, chunk creation, browser testing
- **Day 3-4**: Device optimization, quality validation, production integration

This plan addresses the unique challenges of deploying a 1.7B parameter model while maintaining usability across different devices and deployment scenarios.