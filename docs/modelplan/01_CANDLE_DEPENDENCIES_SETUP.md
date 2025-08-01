# Candle Dependencies & Core Integration Setup

## Overview

This document provides the complete setup process for integrating HuggingFace Candle into the LLMKG system, including dependency configuration, core trait implementation, and infrastructure modifications.

## Cargo.toml Modifications

### Add Candle Ecosystem Dependencies

```toml
[dependencies]
# Core Candle Framework
candle-core = "0.6.0"
candle-nn = "0.6.0"
candle-transformers = "0.6.0"
candle-hub = "0.6.0"

# HuggingFace Integration
hf-hub = "0.3.2"
tokenizers = "0.19.1"

# Model Format Support
safetensors = "0.4.1"
candle-onnx = { version = "0.6.0", optional = true }

# Quantization & Optimization
half = "2.3.1"  # f16 support
byteorder = "1.5.0"  # Endianness handling

# WASM-specific dependencies
[target.'cfg(target_arch = "wasm32")'.dependencies]
candle-wasm-utils = "0.6.0"
wasm-bindgen-futures = "0.4.40"
js-sys = "0.3.67"
```

### Feature Flag Updates

```toml
[features]
default = ["native"]
native = ["nalgebra", "candle-hub/cuda"]  # Enable CUDA for native
wasm = [
    "wasm-bindgen", 
    "wasm-bindgen-futures", 
    "console_error_panic_hook", 
    "wee_alloc", 
    "js-sys", 
    "web-sys", 
    "getrandom",
    "candle-wasm-utils"
]
quantization = ["candle-onnx", "half"]
simd = ["candle-core/accelerate"]  # Use accelerate framework on macOS
cuda = ["candle-core/cuda"]
```

## Core Architecture Integration

### 1. Model Backend Trait Definition

Create `src/models/backend.rs`:

```rust
use candle_core::{Device, Tensor, Result as CandleResult};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use crate::error::{GraphError, Result};
use std::sync::Arc;

/// Universal model backend trait for all ML frameworks
pub trait ModelBackend: Send + Sync {
    /// Generate text from prompt
    fn generate_text(&self, prompt: &str, config: &ModelConfig) -> Result<String>;
    
    /// Get model embeddings for semantic search
    fn get_embeddings(&self, text: &str) -> Result<Vec<f32>>;
    
    /// Check if model is loaded and ready
    fn is_ready(&self) -> bool;
    
    /// Get model memory usage in bytes
    fn memory_usage(&self) -> u64;
    
    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;
}

/// Candle-specific model backend implementation
pub struct CandleModelBackend {
    model: Arc<dyn CandleModel>,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    config: ModelConfig,
    metadata: ModelMetadata,
    logits_processor: LogitsProcessor,
}

/// Universal Candle model trait
pub trait CandleModel: Send + Sync {
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor>;
    fn device(&self) -> &Device;
    fn reset_kv_cache(&mut self);
}
```

### 2. Model Loading Infrastructure

Create `src/models/candle_loader.rs`:

```rust
use candle_core::{Device, DType};
use candle_hub::{api::sync::Api, Repo, RepoType};
use candle_transformers::models::{
    llama::{Llama, LlamaConfig},
    phi3::{Phi3, Phi3Config},
    // Add other model architectures as needed
};
use hf_hub::{api::tokio::Api as AsyncApi, Cache, Repo as HfRepo};
use tokenizers::Tokenizer;
use std::path::PathBuf;

pub struct CandleModelLoader {
    cache_dir: PathBuf,
    device: Device,
    dtype: DType,
    api: Api,
}

impl CandleModelLoader {
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
                .join("huggingface")
                .join("hub")
        });

        let device = Self::detect_best_device()?;
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let api = Api::new()?;

        Ok(Self {
            cache_dir,
            device,
            dtype,
            api,
        })
    }

    fn detect_best_device() -> Result<Device> {
        #[cfg(feature = "cuda")]
        if candle_core::cuda_is_available() {
            return Ok(Device::new_cuda(0)?);
        }
        
        #[cfg(target_arch = "wasm32")]
        return Ok(Device::Cpu);
        
        Ok(Device::Cpu)
    }

    pub async fn load_model(&self, model_id: &str, model_type: ModelType) -> Result<Box<dyn CandleModel>> {
        match model_type {
            ModelType::SmolLM => self.load_llama_based_model(model_id).await,
            ModelType::TinyLlama => self.load_llama_based_model(model_id).await,
            ModelType::OpenELM => self.load_openelm_model(model_id).await,
            ModelType::MiniLM => self.load_bert_based_model(model_id).await,
        }
    }

    async fn load_llama_based_model(&self, model_id: &str) -> Result<Box<dyn CandleModel>> {
        // Download config and weights
        let repo = self.api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let config_path = repo.get("config.json")?;
        let weights_path = repo.get("model.safetensors")?;

        // Load configuration
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        
        // Load model weights
        let tensors = safetensors::SafeTensors::deserialize(&std::fs::read(weights_path)?)?;
        let vb = candle_nn::VarBuilder::from_tensors(tensors, self.dtype, &self.device);
        
        // Create model instance
        let model = Llama::load(vb, &config)?;
        
        Ok(Box::new(LlamaWrapper::new(model, self.device.clone())))
    }
}

#[derive(Debug, Clone)]
pub enum ModelType {
    SmolLM,
    TinyLlama,
    OpenELM,
    MiniLM,
}
```

### 3. Model Registry Integration

Update `src/models/registry.rs`:

```rust
use super::backend::{CandleModelBackend, ModelBackend};
use super::candle_loader::{CandleModelLoader, ModelType};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct EnhancedModelRegistry {
    models: HashMap<String, ModelMetadata>,
    loaded_models: Arc<RwLock<HashMap<String, Arc<dyn ModelBackend>>>>,
    loader: Arc<CandleModelLoader>,
}

impl EnhancedModelRegistry {
    pub fn new() -> Result<Self> {
        let loader = Arc::new(CandleModelLoader::new(None)?);
        
        Ok(Self {
            models: HashMap::new(),
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            loader,
        })
    }

    pub async fn load_model(&self, model_id: &str) -> Result<Arc<dyn ModelBackend>> {
        // Check if already loaded
        {
            let loaded = self.loaded_models.read().await;
            if let Some(model) = loaded.get(model_id) {
                return Ok(model.clone());
            }
        }

        // Load new model
        let metadata = self.get_model(model_id)
            .ok_or_else(|| GraphError::InvalidState(format!("Model {} not registered", model_id)))?;

        let model_type = self.determine_model_type(&metadata.family);
        let candle_model = self.loader.load_model(model_id, model_type).await?;
        
        // Load tokenizer
        let tokenizer = self.load_tokenizer(model_id).await?;
        
        // Create backend
        let backend = Arc::new(CandleModelBackend::new(
            candle_model,
            tokenizer,
            metadata.clone(),
        )?);

        // Cache the loaded model
        {
            let mut loaded = self.loaded_models.write().await;
            loaded.insert(model_id.to_string(), backend.clone());
        }

        Ok(backend)
    }

    fn determine_model_type(&self, family: &str) -> ModelType {
        match family {
            "SmolLM" => ModelType::SmolLM,
            "TinyLlama" => ModelType::TinyLlama,
            "OpenELM" => ModelType::OpenELM,
            "MiniLM" => ModelType::MiniLM,
            _ => ModelType::SmolLM, // Default fallback
        }
    }

    async fn load_tokenizer(&self, model_id: &str) -> Result<Arc<Tokenizer>> {
        let repo = self.loader.api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| GraphError::StorageError(format!("Failed to load tokenizer: {}", e)))?;

        Ok(Arc::new(tokenizer))
    }
}
```

## WASM-Specific Modifications

### 1. WASM Model Loading

Create `src/models/wasm_loader.rs`:

```rust
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use candle_core::Device;

#[cfg(target_arch = "wasm32")]
pub struct WasmModelLoader {
    device: Device,
}

#[cfg(target_arch = "wasm32")]
impl WasmModelLoader {
    pub fn new() -> Self {
        Self {
            device: Device::Cpu, // WASM only supports CPU for now
        }
    }

    pub async fn load_quantized_model(&self, model_url: &str) -> Result<Box<dyn CandleModel>> {
        // Use fetch API to download quantized model weights
        let window = web_sys::window().unwrap();
        let resp = JsFuture::from(window.fetch_with_str(model_url)).await?;
        let resp: web_sys::Response = resp.dyn_into().unwrap();
        let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let bytes = uint8_array.to_vec();

        // Deserialize quantized model
        self.deserialize_quantized_model(&bytes).await
    }

    async fn deserialize_quantized_model(&self, bytes: &[u8]) -> Result<Box<dyn CandleModel>> {
        // Implementation depends on quantization format
        // This is a placeholder for the actual deserialization logic
        todo!("Implement quantized model deserialization")
    }
}
```

### 2. Performance Optimization

Create `src/models/optimization.rs`:

```rust
use candle_core::{Tensor, Device, DType};

pub struct ModelOptimizer;

impl ModelOptimizer {
    /// Quantize model weights to Int8 for WASM deployment
    pub fn quantize_to_int8(weights: &Tensor) -> Result<Tensor> {
        // Find min/max values for quantization range
        let min_val = weights.min(0)?.min(0)?.to_scalar::<f32>()?;
        let max_val = weights.max(0)?.max(0)?.to_scalar::<f32>()?;
        
        // Calculate scale and zero point
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as i8;
        
        // Quantize: q = round(f / scale) + zero_point
        let quantized = ((weights / scale)? + zero_point as f32)?
            .round()?
            .clamp(-128.0, 127.0)?
            .to_dtype(DType::I8)?;
            
        Ok(quantized)
    }

    /// Dequantize Int8 weights back to floating point
    pub fn dequantize_from_int8(quantized: &Tensor, scale: f32, zero_point: i8) -> Result<Tensor> {
        let fp_weights = (quantized.to_dtype(DType::F32)? - zero_point as f32)? * scale;
        Ok(fp_weights)
    }

    /// Optimize model for WASM deployment
    pub fn optimize_for_wasm(model: &mut dyn CandleModel) -> Result<()> {
        // Apply various optimizations:
        // 1. Weight quantization
        // 2. Operator fusion
        // 3. Memory layout optimization
        // 4. Remove unnecessary operations
        
        todo!("Implement WASM-specific optimizations")
    }
}
```

## Integration Steps

### Step 1: Add Dependencies
```bash
# Update Cargo.toml with the dependencies listed above
cargo update
```

### Step 2: Create Core Files
```bash
# Create the backend infrastructure
touch src/models/backend.rs
touch src/models/candle_loader.rs
touch src/models/optimization.rs

# Add WASM-specific files
touch src/models/wasm_loader.rs
```

### Step 3: Update Module Exports
Add to `src/models/mod.rs`:
```rust
pub mod backend;
pub mod candle_loader;
pub mod optimization;

#[cfg(target_arch = "wasm32")]
pub mod wasm_loader;

pub use backend::{ModelBackend, CandleModelBackend};
pub use candle_loader::{CandleModelLoader, ModelType};
```

### Step 4: Update Main Library
Add to `src/lib.rs`:
```rust
pub use crate::models::{
    ModelBackend, CandleModelBackend, CandleModelLoader, ModelType
};
```

## Testing Setup

### Unit Tests
Create `src/models/tests.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_candle_loader_creation() {
        let loader = CandleModelLoader::new(None).unwrap();
        assert!(loader.device.is_cpu() || loader.device.is_cuda());
    }

    #[tokio::test]
    async fn test_model_registry_integration() {
        let registry = EnhancedModelRegistry::new().unwrap();
        // Add model loading tests here
    }
}
```

## Build Configuration

### Native Build
```bash
cargo build --release --features "native,quantization"
```

### WASM Build
```bash
wasm-pack build --target web --features "wasm,quantization" --release
```

## Next Steps

1. **Implement Core Traits**: Start with `ModelBackend` and `CandleModelBackend`
2. **Model Loading**: Implement `CandleModelLoader` for first model (SmolLM-135M)
3. **Testing**: Create integration tests for model loading pipeline
4. **Optimization**: Add quantization support for WASM deployment
5. **Documentation**: Document API changes and usage examples

This foundation provides the infrastructure needed for all subsequent model integrations.