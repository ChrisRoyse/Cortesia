# Directory Overview: Models

## 1. High-Level Summary

The `src/models` directory contains a comprehensive small language model management system for the LLMKG (Large Language Model Knowledge Graph) project. This module provides implementations for state-of-the-art small language models (1M-500M parameters) optimized for efficiency and performance in resource-constrained environments. It includes model definitions, loading utilities, configuration management, and a registry system for discovering and managing multiple model families.

## 2. Tech Stack

- **Languages:** Rust
- **Frameworks:** Custom model management framework
- **Libraries:** 
  - `serde` (JSON serialization/deserialization)
  - `tokio` (async runtime)
  - `walkdir` (directory traversal)
  - `dirs` (system directories)
  - `tempfile` (testing)
- **External Dependencies:** HuggingFace Hub integration (simulated)
- **Architecture:** Modular design with builder patterns and trait-based abstractions

## 3. Directory Structure

```
models/
├── mod.rs              # Main module exports and core types
├── config.rs           # Model configuration structures  
├── loader.rs           # Model loading and caching utilities
├── registry.rs         # Model discovery and management registry
├── utils.rs            # Utility functions for formatting and analysis
├── smollm.rs          # SmolLM model family (135M-1.7B params)
├── tinyllama.rs       # TinyLlama model family (1.1B params)
├── openelm.rs         # OpenELM model family (270M-3B params)
└── minilm.rs          # MiniLM model family (22M-118M params)
```

## 4. File Breakdown

### `mod.rs`

- **Purpose:** Main module definition with core types and public exports
- **Enums:**
  - `ModelSize`: Categorizes models (Tiny, Small, Medium, Large)
- **Structs:**
  - `ModelCapabilities`: Defines model capabilities (text generation, chat, etc.)
  - `ModelMetadata`: Complete model information and specifications
  - `Model`: Main model instance with metadata and configuration
- **Key Methods:**
  - `Model::new(metadata, config)`: Creates new model instance
  - `Model::load()`: Loads model into memory
  - `Model::unload()`: Unloads model from memory
  - `Model::generate_text(prompt, max_tokens)`: Text generation method
  - `Model::is_loaded()`: Checks if model is loaded

### `config.rs`

- **Purpose:** Configuration management for model parameters and settings
- **Structs:**
  - `ModelConfig`: Generation parameters (temperature, top_k, top_p, etc.)
  - `LoadingConfig`: Model loading configuration (device, quantization, batch size)
- **Enums:**
  - `QuantizationConfig`: Quantization options (None, Int8, Int4, F16, BFloat16)
  - `DeviceConfig`: Target device (Cpu, Cuda, Metal)
- **Key Methods:**
  - `ModelConfig::with_temperature(temp)`: Builder method for temperature
  - `ModelConfig::with_top_k(k)`: Builder method for top-k sampling
  - `ModelConfig::set_custom_parameter(key, value)`: Custom parameter setting

### `loader.rs`

- **Purpose:** Model downloading and loading from HuggingFace Hub
- **Enums:**
  - `LoadingState`: Track loading progress (NotLoaded, Downloading, Loading, Loaded, Failed)
- **Structs:**
  - `ModelLoader`: Main loader with caching and state management
  - `AdvancedModelLoader`: Loader with progress callbacks
  - `LoadingStatistics`: Loading performance metrics
- **Key Methods:**
  - `ModelLoader::load_model(model_id, config)`: Async model loading
  - `ModelLoader::download_model(model_id)`: Download from HuggingFace
  - `ModelLoader::unload_model(model_id)`: Unload from memory
  - `ModelLoader::get_loading_state(model_id)`: Check loading status
  - `ModelLoader::clear_cache()`: Cache management

### `registry.rs`

- **Purpose:** Model discovery, registration, and management system
- **Structs:**
  - `ModelRegistry`: Central registry for all available models
  - `RecommendedModels`: Model recommendations for different use cases
  - `RegistryStatistics`: Registry statistics and metrics
  - `ModelFilter`: Filtering criteria for model search
- **Key Methods:**
  - `ModelRegistry::list_models()`: Get all registered models
  - `ModelRegistry::list_models_by_size(size)`: Filter by model size
  - `ModelRegistry::search_models(query)`: Text-based model search
  - `ModelRegistry::get_recommended_models()`: Get use-case recommendations
  - `ModelRegistry::filter_models(filter)`: Advanced filtering

### `utils.rs`

- **Purpose:** Utility functions for model analysis and formatting
- **Enums:**
  - `ModelPrecision`: Precision types (Float32, Float16, Int8, etc.)
  - `PerformanceTier`: Performance categories (UltraFast, Fast, Balanced, Powerful)
- **Structs:**
  - `MemoryRequirements`: Memory usage estimation
  - `DeviceCompatibility`: Device compatibility assessment
  - `ModelComparison`: Comprehensive model comparison metrics
  - `InferenceSpeed`: Speed estimates for different hardware
- **Key Functions:**
  - `format_parameters(params)`: Human-readable parameter counts
  - `format_file_size(bytes)`: Human-readable file sizes
  - `estimate_memory_requirements(params, precision)`: Memory estimation
  - `assess_device_compatibility(params)`: Device compatibility check
  - `recommend_model(models, requirements)`: Model recommendation engine
  - `generate_model_report(metadata)`: Comprehensive model report

### `smollm.rs`

- **Purpose:** SmolLM model family implementation by HuggingFace
- **Enum:**
  - `SmolLMVariant`: All SmolLM variants (135M, 360M, 1.7B, Instruct versions, v2 models)
- **Struct:**
  - `SmolLMBuilder`: Builder pattern for SmolLM models
- **Key Methods:**
  - `SmolLMVariant::parameters()`: Get parameter count
  - `SmolLMVariant::is_instruct()`: Check if instruction-tuned
  - `SmolLMVariant::is_v2()`: Check if version 2
- **Factory Functions:**
  - `smollm_135m()`, `smollm_360m()`, `smollm_1_7b()`: Base models
  - `smollm_135m_instruct()`, `smollm_360m_instruct()`, `smollm_1_7b_instruct()`: Instruct models
  - `smollm2_135m()`, `smollm2_360m()`, `smollm2_1_7b()`: Version 2 models

### `tinyllama.rs`

- **Purpose:** TinyLlama model family implementation (1.1B parameter models)
- **Enum:**
  - `TinyLlamaVariant`: All TinyLlama variants (base, chat versions, intermediate)
- **Struct:**
  - `TinyLlamaBuilder`: Builder pattern for TinyLlama models
- **Key Methods:**
  - `TinyLlamaVariant::is_chat()`: Check if chat-tuned
  - `TinyLlamaVariant::training_tokens()`: Get training token count
  - `TinyLlamaVariant::version()`: Get model version
- **Factory Functions:**
  - `tinyllama_1_1b()`: Base model
  - `tinyllama_1_1b_chat()`: Chat model
  - `tinyllama_1_1b_chat_v0_X()`: Various chat versions
  - `tinyllama_1_1b_intermediate()`: Intermediate checkpoint

### `openelm.rs`

- **Purpose:** OpenELM model family implementation by Apple
- **Enum:**
  - `OpenELMVariant`: All OpenELM variants (270M, 450M, 1.1B, 3B, Instruct versions)
- **Struct:**
  - `OpenELMBuilder`: Builder pattern for OpenELM models
- **Key Methods:**
  - `OpenELMVariant::is_instruct()`: Check if instruction-tuned
  - `OpenELMVariant::context_length()`: Get context window size
  - `OpenELMVariant::vocab_size()`: Get vocabulary size
- **Factory Functions:**
  - `openelm_270m()`, `openelm_450m()`, `openelm_1_1b()`, `openelm_3b()`: Base models
  - `openelm_X_instruct()`: Instruction-tuned versions
- **Helper Functions:**
  - `small_variants()`: Get models in 1M-500M parameter range

### `minilm.rs`

- **Purpose:** MiniLM model family implementation by Microsoft
- **Enums:**
  - `MiniLMVariant`: All MiniLM variants (language models, sentence transformers, cross-encoders)
  - `MiniLMType`: Model type classification
- **Struct:**
  - `MiniLMBuilder`: Builder pattern for MiniLM models
- **Key Methods:**
  - `MiniLMVariant::model_type()`: Get model type (LanguageModel, SentenceTransformer, CrossEncoder)
  - `MiniLMVariant::is_multilingual()`: Check multilingual support
  - `MiniLMVariant::layers()`: Get layer count
  - `MiniLMVariant::hidden_size()`: Get hidden dimension size
- **Factory Functions:**
  - `minilm_l12_h384()`: Standard language model
  - `minilm_multilingual_l12_h384()`: Multilingual model
  - `all_minilm_l6_v2()`, `all_minilm_l12_v2()`: Sentence transformers
  - `ms_marco_minilm_X_v2()`: Cross-encoders
- **Helper Functions:**
  - `language_model_variants()`: Get language model variants only
  - `sentence_transformer_variants()`: Get sentence transformer variants
  - `cross_encoder_variants()`: Get cross-encoder variants

## 5. Key Variables and Logic

### Model Size Categories
- **Tiny**: 1M-50M parameters (ultra-efficient, mobile-ready)
- **Small**: 50M-200M parameters (balanced efficiency/performance)
- **Medium**: 200M-500M parameters (good performance, reasonable resources)
- **Large**: 500M+ parameters (high performance, more resources)

### Model Capabilities
- `text_generation`: Can generate coherent text
- `instruction_following`: Can follow instructions/prompts
- `chat`: Optimized for conversational AI
- `code_generation`: Can generate and understand code
- `reasoning`: Capable of logical reasoning
- `multilingual`: Supports multiple languages

### Loading States
- `NotLoaded`: Model not in memory
- `Downloading(progress)`: Currently downloading with progress %
- `Loading`: Loading into memory
- `Loaded`: Ready for inference
- `Failed(error)`: Loading failed with error message

### Configuration Parameters
- `temperature`: Controls randomness (0.0-2.0)
- `top_k`: Top-k sampling parameter
- `top_p`: Nucleus sampling parameter
- `repetition_penalty`: Penalty for repetitive text
- `max_tokens`: Maximum generation length
- `stop_tokens`: Tokens that end generation

## 6. Dependencies

### Internal Dependencies
- `crate::error::{GraphError, Result}`: Error handling from parent crate
- Cross-module dependencies between model implementations and core types

### External Dependencies
- `serde`: Serialization/deserialization for JSON config
- `tokio`: Async runtime for non-blocking operations
- `walkdir`: Directory traversal for cache management
- `dirs`: System directory discovery
- `std::collections::HashMap`: Key-value storage
- `std::path::{Path, PathBuf}`: File system path handling

## 7. Architecture Patterns

### Builder Pattern
Each model family uses a builder pattern for flexible configuration:
```rust
let model = smollm_360m()
    .with_config(ModelConfig::new().with_temperature(0.8))
    .build()?;
```

### Registry Pattern
Central registry manages all available models with filtering and search capabilities.

### Factory Pattern
Each model family provides factory functions for easy instantiation.

### Strategy Pattern
Different model types (language models, sentence transformers, cross-encoders) implement different capability sets.

## 8. Performance Considerations

### Memory Management
- Lazy loading: Models loaded on demand
- Cache management: Configurable cache directory with cleanup utilities
- Memory estimation: Predictive memory requirements based on parameters and precision

### Async Operations
- Non-blocking downloads from HuggingFace Hub
- Progress tracking for long-running operations
- Concurrent model operations support

### Device Optimization
- Multi-device support (CPU, CUDA, Metal)
- Quantization options for memory efficiency
- Performance tier classification for device recommendation

## 9. Testing

Each module includes comprehensive unit tests covering:
- Model variant enumeration and properties
- Builder pattern functionality
- Configuration management
- Loading and caching operations
- Utility function accuracy
- Error handling scenarios

Tests use `tempfile` for isolated temporary directories and mock implementations for external dependencies.

## 10. Usage Examples

### Basic Model Loading
```rust
let mut loader = ModelLoader::default();
let model = loader.load_model("HuggingFaceTB/SmolLM-360M", None).await?;
let response = model.generate_text("Hello world", Some(50))?;
```

### Registry-Based Discovery
```rust
let registry = ModelRegistry::new();
let small_models = registry.list_models_by_size(ModelSize::Small);
let chat_models = registry.list_models_with_capability(|caps| caps.chat);
```

### Advanced Configuration
```rust
let config = ModelConfig::new()
    .with_temperature(0.8)
    .with_top_k(50)
    .with_max_tokens(100);
    
let model = smollm_360m_instruct()
    .with_config(config)
    .build()?;
```

This models directory provides a complete, production-ready system for managing and working with small language models, with emphasis on efficiency, flexibility, and ease of use.