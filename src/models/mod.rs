//! Native Rust AI Models Module for LLMKG
//! 
//! This module provides native Rust implementations of AI models for Phase 1:
//! - Entity extraction (BERT-based NER models)
//! - Text generation (T5-based models)
//! - Semantic similarity (Sentence transformers)
//! - Tokenization (Native Rust tokenizer)

// Native Rust model implementations
pub mod rust_tokenizer;
pub mod rust_bert_models;
pub mod rust_t5_models;
pub mod rust_embeddings;
pub mod model_loader;

// Re-export key types for convenience
pub use rust_bert_models::{RustBertNER, RustTinyBertNER, Entity};
pub use rust_t5_models::RustT5Small;
pub use rust_embeddings::RustMiniLM;
pub use rust_tokenizer::{RustTokenizer, TokenizedInput};

/// Model types available in the native Rust system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// DistilBERT fine-tuned for Named Entity Recognition (66M params)
    DistilBertNER,
    /// TinyBERT fine-tuned for NER - lightweight alternative (14.5M params)
    TinyBertNER,
    /// T5-Small for text generation and answer synthesis (60M params)
    T5Small,
    /// all-MiniLM-L6-v2 for semantic similarity and embeddings (22M params)
    MiniLM,
}

impl ModelType {
    /// Get the parameter count for this model
    pub fn param_count(&self) -> usize {
        match self {
            ModelType::DistilBertNER => 66_000_000,
            ModelType::TinyBertNER => 14_500_000,
            ModelType::T5Small => 60_000_000,
            ModelType::MiniLM => 22_000_000,
        }
    }
    
    /// Get the recommended batch size for optimal performance
    pub fn recommended_batch_size(&self) -> usize {
        match self {
            ModelType::DistilBertNER => 32,
            ModelType::TinyBertNER => 64,
            ModelType::T5Small => 16,
            ModelType::MiniLM => 64,
        }
    }
    
    /// Get the model description
    pub fn description(&self) -> &'static str {
        match self {
            ModelType::DistilBertNER => "DistilBERT fine-tuned for Named Entity Recognition",
            ModelType::TinyBertNER => "TinyBERT lightweight model for Named Entity Recognition",
            ModelType::T5Small => "T5-Small for text generation and answer synthesis",
            ModelType::MiniLM => "all-MiniLM-L6-v2 for semantic similarity and embeddings",
        }
    }
}

/// Model loading error types
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Model initialization error: {0}")]
    InitializationError(String),
    
    #[error("Invalid model format: {0}")]
    InvalidFormat(String),
    
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    #[error("Matrix operation error: {0}")]
    MatrixError(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;

/// Performance metrics for model inference
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Model type
    pub model: ModelType,
    /// Average inference time in milliseconds
    pub avg_inference_ms: f64,
    /// Peak inference time in milliseconds
    pub peak_inference_ms: f64,
    /// Throughput (items per second)
    pub throughput: f64,
    /// Memory usage in MB
    pub memory_mb: f64,
}

/// Factory for creating native Rust models
pub struct ModelFactory;

impl ModelFactory {
    /// Create a new BERT NER model
    pub fn create_bert_ner() -> RustBertNER {
        RustBertNER::new()
    }
    
    /// Create a new TinyBERT NER model
    pub fn create_tiny_bert_ner() -> RustTinyBertNER {
        RustTinyBertNER::new()
    }
    
    /// Create a new T5 model
    pub fn create_t5_small() -> RustT5Small {
        RustT5Small::new()
    }
    
    /// Create a new MiniLM model
    pub fn create_mini_lm() -> RustMiniLM {
        RustMiniLM::new()
    }
    
    /// Create a new tokenizer
    pub fn create_tokenizer() -> RustTokenizer {
        RustTokenizer::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_type_properties() {
        assert_eq!(ModelType::DistilBertNER.param_count(), 66_000_000);
        assert_eq!(ModelType::TinyBertNER.recommended_batch_size(), 64);
        assert_eq!(ModelType::T5Small.description(), "T5-Small for text generation and answer synthesis");
    }
    
    #[test]
    fn test_model_factory() {
        let bert_ner = ModelFactory::create_bert_ner();
        let tiny_bert = ModelFactory::create_tiny_bert_ner();
        let t5 = ModelFactory::create_t5_small();
        let mini_lm = ModelFactory::create_mini_lm();
        let tokenizer = ModelFactory::create_tokenizer();
        
        // Basic sanity checks that models can be created
        assert_eq!(bert_ner.num_labels, 9);
        assert_eq!(tiny_bert.label_map.len(), 9);
        assert_eq!(t5.model.vocab_size, 32128);
        assert_eq!(mini_lm.transformer.normalize, true);
        // Note: max_length is private, so we'll check it indirectly
        assert!(tokenizer.encode("test", true).input_ids.len() > 0);
    }
    
    #[test]
    fn test_all_model_types() {
        let model_types = [
            ModelType::DistilBertNER,
            ModelType::TinyBertNER,
            ModelType::T5Small,
            ModelType::MiniLM,
        ];
        
        for model_type in model_types {
            assert!(model_type.param_count() > 0);
            assert!(model_type.recommended_batch_size() > 0);
            assert!(!model_type.description().is_empty());
        }
    }
}