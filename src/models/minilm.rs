//! MiniLM - Compact and efficient language models by Microsoft
//!
//! MiniLM is a family of distilled models from the paper "MiniLM: Deep Self-Attention 
//! Distillation for Task-Agnostic Compression of Pre-Trained Transformers". These models
//! are designed to be compact and efficient while maintaining high performance.

use super::{ModelMetadata, ModelCapabilities, ModelSize, Model, ModelConfig};
use crate::error::Result;

/// MiniLM model variants
#[derive(Debug, Clone)]
pub enum MiniLMVariant {
    // Microsoft MiniLM models
    MiniLML12H384,
    MiniLMMultilingualL12H384,
    
    // Sentence Transformers MiniLM models
    AllMiniLML6V2,
    AllMiniLML12V2,
    
    // Cross-encoder MiniLM models
    MsMarcoMiniLML6V2,
    MsMarcoMiniLML12V2,
}

impl MiniLMVariant {
    pub fn huggingface_id(&self) -> &'static str {
        match self {
            MiniLMVariant::MiniLML12H384 => "microsoft/MiniLM-L12-H384-uncased",
            MiniLMVariant::MiniLMMultilingualL12H384 => "microsoft/Multilingual-MiniLM-L12-H384",
            MiniLMVariant::AllMiniLML6V2 => "sentence-transformers/all-MiniLM-L6-v2",
            MiniLMVariant::AllMiniLML12V2 => "sentence-transformers/all-MiniLM-L12-v2",
            MiniLMVariant::MsMarcoMiniLML6V2 => "cross-encoder/ms-marco-MiniLM-L6-v2",
            MiniLMVariant::MsMarcoMiniLML12V2 => "cross-encoder/ms-marco-MiniLM-L12-v2",
        }
    }

    pub fn parameters(&self) -> u64 {
        match self {
            MiniLMVariant::AllMiniLML6V2 => 22_000_000,        // 22M parameters
            MiniLMVariant::MsMarcoMiniLML6V2 => 22_000_000,    // 22M parameters
            MiniLMVariant::MiniLML12H384 => 33_000_000,        // 33M parameters
            MiniLMVariant::AllMiniLML12V2 => 33_000_000,       // 33M parameters
            MiniLMVariant::MsMarcoMiniLML12V2 => 33_000_000,   // 33M parameters
            MiniLMVariant::MiniLMMultilingualL12H384 => 118_000_000, // 118M parameters
        }
    }

    pub fn size_category(&self) -> ModelSize {
        match self {
            MiniLMVariant::AllMiniLML6V2 | MiniLMVariant::MsMarcoMiniLML6V2 => ModelSize::Tiny,
            MiniLMVariant::MiniLML12H384 | MiniLMVariant::AllMiniLML12V2 | MiniLMVariant::MsMarcoMiniLML12V2 => ModelSize::Tiny,
            MiniLMVariant::MiniLMMultilingualL12H384 => ModelSize::Small,
        }
    }

    pub fn model_type(&self) -> MiniLMType {
        match self {
            MiniLMVariant::MiniLML12H384 | MiniLMVariant::MiniLMMultilingualL12H384 => MiniLMType::LanguageModel,
            MiniLMVariant::AllMiniLML6V2 | MiniLMVariant::AllMiniLML12V2 => MiniLMType::SentenceTransformer,
            MiniLMVariant::MsMarcoMiniLML6V2 | MiniLMVariant::MsMarcoMiniLML12V2 => MiniLMType::CrossEncoder,
        }
    }

    pub fn is_multilingual(&self) -> bool {
        matches!(self, MiniLMVariant::MiniLMMultilingualL12H384)
    }

    pub fn layers(&self) -> u32 {
        match self {
            MiniLMVariant::AllMiniLML6V2 | MiniLMVariant::MsMarcoMiniLML6V2 => 6,
            _ => 12,
        }
    }

    pub fn hidden_size(&self) -> u32 {
        match self {
            MiniLMVariant::MiniLML12H384 | MiniLMVariant::MiniLMMultilingualL12H384 => 384,
            MiniLMVariant::AllMiniLML6V2 | MiniLMVariant::AllMiniLML12V2 => 384,
            MiniLMVariant::MsMarcoMiniLML6V2 | MiniLMVariant::MsMarcoMiniLML12V2 => 384,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            MiniLMVariant::MiniLML12H384 => "MiniLM-L12-H384-uncased",
            MiniLMVariant::MiniLMMultilingualL12H384 => "Multilingual-MiniLM-L12-H384",
            MiniLMVariant::AllMiniLML6V2 => "all-MiniLM-L6-v2",
            MiniLMVariant::AllMiniLML12V2 => "all-MiniLM-L12-v2",
            MiniLMVariant::MsMarcoMiniLML6V2 => "ms-marco-MiniLM-L6-v2",
            MiniLMVariant::MsMarcoMiniLML12V2 => "ms-marco-MiniLM-L12-v2",
        }
    }

    pub fn max_seq_length(&self) -> u32 {
        match self {
            MiniLMVariant::AllMiniLML6V2 | MiniLMVariant::AllMiniLML12V2 => 256,
            MiniLMVariant::MsMarcoMiniLML6V2 | MiniLMVariant::MsMarcoMiniLML12V2 => 512,
            _ => 512,
        }
    }
}

/// MiniLM model types
#[derive(Debug, Clone, PartialEq)]
pub enum MiniLMType {
    LanguageModel,      // Standard language model
    SentenceTransformer, // Sentence embedding model
    CrossEncoder,       // Cross-encoder for ranking/classification
}

/// MiniLM model builder
pub struct MiniLMBuilder {
    variant: MiniLMVariant,
    config: ModelConfig,
}

impl MiniLMBuilder {
    pub fn new(variant: MiniLMVariant) -> Self {
        Self {
            variant,
            config: ModelConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = config;
        self
    }

    pub fn build(self) -> Result<Model> {
        let model_type = self.variant.model_type();
        
        let capabilities = ModelCapabilities {
            text_generation: model_type == MiniLMType::LanguageModel,
            instruction_following: false,
            chat: false,
            code_generation: model_type == MiniLMType::LanguageModel,
            reasoning: model_type == MiniLMType::LanguageModel,
            multilingual: self.variant.is_multilingual(),
        };

        let metadata = ModelMetadata {
            name: self.variant.name().to_string(),
            family: "MiniLM".to_string(),
            parameters: self.variant.parameters(),
            size_category: self.variant.size_category(),
            huggingface_id: self.variant.huggingface_id().to_string(),
            architecture: "BERT".to_string(),
            capabilities,
            context_length: self.variant.max_seq_length(),
            vocab_size: if self.variant.is_multilingual() { 250002 } else { 30522 },
            training_tokens: None, // Training token count varies by model
            release_date: "2020-09".to_string(),
            license: "MIT".to_string(),
            description: format!(
                "MiniLM {} is a {} with {} parameters and {} layers. {} designed for {} tasks and offers excellent efficiency while maintaining high performance through deep self-attention distillation.",
                self.variant.name(),
                match model_type {
                    MiniLMType::LanguageModel => "compact language model",
                    MiniLMType::SentenceTransformer => "sentence embedding model",
                    MiniLMType::CrossEncoder => "cross-encoder model",
                },
                format_parameters(self.variant.parameters()),
                self.variant.layers(),
                if self.variant.is_multilingual() { "This multilingual model is" } else { "It is" },
                match model_type {
                    MiniLMType::LanguageModel => "general language understanding",
                    MiniLMType::SentenceTransformer => "sentence similarity and semantic search",
                    MiniLMType::CrossEncoder => "text ranking and classification",
                }
            ),
        };

        Ok(Model::new(metadata, self.config))
    }
}

/// Create MiniLM-L12-H384-uncased model
pub fn minilm_l12_h384() -> MiniLMBuilder {
    MiniLMBuilder::new(MiniLMVariant::MiniLML12H384)
}

/// Create Multilingual-MiniLM-L12-H384 model
pub fn minilm_multilingual_l12_h384() -> MiniLMBuilder {
    MiniLMBuilder::new(MiniLMVariant::MiniLMMultilingualL12H384)
}

/// Create all-MiniLM-L6-v2 sentence transformer model
pub fn all_minilm_l6_v2() -> MiniLMBuilder {
    MiniLMBuilder::new(MiniLMVariant::AllMiniLML6V2)
}

/// Create all-MiniLM-L12-v2 sentence transformer model
pub fn all_minilm_l12_v2() -> MiniLMBuilder {
    MiniLMBuilder::new(MiniLMVariant::AllMiniLML12V2)
}

/// Create ms-marco-MiniLM-L6-v2 cross-encoder model
pub fn ms_marco_minilm_l6_v2() -> MiniLMBuilder {
    MiniLMBuilder::new(MiniLMVariant::MsMarcoMiniLML6V2)
}

/// Create ms-marco-MiniLM-L12-v2 cross-encoder model
pub fn ms_marco_minilm_l12_v2() -> MiniLMBuilder {
    MiniLMBuilder::new(MiniLMVariant::MsMarcoMiniLML12V2)
}

/// Get all available MiniLM variants
pub fn available_variants() -> Vec<MiniLMVariant> {
    vec![
        MiniLMVariant::MiniLML12H384,
        MiniLMVariant::MiniLMMultilingualL12H384,
        MiniLMVariant::AllMiniLML6V2,
        MiniLMVariant::AllMiniLML12V2,
        MiniLMVariant::MsMarcoMiniLML6V2,
        MiniLMVariant::MsMarcoMiniLML12V2,
    ]
}

/// Get language model variants only
pub fn language_model_variants() -> Vec<MiniLMVariant> {
    vec![
        MiniLMVariant::MiniLML12H384,
        MiniLMVariant::MiniLMMultilingualL12H384,
    ]
}

/// Get sentence transformer variants only
pub fn sentence_transformer_variants() -> Vec<MiniLMVariant> {
    vec![
        MiniLMVariant::AllMiniLML6V2,
        MiniLMVariant::AllMiniLML12V2,
    ]
}

/// Get cross-encoder variants only
pub fn cross_encoder_variants() -> Vec<MiniLMVariant> {
    vec![
        MiniLMVariant::MsMarcoMiniLML6V2,
        MiniLMVariant::MsMarcoMiniLML12V2,
    ]
}

/// Format parameter count in human-readable form
fn format_parameters(params: u64) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else if params >= 1_000_000 {
        format!("{}M", params / 1_000_000)
    } else if params >= 1_000 {
        format!("{}K", params / 1_000)
    } else {
        params.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minilm_variants() {
        let variants = available_variants();
        assert_eq!(variants.len(), 6);
        
        let lm_variants = language_model_variants();
        assert_eq!(lm_variants.len(), 2);
        
        let st_variants = sentence_transformer_variants();
        assert_eq!(st_variants.len(), 2);
        
        let ce_variants = cross_encoder_variants();
        assert_eq!(ce_variants.len(), 2);
        
        // Test parameter counts
        assert_eq!(MiniLMVariant::AllMiniLML6V2.parameters(), 22_000_000);
        assert_eq!(MiniLMVariant::MiniLML12H384.parameters(), 33_000_000);
        assert_eq!(MiniLMVariant::MiniLMMultilingualL12H384.parameters(), 118_000_000);
        
        // Test HuggingFace IDs
        assert_eq!(MiniLMVariant::AllMiniLML6V2.huggingface_id(), "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(MiniLMVariant::MiniLML12H384.huggingface_id(), "microsoft/MiniLM-L12-H384-uncased");
        
        // Test model types
        assert_eq!(MiniLMVariant::MiniLML12H384.model_type(), MiniLMType::LanguageModel);
        assert_eq!(MiniLMVariant::AllMiniLML6V2.model_type(), MiniLMType::SentenceTransformer);
        assert_eq!(MiniLMVariant::MsMarcoMiniLML6V2.model_type(), MiniLMType::CrossEncoder);
        
        // Test multilingual
        assert!(MiniLMVariant::MiniLMMultilingualL12H384.is_multilingual());
        assert!(!MiniLMVariant::MiniLML12H384.is_multilingual());
        
        // Test layers and hidden size
        assert_eq!(MiniLMVariant::AllMiniLML6V2.layers(), 6);
        assert_eq!(MiniLMVariant::MiniLML12H384.layers(), 12);
        assert_eq!(MiniLMVariant::AllMiniLML6V2.hidden_size(), 384);
    }

    #[test]
    fn test_minilm_builder() {
        let model = minilm_l12_h384().build().unwrap();
        assert_eq!(model.metadata.parameters, 33_000_000);
        assert_eq!(model.metadata.huggingface_id, "microsoft/MiniLM-L12-H384-uncased");
        assert!(model.metadata.capabilities.text_generation);
        assert!(!model.metadata.capabilities.instruction_following);
        assert!(!model.metadata.capabilities.multilingual);
        assert_eq!(model.metadata.architecture, "BERT");
        
        let multilingual_model = minilm_multilingual_l12_h384().build().unwrap();
        assert!(multilingual_model.metadata.capabilities.multilingual);
        assert_eq!(multilingual_model.metadata.vocab_size, 250002);
        
        let sentence_model = all_minilm_l6_v2().build().unwrap();
        assert!(!sentence_model.metadata.capabilities.text_generation);
        assert_eq!(sentence_model.metadata.context_length, 256);
        assert_eq!(sentence_model.metadata.family, "MiniLM");
    }

    #[test]
    fn test_format_parameters() {
        assert_eq!(format_parameters(22_000_000), "22M");
        assert_eq!(format_parameters(33_000_000), "33M");
        assert_eq!(format_parameters(118_000_000), "118M");
        assert_eq!(format_parameters(1_000), "1K");
        assert_eq!(format_parameters(500), "500");
    }
}