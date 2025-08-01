//! TinyLlama - Compact 1.1B language model pretrained on ~1 trillion tokens
//!
//! TinyLlama is a compact 1.1B language model that builds on the architecture and 
//! tokenizer of Llama 2, achieving better computational efficiency through various
//! advances like FlashAttention.

use super::{ModelMetadata, ModelCapabilities, ModelSize, Model, ModelConfig};
use crate::error::Result;

/// TinyLlama model variants
#[derive(Debug, Clone)]
pub enum TinyLlamaVariant {
    TinyLlama1_1B,
    TinyLlama1_1BChat,
    TinyLlama1_1BChatV0_1,
    TinyLlama1_1BChatV0_3,
    TinyLlama1_1BChatV0_6,
    TinyLlama1_1BChatV1_0,
    TinyLlama1_1BIntermediate,
}

impl TinyLlamaVariant {
    pub fn huggingface_id(&self) -> &'static str {
        match self {
            TinyLlamaVariant::TinyLlama1_1B => "TinyLlama/TinyLlama_v1.1",
            TinyLlamaVariant::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            TinyLlamaVariant::TinyLlama1_1BChatV0_1 => "TinyLlama/TinyLlama-1.1B-Chat-v0.1",
            TinyLlamaVariant::TinyLlama1_1BChatV0_3 => "TinyLlama/TinyLlama-1.1B-Chat-v0.3",
            TinyLlamaVariant::TinyLlama1_1BChatV0_6 => "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            TinyLlamaVariant::TinyLlama1_1BChatV1_0 => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            TinyLlamaVariant::TinyLlama1_1BIntermediate => "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        }
    }

    pub fn parameters(&self) -> u64 {
        1_100_000_000 // All variants have 1.1B parameters
    }

    pub fn size_category(&self) -> ModelSize {
        ModelSize::Large // 1.1B is considered large in our categorization
    }

    pub fn is_chat(&self) -> bool {
        matches!(self, 
            TinyLlamaVariant::TinyLlama1_1BChat |
            TinyLlamaVariant::TinyLlama1_1BChatV0_1 |
            TinyLlamaVariant::TinyLlama1_1BChatV0_3 |
            TinyLlamaVariant::TinyLlama1_1BChatV0_6 |
            TinyLlamaVariant::TinyLlama1_1BChatV1_0
        )
    }

    pub fn version(&self) -> &'static str {
        match self {
            TinyLlamaVariant::TinyLlama1_1B => "v1.1",
            TinyLlamaVariant::TinyLlama1_1BChat => "Chat-v1.0",
            TinyLlamaVariant::TinyLlama1_1BChatV0_1 => "Chat-v0.1",
            TinyLlamaVariant::TinyLlama1_1BChatV0_3 => "Chat-v0.3",
            TinyLlamaVariant::TinyLlama1_1BChatV0_6 => "Chat-v0.6",
            TinyLlamaVariant::TinyLlama1_1BChatV1_0 => "Chat-v1.0",
            TinyLlamaVariant::TinyLlama1_1BIntermediate => "Intermediate-3T",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TinyLlamaVariant::TinyLlama1_1B => "TinyLlama-1.1B",
            TinyLlamaVariant::TinyLlama1_1BChat => "TinyLlama-1.1B-Chat",
            TinyLlamaVariant::TinyLlama1_1BChatV0_1 => "TinyLlama-1.1B-Chat-v0.1",
            TinyLlamaVariant::TinyLlama1_1BChatV0_3 => "TinyLlama-1.1B-Chat-v0.3",
            TinyLlamaVariant::TinyLlama1_1BChatV0_6 => "TinyLlama-1.1B-Chat-v0.6",
            TinyLlamaVariant::TinyLlama1_1BChatV1_0 => "TinyLlama-1.1B-Chat-v1.0",
            TinyLlamaVariant::TinyLlama1_1BIntermediate => "TinyLlama-1.1B-Intermediate-3T",
        }
    }

    pub fn training_tokens(&self) -> u64 {
        match self {
            TinyLlamaVariant::TinyLlama1_1BIntermediate => 3_000_000_000_000, // 3T tokens
            _ => 1_000_000_000_000, // ~1T tokens for standard variants
        }
    }
}

/// TinyLlama model builder
pub struct TinyLlamaBuilder {
    variant: TinyLlamaVariant,
    config: ModelConfig,
}

impl TinyLlamaBuilder {
    pub fn new(variant: TinyLlamaVariant) -> Self {
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
        let capabilities = ModelCapabilities {
            text_generation: true,
            instruction_following: self.variant.is_chat(),
            chat: self.variant.is_chat(),
            code_generation: true,
            reasoning: true,
            multilingual: false,
        };

        let metadata = ModelMetadata {
            name: self.variant.name().to_string(),
            family: "TinyLlama".to_string(),
            parameters: self.variant.parameters(),
            size_category: self.variant.size_category(),
            huggingface_id: self.variant.huggingface_id().to_string(),
            architecture: "Llama".to_string(),
            capabilities,
            context_length: 2048,
            vocab_size: 32000,
            training_tokens: Some(self.variant.training_tokens()),
            release_date: "2024-01".to_string(),
            license: "Apache-2.0".to_string(),
            description: format!(
                "TinyLlama {} is a compact 1.1B language model built on the Llama 2 architecture and tokenizer, trained on {} tokens. {} features FlashAttention for improved efficiency and performs better than models like Pythia-1.4B for commonsense reasoning.",
                self.variant.version(),
                format_tokens(self.variant.training_tokens()),
                if self.variant.is_chat() { "This chat-tuned variant" } else { "It" }
            ),
        };

        Ok(Model::new(metadata, self.config))
    }
}

/// Create TinyLlama-1.1B base model
pub fn tinyllama_1_1b() -> TinyLlamaBuilder {
    TinyLlamaBuilder::new(TinyLlamaVariant::TinyLlama1_1B)
}

/// Create TinyLlama-1.1B-Chat model (latest)
pub fn tinyllama_1_1b_chat() -> TinyLlamaBuilder {
    TinyLlamaBuilder::new(TinyLlamaVariant::TinyLlama1_1BChat)
}

/// Create TinyLlama-1.1B-Chat-v0.1 model
pub fn tinyllama_1_1b_chat_v0_1() -> TinyLlamaBuilder {
    TinyLlamaBuilder::new(TinyLlamaVariant::TinyLlama1_1BChatV0_1)
}

/// Create TinyLlama-1.1B-Chat-v0.3 model
pub fn tinyllama_1_1b_chat_v0_3() -> TinyLlamaBuilder {
    TinyLlamaBuilder::new(TinyLlamaVariant::TinyLlama1_1BChatV0_3)
}

/// Create TinyLlama-1.1B-Chat-v0.6 model
pub fn tinyllama_1_1b_chat_v0_6() -> TinyLlamaBuilder {
    TinyLlamaBuilder::new(TinyLlamaVariant::TinyLlama1_1BChatV0_6)
}

/// Create TinyLlama-1.1B-Chat-v1.0 model
pub fn tinyllama_1_1b_chat_v1_0() -> TinyLlamaBuilder {
    TinyLlamaBuilder::new(TinyLlamaVariant::TinyLlama1_1BChatV1_0)
}

/// Create TinyLlama-1.1B-Intermediate-3T model (trained on 3T tokens)
pub fn tinyllama_1_1b_intermediate() -> TinyLlamaBuilder {
    TinyLlamaBuilder::new(TinyLlamaVariant::TinyLlama1_1BIntermediate)
}

/// Get all available TinyLlama variants
pub fn available_variants() -> Vec<TinyLlamaVariant> {
    vec![
        TinyLlamaVariant::TinyLlama1_1B,
        TinyLlamaVariant::TinyLlama1_1BChat,
        TinyLlamaVariant::TinyLlama1_1BChatV0_1,
        TinyLlamaVariant::TinyLlama1_1BChatV0_3,
        TinyLlamaVariant::TinyLlama1_1BChatV0_6,
        TinyLlamaVariant::TinyLlama1_1BChatV1_0,
        TinyLlamaVariant::TinyLlama1_1BIntermediate,
    ]
}

/// Format token count in human-readable form
fn format_tokens(tokens: u64) -> String {
    if tokens >= 1_000_000_000_000 {
        format!("{:.1}T", tokens as f64 / 1_000_000_000_000.0)
    } else if tokens >= 1_000_000_000 {
        format!("{:.1}B", tokens as f64 / 1_000_000_000.0)
    } else if tokens >= 1_000_000 {
        format!("{}M", tokens / 1_000_000)
    } else if tokens >= 1_000 {
        format!("{}K", tokens / 1_000)
    } else {
        tokens.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tinyllama_variants() {
        let variants = available_variants();
        assert_eq!(variants.len(), 7);
        
        // Test parameter counts
        for variant in &variants {
            assert_eq!(variant.parameters(), 1_100_000_000);
        }
        
        // Test HuggingFace IDs
        assert_eq!(TinyLlamaVariant::TinyLlama1_1B.huggingface_id(), "TinyLlama/TinyLlama_v1.1");
        assert_eq!(TinyLlamaVariant::TinyLlama1_1BChat.huggingface_id(), "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
        
        // Test chat variants
        assert!(TinyLlamaVariant::TinyLlama1_1BChat.is_chat());
        assert!(!TinyLlamaVariant::TinyLlama1_1B.is_chat());
        assert!(TinyLlamaVariant::TinyLlama1_1BChatV0_3.is_chat());
        
        // Test training tokens
        assert_eq!(TinyLlamaVariant::TinyLlama1_1B.training_tokens(), 1_000_000_000_000);
        assert_eq!(TinyLlamaVariant::TinyLlama1_1BIntermediate.training_tokens(), 3_000_000_000_000);
    }

    #[test]
    fn test_tinyllama_builder() {
        let model = tinyllama_1_1b().build().unwrap();
        assert_eq!(model.metadata.parameters, 1_100_000_000);
        assert_eq!(model.metadata.huggingface_id, "TinyLlama/TinyLlama_v1.1");
        assert!(model.metadata.capabilities.text_generation);
        assert!(!model.metadata.capabilities.instruction_following);
        
        let chat_model = tinyllama_1_1b_chat().build().unwrap();
        assert!(chat_model.metadata.capabilities.instruction_following);
        assert!(chat_model.metadata.capabilities.chat);
        assert_eq!(chat_model.metadata.architecture, "Llama");
    }

    #[test]
    fn test_format_tokens() {
        assert_eq!(format_tokens(1_000_000_000_000), "1.0T");
        assert_eq!(format_tokens(3_000_000_000_000), "3.0T");
        assert_eq!(format_tokens(600_000_000_000), "600.0B");
        assert_eq!(format_tokens(1_000_000), "1M");
        assert_eq!(format_tokens(1_000), "1K");
        assert_eq!(format_tokens(500), "500");
    }
}