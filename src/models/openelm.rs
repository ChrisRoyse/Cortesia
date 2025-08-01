//! OpenELM - Open Efficient Language Models by Apple
//!
//! OpenELM is a family of efficient language models with parameters ranging from
//! 270M to 3B, developed by Apple with a focus on energy efficiency and on-device
//! AI applications.

use super::{ModelMetadata, ModelCapabilities, ModelSize, Model, ModelConfig};
use crate::error::Result;

/// OpenELM model variants
#[derive(Debug, Clone)]
pub enum OpenELMVariant {
    OpenELM270M,
    OpenELM450M,
    OpenELM1_1B,
    OpenELM3B,
    OpenELM270MInstruct,
    OpenELM450MInstruct,
    OpenELM1_1BInstruct,
    OpenELM3BInstruct,
}

impl OpenELMVariant {
    pub fn huggingface_id(&self) -> &'static str {
        match self {
            OpenELMVariant::OpenELM270M => "apple/OpenELM-270M",
            OpenELMVariant::OpenELM450M => "apple/OpenELM-450M",
            OpenELMVariant::OpenELM1_1B => "apple/OpenELM-1.1B",
            OpenELMVariant::OpenELM3B => "apple/OpenELM-3B",
            OpenELMVariant::OpenELM270MInstruct => "apple/OpenELM-270M-Instruct",
            OpenELMVariant::OpenELM450MInstruct => "apple/OpenELM-450M-Instruct",
            OpenELMVariant::OpenELM1_1BInstruct => "apple/OpenELM-1.1B-Instruct",
            OpenELMVariant::OpenELM3BInstruct => "apple/OpenELM-3B-Instruct",
        }
    }

    pub fn parameters(&self) -> u64 {
        match self {
            OpenELMVariant::OpenELM270M | OpenELMVariant::OpenELM270MInstruct => 270_000_000,
            OpenELMVariant::OpenELM450M | OpenELMVariant::OpenELM450MInstruct => 450_000_000,
            OpenELMVariant::OpenELM1_1B | OpenELMVariant::OpenELM1_1BInstruct => 1_100_000_000,
            OpenELMVariant::OpenELM3B | OpenELMVariant::OpenELM3BInstruct => 3_000_000_000,
        }
    }

    pub fn size_category(&self) -> ModelSize {
        match self {
            OpenELMVariant::OpenELM270M | OpenELMVariant::OpenELM270MInstruct => ModelSize::Medium,
            OpenELMVariant::OpenELM450M | OpenELMVariant::OpenELM450MInstruct => ModelSize::Medium,
            OpenELMVariant::OpenELM1_1B | OpenELMVariant::OpenELM1_1BInstruct => ModelSize::Large,
            OpenELMVariant::OpenELM3B | OpenELMVariant::OpenELM3BInstruct => ModelSize::Large,
        }
    }

    pub fn is_instruct(&self) -> bool {
        matches!(self, 
            OpenELMVariant::OpenELM270MInstruct | 
            OpenELMVariant::OpenELM450MInstruct |
            OpenELMVariant::OpenELM1_1BInstruct |
            OpenELMVariant::OpenELM3BInstruct
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            OpenELMVariant::OpenELM270M => "OpenELM-270M",
            OpenELMVariant::OpenELM450M => "OpenELM-450M",
            OpenELMVariant::OpenELM1_1B => "OpenELM-1.1B",
            OpenELMVariant::OpenELM3B => "OpenELM-3B",
            OpenELMVariant::OpenELM270MInstruct => "OpenELM-270M-Instruct",
            OpenELMVariant::OpenELM450MInstruct => "OpenELM-450M-Instruct",
            OpenELMVariant::OpenELM1_1BInstruct => "OpenELM-1.1B-Instruct",
            OpenELMVariant::OpenELM3BInstruct => "OpenELM-3B-Instruct",
        }
    }

    pub fn context_length(&self) -> u32 {
        2048 // Standard context length for OpenELM models
    }

    pub fn vocab_size(&self) -> u32 {
        32000 // Standard vocabulary size
    }
}

/// OpenELM model builder
pub struct OpenELMBuilder {
    variant: OpenELMVariant,
    config: ModelConfig,
}

impl OpenELMBuilder {
    pub fn new(variant: OpenELMVariant) -> Self {
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
            instruction_following: self.variant.is_instruct(),
            chat: self.variant.is_instruct(),
            code_generation: true,
            reasoning: true,
            multilingual: true, // OpenELM supports multiple languages
        };

        let metadata = ModelMetadata {
            name: self.variant.name().to_string(),
            family: "OpenELM".to_string(),
            parameters: self.variant.parameters(),
            size_category: self.variant.size_category(),
            huggingface_id: self.variant.huggingface_id().to_string(),
            architecture: "Transformer".to_string(),
            capabilities,
            context_length: self.variant.context_length(),
            vocab_size: self.variant.vocab_size(),
            training_tokens: None, // Training token count not publicly disclosed
            release_date: "2024-04".to_string(),
            license: "Apple Sample Code License".to_string(),
            description: format!(
                "OpenELM {} is an efficient language model with {} parameters developed by Apple. {} focuses on energy efficiency and on-device AI applications, competing well against models like MobileLlama and OLMo while being optimized for mobile and edge devices.",
                self.variant.name(),
                format_parameters(self.variant.parameters()),
                if self.variant.is_instruct() { "This instruction-tuned variant" } else { "It" }
            ),
        };

        Ok(Model::new(metadata, self.config))
    }
}

/// Create OpenELM-270M model
pub fn openelm_270m() -> OpenELMBuilder {
    OpenELMBuilder::new(OpenELMVariant::OpenELM270M)
}

/// Create OpenELM-450M model
pub fn openelm_450m() -> OpenELMBuilder {
    OpenELMBuilder::new(OpenELMVariant::OpenELM450M)
}

/// Create OpenELM-1.1B model
pub fn openelm_1_1b() -> OpenELMBuilder {
    OpenELMBuilder::new(OpenELMVariant::OpenELM1_1B)
}

/// Create OpenELM-3B model
pub fn openelm_3b() -> OpenELMBuilder {
    OpenELMBuilder::new(OpenELMVariant::OpenELM3B)
}

/// Create OpenELM-270M-Instruct model
pub fn openelm_270m_instruct() -> OpenELMBuilder {
    OpenELMBuilder::new(OpenELMVariant::OpenELM270MInstruct)
}

/// Create OpenELM-450M-Instruct model
pub fn openelm_450m_instruct() -> OpenELMBuilder {
    OpenELMBuilder::new(OpenELMVariant::OpenELM450MInstruct)
}

/// Create OpenELM-1.1B-Instruct model
pub fn openelm_1_1b_instruct() -> OpenELMBuilder {
    OpenELMBuilder::new(OpenELMVariant::OpenELM1_1BInstruct)
}

/// Create OpenELM-3B-Instruct model
pub fn openelm_3b_instruct() -> OpenELMBuilder {
    OpenELMBuilder::new(OpenELMVariant::OpenELM3BInstruct)
}

/// Get all available OpenELM variants
pub fn available_variants() -> Vec<OpenELMVariant> {
    vec![
        OpenELMVariant::OpenELM270M,
        OpenELMVariant::OpenELM450M,
        OpenELMVariant::OpenELM1_1B,
        OpenELMVariant::OpenELM3B,
        OpenELMVariant::OpenELM270MInstruct,
        OpenELMVariant::OpenELM450MInstruct,
        OpenELMVariant::OpenELM1_1BInstruct,
        OpenELMVariant::OpenELM3BInstruct,
    ]
}

/// Get models in the 1M-500M parameter range
pub fn small_variants() -> Vec<OpenELMVariant> {
    vec![
        OpenELMVariant::OpenELM270M,
        OpenELMVariant::OpenELM450M,
        OpenELMVariant::OpenELM270MInstruct,
        OpenELMVariant::OpenELM450MInstruct,
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
    fn test_openelm_variants() {
        let variants = available_variants();
        assert_eq!(variants.len(), 8);
        
        let small_variants = small_variants();
        assert_eq!(small_variants.len(), 4);
        
        // Test parameter counts
        assert_eq!(OpenELMVariant::OpenELM270M.parameters(), 270_000_000);
        assert_eq!(OpenELMVariant::OpenELM450M.parameters(), 450_000_000);
        assert_eq!(OpenELMVariant::OpenELM1_1B.parameters(), 1_100_000_000);
        assert_eq!(OpenELMVariant::OpenELM3B.parameters(), 3_000_000_000);
        
        // Test HuggingFace IDs
        assert_eq!(OpenELMVariant::OpenELM270M.huggingface_id(), "apple/OpenELM-270M");
        assert_eq!(OpenELMVariant::OpenELM450M.huggingface_id(), "apple/OpenELM-450M");
        
        // Test instruct variants
        assert!(OpenELMVariant::OpenELM270MInstruct.is_instruct());
        assert!(!OpenELMVariant::OpenELM270M.is_instruct());
        
        // Test size categories
        assert_eq!(OpenELMVariant::OpenELM270M.size_category(), ModelSize::Medium);
        assert_eq!(OpenELMVariant::OpenELM450M.size_category(), ModelSize::Medium);
        assert_eq!(OpenELMVariant::OpenELM1_1B.size_category(), ModelSize::Large);
        assert_eq!(OpenELMVariant::OpenELM3B.size_category(), ModelSize::Large);
    }

    #[test]
    fn test_openelm_builder() {
        let model = openelm_270m().build().unwrap();
        assert_eq!(model.metadata.parameters, 270_000_000);
        assert_eq!(model.metadata.huggingface_id, "apple/OpenELM-270M");
        assert!(model.metadata.capabilities.text_generation);
        assert!(!model.metadata.capabilities.instruction_following);
        assert!(model.metadata.capabilities.multilingual);
        assert_eq!(model.metadata.license, "Apple Sample Code License");
        
        let instruct_model = openelm_450m_instruct().build().unwrap();
        assert!(instruct_model.metadata.capabilities.instruction_following);
        assert!(instruct_model.metadata.capabilities.chat);
        assert_eq!(instruct_model.metadata.family, "OpenELM");
    }

    #[test]
    fn test_format_parameters() {
        assert_eq!(format_parameters(270_000_000), "270M");
        assert_eq!(format_parameters(450_000_000), "450M");
        assert_eq!(format_parameters(1_100_000_000), "1.1B");
        assert_eq!(format_parameters(3_000_000_000), "3.0B");
        assert_eq!(format_parameters(1_000), "1K");
        assert_eq!(format_parameters(500), "500");
    }
}