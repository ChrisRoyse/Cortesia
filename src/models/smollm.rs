//! SmolLM - State-of-the-art small language models by HuggingFace
//!
//! SmolLM is a family of small language models with 135M, 360M, and 1.7B parameters.
//! These models are optimized for efficiency and can run on edge devices while
//! maintaining competitive performance.

use super::{ModelMetadata, ModelCapabilities, ModelSize, Model, ModelConfig};
use crate::error::Result;

/// SmolLM model variants
#[derive(Debug, Clone)]
pub enum SmolLMVariant {
    SmolLM135M,
    SmolLM360M,
    SmolLM1_7B,
    SmolLM135MInstruct,
    SmolLM360MInstruct,
    SmolLM1_7BInstruct,
    SmolLM2_135M,
    SmolLM2_360M,
    SmolLM2_1_7B,
}

impl SmolLMVariant {
    pub fn huggingface_id(&self) -> &'static str {
        match self {
            SmolLMVariant::SmolLM135M => "HuggingFaceTB/SmolLM-135M",
            SmolLMVariant::SmolLM360M => "HuggingFaceTB/SmolLM-360M",
            SmolLMVariant::SmolLM1_7B => "HuggingFaceTB/SmolLM-1.7B",
            SmolLMVariant::SmolLM135MInstruct => "HuggingFaceTB/SmolLM-135M-Instruct",
            SmolLMVariant::SmolLM360MInstruct => "HuggingFaceTB/SmolLM-360M-Instruct",
            SmolLMVariant::SmolLM1_7BInstruct => "HuggingFaceTB/SmolLM-1.7B-Instruct",
            SmolLMVariant::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M",
            SmolLMVariant::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M",
            SmolLMVariant::SmolLM2_1_7B => "HuggingFaceTB/SmolLM2-1.7B",
        }
    }

    pub fn parameters(&self) -> u64 {
        match self {
            SmolLMVariant::SmolLM135M | SmolLMVariant::SmolLM135MInstruct | SmolLMVariant::SmolLM2_135M => 135_000_000,
            SmolLMVariant::SmolLM360M | SmolLMVariant::SmolLM360MInstruct | SmolLMVariant::SmolLM2_360M => 360_000_000,
            SmolLMVariant::SmolLM1_7B | SmolLMVariant::SmolLM1_7BInstruct | SmolLMVariant::SmolLM2_1_7B => 1_700_000_000,
        }
    }

    pub fn size_category(&self) -> ModelSize {
        match self {
            SmolLMVariant::SmolLM135M | SmolLMVariant::SmolLM135MInstruct | SmolLMVariant::SmolLM2_135M => ModelSize::Small,
            SmolLMVariant::SmolLM360M | SmolLMVariant::SmolLM360MInstruct | SmolLMVariant::SmolLM2_360M => ModelSize::Medium,
            SmolLMVariant::SmolLM1_7B | SmolLMVariant::SmolLM1_7BInstruct | SmolLMVariant::SmolLM2_1_7B => ModelSize::Large,
        }
    }

    pub fn is_instruct(&self) -> bool {
        matches!(self, 
            SmolLMVariant::SmolLM135MInstruct | 
            SmolLMVariant::SmolLM360MInstruct | 
            SmolLMVariant::SmolLM1_7BInstruct
        )
    }

    pub fn is_v2(&self) -> bool {
        matches!(self, 
            SmolLMVariant::SmolLM2_135M | 
            SmolLMVariant::SmolLM2_360M | 
            SmolLMVariant::SmolLM2_1_7B
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            SmolLMVariant::SmolLM135M => "SmolLM-135M",
            SmolLMVariant::SmolLM360M => "SmolLM-360M",
            SmolLMVariant::SmolLM1_7B => "SmolLM-1.7B",
            SmolLMVariant::SmolLM135MInstruct => "SmolLM-135M-Instruct",
            SmolLMVariant::SmolLM360MInstruct => "SmolLM-360M-Instruct",
            SmolLMVariant::SmolLM1_7BInstruct => "SmolLM-1.7B-Instruct",
            SmolLMVariant::SmolLM2_135M => "SmolLM2-135M",
            SmolLMVariant::SmolLM2_360M => "SmolLM2-360M",
            SmolLMVariant::SmolLM2_1_7B => "SmolLM2-1.7B",
        }
    }
}

/// SmolLM model builder
pub struct SmolLMBuilder {
    variant: SmolLMVariant,
    config: ModelConfig,
}

impl SmolLMBuilder {
    pub fn new(variant: SmolLMVariant) -> Self {
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
            multilingual: false,
        };

        let metadata = ModelMetadata {
            name: self.variant.name().to_string(),
            family: "SmolLM".to_string(),
            parameters: self.variant.parameters(),
            size_category: self.variant.size_category(),
            huggingface_id: self.variant.huggingface_id().to_string(),
            architecture: "Transformer".to_string(),
            capabilities,
            context_length: 2048,
            vocab_size: 49152,
            training_tokens: Some(600_000_000_000), // 600B tokens
            release_date: if self.variant.is_v2() { "2024-11" } else { "2024-07" }.to_string(),
            license: "Apache-2.0".to_string(),
            description: format!(
                "SmolLM {} is a state-of-the-art small language model with {} parameters, trained on 600B tokens from the SmolLM corpus. {} optimized for efficiency and edge deployment.",
                self.variant.name(),
                format_parameters(self.variant.parameters()),
                if self.variant.is_instruct() { "Instruction-tuned and" } else { "It is" }
            ),
        };

        Ok(Model::new(metadata, self.config))
    }
}

/// Create SmolLM-135M model
pub fn smollm_135m() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM135M)
}

/// Create SmolLM-360M model
pub fn smollm_360m() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM360M)
}

/// Create SmolLM-1.7B model
pub fn smollm_1_7b() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM1_7B)
}

/// Create SmolLM-135M-Instruct model
pub fn smollm_135m_instruct() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM135MInstruct)
}

/// Create SmolLM-360M-Instruct model
pub fn smollm_360m_instruct() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM360MInstruct)
}

/// Create SmolLM-1.7B-Instruct model
pub fn smollm_1_7b_instruct() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM1_7BInstruct)
}

/// Create SmolLM2-135M model (latest version)
pub fn smollm2_135m() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM2_135M)
}

/// Create SmolLM2-360M model (latest version)
pub fn smollm2_360m() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM2_360M)
}

/// Create SmolLM2-1.7B model (latest version)
pub fn smollm2_1_7b() -> SmolLMBuilder {
    SmolLMBuilder::new(SmolLMVariant::SmolLM2_1_7B)
}

/// Get all available SmolLM variants
pub fn available_variants() -> Vec<SmolLMVariant> {
    vec![
        SmolLMVariant::SmolLM135M,
        SmolLMVariant::SmolLM360M,
        SmolLMVariant::SmolLM1_7B,
        SmolLMVariant::SmolLM135MInstruct,
        SmolLMVariant::SmolLM360MInstruct,
        SmolLMVariant::SmolLM1_7BInstruct,
        SmolLMVariant::SmolLM2_135M,
        SmolLMVariant::SmolLM2_360M,
        SmolLMVariant::SmolLM2_1_7B,
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
    fn test_smollm_variants() {
        let variants = available_variants();
        assert_eq!(variants.len(), 9);
        
        // Test parameter counts
        assert_eq!(SmolLMVariant::SmolLM135M.parameters(), 135_000_000);
        assert_eq!(SmolLMVariant::SmolLM360M.parameters(), 360_000_000);
        assert_eq!(SmolLMVariant::SmolLM1_7B.parameters(), 1_700_000_000);
        
        // Test HuggingFace IDs
        assert_eq!(SmolLMVariant::SmolLM135M.huggingface_id(), "HuggingFaceTB/SmolLM-135M");
        assert_eq!(SmolLMVariant::SmolLM360M.huggingface_id(), "HuggingFaceTB/SmolLM-360M");
        
        // Test instruct variants
        assert!(SmolLMVariant::SmolLM135MInstruct.is_instruct());
        assert!(!SmolLMVariant::SmolLM135M.is_instruct());
        
        // Test v2 variants
        assert!(SmolLMVariant::SmolLM2_135M.is_v2());
        assert!(!SmolLMVariant::SmolLM135M.is_v2());
    }

    #[test]
    fn test_smollm_builder() {
        let model = smollm_135m().build().unwrap();
        assert_eq!(model.metadata.parameters, 135_000_000);
        assert_eq!(model.metadata.huggingface_id, "HuggingFaceTB/SmolLM-135M");
        assert!(model.metadata.capabilities.text_generation);
        assert!(!model.metadata.capabilities.instruction_following);
        
        let instruct_model = smollm_135m_instruct().build().unwrap();
        assert!(instruct_model.metadata.capabilities.instruction_following);
        assert!(instruct_model.metadata.capabilities.chat);
    }

    #[test]
    fn test_format_parameters() {
        assert_eq!(format_parameters(135_000_000), "135M");
        assert_eq!(format_parameters(360_000_000), "360M");
        assert_eq!(format_parameters(1_700_000_000), "1.7B");
        assert_eq!(format_parameters(1_000), "1K");
        assert_eq!(format_parameters(500), "500");
    }
}