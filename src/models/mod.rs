//! State-of-the-art small language models (1M-500M parameters)
//! 
//! This module provides a collection of cutting-edge small language models
//! optimized for efficiency and performance in resource-constrained environments.

pub mod smollm;
pub mod tinyllama;
pub mod openelm;
pub mod minilm;
pub mod config;
pub mod loader;
pub mod registry;
pub mod utils;

pub use config::*;
pub use loader::*;
pub use registry::*;
pub use utils::*;

use crate::error::{GraphError, Result};
use serde::{Serialize, Deserialize};

/// Model size categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelSize {
    Tiny,      // 1M-50M parameters
    Small,     // 50M-200M parameters
    Medium,    // 200M-500M parameters
    Large,     // 500M+ parameters
}

/// Model capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub text_generation: bool,
    pub instruction_following: bool,
    pub chat: bool,
    pub code_generation: bool,
    pub reasoning: bool,
    pub multilingual: bool,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub family: String,
    pub parameters: u64,
    pub size_category: ModelSize,
    pub huggingface_id: String,
    pub architecture: String,
    pub capabilities: ModelCapabilities,
    pub context_length: u32,
    pub vocab_size: u32,
    pub training_tokens: Option<u64>,
    pub release_date: String,
    pub license: String,
    pub description: String,
}

/// Model instance
#[derive(Debug)]
pub struct Model {
    pub metadata: ModelMetadata,
    pub loaded: bool,
    pub config: ModelConfig,
}

impl Model {
    pub fn new(metadata: ModelMetadata, config: ModelConfig) -> Self {
        Self {
            metadata,
            loaded: false,
            config,
        }
    }

    pub fn load(&mut self) -> Result<()> {
        // Implementation for loading the model
        self.loaded = true;
        Ok(())
    }

    pub fn unload(&mut self) -> Result<()> {
        // Implementation for unloading the model
        self.loaded = false;
        Ok(())
    }

    pub fn generate_text(&self, prompt: &str, _max_tokens: Option<u32>) -> Result<String> {
        if !self.loaded {
            return Err(GraphError::InvalidState("Model not loaded".to_string()));
        }
        
        // Placeholder implementation
        Ok(format!("Generated response for: {}", prompt))
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn parameter_count(&self) -> u64 {
        self.metadata.parameters
    }

    pub fn size_category(&self) -> &ModelSize {
        &self.metadata.size_category
    }
}