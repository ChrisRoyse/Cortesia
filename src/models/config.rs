//! Model configuration and settings

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Model configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub temperature: f32,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
    pub max_tokens: u32,
    pub seed: Option<u64>,
    pub stop_tokens: Vec<String>,
    pub custom_parameters: HashMap<String, serde_json::Value>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.95),
            repetition_penalty: 1.1,
            max_tokens: 1024,
            seed: None,
            stop_tokens: vec!["</s>".to_string(), "<|endoftext|>".to_string()],
            custom_parameters: HashMap::new(),
        }
    }
}

impl ModelConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn add_stop_token(mut self, token: String) -> Self {
        self.stop_tokens.push(token);
        self
    }

    pub fn set_custom_parameter<T: Into<serde_json::Value>>(mut self, key: String, value: T) -> Self {
        self.custom_parameters.insert(key, value.into());
        self
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationConfig {
    None,
    Int8,
    Int4,
    F16,
    BFloat16,
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(u32), // GPU device index
    Metal,
}

/// Model loading configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadingConfig {
    pub device: DeviceConfig,
    pub quantization: QuantizationConfig,
    pub batch_size: u32,
    pub cache_dir: Option<String>,
    pub trust_remote_code: bool,
    pub revision: Option<String>,
}

impl Default for LoadingConfig {
    fn default() -> Self {
        Self {
            device: DeviceConfig::Cpu,
            quantization: QuantizationConfig::None,
            batch_size: 1,
            cache_dir: None,
            trust_remote_code: false,
            revision: None,
        }
    }
}