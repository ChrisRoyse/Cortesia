//! Core Types for Enhanced Knowledge Storage System
//! 
//! Defines shared types, enums, and structures used throughout the enhanced
//! knowledge storage system.

use std::time::Duration;
use serde::{Serialize, Deserialize};

/// Task complexity levels for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Low,    // Simple text processing, basic entity extraction
    Medium, // Moderate complexity analysis, relationship extraction
    High,   // Complex reasoning, multi-step processing
}

/// Configuration for model resource management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResourceConfig {
    /// Maximum total memory usage across all loaded models (bytes)
    pub max_memory_usage: u64,
    /// Maximum number of models that can be loaded simultaneously
    pub max_concurrent_models: usize,
    /// Time after which idle models are eligible for eviction
    pub idle_timeout: Duration,
    /// Minimum memory threshold below which no eviction occurs
    pub min_memory_threshold: u64,
}

impl Default for ModelResourceConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 2_000_000_000, // 2GB default
            max_concurrent_models: 3,
            idle_timeout: Duration::from_secs(300), // 5 minutes
            min_memory_threshold: 100_000_000, // 100MB minimum
        }
    }
}

/// Processing task with complexity and content
#[derive(Debug, Clone)]
pub struct ProcessingTask {
    pub id: String,
    pub complexity: ComplexityLevel,
    pub content: String,
    pub created_at: std::time::Instant,
}

impl ProcessingTask {
    pub fn new(complexity: ComplexityLevel, content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            complexity,
            content: content.to_string(),
            created_at: std::time::Instant::now(),
        }
    }
    
    pub fn with_id(id: String, complexity: ComplexityLevel, content: &str) -> Self {
        Self {
            id,
            complexity,
            content: content.to_string(),
            created_at: std::time::Instant::now(),
        }
    }
}

/// Result of processing a task
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub task_id: String,
    pub processing_time: Duration,
    pub quality_score: f32,
    pub model_used: String,
    pub success: bool,
    pub output: String,
    pub metadata: ProcessingMetadata,
}

/// Additional metadata about processing
#[derive(Debug, Clone, Default)]
pub struct ProcessingMetadata {
    pub memory_used: u64,
    pub cache_hit: bool,
    pub model_load_time: Option<Duration>,
    pub inference_time: Duration,
}

/// Metadata about a model's capabilities and requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub parameters: u64,
    pub memory_footprint: u64,
    pub complexity_level: ComplexityLevel,
    pub model_type: String,
    pub huggingface_id: String,
    pub supported_tasks: Vec<String>,
}

/// Task complexity assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskComplexity {
    Low,
    Medium,
    High,
}

impl From<ComplexityLevel> for TaskComplexity {
    fn from(level: ComplexityLevel) -> Self {
        match level {
            ComplexityLevel::Low => TaskComplexity::Low,
            ComplexityLevel::Medium => TaskComplexity::Medium,
            ComplexityLevel::High => TaskComplexity::High,
        }
    }
}

/// Errors that can occur in the enhanced knowledge storage system
#[derive(Debug, thiserror::Error)]
pub enum EnhancedStorageError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    
    #[error("Insufficient resources: {0}")]
    InsufficientResources(String),
    
    #[error("Model loading failed: {0}")]
    ModelLoadingFailed(String),
    
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, EnhancedStorageError>;