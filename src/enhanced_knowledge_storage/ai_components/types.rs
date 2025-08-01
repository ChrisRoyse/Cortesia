//! AI Components Types
//! 
//! Core types and configurations for production AI/ML components.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use candle_core::{Device, Tensor};
use thiserror::Error;

/// Errors that can occur in AI components
#[derive(Error, Debug)]
pub enum AIComponentError {
    #[error("Model loading failed: {0}")]
    ModelLoad(String),
    
    #[error("Tokenization error: {0}")]
    Tokenization(String),
    
    #[error("Tensor creation error: {0}")]
    TensorCreation(String),
    
    #[error("Inference error: {0}")]
    Inference(String),
    
    #[error("Postprocessing error: {0}")]
    Postprocessing(String),
    
    #[error("Embedding generation error: {0}")]
    EmbeddingGeneration(String),
    
    #[error("Reasoning error: {0}")]
    ReasoningError(String),
    
    #[error("Caching error: {0}")]
    CachingError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

pub type AIResult<T> = Result<T, AIComponentError>;

/// Entity extracted by AI models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Entity {
    pub name: String,
    pub entity_type: EntityType,
    pub start_pos: usize,
    pub end_pos: usize,
    pub confidence: f32,
    pub context: String,
    pub attributes: HashMap<String, String>,
    pub extracted_at: u64,
}

/// Types of entities that can be extracted
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Technology,
    Concept,
    Date,
    Number,
    Measurement,
    Event,
    Method,
    Other(String),
}

impl EntityType {
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "person" | "per" => EntityType::Person,
            "organization" | "org" => EntityType::Organization,
            "location" | "loc" => EntityType::Location,
            "technology" | "tech" => EntityType::Technology,
            "concept" => EntityType::Concept,
            "date" => EntityType::Date,
            "number" | "num" => EntityType::Number,
            "measurement" => EntityType::Measurement,
            "event" => EntityType::Event,
            "method" => EntityType::Method,
            other => EntityType::Other(other.to_string()),
        }
    }
    
    pub fn to_string(&self) -> String {
        match self {
            EntityType::Person => "Person".to_string(),
            EntityType::Organization => "Organization".to_string(),
            EntityType::Location => "Location".to_string(),
            EntityType::Technology => "Technology".to_string(),
            EntityType::Concept => "Concept".to_string(),
            EntityType::Date => "Date".to_string(),
            EntityType::Number => "Number".to_string(),
            EntityType::Measurement => "Measurement".to_string(),
            EntityType::Event => "Event".to_string(),
            EntityType::Method => "Method".to_string(),
            EntityType::Other(s) => s.clone(),
        }
    }
}

/// Text span within a document
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
}

/// Semantic chunk created by AI chunker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunk {
    pub id: String,
    pub content: String,
    pub start_pos: usize,
    pub end_pos: usize,
    pub start_sentence: usize,
    pub end_sentence: usize,
    pub semantic_coherence: f32,
    pub key_concepts: Vec<String>,
    pub embedding: Vec<f32>,
    pub chunk_type: ChunkType,
}

/// Types of semantic chunks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChunkType {
    Paragraph,
    Section,
    Topic,
    Code,
    List,
    Table,
    Quote,
    Other,
}

/// Configuration for entity extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionConfig {
    pub model_name: String,
    pub min_confidence: f32,
    pub max_sequence_length: usize,
    pub batch_size: usize,
    pub labels: Vec<String>,
    pub device: String,
    pub cache_embeddings: bool,
    pub enable_context_expansion: bool,
}

impl Default for EntityExtractionConfig {
    fn default() -> Self {
        Self {
            model_name: "dbmdz/bert-large-cased-finetuned-conll03-english".to_string(),
            min_confidence: 0.85,
            max_sequence_length: 512,
            batch_size: 32,
            labels: vec![
                "O".to_string(),       // Outside
                "B-PER".to_string(),   // Begin Person
                "I-PER".to_string(),   // Inside Person
                "B-ORG".to_string(),   // Begin Organization
                "I-ORG".to_string(),   // Inside Organization
                "B-LOC".to_string(),   // Begin Location
                "I-LOC".to_string(),   // Inside Location
                "B-MISC".to_string(),  // Begin Miscellaneous
                "I-MISC".to_string(),  // Inside Miscellaneous
            ],
            device: "cpu".to_string(),
            cache_embeddings: true,
            enable_context_expansion: true,
        }
    }
}

/// Configuration for semantic chunking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunkingConfig {
    pub model_name: String,
    pub max_chunk_size: usize,
    pub min_chunk_size: usize,
    pub overlap_size: usize,
    pub similarity_threshold: f32,
    pub min_coherence: f32,
    pub preserve_sentence_boundaries: bool,
    pub enable_topic_modeling: bool,
}

impl Default for SemanticChunkingConfig {
    fn default() -> Self {
        Self {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            max_chunk_size: 1024,
            min_chunk_size: 128,
            overlap_size: 64,
            similarity_threshold: 0.7,
            min_coherence: 0.6,
            preserve_sentence_boundaries: true,
            enable_topic_modeling: false,
        }
    }
}

/// Configuration for reasoning engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub max_path_length: usize,
    pub confidence_threshold: f32,
    pub max_reasoning_time: Duration,
    pub enable_caching: bool,
    pub reasoning_strategy: ReasoningStrategy,
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            max_path_length: 5,
            confidence_threshold: 0.7,
            max_reasoning_time: Duration::from_secs(30),
            enable_caching: true,
            reasoning_strategy: ReasoningStrategy::MultiHop,
        }
    }
}

/// Reasoning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReasoningStrategy {
    DirectMatch,
    SingleHop,
    MultiHop,
    ChainOfThought,
    GraphTraversal,
}

/// Reasoning step in a chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_number: u32,
    pub hypothesis: String,
    pub evidence: Vec<String>,
    pub inference: String,
    pub confidence: f32,
    pub step_type: StepType,
}

/// Types of reasoning steps
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StepType {
    DirectEvidence,
    InferredConnection,
    TransitiveRelation,
    CausalLink,
    TemporalSequence,
    ConceptualBridge,
}

/// Result of reasoning process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResult {
    pub reasoning_chain: Vec<ReasoningStep>,
    pub confidence: f32,
    pub explanation: String,
    pub source_entities: Vec<String>,
    pub target_entities: Vec<String>,
    pub path_length: usize,
    pub processing_time: Duration,
}

/// Label for NER decoding
#[derive(Debug, Clone, PartialEq)]
pub enum Label {
    Outside,
    Begin(EntityType),
    Inside(EntityType),
}

/// Entity builder for constructing entities during extraction
#[derive(Debug)]
pub struct EntityBuilder {
    pub entity_type: EntityType,
    pub start_token: usize,
    pub end_token: usize,
    pub start_pos: usize,
    pub end_pos: usize,
}

impl EntityBuilder {
    pub fn new(entity_type: EntityType, start_token: usize, start_pos: usize) -> Self {
        Self {
            entity_type,
            start_token,
            end_token: start_token,
            start_pos,
            end_pos: start_pos,
        }
    }
    
    pub fn extend(&mut self, end_token: usize, end_pos: usize) {
        self.end_token = end_token;
        self.end_pos = end_pos;
    }
    
    pub fn build(self, original_text: &str) -> AIResult<Entity> {
        let text = if self.end_pos <= original_text.len() {
            original_text[self.start_pos..self.end_pos].to_string()
        } else {
            // Fallback if positions are invalid
            format!("Entity_{}", self.entity_type.to_string())
        };
        
        Ok(Entity {
            name: text.trim().to_string(),
            entity_type: self.entity_type,
            start_pos: self.start_pos,
            end_pos: self.end_pos,
            confidence: 0.9, // Will be updated by caller
            context: String::new(), // Will be filled by caller
            attributes: HashMap::new(),
            extracted_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        })
    }
}

/// Label decoder for NER model outputs
pub struct LabelDecoder {
    labels: Vec<String>,
}

impl LabelDecoder {
    pub fn new(labels: Vec<String>) -> Self {
        Self { labels }
    }
    
    pub fn decode(&self, label_id: usize) -> AIResult<Label> {
        if label_id >= self.labels.len() {
            return Ok(Label::Outside);
        }
        
        let label = &self.labels[label_id];
        
        if label == "O" {
            Ok(Label::Outside)
        } else if label.starts_with("B-") {
            let entity_type = EntityType::from_string(&label[2..]);
            Ok(Label::Begin(entity_type))
        } else if label.starts_with("I-") {
            let entity_type = EntityType::from_string(&label[2..]);
            Ok(Label::Inside(entity_type))
        } else {
            Ok(Label::Outside)
        }
    }
}

/// Confidence calculator for model outputs
pub struct ConfidenceCalculator {
    temperature: f32,
}

impl ConfidenceCalculator {
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
        }
    }
    
    pub fn calculate_confidence(&self, logits: &[f32]) -> f32 {
        // Apply softmax with temperature
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.iter()
            .map(|&x| ((x - max_logit) / self.temperature).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        
        // Return max probability as confidence
        exp_logits.iter().fold(0.0, |a, &b| a.max(b / sum_exp))
    }
}

impl Default for ConfidenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached result type for caching layer
#[derive(Debug, Clone)]
pub enum CachedResult {
    Entities(Vec<Entity>),
    Chunks(Vec<SemanticChunk>),
    Embeddings(Vec<f32>),
    Reasoning(ReasoningResult),
}

/// Performance metrics for AI components
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AIPerformanceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_latency: Duration,
    pub cache_hit_rate: f32,
    pub model_load_time: Duration,
    pub memory_usage: u64,
    pub throughput: f32, // requests per second
}

impl AIPerformanceMetrics {
    pub fn success_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.successful_requests as f32 / self.total_requests as f32) * 100.0
        }
    }
    
    pub fn error_rate(&self) -> f32 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.failed_requests as f32 / self.total_requests as f32) * 100.0
        }
    }
}

/// Device configuration for model execution
#[derive(Debug, Clone)]
pub enum ModelDevice {
    Cpu,
    Cuda(usize), // GPU index
    Metal,       // Apple Silicon
    Auto,        // Automatically detect best device
}

impl Default for ModelDevice {
    fn default() -> Self {
        ModelDevice::Auto
    }
}

impl ModelDevice {
    pub fn to_candle_device(&self) -> AIResult<Device> {
        match self {
            ModelDevice::Cpu => Ok(Device::Cpu),
            ModelDevice::Cuda(idx) => {
                #[cfg(feature = "cuda")]
                {
                    Device::new_cuda(*idx).map_err(|e| AIComponentError::ConfigError(format!("CUDA device error: {e}")))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    log::warn!("CUDA requested but not compiled with CUDA support, falling back to CPU");
                    Ok(Device::Cpu)
                }
            },
            ModelDevice::Metal => {
                #[cfg(feature = "metal")]
                {
                    Device::new_metal(0).map_err(|e| AIComponentError::ConfigError(format!("Metal device error: {e}")))
                }
                #[cfg(not(feature = "metal"))]
                {
                    log::warn!("Metal requested but not compiled with Metal support, falling back to CPU");
                    Ok(Device::Cpu)
                }
            },
            ModelDevice::Auto => {
                // Try CUDA first, then Metal, then CPU
                #[cfg(feature = "cuda")]
                if let Ok(device) = Device::new_cuda(0) {
                    return Ok(device);
                }
                
                #[cfg(feature = "metal")]
                if let Ok(device) = Device::new_metal(0) {
                    return Ok(device);
                }
                
                Ok(Device::Cpu)
            }
        }
    }
}

/// Utility functions for working with cosine similarity
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

/// Average multiple embeddings into a single embedding
pub fn average_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    
    let dim = embeddings[0].len();
    let mut result = vec![0.0; dim];
    
    for embedding in embeddings {
        for (i, &val) in embedding.iter().enumerate() {
            if i < result.len() {
                result[i] += val;
            }
        }
    }
    
    let count = embeddings.len() as f32;
    for val in &mut result {
        *val /= count;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entity_type_conversion() {
        assert_eq!(EntityType::from_string("person"), EntityType::Person);
        assert_eq!(EntityType::from_string("Organization"), EntityType::Organization);
        assert_eq!(EntityType::from_string("unknown"), EntityType::Other("unknown".to_string()));
        
        assert_eq!(EntityType::Person.to_string(), "Person");
        assert_eq!(EntityType::Other("custom".to_string()).to_string(), "custom");
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
        
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_average_embeddings() {
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let avg = average_embeddings(&embeddings);
        assert_eq!(avg, vec![2.5, 3.5, 4.5]);
    }
    
    #[test]
    fn test_confidence_calculator() {
        let calc = ConfidenceCalculator::new();
        let logits = vec![2.0, 1.0, 0.5];
        let confidence = calc.calculate_confidence(&logits);
        assert!(confidence > 0.5 && confidence <= 1.0);
    }
    
    #[test]
    fn test_label_decoder() {
        let labels = vec![
            "O".to_string(),
            "B-PER".to_string(), 
            "I-PER".to_string(),
        ];
        let decoder = LabelDecoder::new(labels);
        
        assert_eq!(decoder.decode(0).unwrap(), Label::Outside);
        assert_eq!(decoder.decode(1).unwrap(), Label::Begin(EntityType::Person));
        assert_eq!(decoder.decode(2).unwrap(), Label::Inside(EntityType::Person));
    }
}