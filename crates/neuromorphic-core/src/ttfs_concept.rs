//! TTFS-encoded concept representation

mod spike_pattern;
mod encoding;

use serde::{Deserialize, Serialize};
use std::time::Duration;

pub use spike_pattern::{SpikePattern, SpikeEvent};
pub use encoding::{TTFSEncoder, EncodingConfig, EncodingError};

/// Convert milliseconds to Duration
pub fn ms_to_duration(ms: f32) -> Duration {
    Duration::from_micros((ms * 1000.0) as u64)
}

/// Convert Duration to milliseconds
pub fn duration_to_ms(duration: Duration) -> f32 {
    duration.as_micros() as f32 / 1000.0
}

/// Time-to-First-Spike encoded concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTFSConcept {
    /// Unique identifier
    pub id: uuid::Uuid,
    
    /// Human-readable name
    pub name: String,
    
    /// Semantic features for neural encoding
    pub semantic_features: Vec<f32>,
    
    /// Spike pattern representation
    pub spike_pattern: SpikePattern,
    
    /// Concept metadata
    pub metadata: ConceptMetadata,
    
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Metadata associated with a concept
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptMetadata {
    /// Source of the concept (e.g., "parsed", "inferred", "manual")
    pub source: String,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    
    /// Parent concept ID if this is a specialized concept
    pub parent_id: Option<uuid::Uuid>,
    
    /// Properties as key-value pairs
    pub properties: std::collections::HashMap<String, String>,
    
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl TTFSConcept {
    /// Create a new TTFS concept
    pub fn new(name: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            name: name.to_string(),
            semantic_features: Vec::new(),
            spike_pattern: SpikePattern::default(),
            metadata: ConceptMetadata::default(),
            created_at: chrono::Utc::now(),
        }
    }
    
    /// Create with semantic features
    pub fn with_features(name: &str, features: Vec<f32>) -> Self {
        let mut concept = Self::new(name);
        
        // Generate spike pattern from features before moving
        let encoder = TTFSEncoder::default();
        concept.spike_pattern = encoder.encode(&features);
        concept.semantic_features = features;
        
        concept
    }
    
    /// Add a property to the concept
    pub fn add_property(&mut self, key: String, value: String) {
        self.metadata.properties.insert(key, value);
    }
    
    /// Set parent concept
    pub fn set_parent(&mut self, parent_id: uuid::Uuid) {
        self.metadata.parent_id = Some(parent_id);
    }
    
    /// Get time-to-first-spike
    pub fn time_to_first_spike(&self) -> Option<Duration> {
        self.spike_pattern.first_spike_time()
    }
}

impl Default for ConceptMetadata {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            confidence: 1.0,
            parent_id: None,
            properties: std::collections::HashMap::new(),
            tags: Vec::new(),
        }
    }
}