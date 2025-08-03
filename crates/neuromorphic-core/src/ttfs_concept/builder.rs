//! Builder pattern for constructing TTFS concepts

use super::{TTFSConcept, ConceptMetadata};
use super::encoding::TTFSEncoder;
use uuid::Uuid;
use std::collections::HashMap;

/// Builder for creating TTFS concepts with validation
pub struct ConceptBuilder {
    name: Option<String>,
    semantic_features: Vec<f32>,
    parent_id: Option<Uuid>,
    properties: HashMap<String, String>,
    tags: Vec<String>,
    source: String,
    confidence: f32,
    encoder: TTFSEncoder,
}

impl Default for ConceptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConceptBuilder {
    /// Create a new concept builder
    pub fn new() -> Self {
        Self {
            name: None,
            semantic_features: Vec::new(),
            parent_id: None,
            properties: HashMap::new(),
            tags: Vec::new(),
            source: "builder".to_string(),
            confidence: 1.0,
            encoder: TTFSEncoder::default(),
        }
    }
    
    /// Create builder with custom encoder
    pub fn with_encoder(encoder: TTFSEncoder) -> Self {
        Self {
            encoder,
            ..Self::new()
        }
    }
    
    /// Set concept name (required)
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    /// Set semantic features directly
    pub fn features(mut self, features: Vec<f32>) -> Self {
        self.semantic_features = features;
        self
    }
    
    /// Extract features from text (placeholder - would use real NLP)
    pub fn features_from_text(mut self, text: &str) -> Self {
        // Simple feature extraction based on text characteristics
        let features = Self::extract_text_features(text);
        self.semantic_features = features;
        self
    }
    
    /// Set parent concept
    pub fn parent(mut self, parent_id: Uuid) -> Self {
        self.parent_id = Some(parent_id);
        self
    }
    
    /// Add a property
    pub fn property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
    
    /// Add multiple properties
    pub fn properties(mut self, props: HashMap<String, String>) -> Self {
        self.properties.extend(props);
        self
    }
    
    /// Add a tag
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
    
    /// Add multiple tags
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags.extend(tags);
        self
    }
    
    /// Set source
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }
    
    /// Set confidence
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
    
    /// Build the concept
    pub fn build(self) -> Result<TTFSConcept, BuilderError> {
        // Validate required fields
        let name = self.name.ok_or(BuilderError::MissingName)?;
        
        // Generate features if not provided
        let features = if self.semantic_features.is_empty() {
            if name.is_empty() {
                return Err(BuilderError::EmptyFeatures);
            }
            Self::extract_text_features(&name)
        } else {
            self.semantic_features
        };
        
        // Validate features
        Self::validate_features(&features)?;
        
        // Encode to spike pattern
        let spike_pattern = self.encoder.encode(&features);
        
        // Create metadata
        let metadata = ConceptMetadata {
            source: self.source,
            confidence: self.confidence,
            parent_id: self.parent_id,
            properties: self.properties,
            tags: self.tags,
        };
        
        // Build concept
        let concept = TTFSConcept {
            id: Uuid::new_v4(),
            name,
            semantic_features: features,
            spike_pattern,
            metadata,
            created_at: chrono::Utc::now(),
        };
        
        Ok(concept)
    }
    
    /// Extract features from text using NLP-inspired techniques
    /// 
    /// This method generates a 128-dimensional feature vector from text using:
    /// - Statistical features (length, capitalization, word count)
    /// - Character n-gram features (bigram and trigram analysis)
    /// - Semantic hashing for word-level features
    /// - Positional encoding for word order sensitivity
    fn extract_text_features(text: &str) -> Vec<f32> {
        let mut features = vec![0.0; 128];
        
        // Statistical features (0-9)
        let text_len = text.len() as f32;
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len() as f32;
        
        // Normalize text length feature
        features[0] = (text_len / 100.0).min(1.0).max(0.0);
        
        // Capitalization ratio
        features[1] = if text_len > 0.0 {
            text.chars().filter(|c| c.is_uppercase()).count() as f32 / text_len
        } else {
            0.0
        };
        
        // Word count feature
        features[2] = (word_count / 50.0).min(1.0).max(0.0);
        
        // Average word length
        features[3] = if word_count > 0.0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count / 10.0
        } else {
            0.0
        }.min(1.0);
        
        // Punctuation density
        features[4] = text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32 / text_len.max(1.0);
        
        // Digit presence
        features[5] = text.chars().filter(|c| c.is_numeric()).count() as f32 / text_len.max(1.0);
        
        // Whitespace ratio
        features[6] = text.chars().filter(|c| c.is_whitespace()).count() as f32 / text_len.max(1.0);
        
        // Unique word ratio
        let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
        features[7] = if word_count > 0.0 {
            unique_words.len() as f32 / word_count
        } else {
            0.0
        };
        
        // Sentence complexity (approximated by commas and semicolons)
        features[8] = text.chars().filter(|&c| c == ',' || c == ';').count() as f32 / word_count.max(1.0);
        
        // Question indicator
        features[9] = if text.contains('?') { 1.0 } else { 0.0 };
        
        // Character n-gram features (10-39)
        use std::collections::HashMap;
        
        // Bigram frequencies
        let text_lower = text.to_lowercase();
        let chars: Vec<char> = text_lower.chars().collect();
        let mut bigram_counts = HashMap::new();
        
        for window in chars.windows(2) {
            if let [a, b] = window {
                let bigram = format!("{}{}", a, b);
                *bigram_counts.entry(bigram).or_insert(0) += 1;
            }
        }
        
        // Top common English bigrams
        let common_bigrams = ["th", "he", "in", "er", "an", "ed", "nd", "to", "en", "es",
                              "of", "te", "at", "on", "ar", "ou", "it", "as", "is", "re"];
        
        for (i, bigram) in common_bigrams.iter().enumerate() {
            let count = bigram_counts.get(*bigram).unwrap_or(&0);
            features[10 + i] = (*count as f32 / chars.len().max(1) as f32).min(1.0);
        }
        
        // Trigram features (30-49)
        let mut trigram_counts = HashMap::new();
        for window in chars.windows(3) {
            if let [a, b, c] = window {
                let trigram = format!("{}{}{}", a, b, c);
                *trigram_counts.entry(trigram).or_insert(0) += 1;
            }
        }
        
        let common_trigrams = ["the", "and", "ing", "ion", "tio", "ent", "ati", "for", "her", "ter",
                               "hat", "tha", "ere", "ate", "his", "con", "res", "ver", "all", "ons"];
        
        for (i, trigram) in common_trigrams.iter().enumerate() {
            let count = trigram_counts.get(*trigram).unwrap_or(&0);
            features[30 + i] = (*count as f32 / chars.len().max(1) as f32).min(1.0);
        }
        
        // Word-level semantic hashing (50-89)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        for (word_idx, word) in words.iter().take(40).enumerate() {
            let mut hasher = DefaultHasher::new();
            word.to_lowercase().hash(&mut hasher);
            let hash = hasher.finish();
            
            // Create word embedding-like feature
            let feature_value = ((hash as f32 / u64::MAX as f32) * 2.0 - 1.0) * 0.5 + 0.5;
            features[50 + word_idx] = feature_value;
            
            // Add positional encoding
            let position_weight = 1.0 - (word_idx as f32 / 40.0);
            features[50 + word_idx] *= position_weight;
        }
        
        // Global text hash features (90-127)
        let mut text_hasher = DefaultHasher::new();
        text.hash(&mut text_hasher);
        let text_hash = text_hasher.finish();
        
        for i in 90..128 {
            // Create diverse hash-based features
            let bit_position = (i - 90) % 64;
            let hash_segment = if i < 109 {
                text_hash
            } else {
                // Create variation by hashing the hash
                let mut variant_hasher = DefaultHasher::new();
                (text_hash, i).hash(&mut variant_hasher);
                variant_hasher.finish()
            };
            
            features[i] = ((hash_segment >> bit_position) & 1) as f32 * 0.3 + 
                         ((hash_segment >> ((bit_position + 16) % 64)) & 1) as f32 * 0.2 +
                         0.3;
        }
        
        features
    }
    
    /// Validate feature vector for correctness and constraints
    /// 
    /// Ensures:
    /// - Feature vector is not empty
    /// - Number of features doesn't exceed 1024 (memory constraint)
    /// - All values are finite (no NaN or Infinity)
    fn validate_features(features: &[f32]) -> Result<(), BuilderError> {
        if features.is_empty() {
            return Err(BuilderError::EmptyFeatures);
        }
        
        if features.len() > 1024 {
            return Err(BuilderError::TooManyFeatures(features.len()));
        }
        
        if features.iter().any(|&f| !f.is_finite()) {
            return Err(BuilderError::InvalidFeatureValue);
        }
        
        Ok(())
    }
}

/// Batch builder for creating multiple concepts efficiently
/// 
/// This builder allows creation of multiple concepts with a shared encoder,
/// reducing overhead and improving performance for bulk operations.
pub struct BatchConceptBuilder {
    encoder: TTFSEncoder,
    concepts: Vec<ConceptBuilder>,
}

impl Default for BatchConceptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchConceptBuilder {
    /// Create new batch builder
    pub fn new() -> Self {
        Self {
            encoder: TTFSEncoder::default(),
            concepts: Vec::new(),
        }
    }
    
    /// Add a concept to the batch
    pub fn add_concept<F>(mut self, configure: F) -> Self 
    where
        F: FnOnce(ConceptBuilder) -> ConceptBuilder,
    {
        let builder = ConceptBuilder::with_encoder(self.encoder.clone());
        let configured = configure(builder);
        self.concepts.push(configured);
        self
    }
    
    /// Build all concepts
    pub fn build_all(self) -> Result<Vec<TTFSConcept>, BatchBuilderError> {
        let mut results = Vec::new();
        let mut errors = Vec::new();
        
        for (idx, builder) in self.concepts.into_iter().enumerate() {
            match builder.build() {
                Ok(concept) => results.push(concept),
                Err(e) => errors.push((idx, e)),
            }
        }
        
        if errors.is_empty() {
            Ok(results)
        } else {
            Err(BatchBuilderError { errors })
        }
    }
}

/// Errors that can occur during building
#[derive(Debug, thiserror::Error)]
pub enum BuilderError {
    #[error("Concept name is required")]
    MissingName,
    
    #[error("Feature vector cannot be empty")]
    EmptyFeatures,
    
    #[error("Too many features: {0} (max 1024)")]
    TooManyFeatures(usize),
    
    #[error("Invalid feature value (NaN or infinity)")]
    InvalidFeatureValue,
}

/// Errors from batch building
#[derive(Debug, thiserror::Error)]
#[error("Batch building failed with {} errors", errors.len())]
pub struct BatchBuilderError {
    pub errors: Vec<(usize, BuilderError)>,
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_builder() {
        let concept = ConceptBuilder::new()
            .name("dog")
            .features(vec![0.8; 64])
            .property("species", "Canis familiaris")
            .tag("animal")
            .tag("pet")
            .confidence(0.95)
            .build()
            .unwrap();
        
        assert_eq!(concept.name, "dog");
        assert_eq!(concept.semantic_features.len(), 64);
        assert_eq!(concept.metadata.properties["species"], "Canis familiaris");
        assert_eq!(concept.metadata.tags.len(), 2);
        assert_eq!(concept.metadata.confidence, 0.95);
    }
    
    #[test]
    fn test_feature_extraction() {
        let concept = ConceptBuilder::new()
            .name("artificial intelligence")
            .features_from_text("AI systems that learn from data")
            .build()
            .unwrap();
        
        assert!(!concept.semantic_features.is_empty());
        assert!(concept.spike_pattern.events.len() > 0);
    }
    
    #[test]
    fn test_validation() {
        // Missing name
        let result = ConceptBuilder::new()
            .features(vec![0.5; 10])
            .build();
        assert!(matches!(result, Err(BuilderError::MissingName)));
        
        // Empty features with no name to generate from
        let result = ConceptBuilder::new()
            .name("")
            .features(vec![])
            .build();
        assert!(matches!(result, Err(BuilderError::EmptyFeatures)));
        
        // Invalid feature values
        let result = ConceptBuilder::new()
            .name("test")
            .features(vec![f32::NAN])
            .build();
        assert!(matches!(result, Err(BuilderError::InvalidFeatureValue)));
    }
    
    #[test]
    fn test_parent_relationship() {
        let parent_id = Uuid::new_v4();
        
        let concept = ConceptBuilder::new()
            .name("puppy")
            .parent(parent_id)
            .features(vec![0.7; 32])
            .build()
            .unwrap();
        
        assert_eq!(concept.metadata.parent_id, Some(parent_id));
    }
    
    #[test]
    fn test_batch_builder() {
        let batch = BatchConceptBuilder::new()
            .add_concept(|b| b.name("cat").features(vec![0.6; 32]))
            .add_concept(|b| b.name("dog").features(vec![0.7; 32]))
            .add_concept(|b| b.name("bird").features(vec![0.5; 32]));
        
        let concepts = batch.build_all().unwrap();
        assert_eq!(concepts.len(), 3);
        assert_eq!(concepts[0].name, "cat");
        assert_eq!(concepts[1].name, "dog");
        assert_eq!(concepts[2].name, "bird");
    }
    
    #[test]
    fn test_fluent_api_comprehensive() {
        // Test the fluent API ergonomics
        let parent_id = Uuid::new_v4();
        
        let concept = ConceptBuilder::new()
            .name("Golden Retriever")
            .features_from_text("A friendly large dog breed known for its golden coat")
            .parent(parent_id)
            .property("breed", "Golden Retriever")
            .property("size", "large")
            .property("temperament", "friendly")
            .tag("animal")
            .tag("pet")
            .tag("dog")
            .tag("large-breed")
            .source("manual_entry")
            .confidence(0.98)
            .build()
            .expect("Should build successfully");
        
        // Verify all properties were set correctly
        assert_eq!(concept.name, "Golden Retriever");
        assert_eq!(concept.metadata.parent_id, Some(parent_id));
        assert_eq!(concept.metadata.properties["breed"], "Golden Retriever");
        assert_eq!(concept.metadata.properties["size"], "large");
        assert_eq!(concept.metadata.properties["temperament"], "friendly");
        assert_eq!(concept.metadata.tags.len(), 4);
        assert!(concept.metadata.tags.contains(&"animal".to_string()));
        assert!(concept.metadata.tags.contains(&"pet".to_string()));
        assert!(concept.metadata.tags.contains(&"dog".to_string()));
        assert!(concept.metadata.tags.contains(&"large-breed".to_string()));
        assert_eq!(concept.metadata.source, "manual_entry");
        assert_eq!(concept.metadata.confidence, 0.98);
        assert!(!concept.semantic_features.is_empty());
        assert!(!concept.spike_pattern.events.is_empty());
    }
    
    #[test] 
    fn test_batch_builder_efficiency() {
        use std::time::Instant;
        
        let start = Instant::now();
        
        // Create a batch of 100 concepts efficiently
        let mut batch = BatchConceptBuilder::new();
        for i in 0..100 {
            batch = batch.add_concept(|b| {
                b.name(format!("concept_{}", i))
                 .features(vec![0.5; 64])
                 .tag("test")
                 .confidence(0.9)
            });
        }
        
        let concepts = batch.build_all().unwrap();
        let duration = start.elapsed();
        
        assert_eq!(concepts.len(), 100);
        assert!(duration.as_millis() < 1000, "Batch building should be fast");
        
        // Verify all concepts were built correctly
        for (i, concept) in concepts.iter().enumerate() {
            assert_eq!(concept.name, format!("concept_{}", i));
            assert_eq!(concept.semantic_features.len(), 64);
            assert_eq!(concept.metadata.tags, vec!["test"]);
            assert_eq!(concept.metadata.confidence, 0.9);
        }
    }
    
    #[test]
    fn test_enhanced_feature_extraction() {
        // Test the enhanced NLP-inspired feature extraction
        let text1 = "The quick brown fox jumps over the lazy dog.";
        let text2 = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG!";
        let text3 = "Is this a question? Yes, it contains punctuation!";
        let text4 = "Numbers: 123, 456.78 and special chars: @#$%";
        
        let features1 = ConceptBuilder::extract_text_features(text1);
        let features2 = ConceptBuilder::extract_text_features(text2);
        let features3 = ConceptBuilder::extract_text_features(text3);
        let features4 = ConceptBuilder::extract_text_features(text4);
        
        // Verify feature dimensions
        assert_eq!(features1.len(), 128);
        assert_eq!(features2.len(), 128);
        assert_eq!(features3.len(), 128);
        assert_eq!(features4.len(), 128);
        
        // Test capitalization feature (index 1)
        assert!(features2[1] > features1[1], "All caps should have higher capitalization ratio");
        
        // Test question indicator (index 9)
        assert_eq!(features3[9], 1.0, "Question text should have question indicator");
        assert_eq!(features1[9], 0.0, "Non-question text should not have question indicator");
        
        // Test digit presence (index 5)
        assert!(features4[5] > features1[5], "Text with numbers should have higher digit presence");
        
        // Test that common bigrams are detected
        // "th" is a common bigram in "the"
        assert!(features1[10] > 0.0, "Should detect 'th' bigram");
        
        // Test that features are normalized (all between 0 and 1)
        for feature in &features1 {
            assert!(*feature >= 0.0 && *feature <= 1.0, "Features should be normalized");
        }
        
        // Test that different texts produce different features
        let similarity = features1.iter()
            .zip(features2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / 128.0;
        
        assert!(similarity > 0.01, "Different texts should produce different features");
    }
    
    #[test]
    fn test_validation_comprehensive() {
        // Test all validation cases
        
        // 1. Missing name
        let result = ConceptBuilder::new().build();
        assert!(matches!(result, Err(BuilderError::MissingName)));
        
        // 2. Empty name with empty features 
        let result = ConceptBuilder::new()
            .name("")
            .build();
        assert!(matches!(result, Err(BuilderError::EmptyFeatures)));
        
        // 3. Too many features
        let result = ConceptBuilder::new()
            .name("test")
            .features(vec![0.5; 2048]) // Over 1024 limit
            .build();
        assert!(matches!(result, Err(BuilderError::TooManyFeatures(2048))));
        
        // 4. Invalid feature values - NaN
        let result = ConceptBuilder::new()
            .name("test") 
            .features(vec![f32::NAN])
            .build();
        assert!(matches!(result, Err(BuilderError::InvalidFeatureValue)));
        
        // 5. Invalid feature values - Infinity
        let result = ConceptBuilder::new()
            .name("test")
            .features(vec![f32::INFINITY])
            .build();
        assert!(matches!(result, Err(BuilderError::InvalidFeatureValue)));
        
        // 6. Valid case should succeed
        let result = ConceptBuilder::new()
            .name("valid_concept")
            .features(vec![0.5; 100])
            .build();
        assert!(result.is_ok());
    }
}