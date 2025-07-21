use crate::core::types::{EntityData, EntityKey};
use crate::error::Result;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Semantic Summary of an entity that preserves essential information for LLM understanding
/// Target: ~150-200 bytes with rich semantic content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSummary {
    /// Core entity type and classification
    pub entity_type: EntityType,
    
    /// Key semantic features extracted from properties
    pub key_features: Vec<KeyFeature>,
    
    /// Compressed embedding that preserves semantic neighborhoods
    pub semantic_embedding: CompactEmbedding,
    
    /// Essential relationships and context
    pub context_hints: Vec<ContextHint>,
    
    /// Metadata for reconstruction guidance
    pub reconstruction_metadata: ReconstructionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityType {
    /// Primary type ID
    pub type_id: u16,
    /// Confidence score for this classification
    pub confidence: f32,
    /// Secondary type if applicable
    pub secondary_type: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyFeature {
    /// Feature name/identifier (compressed)
    pub feature_id: u16,
    /// Feature value (compressed representation)
    pub value: FeatureValue,
    /// Importance score (0.0-1.0)
    pub importance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureValue {
    /// Categorical value
    Category(u16),
    /// Numerical value with optional range
    Numeric { value: f32, range_hint: Option<(f32, f32)> },
    /// Text summary (heavily compressed but meaningful)
    TextSummary { 
        /// Key terms (dictionary indices)
        key_terms: Vec<u16>,
        /// Sentiment/tone indicator
        sentiment: i8,
        /// Text length hint
        length_category: u8,
    },
    /// Temporal information
    Temporal { timestamp: u32, duration_hint: Option<u32> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactEmbedding {
    /// Compressed embedding using learned quantization
    /// Preserves semantic neighborhoods while reducing size
    pub quantized_values: Vec<u8>,
    /// Scaling factors for reconstruction
    pub scale_factors: Vec<f32>,
    /// Dimension reduction mapping
    pub dimension_map: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextHint {
    /// Type of context relationship
    pub context_type: ContextType,
    /// Related entity hint or cluster ID
    pub related_id: u32,
    /// Relationship strength
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextType {
    /// Similar entities cluster
    SemanticCluster,
    /// Temporal relationship
    TemporalSequence,
    /// Causal relationship
    CausalLink,
    /// Spatial/location relationship
    SpatialProximity,
    /// Hierarchical relationship
    Hierarchy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconstructionMetadata {
    /// Original data size for reference
    pub original_size: u32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Quality score of the summary
    pub quality_score: f32,
    /// Hash of original properties for verification
    pub content_hash: u32,
}

/// Semantic Summarizer that creates rich, LLM-friendly summaries
pub struct SemanticSummarizer {
    /// Dictionary for text compression
    term_dictionary: HashMap<String, u16>,
    /// Feature extractors for different entity types
    feature_extractors: HashMap<u16, FeatureExtractor>,
    /// Embedding compressor
    embedding_compressor: EmbeddingCompressor,
    /// Next dictionary ID
    next_term_id: u16,
}

#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Important keywords for this entity type
    pub key_terms: Vec<String>,
    /// Numerical features to extract
    pub numeric_patterns: Vec<String>,
    /// Context patterns to look for
    pub context_patterns: Vec<String>,
}

pub struct EmbeddingCompressor {
    /// Learned principal components for dimension reduction
    principal_components: Vec<Vec<f32>>,
    /// Quantization centroids for each reduced dimension
    quantization_centroids: Vec<Vec<f32>>,
    /// Target dimensions after compression
    target_dimensions: usize,
}

impl SemanticSummarizer {
    pub fn new() -> Self {
        Self {
            term_dictionary: HashMap::new(),
            feature_extractors: Self::create_default_extractors(),
            embedding_compressor: EmbeddingCompressor::new(32), // Reduce to 32D
            next_term_id: 1,
        }
    }

    /// Create a semantic summary that preserves essential information for LLM understanding
    pub fn create_summary(&mut self, entity_data: &EntityData, entity_key: EntityKey) -> Result<SemanticSummary> {
        // 1. Extract entity type and classification
        let entity_type = self.classify_entity(entity_data)?;
        
        // 2. Extract key semantic features from properties
        let key_features = self.extract_key_features(entity_data)?;
        
        // 3. Compress embedding while preserving semantic meaning
        let semantic_embedding = self.compress_embedding(&entity_data.embedding)?;
        
        // 4. Generate context hints
        let context_hints = self.generate_context_hints(entity_data, entity_key)?;
        
        // 5. Create reconstruction metadata
        let original_size = self.estimate_original_size(entity_data);
        let reconstruction_metadata = ReconstructionMetadata {
            original_size,
            compression_ratio: 0.0, // Will be calculated later
            quality_score: 0.85, // Placeholder - would be learned
            content_hash: self.calculate_content_hash(entity_data),
        };

        let summary = SemanticSummary {
            entity_type,
            key_features,
            semantic_embedding,
            context_hints,
            reconstruction_metadata,
        };

        Ok(summary)
    }

    /// Generate an LLM-friendly text representation of the summary
    pub fn to_llm_text(&self, summary: &SemanticSummary) -> String {
        let mut text = String::new();
        
        // Entity classification
        text.push_str(&format!("Entity Type: {} (confidence: {:.2})\n", 
                              summary.entity_type.type_id, 
                              summary.entity_type.confidence));
        
        // Key features
        text.push_str("Key Features:\n");
        for feature in &summary.key_features {
            let value_desc = match &feature.value {
                FeatureValue::Category(cat) => format!("Category {}", cat),
                FeatureValue::Numeric { value, range_hint } => {
                    if let Some((min, max)) = range_hint {
                        format!("Numeric: {:.3} (range: {:.2}-{:.2})", value, min, max)
                    } else {
                        format!("Numeric: {:.3}", value)
                    }
                },
                FeatureValue::TextSummary { key_terms, sentiment, length_category } => {
                    format!("Text: {} key terms, sentiment: {}, length: {}", 
                           key_terms.len(), sentiment, length_category)
                },
                FeatureValue::Temporal { timestamp, duration_hint } => {
                    format!("Time: {}, duration: {:?}", timestamp, duration_hint)
                },
            };
            text.push_str(&format!("  - Feature {}: {} (importance: {:.2})\n", 
                                  feature.feature_id, value_desc, feature.importance));
        }
        
        // Semantic embedding info
        text.push_str(&format!("Semantic Embedding: {} dimensions, {} scale factors\n",
                              summary.semantic_embedding.quantized_values.len(),
                              summary.semantic_embedding.scale_factors.len()));
        
        // Context hints
        if !summary.context_hints.is_empty() {
            text.push_str("Context Relationships:\n");
            for hint in &summary.context_hints {
                text.push_str(&format!("  - {:?} with entity {} (strength: {:.2})\n",
                                      hint.context_type, hint.related_id, hint.strength));
            }
        }
        
        // Metadata
        text.push_str(&format!("Summary Quality: {:.2}, Compression: {:.1}x, Original Size: {} bytes\n",
                              summary.reconstruction_metadata.quality_score,
                              summary.reconstruction_metadata.compression_ratio,
                              summary.reconstruction_metadata.original_size));
        
        text
    }

    /// Estimate how well this summary would help an LLM understand the original
    pub fn estimate_llm_comprehension(&self, summary: &SemanticSummary) -> f32 {
        let mut score = 0.0;
        
        // Entity type clarity
        score += summary.entity_type.confidence * 0.2;
        
        // Feature richness
        let feature_score = (summary.key_features.len() as f32).min(10.0) / 10.0;
        score += feature_score * 0.3;
        
        // Semantic embedding quality
        let embedding_score = (summary.semantic_embedding.quantized_values.len() as f32).min(32.0) / 32.0;
        score += embedding_score * 0.2;
        
        // Context richness
        let context_score = (summary.context_hints.len() as f32).min(5.0) / 5.0;
        score += context_score * 0.2;
        
        // Overall quality
        score += summary.reconstruction_metadata.quality_score * 0.1;
        
        score.min(1.0)
    }

    fn classify_entity(&mut self, entity_data: &EntityData) -> Result<EntityType> {
        // Use existing type_id as primary classification
        let confidence = 0.9; // Would be learned from data
        
        Ok(EntityType {
            type_id: entity_data.type_id,
            confidence,
            secondary_type: None, // Could be inferred from properties
        })
    }

    fn extract_key_features(&mut self, entity_data: &EntityData) -> Result<Vec<KeyFeature>> {
        let mut features = Vec::new();
        
        // Extract from properties text
        let properties = &entity_data.properties;
        
        // 1. Text length feature
        features.push(KeyFeature {
            feature_id: 1,
            value: FeatureValue::Numeric { 
                value: properties.len() as f32, 
                range_hint: Some((0.0, 10000.0)) 
            },
            importance: 0.7,
        });
        
        // 2. Key terms from properties
        let key_terms = self.extract_key_terms(properties);
        if !key_terms.is_empty() {
            features.push(KeyFeature {
                feature_id: 2,
                value: FeatureValue::TextSummary {
                    key_terms,
                    sentiment: self.estimate_sentiment(properties),
                    length_category: self.categorize_text_length(properties.len()),
                },
                importance: 0.9,
            });
        }
        
        // 3. Embedding magnitude feature
        let embedding_magnitude: f32 = entity_data.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        features.push(KeyFeature {
            feature_id: 3,
            value: FeatureValue::Numeric {
                value: embedding_magnitude,
                range_hint: Some((0.0, 2.0)),
            },
            importance: 0.6,
        });
        
        Ok(features)
    }

    fn compress_embedding(&mut self, embedding: &[f32]) -> Result<CompactEmbedding> {
        // Simple but effective compression that preserves semantic meaning
        let compressed = self.embedding_compressor.compress(embedding)?;
        Ok(compressed)
    }

    fn generate_context_hints(&self, _entity_data: &EntityData, _entity_key: EntityKey) -> Result<Vec<ContextHint>> {
        // Would be populated based on graph analysis
        Ok(Vec::new())
    }

    fn extract_key_terms(&mut self, text: &str) -> Vec<u16> {
        // Simple keyword extraction - would be more sophisticated in practice
        let words: Vec<&str> = text.split_whitespace().take(5).collect();
        let mut term_ids = Vec::new();
        
        for word in words {
            let term_id = if let Some(&id) = self.term_dictionary.get(word) {
                id
            } else {
                let id = self.next_term_id;
                self.term_dictionary.insert(word.to_string(), id);
                self.next_term_id += 1;
                id
            };
            term_ids.push(term_id);
        }
        
        term_ids
    }

    fn estimate_sentiment(&self, text: &str) -> i8 {
        // Simple sentiment estimation - would be more sophisticated
        let positive_words = ["good", "great", "excellent", "positive", "success"];
        let negative_words = ["bad", "terrible", "poor", "negative", "failure"];
        
        let text_lower = text.to_lowercase();
        let positive_count = positive_words.iter().filter(|&&word| text_lower.contains(word)).count();
        let negative_count = negative_words.iter().filter(|&&word| text_lower.contains(word)).count();
        
        match positive_count.cmp(&negative_count) {
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
        }
    }

    fn categorize_text_length(&self, length: usize) -> u8 {
        match length {
            0..=50 => 1,      // Short
            51..=200 => 2,    // Medium
            201..=1000 => 3,  // Long
            _ => 4,           // Very long
        }
    }

    fn estimate_original_size(&self, entity_data: &EntityData) -> u32 {
        (entity_data.properties.len() + entity_data.embedding.len() * 4 + 8) as u32
    }

    fn calculate_content_hash(&self, entity_data: &EntityData) -> u32 {
        // Simple hash - would use a proper hash function
        let mut hash = 0u32;
        for byte in entity_data.properties.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
        }
        hash
    }

    fn create_default_extractors() -> HashMap<u16, FeatureExtractor> {
        let mut extractors = HashMap::new();
        
        // Default extractor for all entity types
        extractors.insert(0, FeatureExtractor {
            key_terms: vec!["name".to_string(), "id".to_string(), "type".to_string()],
            numeric_patterns: vec!["count".to_string(), "size".to_string(), "value".to_string()],
            context_patterns: vec!["related".to_string(), "connected".to_string(), "link".to_string()],
        });
        
        extractors
    }
}

impl EmbeddingCompressor {
    pub fn new(target_dimensions: usize) -> Self {
        Self {
            principal_components: Vec::new(),
            quantization_centroids: Vec::new(),
            target_dimensions,
        }
    }

    pub fn compress(&self, embedding: &[f32]) -> Result<CompactEmbedding> {
        // Simple compression that reduces dimensions while preserving semantic meaning
        let chunk_size = embedding.len() / self.target_dimensions.max(1);
        let mut quantized_values = Vec::new();
        let mut scale_factors = Vec::new();
        let mut dimension_map = Vec::new();
        
        for i in 0..self.target_dimensions.min(embedding.len()) {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(embedding.len());
            
            if start < embedding.len() {
                // Average the chunk
                let chunk_sum: f32 = embedding[start..end].iter().sum();
                let chunk_avg = chunk_sum / (end - start) as f32;
                
                // Quantize to u8 with scaling
                let scale = chunk_avg.abs().max(0.01);
                let normalized = (chunk_avg / scale).clamp(-1.0, 1.0);
                let quantized = ((normalized + 1.0) * 127.5) as u8;
                
                quantized_values.push(quantized);
                scale_factors.push(scale);
                dimension_map.push(i as u8);
            }
        }
        
        Ok(CompactEmbedding {
            quantized_values,
            scale_factors,
            dimension_map,
        })
    }

    pub fn decompress(&self, compact: &CompactEmbedding) -> Result<Vec<f32>> {
        let mut result = vec![0.0; compact.quantized_values.len()];
        
        for i in 0..compact.quantized_values.len() {
            let quantized = compact.quantized_values[i];
            let scale = compact.scale_factors.get(i).copied().unwrap_or(1.0);
            
            // Dequantize
            let normalized = (quantized as f32 / 127.5) - 1.0;
            result[i] = normalized * scale;
        }
        
        Ok(result)
    }
}

impl Default for SemanticSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use slotmap::SlotMap;

    // Helper function to create test EntityData
    fn create_test_entity_data() -> EntityData {
        EntityData {
            type_id: 1,
            properties: "This is a test entity with some good properties and excellent features".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    }

    // Helper function to create empty EntityData
    fn create_empty_entity_data() -> EntityData {
        EntityData {
            type_id: 0,
            properties: String::new(),
            embedding: vec![],
        }
    }

    // Helper function to create large embedding
    fn create_large_embedding_entity() -> EntityData {
        EntityData {
            type_id: 2,
            properties: "Large embedding entity".to_string(),
            embedding: (0..128).map(|i| (i as f32) * 0.01).collect(),
        }
    }

    // Helper function to create a test EntityKey
    fn create_test_entity_key() -> EntityKey {
        let mut entities = SlotMap::new();
        entities.insert(())
    }

    #[test]
    fn test_semantic_summarizer_new() {
        let summarizer = SemanticSummarizer::new();
        assert_eq!(summarizer.next_term_id, 1);
        assert!(summarizer.term_dictionary.is_empty());
        assert!(!summarizer.feature_extractors.is_empty());
        assert_eq!(summarizer.embedding_compressor.target_dimensions, 32);
    }

    #[test]
    fn test_create_summary_with_properties() {
        let mut summarizer = SemanticSummarizer::new();
        let entity_data = create_test_entity_data();
        let entity_key = create_test_entity_key();
        
        let result = summarizer.create_summary(&entity_data, entity_key);
        assert!(result.is_ok());
        
        let summary = result.unwrap();
        assert_eq!(summary.entity_type.type_id, 1);
        assert!(summary.entity_type.confidence > 0.0);
        assert!(!summary.key_features.is_empty());
        assert!(!summary.semantic_embedding.quantized_values.is_empty());
        assert!(summary.reconstruction_metadata.original_size > 0);
    }

    #[test]
    fn test_create_summary_with_empty_properties() {
        let mut summarizer = SemanticSummarizer::new();
        let entity_data = create_empty_entity_data();
        let entity_key = create_test_entity_key();
        
        let result = summarizer.create_summary(&entity_data, entity_key);
        assert!(result.is_ok());
        
        let summary = result.unwrap();
        assert_eq!(summary.entity_type.type_id, 0);
        assert!(!summary.key_features.is_empty()); // Should have at least text length feature
    }

    #[test]
    fn test_create_summary_large_embedding() {
        let mut summarizer = SemanticSummarizer::new();
        let entity_data = create_large_embedding_entity();
        let entity_key = create_test_entity_key();
        
        let result = summarizer.create_summary(&entity_data, entity_key);
        assert!(result.is_ok());
        
        let summary = result.unwrap();
        assert!(!summary.semantic_embedding.quantized_values.is_empty());
        assert!(!summary.semantic_embedding.scale_factors.is_empty());
    }

    #[test]
    fn test_to_llm_text_comprehensive() {
        let mut summarizer = SemanticSummarizer::new();
        let entity_data = create_test_entity_data();
        let entity_key = create_test_entity_key();
        
        let summary = summarizer.create_summary(&entity_data, entity_key).unwrap();
        let llm_text = summarizer.to_llm_text(&summary);
        
        // Verify text contains all expected sections
        assert!(llm_text.contains("Entity Type:"));
        assert!(llm_text.contains("Key Features:"));
        assert!(llm_text.contains("Semantic Embedding:"));
        assert!(llm_text.contains("Summary Quality:"));
        assert!(llm_text.contains(&format!("{}", summary.entity_type.type_id)));
        assert!(llm_text.contains(&format!("{:.2}", summary.entity_type.confidence)));
    }

    #[test]
    fn test_to_llm_text_with_empty_summary() {
        let summarizer = SemanticSummarizer::new();
        let summary = SemanticSummary {
            entity_type: EntityType {
                type_id: 0,
                confidence: 0.5,
                secondary_type: None,
            },
            key_features: vec![],
            semantic_embedding: CompactEmbedding {
                quantized_values: vec![],
                scale_factors: vec![],
                dimension_map: vec![],
            },
            context_hints: vec![],
            reconstruction_metadata: ReconstructionMetadata {
                original_size: 0,
                compression_ratio: 1.0,
                quality_score: 0.0,
                content_hash: 0,
            },
        };
        
        let llm_text = summarizer.to_llm_text(&summary);
        
        // Should handle empty summary gracefully
        assert!(llm_text.contains("Entity Type: 0"));
        assert!(llm_text.contains("Key Features:"));
        assert!(!llm_text.contains("Context Relationships:")); // Should not appear for empty hints
    }

    #[test]
    fn test_to_llm_text_with_context_hints() {
        let summarizer = SemanticSummarizer::new();
        let summary = SemanticSummary {
            entity_type: EntityType {
                type_id: 1,
                confidence: 0.9,
                secondary_type: None,
            },
            key_features: vec![],
            semantic_embedding: CompactEmbedding {
                quantized_values: vec![128],
                scale_factors: vec![1.0],
                dimension_map: vec![0],
            },
            context_hints: vec![
                ContextHint {
                    context_type: ContextType::SemanticCluster,
                    related_id: 123,
                    strength: 0.8,
                }
            ],
            reconstruction_metadata: ReconstructionMetadata {
                original_size: 100,
                compression_ratio: 2.0,
                quality_score: 0.85,
                content_hash: 12345,
            },
        };
        
        let llm_text = summarizer.to_llm_text(&summary);
        
        // Should include context relationships section
        assert!(llm_text.contains("Context Relationships:"));
        assert!(llm_text.contains("SemanticCluster"));
        assert!(llm_text.contains("entity 123"));
        assert!(llm_text.contains("strength: 0.80"));
    }

    #[test]
    fn test_estimate_llm_comprehension_high_quality() {
        let summarizer = SemanticSummarizer::new();
        let summary = SemanticSummary {
            entity_type: EntityType {
                type_id: 1,
                confidence: 0.95,
                secondary_type: None,
            },
            key_features: vec![
                KeyFeature {
                    feature_id: 1,
                    value: FeatureValue::Category(1),
                    importance: 0.9,
                },
                KeyFeature {
                    feature_id: 2,
                    value: FeatureValue::Numeric { value: 1.0, range_hint: None },
                    importance: 0.8,
                },
            ],
            semantic_embedding: CompactEmbedding {
                quantized_values: vec![128; 32], // Full 32 dimensions
                scale_factors: vec![1.0; 32],
                dimension_map: (0..32).map(|i| i as u8).collect(),
            },
            context_hints: vec![
                ContextHint {
                    context_type: ContextType::SemanticCluster,
                    related_id: 1,
                    strength: 0.9,
                },
            ],
            reconstruction_metadata: ReconstructionMetadata {
                original_size: 1000,
                compression_ratio: 5.0,
                quality_score: 0.95,
                content_hash: 12345,
            },
        };
        
        let comprehension = summarizer.estimate_llm_comprehension(&summary);
        
        // Should be close to 1.0 for high-quality summary
        assert!(comprehension > 0.8);
        assert!(comprehension <= 1.0);
    }

    #[test]
    fn test_estimate_llm_comprehension_low_quality() {
        let summarizer = SemanticSummarizer::new();
        let summary = SemanticSummary {
            entity_type: EntityType {
                type_id: 0,
                confidence: 0.1,
                secondary_type: None,
            },
            key_features: vec![],
            semantic_embedding: CompactEmbedding {
                quantized_values: vec![],
                scale_factors: vec![],
                dimension_map: vec![],
            },
            context_hints: vec![],
            reconstruction_metadata: ReconstructionMetadata {
                original_size: 10,
                compression_ratio: 1.0,
                quality_score: 0.1,
                content_hash: 0,
            },
        };
        
        let comprehension = summarizer.estimate_llm_comprehension(&summary);
        
        // Should be low for poor-quality summary
        assert!(comprehension < 0.5);
        assert!(comprehension >= 0.0);
    }

    #[test]
    fn test_estimate_llm_comprehension_capped_at_one() {
        let summarizer = SemanticSummarizer::new();
        let summary = SemanticSummary {
            entity_type: EntityType {
                type_id: 1,
                confidence: 1.0,
                secondary_type: None,
            },
            key_features: (0..20).map(|i| KeyFeature {
                feature_id: i,
                value: FeatureValue::Category(1),
                importance: 1.0,
            }).collect(),
            semantic_embedding: CompactEmbedding {
                quantized_values: vec![128; 50], // More than 32
                scale_factors: vec![1.0; 50],
                dimension_map: (0..50).map(|i| i as u8).collect(),
            },
            context_hints: (0..10).map(|i| ContextHint {
                context_type: ContextType::SemanticCluster,
                related_id: i,
                strength: 1.0,
            }).collect(),
            reconstruction_metadata: ReconstructionMetadata {
                original_size: 1000,
                compression_ratio: 5.0,
                quality_score: 1.0,
                content_hash: 12345,
            },
        };
        
        let comprehension = summarizer.estimate_llm_comprehension(&summary);
        
        // Should be capped at 1.0
        assert_eq!(comprehension, 1.0);
    }

    #[test]
    fn test_extract_key_terms() {
        let mut summarizer = SemanticSummarizer::new();
        let text = "hello world test example data";
        
        let term_ids = summarizer.extract_key_terms(text);
        
        assert_eq!(term_ids.len(), 5); // Takes first 5 words
        assert_eq!(summarizer.term_dictionary.len(), 5);
        assert!(summarizer.term_dictionary.contains_key("hello"));
        assert!(summarizer.term_dictionary.contains_key("world"));
        assert_eq!(summarizer.next_term_id, 6); // Should increment for each new term
    }

    #[test]
    fn test_extract_key_terms_reuse_existing() {
        let mut summarizer = SemanticSummarizer::new();
        
        // First extraction
        let term_ids1 = summarizer.extract_key_terms("hello world");
        
        // Second extraction with overlapping terms
        let term_ids2 = summarizer.extract_key_terms("hello test");
        
        assert_eq!(term_ids1[0], term_ids2[0]); // "hello" should have same ID
        assert_ne!(term_ids1[1], term_ids2[1]); // "world" vs "test" should differ
        assert_eq!(summarizer.term_dictionary.len(), 3); // hello, world, test
    }

    #[test]
    fn test_extract_key_terms_empty_text() {
        let mut summarizer = SemanticSummarizer::new();
        let term_ids = summarizer.extract_key_terms("");
        
        assert!(term_ids.is_empty());
        assert!(summarizer.term_dictionary.is_empty());
        assert_eq!(summarizer.next_term_id, 1);
    }

    #[test]
    fn test_estimate_sentiment_positive() {
        let summarizer = SemanticSummarizer::new();
        let text = "This is a great and excellent example of good work";
        
        let sentiment = summarizer.estimate_sentiment(text);
        
        assert_eq!(sentiment, 1); // Should be positive
    }

    #[test]
    fn test_estimate_sentiment_negative() {
        let summarizer = SemanticSummarizer::new();
        let text = "This is terrible and bad work with poor results and failure";
        
        let sentiment = summarizer.estimate_sentiment(text);
        
        assert_eq!(sentiment, -1); // Should be negative
    }

    #[test]
    fn test_estimate_sentiment_neutral() {
        let summarizer = SemanticSummarizer::new();
        let text = "This is a neutral text without sentiment words";
        
        let sentiment = summarizer.estimate_sentiment(text);
        
        assert_eq!(sentiment, 0); // Should be neutral
    }

    #[test]
    fn test_estimate_sentiment_mixed() {
        let summarizer = SemanticSummarizer::new();
        let text = "This has good and bad elements with excellent and terrible aspects";
        
        let sentiment = summarizer.estimate_sentiment(text);
        
        assert_eq!(sentiment, 0); // Should be neutral due to equal positive/negative
    }

    #[test]
    fn test_categorize_text_length() {
        let summarizer = SemanticSummarizer::new();
        
        assert_eq!(summarizer.categorize_text_length(0), 1);    // Short
        assert_eq!(summarizer.categorize_text_length(25), 1);   // Short
        assert_eq!(summarizer.categorize_text_length(50), 1);   // Short
        assert_eq!(summarizer.categorize_text_length(51), 2);   // Medium
        assert_eq!(summarizer.categorize_text_length(100), 2);  // Medium
        assert_eq!(summarizer.categorize_text_length(200), 2);  // Medium
        assert_eq!(summarizer.categorize_text_length(201), 3);  // Long
        assert_eq!(summarizer.categorize_text_length(500), 3);  // Long
        assert_eq!(summarizer.categorize_text_length(1000), 3); // Long
        assert_eq!(summarizer.categorize_text_length(1001), 4); // Very long
        assert_eq!(summarizer.categorize_text_length(5000), 4); // Very long
    }

    #[test]
    fn test_calculate_content_hash() {
        let summarizer = SemanticSummarizer::new();
        let entity1 = create_test_entity_data();
        let entity2 = create_empty_entity_data();
        
        let hash1 = summarizer.calculate_content_hash(&entity1);
        let hash2 = summarizer.calculate_content_hash(&entity2);
        
        assert_ne!(hash1, hash2); // Different content should have different hashes
        
        // Same content should have same hash
        let hash1_again = summarizer.calculate_content_hash(&entity1);
        assert_eq!(hash1, hash1_again);
    }

    #[test]
    fn test_estimate_original_size() {
        let summarizer = SemanticSummarizer::new();
        let entity = create_test_entity_data();
        
        let size = summarizer.estimate_original_size(&entity);
        
        // Should include properties length + embedding size + metadata
        let expected_min = entity.properties.len() + entity.embedding.len() * 4 + 8;
        assert_eq!(size as usize, expected_min);
    }

    #[test]
    fn test_classify_entity() {
        let mut summarizer = SemanticSummarizer::new();
        let entity_data = create_test_entity_data();
        
        let result = summarizer.classify_entity(&entity_data);
        assert!(result.is_ok());
        
        let entity_type = result.unwrap();
        assert_eq!(entity_type.type_id, entity_data.type_id);
        assert!(entity_type.confidence > 0.0);
        assert!(entity_type.secondary_type.is_none());
    }

    #[test]
    fn test_extract_key_features() {
        let mut summarizer = SemanticSummarizer::new();
        let entity_data = create_test_entity_data();
        
        let result = summarizer.extract_key_features(&entity_data);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        assert!(!features.is_empty());
        
        // Should have at least text length, key terms, and embedding magnitude features
        assert!(features.iter().any(|f| f.feature_id == 1)); // Text length
        assert!(features.iter().any(|f| f.feature_id == 2)); // Key terms
        assert!(features.iter().any(|f| f.feature_id == 3)); // Embedding magnitude
    }

    #[test]
    fn test_extract_key_features_empty_properties() {
        let mut summarizer = SemanticSummarizer::new();
        let entity_data = create_empty_entity_data();
        
        let result = summarizer.extract_key_features(&entity_data);
        assert!(result.is_ok());
        
        let features = result.unwrap();
        // Should still have text length and embedding magnitude features
        assert!(features.iter().any(|f| f.feature_id == 1)); // Text length (0)
        assert!(features.iter().any(|f| f.feature_id == 3)); // Embedding magnitude
        // Should not have text summary feature due to empty key terms
        assert!(!features.iter().any(|f| f.feature_id == 2));
    }

    #[test]
    fn test_compress_embedding() {
        let mut summarizer = SemanticSummarizer::new();
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        
        let result = summarizer.compress_embedding(&embedding);
        assert!(result.is_ok());
        
        let compressed = result.unwrap();
        assert!(!compressed.quantized_values.is_empty());
        assert!(!compressed.scale_factors.is_empty());
        assert!(!compressed.dimension_map.is_empty());
        assert_eq!(compressed.quantized_values.len(), compressed.scale_factors.len());
    }

    #[test]
    fn test_generate_context_hints() {
        let summarizer = SemanticSummarizer::new();
        let entity_data = create_test_entity_data();
        let entity_key = create_test_entity_key();
        
        let result = summarizer.generate_context_hints(&entity_data, entity_key);
        assert!(result.is_ok());
        
        let hints = result.unwrap();
        // Currently returns empty vector - this is expected behavior
        assert!(hints.is_empty());
    }

    #[test]
    fn test_embedding_compressor_new() {
        let compressor = EmbeddingCompressor::new(16);
        assert_eq!(compressor.target_dimensions, 16);
        assert!(compressor.principal_components.is_empty());
        assert!(compressor.quantization_centroids.is_empty());
    }

    #[test]
    fn test_embedding_compressor_compress() {
        let compressor = EmbeddingCompressor::new(4);
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        
        let result = compressor.compress(&embedding);
        assert!(result.is_ok());
        
        let compressed = result.unwrap();
        assert_eq!(compressed.quantized_values.len(), 4);
        assert_eq!(compressed.scale_factors.len(), 4);
        assert_eq!(compressed.dimension_map.len(), 4);
    }

    #[test]
    fn test_embedding_compressor_compress_empty() {
        let compressor = EmbeddingCompressor::new(4);
        let embedding = vec![];
        
        let result = compressor.compress(&embedding);
        assert!(result.is_ok());
        
        let compressed = result.unwrap();
        assert!(compressed.quantized_values.is_empty());
        assert!(compressed.scale_factors.is_empty());
        assert!(compressed.dimension_map.is_empty());
    }

    #[test]
    fn test_embedding_compressor_compress_small() {
        let compressor = EmbeddingCompressor::new(8);
        let embedding = vec![0.5, 1.0]; // Smaller than target dimensions
        
        let result = compressor.compress(&embedding);
        assert!(result.is_ok());
        
        let compressed = result.unwrap();
        assert_eq!(compressed.quantized_values.len(), 2); // Should be limited by input size
    }

    #[test]
    fn test_embedding_compressor_decompress() {
        let compressor = EmbeddingCompressor::new(4);
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        
        let compressed = compressor.compress(&embedding).unwrap();
        let result = compressor.decompress(&compressed);
        assert!(result.is_ok());
        
        let decompressed = result.unwrap();
        assert_eq!(decompressed.len(), compressed.quantized_values.len());
        
        // Values should be in reasonable range
        for value in &decompressed {
            assert!(value.abs() <= 2.0); // Should be reasonable after scaling
        }
    }

    #[test]
    fn test_embedding_compressor_roundtrip() {
        let compressor = EmbeddingCompressor::new(4);
        let embedding = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        
        let compressed = compressor.compress(&embedding).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();
        
        // Should preserve general structure (signs, relative magnitudes)
        assert_eq!(decompressed.len(), 4);
        // Due to quantization, exact values won't match, but structure should be preserved
        for value in &decompressed {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn test_feature_value_variants() {
        // Test all FeatureValue variants can be created and used
        let category = FeatureValue::Category(42);
        let numeric = FeatureValue::Numeric { value: 3.14, range_hint: Some((0.0, 10.0)) };
        let text_summary = FeatureValue::TextSummary {
            key_terms: vec![1, 2, 3],
            sentiment: 1,
            length_category: 2,
        };
        let temporal = FeatureValue::Temporal {
            timestamp: 1234567890,
            duration_hint: Some(3600),
        };
        
        // Verify they can be pattern matched (part of to_llm_text functionality)
        match category {
            FeatureValue::Category(id) => assert_eq!(id, 42),
            _ => panic!("Wrong variant"),
        }
        
        match numeric {
            FeatureValue::Numeric { value, range_hint } => {
                assert!((value - 3.14).abs() < f32::EPSILON);
                assert_eq!(range_hint, Some((0.0, 10.0)));
            },
            _ => panic!("Wrong variant"),
        }
        
        match text_summary {
            FeatureValue::TextSummary { key_terms, sentiment, length_category } => {
                assert_eq!(key_terms, vec![1, 2, 3]);
                assert_eq!(sentiment, 1);
                assert_eq!(length_category, 2);
            },
            _ => panic!("Wrong variant"),
        }
        
        match temporal {
            FeatureValue::Temporal { timestamp, duration_hint } => {
                assert_eq!(timestamp, 1234567890);
                assert_eq!(duration_hint, Some(3600));
            },
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_context_type_variants() {
        // Test all ContextType variants
        let types = vec![
            ContextType::SemanticCluster,
            ContextType::TemporalSequence,
            ContextType::CausalLink,
            ContextType::SpatialProximity,
            ContextType::Hierarchy,
        ];
        
        // Verify they can be debug formatted (used in to_llm_text)
        for context_type in types {
            let debug_str = format!("{:?}", context_type);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_create_default_extractors() {
        let extractors = SemanticSummarizer::create_default_extractors();
        
        assert!(!extractors.is_empty());
        assert!(extractors.contains_key(&0)); // Default extractor
        
        let default_extractor = &extractors[&0];
        assert!(!default_extractor.key_terms.is_empty());
        assert!(!default_extractor.numeric_patterns.is_empty());
        assert!(!default_extractor.context_patterns.is_empty());
        assert!(default_extractor.key_terms.contains(&"name".to_string()));
        assert!(default_extractor.numeric_patterns.contains(&"count".to_string()));
        assert!(default_extractor.context_patterns.contains(&"related".to_string()));
    }
}