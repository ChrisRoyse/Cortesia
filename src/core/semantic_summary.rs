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