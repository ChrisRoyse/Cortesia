//! Intelligent Knowledge Processor
//! 
//! Central coordinator that orchestrates advanced entity extraction,
//! relationship mapping, semantic chunking, and context analysis.

use std::sync::Arc;
use std::time::Instant;
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    knowledge_processing::types::*,
};
use super::{
    AdvancedEntityExtractor, EntityExtractionConfig,
    AdvancedRelationshipMapper, RelationshipExtractionConfig,
    SemanticChunker, SemanticChunkingConfig,
    ContextAnalyzer, ContextAnalysisConfig,
};

/// Central intelligent knowledge processor
pub struct IntelligentKnowledgeProcessor {
    model_manager: Arc<ModelResourceManager>,
    entity_extractor: AdvancedEntityExtractor,
    relationship_mapper: AdvancedRelationshipMapper,
    semantic_chunker: SemanticChunker,
    context_analyzer: ContextAnalyzer,
    config: KnowledgeProcessingConfig,
}

impl IntelligentKnowledgeProcessor {
    /// Create new intelligent knowledge processor
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        config: KnowledgeProcessingConfig,
    ) -> Self {
        // Create component configs based on main config
        let entity_config = EntityExtractionConfig {
            model_id: config.entity_extraction_model.clone(),
            min_confidence: config.min_entity_confidence,
            ..Default::default()
        };
        
        let relationship_config = RelationshipExtractionConfig {
            model_id: config.relationship_extraction_model.clone(),
            min_confidence: config.min_relationship_confidence,
            ..Default::default()
        };
        
        let chunking_config = SemanticChunkingConfig {
            model_id: config.semantic_analysis_model.clone(),
            max_chunk_size: config.max_chunk_size,
            min_chunk_size: config.min_chunk_size,
            overlap_strategy: super::semantic_chunker::OverlapStrategy::SemanticOverlap { 
                tokens: config.chunk_overlap_size / 4 // Approximate tokens
            },
            ..Default::default()
        };
        
        let context_config = ContextAnalysisConfig {
            model_id: config.semantic_analysis_model.clone(),
            enable_global_context: config.preserve_context,
            enable_local_context: config.preserve_context,
            ..Default::default()
        };
        
        // Create components
        let entity_extractor = AdvancedEntityExtractor::new(model_manager.clone(), entity_config);
        let relationship_mapper = AdvancedRelationshipMapper::new(model_manager.clone(), relationship_config);
        let semantic_chunker = SemanticChunker::new(model_manager.clone(), chunking_config);
        let context_analyzer = ContextAnalyzer::new(model_manager.clone(), context_config);
        
        Self {
            model_manager,
            entity_extractor,
            relationship_mapper,
            semantic_chunker,
            context_analyzer,
            config,
        }
    }
    
    /// Process knowledge with full pipeline
    pub async fn process_knowledge(
        &self,
        content: &str,
        title: &str,
    ) -> KnowledgeProcessingResult2<KnowledgeProcessingResult> {
        let start_time = Instant::now();
        let document_id = self.generate_document_id(title);
        
        // Step 1: Global Context Analysis
        let global_context = self.context_analyzer
            .analyze_global_context(content, title)
            .await?;
        
        // Step 2: Semantic Structure Detection & Intelligent Chunking
        let chunks = self.semantic_chunker
            .create_semantic_chunks(content)
            .await?;
        
        // Step 3: Enhanced Entity Extraction (per chunk)
        let mut all_entities = Vec::new();
        let mut chunk_entities = Vec::new();
        
        for chunk in &chunks {
            let entities = self.entity_extractor
                .extract_entities_with_context(&chunk.content)
                .await?;
            
            chunk_entities.push(entities.clone());
            all_entities.extend(entities);
        }
        
        // Step 4: Complex Relationship Mapping (per chunk)
        let mut all_relationships = Vec::new();
        let mut chunk_relationships = Vec::new();
        
        for (chunk, entities) in chunks.iter().zip(&chunk_entities) {
            let relationships = self.relationship_mapper
                .extract_complex_relationships(&chunk.content, entities)
                .await?;
            
            chunk_relationships.push(relationships.clone());
            all_relationships.extend(relationships);
        }
        
        // Step 5: Deduplicate and merge global entities/relationships
        let global_entities = self.deduplicate_entities(all_entities);
        let global_relationships = self.deduplicate_relationships(all_relationships);
        
        // Step 6: Enhance chunks with extracted data
        let enhanced_chunks = self.enhance_chunks_with_extractions(
            chunks,
            chunk_entities,
            chunk_relationships,
        );
        
        // Step 7: Build cross-references
        let cross_references = self.context_analyzer
            .build_cross_references(&enhanced_chunks, &global_context)
            .await?;
        
        // Step 8: Validate context preservation
        let context_validation = self.context_analyzer
            .validate_context_preservation(&enhanced_chunks, &cross_references, &global_context);
        
        // Step 9: Analyze document structure
        let document_structure = self.create_document_structure(&enhanced_chunks, &global_context);
        
        // Step 10: Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(
            &enhanced_chunks,
            &global_entities,
            &global_relationships,
            &context_validation,
        );
        
        // Step 11: Create processing metadata
        let processing_metadata = crate::enhanced_knowledge_storage::knowledge_processing::types::ProcessingMetadata {
            processing_time: start_time.elapsed(),
            models_used: vec![
                self.config.entity_extraction_model.clone(),
                self.config.relationship_extraction_model.clone(),
                self.config.semantic_analysis_model.clone(),
            ],
            total_tokens_processed: content.len() / 4, // Rough approximation
            chunks_created: enhanced_chunks.len(),
            entities_extracted: global_entities.len(),
            relationships_extracted: global_relationships.len(),
            memory_usage_peak: 0, // Would need actual memory tracking
        };
        
        Ok(KnowledgeProcessingResult {
            document_id,
            chunks: enhanced_chunks,
            global_entities,
            global_relationships,
            document_structure,
            processing_metadata,
            quality_metrics,
        })
    }
    
    /// Generate unique document ID
    fn generate_document_id(&self, title: &str) -> String {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let title_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            title.hash(&mut hasher);
            hasher.finish()
        };
        
        format!("doc_{}_{:x}", timestamp, title_hash)
    }
    
    /// Deduplicate entities by name and type
    fn deduplicate_entities(&self, mut entities: Vec<ContextualEntity>) -> Vec<ContextualEntity> {
        // Sort by confidence (highest first)
        entities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // Remove duplicates (keep highest confidence version)
        let mut seen = std::collections::HashSet::new();
        entities.retain(|entity| {
            let key = (entity.name.clone(), entity.entity_type.clone());
            seen.insert(key)
        });
        
        entities
    }
    
    /// Deduplicate relationships by source, predicate, target
    fn deduplicate_relationships(&self, mut relationships: Vec<ComplexRelationship>) -> Vec<ComplexRelationship> {
        // Sort by confidence * strength (highest first)
        relationships.sort_by(|a, b| {
            let score_a = a.confidence * a.relationship_strength;
            let score_b = b.confidence * b.relationship_strength;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Remove duplicates (keep highest scoring version)
        let mut seen = std::collections::HashSet::new();
        relationships.retain(|rel| {
            let key = (rel.source.clone(), rel.predicate.clone(), rel.target.clone());
            seen.insert(key)
        });
        
        relationships
    }
    
    /// Enhance chunks with extracted entities and relationships
    fn enhance_chunks_with_extractions(
        &self,
        mut chunks: Vec<SemanticChunk>,
        chunk_entities: Vec<Vec<ContextualEntity>>,
        chunk_relationships: Vec<Vec<ComplexRelationship>>,
    ) -> Vec<SemanticChunk> {
        for (i, chunk) in chunks.iter_mut().enumerate() {
            if let Some(entities) = chunk_entities.get(i) {
                chunk.entities = entities.clone();
                
                // Extract key concepts from entities
                chunk.key_concepts = entities
                    .iter()
                    .filter(|e| e.confidence > 0.8)
                    .map(|e| e.name.clone())
                    .collect();
            }
            
            if let Some(relationships) = chunk_relationships.get(i) {
                chunk.relationships = relationships.clone();
            }
        }
        
        chunks
    }
    
    /// Create document structure from processed chunks
    fn create_document_structure(
        &self,
        chunks: &[SemanticChunk],
        global_context: &super::context_analyzer::GlobalContext,
    ) -> DocumentStructure {
        let sections = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| DocumentSection {
                title: if i == 0 { Some("Beginning".to_string()) } else { None },
                start_pos: chunk.start_pos,
                end_pos: chunk.end_pos,
                section_type: match chunk.chunk_type {
                    ChunkType::Section => SectionType::Body,
                    ChunkType::Paragraph => SectionType::Body,
                    _ => SectionType::Other,
                },
                key_points: chunk.key_concepts.clone(),
            })
            .collect();
        
        let total_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
        let estimated_reading_time = std::time::Duration::from_secs((total_chars / 1000) as u64 * 60 / 200); // ~200 wpm
        
        DocumentStructure {
            sections,
            overall_topic: Some(global_context.document_theme.clone()),
            key_themes: global_context.key_entities.clone(),
            complexity_level: if chunks.len() > 10 { ComplexityLevel::High } 
                             else if chunks.len() > 5 { ComplexityLevel::Medium }
                             else { ComplexityLevel::Low },
            estimated_reading_time,
        }
    }
    
    /// Calculate quality metrics for the processing result
    fn calculate_quality_metrics(
        &self,
        chunks: &[SemanticChunk],
        entities: &[ContextualEntity],
        relationships: &[ComplexRelationship],
        context_validation: &super::context_analyzer::ContextValidationResult,
    ) -> QualityMetrics {
        // Entity extraction quality (based on confidence and coverage)
        let entity_extraction_quality = if entities.is_empty() { 0.0 } else {
            let avg_confidence = entities.iter().map(|e| e.confidence).sum::<f32>() / entities.len() as f32;
            let coverage = entities.len() as f32 / chunks.len().max(1) as f32; // Entities per chunk
            (avg_confidence * 0.7 + coverage.min(1.0) * 0.3).min(1.0)
        };
        
        // Relationship extraction quality
        let relationship_extraction_quality = if relationships.is_empty() { 0.0 } else {
            let avg_confidence = relationships.iter().map(|r| r.confidence).sum::<f32>() / relationships.len() as f32;
            let avg_strength = relationships.iter().map(|r| r.relationship_strength).sum::<f32>() / relationships.len() as f32;
            (avg_confidence * 0.5 + avg_strength * 0.5).min(1.0)
        };
        
        // Semantic coherence (from chunks)
        let semantic_coherence = if chunks.is_empty() { 0.0 } else {
            chunks.iter().map(|c| c.semantic_coherence).sum::<f32>() / chunks.len() as f32
        };
        
        // Context preservation (from validation)
        let context_preservation = context_validation.context_preservation_score;
        
        let mut quality_metrics = QualityMetrics {
            entity_extraction_quality,
            relationship_extraction_quality,
            semantic_coherence,
            context_preservation,
            overall_quality: 0.0, // Will be calculated
        };
        
        quality_metrics.calculate_overall_quality();
        quality_metrics
    }
    
    /// Get processing statistics
    pub fn get_processing_stats(&self, result: &KnowledgeProcessingResult) -> ProcessingStats {
        ProcessingStats {
            total_chunks: result.chunks.len(),
            total_entities: result.global_entities.len(),
            total_relationships: result.global_relationships.len(),
            processing_time: result.processing_metadata.processing_time,
            average_chunk_size: if result.chunks.is_empty() { 0.0 } else {
                result.chunks.iter().map(|c| c.content.len()).sum::<usize>() as f32 / result.chunks.len() as f32
            },
            quality_score: result.quality_metrics.overall_quality,
            models_used: result.processing_metadata.models_used.clone(),
        }
    }
    
    /// Validate processing result
    pub fn validate_processing_result(&self, result: &KnowledgeProcessingResult) -> ProcessingValidation {
        let mut validation_errors = Vec::new();
        let mut validation_warnings = Vec::new();
        
        // Check chunk quality
        for chunk in &result.chunks {
            if chunk.content.len() < self.config.min_chunk_size {
                validation_warnings.push(format!("Chunk {} is below minimum size", chunk.id));
            }
            if chunk.content.len() > self.config.max_chunk_size {
                validation_errors.push(format!("Chunk {} exceeds maximum size", chunk.id));
            }
            if chunk.semantic_coherence < 0.5 {
                validation_warnings.push(format!("Chunk {} has low semantic coherence", chunk.id));
            }
        }
        
        // Check entity quality
        let low_confidence_entities = result.global_entities
            .iter()
            .filter(|e| e.confidence < self.config.min_entity_confidence)
            .count();
        
        if low_confidence_entities > 0 {
            validation_warnings.push(format!("{} entities below confidence threshold", low_confidence_entities));
        }
        
        // Check relationship quality
        let low_confidence_relationships = result.global_relationships
            .iter()
            .filter(|r| r.confidence < self.config.min_relationship_confidence)
            .count();
        
        if low_confidence_relationships > 0 {
            validation_warnings.push(format!("{} relationships below confidence threshold", low_confidence_relationships));
        }
        
        // Check overall quality
        if result.quality_metrics.overall_quality < 0.7 {
            validation_warnings.push("Overall processing quality is below recommended threshold".to_string());
        }
        
        ProcessingValidation {
            is_valid: validation_errors.is_empty(),
            quality_score: result.quality_metrics.overall_quality,
            errors: validation_errors,
            warnings: validation_warnings,
        }
    }
}

/// Statistics about processing performance
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub total_chunks: usize,
    pub total_entities: usize,
    pub total_relationships: usize,
    pub processing_time: std::time::Duration,
    pub average_chunk_size: f32,
    pub quality_score: f32,
    pub models_used: Vec<String>,
}

/// Validation result for processing
#[derive(Debug, Clone)]
pub struct ProcessingValidation {
    pub is_valid: bool,
    pub quality_score: f32,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_intelligent_processor_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let processing_config = KnowledgeProcessingConfig::default();
        
        let processor = IntelligentKnowledgeProcessor::new(model_manager, processing_config);
        
        assert_eq!(processor.config.entity_extraction_model, "smollm2_360m");
        assert_eq!(processor.config.max_chunk_size, 2048);
    }
    
    #[test]
    fn test_document_id_generation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let processing_config = KnowledgeProcessingConfig::default();
        
        let processor = IntelligentKnowledgeProcessor::new(model_manager, processing_config);
        
        let id1 = processor.generate_document_id("Test Title");
        let id2 = processor.generate_document_id("Test Title");
        let id3 = processor.generate_document_id("Different Title");
        
        assert_ne!(id1, id2); // Different timestamps
        assert_ne!(id1, id3); // Different content
        assert!(id1.starts_with("doc_"));
    }
    
    #[test]
    fn test_entity_deduplication() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let processing_config = KnowledgeProcessingConfig::default();
        
        let processor = IntelligentKnowledgeProcessor::new(model_manager, processing_config);
        
        let entities = vec![
            ContextualEntity {
                name: "Einstein".to_string(),
                entity_type: EntityType::Person,
                context: "context1".to_string(),
                confidence: 0.9,
                span: None,
                attributes: std::collections::HashMap::new(),
                extracted_at: 1000,
            },
            ContextualEntity {
                name: "Einstein".to_string(),
                entity_type: EntityType::Person,
                context: "context2".to_string(),
                confidence: 0.8, // Lower confidence
                span: None,
                attributes: std::collections::HashMap::new(),
                extracted_at: 2000,
            },
            ContextualEntity {
                name: "Newton".to_string(),
                entity_type: EntityType::Person,
                context: "context3".to_string(),
                confidence: 0.95,
                span: None,
                attributes: std::collections::HashMap::new(),
                extracted_at: 3000,
            },
        ];
        
        let deduplicated = processor.deduplicate_entities(entities);
        
        assert_eq!(deduplicated.len(), 2); // Einstein and Newton
        assert_eq!(deduplicated[0].confidence, 0.95); // Newton (highest confidence)
        assert_eq!(deduplicated[1].confidence, 0.9);  // Einstein (higher confidence version kept)
    }
    
    #[test]
    fn test_relationship_deduplication() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let processing_config = KnowledgeProcessingConfig::default();
        
        let processor = IntelligentKnowledgeProcessor::new(model_manager, processing_config);
        
        let relationships = vec![
            ComplexRelationship {
                source: "Einstein".to_string(),
                predicate: RelationshipType::CreatedBy,
                target: "Relativity".to_string(),
                context: "context1".to_string(),
                confidence: 0.9,
                supporting_evidence: vec![],
                relationship_strength: 0.8,
                temporal_info: None,
            },
            ComplexRelationship {
                source: "Einstein".to_string(),
                predicate: RelationshipType::CreatedBy,
                target: "Relativity".to_string(),
                context: "context2".to_string(),
                confidence: 0.8,
                supporting_evidence: vec![],
                relationship_strength: 0.7, // Lower score
                temporal_info: None,
            },
        ];
        
        let deduplicated = processor.deduplicate_relationships(relationships);
        
        assert_eq!(deduplicated.len(), 1); // Only one relationship kept
        assert_eq!(deduplicated[0].confidence, 0.9); // Higher confidence version kept
    }
    
    #[tokio::test]
    async fn test_processing_validation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let processing_config = KnowledgeProcessingConfig::default();
        
        let processor = IntelligentKnowledgeProcessor::new(model_manager, processing_config);
        
        // Create a mock processing result
        let result = KnowledgeProcessingResult {
            document_id: "test_doc".to_string(),
            chunks: vec![
                SemanticChunk {
                    id: "chunk1".to_string(),
                    content: "x".repeat(100), // Below min size
                    start_pos: 0,
                    end_pos: 100,
                    semantic_coherence: 0.3, // Low coherence
                    key_concepts: vec![],
                    entities: vec![],
                    relationships: vec![],
                    chunk_type: ChunkType::Paragraph,
                    overlap_with_previous: None,
                    overlap_with_next: None,
                },
            ],
            global_entities: vec![],
            global_relationships: vec![],
            document_structure: DocumentStructure {
                sections: vec![],
                overall_topic: None,
                key_themes: vec![],
                complexity_level: ComplexityLevel::Low,
                estimated_reading_time: std::time::Duration::from_secs(60),
            },
            processing_metadata: ProcessingMetadata {
                processing_time: std::time::Duration::from_millis(100),
                models_used: vec!["test_model".to_string()],
                total_tokens_processed: 100,
                chunks_created: 1,
                entities_extracted: 0,
                relationships_extracted: 0,
                memory_usage_peak: 1000000,
            },
            quality_metrics: QualityMetrics {
                entity_extraction_quality: 0.5,
                relationship_extraction_quality: 0.5,
                semantic_coherence: 0.3,
                context_preservation: 0.5,
                overall_quality: 0.45,
            },
        };
        
        let validation = processor.validate_processing_result(&result);
        
        assert!(validation.is_valid); // No errors, only warnings
        assert!(!validation.warnings.is_empty()); // Should have warnings
        assert_eq!(validation.quality_score, 0.45);
    }
}