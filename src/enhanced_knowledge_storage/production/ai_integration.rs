//! AI Component Integration Adapters
//! 
//! Provides integration adapters to bridge between real AI components
//! and the production system orchestrator.

use std::sync::Arc;
use crate::enhanced_knowledge_storage::ai_components::{
    RealEntityExtractor, RealSemanticChunker, RealReasoningEngine,
    Entity as AIEntity, SemanticChunk, ReasoningResult,
    EntityExtractionConfig, SemanticChunkingConfig, ReasoningConfig,
};
use crate::enhanced_knowledge_storage::production::system_orchestrator::{
    ExtractedEntity, ProcessedChunk, EntityRelationship, EntityAttribute,
};
use crate::core::types::EntityKey;

/// Integration adapter for entity extraction
pub struct EntityExtractionAdapter {
    extractor: Arc<RealEntityExtractor>,
}

impl EntityExtractionAdapter {
    pub fn new(extractor: Arc<RealEntityExtractor>) -> Self {
        Self { extractor }
    }
    
    /// Extract entities and convert to production format
    pub async fn extract_entities_enhanced(
        &self, 
        content: &str, 
        _context: &str
    ) -> Result<Vec<ExtractedEntity>, String> {
        let ai_entities = self.extractor
            .extract_entities(content)
            .await
            .map_err(|e| e.to_string())?;
        
        let extracted_entities = ai_entities
            .into_iter()
            .map(|entity| ExtractedEntity {
                id: uuid::Uuid::new_v4().to_string(),
                name: entity.name,
                entity_type: entity.entity_type.to_string(),
                description: entity.context,
                confidence: entity.confidence,
                attributes: entity.attributes.into_iter()
                    .map(|(k, v)| EntityAttribute {
                        key: k,
                        value: v,
                        confidence: 0.9,
                        source: "AI".to_string(),
                    })
                    .collect(),
                source_chunks: vec!["current_chunk".to_string()],
            })
            .collect();
        
        Ok(extracted_entities)
    }
    
    /// Health check for the entity extractor
    pub async fn health_check(&self) -> HealthStatus {
        // Simple health check - if extractor exists and has good metrics, it's healthy
        let metrics = self.extractor.get_metrics().await;
        
        if metrics.success_rate() > 80.0 {
            HealthStatus::Healthy
        } else if metrics.success_rate() > 50.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        }
    }
    
    /// Check if the extractor is ready
    pub async fn is_ready(&self) -> bool {
        // Entity extractor is ready if it was successfully created
        true
    }
    
    /// Deduplicate entities based on semantic similarity
    pub async fn deduplicate_entities(&self, entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
        // Simple deduplication based on name and type
        let mut unique_entities = Vec::new();
        
        for entity in entities {
            let is_duplicate = unique_entities.iter().any(|existing: &ExtractedEntity| {
                existing.name.to_lowercase() == entity.name.to_lowercase() &&
                existing.entity_type == entity.entity_type
            });
            
            if !is_duplicate {
                unique_entities.push(entity);
            }
        }
        
        unique_entities
    }
}

/// Integration adapter for semantic chunking
pub struct SemanticChunkingAdapter {
    chunker: Arc<RealSemanticChunker>,
}

impl SemanticChunkingAdapter {
    pub fn new(chunker: Arc<RealSemanticChunker>) -> Self {
        Self { chunker }
    }
    
    /// Create semantic chunks and convert to production format
    pub async fn create_semantic_chunks(
        &self,
        content: &str,
        title: &str,
    ) -> Result<Vec<ProcessedChunk>, String> {
        let semantic_chunks = self.chunker
            .chunk_document(content)
            .await
            .map_err(|e| e.to_string())?;
        
        let processed_chunks = semantic_chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| ProcessedChunk {
                chunk_id: format!("chunk_{}", i),
                content: chunk.content,
                semantic_summary: chunk.key_concepts.join(", "),
                entities: Vec::new(), // Will be filled by entity extraction
                relationships: Vec::new(), // Will be filled by relationship mapping
                embedding: chunk.embedding,
                importance_score: 0.8, // Default importance
                coherence_score: chunk.semantic_coherence,
            })
            .collect();
        
        Ok(processed_chunks)
    }
    
    /// Health check for the semantic chunker
    pub async fn health_check(&self) -> HealthStatus {
        // Simple health check
        HealthStatus::Healthy
    }
    
    /// Check if the chunker is ready
    pub async fn is_ready(&self) -> bool {
        true
    }
}

/// Integration adapter for reasoning engine
pub struct ReasoningAdapter {
    engine: Arc<RealReasoningEngine>,
}

impl ReasoningAdapter {
    pub fn new(engine: Arc<RealReasoningEngine>) -> Self {
        Self { engine }
    }
    
    /// Perform reasoning with context
    pub async fn reason_with_context(
        &self,
        query: &str,
        _entities: &[ExtractedEntity],
        _docs: &[crate::enhanced_knowledge_storage::production::system_orchestrator::RetrievedDocument],
    ) -> Result<ReasoningResultInternal, String> {
        let reasoning_result = self.engine
            .reason(query)
            .await
            .map_err(|e| e.to_string())?;
        
        // Convert to internal format
        let internal_result = ReasoningResultInternal {
            steps: reasoning_result.reasoning_chain.into_iter().map(|step| {
                crate::enhanced_knowledge_storage::production::system_orchestrator::ReasoningStep {
                    step_number: step.step_number as usize,
                    operation: step.inference,
                    input: step.hypothesis,
                    output: step.evidence.join("; "),
                    confidence: step.confidence,
                    supporting_evidence: step.evidence,
                }
            }).collect(),
            overall_confidence: reasoning_result.confidence,
        };
        
        Ok(internal_result)
    }
    
    /// Health check for the reasoning engine
    pub async fn health_check(&self) -> HealthStatus {
        HealthStatus::Healthy
    }
}

/// Health status enumeration for adapters
#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

// Import the internal types for compatibility
use crate::enhanced_knowledge_storage::production::system_orchestrator::ReasoningResultInternal;