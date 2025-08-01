//! Context Analyzer
//! 
//! Analyzes and preserves context across processing steps to maintain
//! semantic coherence and prevent information loss at boundaries.

use std::sync::Arc;
use std::collections::HashMap;
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    knowledge_processing::types::*,
};

/// Context analyzer for maintaining semantic coherence
pub struct ContextAnalyzer {
    model_manager: Arc<ModelResourceManager>,
    config: ContextAnalysisConfig,
}

/// Configuration for context analysis
#[derive(Debug, Clone)]
pub struct ContextAnalysisConfig {
    pub model_id: String,
    pub context_window_size: usize,
    pub cross_reference_threshold: f32,
    pub coherence_threshold: f32,
    pub enable_global_context: bool,
    pub enable_local_context: bool,
}

impl Default for ContextAnalysisConfig {
    fn default() -> Self {
        Self {
            model_id: "smollm2_360m".to_string(),
            context_window_size: 512, // characters
            cross_reference_threshold: 0.6,
            coherence_threshold: 0.7,
            enable_global_context: true,
            enable_local_context: true,
        }
    }
}

/// Global context information for a document
#[derive(Debug, Clone)]
pub struct GlobalContext {
    pub document_theme: String,
    pub key_entities: Vec<String>,
    pub main_relationships: Vec<String>,
    pub conceptual_framework: Vec<String>,
    pub context_preservation_score: f32,
}

/// Cross-reference between content pieces
#[derive(Debug, Clone)]
pub struct CrossReference {
    pub source_id: String,
    pub target_id: String,
    pub reference_type: CrossReferenceType,
    pub confidence: f32,
    pub bridging_content: Option<String>,
}

/// Types of cross-references
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossReferenceType {
    EntityMention,      // Same entity mentioned in different chunks
    ConceptualLink,     // Related concepts across chunks
    CausalConnection,   // Cause-effect relationship spans chunks
    TemporalSequence,   // Time-based sequence across chunks
    ArgumentFlow,       // Logical argument continues across chunks
    DefinitionUsage,    // Definition in one chunk, usage in another
}

impl ContextAnalyzer {
    /// Create new context analyzer
    pub fn new(model_manager: Arc<ModelResourceManager>, config: ContextAnalysisConfig) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Analyze global context of a document
    pub async fn analyze_global_context(&self, content: &str, title: &str) -> KnowledgeProcessingResult2<GlobalContext> {
        if !self.config.enable_global_context {
            return Ok(GlobalContext {
                document_theme: title.to_string(),
                key_entities: Vec::new(),
                main_relationships: Vec::new(),
                conceptual_framework: Vec::new(),
                context_preservation_score: 0.5,
            });
        }
        
        let prompt = format!(
            r#"Analyze this document for global context information. Provide:

1. Main document theme (1-2 sentences)
2. Key entities (5-10 most important entities)
3. Main relationships (3-5 core relationships)
4. Conceptual framework (key concepts and ideas)

Return as JSON:
{{
  "document_theme": "brief description of main theme",
  "key_entities": ["entity1", "entity2", "entity3"],
  "main_relationships": ["relationship1", "relationship2"],
  "conceptual_framework": ["concept1", "concept2", "concept3"]
}}

Title: {title}

Document:
{content}

JSON Response:"#
        );
        
        let task = ProcessingTask::new(ComplexityLevel::High, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| KnowledgeProcessingError::ModelError(e.to_string()))?;
        
        self.parse_global_context_response(&result.output)
    }
    
    /// Parse global context response from model
    fn parse_global_context_response(&self, response: &str) -> KnowledgeProcessingResult2<GlobalContext> {
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(KnowledgeProcessingError::JsonError)?;
        
        let document_theme = parsed["document_theme"].as_str().unwrap_or("").to_string();
        
        let key_entities = if let Some(entities_array) = parsed["key_entities"].as_array() {
            entities_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };
        
        let main_relationships = if let Some(rels_array) = parsed["main_relationships"].as_array() {
            rels_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };
        
        let conceptual_framework = if let Some(concepts_array) = parsed["conceptual_framework"].as_array() {
            concepts_array
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };
        
        // Calculate context preservation score based on content richness
        let context_preservation_score = self.calculate_context_preservation_score(
            &key_entities,
            &main_relationships,
            &conceptual_framework,
        );
        
        Ok(GlobalContext {
            document_theme,
            key_entities,
            main_relationships,
            conceptual_framework,
            context_preservation_score,
        })
    }
    
    /// Calculate context preservation score
    fn calculate_context_preservation_score(
        &self,
        entities: &[String],
        relationships: &[String],
        concepts: &[String],
    ) -> f32 {
        let entity_score = (entities.len() as f32 / 10.0).min(1.0) * 0.4;
        let relationship_score = (relationships.len() as f32 / 5.0).min(1.0) * 0.3;
        let concept_score = (concepts.len() as f32 / 5.0).min(1.0) * 0.3;
        
        entity_score + relationship_score + concept_score
    }
    
    /// Build cross-references between chunks
    pub async fn build_cross_references(
        &self,
        chunks: &[SemanticChunk],
        global_context: &GlobalContext,
    ) -> KnowledgeProcessingResult2<Vec<CrossReference>> {
        let mut cross_references = Vec::new();
        
        // Find entity-based cross-references
        let entity_refs = self.find_entity_cross_references(chunks, &global_context.key_entities);
        cross_references.extend(entity_refs);
        
        // Find conceptual cross-references
        let concept_refs = self.find_conceptual_cross_references(chunks, &global_context.conceptual_framework).await?;
        cross_references.extend(concept_refs);
        
        // Find temporal and causal cross-references
        let temporal_refs = self.find_temporal_cross_references(chunks).await?;
        cross_references.extend(temporal_refs);
        
        // Filter by confidence threshold
        cross_references.retain(|cr| cr.confidence >= self.config.cross_reference_threshold);
        
        Ok(cross_references)
    }
    
    /// Find entity-based cross-references
    fn find_entity_cross_references(
        &self,
        chunks: &[SemanticChunk],
        key_entities: &[String],
    ) -> Vec<CrossReference> {
        let mut cross_references = Vec::new();
        
        // Track which chunks contain which entities
        let mut entity_chunks: HashMap<String, Vec<String>> = HashMap::new();
        
        for chunk in chunks {
            for entity in key_entities {
                if chunk.content.contains(entity) || 
                   chunk.content.to_lowercase().contains(&entity.to_lowercase()) {
                    entity_chunks.entry(entity.clone()).or_default().push(chunk.id.clone());
                }
            }
        }
        
        // Create cross-references for shared entities
        for (entity, chunk_ids) in entity_chunks {
            if chunk_ids.len() > 1 {
                for i in 0..chunk_ids.len() {
                    for j in i + 1..chunk_ids.len() {
                        cross_references.push(CrossReference {
                            source_id: chunk_ids[i].clone(),
                            target_id: chunk_ids[j].clone(),
                            reference_type: CrossReferenceType::EntityMention,
                            confidence: 0.8, // High confidence for direct entity matches
                            bridging_content: Some(entity.clone()),
                        });
                    }
                }
            }
        }
        
        cross_references
    }
    
    /// Find conceptual cross-references using AI analysis
    async fn find_conceptual_cross_references(
        &self,
        chunks: &[SemanticChunk],
        conceptual_framework: &[String],
    ) -> KnowledgeProcessingResult2<Vec<CrossReference>> {
        if chunks.len() < 2 {
            return Ok(Vec::new());
        }
        
        let mut cross_references = Vec::new();
        
        // Analyze pairs of chunks for conceptual connections
        for i in 0..chunks.len() {
            for j in i + 1..chunks.len() {
                if let Some(cross_ref) = self.analyze_chunk_pair_for_concepts(
                    &chunks[i],
                    &chunks[j],
                    conceptual_framework,
                ).await? {
                    cross_references.push(cross_ref);
                }
            }
        }
        
        Ok(cross_references)
    }
    
    /// Analyze a pair of chunks for conceptual connections
    async fn analyze_chunk_pair_for_concepts(
        &self,
        chunk1: &SemanticChunk,
        chunk2: &SemanticChunk,
        concepts: &[String],
    ) -> KnowledgeProcessingResult2<Option<CrossReference>> {
        let prompt = format!(
            r#"Analyze these two text chunks for conceptual connections. Consider:

Key concepts to look for: {concepts:?}

Determine if there are meaningful conceptual links between the chunks such as:
- Related ideas or themes
- Cause and effect relationships
- Definition and usage
- Argument flow or logical progression

If a strong connection exists, provide:
{{
  "has_connection": true,
  "connection_type": "ConceptualLink|CausalConnection|DefinitionUsage|ArgumentFlow",
  "confidence": 0.75,
  "bridging_content": "description of the connection"
}}

If no strong connection exists:
{{
  "has_connection": false
}}

Chunk 1:
{chunk1_content}

Chunk 2:
{chunk2_content}

JSON Response:"#,
            concepts = concepts,
            chunk1_content = &chunk1.content[..chunk1.content.len().min(500)], // Limit for prompt size
            chunk2_content = &chunk2.content[..chunk2.content.len().min(500)],
        );
        
        let task = ProcessingTask::new(ComplexityLevel::Medium, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| KnowledgeProcessingError::ModelError(e.to_string()))?;
        
        self.parse_conceptual_connection_response(&result.output, &chunk1.id, &chunk2.id)
    }
    
    /// Parse conceptual connection response
    fn parse_conceptual_connection_response(
        &self,
        response: &str,
        chunk1_id: &str,
        chunk2_id: &str,
    ) -> KnowledgeProcessingResult2<Option<CrossReference>> {
        let json_start = response.find('{').unwrap_or(0);
        let json_end = response.rfind('}').map(|i| i + 1).unwrap_or(response.len());
        let json_str = &response[json_start..json_end];
        
        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(KnowledgeProcessingError::JsonError)?;
        
        let has_connection = parsed["has_connection"].as_bool().unwrap_or(false);
        
        if !has_connection {
            return Ok(None);
        }
        
        let connection_type_str = parsed["connection_type"].as_str().unwrap_or("ConceptualLink");
        let confidence = parsed["confidence"].as_f64().unwrap_or(0.5) as f32;
        let bridging_content = parsed["bridging_content"].as_str().map(|s| s.to_string());
        
        let reference_type = match connection_type_str {
            "ConceptualLink" => CrossReferenceType::ConceptualLink,
            "CausalConnection" => CrossReferenceType::CausalConnection,
            "DefinitionUsage" => CrossReferenceType::DefinitionUsage,
            "ArgumentFlow" => CrossReferenceType::ArgumentFlow,
            _ => CrossReferenceType::ConceptualLink,
        };
        
        Ok(Some(CrossReference {
            source_id: chunk1_id.to_string(),
            target_id: chunk2_id.to_string(),
            reference_type,
            confidence,
            bridging_content,
        }))
    }
    
    /// Find temporal cross-references
    async fn find_temporal_cross_references(
        &self,
        chunks: &[SemanticChunk],
    ) -> KnowledgeProcessingResult2<Vec<CrossReference>> {
        let mut cross_references = Vec::new();
        
        // Look for temporal indicators in adjacent chunks
        for i in 0..chunks.len().saturating_sub(1) {
            let chunk1 = &chunks[i];
            let chunk2 = &chunks[i + 1];
            
            if self.has_temporal_indicators(&chunk1.content) || self.has_temporal_indicators(&chunk2.content) {
                cross_references.push(CrossReference {
                    source_id: chunk1.id.clone(),
                    target_id: chunk2.id.clone(),
                    reference_type: CrossReferenceType::TemporalSequence,
                    confidence: 0.6, // Moderate confidence for temporal heuristics
                    bridging_content: Some("Temporal sequence detected".to_string()),
                });
            }
        }
        
        Ok(cross_references)
    }
    
    /// Check if content has temporal indicators
    fn has_temporal_indicators(&self, content: &str) -> bool {
        let temporal_words = [
            "then", "next", "after", "before", "during", "while", "when",
            "first", "second", "finally", "meanwhile", "subsequently",
            "earlier", "later", "previously", "following", "since"
        ];
        
        let lower_content = content.to_lowercase();
        temporal_words.iter().any(|&word| lower_content.contains(word))
    }
    
    /// Validate context preservation across chunks
    pub fn validate_context_preservation(
        &self,
        chunks: &[SemanticChunk],
        cross_references: &[CrossReference],
        global_context: &GlobalContext,
    ) -> ContextValidationResult {
        let total_chunks = chunks.len();
        let connected_chunks = self.count_connected_chunks(chunks, cross_references);
        let coverage_score = if total_chunks == 0 { 0.0 } else { connected_chunks as f32 / total_chunks as f32 };
        
        let avg_coherence = if chunks.is_empty() { 0.0 } else {
            chunks.iter().map(|c| c.semantic_coherence).sum::<f32>() / chunks.len() as f32
        };
        
        let cross_reference_density = if total_chunks <= 1 { 0.0 } else {
            cross_references.len() as f32 / (total_chunks * (total_chunks - 1) / 2) as f32
        };
        
        let context_preservation_score = (
            coverage_score * 0.4 +
            avg_coherence * 0.3 +
            cross_reference_density * 0.2 +
            global_context.context_preservation_score * 0.1
        ).min(1.0);
        
        ContextValidationResult {
            context_preservation_score,
            coverage_score,
            average_coherence: avg_coherence,
            cross_reference_density,
            connected_chunks_count: connected_chunks,
            total_chunks_count: total_chunks,
            validation_passed: context_preservation_score >= self.config.coherence_threshold,
        }
    }
    
    /// Count chunks that have cross-references
    fn count_connected_chunks(&self, _chunks: &[SemanticChunk], cross_references: &[CrossReference]) -> usize {
        let mut connected_chunk_ids = std::collections::HashSet::new();
        
        for cross_ref in cross_references {
            connected_chunk_ids.insert(&cross_ref.source_id);
            connected_chunk_ids.insert(&cross_ref.target_id);
        }
        
        connected_chunk_ids.len()
    }
    
    /// Get context analysis statistics
    pub fn get_context_stats(
        &self,
        cross_references: &[CrossReference],
        validation_result: &ContextValidationResult,
    ) -> ContextAnalysisStats {
        let mut type_counts = HashMap::new();
        let mut total_confidence = 0.0;
        
        for cross_ref in cross_references {
            *type_counts.entry(format!("{:?}", cross_ref.reference_type)).or_insert(0) += 1;
            total_confidence += cross_ref.confidence;
        }
        
        let average_confidence = if cross_references.is_empty() { 0.0 } else {
            total_confidence / cross_references.len() as f32
        };
        
        ContextAnalysisStats {
            total_cross_references: cross_references.len(),
            average_confidence,
            reference_type_distribution: type_counts,
            context_preservation_score: validation_result.context_preservation_score,
            coverage_score: validation_result.coverage_score,
            connected_chunks_ratio: validation_result.connected_chunks_count as f32 / validation_result.total_chunks_count.max(1) as f32,
        }
    }
}

/// Result of context validation
#[derive(Debug, Clone)]
pub struct ContextValidationResult {
    pub context_preservation_score: f32,
    pub coverage_score: f32,
    pub average_coherence: f32,
    pub cross_reference_density: f32,
    pub connected_chunks_count: usize,
    pub total_chunks_count: usize,
    pub validation_passed: bool,
}

/// Statistics about context analysis
#[derive(Debug, Clone)]
pub struct ContextAnalysisStats {
    pub total_cross_references: usize,
    pub average_confidence: f32,
    pub reference_type_distribution: HashMap<String, usize>,
    pub context_preservation_score: f32,
    pub coverage_score: f32,
    pub connected_chunks_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_context_analyzer_creation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config).await.unwrap());
        let context_config = ContextAnalysisConfig::default();
        
        let analyzer = ContextAnalyzer::new(model_manager, context_config);
        
        assert_eq!(analyzer.config.model_id, "smollm2_360m");
        assert!(analyzer.config.enable_global_context);
    }
    
    #[tokio::test]
    async fn test_context_preservation_score_calculation() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config).await.unwrap());
        let context_config = ContextAnalysisConfig::default();
        
        let analyzer = ContextAnalyzer::new(model_manager, context_config);
        
        let entities = vec!["Entity1".to_string(), "Entity2".to_string()];
        let relationships = vec!["Relationship1".to_string()];
        let concepts = vec!["Concept1".to_string()];
        
        let score = analyzer.calculate_context_preservation_score(&entities, &relationships, &concepts);
        
        // Score should be: (2/10)*0.4 + (1/5)*0.3 + (1/5)*0.3 = 0.08 + 0.06 + 0.06 = 0.2
        assert!((score - 0.2).abs() < 0.01);
    }
    
    #[tokio::test]
    async fn test_temporal_indicators_detection() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config).await.unwrap());
        let context_config = ContextAnalysisConfig::default();
        
        let analyzer = ContextAnalyzer::new(model_manager, context_config);
        
        assert!(analyzer.has_temporal_indicators("First, we need to understand this. Then, we can proceed."));
        assert!(analyzer.has_temporal_indicators("After the experiment, the results were clear."));
        assert!(!analyzer.has_temporal_indicators("This is a simple statement without temporal markers."));
    }
    
    #[tokio::test]
    async fn test_entity_cross_references() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config).await.unwrap());
        let context_config = ContextAnalysisConfig::default();
        
        let analyzer = ContextAnalyzer::new(model_manager, context_config);
        
        let chunks = vec![
            SemanticChunk {
                id: "chunk1".to_string(),
                content: "Einstein developed the theory of relativity.".to_string(),
                start_pos: 0,
                end_pos: 45,
                semantic_coherence: 0.8,
                key_concepts: Vec::new(),
                entities: Vec::new(),
                relationships: Vec::new(),
                chunk_type: ChunkType::Paragraph,
                overlap_with_previous: None,
                overlap_with_next: None,
            },
            SemanticChunk {
                id: "chunk2".to_string(),
                content: "Einstein was born in Germany.".to_string(),
                start_pos: 46,
                end_pos: 75,
                semantic_coherence: 0.7,
                key_concepts: Vec::new(),
                entities: Vec::new(),
                relationships: Vec::new(),
                chunk_type: ChunkType::Paragraph,
                overlap_with_previous: None,
                overlap_with_next: None,
            },
        ];
        
        let key_entities = vec!["Einstein".to_string()];
        let cross_refs = analyzer.find_entity_cross_references(&chunks, &key_entities);
        
        assert_eq!(cross_refs.len(), 1);
        assert_eq!(cross_refs[0].reference_type, CrossReferenceType::EntityMention);
        assert_eq!(cross_refs[0].bridging_content, Some("Einstein".to_string()));
    }
    
    #[tokio::test]
    async fn test_connected_chunks_counting() {
        let config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(config).await.unwrap());
        let context_config = ContextAnalysisConfig::default();
        
        let analyzer = ContextAnalyzer::new(model_manager, context_config);
        
        let chunks = vec![
            SemanticChunk {
                id: "chunk1".to_string(),
                content: "Content 1".to_string(),
                start_pos: 0,
                end_pos: 9,
                semantic_coherence: 0.8,
                key_concepts: Vec::new(),
                entities: Vec::new(),
                relationships: Vec::new(),
                chunk_type: ChunkType::Paragraph,
                overlap_with_previous: None,
                overlap_with_next: None,
            },
            SemanticChunk {
                id: "chunk2".to_string(),
                content: "Content 2".to_string(),
                start_pos: 10,
                end_pos: 19,
                semantic_coherence: 0.7,
                key_concepts: Vec::new(),
                entities: Vec::new(),
                relationships: Vec::new(),
                chunk_type: ChunkType::Paragraph,
                overlap_with_previous: None,
                overlap_with_next: None,
            },
        ];
        
        let cross_references = vec![
            CrossReference {
                source_id: "chunk1".to_string(),
                target_id: "chunk2".to_string(),
                reference_type: CrossReferenceType::ConceptualLink,
                confidence: 0.8,
                bridging_content: None,
            },
        ];
        
        let connected_count = analyzer.count_connected_chunks(&chunks, &cross_references);
        assert_eq!(connected_count, 2); // Both chunks are connected
    }
}