//! Knowledge Layers Management
//! 
//! Manages the creation, organization, and manipulation of hierarchical
//! knowledge layers from processed document content.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::Duration;
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    knowledge_processing::types::*,
    hierarchical_storage::types::*,
};

/// Manager for creating and organizing knowledge layers
pub struct KnowledgeLayerManager {
    model_manager: Arc<ModelResourceManager>,
    config: HierarchicalStorageConfig,
}

impl KnowledgeLayerManager {
    /// Create new knowledge layer manager
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        config: HierarchicalStorageConfig,
    ) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Create hierarchical layers from processed knowledge
    pub async fn create_hierarchical_layers(
        &self,
        processing_result: &KnowledgeProcessingResult,
    ) -> HierarchicalStorageResult<Vec<KnowledgeLayer>> {
        let mut layers = Vec::new();
        
        // Step 1: Create document layer (top level)
        let document_layer = self.create_document_layer(processing_result).await?;
        layers.push(document_layer);
        
        // Step 2: Create section layers from document structure
        let section_layers = self.create_section_layers(processing_result).await?;
        layers.extend(section_layers);
        
        // Step 3: Create paragraph layers from chunks
        let paragraph_layers = self.create_paragraph_layers(processing_result).await?;
        let paragraph_count = paragraph_layers.len();
        layers.extend(paragraph_layers);
        
        // Step 4: Create sentence layers within paragraphs
        let sentence_layers = self.create_sentence_layers(&layers[layers.len() - paragraph_count..]).await?;
        layers.extend(sentence_layers);
        
        // Step 5: Create entity-focused layers
        let entity_layers = self.create_entity_layers(processing_result).await?;
        layers.extend(entity_layers);
        
        // Step 6: Create relationship-focused layers
        let relationship_layers = self.create_relationship_layers(processing_result).await?;
        layers.extend(relationship_layers);
        
        // Step 7: Create concept layers for thematic groupings
        let concept_layers = self.create_concept_layers(processing_result).await?;
        layers.extend(concept_layers);
        
        // Step 8: Calculate importance scores and semantic embeddings
        self.enhance_layers_with_analysis(&mut layers).await?;
        
        // Step 9: Establish parent-child relationships
        self.establish_layer_hierarchy(&mut layers)?;
        
        Ok(layers)
    }
    
    /// Create the top-level document layer
    async fn create_document_layer(
        &self,
        processing_result: &KnowledgeProcessingResult,
    ) -> HierarchicalStorageResult<KnowledgeLayer> {
        let document_content = processing_result.chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        
        let layer_content = LayerContent {
            raw_text: document_content.clone(),
            processed_text: document_content.clone(),
            key_phrases: processing_result.global_entities
                .iter()
                .take(10)
                .map(|e| e.name.clone())
                .collect(),
            summary: processing_result.document_structure.overall_topic.clone(),
            metadata: LayerMetadata {
                word_count: document_content.split_whitespace().count(),
                character_count: document_content.len(),
                complexity_level: processing_result.document_structure.complexity_level.clone(),
                reading_time: processing_result.document_structure.estimated_reading_time,
                tags: processing_result.document_structure.key_themes.clone(),
                custom_attributes: HashMap::new(),
            },
        };
        
        Ok(KnowledgeLayer {
            layer_id: format!("{}_document", processing_result.document_id),
            layer_type: LayerType::Document,
            parent_layer_id: None,
            child_layer_ids: Vec::new(), // Will be populated later
            content: layer_content,
            entities: processing_result.global_entities.clone(),
            relationships: processing_result.global_relationships.clone(),
            semantic_embedding: None, // Will be calculated later
            importance_score: 1.0, // Document layer has maximum importance
            coherence_score: processing_result.quality_metrics.semantic_coherence,
            position: LayerPosition {
                start_offset: 0,
                end_offset: document_content.len(),
                depth_level: 0,
                sequence_number: 0,
            },
        })
    }
    
    /// Create section layers from document structure
    async fn create_section_layers(
        &self,
        processing_result: &KnowledgeProcessingResult,
    ) -> HierarchicalStorageResult<Vec<KnowledgeLayer>> {
        let mut layers = Vec::new();
        
        for (i, section) in processing_result.document_structure.sections.iter().enumerate() {
            let section_content = if section.end_pos > section.start_pos {
                // Extract content from original document based on positions
                // For now, we'll use a placeholder approach
                format!("Section content from {} to {}", section.start_pos, section.end_pos)
            } else {
                "Section content not available".to_string()
            };
            
            let layer_content = LayerContent {
                raw_text: section_content.clone(),
                processed_text: section_content.clone(),
                key_phrases: section.key_points.clone(),
                summary: section.title.clone(),
                metadata: LayerMetadata {
                    word_count: section_content.split_whitespace().count(),
                    character_count: section_content.len(),
                    complexity_level: ComplexityLevel::Medium,
                    reading_time: Duration::from_secs((section_content.len() / 1000) as u64 * 30),
                    tags: vec![format!("{:?}", section.section_type)],
                    custom_attributes: HashMap::new(),
                },
            };
            
            let layer = KnowledgeLayer {
                layer_id: format!("{}_section_{}", processing_result.document_id, i),
                layer_type: LayerType::Section,
                parent_layer_id: Some(format!("{}_document", processing_result.document_id)),
                child_layer_ids: Vec::new(),
                content: layer_content,
                entities: Vec::new(), // Will be populated from relevant chunks
                relationships: Vec::new(),
                semantic_embedding: None,
                importance_score: 0.8, // High importance for sections
                coherence_score: 0.8,
                position: LayerPosition {
                    start_offset: section.start_pos,
                    end_offset: section.end_pos,
                    depth_level: 1,
                    sequence_number: i as u32,
                },
            };
            
            layers.push(layer);
        }
        
        Ok(layers)
    }
    
    /// Create paragraph layers from semantic chunks
    async fn create_paragraph_layers(
        &self,
        processing_result: &KnowledgeProcessingResult,
    ) -> HierarchicalStorageResult<Vec<KnowledgeLayer>> {
        let mut layers = Vec::new();
        
        for (i, chunk) in processing_result.chunks.iter().enumerate() {
            let layer_content = LayerContent {
                raw_text: chunk.content.clone(),
                processed_text: chunk.content.clone(),
                key_phrases: chunk.key_concepts.clone(),
                summary: if chunk.content.len() > 200 {
                    Some(format!("{}...", &chunk.content[..197]))
                } else {
                    Some(chunk.content.clone())
                },
                metadata: LayerMetadata {
                    word_count: chunk.content.split_whitespace().count(),
                    character_count: chunk.content.len(),
                    complexity_level: if chunk.semantic_coherence > 0.8 { 
                        ComplexityLevel::Low 
                    } else if chunk.semantic_coherence > 0.6 { 
                        ComplexityLevel::Medium 
                    } else { 
                        ComplexityLevel::High 
                    },
                    reading_time: Duration::from_secs((chunk.content.len() / 1000) as u64 * 15),
                    tags: vec![format!("{:?}", chunk.chunk_type)],
                    custom_attributes: HashMap::new(),
                },
            };
            
            // Find appropriate parent section
            let parent_section_id = self.find_parent_section_for_chunk(
                chunk, 
                &processing_result.document_structure.sections,
                &processing_result.document_id
            );
            
            let layer = KnowledgeLayer {
                layer_id: format!("{}_paragraph_{}", processing_result.document_id, i),
                layer_type: LayerType::Paragraph,
                parent_layer_id: parent_section_id,
                child_layer_ids: Vec::new(),
                content: layer_content,
                entities: chunk.entities.clone(),
                relationships: chunk.relationships.clone(),
                semantic_embedding: None,
                importance_score: chunk.semantic_coherence * 0.7, // Base on coherence
                coherence_score: chunk.semantic_coherence,
                position: LayerPosition {
                    start_offset: chunk.start_pos,
                    end_offset: chunk.end_pos,
                    depth_level: 2,
                    sequence_number: i as u32,
                },
            };
            
            layers.push(layer);
        }
        
        Ok(layers)
    }
    
    /// Create sentence layers within paragraphs
    async fn create_sentence_layers(
        &self,
        paragraph_layers: &[KnowledgeLayer],
    ) -> HierarchicalStorageResult<Vec<KnowledgeLayer>> {
        let mut layers = Vec::new();
        
        for paragraph in paragraph_layers {
            if paragraph.layer_type != LayerType::Paragraph {
                continue;
            }
            
            let sentences = self.split_into_sentences(&paragraph.content.raw_text);
            let mut current_offset = paragraph.position.start_offset;
            
            for (i, sentence) in sentences.iter().enumerate() {
                if sentence.trim().is_empty() {
                    continue;
                }
                
                let layer_content = LayerContent {
                    raw_text: sentence.clone(),
                    processed_text: sentence.clone(),
                    key_phrases: self.extract_key_phrases_from_sentence(sentence),
                    summary: Some(sentence.clone()),
                    metadata: LayerMetadata {
                        word_count: sentence.split_whitespace().count(),
                        character_count: sentence.len(),
                        complexity_level: ComplexityLevel::Low,
                        reading_time: Duration::from_secs((sentence.len() / 1000) as u64 * 5),
                        tags: Vec::new(),
                        custom_attributes: HashMap::new(),
                    },
                };
                
                // Extract entities that appear in this sentence
                let sentence_entities = paragraph.entities
                    .iter()
                    .filter(|entity| sentence.contains(&entity.name))
                    .cloned()
                    .collect();
                
                let layer = KnowledgeLayer {
                    layer_id: format!("{}_sentence_{}", paragraph.layer_id, i),
                    layer_type: LayerType::Sentence,
                    parent_layer_id: Some(paragraph.layer_id.clone()),
                    child_layer_ids: Vec::new(),
                    content: layer_content,
                    entities: sentence_entities,
                    relationships: Vec::new(), // Relationships are typically at paragraph level
                    semantic_embedding: None,
                    importance_score: 0.3, // Lower importance for individual sentences
                    coherence_score: 0.7,
                    position: LayerPosition {
                        start_offset: current_offset,
                        end_offset: current_offset + sentence.len(),
                        depth_level: 3,
                        sequence_number: i as u32,
                    },
                };
                
                current_offset += sentence.len() + 1; // +1 for separator
                layers.push(layer);
            }
        }
        
        Ok(layers)
    }
    
    /// Create entity-focused layers
    async fn create_entity_layers(
        &self,
        processing_result: &KnowledgeProcessingResult,
    ) -> HierarchicalStorageResult<Vec<KnowledgeLayer>> {
        let mut layers = Vec::new();
        
        for (i, entity) in processing_result.global_entities.iter().enumerate() {
            // Collect all contexts where this entity appears
            let contexts: Vec<String> = processing_result.chunks
                .iter()
                .filter(|chunk| chunk.entities.iter().any(|e| e.name == entity.name))
                .map(|chunk| chunk.content.clone())
                .collect();
            
            let combined_context = contexts.join("\n---\n");
            
            let layer_content = LayerContent {
                raw_text: combined_context.clone(),
                processed_text: combined_context.clone(),
                key_phrases: vec![entity.name.clone()],
                summary: Some(format!("All contexts for entity: {}", entity.name)),
                metadata: LayerMetadata {
                    word_count: combined_context.split_whitespace().count(),
                    character_count: combined_context.len(),
                    complexity_level: ComplexityLevel::Medium,
                    reading_time: Duration::from_secs((combined_context.len() / 1000) as u64 * 10),
                    tags: vec![entity.entity_type.to_string()],
                    custom_attributes: entity.attributes.clone(),
                },
            };
            
            let layer = KnowledgeLayer {
                layer_id: format!("{}_entity_{}_{}", processing_result.document_id, entity.entity_type.to_string().to_lowercase(), i),
                layer_type: LayerType::Entity,
                parent_layer_id: Some(format!("{}_document", processing_result.document_id)),
                child_layer_ids: Vec::new(),
                content: layer_content,
                entities: vec![entity.clone()],
                relationships: processing_result.global_relationships
                    .iter()
                    .filter(|rel| rel.source == entity.name || rel.target == entity.name)
                    .cloned()
                    .collect(),
                semantic_embedding: None,
                importance_score: entity.confidence * 0.6,
                coherence_score: entity.confidence,
                position: LayerPosition {
                    start_offset: 0,
                    end_offset: combined_context.len(),
                    depth_level: 4,
                    sequence_number: i as u32,
                },
            };
            
            layers.push(layer);
        }
        
        Ok(layers)
    }
    
    /// Create relationship-focused layers
    async fn create_relationship_layers(
        &self,
        processing_result: &KnowledgeProcessingResult,
    ) -> HierarchicalStorageResult<Vec<KnowledgeLayer>> {
        let mut layers = Vec::new();
        
        for (i, relationship) in processing_result.global_relationships.iter().enumerate() {
            let relationship_description = format!(
                "{} {} {}",
                relationship.source,
                relationship.predicate.to_string(),
                relationship.target
            );
            
            let layer_content = LayerContent {
                raw_text: relationship.context.clone(),
                processed_text: relationship.context.clone(),
                key_phrases: vec![
                    relationship.source.clone(),
                    relationship.target.clone(),
                    relationship.predicate.to_string(),
                ],
                summary: Some(relationship_description.clone()),
                metadata: LayerMetadata {
                    word_count: relationship.context.split_whitespace().count(),
                    character_count: relationship.context.len(),
                    complexity_level: ComplexityLevel::Medium,
                    reading_time: Duration::from_secs((relationship.context.len() / 1000) as u64 * 8),
                    tags: vec![relationship.predicate.to_string()],
                    custom_attributes: HashMap::new(),
                },
            };
            
            let layer = KnowledgeLayer {
                layer_id: format!("{}_relationship_{}_{}", processing_result.document_id, relationship.predicate.to_string().to_lowercase(), i),
                layer_type: LayerType::Relationship,
                parent_layer_id: Some(format!("{}_document", processing_result.document_id)),
                child_layer_ids: Vec::new(),
                content: layer_content,
                entities: Vec::new(),
                relationships: vec![relationship.clone()],
                semantic_embedding: None,
                importance_score: relationship.confidence * relationship.relationship_strength * 0.5,
                coherence_score: relationship.confidence,
                position: LayerPosition {
                    start_offset: 0,
                    end_offset: relationship.context.len(),
                    depth_level: 4,
                    sequence_number: i as u32,
                },
            };
            
            layers.push(layer);
        }
        
        Ok(layers)
    }
    
    /// Create concept layers for thematic groupings
    async fn create_concept_layers(
        &self,
        processing_result: &KnowledgeProcessingResult,
    ) -> HierarchicalStorageResult<Vec<KnowledgeLayer>> {
        let mut layers = Vec::new();
        
        for (i, theme) in processing_result.document_structure.key_themes.iter().enumerate() {
            // Find chunks related to this theme
            let related_chunks: Vec<&SemanticChunk> = processing_result.chunks
                .iter()
                .filter(|chunk| {
                    chunk.key_concepts.iter().any(|concept| 
                        concept.to_lowercase().contains(&theme.to_lowercase()) ||
                        theme.to_lowercase().contains(&concept.to_lowercase())
                    )
                })
                .collect();
            
            if related_chunks.is_empty() {
                continue;
            }
            
            let combined_content = related_chunks
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            
            let layer_content = LayerContent {
                raw_text: combined_content.clone(),
                processed_text: combined_content.clone(),
                key_phrases: vec![theme.clone()],
                summary: Some(format!("Conceptual grouping for theme: {}", theme)),
                metadata: LayerMetadata {
                    word_count: combined_content.split_whitespace().count(),
                    character_count: combined_content.len(),
                    complexity_level: ComplexityLevel::Medium,
                    reading_time: Duration::from_secs((combined_content.len() / 1000) as u64 * 12),
                    tags: vec![theme.clone()],
                    custom_attributes: HashMap::new(),
                },
            };
            
            let layer = KnowledgeLayer {
                layer_id: format!("{}_concept_{}_{}", processing_result.document_id, theme.to_lowercase().replace(" ", "_"), i),
                layer_type: LayerType::Concept,
                parent_layer_id: Some(format!("{}_document", processing_result.document_id)),
                child_layer_ids: Vec::new(),
                content: layer_content,
                entities: related_chunks
                    .iter()
                    .flat_map(|chunk| &chunk.entities)
                    .cloned()
                    .collect(),
                relationships: related_chunks
                    .iter()
                    .flat_map(|chunk| &chunk.relationships)
                    .cloned()
                    .collect(),
                semantic_embedding: None,
                importance_score: 0.6,
                coherence_score: 0.7,
                position: LayerPosition {
                    start_offset: 0,
                    end_offset: combined_content.len(),
                    depth_level: 3,
                    sequence_number: i as u32,
                },
            };
            
            layers.push(layer);
        }
        
        Ok(layers)
    }
    
    /// Enhance layers with semantic analysis
    async fn enhance_layers_with_analysis(
        &self,
        layers: &mut [KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        for layer in layers.iter_mut() {
            // Calculate importance score based on content analysis
            layer.importance_score = self.calculate_importance_score(layer).await?;
            
            // Generate semantic embedding (placeholder - would use actual embedding model)
            layer.semantic_embedding = Some(self.generate_semantic_embedding(&layer.content.processed_text).await?);
            
            // Update coherence score based on enhanced analysis
            layer.coherence_score = self.calculate_coherence_score(layer).await?;
        }
        
        Ok(())
    }
    
    /// Establish parent-child relationships between layers
    fn establish_layer_hierarchy(
        &self,
        layers: &mut [KnowledgeLayer],
    ) -> HierarchicalStorageResult<()> {
        // Create lookup map for layers by ID
        let mut layer_map: HashMap<String, usize> = HashMap::new();
        for (i, layer) in layers.iter().enumerate() {
            layer_map.insert(layer.layer_id.clone(), i);
        }
        
        // Update child_layer_ids for parent layers
        for i in 0..layers.len() {
            if let Some(parent_id) = &layers[i].parent_layer_id.clone() {
                if let Some(&parent_index) = layer_map.get(parent_id) {
                    layers[parent_index].child_layer_ids.push(layers[i].layer_id.clone());
                }
            }
        }
        
        Ok(())
    }
    
    // Helper methods
    
    /// Find appropriate parent section for a chunk
    fn find_parent_section_for_chunk(
        &self,
        chunk: &SemanticChunk,
        sections: &[DocumentSection],
        document_id: &str,
    ) -> Option<String> {
        for (i, section) in sections.iter().enumerate() {
            if chunk.start_pos >= section.start_pos && chunk.end_pos <= section.end_pos {
                return Some(format!("{}_section_{}", document_id, i));
            }
        }
        // Fallback to document layer
        Some(format!("{}_document", document_id))
    }
    
    /// Split text into sentences
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        // Simple sentence splitting - could be enhanced with NLP
        text.split(&['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
    
    /// Extract key phrases from a sentence
    fn extract_key_phrases_from_sentence(&self, sentence: &str) -> Vec<String> {
        // Simple approach - extract longer words
        sentence
            .split_whitespace()
            .filter(|word| word.len() > 4)
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty())
            .collect()
    }
    
    /// Calculate importance score for a layer
    async fn calculate_importance_score(&self, layer: &KnowledgeLayer) -> HierarchicalStorageResult<f32> {
        let mut score = 0.0;
        
        // Base score on layer type
        score += match layer.layer_type {
            LayerType::Document => 1.0,
            LayerType::Section => 0.8,
            LayerType::Paragraph => 0.6,
            LayerType::Sentence => 0.3,
            LayerType::Entity => 0.7,
            LayerType::Relationship => 0.5,
            LayerType::Concept => 0.6,
        };
        
        // Adjust based on content length
        let content_length_factor = (layer.content.raw_text.len() as f32 / 1000.0).min(1.0);
        score *= 0.7 + 0.3 * content_length_factor;
        
        // Adjust based on entity and relationship count
        let entity_factor = (layer.entities.len() as f32 / 10.0).min(1.0);
        let relationship_factor = (layer.relationships.len() as f32 / 5.0).min(1.0);
        score *= 0.6 + 0.2 * entity_factor + 0.2 * relationship_factor;
        
        Ok(score.min(1.0))
    }
    
    /// Generate semantic embedding for text
    async fn generate_semantic_embedding(&self, text: &str) -> HierarchicalStorageResult<Vec<f32>> {
        // Placeholder implementation - would use actual embedding model
        let prompt = format!("Generate semantic representation for: {}", &text[..text.len().min(500)]);
        let task = ProcessingTask::new(ComplexityLevel::Low, &prompt);
        
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| HierarchicalStorageError::SemanticAnalysisError(e.to_string()))?;
        
        // Simple hash-based embedding (would be replaced with actual embeddings)
        let mut embedding = Vec::new();
        let hash_seed = result.output.len() % 256;
        for i in 0..384 { // 384-dimensional embedding
            let value = ((hash_seed + i) as f32 / 256.0) * 2.0 - 1.0;
            embedding.push(value);
        }
        
        Ok(embedding)
    }
    
    /// Calculate coherence score for a layer
    async fn calculate_coherence_score(&self, layer: &KnowledgeLayer) -> HierarchicalStorageResult<f32> {
        // Base coherence on existing score
        let mut coherence = layer.coherence_score;
        
        // Adjust based on entities and relationships consistency
        if !layer.entities.is_empty() {
            let avg_entity_confidence = layer.entities
                .iter()
                .map(|e| e.confidence)
                .sum::<f32>() / layer.entities.len() as f32;
            coherence = (coherence + avg_entity_confidence) / 2.0;
        }
        
        if !layer.relationships.is_empty() {
            let avg_relationship_confidence = layer.relationships
                .iter()
                .map(|r| r.confidence)
                .sum::<f32>() / layer.relationships.len() as f32;
            coherence = (coherence + avg_relationship_confidence) / 2.0;
        }
        
        Ok(coherence.min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_knowledge_layer_manager_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = KnowledgeLayerManager::new(model_manager, storage_config);
        
        assert_eq!(manager.config.max_layers_per_document, 1000);
    }
    
    #[test]
    fn test_sentence_splitting() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = KnowledgeLayerManager::new(model_manager, storage_config);
        
        let text = "This is the first sentence. This is the second sentence! And this is the third?";
        let sentences = manager.split_into_sentences(text);
        
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "This is the first sentence");
        assert_eq!(sentences[1], "This is the second sentence");
        assert_eq!(sentences[2], "And this is the third");
    }
    
    #[test]
    fn test_key_phrase_extraction() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let storage_config = HierarchicalStorageConfig::default();
        
        let manager = KnowledgeLayerManager::new(model_manager, storage_config);
        
        let sentence = "The artificial intelligence algorithm processes natural language effectively.";
        let phrases = manager.extract_key_phrases_from_sentence(sentence);
        
        assert!(phrases.contains(&"artificial".to_string()));
        assert!(phrases.contains(&"intelligence".to_string()));
        assert!(phrases.contains(&"algorithm".to_string()));
        assert!(phrases.contains(&"processes".to_string()));
        assert!(phrases.contains(&"natural".to_string()));
        assert!(phrases.contains(&"language".to_string()));
        assert!(phrases.contains(&"effectively".to_string()));
    }
}