//! Context Aggregator
//! 
//! Aggregates and synthesizes context from multiple retrieved items
//! to provide comprehensive answers with proper context preservation.

use std::sync::Arc;
use std::collections::HashMap;
use crate::enhanced_knowledge_storage::{
    types::*,
    model_management::ModelResourceManager,
    retrieval_system::types::*,
};

/// Context aggregator for synthesizing retrieved information
pub struct ContextAggregator {
    model_manager: Arc<ModelResourceManager>,
    config: RetrievalConfig,
}

impl ContextAggregator {
    /// Create new context aggregator
    pub fn new(
        model_manager: Arc<ModelResourceManager>,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            model_manager,
            config,
        }
    }
    
    /// Aggregate context from retrieved items
    pub async fn aggregate_context(
        &self,
        retrieved_items: &[RetrievedItem],
        query: &ProcessedQuery,
        reasoning_chain: Option<&ReasoningChain>,
    ) -> RetrievalResult2<AggregatedContext> {
        // Step 1: Select primary content
        let primary_content = self.select_primary_content(retrieved_items, query).await?;
        
        // Step 2: Gather supporting contexts
        let supporting_contexts = self.gather_supporting_contexts(
            retrieved_items,
            &primary_content,
            query,
        ).await?;
        
        // Step 3: Create entity summary
        let entity_summary = self.create_entity_summary(retrieved_items);
        
        // Step 4: Create relationship summary
        let relationship_summary = self.create_relationship_summary(retrieved_items);
        
        // Step 5: Analyze temporal flow if applicable
        let temporal_flow = if query.understanding.temporal_context.is_some() {
            Some(self.analyze_temporal_flow(retrieved_items, query).await?)
        } else {
            None
        };
        
        // Step 6: Calculate coherence score
        let coherence_score = self.calculate_coherence_score(
            &primary_content,
            &supporting_contexts,
            reasoning_chain,
        );
        
        Ok(AggregatedContext {
            primary_content,
            supporting_contexts,
            entity_summary,
            relationship_summary,
            temporal_flow,
            coherence_score,
        })
    }
    
    /// Select the most relevant content as primary
    async fn select_primary_content(
        &self,
        items: &[RetrievedItem],
        query: &ProcessedQuery,
    ) -> RetrievalResult2<String> {
        if items.is_empty() {
            return Ok("No relevant content found.".to_string());
        }
        
        // Sort by relevance and importance
        let mut sorted_items = items.to_vec();
        sorted_items.sort_by(|a, b| {
            let score_a = a.relevance_score * a.importance_score;
            let score_b = b.relevance_score * b.importance_score;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Take top items for synthesis
        let top_items = sorted_items.iter().take(3).collect::<Vec<_>>();
        
        // Synthesize primary content
        self.synthesize_primary_content(&top_items, query).await
    }
    
    /// Synthesize primary content from top items
    async fn synthesize_primary_content(
        &self,
        items: &[&RetrievedItem],
        query: &ProcessedQuery,
    ) -> RetrievalResult2<String> {
        let contents = items
            .iter()
            .map(|item| item.content.as_str())
            .collect::<Vec<_>>();
        
        let prompt = format!(
            r#"Synthesize a comprehensive answer from these sources:

Query: "{}"

Sources:
{}

Create a coherent, complete answer that directly addresses the query.
Combine information naturally without repetition.

Synthesized answer:"#,
            query.original_query.natural_language_query,
            contents.join("\n\n---\n\n")
        );
        
        let task = ProcessingTask::new(ComplexityLevel::High, &prompt);
        let result = self.model_manager
            .process_with_optimal_model(task)
            .await
            .map_err(|e| RetrievalError::ContextAggregationError(e.to_string()))?;
        
        Ok(result.output.trim().to_string())
    }
    
    /// Gather supporting contexts
    async fn gather_supporting_contexts(
        &self,
        items: &[RetrievedItem],
        primary_content: &str,
        query: &ProcessedQuery,
    ) -> RetrievalResult2<Vec<SupportingContext>> {
        let mut supporting = Vec::new();
        
        for item in items.iter().skip(3).take(5) {
            // Calculate relevance to primary content
            let relevance = self.calculate_relevance_to_primary(
                &item.content,
                primary_content,
                query,
            ).await?;
            
            if relevance > 0.5 {
                // Extract relevant snippet
                let snippet = self.extract_relevant_snippet(
                    &item.content,
                    &query.understanding.extracted_entities,
                    200,
                );
                
                supporting.push(SupportingContext {
                    layer_id: item.layer_id.clone(),
                    content_snippet: snippet,
                    relevance_to_primary: relevance,
                    link_path: item.semantic_links
                        .iter()
                        .map(|link| link.link_type.clone())
                        .collect(),
                });
            }
        }
        
        Ok(supporting)
    }
    
    /// Calculate relevance to primary content
    async fn calculate_relevance_to_primary(
        &self,
        content: &str,
        primary: &str,
        query: &ProcessedQuery,
    ) -> RetrievalResult2<f32> {
        // Extract key terms from primary content
        let primary_terms = self.extract_key_terms(primary);
        let content_terms = self.extract_key_terms(content);
        
        // Calculate term overlap
        let overlap_count = primary_terms
            .iter()
            .filter(|term| content_terms.contains(term))
            .count();
        
        let term_overlap = if primary_terms.is_empty() {
            0.0
        } else {
            overlap_count as f32 / primary_terms.len() as f32
        };
        
        // Boost if contains query entities
        let entity_boost = query.understanding.extracted_entities
            .iter()
            .filter(|entity| content.to_lowercase().contains(&entity.to_lowercase()))
            .count() as f32 * 0.1;
        
        Ok((term_overlap + entity_boost).min(1.0))
    }
    
    /// Create entity summary from retrieved items
    fn create_entity_summary(&self, items: &[RetrievedItem]) -> HashMap<String, EntityContext> {
        let mut entity_map: HashMap<String, Vec<EntityOccurrence>> = HashMap::new();
        
        // Collect all entity occurrences
        for item in items {
            for entity in &item.match_explanation.matched_entities {
                entity_map.entry(entity.clone()).or_default().push(EntityOccurrence {
                    layer_id: item.layer_id.clone(),
                    context: self.extract_entity_context(&item.content, entity),
                    confidence: item.relevance_score,
                });
            }
        }
        
        // Create entity contexts
        let mut summary = HashMap::new();
        for (entity_name, occurrences) in entity_map {
            let importance = occurrences.len() as f32 / items.len() as f32;
            
            summary.insert(
                entity_name.clone(),
                EntityContext {
                    entity_name: entity_name.clone(),
                    entity_type: self.infer_entity_type(&entity_name),
                    occurrences,
                    relationships: Vec::new(), // Would extract from items
                    importance_in_context: importance,
                }
            );
        }
        
        summary
    }
    
    /// Create relationship summary
    fn create_relationship_summary(&self, items: &[RetrievedItem]) -> Vec<RelationshipContext> {
        let mut relationships = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        // Collect unique relationships
        for item in items {
            // Extract relationships from content (simplified)
            let extracted = self.extract_relationships_from_content(&item.content);
            
            for (source, predicate, target) in extracted {
                let key = format!("{source}-{predicate}-{target}");
                if !seen.contains(&key) {
                    seen.insert(key);
                    
                    relationships.push(RelationshipContext {
                        source: source.clone(),
                        predicate: predicate.clone(),
                        target: target.clone(),
                        occurrences: vec![item.layer_id.clone()],
                        strength: item.relevance_score,
                    });
                }
            }
        }
        
        relationships
    }
    
    /// Analyze temporal flow
    async fn analyze_temporal_flow(
        &self,
        items: &[RetrievedItem],
        _query: &ProcessedQuery,
    ) -> RetrievalResult2<TemporalFlow> {
        let mut events = Vec::new();
        
        // Extract temporal events from items
        for (i, item) in items.iter().enumerate() {
            if let Some(event) = self.extract_temporal_event(&item.content, i as u32) {
                events.push(event);
            }
        }
        
        // Sort events by sequence position
        events.sort_by_key(|e| e.sequence_position);
        
        // Calculate sequence confidence
        let sequence_confidence = if events.len() > 1 {
            0.7 // Would calculate based on temporal markers
        } else {
            0.3
        };
        
        Ok(TemporalFlow {
            events,
            sequence_confidence,
        })
    }
    
    /// Calculate coherence score
    fn calculate_coherence_score(
        &self,
        primary_content: &str,
        supporting_contexts: &[SupportingContext],
        reasoning_chain: Option<&ReasoningChain>,
    ) -> f32 {
        let mut score = 0.5; // Base score
        
        // Boost for having supporting contexts
        if !supporting_contexts.is_empty() {
            let avg_relevance = supporting_contexts
                .iter()
                .map(|ctx| ctx.relevance_to_primary)
                .sum::<f32>() / supporting_contexts.len() as f32;
            score += avg_relevance * 0.2;
        }
        
        // Boost for reasoning chain
        if let Some(chain) = reasoning_chain {
            score += chain.confidence * 0.3;
        }
        
        // Penalize if primary content is too short
        if primary_content.len() < 100 {
            score *= 0.8;
        }
        
        score.min(1.0)
    }
    
    // Helper methods
    
    /// Extract key terms from text
    fn extract_key_terms(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|word| word.len() > 4 && !self.is_stop_word(word))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }
    
    /// Check if word is stop word
    fn is_stop_word(&self, word: &str) -> bool {
        const STOP_WORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "been", "be",
        ];
        STOP_WORDS.contains(&word)
    }
    
    /// Extract relevant snippet around key terms
    fn extract_relevant_snippet(&self, content: &str, key_terms: &[String], max_length: usize) -> String {
        // Find first occurrence of any key term
        let content_lower = content.to_lowercase();
        let mut best_pos = None;
        
        for term in key_terms {
            if let Some(pos) = content_lower.find(&term.to_lowercase()) {
                best_pos = Some(pos);
                break;
            }
        }
        
        let start_pos = best_pos.unwrap_or(0);
        let snippet_start = start_pos.saturating_sub(50);
        let snippet_end = (start_pos + max_length).min(content.len());
        
        let snippet = &content[snippet_start..snippet_end];
        
        // Add ellipsis if truncated
        if snippet_start > 0 || snippet_end < content.len() {
            format!("...{}...", snippet.trim())
        } else {
            snippet.to_string()
        }
    }
    
    /// Extract entity context
    fn extract_entity_context(&self, content: &str, entity: &str) -> String {
        let content_lower = content.to_lowercase();
        let entity_lower = entity.to_lowercase();
        
        if let Some(pos) = content_lower.find(&entity_lower) {
            let start = pos.saturating_sub(30);
            let end = (pos + entity.len() + 30).min(content.len());
            content[start..end].to_string()
        } else {
            String::new()
        }
    }
    
    /// Infer entity type from name
    fn infer_entity_type(&self, entity_name: &str) -> EntityType {
        // Simple heuristics
        if entity_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
            EntityType::Person // Capitalized names often indicate people
        } else if entity_name.contains("ation") || entity_name.contains("ment") {
            EntityType::Organization
        } else {
            EntityType::Concept
        }
    }
    
    /// Extract relationships from content
    fn extract_relationships_from_content(&self, content: &str) -> Vec<(String, String, String)> {
        let mut relationships = Vec::new();
        
        // Simple pattern matching for relationships
        let patterns = [
            ("is", "is_a"),
            ("was", "was_a"),
            ("created", "created_by"),
            ("developed", "developed_by"),
            ("invented", "invented_by"),
            ("discovered", "discovered_by"),
        ];
        
        for (verb, predicate) in patterns {
            if content.contains(verb) {
                // Extract subject and object around the verb (simplified)
                let parts: Vec<&str> = content.split(verb).collect();
                if parts.len() >= 2 {
                    let subject = parts[0].split_whitespace().last().unwrap_or("").to_string();
                    let object = parts[1].split_whitespace().next().unwrap_or("").to_string();
                    
                    if !subject.is_empty() && !object.is_empty() {
                        relationships.push((subject, predicate.to_string(), object));
                    }
                }
            }
        }
        
        relationships
    }
    
    /// Extract temporal event from content
    fn extract_temporal_event(&self, content: &str, sequence_pos: u32) -> Option<TemporalEvent> {
        // Look for temporal markers
        let temporal_markers = ["before", "after", "during", "then", "next", "first", "finally"];
        
        let mut found_marker = None;
        for marker in temporal_markers {
            if content.to_lowercase().contains(marker) {
                found_marker = Some(marker.to_string());
                break;
            }
        }
        
        if found_marker.is_some() || sequence_pos == 0 {
            Some(TemporalEvent {
                description: self.extract_relevant_snippet(content, &[], 100),
                layer_ids: vec![], // Would be filled from item
                temporal_marker: found_marker,
                sequence_position: sequence_pos,
            })
        } else {
            None
        }
    }
}

// Import ProcessedQuery from query_processor
use crate::enhanced_knowledge_storage::retrieval_system::query_processor::ProcessedQuery;

// Import EntityType from knowledge processing
use crate::enhanced_knowledge_storage::knowledge_processing::types::EntityType;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_knowledge_storage::model_management::ModelResourceManager;
    
    #[tokio::test]
    async fn test_context_aggregator_creation() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let aggregator = ContextAggregator::new(model_manager, retrieval_config);
        
        assert!(aggregator.config.enable_result_reranking);
    }
    
    #[test]
    fn test_key_term_extraction() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let aggregator = ContextAggregator::new(model_manager, retrieval_config);
        
        let text = "The quantum physics experiment demonstrated wave-particle duality";
        let terms = aggregator.extract_key_terms(text);
        
        assert!(terms.contains(&"quantum".to_string()));
        assert!(terms.contains(&"physics".to_string()));
        assert!(terms.contains(&"experiment".to_string()));
        assert!(terms.contains(&"demonstrated".to_string()));
        assert!(terms.contains(&"wave-particle".to_string()));
        assert!(terms.contains(&"duality".to_string()));
    }
    
    #[test]
    fn test_snippet_extraction() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let aggregator = ContextAggregator::new(model_manager, retrieval_config);
        
        let content = "This is a very long content that contains many words before the important keyword appears here, and it continues with much more text after the keyword to ensure truncation happens. This should definitely be long enough to trigger the ellipsis functionality.";
        let key_terms = vec!["keyword".to_string()];
        
        let snippet = aggregator.extract_relevant_snippet(content, &key_terms, 50);
        
        assert!(snippet.contains("keyword"));
        assert!(snippet.contains("..."));
    }
    
    #[test]
    fn test_entity_type_inference() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let aggregator = ContextAggregator::new(model_manager, retrieval_config);
        
        assert_eq!(aggregator.infer_entity_type("Einstein"), EntityType::Person);
        assert_eq!(aggregator.infer_entity_type("organization"), EntityType::Organization);
        assert_eq!(aggregator.infer_entity_type("government"), EntityType::Organization);
        assert_eq!(aggregator.infer_entity_type("physics"), EntityType::Concept);
    }
    
    #[test]
    fn test_relationship_extraction() {
        let model_config = ModelResourceConfig::default();
        let model_manager = Arc::new(ModelResourceManager::new(model_config));
        let retrieval_config = RetrievalConfig::default();
        
        let aggregator = ContextAggregator::new(model_manager, retrieval_config);
        
        let content = "Einstein created the theory of relativity";
        let relationships = aggregator.extract_relationships_from_content(content);
        
        assert!(!relationships.is_empty());
        let (source, predicate, target) = &relationships[0];
        assert_eq!(source, "Einstein");
        assert_eq!(predicate, "created_by");
        assert_eq!(target, "the");
    }
}