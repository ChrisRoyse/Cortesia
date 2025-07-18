use crate::core::graph::KnowledgeGraph;
use crate::error::Result;
use crate::query::clustering::Community;
use crate::text::TextCompressor;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Fast community summarization for MCP responses
/// This is designed to provide quick summaries for external LLMs, not use LLMs internally
pub struct CommunitySummarizer {
    text_compressor: TextCompressor,
    cache: Arc<RwLock<HashMap<u32, CommunitySummary>>>,
}

impl CommunitySummarizer {
    pub fn new() -> Self {
        Self {
            text_compressor: TextCompressor::new(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Summarize a community in sub-millisecond time
    pub async fn summarize_community(&self, community: &Community, graph: &KnowledgeGraph) -> Result<CommunitySummary> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&community.id) {
                return Ok(cached.clone());
            }
        }

        // Extract entity information
        let entity_info = self.extract_entity_information(community, graph).await?;
        
        // Build fast summary
        let summary_text = self.build_fast_summary(&entity_info);
        
        // Create summary
        let summary = CommunitySummary {
            community_id: community.id,
            entity_count: community.entities.len(),
            summary: self.text_compressor.compress(&summary_text),
            key_entities: self.identify_key_entities(&entity_info),
            confidence_score: self.calculate_confidence_score(&entity_info),
        };

        // Cache the result
        {
            let mut cache = self.cache.write().await;
            cache.insert(community.id, summary.clone());
        }

        Ok(summary)
    }

    /// Extract entity information - must be fast
    async fn extract_entity_information(&self, community: &Community, graph: &KnowledgeGraph) -> Result<Vec<EntityInfo>> {
        let mut entity_infos = Vec::with_capacity(community.entities.len());
        
        for &entity_id in &community.entities {
            if let Ok((_meta, data)) = graph.get_entity(entity_id) {
                let neighbors = graph.get_neighbors(entity_id);
                
                let internal_connections = neighbors.iter()
                    .filter(|n| community.entities.contains(n))
                    .count();
                
                let external_connections = neighbors.len() - internal_connections;
                
                entity_infos.push(EntityInfo {
                    id: entity_id,
                    properties: data.properties,
                    neighbor_count: neighbors.len(),
                    internal_connections,
                    external_connections,
                });
            }
        }
        
        Ok(entity_infos)
    }

    /// Build a fast summary without LLM
    fn build_fast_summary(&self, entity_info: &[EntityInfo]) -> String {
        if entity_info.is_empty() {
            return "Empty community.".to_string();
        }

        let mut summary_parts = Vec::new();
        
        // Add entity count
        summary_parts.push(format!("Community with {} entities", entity_info.len()));
        
        // Analyze connectivity
        let total_internal: usize = entity_info.iter().map(|e| e.internal_connections).sum();
        let total_external: usize = entity_info.iter().map(|e| e.external_connections).sum();
        
        if total_internal > total_external {
            summary_parts.push("highly interconnected internally".to_string());
        } else {
            summary_parts.push("loosely connected with many external links".to_string());
        }
        
        // Extract common properties
        let properties: Vec<&str> = entity_info.iter()
            .flat_map(|e| e.properties.split_whitespace())
            .collect();
        
        if !properties.is_empty() {
            summary_parts.push(format!("containing entities related to: {}", 
                properties.iter().take(5).cloned().collect::<Vec<_>>().join(", ")));
        }
        
        summary_parts.join(", ")
    }

    fn identify_key_entities(&self, entity_info: &[EntityInfo]) -> Vec<u32> {
        let mut scored_entities: Vec<(u32, f64)> = entity_info.iter()
            .map(|e| {
                let centrality_score = e.internal_connections as f64 / (e.neighbor_count as f64 + 1.0);
                let hub_score = e.external_connections as f64 / (e.neighbor_count as f64 + 1.0);
                let total_score = centrality_score + hub_score;
                (e.id, total_score)
            })
            .collect();
        
        scored_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_entities.into_iter()
            .take(5)
            .map(|(id, _)| id)
            .collect()
    }

    fn calculate_confidence_score(&self, entity_info: &[EntityInfo]) -> f64 {
        if entity_info.is_empty() {
            return 0.0;
        }
        
        let total_connections: usize = entity_info.iter()
            .map(|e| e.internal_connections)
            .sum();
        
        let total_possible = entity_info.len() * (entity_info.len() - 1) / 2;
        
        if total_possible == 0 {
            1.0
        } else {
            (total_connections as f64 / total_possible as f64).min(1.0)
        }
    }

    pub async fn invalidate_cache(&self, community_id: u32) {
        let mut cache = self.cache.write().await;
        cache.remove(&community_id);
    }

    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

/// Information about an entity for summarization
#[derive(Debug, Clone)]
struct EntityInfo {
    id: u32,
    properties: String,
    neighbor_count: usize,
    internal_connections: usize,
    external_connections: usize,
}

/// Fast, compact summary of a community
#[derive(Debug, Clone)]
pub struct CommunitySummary {
    pub community_id: u32,
    pub entity_count: usize,
    pub summary: String, // 50-100 words max
    pub key_entities: Vec<u32>,
    pub confidence_score: f64,
}

impl CommunitySummary {
    /// Convert to a format suitable for LLM context
    pub fn to_llm_context(&self) -> String {
        format!(
            "Community {}: {} (entities: {}, confidence: {:.2})",
            self.community_id,
            self.summary,
            self.entity_count,
            self.confidence_score
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fast_summarization() {
        let summarizer = CommunitySummarizer::new();
        
        // Test performance - should be sub-millisecond
        let start = std::time::Instant::now();
        
        let entity_info = vec![
            EntityInfo {
                id: 1,
                properties: "test entity one".to_string(),
                neighbor_count: 5,
                internal_connections: 3,
                external_connections: 2,
            },
            EntityInfo {
                id: 2,
                properties: "test entity two".to_string(),
                neighbor_count: 4,
                internal_connections: 3,
                external_connections: 1,
            },
        ];
        
        let summary = summarizer.build_fast_summary(&entity_info);
        
        let elapsed = start.elapsed();
        assert!(elapsed.as_micros() < 1000); // Should be under 1ms
        
        assert!(!summary.is_empty());
        assert!(summary.contains("2 entities"));
    }
}